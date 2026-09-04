from __future__ import annotations

"""Reference CANDI candidate selection and SANA adaptation."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import chi2

from src.baselines.online.adaptive import AdaptiveStreamingBaselineBase
from src.models.online_adapter_modules import SANA


class CANDIStreamingBaseline(AdaptiveStreamingBaselineBase):
    method_name = "candi"

    def __init__(
        self,
        *,
        train_sequence: np.ndarray,
        input_dim: int | None = None,
        window_size: int = 20,
        threshold_quantile: float = 0.995,
        online_variant: str = "reference_adapter_redlamp_encoder",
        seed: int = 0,
        encoder_family: str = "cnn_simple",
        encoder_dim: int = 128,
        cnn_num_layers: int = 3,
        cnn_kernel_size: int = 3,
        cnn_hidden_channels: int = 64,
        cnn_dropout: float = 0.1,
        mlp_num_linear_layers: int = 3,
        pretrained_encoder_checkpoint: str | Path | None = None,
        pretrained_model_checkpoint: str | Path | None = None,
        candi_use_fpm: bool = True,
        candi_use_sana: bool = True,
        candi_use_hard: bool = True,
        candi_use_moderate: bool = True,
        candi_min_samples: int = 16,
        candi_steps: int = 1,
        candi_anomaly_ratio: float = 0.5,
        sana_type: str = "TCN_iTrans",
        sana_d_model: int = 512,
        sana_n_heads: int = 8,
        sana_d_ff: int = 512,
        sana_dropout: float = 0.0,
        sana_gating_init: float = 0.0,
        adaptation_learning_rate: float = 1.0e-4,
        adaptation_weight_decay: float = 1.0e-4,
        adaptation_optimizer: str = "sgd",
        adaptation_momentum: float = 0.9,
        adaptation_dampening: float = 0.0,
        adaptation_nesterov: bool = True,
        adaptation_batch_size: int = 1,
    ) -> None:
        if candi_min_samples <= 0:
            raise ValueError("candi_min_samples must be positive")
        if candi_steps != 1:
            raise ValueError("CANDI reference contract requires exactly one step")
        if not 0.0 < candi_anomaly_ratio <= 100.0:
            raise ValueError("candi_anomaly_ratio must be in (0, 100]")
        self.candi_use_fpm = bool(candi_use_fpm)
        self.candi_use_sana = bool(candi_use_sana)
        self.candi_use_hard = bool(candi_use_hard)
        self.candi_use_moderate = bool(candi_use_moderate)
        self.candi_min_samples = int(candi_min_samples)
        self.candi_steps = int(candi_steps)
        self.candi_anomaly_ratio = float(candi_anomaly_ratio)
        self.sana_type = sana_type
        self.sana_d_model = int(sana_d_model)
        self.sana_n_heads = int(sana_n_heads)
        self.sana_d_ff = int(sana_d_ff)
        self.sana_dropout = float(sana_dropout)
        self.sana_gating_init = float(sana_gating_init)
        self._sana_in: SANA | None = None
        self._sana_out: SANA | None = None
        self._val_representations: torch.Tensor | None = None
        self._representation_cov_inv: torch.Tensor | None = None
        self._hard_representations: torch.Tensor | None = None
        self._moderate_representations: torch.Tensor | None = None
        self._q1: float | None = None
        self._q3: float | None = None
        self._hard_pool: list[torch.Tensor] = []
        self._moderate_pool: list[torch.Tensor] = []
        self._active_test_labels: np.ndarray | None = None
        self.total_samples_to_adapt_hard = 0
        self.total_samples_to_adapt_moderate = 0
        self.total_anomalies_in_hard = 0
        self.total_anomalies_in_moderate = 0
        super().__init__(
            train_sequence=train_sequence,
            input_dim=input_dim,
            window_size=window_size,
            threshold_quantile=threshold_quantile,
            online_variant=online_variant,
            seed=seed,
            encoder_family=encoder_family,
            encoder_dim=encoder_dim,
            cnn_num_layers=cnn_num_layers,
            cnn_kernel_size=cnn_kernel_size,
            cnn_hidden_channels=cnn_hidden_channels,
            cnn_dropout=cnn_dropout,
            mlp_num_linear_layers=mlp_num_linear_layers,
            pretrained_encoder_checkpoint=pretrained_encoder_checkpoint,
            pretrained_model_checkpoint=pretrained_model_checkpoint,
            adaptation_learning_rate=adaptation_learning_rate,
            adaptation_weight_decay=adaptation_weight_decay,
            adaptation_optimizer=adaptation_optimizer,
            adaptation_momentum=adaptation_momentum,
            adaptation_dampening=adaptation_dampening,
            adaptation_nesterov=adaptation_nesterov,
            adaptation_batch_size=adaptation_batch_size,
        )

    def _initialize_method_state(self) -> None:
        if self.backbone_ is None:
            raise RuntimeError("model must be loaded before CANDI state")
        if self.candi_use_sana:
            self._sana_in = SANA(
                input_dim=self.input_dim,
                window_size=self.window_size,
                sana_type=self.sana_type,
                d_model=self.sana_d_model,
                n_heads=self.sana_n_heads,
                d_ff=self.sana_d_ff,
                dropout=self.sana_dropout,
                gating_init=self.sana_gating_init,
            ).to(self.backbone_device)
            self._sana_out = SANA(
                input_dim=self.input_dim,
                window_size=self.window_size,
                sana_type=self.sana_type,
                d_model=self.sana_d_model,
                n_heads=self.sana_n_heads,
                d_ff=self.sana_d_ff,
                dropout=self.sana_dropout,
                gating_init=self.sana_gating_init,
            ).to(self.backbone_device)
            for parameter in self.backbone_.parameters():
                parameter.requires_grad_(False)
            trainable_modules = list(self._sana_in.parameters()) + list(
                self._sana_out.parameters()
            )
        else:
            for parameter in self.backbone_.parameters():
                parameter.requires_grad_(True)
            trainable_modules = list(self.backbone_.parameters())
        self.optimizer_ = self._build_optimizer(trainable_modules)

    def _candi_input(self, x: torch.Tensor) -> torch.Tensor:
        if self._sana_in is None:
            return x
        return x + self._sana_in(x)

    def _candi_reconstruction(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction = self.backbone_(self._candi_input(x))
        if self._sana_out is not None:
            reconstruction = reconstruction - self._sana_out(reconstruction)
        return reconstruction

    def _score_tensor(self, x: torch.Tensor) -> tuple[float, float]:
        self.backbone_.eval()
        if self._sana_in is not None:
            self._sana_in.eval()
        if self._sana_out is not None:
            self._sana_out.eval()
        with torch.no_grad():
            reconstruction = self._candi_reconstruction(x)
            score = float(F.mse_loss(reconstruction, x).cpu())
            representation = self.backbone_.get_representations(x)
            latent_score = float(representation.abs().mean().cpu())
        return score, latent_score

    def _score_tensor_batch(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        self.backbone_.eval()
        if self._sana_in is not None:
            self._sana_in.eval()
        if self._sana_out is not None:
            self._sana_out.eval()
        with torch.no_grad():
            reconstruction = self._candi_reconstruction(x)
            errors = (reconstruction - x) ** 2
            scores = errors.mean(dim=(1, 2))
            representations = self.backbone_.get_representations(x)
            latent_scores = representations.abs().mean(dim=1)
        return (
            scores.cpu().numpy().astype(np.float64),
            latent_scores.cpu().numpy().astype(np.float64),
        )

    def _calibration_complete(
        self, validation_windows: np.ndarray, validation_scores: np.ndarray
    ) -> None:
        if not self.candi_use_fpm:
            return
        if validation_windows.shape[0] < 2:
            raise ValueError("CANDI FPM requires at least two validation windows")
        windows = torch.as_tensor(validation_windows, dtype=torch.float32)
        with torch.no_grad():
            representations = self.backbone_.get_representations(windows)
        self._val_representations = representations
        covariance = torch.cov(representations.T)
        self._representation_cov_inv = torch.linalg.pinv(covariance)
        topk = int(len(validation_scores) * self.candi_anomaly_ratio / 100.0)
        if topk <= 0:
            raise ValueError(
                "CANDI hard references require enough validation windows for top-k"
            )
        topk_indices = np.argpartition(validation_scores, -topk)[-topk:]
        self._hard_representations = representations[topk_indices]
        self._q1 = float(np.percentile(validation_scores, 25))
        self._q3 = float(np.percentile(validation_scores, 75))
        moderate_mask = (validation_scores > self._q1) & (validation_scores < self._q3)
        moderate_indices = np.flatnonzero(moderate_mask)
        if moderate_indices.size == 0:
            raise ValueError("CANDI FPM produced no moderate validation references")
        self._moderate_representations = representations[moderate_indices]

    def _mahalanobis_similarity(
        self, representation: torch.Tensor, references: torch.Tensor
    ) -> bool:
        if self._representation_cov_inv is None:
            raise RuntimeError("CANDI covariance has not been initialized")
        difference = representation[:, None, :] - references[None, :, :]
        distances = torch.sum(
            (difference @ self._representation_cov_inv) * difference, dim=-1
        )
        chi2_threshold = float(chi2.ppf(0.05, df=representation.shape[1]))
        return bool((distances < chi2_threshold).any().cpu())

    def _collect_candidates(
        self, x: torch.Tensor, scores: np.ndarray, threshold: float
    ) -> tuple[bool, bool]:
        score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
        if score_values.size == 1 and x.shape[0] > 1:
            score_values = np.repeat(score_values, x.shape[0])
        if score_values.size != x.shape[0]:
            raise ValueError("CANDI scores do not match the adaptation batch")
        hard_selected = False
        moderate_selected = False
        for sample_index, sample in enumerate(x):
            hard, moderate = self._collect_one_candidate(
                sample.unsqueeze(0),
                float(score_values[sample_index]),
                threshold,
                sample_index,
            )
            hard_selected = hard_selected or hard
            moderate_selected = moderate_selected or moderate
        return hard_selected, moderate_selected

    def _collect_one_candidate(
        self,
        x: torch.Tensor,
        score: float,
        threshold: float,
        sample_index: int,
    ) -> tuple[bool, bool]:
        if not self.candi_use_fpm:
            if score < threshold and self.candi_use_moderate:
                self._moderate_pool.append(x.detach().clone())
                self._record_selected_label("moderate", sample_index)
                return False, True
            return False, False
        with torch.no_grad():
            representation = self.backbone_.get_representations(x)
        hard_selected = False
        moderate_selected = False
        if self.candi_use_hard and self._hard_representations is not None:
            hard_selected = score > threshold and self._mahalanobis_similarity(
                representation, self._hard_representations
            )
            if hard_selected:
                self._hard_pool.append(x.detach().clone())
                self._record_selected_label("hard", sample_index)
        if self.candi_use_moderate and self._moderate_representations is not None:
            moderate_selected = score < threshold and self._mahalanobis_similarity(
                representation, self._moderate_representations
            )
            if moderate_selected:
                self._moderate_pool.append(x.detach().clone())
                self._record_selected_label("moderate", sample_index)
        return hard_selected, moderate_selected

    def _record_selected_label(self, pool_name: str, sample_index: int) -> None:
        if self._active_test_labels is None:
            return
        if sample_index >= len(self._active_test_labels):
            raise ValueError("CANDI test-label diagnostics do not match batch size")
        label = int(self._active_test_labels[sample_index])
        if pool_name == "hard":
            self.total_samples_to_adapt_hard += 1
            self.total_anomalies_in_hard += int(label == 1)
        else:
            self.total_samples_to_adapt_moderate += 1
            self.total_anomalies_in_moderate += int(label == 1)

    def _adapt_pool(self, pool: list[torch.Tensor]) -> float:
        if not pool:
            raise ValueError("CANDI cannot adapt an empty pool")
        # step 1: Combine the selected pool and enter training mode.
        batch = torch.cat(pool, dim=0)
        was_training = self.backbone_.training
        self.backbone_.train()
        if self._sana_in is not None:
            self._sana_in.train()
        if self._sana_out is not None:
            self._sana_out.train()
        loss = torch.zeros((), dtype=batch.dtype)
        # step 2: Repeat the CANDI adaptation update for the configured steps.
        for _ in range(self.candi_steps):
            # step 3: Compute the reconstruction loss for the selected pool.
            loss = F.mse_loss(self._candi_reconstruction(batch), batch)
            if not torch.isfinite(loss):
                self.backbone_.train(was_training)
                raise FloatingPointError("CANDI SANA loss is not finite")
            # step 4: Backpropagate and update the CANDI parameters.
            self.optimizer_.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer_.step()

        # step 5: Restore evaluation mode and return the final adaptation loss.
        self.backbone_.train(was_training)
        self.backbone_.eval()
        return float(loss.detach().cpu())

    def _adapt_tensor(
        self, x: torch.Tensor, score: np.ndarray, threshold: float
    ) -> dict[str, Any]:
        hard_selected, moderate_selected = self._collect_candidates(x, score, threshold)
        losses: list[float] = []
        if len(self._hard_pool) >= self.candi_min_samples:
            losses.append(self._adapt_pool(self._hard_pool))
            self._hard_pool = []
        if len(self._moderate_pool) >= self.candi_min_samples:
            losses.append(self._adapt_pool(self._moderate_pool))
            self._moderate_pool = []
        did_update = bool(losses)
        if hard_selected and moderate_selected:
            decision = "candi_hard_and_moderate_candidate"
        elif hard_selected:
            decision = "candi_hard_candidate"
        elif moderate_selected:
            decision = "candi_moderate_candidate"
        else:
            decision = "candi_no_candidate"
        return {
            "decision": decision,
            "did_update": did_update,
            "loss_total": float(np.mean(losses)) if losses else None,
            "candidate_pool_hard_size": len(self._hard_pool),
            "candidate_pool_moderate_size": len(self._moderate_pool),
            "verification_buffer_size": len(self._hard_pool) + len(self._moderate_pool),
            "total_samples_to_adapt_hard": self.total_samples_to_adapt_hard,
            "total_samples_to_adapt_moderate": self.total_samples_to_adapt_moderate,
            "total_anomalies_in_hard": self.total_anomalies_in_hard,
            "total_anomalies_in_moderate": self.total_anomalies_in_moderate,
        }

    def _method_metadata(self) -> dict[str, Any]:
        metadata = super()._method_metadata()
        metadata.update(
            {
                "policy": "reference_candi",
                "update_policy": "fpm_hard_moderate_pools_then_sana_mse",
                "candi_use_fpm": self.candi_use_fpm,
                "candi_use_sana": self.candi_use_sana,
                "candi_use_hard": self.candi_use_hard,
                "candi_use_moderate": self.candi_use_moderate,
                "candi_min_samples": self.candi_min_samples,
                "candi_steps": self.candi_steps,
                "candi_anomaly_ratio": self.candi_anomaly_ratio,
                "sana_type": self.sana_type,
                "sana_d_model": self.sana_d_model,
                "sana_n_heads": self.sana_n_heads,
                "sana_d_ff": self.sana_d_ff,
                "sana_gating_init": self.sana_gating_init,
                "trainable_surface": (
                    "sana_in_and_sana_out"
                    if self.candi_use_sana
                    else "redlamp_encoder_and_adapter_decoder"
                ),
                "test_label_usage": "metrics_only",
            }
        )
        return metadata
