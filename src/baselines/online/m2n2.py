from __future__ import annotations

"""Reference M2N2 update rule on the repository RedLamp encoder contract."""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.baselines.online.adaptive import AdaptiveStreamingBaselineBase
from src.models.online_adapter_modules import Detrender


class M2N2StreamingBaseline(AdaptiveStreamingBaselineBase):
    method_name = "m2n2"

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
        m2n2_gamma: float = 0.99999,
        m2n2_steps: int = 1,
        adaptation_learning_rate: float = 1.0e-4,
        adaptation_weight_decay: float = 1.0e-4,
        adaptation_optimizer: str = "sgd",
        adaptation_momentum: float = 0.9,
        adaptation_dampening: float = 0.0,
        adaptation_nesterov: bool = True,
        adaptation_batch_size: int = 1,
    ) -> None:
        self.m2n2_gamma = float(m2n2_gamma)
        self.m2n2_steps = int(m2n2_steps)
        if self.m2n2_steps != 1:
            raise ValueError("M2N2 reference contract requires exactly one step")
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
            raise RuntimeError("model must be loaded before M2N2 state")
        self.detrender = Detrender(
            num_features=self.input_dim, gamma=self.m2n2_gamma
        ).to(self.backbone_device)
        self.optimizer_ = self._build_optimizer(self.backbone_.parameters())

    def _score_tensor(self, x: torch.Tensor) -> tuple[float, float]:
        self.backbone_.eval()
        with torch.no_grad():
            normalized = self.detrender.normalize(x)
            reconstruction = self.detrender.denormalize(self.backbone_(normalized))
            errors = (reconstruction - x) ** 2
            score = float(errors.mean().cpu())
            latent_score = float(
                self.backbone_.get_representations(normalized).abs().mean().cpu()
            )
        return score, latent_score

    def _score_tensor_batch(
        self, x: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        self.backbone_.eval()
        with torch.no_grad():
            normalized = self.detrender.normalize(x)
            reconstruction = self.detrender.denormalize(self.backbone_(normalized))
            errors = (reconstruction - x) ** 2
            scores = errors.mean(dim=(1, 2))
            representations = self.backbone_.get_representations(normalized)
            latent_scores = representations.abs().mean(dim=1)
        return (
            scores.cpu().numpy().astype(np.float64),
            latent_scores.cpu().numpy().astype(np.float64),
        )

    def _adapt_tensor(
        self, x: torch.Tensor, score: np.ndarray, threshold: float
    ) -> dict[str, Any]:
        del score
        if self.optimizer_ is None:
            raise RuntimeError("M2N2 optimizer is not initialized")
        self.detrender.update_statistics(x)
        was_training = self.backbone_.training
        self.backbone_.train()
        normalized = self.detrender.normalize(x)
        reconstruction = self.detrender.denormalize(self.backbone_(normalized))
        timestep_error = torch.mean((reconstruction - x) ** 2, dim=-1)
        pseudo_anomaly = timestep_error >= float(threshold)
        normal_mask = pseudo_anomaly == 0
        loss = (timestep_error * normal_mask).mean()
        if not torch.isfinite(loss):
            self.backbone_.train(was_training)
            raise FloatingPointError("M2N2 masked reconstruction loss is not finite")
        self.optimizer_.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer_.step()
        self.backbone_.train(was_training)
        return {
            "decision": "m2n2_masked_update",
            "did_update": True,
            "loss_total": float(loss.detach().cpu()),
            "mask_count": int(normal_mask.sum().detach().cpu()),
            "pseudo_anomaly_count": int(pseudo_anomaly.sum().detach().cpu()),
        }

    def _method_metadata(self) -> dict[str, Any]:
        metadata = super()._method_metadata()
        metadata.update(
            {
                "policy": "reference_m2n2",
                "update_policy": "detrender_pseudo_masked_reconstruction",
                "normalizer": "mean_only_detrender",
                "m2n2_gamma": self.m2n2_gamma,
                "m2n2_steps": self.m2n2_steps,
                "trainable_surface": "redlamp_encoder_and_adapter_decoder",
                "test_label_usage": "metrics_only",
            }
        )
        return metadata
