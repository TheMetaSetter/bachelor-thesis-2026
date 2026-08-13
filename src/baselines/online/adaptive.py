from __future__ import annotations

"""Method-owned causal runtime for the M2N2 and CANDI adapters."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import chi2

from src.baselines.online.base import (
    OnlineStreamingBaselineProtocol,
    absolute_stream_offset,
    as_2d_sequence,
)
from src.models.online_adapter_modules import Detrender
from src.models.online_redlamp_reconstruction import (
    RedLampReconstructionCheckpoint,
    RedLampReconstructionModel,
    load_redlamp_reconstruction_checkpoint,
)
from src.protocols.threshold_artifact import build_threshold_artifact


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _finite_values(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def _metric_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "online/step": record["stream_step"],
        "online/raw_point_score": record["raw_point_score"],
        "online/ewma_point_score": record["ewma_point_score"],
        "online/threshold": record["threshold"],
        "online/prediction": record["prediction"],
        "online/did_update": record["did_update"],
        "online/loss_total": record["loss_total"],
        "online/triage_decision": record["triage_decision"],
        "online/verification_buffer_size": record["online/verification_buffer_size"],
        "online/adaptation_mask_count": record["adaptation_mask_count"],
        "online/candidate_pool_hard_size": record["candidate_pool_hard_size"],
        "online/candidate_pool_moderate_size": record["candidate_pool_moderate_size"],
    }


def _window_matrix(sequence: np.ndarray, window_size: int) -> np.ndarray:
    if sequence.shape[0] < window_size:
        return np.zeros((0, window_size, sequence.shape[1]), dtype=np.float32)
    return np.stack(
        [
            sequence[start : start + window_size]
            for start in range(sequence.shape[0] - window_size + 1)
        ],
        axis=0,
    ).astype(np.float32, copy=False)


@dataclass(frozen=True)
class AdaptiveStreamingCalibration:
    threshold_artifact: dict[str, Any]
    threshold_value: float
    threshold_source: str
    validation_point_scores: np.ndarray
    validation_ewma_scores: np.ndarray
    method_metadata: dict[str, Any]


class AdaptiveStreamingBaselineBase(OnlineStreamingBaselineProtocol):
    """Shared model/checkpoint and causal stream lifecycle.

    Subclasses implement only their reference-specific state and update rule.
    The lifecycle is always ``score -> record -> adapt`` for one batch.
    """

    method_name = "adaptive_streaming"

    def __init__(
        self,
        *,
        train_sequence: np.ndarray,
        input_dim: int | None = None,
        window_size: int = 20,
        threshold_quantile: float = 0.995,
        online_variant: str = "reference_adapter_redlamp",
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
        adaptation_learning_rate: float = 1.0e-4,
        adaptation_weight_decay: float = 1.0e-4,
        adaptation_optimizer: str = "sgd",
        adaptation_momentum: float = 0.9,
        adaptation_dampening: float = 0.0,
        adaptation_nesterov: bool = True,
        adaptation_batch_size: int = 1,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0, 1)")
        if online_variant == "main":
            raise ValueError(
                "M2N2/CANDI approximation artifacts use online_variant='main'; "
                "use 'reference_adapter_redlamp_encoder' for the method adapter"
            )
        if encoder_family != "cnn_simple":
            raise ValueError(
                "reference_adapter_redlamp requires encoder_family='cnn_simple'"
            )
        if encoder_dim <= 0 or cnn_num_layers < 2:
            raise ValueError("encoder dimensions are invalid")
        if cnn_kernel_size <= 0 or cnn_hidden_channels <= 0:
            raise ValueError("CNN dimensions must be positive")
        if not 0.0 <= cnn_dropout <= 1.0:
            raise ValueError("cnn_dropout must be between 0 and 1")
        if mlp_num_linear_layers < 2:
            raise ValueError("mlp_num_linear_layers must be at least 2")
        if adaptation_learning_rate <= 0.0:
            raise ValueError("adaptation_learning_rate must be positive")
        if adaptation_weight_decay < 0.0:
            raise ValueError("adaptation_weight_decay must be non-negative")
        if adaptation_optimizer not in {"sgd", "adamw"}:
            raise ValueError("adaptation_optimizer must be 'sgd' or 'adamw'")
        if adaptation_momentum < 0.0 or adaptation_dampening < 0.0:
            raise ValueError("optimizer momentum and dampening must be non-negative")
        if adaptation_nesterov and (
            adaptation_optimizer != "sgd"
            or adaptation_momentum <= 0.0
            or adaptation_dampening != 0.0
        ):
            raise ValueError(
                "Nesterov SGD requires optimizer='sgd', positive momentum and zero dampening"
            )
        if adaptation_batch_size <= 0:
            raise ValueError("adaptation_batch_size must be positive")
        checkpoint_value = pretrained_model_checkpoint or pretrained_encoder_checkpoint
        if checkpoint_value is None:
            raise ValueError(
                "pretrained_model_checkpoint is required; encoder-only checkpoints "
                "are not valid for reference_adapter_redlamp"
            )

        train_array = as_2d_sequence(train_sequence)
        if train_array.shape[0] < window_size:
            raise ValueError("train_sequence must be long enough for one window")
        inferred_input_dim = int(train_array.shape[1])
        if input_dim is not None and int(input_dim) != inferred_input_dim:
            raise ValueError(
                f"input_dim={input_dim} does not match train_sequence feature dimension "
                f"{inferred_input_dim}"
            )
        self.window_size = int(window_size)
        self.input_dim = inferred_input_dim
        self.threshold_quantile = float(threshold_quantile)
        self.online_variant = str(online_variant)
        self.seed = int(seed)
        self.encoder_family = str(encoder_family)
        self.encoder_dim = int(encoder_dim)
        self.cnn_num_layers = int(cnn_num_layers)
        self.cnn_kernel_size = int(cnn_kernel_size)
        self.cnn_hidden_channels = int(cnn_hidden_channels)
        self.cnn_dropout = float(cnn_dropout)
        self.mlp_num_linear_layers = int(mlp_num_linear_layers)
        self.pretrained_model_checkpoint = str(checkpoint_value)
        self.adaptation_learning_rate = float(adaptation_learning_rate)
        self.adaptation_weight_decay = float(adaptation_weight_decay)
        self.adaptation_optimizer = str(adaptation_optimizer)
        self.adaptation_momentum = float(adaptation_momentum)
        self.adaptation_dampening = float(adaptation_dampening)
        self.adaptation_nesterov = bool(adaptation_nesterov)
        self.adaptation_batch_size = int(adaptation_batch_size)
        self.backbone_device = torch.device("cpu")
        self.backbone_: RedLampReconstructionModel | None = None
        self.checkpoint_identity_: RedLampReconstructionCheckpoint | None = None
        self.optimizer_: torch.optim.Optimizer | None = None
        self.calibration_: AdaptiveStreamingCalibration | None = None
        self._fit_backbone()
        self._initialize_method_state()

    def _fit_backbone(self) -> None:
        torch.manual_seed(self.seed)
        self.backbone_ = RedLampReconstructionModel(
            input_dim=self.input_dim,
            window_size=self.window_size,
            latent_dim=self.encoder_dim,
            hidden_channels=self.cnn_hidden_channels,
            kernel_size=self.cnn_kernel_size,
            num_layers=self.cnn_num_layers,
            dropout=self.cnn_dropout,
            mlp_num_linear_layers=self.mlp_num_linear_layers,
        ).to(self.backbone_device)
        checkpoint_path = Path(self.pretrained_model_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = REPOSITORY_ROOT / checkpoint_path
        self.checkpoint_identity_ = load_redlamp_reconstruction_checkpoint(
            model=self.backbone_, checkpoint_path=checkpoint_path
        )
        self.backbone_.eval()

    def _initialize_method_state(self) -> None:
        """Allow a subclass to allocate state after checkpoint validation."""

    def _calibration_complete(
        self, validation_windows: np.ndarray, validation_scores: np.ndarray
    ) -> None:
        del validation_windows, validation_scores

    def _score_tensor(self, x: torch.Tensor) -> tuple[float, float]:
        if self.backbone_ is None:
            raise RuntimeError("model has not been initialized")
        self.backbone_.eval()
        with torch.no_grad():
            reconstruction = self.backbone_(x)
            errors = (reconstruction - x) ** 2
            score = float(errors.mean().cpu())
            representation = self.backbone_.get_representations(x)
            latent_score = float(representation.abs().mean().cpu())
        return score, latent_score

    def _score_tensor_batch(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if self.backbone_ is None:
            raise RuntimeError("model has not been initialized")
        self.backbone_.eval()
        with torch.no_grad():
            reconstruction = self.backbone_(x)
            errors = (reconstruction - x) ** 2
            scores = errors.mean(dim=(1, 2))
            representations = self.backbone_.get_representations(x)
            latent_scores = representations.abs().mean(dim=1)
        return (
            scores.cpu().numpy().astype(np.float64),
            latent_scores.cpu().numpy().astype(np.float64),
        )

    def _adapt_tensor(
        self, x: torch.Tensor, score: float, threshold: float
    ) -> dict[str, Any]:
        del x, score, threshold
        raise NotImplementedError

    def _build_optimizer(self, parameters: Any) -> torch.optim.Optimizer:
        trainable_parameters = [
            parameter for parameter in parameters if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError("adapter optimizer received no trainable parameters")
        if self.adaptation_optimizer == "sgd":
            return torch.optim.SGD(
                trainable_parameters,
                lr=self.adaptation_learning_rate,
                momentum=self.adaptation_momentum,
                dampening=self.adaptation_dampening,
                weight_decay=self.adaptation_weight_decay,
                nesterov=self.adaptation_nesterov,
            )
        return torch.optim.AdamW(
            trainable_parameters,
            lr=self.adaptation_learning_rate,
            weight_decay=self.adaptation_weight_decay,
        )

    def _method_metadata(self) -> dict[str, Any]:
        if self.checkpoint_identity_ is None:
            raise RuntimeError("checkpoint identity is unavailable")
        return {
            "method": self.method_name,
            "online_variant": self.online_variant,
            "method_contract": "reference_adapter_redlamp_encoder",
            "model_family": "RedLamp",
            "encoder_family": self.encoder_family,
            "input_dim": self.input_dim,
            "window_size": self.window_size,
            "encoder_dim": self.encoder_dim,
            "cnn_num_layers": self.cnn_num_layers,
            "cnn_kernel_size": self.cnn_kernel_size,
            "cnn_hidden_channels": self.cnn_hidden_channels,
            "cnn_dropout": self.cnn_dropout,
            "mlp_num_linear_layers": self.mlp_num_linear_layers,
            "checkpoint_path": self.checkpoint_identity_.checkpoint_path,
            "checkpoint_role": self.checkpoint_identity_.checkpoint_role,
            "checkpoint_contract": self.checkpoint_identity_.checkpoint_contract,
            "checkpoint_sha256": self.checkpoint_identity_.checkpoint_sha256,
            "checkpoint_epoch": self.checkpoint_identity_.epoch,
            "adaptation_learning_rate": self.adaptation_learning_rate,
            "adaptation_weight_decay": self.adaptation_weight_decay,
            "adaptation_optimizer": self.adaptation_optimizer,
            "adaptation_momentum": self.adaptation_momentum,
            "adaptation_dampening": self.adaptation_dampening,
            "adaptation_nesterov": self.adaptation_nesterov,
            "adaptation_batch_size": self.adaptation_batch_size,
        }

    def _score_validation_sequences(
        self, clean_validation_sequences: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        all_windows: list[np.ndarray] = []
        all_scores: list[float] = []
        for sequence in clean_validation_sequences:
            sequence_array = as_2d_sequence(sequence["x"])
            windows = _window_matrix(sequence_array, self.window_size)
            all_windows.append(windows)
            for start in range(0, windows.shape[0], 256):
                batch = torch.as_tensor(
                    windows[start : start + 256], dtype=torch.float32
                )
                scores, _ = self._score_tensor_batch(batch)
                all_scores.extend(scores.tolist())
        if not all_scores:
            raise ValueError("Validation stream produced no windows")
        return np.concatenate(all_windows, axis=0), np.asarray(
            all_scores, dtype=np.float64
        )

    def calibrate(
        self,
        *,
        clean_validation_sequences: list[dict[str, Any]],
        protocol_config: dict[str, Any],
        device: str,
    ) -> dict[str, Any]:
        del device
        if not clean_validation_sequences:
            raise ValueError("clean_validation_sequences must not be empty")
        validation_windows, validation_window_scores = self._score_validation_sequences(
            clean_validation_sequences
        )
        self._calibration_complete(validation_windows, validation_window_scores)

        current_weight = float(protocol_config["online_ewma_current_weight"])
        previous_weight = float(protocol_config["online_ewma_previous_weight"])
        point_scores = _finite_values(validation_window_scores)
        ewma_scores = np.empty_like(point_scores)
        for index, score in enumerate(point_scores):
            ewma_scores[index] = (
                score
                if index == 0
                else (current_weight * score + previous_weight * ewma_scores[index - 1])
            )
        threshold_value = float(np.nanquantile(point_scores, self.threshold_quantile))
        ewma_threshold = float(np.nanquantile(ewma_scores, self.threshold_quantile))
        entity_id = str(
            clean_validation_sequences[0].get("meta", {}).get("entity_id", "unknown")
        )
        checkpoint_sha256 = (
            self.checkpoint_identity_.checkpoint_sha256
            if self.checkpoint_identity_ is not None
            else None
        )
        artifact = build_threshold_artifact(
            method_name=self.method_name,
            variant_name=self.online_variant,
            entity_id=entity_id,
            seed=self.seed,
            window_size=self.window_size,
            offline_point_threshold=threshold_value,
            online_ewma_point_threshold=ewma_threshold,
            quantile=self.threshold_quantile,
            ewma_current_weight=current_weight,
            ewma_previous_weight=previous_weight,
            checkpoint_sha256=checkpoint_sha256,
            created_by=f"{__name__}:{type(self).__name__}",
            config_path="configs/protocol/smd_window20_cleanval_q995_ewma09.yaml",
        )
        self.calibration_ = AdaptiveStreamingCalibration(
            threshold_artifact=artifact,
            threshold_value=threshold_value,
            threshold_source="clean_validation_stride1_raw_window",
            validation_point_scores=point_scores,
            validation_ewma_scores=ewma_scores,
            method_metadata=self._method_metadata(),
        )
        return {
            "threshold_artifact": artifact,
            "threshold_value": threshold_value,
            "threshold_source": self.calibration_.threshold_source,
            "validation_point_scores": point_scores,
            "validation_ewma_scores": ewma_scores,
            "method_metadata": self.calibration_.method_metadata,
        }

    def run_sequence(
        self,
        *,
        sequence: dict[str, Any],
        threshold_value: float,
        protocol_config: dict[str, Any],
        device: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        del device
        if self.calibration_ is None:
            raise RuntimeError("Call calibrate() before run_sequence().")
        sequence_array = as_2d_sequence(sequence["x"])
        if sequence_array.shape[0] < self.window_size:
            return [], []
        current_weight = float(protocol_config["online_ewma_current_weight"])
        previous_weight = float(protocol_config["online_ewma_previous_weight"])
        absolute_offset = absolute_stream_offset(sequence)
        entity_id = str(sequence.get("meta", {}).get("entity_id", "unknown"))
        metric_history: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        previous_ewma: float | None = None
        windows = _window_matrix(sequence_array, self.window_size)
        for batch_start in range(0, windows.shape[0], self.adaptation_batch_size):
            batch_windows = windows[
                batch_start : batch_start + self.adaptation_batch_size
            ]
            tensor_batch = torch.as_tensor(batch_windows, dtype=torch.float32)
            if tensor_batch.shape[0] == 1:
                raw_score, latent_score = self._score_tensor(tensor_batch)
                raw_scores = np.asarray([raw_score], dtype=np.float64)
                latent_scores = np.asarray([latent_score], dtype=np.float64)
            else:
                raw_scores, latent_scores = self._score_tensor_batch(tensor_batch)
            batch_records: list[dict[str, Any]] = []
            for batch_offset, raw_score in enumerate(raw_scores):
                stream_step = batch_start + batch_offset + 1
                ewma_score = (
                    float(raw_score)
                    if previous_ewma is None
                    else (
                        current_weight * float(raw_score)
                        + previous_weight * previous_ewma
                    )
                )
                previous_ewma = ewma_score
                window_start = absolute_offset + stream_step - 1
                window_end = window_start + self.window_size
                record = {
                    "entity_id": entity_id,
                    "point_index": window_end - 1,
                    "window_start_index": window_start,
                    "window_end_index": window_end,
                    "stream_step": stream_step,
                    "raw_point_score": float(raw_score),
                    "ewma_point_score": ewma_score,
                    "latent_window_score": float(latent_scores[batch_offset]),
                    "threshold": float(threshold_value),
                    "prediction": int(float(raw_score) > float(threshold_value)),
                    "online_variant": self.online_variant,
                    "triage_decision": "pending_adaptation",
                    "did_update": False,
                    "loss_total": None,
                    "online/verification_buffer_size": 0,
                    "adaptation_mask_count": 0,
                    "candidate_pool_hard_size": 0,
                    "candidate_pool_moderate_size": 0,
                }
                records.append(record)
                batch_records.append(record)

            if hasattr(self, "_active_test_labels"):
                point_labels = sequence.get("point_labels")
                if point_labels is None:
                    self._active_test_labels = None
                else:
                    labels_array = np.asarray(point_labels).reshape(-1)
                    endpoint_indices = (
                        np.arange(
                            batch_start,
                            batch_start + len(batch_records),
                        )
                        + self.window_size
                        - 1
                    )
                    if int(endpoint_indices[-1]) >= len(labels_array):
                        raise ValueError("test labels do not cover the stream windows")
                    self._active_test_labels = labels_array[endpoint_indices]
            update = self._adapt_tensor(
                tensor_batch, raw_scores, float(threshold_value)
            )
            for record in batch_records:
                record.update(
                    {
                        "triage_decision": str(update["decision"]),
                        "did_update": bool(update["did_update"]),
                        "loss_total": update.get("loss_total"),
                        "online/verification_buffer_size": int(
                            update.get("verification_buffer_size", 0)
                        ),
                        "adaptation_mask_count": int(update.get("mask_count", 0)),
                        "candidate_pool_hard_size": int(
                            update.get("candidate_pool_hard_size", 0)
                        ),
                        "candidate_pool_moderate_size": int(
                            update.get("candidate_pool_moderate_size", 0)
                        ),
                        "total_samples_to_adapt_hard": int(
                            update.get("total_samples_to_adapt_hard", 0)
                        ),
                        "total_samples_to_adapt_moderate": int(
                            update.get("total_samples_to_adapt_moderate", 0)
                        ),
                        "total_anomalies_in_hard": int(
                            update.get("total_anomalies_in_hard", 0)
                        ),
                        "total_anomalies_in_moderate": int(
                            update.get("total_anomalies_in_moderate", 0)
                        ),
                    }
                )
            metric_history.extend(
                _metric_from_record(record) for record in batch_records
            )
        return metric_history, records
