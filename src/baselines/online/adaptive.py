from __future__ import annotations

"""Shared adaptive streaming baseline core.

₍^. .^₎⟆ Adaptive stream flow

train sequence
  -> reference mean/std
  -> stride-1 window scoring
  -> EWMA threshold calibration
  -> policy-specific online updates
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.baselines.online.base import (
    OnlineStreamingBaselineProtocol,
    absolute_stream_offset,
    as_2d_sequence,
    build_online_thresholds,
    causal_point_scores_from_windows,
    smooth_point_scores,
)
from src.engine.online_tta.triage import classify_legacy_baseline_window
from src.models.simple_window_cnn_autoencoder import SimpleWindowCnnAutoencoder
from src.protocols.threshold_artifact import build_threshold_artifact
from src.baselines.online.redlamp_encoder_checkpoint import (
    RedLampEncoderCheckpoint,
    load_redlamp_encoder_checkpoint,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _sequence_metadata(sequence: dict[str, Any]) -> dict[str, Any]:
    return dict(sequence.get("meta", {}))


def _finite_values(array: np.ndarray) -> np.ndarray:
    flat_array = np.asarray(array, dtype=np.float64).reshape(-1)
    return flat_array[np.isfinite(flat_array)]


@dataclass(frozen=True)
class AdaptiveStreamingCalibration:
    threshold_artifact: dict[str, Any]
    threshold_value: float
    threshold_source: str
    validation_point_scores: np.ndarray
    validation_ewma_scores: np.ndarray
    method_metadata: dict[str, Any]


class AdaptiveStreamingBaselineBase(OnlineStreamingBaselineProtocol):
    method_name = "adaptive_streaming"

    def __init__(
        self,
        *,
        train_sequence: np.ndarray,
        input_dim: int | None = None,
        window_size: int = 20,
        threshold_quantile: float = 0.99,
        online_variant: str = "main",
        seed: int = 0,
        adaptation_momentum: float = 0.02,
        encoder_family: str = "cnn_simple",
        encoder_dim: int = 128,
        cnn_num_layers: int = 3,
        cnn_kernel_size: int = 3,
        cnn_hidden_channels: int = 64,
        cnn_dropout: float = 0.1,
        pretrained_encoder_checkpoint: str | Path | None = None,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0, 1)")
        if not 0.0 < adaptation_momentum <= 1.0:
            raise ValueError("adaptation_momentum must be in (0, 1]")
        if encoder_family != "cnn_simple":
            raise ValueError("M2N2 and CANDI require encoder_family='cnn_simple'")
        if encoder_dim <= 0:
            raise ValueError("encoder_dim must be positive")
        if cnn_num_layers < 2:
            raise ValueError("cnn_num_layers must be at least 2")
        if cnn_kernel_size <= 0 or cnn_hidden_channels <= 0:
            raise ValueError("CNN dimensions must be positive")
        if not 0.0 <= cnn_dropout <= 1.0:
            raise ValueError("cnn_dropout must be between 0 and 1")
        if pretrained_encoder_checkpoint is None:
            raise ValueError(
                "pretrained_encoder_checkpoint is required for M2N2 and CANDI"
            )
        self.window_size = int(window_size)
        self.input_dim = None if input_dim is None else int(input_dim)
        self.threshold_quantile = float(threshold_quantile)
        self.online_variant = str(online_variant)
        self.seed = int(seed)
        self.adaptation_momentum = float(adaptation_momentum)
        self.encoder_family = str(encoder_family)
        self.encoder_dim = int(encoder_dim)
        self.cnn_num_layers = int(cnn_num_layers)
        self.cnn_kernel_size = int(cnn_kernel_size)
        self.cnn_hidden_channels = int(cnn_hidden_channels)
        self.cnn_dropout = float(cnn_dropout)
        self.pretrained_encoder_checkpoint = str(pretrained_encoder_checkpoint)
        self.checkpoint_identity_: RedLampEncoderCheckpoint | None = None
        self.backbone_device = "cpu"
        self.backbone_: SimpleWindowCnnAutoencoder | None = None
        self.reference_mean_: np.ndarray | None = None
        self.reference_std_: np.ndarray | None = None
        self.calibration_: AdaptiveStreamingCalibration | None = None
        self.fit(train_sequence)

    def fit(self, train_sequence: np.ndarray) -> "AdaptiveStreamingBaselineBase":
        train_array = as_2d_sequence(train_sequence)
        if self.input_dim is None:
            self.input_dim = int(train_array.shape[1])
        self.reference_mean_ = np.mean(train_array, axis=0)
        self.reference_std_ = np.maximum(np.std(train_array, axis=0, ddof=1), 1.0e-3)
        self._fit_backbone(train_array)
        self.calibration_ = None
        return self

    def _fit_backbone(self, train_sequence: np.ndarray) -> None:
        if self.reference_mean_ is None or self.reference_std_ is None:
            raise RuntimeError("Reference statistics must be fitted first")
        if train_sequence.shape[0] < self.window_size:
            raise ValueError("train_sequence must be long enough for one window")
        if self.input_dim is not None and train_sequence.shape[1] != self.input_dim:
            raise ValueError(
                f"input_dim={self.input_dim} does not match train_sequence feature dimension "
                f"{train_sequence.shape[1]}"
            )

        torch.manual_seed(self.seed)
        self.backbone_ = SimpleWindowCnnAutoencoder(
            input_dim=train_sequence.shape[1],
            latent_dim=self.encoder_dim,
            hidden_channels=self.cnn_hidden_channels,
            kernel_size=self.cnn_kernel_size,
            num_layers=self.cnn_num_layers,
            dropout=self.cnn_dropout,
        ).to(self.backbone_device)
        checkpoint_path = Path(self.pretrained_encoder_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = REPOSITORY_ROOT / checkpoint_path
        self.checkpoint_identity_ = load_redlamp_encoder_checkpoint(
            encoder=self.backbone_.encoder,
            checkpoint_path=checkpoint_path,
        )
        for parameter in self.backbone_.encoder.parameters():
            parameter.requires_grad_(False)
        self.backbone_.eval()

    def _score_backbone_windows(
        self, query_sequence: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.backbone_ is None:
            raise RuntimeError("Call fit() before scoring")
        if query_sequence.shape[0] < self.window_size:
            empty = np.zeros(0, dtype=np.float64)
            return empty, empty
        if self.reference_mean_ is None or self.reference_std_ is None:
            raise RuntimeError("Call fit() before scoring")
        windows = np.stack(
            [
                query_sequence[start : start + self.window_size]
                for start in range(query_sequence.shape[0] - self.window_size + 1)
            ],
            axis=0,
        )
        normalized = (
            windows - self.reference_mean_[None, None, :]
        ) / self.reference_std_[None, None, :]
        with torch.no_grad():
            reconstruction, latent = self.backbone_(
                torch.as_tensor(normalized, dtype=torch.float32).to(
                    self.backbone_device
                )
            )
            raw_scores = torch.mean(
                (
                    reconstruction
                    - torch.as_tensor(normalized, dtype=torch.float32).to(
                        self.backbone_device
                    )
                )
                ** 2,
                dim=(1, 2),
            )
            latent_scores = torch.mean(torch.abs(latent), dim=(1, 2))
        return (
            raw_scores.detach().cpu().numpy().astype(np.float64),
            latent_scores.detach().cpu().numpy().astype(np.float64),
        )

    def _score_window_scores(
        self, query_sequence: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.backbone_ is not None:
            return self._score_backbone_windows(query_sequence)
        if self.reference_mean_ is None or self.reference_std_ is None:
            raise RuntimeError("Call fit() before scoring.")
        query_array = as_2d_sequence(query_sequence)
        if query_array.shape[0] < self.window_size:
            empty = np.zeros(0, dtype=np.float64)
            return empty, empty
        raw_scores: list[float] = []
        latent_scores: list[float] = []
        for start_index in range(0, query_array.shape[0] - self.window_size + 1):
            window = query_array[start_index : start_index + self.window_size]
            normalized = (window - self.reference_mean_[None, :]) / self.reference_std_[
                None, :
            ]
            raw_scores.append(float(np.mean(normalized**2)))
            latent_scores.append(float(np.mean(np.abs(normalized))))
        return (
            np.asarray(raw_scores, dtype=np.float64),
            np.asarray(latent_scores, dtype=np.float64),
        )

    def _update_reference(self, window: np.ndarray) -> None:
        assert self.reference_mean_ is not None and self.reference_std_ is not None
        momentum = self.adaptation_momentum
        window_mean = np.mean(window, axis=0)
        window_std = np.maximum(np.std(window, axis=0, ddof=1), 1.0e-3)
        self.reference_mean_ = (
            1.0 - momentum
        ) * self.reference_mean_ + momentum * window_mean
        self.reference_std_ = np.maximum(
            (1.0 - momentum) * self.reference_std_ + momentum * window_std,
            1.0e-3,
        )

    def _should_update(
        self,
        *,
        triage_decision: str,
        raw_point_score: float,
        ewma_point_score: float,
        threshold_value: float,
    ) -> bool:
        raise NotImplementedError

    def _method_metadata(self) -> dict[str, Any]:
        return {
            "method": self.method_name,
            "adaptation_momentum": self.adaptation_momentum,
            "online_variant": self.online_variant,
            **self._backbone_metadata(),
        }

    def _backbone_metadata(self) -> dict[str, Any]:
        if self.checkpoint_identity_ is None:
            raise RuntimeError("RedLamp encoder checkpoint has not been loaded")
        return {
            "encoder_family": self.encoder_family,
            "input_dim": self.input_dim,
            "encoder_dim": self.encoder_dim,
            "cnn_num_layers": self.cnn_num_layers,
            "cnn_kernel_size": self.cnn_kernel_size,
            "cnn_hidden_channels": self.cnn_hidden_channels,
            "cnn_dropout": self.cnn_dropout,
            "encoder_source": "RedLamp",
            "pretrained_encoder_checkpoint": self.pretrained_encoder_checkpoint,
            "resolved_checkpoint_path": self.checkpoint_identity_.checkpoint_path,
            "checkpoint_role": self.checkpoint_identity_.checkpoint_role,
            "checkpoint_sha256": self.checkpoint_identity_.checkpoint_sha256,
            "checkpoint_epoch": self.checkpoint_identity_.epoch,
        }

    def calibrate(
        self,
        *,
        clean_validation_sequences: list[dict[str, Any]],
        protocol_config: dict[str, Any],
        device: str,
    ) -> dict[str, Any]:
        del device
        if self.reference_mean_ is None or self.reference_std_ is None:
            raise RuntimeError("Call fit() before calibrate().")
        if not clean_validation_sequences:
            raise ValueError("clean_validation_sequences must not be empty")

        current_weight = float(protocol_config["online_ewma_current_weight"])
        previous_weight = float(protocol_config["online_ewma_previous_weight"])
        validation_point_scores: list[float] = []
        validation_ewma_scores: list[float] = []
        for sequence in clean_validation_sequences:
            raw_window_scores, _ = self._score_window_scores(sequence["x"])
            raw_point_scores = causal_point_scores_from_windows(
                window_scores=raw_window_scores,
                sequence_length=int(sequence["x"].shape[0]),
                window_size=self.window_size,
            )
            ewma_point_scores = smooth_point_scores(
                raw_point_scores,
                current_weight=current_weight,
                previous_weight=previous_weight,
            )
            validation_point_scores.extend(_finite_values(raw_point_scores).tolist())
            validation_ewma_scores.extend(_finite_values(ewma_point_scores).tolist())

        if not validation_ewma_scores:
            raise ValueError("Validation stream produced no finite EWMA scores")

        offline_point_threshold = float(
            np.nanquantile(
                np.asarray(validation_point_scores, dtype=np.float64),
                self.threshold_quantile,
            )
        )
        online_ewma_point_threshold = float(
            np.nanquantile(
                np.asarray(validation_ewma_scores, dtype=np.float64),
                self.threshold_quantile,
            )
        )
        entity_id = str(
            _sequence_metadata(clean_validation_sequences[0]).get(
                "entity_id", "unknown"
            )
        )
        threshold_artifact = build_threshold_artifact(
            method_name=self.method_name,
            variant_name=self.online_variant,
            entity_id=entity_id,
            seed=self.seed,
            window_size=self.window_size,
            offline_point_threshold=offline_point_threshold,
            online_ewma_point_threshold=online_ewma_point_threshold,
            quantile=self.threshold_quantile,
            ewma_current_weight=current_weight,
            ewma_previous_weight=previous_weight,
            checkpoint_sha256=(
                self.checkpoint_identity_.checkpoint_sha256
                if self.checkpoint_identity_ is not None
                else None
            ),
            created_by=f"{__name__}:{type(self).__name__}",
            config_path="configs/protocol/smd_window20_cleanval_q99_ewma09.yaml",
        )
        self.calibration_ = AdaptiveStreamingCalibration(
            threshold_artifact=threshold_artifact,
            threshold_value=online_ewma_point_threshold,
            threshold_source="clean_validation_stride1_ewma",
            validation_point_scores=np.asarray(
                validation_point_scores, dtype=np.float64
            ),
            validation_ewma_scores=np.asarray(validation_ewma_scores, dtype=np.float64),
            method_metadata=self._method_metadata(),
        )
        return {
            "threshold_artifact": threshold_artifact,
            "threshold_value": online_ewma_point_threshold,
            "threshold_source": "clean_validation_stride1_ewma",
            "validation_point_scores": self.calibration_.validation_point_scores,
            "validation_ewma_scores": self.calibration_.validation_ewma_scores,
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

        raw_window_scores, latent_window_scores = self._score_window_scores(
            sequence["x"]
        )
        raw_point_scores = causal_point_scores_from_windows(
            window_scores=raw_window_scores,
            sequence_length=int(sequence["x"].shape[0]),
            window_size=self.window_size,
        )
        latent_point_scores = causal_point_scores_from_windows(
            window_scores=latent_window_scores,
            sequence_length=int(sequence["x"].shape[0]),
            window_size=self.window_size,
        )
        ewma_point_scores = smooth_point_scores(
            raw_point_scores,
            current_weight=float(protocol_config["online_ewma_current_weight"]),
            previous_weight=float(protocol_config["online_ewma_previous_weight"]),
        )
        thresholds = build_online_thresholds(threshold_value=threshold_value)
        metric_history: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        entity_id = str(sequence["meta"]["entity_id"])
        absolute_offset = absolute_stream_offset(sequence)
        sequence_array = as_2d_sequence(sequence["x"])
        for point_index, (raw_score, latent_score, ewma_score) in enumerate(
            zip(raw_point_scores, latent_point_scores, ewma_point_scores, strict=True)
        ):
            if np.isnan(ewma_score):
                continue
            triage_decision = classify_legacy_baseline_window(
                input_window_score=float(raw_score),
                latent_window_score=float(latent_score),
                thresholds=thresholds,
            )
            did_update = self._should_update(
                triage_decision=triage_decision,
                raw_point_score=float(raw_score),
                ewma_point_score=float(ewma_score),
                threshold_value=float(threshold_value),
            )
            if did_update:
                window_end_index = point_index + 1
                window_start_index = window_end_index - self.window_size
                self._update_reference(
                    sequence_array[window_start_index:window_end_index]
                )
            window_end_index = point_index + 1
            window_start_index = window_end_index - self.window_size
            prediction = int(ewma_score > float(threshold_value))
            records.append(
                {
                    "entity_id": entity_id,
                    "point_index": absolute_offset + point_index,
                    "window_start_index": absolute_offset + window_start_index,
                    "window_end_index": absolute_offset + window_end_index,
                    "raw_point_score": float(raw_score),
                    "ewma_point_score": float(ewma_score),
                    "latent_window_score": float(latent_score),
                    "threshold": float(threshold_value),
                    "prediction": prediction,
                    "online_variant": self.online_variant,
                    "triage_decision": triage_decision,
                    "did_update": did_update,
                    "loss_total": None,
                }
            )
            metric_history.append(
                {
                    "online/step": len(metric_history) + 1,
                    "online/raw_point_score": float(raw_score),
                    "online/ewma_point_score": float(ewma_score),
                    "online/threshold": float(threshold_value),
                    "online/prediction": prediction,
                    "online/did_update": did_update,
                    "online/loss_total": None,
                    "online/triage_decision": triage_decision,
                    "online/verification_buffer_size": 0,
                }
            )
        return metric_history, records
