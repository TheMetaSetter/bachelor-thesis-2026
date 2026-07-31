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
from typing import Any

import numpy as np

from src.baselines.online.base import (
    OnlineStreamingBaselineProtocol,
    as_2d_sequence,
    build_online_thresholds,
    causal_point_scores_from_windows,
    smooth_point_scores,
)
from src.engine.online_tta.triage import classify_legacy_baseline_window
from src.protocols.threshold_artifact import build_threshold_artifact


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
        window_size: int = 20,
        threshold_quantile: float = 0.99,
        online_variant: str = "A0",
        seed: int = 0,
        adaptation_momentum: float = 0.02,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0, 1)")
        if not 0.0 < adaptation_momentum <= 1.0:
            raise ValueError("adaptation_momentum must be in (0, 1]")
        self.window_size = int(window_size)
        self.threshold_quantile = float(threshold_quantile)
        self.online_variant = str(online_variant)
        self.seed = int(seed)
        self.adaptation_momentum = float(adaptation_momentum)
        self.reference_mean_: np.ndarray | None = None
        self.reference_std_: np.ndarray | None = None
        self.calibration_: AdaptiveStreamingCalibration | None = None
        self.fit(train_sequence)

    def fit(self, train_sequence: np.ndarray) -> "AdaptiveStreamingBaselineBase":
        train_array = as_2d_sequence(train_sequence)
        self.reference_mean_ = np.mean(train_array, axis=0)
        self.reference_std_ = np.maximum(np.std(train_array, axis=0, ddof=1), 1.0e-3)
        self.calibration_ = None
        return self

    def _score_window_scores(
        self, query_sequence: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
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
                    "point_index": point_index,
                    "window_start_index": window_start_index,
                    "window_end_index": window_end_index,
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
