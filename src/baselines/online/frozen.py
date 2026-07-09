from __future__ import annotations

"""Frozen streaming baselines for the online benchmark.

₍^. .^₎⟆ Frozen stream flow

train sequence
  -> build a reference scorer
  -> stride-1 query windows
  -> causal endpoint scores
  -> EWMA threshold calibration
  -> no parameter updates on test stream
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import stumpy
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

from src.baselines.online.base import (
    OnlineStreamingBaselineProtocol,
    as_2d_sequence,
    build_online_thresholds,
    causal_point_scores_from_windows,
    build_stride1_window_matrix,
    smooth_point_scores,
)
from src.baselines.traditional.base import build_window_matrix
from src.baselines.traditional.stumpy_channel_ab import (
    compute_stumpy_channel_ab_subsequence_scores,
)
from src.engine.online_tta.triage import classify_online_window
from src.protocols.threshold_artifact import build_threshold_artifact


def _sequence_metadata(sequence: dict[str, Any]) -> dict[str, Any]:
    return dict(sequence.get("meta", {}))


def _finite_values(array: np.ndarray) -> np.ndarray:
    flat_array = np.asarray(array, dtype=np.float64).reshape(-1)
    return flat_array[np.isfinite(flat_array)]


@dataclass(frozen=True)
class FrozenStreamingCalibration:
    threshold_artifact: dict[str, Any]
    threshold_value: float
    threshold_source: str
    validation_point_scores: np.ndarray
    validation_ewma_scores: np.ndarray
    method_metadata: dict[str, Any]


class _FrozenStreamingBaseline(OnlineStreamingBaselineProtocol):
    method_name = "frozen_streaming"

    def __init__(
        self,
        *,
        train_sequence: np.ndarray,
        window_size: int = 20,
        threshold_quantile: float = 0.99,
        online_variant: str = "A0",
        seed: int = 0,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0, 1)")
        self.window_size = int(window_size)
        self.threshold_quantile = float(threshold_quantile)
        self.online_variant = str(online_variant)
        self.seed = int(seed)
        self.reference_sequence_: np.ndarray | None = None
        self.calibration_: FrozenStreamingCalibration | None = None
        self.fit(train_sequence)

    def fit(self, train_sequence: np.ndarray) -> "_FrozenStreamingBaseline":
        self.reference_sequence_ = as_2d_sequence(train_sequence).copy()
        self._fit_reference(self.reference_sequence_)
        self.calibration_ = None
        return self

    def _fit_reference(self, train_sequence: np.ndarray) -> None:
        raise NotImplementedError

    def _score_window_scores(self, query_sequence: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _score_sequence(self, query_sequence: np.ndarray) -> np.ndarray:
        query_array = as_2d_sequence(query_sequence)
        window_scores = self._score_window_scores(query_array)
        return causal_point_scores_from_windows(
            window_scores=window_scores,
            sequence_length=query_array.shape[0],
            window_size=self.window_size,
        )

    def calibrate(
        self,
        *,
        clean_validation_sequences: list[dict[str, Any]],
        protocol_config: dict[str, Any],
        device: str,
    ) -> dict[str, Any]:
        del device
        if self.reference_sequence_ is None:
            raise RuntimeError("Call fit() before calibrate().")
        if not clean_validation_sequences:
            raise ValueError("clean_validation_sequences must not be empty")

        current_weight = float(protocol_config["online_ewma_current_weight"])
        previous_weight = float(protocol_config["online_ewma_previous_weight"])
        validation_point_scores: list[float] = []
        validation_ewma_scores: list[float] = []
        for sequence in clean_validation_sequences:
            raw_point_scores = self._score_sequence(sequence["x"])
            smoothed_scores = smooth_point_scores(
                raw_point_scores,
                current_weight=current_weight,
                previous_weight=previous_weight,
            )
            validation_point_scores.extend(_finite_values(raw_point_scores).tolist())
            validation_ewma_scores.extend(_finite_values(smoothed_scores).tolist())

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
        self.calibration_ = FrozenStreamingCalibration(
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
        raw_point_scores = self._score_sequence(sequence["x"])
        current_weight = float(protocol_config["online_ewma_current_weight"])
        previous_weight = float(protocol_config["online_ewma_previous_weight"])
        ewma_point_scores = smooth_point_scores(
            raw_point_scores,
            current_weight=current_weight,
            previous_weight=previous_weight,
        )

        metric_history: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        entity_id = str(sequence["meta"]["entity_id"])
        for step_index, (raw_score, ewma_score) in enumerate(
            zip(raw_point_scores, ewma_point_scores, strict=True)
        ):
            if np.isnan(ewma_score):
                continue
            prediction = int(ewma_score > float(threshold_value))
            triage_decision = classify_online_window(
                input_window_score=float(raw_score),
                latent_window_score=float(raw_score),
                thresholds=build_online_thresholds(threshold_value=threshold_value),
            )
            window_end_index = step_index + 1
            window_start_index = window_end_index - self.window_size
            record = {
                "entity_id": entity_id,
                "point_index": window_end_index - 1,
                "window_start_index": window_start_index,
                "window_end_index": window_end_index,
                "raw_point_score": float(raw_score),
                "ewma_point_score": float(ewma_score),
                "latent_window_score": float(raw_score),
                "threshold": float(threshold_value),
                "prediction": prediction,
                "online_variant": self.online_variant,
                "triage_decision": triage_decision,
                "did_update": False,
                "loss_total": None,
            }
            metric_history.append(
                {
                    "online/step": len(metric_history) + 1,
                    "online/raw_point_score": float(raw_score),
                    "online/ewma_point_score": float(ewma_score),
                    "online/threshold": float(threshold_value),
                    "online/prediction": prediction,
                    "online/did_update": False,
                    "online/loss_total": None,
                    "online/triage_decision": triage_decision,
                    "online/verification_buffer_size": 0,
                    "online/ttl_buffer_size": 0,
                }
            )
            records.append(record)
        return metric_history, records

    def _method_metadata(self) -> dict[str, Any]:
        return {"method": self.method_name, "online_variant": self.online_variant}


class StumpyChannelABStreamingBaseline(_FrozenStreamingBaseline):
    method_name = "stumpy"

    def __init__(
        self,
        *,
        train_sequence: np.ndarray,
        window_size: int = 20,
        normalize: bool = True,
        p: float = 2.0,
        threshold_quantile: float = 0.99,
        online_variant: str = "A0",
        seed: int = 0,
    ) -> None:
        self.normalize = bool(normalize)
        self.p = float(p)
        super().__init__(
            train_sequence=train_sequence,
            window_size=window_size,
            threshold_quantile=threshold_quantile,
            online_variant=online_variant,
            seed=seed,
        )

    def _fit_reference(self, train_sequence: np.ndarray) -> None:
        self.reference_sequence_ = as_2d_sequence(train_sequence).copy()

    def _score_window_scores(self, query_sequence: np.ndarray) -> np.ndarray:
        assert self.reference_sequence_ is not None
        query_array = as_2d_sequence(query_sequence)
        if query_array.shape[0] < self.window_size:
            return np.zeros(0, dtype=np.float64)
        channel_scores = compute_stumpy_channel_ab_subsequence_scores(
            query_sequence=query_array,
            reference_sequence=self.reference_sequence_,
            window_size=self.window_size,
            normalize=self.normalize,
            p=self.p,
        )
        if channel_scores.size == 0:
            return np.zeros(0, dtype=np.float64)
        return np.nanmax(channel_scores, axis=1)

    def _method_metadata(self) -> dict[str, Any]:
        return {
            "method": self.method_name,
            "normalize": self.normalize,
            "p": self.p,
            "aggregation": "channel_max",
            "online_variant": self.online_variant,
        }


class KMeansADStreamingBaseline(_FrozenStreamingBaseline):
    method_name = "kmeans_ad"

    def __init__(
        self,
        *,
        train_sequence: np.ndarray,
        window_size: int = 20,
        n_clusters: int = 20,
        normalize_windows: bool = True,
        threshold_quantile: float = 0.99,
        online_variant: str = "A0",
        seed: int = 0,
    ) -> None:
        self.n_clusters = int(n_clusters)
        self.normalize_windows = bool(normalize_windows)
        self.model_: KMeans | None = None
        super().__init__(
            train_sequence=train_sequence,
            window_size=window_size,
            threshold_quantile=threshold_quantile,
            online_variant=online_variant,
            seed=seed,
        )

    def _fit_reference(self, train_sequence: np.ndarray) -> None:
        window_matrix, _ = build_window_matrix(
            train_sequence,
            window_size=self.window_size,
            normalize=self.normalize_windows,
        )
        if window_matrix.shape[0] == 0:
            raise ValueError("train_sequence must be long enough for one window")
        n_clusters = min(self.n_clusters, window_matrix.shape[0])
        self.model_ = KMeans(n_clusters=n_clusters, n_init=10, random_state=self.seed)
        self.model_.fit(window_matrix)

    def _score_window_scores(self, query_sequence: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() before score_sequence().")
        window_matrix, _ = build_stride1_window_matrix(
            query_sequence,
            window_size=self.window_size,
            normalize=self.normalize_windows,
        )
        if window_matrix.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        clusters = self.model_.predict(window_matrix)
        distances = np.linalg.norm(
            window_matrix - self.model_.cluster_centers_[clusters],
            axis=1,
        )
        return distances.astype(np.float64, copy=False)

    def _method_metadata(self) -> dict[str, Any]:
        return {
            "method": self.method_name,
            "window_normalization": (
                "per_window_row_zscore_ddof1" if self.normalize_windows else "none"
            ),
            "online_variant": self.online_variant,
        }


class IForestStreamingBaseline(_FrozenStreamingBaseline):
    method_name = "iforest"

    def __init__(
        self,
        *,
        train_sequence: np.ndarray,
        window_size: int = 20,
        n_estimators: int = 100,
        max_samples: str | int | float = "auto",
        max_features: float = 1.0,
        contamination: str | float = "auto",
        normalize_windows: bool = True,
        threshold_quantile: float = 0.99,
        online_variant: str = "A0",
        seed: int = 0,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.max_features = max_features
        self.contamination = contamination
        self.normalize_windows = bool(normalize_windows)
        self.model_: IsolationForest | None = None
        super().__init__(
            train_sequence=train_sequence,
            window_size=window_size,
            threshold_quantile=threshold_quantile,
            online_variant=online_variant,
            seed=seed,
        )

    def _fit_reference(self, train_sequence: np.ndarray) -> None:
        window_matrix, _ = build_window_matrix(
            train_sequence,
            window_size=self.window_size,
            normalize=self.normalize_windows,
        )
        if window_matrix.shape[0] == 0:
            raise ValueError("train_sequence must be long enough for one window")
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            contamination=self.contamination,
            random_state=self.seed,
        )
        self.model_.fit(window_matrix)

    def _score_window_scores(self, query_sequence: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() before score_sequence().")
        window_matrix, _ = build_stride1_window_matrix(
            query_sequence,
            window_size=self.window_size,
            normalize=self.normalize_windows,
        )
        if window_matrix.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        return (-self.model_.decision_function(window_matrix)).astype(
            np.float64, copy=False
        )

    def _method_metadata(self) -> dict[str, Any]:
        return {
            "method": self.method_name,
            "window_normalization": (
                "per_window_row_zscore_ddof1" if self.normalize_windows else "none"
            ),
            "anomaly_score_sign": "negative_decision_function",
            "online_variant": self.online_variant,
        }
