from __future__ import annotations

"""Window-level Isolation Forest baseline with clean-validation calibration."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

from src.baselines.traditional.base import (
    RobustPointCalibration,
    TraditionalBaselineProtocol,
    build_window_matrix,
    fit_robust_point_calibration,
    pointify_nonoverlap_tail_window_scores,
    transform_point_scores,
)


@dataclass(frozen=True)
class IsolationForestCalibration:
    robust_point_calibration: RobustPointCalibration
    method_metadata: dict[str, Any]
    validation_window_scores: np.ndarray


class IForestWindowBaseline(TraditionalBaselineProtocol):
    def __init__(
        self,
        *,
        window_size: int = 20,
        n_estimators: int = 100,
        max_samples: str | int | float = "auto",
        max_features: float = 1.0,
        contamination: str | float = "auto",
        normalize_windows: bool = True,
        threshold_quantile: float = 0.99,
        random_state: int = 0,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if not 0.0 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0, 1)")
        self.window_size = int(window_size)
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.max_features = max_features
        self.contamination = contamination
        self.normalize_windows = bool(normalize_windows)
        self.threshold_quantile = float(threshold_quantile)
        self.random_state = int(random_state)
        self.model_: IsolationForest | None = None
        self.calibration_: IsolationForestCalibration | None = None

    def fit(self, train_sequence: np.ndarray) -> "IForestWindowBaseline":
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
            random_state=self.random_state,
        )
        self.model_.fit(window_matrix)
        self.calibration_ = None
        return self

    def _score_windows(self, query_sequence: np.ndarray) -> tuple[np.ndarray, int]:
        if self.model_ is None:
            raise RuntimeError("Call fit() before score_sequence().")
        window_matrix, _ = build_window_matrix(
            query_sequence,
            window_size=self.window_size,
            normalize=self.normalize_windows,
        )
        if window_matrix.shape[0] == 0:
            return np.zeros(0, dtype=np.float64), int(
                np.asarray(query_sequence).shape[0]
            )
        # IsolationForest returns larger values for normal points, so invert them.
        anomaly_scores = -self.model_.decision_function(window_matrix)
        return anomaly_scores.astype(np.float64, copy=False), int(
            np.asarray(query_sequence).shape[0]
        )

    def calibrate(self, clean_validation_sequence: np.ndarray) -> dict[str, Any]:
        if self.model_ is None:
            raise RuntimeError("Call fit() before calibrate().")
        validation_window_scores, sequence_length = self._score_windows(
            clean_validation_sequence
        )
        point_scores, covered_mask = pointify_nonoverlap_tail_window_scores(
            sequence_length=sequence_length,
            window_scores=validation_window_scores,
            window_size=self.window_size,
        )
        robust_point_calibration = fit_robust_point_calibration(
            validation_point_scores=point_scores,
            threshold_quantile=self.threshold_quantile,
        )
        self.calibration_ = IsolationForestCalibration(
            robust_point_calibration=robust_point_calibration,
            method_metadata={
                "window_normalization": "per_window_row_zscore_ddof1"
                if self.normalize_windows
                else "none",
                "point_calibration": "clean_validation_median_iqr",
                "anomaly_score_sign": "negative_decision_function",
            },
            validation_window_scores=validation_window_scores,
        )
        return {
            "threshold": robust_point_calibration.threshold,
            "point_center": robust_point_calibration.center,
            "point_scale": robust_point_calibration.scale,
            "validation_point_scores": robust_point_calibration.validation_point_scores,
            "validation_covered_mask": robust_point_calibration.validation_covered_mask,
            "validation_window_scores": validation_window_scores,
            "method_metadata": self.calibration_.method_metadata,
            "validation_point_coverage": covered_mask,
        }

    def score_sequence(self, query_sequence: np.ndarray) -> np.ndarray:
        if self.calibration_ is None:
            raise RuntimeError("Call calibrate() before score_sequence().")
        window_scores, sequence_length = self._score_windows(query_sequence)
        point_scores, _ = pointify_nonoverlap_tail_window_scores(
            sequence_length=sequence_length,
            window_scores=window_scores,
            window_size=self.window_size,
        )
        return transform_point_scores(
            point_scores, self.calibration_.robust_point_calibration
        )
