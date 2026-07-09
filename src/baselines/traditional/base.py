from __future__ import annotations

"""Shared protocol and small helpers for traditional offline baselines.

₍^. .^₎⟆ Traditional baseline flow

raw sequence
  -> non-overlap tail windows
  -> method-specific window scorer
  -> point-level calibration on clean validation
  -> point scores for benchmark artifacts
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.protocols.point_scores import (
    average_overlapping_point_scores,
    build_nonoverlap_tail_window_starts,
)


@runtime_checkable
class TraditionalBaselineProtocol(Protocol):
    def fit(self, train_sequence: np.ndarray) -> "TraditionalBaselineProtocol": ...

    def calibrate(self, clean_validation_sequence: np.ndarray) -> dict[str, Any]: ...

    def score_sequence(self, query_sequence: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class RobustPointCalibration:
    center: float
    scale: float
    threshold: float
    validation_point_scores: np.ndarray
    validation_covered_mask: np.ndarray


def as_2d_sequence(sequence: np.ndarray) -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Expected a 2D array with shape [T, D]")
    return array


def _row_standardize(matrix: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    row_mean = np.nanmean(matrix, axis=1, keepdims=True)
    row_std = np.nanstd(matrix, axis=1, keepdims=True, ddof=1)
    row_std = np.where(np.isfinite(row_std) & (row_std >= eps), row_std, 1.0)
    return (matrix - row_mean) / row_std


def build_window_matrix(
    sequence: np.ndarray,
    window_size: int,
    normalize: bool,
) -> tuple[np.ndarray, list[int]]:
    sequence_array = as_2d_sequence(sequence)
    window_starts = build_nonoverlap_tail_window_starts(
        sequence_length=sequence_array.shape[0],
        window_size=window_size,
    )
    if not window_starts:
        return np.zeros(
            (0, window_size * sequence_array.shape[1]), dtype=np.float64
        ), []

    window_rows = [
        sequence_array[start_index : start_index + window_size].reshape(-1)
        for start_index in window_starts
    ]
    window_matrix = np.stack(window_rows, axis=0).astype(np.float64, copy=False)
    if normalize:
        window_matrix = _row_standardize(window_matrix)
    return window_matrix, window_starts


def pointify_nonoverlap_tail_window_scores(
    *,
    sequence_length: int,
    window_scores: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    window_starts = build_nonoverlap_tail_window_starts(sequence_length, window_size)
    if len(window_starts) != int(np.asarray(window_scores).shape[0]):
        raise ValueError("window_scores length must match the derived window starts")
    window_score_arrays = [
        np.full(window_size, float(score), dtype=np.float64)
        for score in np.asarray(window_scores, dtype=np.float64).reshape(-1)
    ]
    return average_overlapping_point_scores(
        sequence_length=sequence_length,
        window_scores=window_score_arrays,
        window_starts=window_starts,
        window_size=window_size,
    )


def fit_robust_point_calibration(
    *,
    validation_point_scores: np.ndarray,
    threshold_quantile: float,
    eps: float = 1.0e-8,
) -> RobustPointCalibration:
    point_scores = np.asarray(validation_point_scores, dtype=np.float64).reshape(-1)
    covered_mask = np.isfinite(point_scores)
    if not np.any(covered_mask):
        raise ValueError(
            "Validation point scores must contain at least one finite value"
        )

    covered_scores = point_scores[covered_mask]
    center = float(np.nanmedian(covered_scores))
    q75 = float(np.nanpercentile(covered_scores, 75))
    q25 = float(np.nanpercentile(covered_scores, 25))
    scale = max(q75 - q25, eps)
    transformed = (point_scores - center) / scale
    threshold = float(np.nanquantile(transformed[covered_mask], threshold_quantile))
    return RobustPointCalibration(
        center=center,
        scale=scale,
        threshold=threshold,
        validation_point_scores=transformed,
        validation_covered_mask=covered_mask,
    )


def transform_point_scores(
    point_scores: np.ndarray,
    calibration: RobustPointCalibration,
) -> np.ndarray:
    return (
        np.asarray(point_scores, dtype=np.float64).reshape(-1) - calibration.center
    ) / calibration.scale
