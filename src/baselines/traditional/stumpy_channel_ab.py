from __future__ import annotations

"""Fair STUMPY channel-wise AB-join baseline.

₍^. .^₎⟆ Baseline flow

train reference
  -> per-channel STUMPY AB-join
  -> robust clean-val channel calibration
  -> non-overlap tail window scores
  -> point-level offline benchmark scores
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import stumpy

from src.baselines.traditional.base import TraditionalBaselineProtocol
from src.protocols.point_scores import (
    average_overlapping_point_scores,
    build_nonoverlap_tail_window_starts,
)


def _as_2d_sequence(sequence: np.ndarray) -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Expected a 2D array with shape [T, D]")
    return array


def _extract_profile_scores(profile: Any) -> np.ndarray:
    if hasattr(profile, "P_"):
        return np.asarray(profile.P_, dtype=np.float64)
    return np.asarray(profile[:, 0], dtype=np.float64)


def _channel_subsequence_scores(
    query_channel: np.ndarray,
    reference_channel: np.ndarray,
    window_size: int,
    normalize: bool,
    p: float,
) -> np.ndarray:
    if query_channel.shape[0] < window_size:
        return np.zeros(0, dtype=np.float64)
    if np.isclose(np.nanstd(query_channel), 0.0) or np.isclose(
        np.nanstd(reference_channel), 0.0
    ):
        subsequence_count = query_channel.shape[0] - window_size + 1
        return np.zeros(subsequence_count, dtype=np.float64)

    profile = stumpy.stump(
        T_A=np.asarray(query_channel, dtype=np.float64),
        m=window_size,
        T_B=np.asarray(reference_channel, dtype=np.float64),
        ignore_trivial=False,
        normalize=normalize,
        p=p,
    )
    return _extract_profile_scores(profile)


def compute_stumpy_channel_ab_subsequence_scores(
    query_sequence: np.ndarray,
    reference_sequence: np.ndarray,
    window_size: int,
    normalize: bool = True,
    p: float = 2.0,
) -> np.ndarray:
    query_array = _as_2d_sequence(query_sequence)
    reference_array = _as_2d_sequence(reference_sequence)
    if query_array.shape[1] != reference_array.shape[1]:
        raise ValueError("query and reference sequences must have the same channels")

    channel_scores: list[np.ndarray] = []
    for channel_index in range(query_array.shape[1]):
        channel_scores.append(
            _channel_subsequence_scores(
                query_channel=query_array[:, channel_index],
                reference_channel=reference_array[:, channel_index],
                window_size=window_size,
                normalize=normalize,
                p=p,
            )
        )
    if not channel_scores:
        return np.zeros((0, 0), dtype=np.float64)
    return np.stack(channel_scores, axis=1)


def _fit_channel_calibration(
    validation_subsequence_scores: np.ndarray,
    eps: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(validation_subsequence_scores, axis=0)
    q75 = np.nanpercentile(validation_subsequence_scores, 75, axis=0)
    q25 = np.nanpercentile(validation_subsequence_scores, 25, axis=0)
    iqr = np.maximum(q75 - q25, eps)
    return med, iqr


def _transform_channel_scores(
    subsequence_scores: np.ndarray,
    channel_median: np.ndarray,
    channel_iqr: np.ndarray,
) -> np.ndarray:
    return (subsequence_scores - channel_median[None, :]) / channel_iqr[None, :]


def _aggregate_channel_scores(subsequence_scores_z: np.ndarray) -> np.ndarray:
    return np.nanmax(subsequence_scores_z, axis=1)


def _window_arrays_from_scores(
    window_scores: np.ndarray,
    window_size: int,
) -> list[np.ndarray]:
    return [
        np.full(window_size, float(score), dtype=np.float64) for score in window_scores
    ]


def _pointify_offline_subsequence_scores(
    sequence_length: int,
    subsequence_scores: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    window_starts = build_nonoverlap_tail_window_starts(sequence_length, window_size)
    if not window_starts:
        return (
            np.full(sequence_length, np.nan, dtype=np.float64),
            np.zeros(sequence_length, dtype=bool),
        )
    window_scores = np.asarray(subsequence_scores, dtype=np.float64)[window_starts]
    point_scores, covered_mask = average_overlapping_point_scores(
        sequence_length=sequence_length,
        window_scores=_window_arrays_from_scores(window_scores, window_size),
        window_starts=window_starts,
        window_size=window_size,
    )
    return point_scores, covered_mask


@dataclass
class StumpyChannelABCalibration:
    channel_median: np.ndarray
    channel_iqr: np.ndarray
    threshold: float
    validation_point_scores: np.ndarray
    validation_covered_mask: np.ndarray


class StumpyChannelABFrozenTrainRef(TraditionalBaselineProtocol):
    def __init__(
        self,
        window_size: int = 20,
        normalize: bool = True,
        p: float = 2.0,
        threshold_quantile: float = 0.99,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0, 1)")
        self.window_size = int(window_size)
        self.normalize = bool(normalize)
        self.p = float(p)
        self.threshold_quantile = float(threshold_quantile)
        self.reference_sequence_: np.ndarray | None = None
        self.calibration_: StumpyChannelABCalibration | None = None

    def fit(self, train_sequence: np.ndarray) -> "StumpyChannelABFrozenTrainRef":
        self.reference_sequence_ = _as_2d_sequence(train_sequence).copy()
        self.calibration_ = None
        return self

    def calibrate(self, clean_validation_sequence: np.ndarray) -> dict[str, Any]:
        if self.reference_sequence_ is None:
            raise RuntimeError("Call fit() before calibrate().")
        validation_sequence = _as_2d_sequence(clean_validation_sequence)
        validation_subsequence_scores = compute_stumpy_channel_ab_subsequence_scores(
            query_sequence=validation_sequence,
            reference_sequence=self.reference_sequence_,
            window_size=self.window_size,
            normalize=self.normalize,
            p=self.p,
        )
        channel_median, channel_iqr = _fit_channel_calibration(
            validation_subsequence_scores
        )
        validation_subsequence_window_scores = _aggregate_channel_scores(
            _transform_channel_scores(
                validation_subsequence_scores,
                channel_median,
                channel_iqr,
            )
        )
        validation_point_scores, validation_covered_mask = (
            _pointify_offline_subsequence_scores(
                sequence_length=validation_sequence.shape[0],
                subsequence_scores=validation_subsequence_window_scores,
                window_size=self.window_size,
            )
        )
        covered_point_scores = validation_point_scores[validation_covered_mask]
        if covered_point_scores.size == 0:
            raise ValueError(
                "Validation sequence must produce at least one covered point"
            )
        threshold = float(np.nanquantile(covered_point_scores, self.threshold_quantile))
        calibration = StumpyChannelABCalibration(
            channel_median=channel_median,
            channel_iqr=channel_iqr,
            threshold=threshold,
            validation_point_scores=validation_point_scores,
            validation_covered_mask=validation_covered_mask,
        )
        self.calibration_ = calibration
        return {
            "threshold": threshold,
            "channel_median": channel_median,
            "channel_iqr": channel_iqr,
            "validation_point_scores": validation_point_scores,
            "validation_covered_mask": validation_covered_mask,
        }

    def score_sequence(self, query_sequence: np.ndarray) -> np.ndarray:
        if self.reference_sequence_ is None:
            raise RuntimeError("Call fit() before score_sequence().")
        if self.calibration_ is None:
            raise RuntimeError("Call calibrate() before score_sequence().")

        query_array = _as_2d_sequence(query_sequence)
        subsequence_scores = compute_stumpy_channel_ab_subsequence_scores(
            query_sequence=query_array,
            reference_sequence=self.reference_sequence_,
            window_size=self.window_size,
            normalize=self.normalize,
            p=self.p,
        )
        if subsequence_scores.shape[0] == 0:
            return np.full(query_array.shape[0], np.nan, dtype=np.float64)
        aggregated_subsequence_scores = _aggregate_channel_scores(
            _transform_channel_scores(
                subsequence_scores,
                self.calibration_.channel_median,
                self.calibration_.channel_iqr,
            )
        )
        point_scores, _ = _pointify_offline_subsequence_scores(
            sequence_length=query_array.shape[0],
            subsequence_scores=aggregated_subsequence_scores,
            window_size=self.window_size,
        )
        return point_scores
