from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def build_nonoverlap_tail_window_starts(
    sequence_length: int,
    window_size: int,
) -> list[int]:
    """Build non-overlap starts plus one final end-aligned tail window.

    ₍^. .^₎⟆ Offline benchmark path

    sequence
      -> regular starts: 0, L, 2L, ...
      -> one tail start if the final points are not covered
      -> score windows
      -> average only the tail-overlap points
    """
    if sequence_length < window_size:
        return []

    starts = list(range(0, sequence_length - window_size + 1, window_size))
    tail_start = sequence_length - window_size
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def average_overlapping_point_scores(
    sequence_length: int,
    window_scores: Sequence[np.ndarray],
    window_starts: Sequence[int],
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    score_sum = np.zeros(sequence_length, dtype=float)
    score_count = np.zeros(sequence_length, dtype=float)

    for start_index, scores in zip(window_starts, window_scores, strict=True):
        score_array = np.asarray(scores, dtype=float)
        if score_array.shape[0] != window_size:
            raise ValueError("Each window score array must match window_size")
        end_index = min(start_index + window_size, sequence_length)
        usable_length = end_index - start_index
        score_sum[start_index:end_index] += score_array[:usable_length]
        score_count[start_index:end_index] += 1.0

    point_scores = np.full(sequence_length, np.nan, dtype=float)
    covered_mask = score_count > 0.0
    point_scores[covered_mask] = score_sum[covered_mask] / score_count[covered_mask]
    return point_scores, covered_mask


def window_scores_to_causal_endpoint_scores(
    window_scores: Sequence[float],
    sequence_length: int,
    window_size: int,
) -> np.ndarray:
    """Put each stride-1 window score on its endpoint point."""
    point_scores = np.full(sequence_length, np.nan, dtype=float)
    for offset, score in enumerate(window_scores):
        point_index = offset + window_size - 1
        if point_index >= sequence_length:
            break
        point_scores[point_index] = float(score)
    return point_scores


def ewma_scores(
    point_scores: np.ndarray,
    current_weight: float,
    previous_weight: float,
) -> np.ndarray:
    if not np.isclose(current_weight + previous_weight, 1.0):
        raise ValueError("EWMA weights must sum to 1.0")

    raw_scores = np.asarray(point_scores, dtype=float)
    smoothed = np.full(raw_scores.shape, np.nan, dtype=float)
    previous_score: float | None = None

    for index, score in enumerate(raw_scores):
        if np.isnan(score):
            continue
        if previous_score is None:
            smoothed[index] = score
        else:
            smoothed[index] = current_weight * score + previous_weight * previous_score
        previous_score = float(smoothed[index])

    return smoothed
