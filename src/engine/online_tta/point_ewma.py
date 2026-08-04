"""Point-wise EWMA state for one active causal window."""

from __future__ import annotations

import torch


def update_window_point_ewma(
    *,
    previous_scores: dict[int, float],
    absolute_indices: torch.Tensor,
    window_point_scores: torch.Tensor,
    current_weight: float,
    previous_weight: float,
) -> tuple[torch.Tensor, dict[int, float]]:
    """Update only points in the current causal window and return its EWMA."""
    if absolute_indices.ndim != 1 or window_point_scores.ndim != 1:
        raise ValueError("point EWMA expects one-dimensional index and score vectors")
    if absolute_indices.shape != window_point_scores.shape:
        raise ValueError("point EWMA index and score vectors must have the same shape")
    current_values: list[float] = []
    active_scores: dict[int, float] = {}
    for index, score in zip(absolute_indices.tolist(), window_point_scores.tolist()):
        point_index = int(index)
        current_score = float(score)
        previous_score = previous_scores.get(point_index)
        ewma_score = current_score if previous_score is None else (
            previous_weight * previous_score + current_weight * current_score
        )
        current_values.append(ewma_score)
        active_scores[point_index] = ewma_score
    return window_point_scores.new_tensor(current_values), active_scores
