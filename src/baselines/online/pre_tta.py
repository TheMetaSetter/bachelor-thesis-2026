from __future__ import annotations

"""Pre-TTA score collection for the reference adapter benchmark flow."""

from typing import Any

import numpy as np
import torch

from src.baselines.online.adaptive import AdaptiveStreamingBaselineBase, _window_matrix


def score_sequence_before_adaptation(
    baseline: AdaptiveStreamingBaselineBase,
    sequence: dict[str, Any],
) -> np.ndarray:
    """Score all windows without changing model, optimizer or adapter state."""
    sequence_array = np.asarray(sequence["x"], dtype=np.float64)
    windows = _window_matrix(sequence_array, baseline.window_size)
    if windows.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    scores: list[float] = []
    for batch_start in range(0, windows.shape[0], baseline.adaptation_batch_size):
        batch = torch.as_tensor(
            windows[batch_start : batch_start + baseline.adaptation_batch_size],
            dtype=torch.float32,
        )
        batch_scores, _ = baseline._score_tensor_batch(batch)
        scores.extend(batch_scores.tolist())
    return np.asarray(scores, dtype=np.float64)
