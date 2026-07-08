from __future__ import annotations

import numpy as np

from src.engine.thresholding import (
    select_clean_validation_point_threshold,
    select_online_ewma_threshold,
)
from src.protocols.point_scores import ewma_scores


def test_ewma_keeps_nan_warmup_and_uses_previous_score_after_warmup() -> None:
    raw_scores = np.array([np.nan, np.nan, 1.0, 3.0, 5.0], dtype=float)

    smoothed = ewma_scores(raw_scores, current_weight=0.9, previous_weight=0.1)

    assert np.isnan(smoothed[:2]).all()
    assert np.allclose(smoothed[2:], [1.0, 2.8, 4.78])


def test_online_threshold_uses_nan_safe_quantile() -> None:
    ewma_point_scores = np.array([np.nan, np.nan, 1.0, 2.0, 3.0], dtype=float)

    threshold = select_online_ewma_threshold(ewma_point_scores, quantile=0.5)

    assert threshold == 2.0


def test_clean_validation_threshold_uses_nan_safe_quantile() -> None:
    point_scores = np.array([np.nan, 0.0, 1.0, 2.0, 3.0], dtype=float)

    threshold = select_clean_validation_point_threshold(point_scores, quantile=0.5)

    assert threshold == 1.5
