from __future__ import annotations

import numpy as np

from src.engine.thresholding import select_online_ewma_threshold
from src.protocols.point_scores import ewma_scores


def test_online_ewma_threshold_ignores_warmup_nans() -> None:
    point_scores = np.array([np.nan, np.nan, 1.0, 3.0, 5.0], dtype=float)
    ewma = ewma_scores(point_scores, current_weight=0.9, previous_weight=0.1)

    assert np.isnan(ewma[0])
    assert np.isnan(ewma[1])
    assert ewma[2] == 1.0
    assert np.isclose(ewma[3], 2.8)
    assert np.isclose(ewma[4], 4.78)
    assert np.isclose(select_online_ewma_threshold(ewma, quantile=0.5), 2.8)
