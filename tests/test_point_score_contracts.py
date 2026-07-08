from __future__ import annotations

import numpy as np

from src.protocols.point_scores import (
    average_overlapping_point_scores,
    window_scores_to_causal_endpoint_scores,
)


def test_tail_overlap_point_scores_are_averaged() -> None:
    window_scores = [
        np.full(20, 1.0),
        np.full(20, 2.0),
        np.full(20, 3.0),
        np.full(20, 4.0),
        np.full(20, 10.0),
    ]

    point_scores, covered_mask = average_overlapping_point_scores(
        sequence_length=95,
        window_scores=window_scores,
        window_starts=[0, 20, 40, 60, 75],
        window_size=20,
    )

    assert point_scores.shape == (95,)
    assert covered_mask.shape == (95,)
    assert covered_mask.all()
    assert np.allclose(point_scores[74], 4.0)
    assert np.allclose(point_scores[75:80], 7.0)
    assert np.allclose(point_scores[80:95], 10.0)


def test_causal_endpoint_scores_leave_warmup_as_nan() -> None:
    point_scores = window_scores_to_causal_endpoint_scores(
        window_scores=[0.2, 0.4, 0.9],
        sequence_length=7,
        window_size=3,
    )

    assert np.isnan(point_scores[:2]).all()
    assert np.allclose(point_scores[2:5], [0.2, 0.4, 0.9])
    assert np.isnan(point_scores[5:]).all()
