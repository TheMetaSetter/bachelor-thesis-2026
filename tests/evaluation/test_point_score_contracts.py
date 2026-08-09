from __future__ import annotations

import numpy as np
import pytest
import torch

from src.protocols.point_score_calibration import (
    POINT_SCORE_MAD_NORMALIZER,
    PointScoreCalibration,
    fit_mad_logistic_calibration,
    transform_point_scores,
)
from src.protocols.point_scores import (
    average_overlapping_point_scores,
    window_scores_to_causal_endpoint_scores,
)


def test_mad_logistic_calibration_uses_raw_scores() -> None:
    calibration = fit_mad_logistic_calibration(np.asarray([1.0, 2.0, 3.0, 4.0]))

    assert calibration.center == 2.5
    assert calibration.tau == pytest.approx(1.0 / POINT_SCORE_MAD_NORMALIZER)
    transformed = transform_point_scores(np.asarray([1.0, 2.5, 4.0]), calibration)
    assert np.all(np.diff(transformed) > 0.0)
    assert np.all((transformed > 0.0) & (transformed < 1.0))


def test_transform_preserves_torch_shape_and_formula() -> None:
    calibration = PointScoreCalibration(center=1.0, tau=0.5)
    raw_scores = torch.tensor([[0.5, 1.0, 1.5]])

    transformed = transform_point_scores(raw_scores, calibration)

    assert transformed.shape == raw_scores.shape
    assert torch.allclose(transformed, torch.sigmoid((raw_scores - 1.0) / 0.5))


@pytest.mark.parametrize(
    "raw_scores",
    [np.asarray([]), np.asarray([1.0, np.nan]), np.asarray([2.0, 2.0])],
)
def test_mad_logistic_calibration_rejects_invalid_scale_inputs(raw_scores) -> None:
    with pytest.raises(ValueError):
        fit_mad_logistic_calibration(raw_scores)


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
