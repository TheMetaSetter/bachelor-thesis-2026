from __future__ import annotations

import numpy as np

from src.baselines.traditional.stumpy_channel_ab import (
    StumpyChannelABFrozenTrainRef,
    compute_stumpy_channel_ab_subsequence_scores,
)


def test_stumpy_channel_ab_fits_calibrates_and_scores() -> None:
    rng = np.random.default_rng(7)
    train = rng.normal(size=(80, 3))
    clean_validation = rng.normal(size=(95, 3))
    test = rng.normal(size=(95, 3))

    baseline = StumpyChannelABFrozenTrainRef(window_size=20)
    assert baseline.fit(train) is baseline

    calibration = baseline.calibrate(clean_validation)
    assert np.isfinite(calibration["threshold"])
    assert calibration["method_metadata"]["point_calibration"] == (
        "clean_validation_median_iqr"
    )
    assert calibration["validation_point_scores"].shape == (95,)
    assert calibration["validation_covered_mask"].shape == (95,)
    assert calibration["validation_covered_mask"].all()

    test_point_scores = baseline.score_sequence(test)
    assert test_point_scores.shape == (95,)
    assert np.isfinite(test_point_scores).all()


def test_stumpy_channel_ab_constant_channel_is_ignored() -> None:
    train = np.stack(
        [
            np.linspace(0.0, 1.0, 60),
            np.zeros(60),
        ],
        axis=1,
    )
    query = np.stack(
        [
            np.linspace(0.2, 1.2, 60),
            np.zeros(60),
        ],
        axis=1,
    )

    subsequence_scores = compute_stumpy_channel_ab_subsequence_scores(
        query_sequence=query,
        reference_sequence=train,
        window_size=20,
    )

    assert subsequence_scores.shape == (41, 2)
    assert np.allclose(subsequence_scores[:, 1], 0.0)
