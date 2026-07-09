from __future__ import annotations

import numpy as np

from src.baselines.traditional.kmeans_ad import KMeansADWindowBaseline


def test_kmeansad_window_baseline_fits_calibrates_and_scores() -> None:
    train_sequence = np.stack(
        [
            np.linspace(0.0, 1.0, 40),
            np.linspace(1.0, 0.0, 40),
        ],
        axis=1,
    )
    validation_sequence = train_sequence + 0.05
    query_sequence = train_sequence + 0.1

    baseline = KMeansADWindowBaseline(window_size=10, n_clusters=2)
    baseline.fit(train_sequence)
    calibration = baseline.calibrate(validation_sequence)
    scores = baseline.score_sequence(query_sequence)

    assert scores.shape == (40,)
    assert np.isfinite(scores).all()
    assert np.isfinite(calibration["threshold"])
    assert calibration["method_metadata"]["window_normalization"].startswith(
        "per_window"
    )
