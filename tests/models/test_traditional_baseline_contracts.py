from __future__ import annotations

import numpy as np

from src.baselines.traditional.iforest import IForestWindowBaseline
from src.baselines.traditional.kmeans_ad import KMeansADWindowBaseline
from src.baselines.traditional.base import TraditionalBaselineProtocol
from src.baselines.traditional.stumpy_channel_ab import StumpyChannelABFrozenTrainRef


def test_stumpy_channel_ab_baseline_satisfies_traditional_protocol() -> None:
    baseline = StumpyChannelABFrozenTrainRef(window_size=20)

    assert isinstance(baseline, TraditionalBaselineProtocol)

    train = np.random.default_rng(0).normal(size=(50, 2))
    clean_validation = np.random.default_rng(1).normal(size=(60, 2))

    baseline.fit(train)
    baseline.calibrate(clean_validation)

    assert isinstance(baseline.score_sequence(clean_validation), np.ndarray)


def test_traditional_baselines_expose_native_point_and_window_scores() -> None:
    train = np.random.default_rng(0).normal(size=(60, 2))
    query = np.random.default_rng(1).normal(size=(60, 2))
    baselines = [
        IForestWindowBaseline(window_size=20, n_estimators=2, random_state=0),
        KMeansADWindowBaseline(window_size=20, n_clusters=2, random_state=0),
        StumpyChannelABFrozenTrainRef(window_size=20),
    ]

    for baseline in baselines:
        baseline.fit(train)
        native = baseline.native_score(query)
        assert set(native) == {
            "window_scores",
            "point_scores",
            "covered_point_mask",
        }
        assert native["window_scores"].shape == (3,)
        assert native["point_scores"].shape == (60,)
        assert native["covered_point_mask"].shape == (60,)
