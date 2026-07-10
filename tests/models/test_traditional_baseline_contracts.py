from __future__ import annotations

import numpy as np

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
