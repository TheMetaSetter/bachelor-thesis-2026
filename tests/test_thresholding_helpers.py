from __future__ import annotations

import numpy as np

from src.engine.thresholding import (
    build_checkpoint_evaluation_metadata,
    resolve_evaluation_threshold,
)


def test_resolve_evaluation_threshold_uses_positive_support_when_available() -> None:
    threshold, threshold_source = resolve_evaluation_threshold(
        np.asarray([0.0, 0.0, 1.0, 2.0], dtype=np.float32),
        quantile=0.5,
    )

    assert threshold > 0.0
    assert threshold_source == "positive_support_quantile_0.5"


def test_build_checkpoint_evaluation_metadata_keeps_base_state_when_threshold_missing() -> (
    None
):
    metadata = build_checkpoint_evaluation_metadata(
        checkpoint_monitor_metric="val_loss",
        epoch_metrics={"val_loss": 1.0},
        base_extra_state={"memory_initialized": False},
    )

    assert metadata == {"memory_initialized": False}
