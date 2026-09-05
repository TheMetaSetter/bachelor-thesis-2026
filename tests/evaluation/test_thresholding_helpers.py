from __future__ import annotations

import numpy as np
import pytest

from src.engine.thresholding import (
    build_checkpoint_evaluation_metadata,
    resolve_evaluation_threshold,
    select_synthetic_validation_normal_point_threshold,
)


def test_synthetic_normal_threshold_uses_only_finite_normal_scores() -> None:
    threshold = select_synthetic_validation_normal_point_threshold(
        np.asarray([1.0, 2.0, 100.0, np.nan, np.inf, 3.0]),
        np.asarray([0, 0, 1, 0, 0, 0]),
        quantile=0.99,
    )

    assert threshold == np.quantile(np.asarray([1.0, 2.0, 3.0]), 0.99)


def test_synthetic_normal_threshold_rejects_empty_or_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        select_synthetic_validation_normal_point_threshold(
            np.asarray([1.0]), np.asarray([0, 0]), quantile=0.99
        )

    with pytest.raises(ValueError, match="normal finite"):
        select_synthetic_validation_normal_point_threshold(
            np.asarray([np.nan, np.inf]), np.asarray([0, 0]), quantile=0.99
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
