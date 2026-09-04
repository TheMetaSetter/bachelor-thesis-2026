from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.visualization.plot_raw_input_mse_histograms import (
    build_histogram_summary,
    load_raw_score_arrays,
)
from src.protocols.threshold_artifact import build_threshold_artifact


def _raw_artifact() -> dict[str, object]:
    return build_threshold_artifact(
        method_name="THESIS",
        variant_name="A2",
        entity_id="machine-1-6",
        seed=6,
        window_size=20,
        offline_point_threshold=2.0,
        online_ewma_point_threshold=3.0,
        quantile=0.99,
        ewma_current_weight=0.9,
        ewma_previous_weight=0.1,
        created_by="pytest",
        config_path="raw.yaml",
        checkpoint_sha256="checkpoint-sha",
        score_space="raw_input",
        input_window_threshold=4.0,
        latent_window_low_threshold=5.0,
        latent_window_high_threshold=6.0,
    )


def test_histogram_loader_separates_point_and_window_label_categories(tmp_path) -> None:
    score_path = tmp_path / "scores.npz"
    np.savez(
        score_path,
        raw_input_point_mse=np.array([0.1, 2.5, 0.2, 3.0]),
        point_labels=np.array([0, 1, 0, 1]),
        raw_input_window_mse=np.array([0.3, 4.5, 0.4]),
        window_labels=np.array([0, 1, 0]),
    )
    arrays = load_raw_score_arrays(score_path, _raw_artifact())

    summary = build_histogram_summary(
        arrays,
        point_threshold=2.0,
        window_threshold=4.0,
    )

    assert summary["point"]["normal_count"] == 2
    assert summary["point"]["anomalous_count"] == 2
    assert summary["window"]["normal_count"] == 2
    assert summary["window"]["anomalous_count"] == 1
    assert summary["point"]["above_threshold_count"] == 2
    assert summary["window"]["above_threshold_count"] == 1


def test_histogram_loader_rejects_non_finite_score(tmp_path) -> None:
    score_path = tmp_path / "scores.npz"
    np.savez(
        score_path,
        raw_input_point_mse=np.array([np.nan]),
        point_labels=np.array([0]),
        raw_input_window_mse=np.array([0.1]),
        window_labels=np.array([0]),
    )

    with pytest.raises(ValueError, match="finite"):
        load_raw_score_arrays(score_path, _raw_artifact())
