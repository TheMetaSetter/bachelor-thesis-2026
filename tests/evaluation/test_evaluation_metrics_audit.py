from __future__ import annotations

import numpy as np

from src.metrics.pointwise import compute_pointwise_metrics


def test_compute_pointwise_metrics_emits_label_and_score_diagnostics() -> None:
    metrics = compute_pointwise_metrics(
        point_labels=np.array([1, 1, 1, 1], dtype=np.int64),
        point_scores=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        threshold=0.25,
    )

    assert metrics["unique_label_count"] == 1
    assert metrics["n_pos"] == 4
    assert metrics["n_neg"] == 0
    assert metrics["positive_ratio"] == 1.0
    assert metrics["score_min"] == 0.1
    assert metrics["score_max"] == 0.4
    assert metrics["predicted_positive_count"] == 2
    assert metrics["is_single_class_label_regime"] == 1.0
