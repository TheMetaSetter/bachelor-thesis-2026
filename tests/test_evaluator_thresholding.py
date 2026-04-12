from __future__ import annotations

import numpy as np
import torch

from src.engine.evaluator import select_point_score_threshold
from src.metrics.pointwise import (
    compute_binary_classification_metrics,
    compute_pointwise_curve_payload,
    compute_pointwise_metrics,
)


def test_select_point_score_threshold_ignores_zero_mass_when_positive_scores_exist() -> None:
    point_scores = np.array([0.0, 0.0, 0.0, 0.2, 0.4, 0.8], dtype=np.float32)

    threshold = select_point_score_threshold(point_scores, quantile=0.5)

    assert threshold > 0.0


def test_compute_pointwise_metrics_uses_strict_threshold_comparison() -> None:
    point_labels = np.array([0, 1, 0], dtype=np.int64)
    point_scores = np.array([0.0, 0.2, 0.0], dtype=np.float32)

    metrics = compute_pointwise_metrics(
        point_labels=point_labels,
        point_scores=point_scores,
        threshold=0.0,
    )

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["fpr"] == 0.0


def test_compute_binary_classification_metrics_reports_expected_values() -> None:
    logits = torch.tensor(
        [
            [4.0, 0.1],
            [0.1, 4.0],
            [3.0, 0.2],
            [0.2, 3.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    metrics = compute_binary_classification_metrics(logits=logits, labels=labels)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["fpr"] == 0.0


def test_pointwise_metric_helpers_handle_single_class_labels_safely() -> None:
    point_labels = np.zeros(4, dtype=np.int64)
    point_scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    metrics = compute_pointwise_metrics(
        point_labels=point_labels,
        point_scores=point_scores,
        threshold=0.5,
    )
    curves = compute_pointwise_curve_payload(
        point_labels=point_labels,
        point_scores=point_scores,
    )

    assert np.isnan(metrics["roc_auc"])
    assert not np.isnan(metrics["fpr"])
    assert "roc_curve" in curves
    assert "pr_curve" in curves
