from __future__ import annotations

import math

import numpy as np

from src.metrics.pointwise import (
    build_threshold_aware_range_labels,
    compute_vus_pr_exact_naive,
    extract_binary_anomaly_ranges,
)


def test_extract_binary_anomaly_ranges_returns_end_exclusive_ranges() -> None:
    labels = np.array([0, 1, 1, 0, 1, 0, 1, 1], dtype=np.int64)

    ranges = extract_binary_anomaly_ranges(labels)

    assert ranges == [(1, 3), (4, 5), (6, 8)]


def test_build_threshold_aware_range_labels_keeps_original_labels_for_zero_buffer() -> (
    None
):
    labels = np.array([0, 1, 1, 0], dtype=np.int64)
    predictions = np.array([0, 0, 1, 0], dtype=np.int64)

    range_labels = build_threshold_aware_range_labels(
        point_labels=labels,
        binary_predictions=predictions,
        buffer_size=0,
    )

    assert np.array_equal(range_labels, labels.astype(np.float64))


def test_build_threshold_aware_range_labels_extends_only_predicted_buffer_side() -> (
    None
):
    labels = np.array([0, 0, 1, 1, 0, 0], dtype=np.int64)
    predictions = np.array([0, 1, 0, 0, 0, 0], dtype=np.int64)

    range_labels = build_threshold_aware_range_labels(
        point_labels=labels,
        binary_predictions=predictions,
        buffer_size=2,
    )

    assert range_labels[2] == 1.0
    assert range_labels[3] == 1.0
    assert range_labels[1] > 0.0
    assert range_labels[0] == 0.0
    assert range_labels[4] == 0.0
    assert range_labels[5] == 0.0


def test_compute_vus_pr_exact_naive_returns_one_for_perfect_scores() -> None:
    labels = np.array([0, 1, 1, 0, 0, 1], dtype=np.int64)
    scores = np.array([0.0, 0.9, 0.8, 0.1, 0.2, 0.95], dtype=np.float64)

    vus_pr = compute_vus_pr_exact_naive(
        point_labels=labels,
        point_scores=scores,
        max_buffer_size=2,
        num_thresholds=20,
    )

    assert math.isclose(vus_pr, 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_compute_vus_pr_exact_naive_returns_nan_for_single_class_labels() -> None:
    labels = np.array([0, 0, 0, 0], dtype=np.int64)
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    vus_pr = compute_vus_pr_exact_naive(
        point_labels=labels,
        point_scores=scores,
        max_buffer_size=2,
        num_thresholds=20,
    )

    assert math.isnan(vus_pr)
