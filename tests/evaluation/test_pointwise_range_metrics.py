from __future__ import annotations

import math

import numpy as np
import pytest

from src.metrics.pointwise import (
    compute_affiliation_f1,
    compute_vus_pr_exact_naive,
    compute_vus_roc_exact_naive,
)


def test_compute_vus_roc_exact_naive_returns_one_for_perfect_scores() -> None:
    labels = np.array([0, 1, 1, 0, 0, 1], dtype=np.int64)
    scores = np.array([0.0, 0.9, 0.8, 0.1, 0.2, 0.95], dtype=np.float64)

    vus_roc = compute_vus_roc_exact_naive(
        point_labels=labels,
        point_scores=scores,
        max_buffer_size=2,
        num_thresholds=20,
    )

    assert math.isclose(vus_roc, 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_compute_affiliation_f1_returns_one_for_perfect_thresholded_predictions() -> (
    None
):
    labels = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64)
    scores = np.array([0.0, 0.9, 0.8, 0.1, 0.2, 0.95, 0.85, 0.0], dtype=np.float64)

    affiliation_f1 = compute_affiliation_f1(
        point_labels=labels,
        point_scores=scores,
        threshold=0.5,
    )

    assert math.isclose(affiliation_f1, 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_vus_pr_and_vus_roc_reject_shape_mismatches() -> None:
    labels = np.array([0, 1, 0], dtype=np.int64)
    scores = np.array([0.1, 0.2], dtype=np.float64)

    with pytest.raises(ValueError, match="same shape"):
        compute_vus_pr_exact_naive(
            point_labels=labels,
            point_scores=scores,
            max_buffer_size=2,
            num_thresholds=20,
        )

    with pytest.raises(ValueError, match="same shape"):
        compute_vus_roc_exact_naive(
            point_labels=labels,
            point_scores=scores,
            max_buffer_size=2,
            num_thresholds=20,
        )


def test_compute_affiliation_f1_returns_nan_for_single_class_labels() -> None:
    labels = np.zeros(4, dtype=np.int64)
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    affiliation_f1 = compute_affiliation_f1(
        point_labels=labels,
        point_scores=scores,
        threshold=0.5,
    )

    assert math.isnan(affiliation_f1)
