from __future__ import annotations

import math

import numpy as np

from src.metrics.pointwise import (
    FPR_BUDGETS,
    compute_budgeted_vus_metrics,
)


def test_budgeted_vus_uses_only_the_three_locked_fpr_budgets() -> None:
    assert FPR_BUDGETS == (0.001, 0.005, 0.01)


def test_budgeted_vus_is_perfect_for_separable_scores() -> None:
    labels = np.array([0, 1, 1, 0, 0, 1], dtype=np.int64)
    scores = np.array([0.0, 0.9, 0.8, 0.1, 0.2, 0.95], dtype=np.float64)

    metrics = compute_budgeted_vus_metrics(
        point_labels=labels,
        point_scores=scores,
        max_buffer_size=2,
        num_thresholds=20,
    )

    assert set(metrics) == {"vus_pr", "vus_roc"}
    assert set(metrics["vus_pr"]) == {"0.001", "0.005", "0.01"}
    assert set(metrics["vus_roc"]) == {"0.001", "0.005", "0.01"}
    for budget in FPR_BUDGETS:
        key = str(budget)
        assert math.isclose(metrics["vus_pr"][key], 1.0, abs_tol=1e-6)
        assert math.isclose(metrics["vus_roc"][key], 1.0, abs_tol=1e-6)
