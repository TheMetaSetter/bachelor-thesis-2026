from __future__ import annotations

import math

import numpy as np

from scripts.ops.aggregate_online_table3_metrics import _compute_metric_values


def test_online_metric_values_include_pointwise_fpr() -> None:
    labels = np.asarray([0, 0, 1, 0], dtype=np.int64)
    scores = np.asarray([0.1, 0.9, 0.8, 0.2], dtype=np.float64)

    metrics = _compute_metric_values(labels, scores, threshold=0.5)

    assert math.isclose(metrics["fpr"], 1.0 / 3.0)
