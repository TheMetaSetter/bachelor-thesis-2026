from __future__ import annotations

import math

import numpy as np

from src.metrics.affiliation import (
    _affiliation_partition,
    compute_affiliation_precision_recall,
)


def test_compute_affiliation_precision_recall_returns_one_for_perfect_events() -> None:
    labels = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64)
    predictions = labels.copy()

    precision, recall = compute_affiliation_precision_recall(labels, predictions)

    assert math.isclose(precision, 1.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(recall, 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_affiliation_partition_preserves_one_slot_per_input_event_in_each_zone() -> (
    None
):
    predicted_events = [(0.0, 1.0), (3.0, 4.0)]
    affiliation_zones = [(0.0, 2.0), (2.0, 5.0)]

    partition = _affiliation_partition(predicted_events, affiliation_zones)

    assert partition == [[(0.0, 1.0), None], [None, (3.0, 4.0)]]
