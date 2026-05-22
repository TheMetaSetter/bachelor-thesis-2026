from __future__ import annotations

import torch

from src.metrics.classification_diagnostics import (
    compute_hard_prediction_ratio,
    compute_row_normalized_confusion_matrix,
)


def test_hard_prediction_ratio_sums_to_one() -> None:
    logits = torch.tensor(
        [[3.0, 1.0, 0.0], [0.0, 4.0, 1.0], [0.2, 0.3, 0.9], [2.0, 0.1, 0.1]]
    )
    ratios = compute_hard_prediction_ratio(logits, ("0", "1", "2"))
    assert abs(sum(ratios.values()) - 1.0) < 1.0e-6


def test_row_normalized_confusion_rows_sum_to_one_when_support_exists() -> None:
    logits = torch.tensor([[3.0, 1.0], [0.2, 2.0], [1.0, 0.1], [0.1, 4.0]])
    labels = torch.tensor([0, 1, 0, 1])
    payload = compute_row_normalized_confusion_matrix(logits, labels, ("0", "1"))

    for row, support in zip(payload["row_normalized"], payload["support"]):
        if support > 0:
            assert abs(sum(row) - 1.0) < 1.0e-6
