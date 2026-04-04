from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(metric_function: Any, *args: Any, **kwargs: Any) -> float:
    try:
        return float(metric_function(*args, **kwargs))
    except ValueError:
        return float("nan")


def compute_pointwise_metrics(
    point_labels: np.ndarray,
    point_scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    # Use a strict comparison so a collapsed zero threshold does not mark every
    # zero-valued point as anomalous.
    binary_predictions = (point_scores > threshold).astype(np.int64)
    return {
        "roc_auc": _safe_metric(roc_auc_score, point_labels, point_scores),
        "pr_auc": _safe_metric(average_precision_score, point_labels, point_scores),
        "precision": _safe_metric(precision_score, point_labels, binary_predictions, zero_division=0),
        "recall": _safe_metric(recall_score, point_labels, binary_predictions, zero_division=0),
        "f1": _safe_metric(f1_score, point_labels, binary_predictions, zero_division=0),
    }
