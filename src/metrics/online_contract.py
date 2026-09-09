from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.metrics.pointwise import compute_pointwise_metrics


def compute_final_online_metrics(
    *,
    point_labels: Any,
    point_scores: Any,
    threshold: float,
    vus_max_buffer_size: int,
    vus_num_thresholds: int,
) -> dict[str, Any]:
    labels = np.asarray(point_labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(point_scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or scores.size == 0:
        raise ValueError("online metric inputs must not be empty")
    if labels.size != scores.size:
        raise ValueError(
            f"online metric label/score length mismatch: {labels.size} != {scores.size}"
        )
    metrics = compute_pointwise_metrics(
        point_labels=labels,
        point_scores=scores,
        threshold=float(threshold),
        vus_max_buffer_size=int(vus_max_buffer_size),
        vus_num_thresholds=int(vus_num_thresholds),
    )
    scalar_values = (
        metrics["vus_pr"],
        metrics["affiliation_f1"],
        metrics["vus_roc"],
        metrics["fpr"],
    )
    budget_values = metrics["vus_pr_at_fpr_budget"].values()
    if not all(math.isfinite(float(value)) for value in scalar_values):
        raise ValueError("online aggregate metrics are not finite")
    if not all(math.isfinite(float(value)) for value in budget_values):
        raise ValueError("budgeted online metrics are not finite")
    return {
        "vus_pr_at_fpr_budget": dict(metrics["vus_pr_at_fpr_budget"]),
        "vus_pr": float(metrics["vus_pr"]),
        "affiliation_f1": float(metrics["affiliation_f1"]),
        "vus_roc": float(metrics["vus_roc"]),
        "fpr": float(metrics["fpr"]),
    }
