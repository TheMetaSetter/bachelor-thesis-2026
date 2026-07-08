from __future__ import annotations

from typing import Any

import numpy as np


def select_point_score_threshold(
    point_scores: np.ndarray,
    quantile: float = 0.99,
) -> float:
    # Smoke runs can produce many exact zeros, so selecting from positive
    # support avoids an "everything is anomalous" threshold.
    positive_scores = point_scores[point_scores > 0.0]
    reference_scores = positive_scores if positive_scores.size > 0 else point_scores
    threshold = float(np.quantile(reference_scores, quantile))
    if threshold <= 0.0 and positive_scores.size > 0:
        threshold = float(np.min(positive_scores))
    return threshold


def _select_nan_safe_quantile(point_scores: np.ndarray, quantile: float) -> float:
    finite_scores = np.asarray(point_scores, dtype=float)
    finite_scores = finite_scores[~np.isnan(finite_scores)]
    if finite_scores.size == 0:
        raise ValueError("Cannot select a threshold from only NaN scores")
    return float(np.nanquantile(finite_scores, quantile))


def select_clean_validation_point_threshold(
    clean_validation_point_scores: np.ndarray,
    quantile: float,
) -> float:
    """Select the official offline point threshold from clean validation only."""
    return _select_nan_safe_quantile(clean_validation_point_scores, quantile)


def select_online_ewma_threshold(
    clean_validation_ewma_scores: np.ndarray,
    quantile: float,
) -> float:
    """Select the online TTA threshold after clean-val stride-1 EWMA simulation."""
    return _select_nan_safe_quantile(clean_validation_ewma_scores, quantile)


def resolve_evaluation_threshold(
    point_scores: np.ndarray,
    *,
    point_score_threshold: float | None = None,
    threshold_source: str | None = None,
    quantile: float = 0.99,
) -> tuple[float, str]:
    if point_score_threshold is None:
        threshold = select_point_score_threshold(point_scores, quantile=quantile)
        return threshold, f"positive_support_quantile_{quantile}"
    return float(point_score_threshold), threshold_source or "provided_threshold"


def resolve_checkpoint_threshold_metric_name(
    checkpoint_monitor_metric: str,
) -> str | None:
    if checkpoint_monitor_metric.startswith("val_synth_"):
        return "val_synth_threshold"
    if checkpoint_monitor_metric.startswith("val_"):
        return "val_threshold"
    return None


def build_checkpoint_evaluation_metadata(
    *,
    checkpoint_monitor_metric: str,
    epoch_metrics: dict[str, Any],
    base_extra_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    checkpoint_metadata = dict(base_extra_state or {})
    threshold_metric_name = resolve_checkpoint_threshold_metric_name(
        checkpoint_monitor_metric
    )
    if threshold_metric_name is None or threshold_metric_name not in epoch_metrics:
        return checkpoint_metadata or None
    checkpoint_metadata["evaluation_threshold"] = float(
        epoch_metrics[threshold_metric_name]
    )
    checkpoint_metadata["evaluation_threshold_metric_name"] = threshold_metric_name
    checkpoint_metadata["evaluation_threshold_source"] = (
        f"checkpoint::{threshold_metric_name}"
    )
    return checkpoint_metadata
