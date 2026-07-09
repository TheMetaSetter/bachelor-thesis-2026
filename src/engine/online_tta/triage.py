from __future__ import annotations

from typing import Any


def _resolve_threshold(
    thresholds: dict[str, Any],
    preferred_keys: tuple[str, ...],
    fallback: float,
) -> float:
    for key in preferred_keys:
        if key in thresholds:
            return float(thresholds[key])
    return float(fallback)


def classify_online_window(
    input_window_score: float,
    latent_window_score: float,
    thresholds: dict[str, Any],
) -> str:
    strong_anomaly_threshold = _resolve_threshold(
        thresholds,
        ("strong_anomaly_threshold", "input_window_high_threshold"),
        fallback=1.0,
    )
    hard_old_normality_threshold = _resolve_threshold(
        thresholds,
        ("hard_old_normality_threshold", "input_window_low_threshold"),
        fallback=0.0,
    )
    pnn_candidate_input_threshold = _resolve_threshold(
        thresholds,
        ("pnn_candidate_input_threshold", "input_window_candidate_threshold"),
        fallback=strong_anomaly_threshold,
    )
    pnn_candidate_latent_threshold = _resolve_threshold(
        thresholds,
        ("pnn_candidate_latent_threshold", "latent_window_candidate_threshold"),
        fallback=strong_anomaly_threshold,
    )
    if (
        input_window_score >= strong_anomaly_threshold
        or latent_window_score >= strong_anomaly_threshold
    ):
        return "strong_anomaly"
    if (
        input_window_score <= hard_old_normality_threshold
        and latent_window_score <= hard_old_normality_threshold
    ):
        return "hard_old_normality"
    if (
        input_window_score <= pnn_candidate_input_threshold
        and latent_window_score >= pnn_candidate_latent_threshold
    ):
        return "pnn_candidate"
    return "gray_zone"
