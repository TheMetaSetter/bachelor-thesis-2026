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
    required = (
        "input_window_threshold",
        "latent_window_low_threshold",
        "latent_window_high_threshold",
    )
    missing = [key for key in required if key not in thresholds]
    if missing:
        raise KeyError(f"full-spec triage thresholds missing keys: {missing}")
    input_threshold = float(thresholds[required[0]])
    latent_low = float(thresholds[required[1]])
    latent_high = float(thresholds[required[2]])
    if latent_low > latent_high:
        raise ValueError("latent low threshold must not exceed high threshold")
    if input_window_score <= input_threshold:
        return "normal"
    if latent_window_score <= latent_low:
        return "hard_old_normality"
    if latent_window_score <= latent_high:
        return "gray_zone"
    return "strong_anomaly"


def classify_legacy_baseline_window(
    input_window_score: float,
    latent_window_score: float,
    thresholds: dict[str, Any],
) -> str:
    """Keep historical baseline regions isolated from THESIS full-spec triage."""
    strong = _resolve_threshold(thresholds, ("strong_anomaly_threshold",), 1.0)
    hard_old = _resolve_threshold(thresholds, ("hard_old_normality_threshold",), 0.0)
    pnn_input = _resolve_threshold(thresholds, ("pnn_candidate_input_threshold",), strong)
    pnn_latent = _resolve_threshold(thresholds, ("pnn_candidate_latent_threshold",), strong)
    if input_window_score >= strong or latent_window_score >= strong:
        return "strong_anomaly"
    if input_window_score <= hard_old and latent_window_score <= hard_old:
        return "hard_old_normality"
    if input_window_score <= pnn_input and latent_window_score >= pnn_latent:
        return "pnn_candidate"
    return "gray_zone"
