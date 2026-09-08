from __future__ import annotations

from typing import Any

SMD_BENCHMARK_ENTITIES = ("machine-1-6", "machine-3-4", "machine-3-9")
SMD_BENCHMARK_SEEDS = (6, 8, 36)
OFFLINE_POINT_THRESHOLD_SOURCES = {
    "clean_validation",
    "synthetic_validation_normal",
}
OFFLINE_WINDOW_THRESHOLD_SOURCES = OFFLINE_POINT_THRESHOLD_SOURCES


def _require_key(config: dict[str, Any], key: str) -> None:
    if key not in config:
        raise ValueError(f"Protocol config is missing required key: {key}")


def _require_equal(config: dict[str, Any], key: str, expected_value: Any) -> None:
    _require_key(config, key)
    if config[key] != expected_value:
        raise ValueError(f"{key} must be {expected_value!r}, got {config[key]!r}")


def _require_quantile(config: dict[str, Any], key: str) -> None:
    _require_key(config, key)
    value = float(config[key])
    if not 0.0 < value < 1.0:
        raise ValueError(f"{key} must be between 0 and 1")


def validate_protocol_config(
    config: dict[str, Any], *, require_score_identity: bool = True
) -> None:
    """Validate locked SMD benchmark rules before any run starts.

    ( ˶˘ ³˘)♡ Fairness guard

    config
      -> clean validation decides thresholds
      -> test labels are metrics-only
      -> online calibration uses stride=1 EWMA
    """
    if config.get("offline_threshold_split") == "test":
        raise ValueError("test labels cannot be used for threshold selection")
    if config.get("test_label_usage") != "metrics_only":
        raise ValueError("test labels must be used for metrics_only")

    score_space = config.get("score_space")
    if require_score_identity and score_space is None:
        _require_key(config, "score_space")
    if score_space not in {None, "raw_input", "normalized_input"}:
        raise ValueError("score_space must be raw_input or normalized_input")
    if require_score_identity or "point_score_transform" in config:
        _require_equal(config, "point_score_transform", "identity")

    _require_equal(config, "window_size", 20)
    _require_equal(config, "offline_tail_policy", "end_align")
    offline_threshold_split = config.get("offline_threshold_split")
    if offline_threshold_split not in {"clean_validation", "synthetic_validation"}:
        raise ValueError(
            "offline_threshold_split must be clean_validation or synthetic_validation"
        )
    offline_point_source = config.get(
        "offline_point_threshold_source_split", "clean_validation"
    )
    if offline_point_source not in OFFLINE_POINT_THRESHOLD_SOURCES:
        raise ValueError(
            "offline_point_threshold_source_split must be one of "
            f"{sorted(OFFLINE_POINT_THRESHOLD_SOURCES)!r}, "
            f"got {offline_point_source!r}"
        )
    offline_window_source = config.get(
        "offline_window_threshold_source_split", "clean_validation"
    )
    if offline_window_source not in OFFLINE_WINDOW_THRESHOLD_SOURCES:
        raise ValueError(
            "offline_window_threshold_source_split must be one of "
            f"{sorted(OFFLINE_WINDOW_THRESHOLD_SOURCES)!r}, "
            f"got {offline_window_source!r}"
        )
    if score_space == "normalized_input":
        if offline_threshold_split != "synthetic_validation":
            raise ValueError(
                "normalized_input requires synthetic_validation thresholds"
            )
        if offline_point_source != "synthetic_validation_normal":
            raise ValueError("normalized_input requires synthetic-normal point scores")
        if offline_window_source != "synthetic_validation_normal":
            raise ValueError("normalized_input requires synthetic-normal window scores")
    _require_equal(config, "online_window_stride", 1)
    _require_equal(config, "online_threshold_split", "clean_validation")
    _require_equal(config, "test_label_usage", "metrics_only")
    _require_equal(config, "point_adjustment", False)
    _require_quantile(config, "offline_threshold_quantile")
    _require_quantile(config, "online_threshold_quantile")

    current_weight = float(config.get("online_ewma_current_weight", -1.0))
    previous_weight = float(config.get("online_ewma_previous_weight", -1.0))
    if not current_weight > previous_weight > 0.0:
        raise ValueError("online EWMA weights must be positive and current-heavy")
    if abs((current_weight + previous_weight) - 1.0) > 1e-8:
        raise ValueError("online EWMA weights must sum to 1.0")
