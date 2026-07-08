from __future__ import annotations

from typing import Any

SMD_BENCHMARK_ENTITIES = ("machine-1-6", "machine-3-4", "machine-3-9")
SMD_BENCHMARK_SEEDS = (6, 8, 36)


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


def validate_protocol_config(config: dict[str, Any]) -> None:
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

    _require_equal(config, "window_size", 20)
    _require_equal(config, "offline_tail_policy", "end_align")
    _require_equal(config, "offline_threshold_split", "clean_validation")
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
