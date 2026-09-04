from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.protocols.smd_benchmark_protocol import (
    SMD_BENCHMARK_ENTITIES,
    SMD_BENCHMARK_SEEDS,
    validate_protocol_config,
)


def test_smd_protocol_config_contains_locked_entities_and_seeds() -> None:
    assert SMD_BENCHMARK_ENTITIES == ("machine-1-6", "machine-3-4", "machine-3-9")
    assert SMD_BENCHMARK_SEEDS == (6, 8, 36)


def test_protocol_yaml_matches_locked_online_offline_rules() -> None:
    config_path = Path("configs/protocol/smd_window20_cleanval_q99_ewma09.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    validate_protocol_config(config)

    assert config["window_size"] == 20
    assert config["offline_tail_policy"] == "end_align"
    assert config["offline_threshold_split"] == "clean_validation"
    assert config["online_window_stride"] == 1
    assert config["online_ewma_current_weight"] == 0.9
    assert config["test_label_usage"] == "metrics_only"
    assert config["score_space"] == "raw_input"
    assert config["point_score_transform"] == "identity"


def test_protocol_config_rejects_label_leakage() -> None:
    bad_config = {
        "protocol_name": "bad",
        "window_size": 20,
        "offline_tail_policy": "end_align",
        "offline_threshold_split": "test",
        "offline_threshold_quantile": 0.99,
        "online_window_stride": 1,
        "online_threshold_split": "clean_validation",
        "online_threshold_quantile": 0.99,
        "online_ewma_current_weight": 0.9,
        "online_ewma_previous_weight": 0.1,
        "test_label_usage": "threshold_and_metrics",
        "point_adjustment": False,
    }

    with pytest.raises(ValueError, match="test labels"):
        validate_protocol_config(bad_config)


def test_protocol_config_rejects_missing_or_mismatched_raw_score_identity() -> None:
    config = yaml.safe_load(
        Path("configs/protocol/smd_window20_cleanval_q99_ewma09.yaml").read_text(
            encoding="utf-8"
        )
    )
    config.pop("score_space")
    with pytest.raises(ValueError, match="score_space"):
        validate_protocol_config(config)

    config["score_space"] = "normalized_input"
    with pytest.raises(ValueError, match="score_space"):
        validate_protocol_config(config)

    config["score_space"] = "raw_input"
    config["point_score_transform"] = "sigmoid"
    with pytest.raises(ValueError, match="point_score_transform"):
        validate_protocol_config(config)
