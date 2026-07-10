from __future__ import annotations

from pathlib import Path

import yaml

from scripts.generate_offline_benchmark_configs import (
    BENCHMARK_ENTITY_IDS,
    BENCHMARK_METHODS,
    BENCHMARK_SEEDS,
    generate_offline_benchmark_configs,
)


def test_generate_offline_benchmark_configs_writes_all_expected_files() -> None:
    generated_paths = generate_offline_benchmark_configs()

    assert (
        len(generated_paths)
        == len(BENCHMARK_METHODS) * len(BENCHMARK_ENTITY_IDS) * len(BENCHMARK_SEEDS) * 2
    )
    sample_path = Path(
        "configs/experiment/offline_benchmark/stumpy/"
        "smd__stumpy_channel_ab__offline__machine_1_6__w20__seed6__main.yaml"
    )
    assert sample_path.exists()
    sample_config = yaml.safe_load(sample_path.read_text(encoding="utf-8"))
    assert sample_config["baseline_name"] == "stumpy_channel_ab"
    assert sample_config["window_size"] == 20
    assert sample_config["protocol_config_path"].endswith(
        "configs/protocol/smd_window20_cleanval_q99_ewma09.yaml"
    )
