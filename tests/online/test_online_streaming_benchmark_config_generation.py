from __future__ import annotations

from pathlib import Path

import yaml

from scripts.generate_online_streaming_benchmark_configs import (
    BENCHMARK_ENTITY_IDS,
    BENCHMARK_METHODS,
    BENCHMARK_METHOD_VARIANTS,
    BENCHMARK_SEEDS,
    BENCHMARK_VARIANTS,
    generate_online_streaming_benchmark_configs,
)


def test_generate_online_streaming_benchmark_configs_writes_all_expected_files() -> (
    None
):
    generated_paths = generate_online_streaming_benchmark_configs()

    expected_count = sum(
        len(BENCHMARK_METHOD_VARIANTS[method])
        * len(BENCHMARK_ENTITY_IDS)
        * len(BENCHMARK_SEEDS)
        * 2
        for method in BENCHMARK_METHODS
    )
    assert len(generated_paths) == expected_count
    sample_path = Path(
        "configs/experiment/online_benchmark/candi/"
        "smd__candi__online_main__machine_1_6__w20__seed6__main.yaml"
    )
    assert sample_path.exists()
    sample_config = yaml.safe_load(sample_path.read_text(encoding="utf-8"))
    assert sample_config["baseline_name"] == "candi"
    assert sample_config["online_variant"] == "main"
    assert sample_config["baseline_kwargs"]["encoder_dim"] == 128
    assert sample_config["baseline_kwargs"]["pretrained_encoder_checkpoint"] == (
        "outputs/benchmark/smd/redlamp_baseline/machine_1_6/seed6/checkpoints/best.pt"
    )
    assert sample_config["window_size"] == 20
    assert sample_config["task_overrides"]["max_online_steps"] is None
    assert sample_config["task_overrides"]["absolute_start_index"] == 146
    assert sample_config["task_overrides"]["absolute_end_index"] == 2200
    assert sample_config["protocol_config_path"].endswith(
        "configs/protocol/smd_window20_cleanval_q99_ewma09.yaml"
    )

    smoke_path = Path(
        "configs/experiment/online_benchmark/candi/"
        "smd__candi__online_main__machine_1_6__w20__seed6__smoke.yaml"
    )
    assert smoke_path.exists()
    smoke_config = yaml.safe_load(smoke_path.read_text(encoding="utf-8"))
    assert smoke_config["task_overrides"]["max_online_steps"] == 16
    assert smoke_config["task_overrides"]["absolute_start_index"] == 146
    assert smoke_config["task_overrides"]["absolute_end_index"] == 2200
