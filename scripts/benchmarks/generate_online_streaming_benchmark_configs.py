from __future__ import annotations

"""Generate online streaming benchmark configs for the fair baseline sweep."""

import argparse
from pathlib import Path
from typing import Any

from scripts.benchmarks._config_generation_helpers import (
    entity_token,
    write_yaml_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ONLINE_STREAMING_BENCHMARK_CONFIG_ROOT = (
    REPOSITORY_ROOT / "configs" / "experiment" / "online_benchmark"
)
WINDOW_SIZE = 20
BENCHMARK_ENTITY_IDS = ("machine_1_6", "machine_3_4", "machine_3_9")
BENCHMARK_SEEDS = (6, 8, 36)
ONLINE_STREAM_RANGES = {
    "machine_1_6": (146, 2200),
    "machine_3_4": (2634, 6116),
    "machine_3_9": (1099, 10807),
}
BENCHMARK_METHODS = ("candi", "m2n2", "stumpy", "kmeans_ad", "iforest")
BENCHMARK_METHOD_VARIANTS = {
    "candi": ("main",),
    "m2n2": ("main",),
    "stumpy": ("main",),
    "kmeans_ad": ("main",),
    "iforest": ("main",),
}
BENCHMARK_VARIANTS = tuple(
    sorted(
        {
            variant
            for variants in BENCHMARK_METHOD_VARIANTS.values()
            for variant in variants
        }
    )
)
PROTOCOL_CONFIG_PATH = "configs/protocol/smd_window20_cleanval_q99_ewma09.yaml"
METHOD_DIRECTORY_NAMES = {
    "candi": "candi",
    "m2n2": "m2n2",
    "stumpy": "stumpy",
    "kmeans_ad": "kmeans_ad",
    "iforest": "iforest",
}


def _entity_token(entity_id: str) -> str:
    return entity_token(entity_id)


def _mode_name(smoke: bool) -> str:
    return "smoke" if smoke else "main"


def _benchmark_name(
    *,
    method: str,
    online_variant: str,
    entity_id: str,
    seed: int,
    smoke: bool,
) -> str:
    return (
        f"smd__{method}__online_{online_variant}"
        f"__{_entity_token(entity_id)}__w{WINDOW_SIZE}__seed{seed}__{_mode_name(smoke)}"
    )


def _output_dir(
    method: str, online_variant: str, entity_id: str, seed: int, smoke: bool
) -> str:
    root_dir = "benchmark_smoke" if smoke else "benchmark"
    return (
        f"outputs/{root_dir}/online_streaming/smd/{method}/{online_variant}"
        f"/{_entity_token(entity_id)}/seed{seed}"
    )


def _data_config_path(entity_id: str) -> str:
    return f"configs/data/smd_benchmark_{_entity_token(entity_id)}_window20.yaml"


def _online_stream_range(entity_id: str) -> dict[str, int]:
    start_index, end_index = ONLINE_STREAM_RANGES[entity_id]
    return {
        "absolute_start_index": start_index,
        "absolute_end_index": end_index,
    }


def _baseline_kwargs(
    method: str, entity_id: str, seed: int, smoke: bool
) -> dict[str, Any]:
    pretrained_encoder_checkpoint = (
        f"outputs/benchmark/smd/redlamp_baseline/{entity_id}/seed{seed}"
        "/checkpoints/best.pt"
    )
    if method == "candi":
        return {
            "input_dim": 38,
            "window_size": WINDOW_SIZE,
            "threshold_quantile": 0.99,
            "adaptation_momentum": 0.02 if smoke else 0.02,
            "seed": seed,
            "encoder_family": "cnn_simple",
            "encoder_dim": 128,
            "cnn_num_layers": 3,
            "cnn_kernel_size": 3,
            "cnn_hidden_channels": 64,
            "cnn_dropout": 0.1,
            "pretrained_encoder_checkpoint": pretrained_encoder_checkpoint,
        }
    if method == "m2n2":
        return {
            "input_dim": 38,
            "window_size": WINDOW_SIZE,
            "threshold_quantile": 0.99,
            "adaptation_momentum": 0.01 if smoke else 0.01,
            "seed": seed,
            "encoder_family": "cnn_simple",
            "encoder_dim": 128,
            "cnn_num_layers": 3,
            "cnn_kernel_size": 3,
            "cnn_hidden_channels": 64,
            "cnn_dropout": 0.1,
            "pretrained_encoder_checkpoint": pretrained_encoder_checkpoint,
        }
    if method == "stumpy":
        return {
            "window_size": WINDOW_SIZE,
            "normalize": True,
            "p": 2.0,
            "threshold_quantile": 0.99,
            "seed": seed,
        }
    if method == "kmeans_ad":
        return {
            "window_size": WINDOW_SIZE,
            "n_clusters": 4 if smoke else 20,
            "normalize_windows": True,
            "threshold_quantile": 0.99,
            "seed": seed,
        }
    if method == "iforest":
        return {
            "window_size": WINDOW_SIZE,
            "n_estimators": 20 if smoke else 100,
            "normalize_windows": True,
            "threshold_quantile": 0.99,
            "seed": seed,
        }
    raise ValueError(f"Unknown online streaming baseline method: {method}")


def build_online_streaming_benchmark_config(
    *,
    method: str,
    online_variant: str,
    entity_id: str,
    seed: int,
    smoke: bool,
) -> dict[str, Any]:
    benchmark_name = _benchmark_name(
        method=method,
        online_variant=online_variant,
        entity_id=entity_id,
        seed=seed,
        smoke=smoke,
    )
    return {
        "benchmark_name": benchmark_name,
        "baseline_name": method,
        "online_variant": online_variant,
        "entity_id": entity_id,
        "seed": seed,
        "device": "cpu",
        "window_size": WINDOW_SIZE,
        "data_config_path": _data_config_path(entity_id),
        "protocol_config_path": PROTOCOL_CONFIG_PATH,
        "output_dir": _output_dir(method, online_variant, entity_id, seed, smoke),
        "baseline_kwargs": _baseline_kwargs(method, entity_id, seed, smoke),
        "task_overrides": {
            **_online_stream_range(entity_id),
            "max_online_steps": 16 if smoke else None,
        },
        "benchmark_mode": _mode_name(smoke),
        "logging": {
            "use_wandb": not smoke,
            "wandb_project": "bachelor-thesis-2026",
            "wandb_mode": "disabled" if smoke else "online",
            "wandb_job_type": "online_benchmark",
            "wandb_run_name": benchmark_name,
            "wandb_tags": [
                "online-benchmark",
                method,
                online_variant,
                entity_id,
                f"seed{seed}",
                f"window{WINDOW_SIZE}",
                _mode_name(smoke),
            ],
        },
    }


def generate_online_streaming_benchmark_configs() -> list[Path]:
    generated_paths: list[Path] = []
    for method in BENCHMARK_METHODS:
        for online_variant in BENCHMARK_METHOD_VARIANTS[method]:
            for entity_id in BENCHMARK_ENTITY_IDS:
                for seed in BENCHMARK_SEEDS:
                    for smoke in (False, True):
                        config = build_online_streaming_benchmark_config(
                            method=method,
                            online_variant=online_variant,
                            entity_id=entity_id,
                            seed=seed,
                            smoke=smoke,
                        )
                        config_path = (
                            ONLINE_STREAMING_BENCHMARK_CONFIG_ROOT
                            / METHOD_DIRECTORY_NAMES[method]
                            / (
                                f"{_benchmark_name(method=method, online_variant=online_variant, entity_id=entity_id, seed=seed, smoke=smoke)}.yaml"
                            )
                        )
                        write_yaml_config(config_path, config)
                        generated_paths.append(config_path)
    return generated_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-count", action="store_true")
    args = parser.parse_args()
    generated_paths = generate_online_streaming_benchmark_configs()
    if args.print_count:
        print(len(generated_paths))
        return
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
