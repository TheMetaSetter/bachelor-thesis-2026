from __future__ import annotations

"""Generate offline benchmark configs for the fair baseline sweep.

₍^. .^₎⟆ Config matrix

method + entity + seed + mode
  -> one compact YAML
  -> one offline benchmark runner input
"""

import argparse
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
OFFLINE_BENCHMARK_CONFIG_ROOT = (
    REPOSITORY_ROOT / "configs" / "experiment" / "offline_benchmark"
)
WINDOW_SIZE = 20
BENCHMARK_ENTITY_IDS = ("machine-1-6", "machine-3-4", "machine-3-9")
BENCHMARK_SEEDS = (6, 8, 36)
BENCHMARK_METHODS = ("stumpy_channel_ab", "kmeans_ad", "iforest")
PROTOCOL_CONFIG_PATH = "configs/protocol/smd_window20_cleanval_q99_ewma09.yaml"
METHOD_DIRECTORY_NAMES = {
    "stumpy_channel_ab": "stumpy",
    "kmeans_ad": "kmeans_ad",
    "iforest": "iforest",
}


def _entity_token(entity_id: str) -> str:
    return entity_id.replace("-", "_")


def _mode_name(smoke: bool) -> str:
    return "smoke" if smoke else "main"


def _benchmark_name(method: str, entity_id: str, seed: int, smoke: bool) -> str:
    return (
        f"smd__{method}__offline__{_entity_token(entity_id)}"
        f"__w{WINDOW_SIZE}__seed{seed}__{_mode_name(smoke)}"
    )


def _output_dir(method: str, entity_id: str, seed: int, smoke: bool) -> str:
    root_dir = "benchmark_smoke" if smoke else "benchmark"
    return (
        f"outputs/{root_dir}/smd/offline_benchmark/{method}"
        f"/{_entity_token(entity_id)}/seed{seed}"
    )


def _data_config_path(entity_id: str) -> str:
    return f"configs/data/smd_benchmark_{_entity_token(entity_id)}_window20.yaml"


def _baseline_kwargs(method: str, seed: int, smoke: bool) -> dict[str, Any]:
    if method == "stumpy_channel_ab":
        return {
            "window_size": WINDOW_SIZE,
            "normalize": True,
            "p": 2.0,
            "threshold_quantile": 0.99,
        }
    if method == "kmeans_ad":
        return {
            "window_size": WINDOW_SIZE,
            "n_clusters": 4 if smoke else 20,
            "normalize_windows": True,
            "threshold_quantile": 0.99,
            "random_state": seed,
        }
    if method == "iforest":
        return {
            "window_size": WINDOW_SIZE,
            "n_estimators": 20 if smoke else 100,
            "normalize_windows": True,
            "threshold_quantile": 0.99,
            "random_state": seed,
        }
    raise ValueError(f"Unknown offline baseline method: {method}")


def build_offline_benchmark_config(
    *,
    method: str,
    entity_id: str,
    seed: int,
    smoke: bool,
) -> dict[str, Any]:
    benchmark_name = _benchmark_name(method, entity_id, seed, smoke)
    return {
        "benchmark_name": benchmark_name,
        "baseline_name": method,
        "entity_id": entity_id,
        "seed": seed,
        "window_size": WINDOW_SIZE,
        "data_config_path": _data_config_path(entity_id),
        "protocol_config_path": PROTOCOL_CONFIG_PATH,
        "output_dir": _output_dir(method, entity_id, seed, smoke),
        "baseline_kwargs": _baseline_kwargs(method, seed, smoke),
    }


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def generate_offline_benchmark_configs() -> list[Path]:
    generated_paths: list[Path] = []
    for method in BENCHMARK_METHODS:
        for entity_id in BENCHMARK_ENTITY_IDS:
            for seed in BENCHMARK_SEEDS:
                for smoke in (False, True):
                    config = build_offline_benchmark_config(
                        method=method,
                        entity_id=entity_id,
                        seed=seed,
                        smoke=smoke,
                    )
                    config_path = (
                        OFFLINE_BENCHMARK_CONFIG_ROOT
                        / METHOD_DIRECTORY_NAMES[method]
                        / (f"{_benchmark_name(method, entity_id, seed, smoke)}.yaml")
                    )
                    _write_config(config_path, config)
                    generated_paths.append(config_path)
    return generated_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-count", action="store_true")
    args = parser.parse_args()
    generated_paths = generate_offline_benchmark_configs()
    if args.print_count:
        print(len(generated_paths))
        return
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
