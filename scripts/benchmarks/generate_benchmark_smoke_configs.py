from __future__ import annotations

"""Generate GPU smoke configs for the THESIS benchmark-smoke matrix."""

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from scripts.benchmarks.generate_smd_benchmark_configs import (
    BENCHMARK_ENTITY_IDS,
    BENCHMARK_SEEDS,
    WINDOW_SIZE,
    build_offline_benchmark_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SMOKE_CONFIG_DIR = (
    REPOSITORY_ROOT / "configs" / "experiment" / "benchmark_smoke" / "thesis"
)


def _entity_token(entity_id: str) -> str:
    return entity_id.replace("-", "_")


def _experiment_name(entity_id: str, seed: int) -> str:
    return (
        f"smd__thesis_multitask__benchmark-two-stage-{_entity_token(entity_id)}"
        f"__w{WINDOW_SIZE}__seed{seed}__smoke"
    )


def _output_dir(entity_id: str, seed: int) -> str:
    return (
        f"outputs/benchmark_smoke/smd/thesis_multitask/{_entity_token(entity_id)}"
        f"/seed{seed}"
    )


def build_benchmark_smoke_config(
    *,
    entity_id: str,
    seed: int,
) -> dict[str, Any]:
    config = deepcopy(
        build_offline_benchmark_config(
            variant="O0",
            entity_id=entity_id,
            seed=seed,
            smoke=True,
        )
    )
    config["experiment_name"] = _experiment_name(entity_id, seed)
    config["device"] = "cuda"
    config["output_dir"] = _output_dir(entity_id, seed)
    config["checkpoint_dir"] = f"{config['output_dir']}/checkpoints"
    config["epochs"] = 3
    config["two_stage"]["expected_total_training_epochs"] = 3
    config["two_stage"]["stage_a_multitask_epochs"] = 2
    config["two_stage"]["stage_b_fusion_finetuning_epochs"] = 1
    logging_config = dict(config["logging"])
    logging_config["wandb_run_name"] = config["experiment_name"]
    logging_config["wandb_tags"] = [
        "benchmark",
        "thesis-two-stage",
        "smd",
        entity_id,
        "smoke",
    ]
    config["logging"] = logging_config
    return config


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def generate_benchmark_smoke_configs() -> list[Path]:
    generated_paths: list[Path] = []
    for entity_id in BENCHMARK_ENTITY_IDS:
        for seed in BENCHMARK_SEEDS:
            config = build_benchmark_smoke_config(
                entity_id=entity_id,
                seed=seed,
            )
            config_path = BENCHMARK_SMOKE_CONFIG_DIR / (
                f"smd__thesis_multitask__benchmark-two-stage-"
                f"{_entity_token(entity_id)}__w{WINDOW_SIZE}__seed{seed}__smoke.yaml"
            )
            _write_config(config_path, config)
            generated_paths.append(config_path)
    return generated_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-count", action="store_true")
    args = parser.parse_args()
    generated_paths = generate_benchmark_smoke_configs()
    if args.print_count:
        print(len(generated_paths))
        return
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
