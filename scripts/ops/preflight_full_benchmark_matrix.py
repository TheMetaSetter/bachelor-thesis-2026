from __future__ import annotations

"""Validate the complete fair benchmark matrix without running training.

₍^. .^₎⟆ Safe preflight

config files
  -> coverage and naming checks
  -> protocol and epoch checks
  -> runner command plan
  -> JSON report
"""

import argparse
import json
import torch
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.core.config import load_experiment_config
from src.core.config_experiment_validation import validate_experiment_config
from src.engine.online_tta.checkpoint_resolution import resolve_stage_b_checkpoint
from src.protocols.smd_benchmark_protocol import (
    SMD_BENCHMARK_ENTITIES,
    SMD_BENCHMARK_SEEDS,
    validate_protocol_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WINDOW_SIZE = 20
OFFLINE_TRADITIONAL_METHODS = ("stumpy", "kmeans_ad", "iforest")
ONLINE_TRADITIONAL_METHODS = ("stumpy", "kmeans_ad", "iforest")
ONLINE_NEURAL_METHODS = ("candi", "m2n2")
ONLINE_STREAM_RANGES = {
    "machine_1_6": (146, 2200),
    "machine_3_4": (2634, 6116),
    "machine_3_9": (1099, 10807),
}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def _paths(root: Path, pattern: str) -> list[Path]:
    return sorted(root.glob(pattern))


def _require_count(paths: list[Path], expected: int, label: str) -> None:
    if len(paths) != expected:
        raise ValueError(f"{label}: expected {expected} files, found {len(paths)}")


def _validate_thesis_offline(paths: list[Path]) -> None:
    _require_count(paths, 18, "THESIS offline configs")
    for path in paths:
        config = load_experiment_config(path)
        validate_experiment_config(config)
        # The offline THESIS budget is 30 epochs in total.
        # We split it into 25 epochs for Stage A and 5 epochs for Stage B so
        # the preflight check can catch any config that still carries the old
        # 10-epoch assumption.
        if config["epochs"] != 30:
            raise ValueError(f"THESIS offline must use 30 epochs: {path}")
        two_stage = config["two_stage"]
        if (
            two_stage["stage_a_multitask_epochs"],
            two_stage["stage_b_fusion_finetuning_epochs"],
        ) != (25, 5):
            raise ValueError(f"THESIS two-stage budget must be 25+5: {path}")


def _validate_thesis_online(paths: list[Path]) -> None:
    _require_count(paths, 54, "THESIS online configs")
    expected_variants = {"O0_A0", "O0_A1", "O0_A2", "O1_A0", "O1_A1", "O1_A2"}
    observed_variants = {path.name.split("__")[3] for path in paths}
    if observed_variants != expected_variants:
        raise ValueError(f"THESIS online variants are incomplete: {observed_variants}")
    for path in paths:
        config = load_experiment_config(path)
        resolve_stage_b_checkpoint(config)


def _validate_wrapper_configs(
    paths: list[Path], expected_count: int, label: str
) -> None:
    _require_count(paths, expected_count, label)
    for path in paths:
        config = _load_yaml(path)
        required_keys = {"data_config_path", "protocol_config_path", "output_dir"}
        missing_keys = sorted(required_keys - set(config))
        if missing_keys:
            raise ValueError(f"{path} is missing wrapper keys: {missing_keys}")
        if config.get("window_size") != WINDOW_SIZE:
            raise ValueError(f"Window size must be 20: {path}")


def _validate_online_baseline_configs(paths: list[Path], method: str) -> None:
    _validate_wrapper_configs(paths, 9, f"online {method} configs")
    for path in paths:
        config = _load_yaml(path)
        if config.get("online_variant") != "main":
            raise ValueError(f"Online baseline must use variant main: {path}")
        entity_id = str(config["entity_id"])
        expected_range = ONLINE_STREAM_RANGES[entity_id]
        task_overrides = config.get("task_overrides", {})
        observed_range = (
            task_overrides.get("absolute_start_index"),
            task_overrides.get("absolute_end_index"),
        )
        if observed_range != expected_range:
            raise ValueError(f"Online range is incorrect: {path}")
        if method in ONLINE_NEURAL_METHODS:
            baseline_kwargs = config.get("baseline_kwargs", {})
            if baseline_kwargs.get("encoder_dim") != 128:
                raise ValueError(f"RedLamp latent dimension must be 128: {path}")
            if not baseline_kwargs.get("pretrained_encoder_checkpoint"):
                raise ValueError(f"RedLamp checkpoint is missing: {path}")


def _validate_redlamp_configs() -> None:
    paths = _paths(
        REPOSITORY_ROOT / "configs/experiment/benchmark/baseline",
        "*__main.yaml",
    )
    _require_count(paths, 9, "RedLamp offline configs")
    for path in paths:
        config = load_experiment_config(path)
        validate_experiment_config(config)
        if config["epochs"] != 30:
            raise ValueError(f"RedLamp benchmark must use 30 epochs: {path}")


def build_preflight_report() -> dict[str, Any]:
    protocol_path = (
        REPOSITORY_ROOT / "configs/protocol/smd_window20_cleanval_q99_ewma09.yaml"
    )
    protocol = _load_yaml(protocol_path)
    validate_protocol_config(protocol)
    offline_root = REPOSITORY_ROOT / "configs/experiment/offline_benchmark"
    online_root = REPOSITORY_ROOT / "configs/experiment/online_benchmark"
    thesis_offline = _paths(offline_root / "thesis", "*__main.yaml")
    thesis_online = _paths(online_root / "thesis", "*__main.yaml")
    _validate_thesis_offline(thesis_offline)
    _validate_thesis_online(thesis_online)
    _validate_redlamp_configs()
    for method in OFFLINE_TRADITIONAL_METHODS:
        _validate_wrapper_configs(
            _paths(offline_root / method, "*__main.yaml"),
            9,
            f"offline {method} configs",
        )
    for method in ONLINE_TRADITIONAL_METHODS + ONLINE_NEURAL_METHODS:
        _validate_online_baseline_configs(
            _paths(online_root / method, "*online_main*__main.yaml"), method
        )
    return {
        "status": "ready",
        "protocol": protocol_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "entities": list(SMD_BENCHMARK_ENTITIES),
        "seeds": list(SMD_BENCHMARK_SEEDS),
        "offline": {"thesis": len(thesis_offline), "redlamp": 9, "traditional": 27},
        "online": {"thesis": len(thesis_online), "baselines": 45},
        "threshold_safety": {
            "offline_source": protocol["offline_threshold_split"],
            "online_source": protocol["online_threshold_split"],
            "test_label_usage": protocol["test_label_usage"],
            "point_adjustment": protocol["point_adjustment"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required but no CUDA device is available")
    report = build_preflight_report()
    if args.require_cuda:
        report["cuda"] = {
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
        }
    print(
        json.dumps(report, indent=2, sort_keys=True)
        if args.json
        else "benchmark matrix: ready"
    )


if __name__ == "__main__":
    main()
