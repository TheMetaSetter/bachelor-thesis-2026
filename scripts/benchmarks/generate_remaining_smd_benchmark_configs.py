from __future__ import annotations

"""Generate the approved remaining-SMD benchmark matrix."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.benchmarks._config_generation_helpers import entity_token, write_yaml_config
from src.core.artifact_naming import (
    build_wandb_smoke_run_name,
    wandb_entity_token,
    wandb_method_display_token,
)
from scripts.benchmarks.generate_offline_benchmark_configs import (
    build_offline_benchmark_config as build_traditional_offline_config,
)
from scripts.benchmarks.generate_smd_benchmark_configs import (
    build_offline_benchmark_config as build_thesis_offline_config,
)
from scripts.benchmarks.generate_online_streaming_benchmark_configs import (
    _baseline_kwargs as build_online_baseline_kwargs,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_ENTITY_IDS = ("machine-1-6", "machine-3-4", "machine-3-9")
SEED_VALUES = (6, 8, 36)
THESIS_OFFLINE_VARIANTS = ("O0", "O1")
THESIS_ONLINE_VARIANTS = ("A0", "A1", "A2")
OFFLINE_BASELINES = ("stumpy_channel_ab", "kmeans_ad", "iforest")
ONLINE_BASELINES = ("candi", "m2n2", "stumpy", "kmeans_ad", "iforest")
WINDOW_SIZE = 20
ONLINE_SUBSEQUENCE_LENGTH = 2048
PROTOCOL_CONFIG = REPOSITORY_ROOT / "configs/protocol/smd_window20_cleanval_q99_ewma09.yaml"


def discover_remaining_entities(
    dataset_root: Path,
    excluded_entity_ids: tuple[str, ...] = EXCLUDED_ENTITY_IDS,
    selected_entity_ids: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Return valid SMD entities after applying the explicit exclusion list."""
    split_dirs = {split: dataset_root / split for split in ("train", "test", "test_label")}
    missing_dirs = [str(path) for path in split_dirs.values() if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(f"Missing SMD split directories: {missing_dirs}")
    train_ids = {path.stem for path in split_dirs["train"].glob("*.txt")}
    if not train_ids:
        raise FileNotFoundError(f"No SMD train files found in {split_dirs['train']}")
    for split_name, split_dir in split_dirs.items():
        split_ids = {path.stem for path in split_dir.glob("*.txt")}
        missing_ids = sorted(train_ids - split_ids)
        if missing_ids:
            raise FileNotFoundError(
                f"SMD split {split_name!r} is missing entities: {missing_ids}"
            )
    remaining = sorted(train_ids - set(excluded_entity_ids))
    if not remaining:
        raise ValueError("The exclusion list removed every discovered SMD entity")
    if selected_entity_ids is not None:
        selected = tuple(sorted(set(selected_entity_ids)))
        excluded_selected = sorted(set(selected) & set(excluded_entity_ids))
        if excluded_selected:
            raise ValueError(
                "Selected entity IDs are explicitly excluded: "
                f"{excluded_selected}"
            )
        unknown_selected = sorted(set(selected) - set(remaining))
        if unknown_selected:
            raise ValueError(
                "Selected entity IDs are absent from the valid SMD splits: "
                f"{unknown_selected}"
            )
        if not selected:
            raise ValueError("At least one selected entity ID is required")
        return selected
    return tuple(remaining)


def select_short_online_range(
    label_path: Path,
    *,
    length: int = ONLINE_SUBSEQUENCE_LENGTH,
) -> dict[str, int]:
    """Select one deterministic short test range containing ground-truth anomalies."""
    if length <= 0:
        raise ValueError("length must be positive")
    labels = np.loadtxt(label_path, dtype=np.int64).reshape(-1)
    if labels.size < length:
        raise ValueError(
            f"{label_path} has {labels.size} labels, fewer than the requested "
            f"online range length {length}"
        )
    anomaly_mask = labels != 0
    if not anomaly_mask.any():
        raise ValueError(f"{label_path} contains no ground-truth anomaly events")
    prefix = np.concatenate(([0], np.cumsum(anomaly_mask, dtype=np.int64)))
    anomaly_counts = prefix[length:] - prefix[:-length]
    best_start = int(np.argmax(anomaly_counts))
    return {
        "absolute_start_index": best_start,
        "absolute_end_index": best_start + length,
        "length": length,
        "anomaly_points": int(anomaly_counts[best_start]),
    }


def mode_settings(
    *,
    smoke: bool,
    stage_a_epochs: int | None = None,
    stage_b_epochs: int | None = None,
    max_online_steps: int | None = None,
) -> dict[str, Any]:
    if smoke:
        settings = {
            "mode": "smoke",
            "stage_a_epochs": 3,
            "stage_b_epochs": 2,
            "redlamp_epochs": 5,
            "max_online_steps": 16,
            "vus_max_buffer_size": 10,
            "vus_num_thresholds": 20,
        }
    else:
        settings = {
            "mode": "wet",
            "stage_a_epochs": 25,
            "stage_b_epochs": 5,
            "redlamp_epochs": 30,
            "max_online_steps": None,
            "vus_max_buffer_size": 20,
            "vus_num_thresholds": 200,
        }
    for name, value in (
        ("stage_a_epochs", stage_a_epochs),
        ("stage_b_epochs", stage_b_epochs),
        ("max_online_steps", max_online_steps),
    ):
        if value is not None:
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            settings[name] = value
    return settings


def _run_output_root(output_root: Path, entity_id: str, seed: int, method: str) -> Path:
    return output_root / entity_token(entity_id) / f"seed{seed}" / method


def _resource_class(*, runner: str, method: str) -> str:
    if runner in {"thesis_offline", "redlamp", "thesis_online"}:
        return "gpu"
    if runner == "offline_baseline":
        return "cpu"
    if runner == "online_baseline":
        return "gpu" if method in {"candi", "m2n2"} else "cpu"
    raise ValueError(f"Unsupported runner/method pair: {runner}/{method}")


def _add_run(
    plan: list[dict[str, Any]],
    *,
    output_root: Path,
    phase: str,
    method: str,
    entity_id: str,
    seed: int,
    variant: str | None = None,
    online_range: dict[str, int] | None = None,
    runner: str,
) -> None:
    variant_token = f"-{variant}" if variant else ""
    run_id = f"{phase[:3]}-{method}{variant_token}-{entity_token(entity_id)}-s{seed}"
    output_dir = _run_output_root(output_root, entity_id, seed, method)
    if variant:
        output_dir = output_dir / variant
    output_dir = output_dir / phase
    report_name = {
        "thesis_offline": "thesis_offline_benchmark_report.json",
        "redlamp": "evaluation_metrics.json",
        "offline_baseline": "offline_benchmark_report.json",
        "thesis_online": "thesis_online_A0_benchmark_report.json",
        "online_baseline": "online_streaming_benchmark_report.json",
    }[runner]
    if runner == "thesis_online":
        report_name = f"thesis_online_{str(variant).split('-', 1)[1]}_benchmark_report.json"
    run = {
        "run_id": run_id,
        "phase": phase,
        "method": method,
        "variant": variant,
        "entity_id": entity_id,
        "seed": seed,
        "runner": runner,
        "output_dir": str(output_dir),
        "report_path": str(output_dir / "benchmark" / report_name)
        if runner != "redlamp"
        else str(output_dir / report_name),
        "resource_class": _resource_class(runner=runner, method=method),
        "phase_group": phase,
        "dependencies": {},
    }
    if runner == "thesis_online":
        offline_variant, _ = str(variant).split("-", 1)
        offline_output = (
            _run_output_root(output_root, entity_id, seed, "thesis")
            / offline_variant
            / "offline"
        )
        run["dependencies"] = {
            "stage_b_checkpoint": str(
                offline_output
                / "two_stage/stage_b_fusion_finetuning/checkpoints/best.pt"
            ),
            "threshold_artifact": str(offline_output / "thresholds/thresholds.json"),
        }
    elif runner == "online_baseline" and method in {"candi", "m2n2"}:
        redlamp_output = _run_output_root(
            output_root, entity_id, seed, "redlamp_baseline"
        ) / "offline"
        run["dependencies"] = {
            "redlamp_checkpoint": str(redlamp_output / "checkpoints/best.pt")
        }
    if online_range is not None:
        run["online_range"] = dict(online_range)
    plan.append(run)


def build_matrix_plan(
    *,
    entity_ids: tuple[str, ...],
    seed_values: tuple[int, ...] = SEED_VALUES,
    smoke: bool,
    output_root: Path,
    online_ranges: dict[str, dict[str, int]] | None = None,
    main_method_only: bool = False,
) -> list[dict[str, Any]]:
    """Build the minimal run identity manifest without touching the dataset."""
    plan: list[dict[str, Any]] = []
    for entity_id in entity_ids:
        for seed in seed_values:
            for variant in THESIS_OFFLINE_VARIANTS:
                _add_run(
                    plan,
                    output_root=output_root,
                    phase="offline",
                    method="thesis",
                    variant=variant,
                    entity_id=entity_id,
                    seed=seed,
                    runner="thesis_offline",
                )
            if not main_method_only:
                _add_run(
                    plan,
                    output_root=output_root,
                    phase="offline",
                    method="redlamp_baseline",
                    entity_id=entity_id,
                    seed=seed,
                    runner="redlamp",
                )
                for method in OFFLINE_BASELINES:
                    _add_run(
                        plan,
                        output_root=output_root,
                        phase="offline",
                        method=method,
                        entity_id=entity_id,
                        seed=seed,
                        runner="offline_baseline",
                    )
            for offline_variant in THESIS_OFFLINE_VARIANTS:
                for online_variant in THESIS_ONLINE_VARIANTS:
                    _add_run(
                        plan,
                        output_root=output_root,
                        phase="online",
                        method="thesis",
                        variant=f"{offline_variant}-{online_variant}",
                        entity_id=entity_id,
                        seed=seed,
                        online_range=(online_ranges or {}).get(entity_id),
                        runner="thesis_online",
                    )
            if not main_method_only:
                for method in ONLINE_BASELINES:
                    _add_run(
                        plan,
                        output_root=output_root,
                        phase="online",
                        method=method,
                        entity_id=entity_id,
                        seed=seed,
                        online_range=(online_ranges or {}).get(entity_id),
                        runner="online_baseline",
                    )
    return plan


def _data_config(dataset_root: Path, entity_id: str, smoke: bool) -> dict[str, Any]:
    config: dict[str, Any] = {
        "dataset_name": "smd",
        "root_dir": str(dataset_root.resolve()),
        "window_size": WINDOW_SIZE,
        "stride": 1,
        "train_stride": 1,
        "val_stride": WINDOW_SIZE,
        "test_stride": WINDOW_SIZE,
        "batch_size": 256 if smoke else 512,
        "num_workers": 4,
        "validation_split_ratio": 0.2,
        "entity_ids": [entity_id],
        "shuffle_train": False,
    }
    if smoke:
        config.update(
            max_train_windows=2048,
            max_val_windows=2048,
            max_test_windows=2048,
        )
    return config


def _common_logging(
    *,
    mode: str,
    run_name: str,
    tags: list[str],
    phase_token: str,
    identity_tokens: list[str],
    entity_id: str,
    seed: int,
) -> dict[str, Any]:
    smoke = mode == "smoke"
    return {
        "use_wandb": True,
        "wandb_project": "bachelor-thesis-2026",
        "wandb_mode": "online",
        "wandb_run_name": (
            build_wandb_smoke_run_name(
                phase_token=phase_token,
                identity_tokens=identity_tokens,
                entity_token=wandb_entity_token(entity_id),
                seed=seed,
            )
            if smoke
            else run_name
        ),
        "wandb_tags": tags,
        "log_hard_prediction_ratio": not smoke,
        "log_row_normalized_confusion_matrix": not smoke,
        "diagnostics_stages_for_classification": ["train", "val_synth"],
    }


def _patch_common(config: dict[str, Any], run: dict[str, Any], data_path: Path) -> dict[str, Any]:
    config["data_config_path"] = str(data_path)
    config["output_dir"] = run["output_dir"]
    config["checkpoint_dir"] = str(Path(run["output_dir"]) / "checkpoints")
    return config


def _thesis_offline_config(run: dict[str, Any], data_path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    variant = str(run["variant"])
    config = build_thesis_offline_config(
        variant=variant,
        entity_id=str(run["entity_id"]),
        seed=int(run["seed"]),
        smoke=settings["mode"] == "smoke",
    )
    config = _patch_common(config, run, data_path)
    config["experiment_name"] = run["run_id"]
    config["epochs"] = settings["stage_a_epochs"] + settings["stage_b_epochs"]
    config["two_stage"].update(
        expected_total_training_epochs=config["epochs"],
        stage_a_multitask_epochs=settings["stage_a_epochs"],
        stage_b_fusion_finetuning_epochs=settings["stage_b_epochs"],
    )
    config["evaluation"].update(
        vus_max_buffer_size=settings["vus_max_buffer_size"],
        vus_num_thresholds=settings["vus_num_thresholds"],
        retention_policy="summary_only",
    )
    config["logging"] = _common_logging(
        mode=settings["mode"],
        run_name=run["run_id"],
        tags=["benchmark", "thesis", "offline", variant.lower(), str(run["entity_id"])],
        phase_token="off",
        identity_tokens=[variant],
        entity_id=str(run["entity_id"]),
        seed=int(run["seed"]),
    )
    return config


def _redlamp_config(run: dict[str, Any], data_path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(run["output_dir"])
    entity_id = str(run["entity_id"])
    seed = int(run["seed"])
    config: dict[str, Any] = {
        "experiment_name": run["run_id"],
        "seed": seed,
        "device": "cuda",
        "output_dir": str(output_dir),
        "checkpoint_dir": str(output_dir / "checkpoints"),
        "data_config_path": str(data_path),
        "model_config_path": str(REPOSITORY_ROOT / "configs/model/redlamp_baseline_comparative_smd.yaml"),
        "task_config_path": str(REPOSITORY_ROOT / "configs/task/multitask_tsad_redlamp_multiclass_window20_benchmark_fixed_synth.yaml"),
        "optimizer": {
            "optimizer_name": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 0.5,
            "scheduler": {
                "scheduler_name": "cosine",
                "warmup_epochs": 1 if settings["mode"] == "smoke" else 5,
                "warmup_start_lr": 0.001,
                "cosine_end_lr": 0.0,
                "cosine_after_warmup": True,
            },
        },
        "checkpoint_monitor_metric": "val_synth_vus_pr",
        "epochs": settings["redlamp_epochs"],
        "evaluation": {
            "vus_max_buffer_size": settings["vus_max_buffer_size"],
            "vus_num_thresholds": settings["vus_num_thresholds"],
            "retention_policy": "summary_only",
        },
        "logging": _common_logging(
            mode=settings["mode"],
            run_name=run["run_id"],
            tags=["benchmark", "redlamp", "offline", entity_id, f"seed{seed}"],
            phase_token="off",
            identity_tokens=[wandb_method_display_token("redlamp_baseline")],
            entity_id=entity_id,
            seed=seed,
        ),
    }
    if settings["mode"] == "smoke":
        config["data_overrides"] = {
            "batch_size": 256,
            "num_workers": 4,
            "max_train_windows": 2048,
            "max_val_windows": 2048,
            "max_test_windows": 2048,
        }
    return config


def _traditional_offline_config(run: dict[str, Any], data_path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    config = build_traditional_offline_config(
        method=str(run["method"]),
        entity_id=str(run["entity_id"]),
        seed=int(run["seed"]),
        smoke=settings["mode"] == "smoke",
    )
    config = _patch_common(config, run, data_path)
    config["benchmark_name"] = run["run_id"]
    config["protocol_config_path"] = str(PROTOCOL_CONFIG)
    config["evaluation"] = {
        "vus_max_buffer_size": settings["vus_max_buffer_size"],
        "vus_num_thresholds": settings["vus_num_thresholds"],
        "retention_policy": "summary_only",
    }
    config["data_overrides"] = {"num_workers": 0}
    config["logging"] = _common_logging(
        mode=settings["mode"],
        run_name=run["run_id"],
        tags=[
            "benchmark",
            "traditional",
            "offline",
            str(run["method"]),
            str(run["entity_id"]),
            f"seed{int(run['seed'])}",
        ],
        phase_token="off",
        identity_tokens=[wandb_method_display_token(str(run["method"]))],
        entity_id=str(run["entity_id"]),
        seed=int(run["seed"]),
    )
    config["logging"]["wandb_job_type"] = "offline_benchmark"
    return config


def _thesis_online_config(run: dict[str, Any], data_path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    offline_variant, online_variant = str(run["variant"]).split("-", 1)
    output_dir = Path(run["output_dir"])
    offline_output = output_dir.parents[1] / offline_variant / "offline"
    stage_b_checkpoint = offline_output / "two_stage/stage_b_fusion_finetuning/checkpoints/best.pt"
    threshold_path = offline_output / "thresholds/thresholds.json"
    online_range = run.get("online_range")
    if online_range is None:
        raise ValueError(f"Missing online_range for run {run['run_id']}")
    config: dict[str, Any] = {
        "experiment_name": run["run_id"],
        "seed": int(run["seed"]),
        "offline_variant": offline_variant,
        "online_variant": online_variant,
        "experiment_variant": f"online_tta_{online_variant.lower()}_v1",
        "device": "cuda",
        "output_dir": str(output_dir),
        "checkpoint_dir": str(output_dir / "checkpoints"),
        "data_config_path": str(data_path),
        "model_config_path": str(REPOSITORY_ROOT / "configs/model/online_adaptation.yaml"),
        "task_config_path": str(REPOSITORY_ROOT / "configs/task/online_adaptation.yaml"),
        "data_overrides": {
            "window_size": WINDOW_SIZE,
            "stride": 1,
            "train_stride": 1,
            "val_stride": WINDOW_SIZE,
            "test_stride": WINDOW_SIZE,
            "shuffle_train": False,
            "batch_size": 1,
            "num_workers": 2,
        },
        "model_overrides": {
            "input_dim": 38,
            "encoder_dim": 64,
            "hidden_dim": 32,
            "projector_hidden_dim": 64,
            "projector_dropout": 0.0,
            "enable_prototype_alignment": False,
            "lambda_align": 1.0,
            "lambda_proto": 0.1,
            "lambda_anchor": 0.001,
            "score_source": "projected_hidden",
        },
        "task_overrides": {
            "offline_variant": offline_variant,
            "entity_id": str(run["entity_id"]),
            "seed": int(run["seed"]),
            "benchmark_mode": "smoke" if settings["mode"] == "smoke" else "main",
            "stage_name": "stage_b_fusion_finetuning",
            "reference_checkpoint_path": str(stage_b_checkpoint),
            "threshold_artifact_path": str(threshold_path),
            "warm_start_projector": False,
            "target_param_group": "projector_params",
            "clean_stream_only": True,
            "absolute_start_index": int(online_range["absolute_start_index"]),
            "absolute_end_index": int(online_range["absolute_end_index"]),
            "max_online_steps": settings["max_online_steps"],
            "log_every_n_steps": 1,
            "checkpoint_every_n_steps": 8 if settings["mode"] == "smoke" else 50,
            "reset_policy": "disabled",
            "reset_alignment_threshold": 0.0,
        },
        "optimizer": {"optimizer_name": "adamw", "learning_rate": 0.001, "weight_decay": 0.0},
        "epochs": 1,
        "evaluation": {
            "vus_max_buffer_size": settings["vus_max_buffer_size"],
            "vus_num_thresholds": settings["vus_num_thresholds"],
            "retention_policy": "summary_only",
        },
        "logging": _common_logging(
            mode=settings["mode"],
            run_name=run["run_id"],
            tags=["benchmark", "thesis", "online", offline_variant.lower(), online_variant.lower(), str(run["entity_id"])],
            phase_token="on",
            identity_tokens=[offline_variant, online_variant],
            entity_id=str(run["entity_id"]),
            seed=int(run["seed"]),
        ),
    }
    config["logging"]["wandb_job_type"] = "online_benchmark"
    return config


def _online_baseline_config(run: dict[str, Any], data_path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    method = str(run["method"])
    entity_id = str(run["entity_id"])
    seed = int(run["seed"])
    online_variant = (
        "reference_adapter_redlamp_encoder"
        if method in {"candi", "m2n2"}
        else "main"
    )
    kwargs = build_online_baseline_kwargs(method, entity_id, seed, settings["mode"] == "smoke")
    kwargs["threshold_quantile"] = 0.99
    if method in {"candi", "m2n2"}:
        kwargs["pretrained_encoder_checkpoint"] = str(
            _run_output_root(Path(run["output_dir"]).parents[3], entity_id, seed, "redlamp_baseline")
            / "offline/checkpoints/best.pt"
        )
    online_range = run.get("online_range")
    if online_range is None:
        raise ValueError(f"Missing online_range for run {run['run_id']}")
    return {
        "benchmark_name": run["run_id"],
        "baseline_name": method,
        "online_variant": online_variant,
        "entity_id": entity_id,
        "seed": seed,
        "device": "cuda" if method in {"candi", "m2n2"} else "cpu",
        "data_overrides": {
            "num_workers": 2 if method in {"candi", "m2n2"} else 0
        },
        "window_size": WINDOW_SIZE,
        "data_config_path": str(data_path),
        "protocol_config_path": str(PROTOCOL_CONFIG),
        "output_dir": run["output_dir"],
        "baseline_kwargs": kwargs,
        "task_overrides": {
            "absolute_start_index": int(online_range["absolute_start_index"]),
            "absolute_end_index": int(online_range["absolute_end_index"]),
            "max_online_steps": settings["max_online_steps"],
        },
        "benchmark_mode": settings["mode"],
        "retention_policy": "summary_only",
        "logging": {
            **_common_logging(
                mode=settings["mode"],
                run_name=run["run_id"],
                tags=["benchmark", "online", method, entity_id, f"seed{seed}"],
                phase_token="on",
                identity_tokens=[wandb_method_display_token(method), online_variant],
                entity_id=entity_id,
                seed=seed,
            ),
            "wandb_job_type": "online_benchmark",
        },
    }


def _build_config(run: dict[str, Any], data_path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    builders = {
        "thesis_offline": _thesis_offline_config,
        "redlamp": _redlamp_config,
        "offline_baseline": _traditional_offline_config,
        "thesis_online": _thesis_online_config,
        "online_baseline": _online_baseline_config,
    }
    config = builders[str(run["runner"])](run, data_path, settings)
    logging_config = config.get("logging")
    if not isinstance(logging_config, dict):
        raise ValueError(f"W&B logging must be enabled for {run['run_id']}")
    if (
        logging_config.get("use_wandb") is not True
        or logging_config.get("wandb_mode") != "online"
        or logging_config.get("wandb_project") != "bachelor-thesis-2026"
    ):
        raise ValueError(f"W&B logging must be enabled for {run['run_id']}")
    return config


def write_matrix_configs(
    *,
    dataset_root: Path,
    output_root: Path,
    smoke: bool,
    seed_values: tuple[int, ...] = SEED_VALUES,
    selected_entity_ids: tuple[str, ...] | None = None,
    main_method_only: bool = False,
    stage_a_epochs: int | None = None,
    stage_b_epochs: int | None = None,
    max_online_steps: int | None = None,
) -> Path:
    settings = mode_settings(
        smoke=smoke,
        stage_a_epochs=stage_a_epochs,
        stage_b_epochs=stage_b_epochs,
        max_online_steps=max_online_steps,
    )
    output_root = output_root.resolve()
    entities = discover_remaining_entities(
        dataset_root, selected_entity_ids=selected_entity_ids
    )
    online_ranges = {
        entity_id: select_short_online_range(
            dataset_root / "test_label" / f"{entity_id}.txt"
        )
        for entity_id in entities
    }
    plan = build_matrix_plan(
        entity_ids=entities,
        seed_values=seed_values,
        smoke=smoke,
        output_root=output_root,
        online_ranges=online_ranges,
        main_method_only=main_method_only,
    )
    data_paths: dict[str, Path] = {}
    for entity_id in entities:
        data_path = output_root / "generated_configs/data" / f"{entity_token(entity_id)}.yaml"
        write_yaml_config(data_path, _data_config(dataset_root, entity_id, smoke))
        data_paths[entity_id] = data_path
    for run in plan:
        config_path = output_root / "generated_configs/runs" / f"{run['run_id']}.yaml"
        run["config_path"] = str(config_path)
        write_yaml_config(config_path, _build_config(run, data_paths[str(run["entity_id"])], settings))
    manifest = {
        "mode": settings["mode"],
        "dataset_root": str(dataset_root.resolve()),
        "output_root": str(output_root),
        "entities": list(entities),
        "seeds": list(seed_values),
        "online_subsequence": {
            "selection": "maximum ground-truth anomaly-point count; earliest tie",
            "length": ONLINE_SUBSEQUENCE_LENGTH,
            "ranges": online_ranges,
        },
        "metrics": [
            "VUS-PR@FPR-budget",
            "VUS-PR",
            "Affiliation F1-score",
            "VUS-ROC",
            "raw-FPR",
        ],
        "runs": plan,
    }
    manifest_path = output_root / "remaining_smd_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--entity-id", action="append", dest="entity_ids")
    parser.add_argument("--main-method-only", action="store_true")
    parser.add_argument("--stage-a-epochs", type=int)
    parser.add_argument("--stage-b-epochs", type=int)
    parser.add_argument("--max-online-steps", type=int)
    args = parser.parse_args()
    settings = mode_settings(
        smoke=args.smoke,
        stage_a_epochs=args.stage_a_epochs,
        stage_b_epochs=args.stage_b_epochs,
        max_online_steps=args.max_online_steps,
    )
    selected_entity_ids = (
        tuple(args.entity_ids) if args.entity_ids is not None else None
    )
    entities = discover_remaining_entities(
        args.dataset_root, selected_entity_ids=selected_entity_ids
    )
    plan = build_matrix_plan(
        entity_ids=entities,
        smoke=args.smoke,
        output_root=args.output_root,
        main_method_only=args.main_method_only,
    )
    if args.dry_run:
        print(json.dumps({"mode": settings["mode"], "entities": list(entities), "runs": len(plan), "online_subsequence_length": ONLINE_SUBSEQUENCE_LENGTH, "metrics": ["VUS-PR@FPR-budget", "VUS-PR", "Affiliation F1-score", "VUS-ROC", "raw-FPR"]}, indent=2))
        return
    print(
        write_matrix_configs(
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            smoke=args.smoke,
            selected_entity_ids=selected_entity_ids,
            main_method_only=args.main_method_only,
            stage_a_epochs=args.stage_a_epochs,
            stage_b_epochs=args.stage_b_epochs,
            max_online_steps=args.max_online_steps,
        )
    )


if __name__ == "__main__":
    main()
