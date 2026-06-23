from __future__ import annotations

"""Server preflight checks for three-stage offline pre-training runs."""

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPOSITORY_ROOT))

from scripts.run_three_stage_offline_pretraining import (  # noqa: E402
    build_three_stage_training_plan,
    compute_three_stage_total_training_epochs,
    STATISTICAL_PROCEDURE_NAMES,
)
from src.core.config import load_experiment_config  # noqa: E402
from src.core.console import console_print  # noqa: E402
from src.data.loaders import build_smd_dataset_bundle  # noqa: E402


def _resolve_repo_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (REPOSITORY_ROOT / path).resolve()


def _validate_gpu_requirements(
    *,
    device: str,
    gpu_index: int,
    required_gpu_name_substring: str | None,
) -> dict[str, Any]:
    if device != "cuda":
        return {
            "status": "skipped_for_cpu_config",
            "device_count": 0,
            "selected_gpu_index": gpu_index,
            "selected_gpu_name": None,
            "required_gpu_name_substring": required_gpu_name_substring,
        }

    if not torch.cuda.is_available():
        return {
            "status": "cuda_unavailable",
            "device_count": 0,
            "selected_gpu_index": gpu_index,
            "selected_gpu_name": None,
            "required_gpu_name_substring": required_gpu_name_substring,
        }

    device_count = int(torch.cuda.device_count())
    if gpu_index < 0 or gpu_index >= device_count:
        return {
            "status": "gpu_index_out_of_range",
            "device_count": device_count,
            "selected_gpu_index": gpu_index,
            "selected_gpu_name": None,
            "required_gpu_name_substring": required_gpu_name_substring,
        }

    selected_gpu_name = str(torch.cuda.get_device_name(gpu_index))
    if required_gpu_name_substring and required_gpu_name_substring not in selected_gpu_name:
        return {
            "status": "gpu_name_mismatch",
            "device_count": device_count,
            "selected_gpu_index": gpu_index,
            "selected_gpu_name": selected_gpu_name,
            "required_gpu_name_substring": required_gpu_name_substring,
        }

    return {
        "status": "ok",
        "device_count": device_count,
        "selected_gpu_index": gpu_index,
        "selected_gpu_name": selected_gpu_name,
        "required_gpu_name_substring": required_gpu_name_substring,
    }


def _count_uncapped_windows(sequence_length: int, window_size: int, stride: int) -> int:
    if sequence_length < window_size:
        return 0
    return ((sequence_length - window_size) // stride) + 1


def _compute_test_window_anomaly_rate_from_sequences(
    test_sequences: list[dict[str, Any]],
    *,
    window_size: int,
    stride: int,
) -> float:
    total_windows = 0
    anomalous_windows = 0
    for sequence in test_sequences:
        point_labels = sequence["point_labels"]
        if point_labels is None:
            continue
        sequence_length = int(point_labels.shape[0])
        if sequence_length < window_size:
            continue
        for start_index in range(0, sequence_length - window_size + 1, stride):
            end_index = start_index + window_size
            total_windows += 1
            if int(torch.count_nonzero(point_labels[start_index:end_index]).item()) > 0:
                anomalous_windows += 1
    if total_windows == 0:
        raise ValueError("Cannot compute test window anomaly rate from zero windows")
    return anomalous_windows / total_windows


def _build_launch_readiness_summary(
    *,
    total_training_epochs: int,
    device: str,
    tmux_available: bool,
    data_readiness: dict[str, Any],
    gpu_validation: dict[str, Any],
) -> dict[str, Any]:
    is_exact_300_epoch_run = total_training_epochs == 300
    uses_uncapped_test_windows = not bool(
        data_readiness.get("evaluation_uses_capped_test_windows", True)
    )
    device_is_cuda = device == "cuda"
    gpu_ready = str(gpu_validation["status"]) == "ok"
    tmux_ready = tmux_available
    ready_for_server_launch = (
        is_exact_300_epoch_run
        and uses_uncapped_test_windows
        and device_is_cuda
        and gpu_ready
        and tmux_ready
    )
    return {
        "status": (
            "ready_for_rtx3090_tmux_launch"
            if ready_for_server_launch
            else "not_ready_for_server_launch"
        ),
        "is_exact_300_epoch_run": is_exact_300_epoch_run,
        "uses_uncapped_test_windows": uses_uncapped_test_windows,
        "device_is_cuda": device_is_cuda,
        "gpu_ready": gpu_ready,
        "tmux_ready": tmux_ready,
    }


def _write_preflight_summary_artifact(summary: dict[str, Any]) -> str:
    output_dir = Path(str(summary["output_dir"]))
    summary_path = output_dir / "three_stage" / "server_preflight_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_with_path = dict(summary)
    summary_with_path["preflight_summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary_with_path, indent=2), encoding="utf-8")
    return str(summary_path)


def _build_data_readiness_summary(data_config: dict[str, Any]) -> dict[str, Any]:
    dataset_name = str(data_config["dataset_name"])
    if dataset_name != "smd":
        return {
            "dataset_name": dataset_name,
            "status": "unsupported_for_three_stage_server_preflight",
        }

    dataset_bundle = build_smd_dataset_bundle(data_config)
    raw_sequences = dataset_bundle["raw_sequences"]
    datasets = dataset_bundle["datasets"]
    window_size = int(data_config["window_size"])
    stride = int(data_config["stride"])

    raw_sequence_lengths_by_split = {
        split_name: [
            int(sequence["meta"]["sequence_length"])
            for sequence in raw_sequences[split_name]
        ]
        for split_name in ["train", "val", "test"]
    }
    uncapped_window_counts_by_split = {
        split_name: sum(
            _count_uncapped_windows(
                sequence_length=sequence_length,
                window_size=window_size,
                stride=stride,
            )
            for sequence_length in raw_sequence_lengths_by_split[split_name]
        )
        for split_name in ["train", "val", "test"]
    }
    actual_window_counts_by_split = {
        split_name: len(datasets[split_name]) for split_name in ["train", "val", "test"]
    }
    max_window_caps_by_split = {
        split_name: data_config.get(f"max_{split_name}_windows")
        for split_name in ["train", "val", "test"]
    }

    return {
        "dataset_name": dataset_name,
        "status": "ok",
        "selected_entity_ids": list(data_config.get("entity_ids", [])),
        "raw_sequence_lengths_by_split": raw_sequence_lengths_by_split,
        "actual_window_counts_by_split": actual_window_counts_by_split,
        "uncapped_window_counts_by_split": uncapped_window_counts_by_split,
        "max_window_caps_by_split": max_window_caps_by_split,
        "evaluation_uses_capped_test_windows": (
            max_window_caps_by_split["test"] is not None
        ),
        "test_window_anomaly_rate": _compute_test_window_anomaly_rate_from_sequences(
            raw_sequences["test"],
            window_size=window_size,
            stride=stride,
        ),
    }


def build_server_preflight_summary(
    experiment_config_path: str,
    *,
    gpu_index: int = 0,
    required_gpu_name_substring: str | None = "RTX 3090",
) -> dict[str, Any]:
    resolved_experiment_config_path = _resolve_repo_path(experiment_config_path)
    experiment_config = load_experiment_config(resolved_experiment_config_path)
    training_plan = build_three_stage_training_plan(experiment_config)

    data_root = _resolve_repo_path(str(experiment_config["data"]["root_dir"]))
    output_dir = _resolve_repo_path(str(experiment_config["output_dir"]))

    total_training_epochs = compute_three_stage_total_training_epochs(
        experiment_config["three_stage"]
    )
    data_readiness = _build_data_readiness_summary(experiment_config["data"])
    tmux_available = shutil.which("tmux") is not None
    gpu_validation = _validate_gpu_requirements(
        device=str(experiment_config["device"]),
        gpu_index=gpu_index,
        required_gpu_name_substring=required_gpu_name_substring,
    )
    summary = {
        "experiment_name": str(experiment_config["experiment_name"]),
        "experiment_config_path": str(resolved_experiment_config_path),
        "device": str(experiment_config["device"]),
        "entity_ids": list(experiment_config["data"].get("entity_ids", [])),
        "data_root": str(data_root),
        "data_root_exists": data_root.exists(),
        "output_dir": str(output_dir),
        "window_size": int(experiment_config["data"]["window_size"]),
        "stride": int(experiment_config["data"]["stride"]),
        "batch_size": int(experiment_config["data"]["batch_size"]),
        "tmux_required": True,
        "tmux_available": tmux_available,
        "python_executable": sys.executable,
        "total_training_epochs": total_training_epochs,
        "phases": [phase_record["phase_name"] for phase_record in training_plan],
        "optimizer_training_phase_names": [
            phase_record["phase_name"] for phase_record in training_plan
        ],
        "statistical_procedure_names": list(STATISTICAL_PROCEDURE_NAMES),
        "phase_epoch_ranges": training_plan,
        "data_readiness": data_readiness,
        "gpu_validation": gpu_validation,
        "launch_readiness": _build_launch_readiness_summary(
            total_training_epochs=total_training_epochs,
            device=str(experiment_config["device"]),
            tmux_available=tmux_available,
            data_readiness=data_readiness,
            gpu_validation=gpu_validation,
        ),
    }
    summary["preflight_summary_path"] = _write_preflight_summary_artifact(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate server readiness for three-stage offline pre-training"
    )
    parser.add_argument(
        "--experiment-config",
        required=True,
        help="Path to the three-stage experiment config",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="CUDA device index that should be reserved for the run",
    )
    parser.add_argument(
        "--required-gpu-name-substring",
        default="RTX 3090",
        help="Substring that must appear in the selected GPU name for cuda configs",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the resolved preflight summary as JSON",
    )
    parser.add_argument(
        "--require-launch-ready",
        action="store_true",
        help="Exit non-zero unless this config is ready for the real RTX 3090 tmux launch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_server_preflight_summary(
        args.experiment_config,
        gpu_index=args.gpu_index,
        required_gpu_name_substring=args.required_gpu_name_substring,
    )
    console_print(
        "THREE_STAGE",
        "Built server preflight summary",
        experiment_name=summary["experiment_name"],
        device=summary["device"],
        total_training_epochs=summary["total_training_epochs"],
        phases=summary["phases"],
        data_root_exists=summary["data_root_exists"],
        gpu_validation_status=summary["gpu_validation"]["status"],
        launch_readiness_status=summary["launch_readiness"]["status"],
        preflight_summary_path=summary["preflight_summary_path"],
    )
    if args.print_json:
        print(json.dumps(summary, indent=2))

    gpu_status = str(summary["gpu_validation"]["status"])
    if not summary["data_root_exists"]:
        raise SystemExit("Data root does not exist for this experiment config")
    if summary["device"] == "cuda" and gpu_status != "ok":
        raise SystemExit(
            "GPU validation failed for cuda experiment config: "
            f"{json.dumps(summary['gpu_validation'])}"
        )
    if args.require_launch_ready and (
        summary["launch_readiness"]["status"] != "ready_for_rtx3090_tmux_launch"
    ):
        raise SystemExit(
            "Launch readiness check failed: "
            f"{json.dumps(summary['launch_readiness'])}"
        )


if __name__ == "__main__":
    main()
