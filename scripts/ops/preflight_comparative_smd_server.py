from __future__ import annotations

"""Server preflight checks for comparative SMD tmux shards."""

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPOSITORY_ROOT))

from scripts.experiments.run_comparative_smd_experiments import (
    build_comparative_run_plan,
)  # noqa: E402


def _validate_gpu_requirements(
    *,
    gpu_index: int,
    required_gpu_name_substring: str | None,
) -> dict[str, Any]:
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
    if (
        required_gpu_name_substring
        and required_gpu_name_substring not in selected_gpu_name
    ):
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


def _validate_devices_are_cuda(
    run_records: list[dict[str, Any]],
) -> dict[str, Any]:
    devices = [str(run_record.get("device", "missing")) for run_record in run_records]
    if all(device == "cuda" for device in devices):
        return {"status": "all_cuda", "devices": devices}
    return {"status": "mixed_or_non_cuda_configs", "devices": devices}


def _summarize_smoke_devices(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    devices = [str(run_record.get("device", "missing")) for run_record in run_records]
    if not devices:
        return {"status": "no_smoke_runs", "devices": devices}
    if all(device == "cpu" for device in devices):
        return {"status": "all_cpu", "devices": devices}
    if all(device == "cuda" for device in devices):
        return {"status": "all_cuda", "devices": devices}
    return {"status": "mixed_smoke_devices", "devices": devices}


def _write_preflight_summary_artifact(summary: dict[str, Any]) -> str:
    report_dir = Path(str(summary["report_dir"]))
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "comparative_server_preflight_summary.json"
    summary_with_path = dict(summary)
    summary_with_path["preflight_summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary_with_path, indent=2), encoding="utf-8")
    return str(summary_path)


def _build_launch_readiness_summary(
    *,
    main_device_validation: dict[str, Any],
    smoke_device_validation: dict[str, Any],
    tmux_available: bool,
    gpu_validation: dict[str, Any],
    data_roots_exist: bool,
    artifact_paths_unique: bool,
) -> dict[str, Any]:
    all_devices_are_cuda = str(main_device_validation["status"]) == "all_cuda"
    smoke_profile_compatible = str(smoke_device_validation["status"]) in {
        "no_smoke_runs",
        "all_cpu",
        "all_cuda",
    }
    gpu_ready = str(gpu_validation["status"]) == "ok"
    tmux_ready = tmux_available
    ready_for_launch = (
        all_devices_are_cuda
        and smoke_profile_compatible
        and gpu_ready
        and tmux_ready
        and data_roots_exist
        and artifact_paths_unique
    )
    return {
        "status": (
            "ready_for_comparative_tmux_launch"
            if ready_for_launch
            else "not_ready_for_comparative_launch"
        ),
        "all_devices_are_cuda": all_devices_are_cuda,
        "smoke_profile_compatible": smoke_profile_compatible,
        "gpu_ready": gpu_ready,
        "tmux_ready": tmux_ready,
        "data_roots_exist": data_roots_exist,
        "artifact_paths_unique": artifact_paths_unique,
    }


def build_comparative_preflight_summary(
    *,
    config_paths: list[str | Path],
    smoke_config_paths: list[str | Path] | None = None,
    report_dir: str | Path,
    gpu_index: int,
    required_gpu_name_substring: str | None,
    data_num_workers_override: int | None = None,
) -> dict[str, Any]:
    smoke_config_paths = smoke_config_paths or []
    run_plan = build_comparative_run_plan(
        config_paths=config_paths,
        smoke_config_paths=smoke_config_paths,
        report_dir=report_dir,
        data_num_workers_override=data_num_workers_override,
    )
    smoke_run_records = list(run_plan["smoke_runs"])
    main_run_records = list(run_plan["main_runs"])
    all_run_records = smoke_run_records + main_run_records
    main_device_validation = _validate_devices_are_cuda(main_run_records)
    smoke_device_validation = _summarize_smoke_devices(smoke_run_records)
    gpu_validation = _validate_gpu_requirements(
        gpu_index=gpu_index,
        required_gpu_name_substring=required_gpu_name_substring,
    )
    tmux_available = shutil.which("tmux") is not None
    summary = {
        "report_dir": str(Path(run_plan["report_dir"]).resolve()),
        "manifest_path": str(Path(run_plan["manifest_path"]).resolve()),
        "execution_report_path": str(Path(run_plan["execution_report_path"]).resolve()),
        "requested_gpu_index": int(gpu_index),
        "required_gpu_name_substring": required_gpu_name_substring,
        "data_num_workers_override": data_num_workers_override,
        "config_paths": [str(Path(path).resolve()) for path in config_paths],
        "smoke_config_paths": [
            str(Path(path).resolve()) for path in smoke_config_paths
        ],
        "experiment_names": [str(run["experiment_name"]) for run in all_run_records],
        "dataset_roots": [str(run["dataset_root"]) for run in all_run_records],
        "output_dirs": [str(run["output_dir"]) for run in all_run_records],
        "checkpoint_dirs": [str(run["checkpoint_dir"]) for run in all_run_records],
        "tmux_required": True,
        "tmux_available": tmux_available,
        "device_validation": main_device_validation,
        "main_device_validation": main_device_validation,
        "smoke_device_validation": smoke_device_validation,
        "gpu_validation": gpu_validation,
    }
    summary["launch_readiness"] = _build_launch_readiness_summary(
        main_device_validation=main_device_validation,
        smoke_device_validation=smoke_device_validation,
        tmux_available=tmux_available,
        gpu_validation=gpu_validation,
        data_roots_exist=True,
        artifact_paths_unique=True,
    )
    summary["preflight_summary_path"] = _write_preflight_summary_artifact(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate server readiness for comparative SMD tmux shards"
    )
    parser.add_argument("--config-paths", nargs="+", required=True)
    parser.add_argument("--smoke-config-paths", nargs="*", default=[])
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--required-gpu-name-substring", default="")
    parser.add_argument("--data-num-workers-override", type=int, default=None)
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--require-launch-ready", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_comparative_preflight_summary(
        config_paths=list(args.config_paths),
        smoke_config_paths=list(args.smoke_config_paths),
        report_dir=args.report_dir,
        gpu_index=args.gpu_index,
        required_gpu_name_substring=args.required_gpu_name_substring,
        data_num_workers_override=args.data_num_workers_override,
    )
    if args.print_json:
        print(json.dumps(summary, indent=2))
    if args.require_launch_ready and (
        summary["launch_readiness"]["status"] != "ready_for_comparative_tmux_launch"
    ):
        raise SystemExit(
            f"Launch readiness check failed: {json.dumps(summary['launch_readiness'])}"
        )


if __name__ == "__main__":
    main()
