from __future__ import annotations

import json
from pathlib import Path
import subprocess

from scripts.preflight_three_stage_server import build_server_preflight_summary


def test_server_preflight_summary_reads_smoke_config_and_reports_cpu_mode() -> None:
    summary = build_server_preflight_summary(
        "configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml",
        required_gpu_name_substring="RTX 3090",
    )

    assert summary["experiment_name"].endswith("smoke_seed11")
    assert summary["device"] == "cpu"
    assert summary["data_root_exists"] is True
    assert summary["tmux_required"] is True
    assert summary["total_training_epochs"] == 5
    assert summary["phases"] == [
        "stage1_classification",
        "stage1_reconstruction",
        "stage2_recovery",
        "stage3_memory_initialization_and_fusion_warmup",
        "multitask_pretraining",
    ]
    assert summary["optimizer_training_phase_names"] == summary["phases"]
    assert summary["statistical_procedure_names"] == [
        "stage2_mtz_parameter_zipping",
        "stage3_memory_initialization",
    ]
    assert summary["gpu_validation"]["status"] == "skipped_for_cpu_config"
    assert summary["launch_readiness"]["status"] == "not_ready_for_server_launch"
    assert summary["launch_readiness"]["is_exact_300_epoch_run"] is False
    assert summary["launch_readiness"]["uses_uncapped_test_windows"] is False
    assert summary["launch_readiness"]["gpu_ready"] is False
    assert summary["preflight_summary_path"].endswith(
        "three_stage/server_preflight_summary.json"
    )
    assert summary["data_readiness"]["selected_entity_ids"] == ["machine-3-4"]
    assert summary["data_readiness"]["raw_sequence_lengths_by_split"]["train"] == [18950]
    assert summary["data_readiness"]["raw_sequence_lengths_by_split"]["val"] == [4737]
    assert summary["data_readiness"]["raw_sequence_lengths_by_split"]["test"] == [23687]
    assert summary["data_readiness"]["actual_window_counts_by_split"] == {
        "train": 64,
        "val": 32,
        "test": 64,
    }
    assert summary["data_readiness"]["uncapped_window_counts_by_split"] == {
        "train": 18931,
        "val": 4718,
        "test": 23668,
    }
    assert summary["data_readiness"]["max_window_caps_by_split"] == {
        "train": 64,
        "val": 32,
        "test": 64,
    }
    assert summary["data_readiness"]["evaluation_uses_capped_test_windows"] is True
    assert summary["data_readiness"]["test_window_anomaly_rate"] == 1129 / 23668
    saved_summary = json.loads(
        Path(summary["preflight_summary_path"]).read_text(encoding="utf-8")
    )
    assert saved_summary["experiment_name"] == summary["experiment_name"]
    assert saved_summary["launch_readiness"] == summary["launch_readiness"]


def test_server_preflight_summary_reads_main_300_epoch_config_without_test_window_cap() -> None:
    summary = build_server_preflight_summary(
        "configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml",
        required_gpu_name_substring="RTX 3090",
    )

    assert summary["experiment_name"].endswith("seed11")
    assert summary["device"] == "cuda"
    assert summary["total_training_epochs"] == 300
    assert summary["phases"] == [
        "stage1_classification",
        "stage1_reconstruction",
        "stage2_recovery",
        "stage3_memory_initialization_and_fusion_warmup",
        "multitask_pretraining",
    ]
    assert summary["optimizer_training_phase_names"] == summary["phases"]
    assert summary["statistical_procedure_names"] == [
        "stage2_mtz_parameter_zipping",
        "stage3_memory_initialization",
    ]
    assert summary["gpu_validation"]["status"] in {
        "ok",
        "cuda_unavailable",
        "gpu_name_mismatch",
        "gpu_index_out_of_range",
    }
    assert summary["launch_readiness"]["is_exact_300_epoch_run"] is True
    assert summary["launch_readiness"]["uses_uncapped_test_windows"] is True
    assert summary["launch_readiness"]["device_is_cuda"] is True
    assert summary["launch_readiness"]["tmux_ready"] == summary["tmux_available"]
    assert summary["launch_readiness"]["status"] in {
        "ready_for_rtx3090_tmux_launch",
        "not_ready_for_server_launch",
    }
    assert summary["data_readiness"]["actual_window_counts_by_split"] == {
        "train": 18931,
        "val": 4718,
        "test": 23668,
    }
    assert summary["data_readiness"]["uncapped_window_counts_by_split"] == {
        "train": 18931,
        "val": 4718,
        "test": 23668,
    }
    assert summary["data_readiness"]["max_window_caps_by_split"] == {
        "train": None,
        "val": None,
        "test": None,
    }
    assert summary["data_readiness"]["evaluation_uses_capped_test_windows"] is False


def test_server_preflight_cli_can_reject_non_launch_ready_smoke_config() -> None:
    completed = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/preflight_three_stage_server.py",
            "--experiment-config",
            "configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml",
            "--require-launch-ready",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "Launch readiness check failed" in completed.stderr


def test_launch_readiness_requires_tmux_when_other_conditions_are_met(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.preflight_three_stage_server.shutil.which",
        lambda executable: None if executable == "tmux" else "/usr/bin/fake",
    )
    monkeypatch.setattr(
        "scripts.preflight_three_stage_server.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "scripts.preflight_three_stage_server.torch.cuda.device_count",
        lambda: 1,
    )
    monkeypatch.setattr(
        "scripts.preflight_three_stage_server.torch.cuda.get_device_name",
        lambda index: "NVIDIA GeForce RTX 3090",
    )

    summary = build_server_preflight_summary(
        "configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml",
        required_gpu_name_substring="RTX 3090",
    )

    assert summary["tmux_available"] is False
    assert summary["gpu_validation"]["status"] == "ok"
    assert summary["launch_readiness"]["tmux_ready"] is False
    assert summary["launch_readiness"]["gpu_ready"] is True
    assert summary["launch_readiness"]["status"] == "not_ready_for_server_launch"
