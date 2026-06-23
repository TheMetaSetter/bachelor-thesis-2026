from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from src.core.config import load_experiment_config
from scripts.run_three_stage_offline_pretraining import (
    build_three_stage_training_plan,
    build_three_stage_execution_commands,
    compute_three_stage_total_training_epochs,
    execute_three_stage_plan,
    materialize_three_stage_run_manifest,
    validate_three_stage_epoch_budget,
)


def test_three_stage_epoch_budget_helpers_resolve_exact_300_training_epochs() -> None:
    experiment_config = load_experiment_config(
        "configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml"
    )

    validate_three_stage_epoch_budget(experiment_config)
    training_plan = build_three_stage_training_plan(experiment_config)

    assert [phase["phase_name"] for phase in training_plan] == [
        "stage1_classification",
        "stage1_reconstruction",
        "stage2_recovery",
        "stage3_memory_initialization_and_fusion_warmup",
        "multitask_pretraining",
    ]
    assert [phase["epochs"] for phase in training_plan] == [50, 70, 20, 20, 140]
    assert compute_three_stage_total_training_epochs(experiment_config["three_stage"]) == 300


def test_three_stage_epoch_budget_helpers_reject_budget_drift() -> None:
    experiment_config = load_experiment_config(
        "configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml"
    )
    experiment_config["three_stage"]["multitask_pretraining_epochs"] = 141

    with pytest.raises(ValueError, match="exactly 300"):
        validate_three_stage_epoch_budget(experiment_config)


def test_materialize_three_stage_run_manifest_writes_stage_configs_with_contiguous_epoch_ranges(
    tmp_path,
) -> None:
    experiment_config = load_experiment_config(
        "configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml"
    )
    experiment_config["output_dir"] = str(tmp_path / "outputs")
    experiment_config["checkpoint_dir"] = str(tmp_path / "outputs" / "checkpoints")

    manifest = materialize_three_stage_run_manifest(experiment_config)

    assert manifest["experiment_name"] == experiment_config["experiment_name"]
    assert manifest["total_training_epochs"] == 5
    assert len(manifest["training_stages"]) == 5
    assert manifest["training_stages"][0]["phase_name"] == "stage1_classification"
    assert manifest["training_stages"][0]["global_epoch_start"] == 1
    assert manifest["training_stages"][0]["global_epoch_end"] == 1
    assert manifest["training_stages"][-1]["phase_name"] == "multitask_pretraining"
    assert manifest["training_stages"][-1]["global_epoch_end"] == 5
    stage3_record = manifest["training_stages"][3]
    assert stage3_record["phase_name"] == "stage3_memory_initialization_and_fusion_warmup"
    assert (
        stage3_record["semantic_stage_label"]
        == "Stage 3: Memory Initialization and Fusion Warm-Up"
    )
    assert stage3_record["memory_initialization_substep"] is True
    assert stage3_record["fusion_warmup_substep"] is True

    manifest_path = tmp_path / "outputs" / "three_stage" / "three_stage_manifest.json"
    assert manifest_path.exists()
    for stage_record in manifest["training_stages"]:
        assert Path(stage_record["config_path"]).exists()
        loaded_stage_config = load_experiment_config(stage_record["config_path"])
        assert loaded_stage_config["epochs"] == stage_record["epochs"]
        assert loaded_stage_config["data"]["entity_ids"] == ["machine-3-4"]
        assert loaded_stage_config["data"]["stride"] == 1
        assert loaded_stage_config["three_stage_phase"] == stage_record["phase_name"]
        assert loaded_stage_config["model"]["training_phase"] == stage_record["phase_name"]
        if stage_record["phase_name"] == "stage2_recovery":
            assert loaded_stage_config["initialization_checkpoint_path"].endswith(
                "stage2_recovery_init.pt"
            )
        if stage_record["phase_name"] == "stage3_memory_initialization_and_fusion_warmup":
            assert loaded_stage_config["initialization_checkpoint_path"].endswith(
                "stage2_recovery/checkpoints/best.pt"
            )
        if stage_record["phase_name"] == "multitask_pretraining":
            assert loaded_stage_config["initialization_checkpoint_path"].endswith(
                "stage3_memory_initialization_and_fusion_warmup/checkpoints/best.pt"
            )


def test_generated_stage_yaml_is_self_consistent_before_loader_resolution(tmp_path) -> None:
    experiment_config = load_experiment_config(
        "configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml"
    )
    experiment_config["output_dir"] = str(tmp_path / "outputs")
    experiment_config["checkpoint_dir"] = str(tmp_path / "outputs" / "checkpoints")

    manifest = materialize_three_stage_run_manifest(experiment_config)
    stage3_record = manifest["training_stages"][3]
    raw_stage3_yaml = Path(stage3_record["config_path"]).read_text(encoding="utf-8")

    assert (
        "stage3_memory_initialization_and_fusion_warmup_epochs" in raw_stage3_yaml
    )
    assert "stage3_prototype_warmup_epochs" not in raw_stage3_yaml
    assert "training_phase: stage3_memory_initialization_and_fusion_warmup" in raw_stage3_yaml
    assert "training_phase: multitask_pretraining" not in raw_stage3_yaml


def test_three_stage_execution_commands_follow_stage_manifest_order(tmp_path) -> None:
    experiment_config = load_experiment_config(
        "configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml"
    )
    experiment_config["output_dir"] = str(tmp_path / "outputs")
    experiment_config["checkpoint_dir"] = str(tmp_path / "outputs" / "checkpoints")
    manifest = materialize_three_stage_run_manifest(experiment_config)

    commands = build_three_stage_execution_commands(manifest)

    assert len(commands["training"]) == 5
    assert commands["training"][0][1].endswith("scripts/train.py")
    assert commands["training"][0][2] == "--experiment-config"
    assert commands["training"][0][3].endswith("01_stage1_classification.yaml")
    assert commands["training"][-1][3].endswith("05_multitask_pretraining.yaml")
    assert commands["evaluation"][1].endswith("scripts/evaluate.py")
    assert commands["evaluation"][2] == "--experiment-config"
    assert commands["evaluation"][3].endswith("06_evaluation_reference.yaml")
    assert commands["evaluation"][4] == "--checkpoint-path"
    assert commands["evaluation"][5].endswith(
        "multitask_pretraining/checkpoints/best.pt"
    )


def test_execute_three_stage_plan_dry_run_writes_report_without_spawning_processes(
    tmp_path, monkeypatch
) -> None:
    experiment_config = load_experiment_config(
        "configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml"
    )
    experiment_config["output_dir"] = str(tmp_path / "outputs")
    experiment_config["checkpoint_dir"] = str(tmp_path / "outputs" / "checkpoints")
    manifest = materialize_three_stage_run_manifest(experiment_config)

    called_commands: list[list[str]] = []

    def _fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        called_commands.append(list(args[0]))
        raise AssertionError("dry_run must not call subprocess.run")

    monkeypatch.setattr(
        "scripts.run_three_stage_offline_pretraining.subprocess.run",
        _fake_run,
    )

    execution_report = execute_three_stage_plan(manifest, dry_run=True)

    assert called_commands == []
    assert execution_report["dry_run"] is True
    assert execution_report["executed_stage_names"] == [
        "stage1_classification",
        "stage1_reconstruction",
        "stage2_recovery",
        "stage3_memory_initialization_and_fusion_warmup",
        "multitask_pretraining",
        "evaluation",
    ]
    assert execution_report["manifest_path"].endswith("three_stage_manifest.json")
    assert execution_report["execution_report_path"].endswith(
        "three_stage_execution_report.json"
    )
    assert execution_report["started_at_utc"].endswith("Z")
    assert execution_report["finished_at_utc"].endswith("Z")
    assert execution_report["stage2_recovery_initialization_checkpoint_path"].endswith(
        "three_stage/initializations/stage2_recovery_init.pt"
    )
    assert execution_report["evaluation_checkpoint_path"].endswith(
        "multitask_pretraining/checkpoints/best.pt"
    )
    assert execution_report["server_preflight_summary_path"].endswith(
        "three_stage/server_preflight_summary.json"
    )
    assert execution_report["optimizer_training_phase_names"] == [
        "stage1_classification",
        "stage1_reconstruction",
        "stage2_recovery",
        "stage3_memory_initialization_and_fusion_warmup",
        "multitask_pretraining",
    ]
    assert execution_report["optimizer_training_total_epochs"] == 5
    assert execution_report["statistical_procedure_names"] == [
        "stage2_mtz_parameter_zipping",
        "stage3_memory_initialization",
    ]
    assert (
        tmp_path / "outputs" / "three_stage" / "three_stage_execution_report.json"
    ).exists()


def test_execute_three_stage_plan_runs_training_stages_then_evaluation_in_order(
    tmp_path, monkeypatch
) -> None:
    experiment_config = load_experiment_config(
        "configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml"
    )
    experiment_config["output_dir"] = str(tmp_path / "outputs")
    experiment_config["checkpoint_dir"] = str(tmp_path / "outputs" / "checkpoints")
    manifest = materialize_three_stage_run_manifest(experiment_config)

    called_commands: list[list[str]] = []

    class _CompletedProcess:
        def __init__(self) -> None:
            self.returncode = 0

    def _fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        called_commands.append(list(args[0]))
        return _CompletedProcess()

    monkeypatch.setattr(
        "scripts.run_three_stage_offline_pretraining.subprocess.run",
        _fake_run,
    )
    monkeypatch.setattr(
        "scripts.run_three_stage_offline_pretraining._prepare_stage2_recovery_initialization_checkpoint",
        lambda manifest: tmp_path / "outputs" / "three_stage" / "initializations" / "stage2_recovery_init.pt",
    )

    execution_report = execute_three_stage_plan(manifest, dry_run=False)

    assert len(called_commands) == 6
    assert called_commands[0][1].endswith("scripts/train.py")
    assert called_commands[0][3].endswith("01_stage1_classification.yaml")
    assert called_commands[4][3].endswith("05_multitask_pretraining.yaml")
    assert called_commands[5][1].endswith("scripts/evaluate.py")
    assert execution_report["dry_run"] is False
    assert execution_report["executed_stage_names"][-1] == "evaluation"


def test_execute_three_stage_plan_writes_failure_report_when_a_stage_subprocess_fails(
    tmp_path, monkeypatch
) -> None:
    experiment_config = load_experiment_config(
        "configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml"
    )
    experiment_config["output_dir"] = str(tmp_path / "outputs")
    experiment_config["checkpoint_dir"] = str(tmp_path / "outputs" / "checkpoints")
    manifest = materialize_three_stage_run_manifest(experiment_config)

    called_commands: list[list[str]] = []

    def _fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        command = list(args[0])
        called_commands.append(command)
        if command[3].endswith("02_stage1_reconstruction.yaml"):
            raise subprocess.CalledProcessError(returncode=17, cmd=command)
        return type("_CompletedProcess", (), {"returncode": 0})()

    monkeypatch.setattr(
        "scripts.run_three_stage_offline_pretraining.subprocess.run",
        _fake_run,
    )

    with pytest.raises(subprocess.CalledProcessError):
        execute_three_stage_plan(manifest, dry_run=False)

    report_path = tmp_path / "outputs" / "three_stage" / "three_stage_execution_report.json"
    assert report_path.exists()
    execution_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert execution_report["status"] == "failed"
    assert execution_report["started_at_utc"].endswith("Z")
    assert execution_report["failed_at_utc"].endswith("Z")
    assert execution_report["failed_stage_name"] == "stage1_reconstruction"
    assert execution_report["failed_command"] == called_commands[1]
    assert execution_report["completed_stage_names"] == ["stage1_classification"]
    assert execution_report["executed_stage_names"] == [
        "stage1_classification",
        "stage1_reconstruction",
    ]
