from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from scripts.run_comparative_smd_experiments import (
    build_comparative_run_plan,
    execute_comparative_run_plan,
)
from src.core.config import load_experiment_config


def _write_placeholder_config(config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("placeholder: true\n", encoding="utf-8")


def _build_stub_config(
    *,
    experiment_name: str,
    dataset_root: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    entity_id: str,
    seed: int,
    model_name: str,
) -> dict[str, object]:
    config: dict[str, object] = {
        "experiment_name": experiment_name,
        "seed": seed,
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "data": {
            "root_dir": str(dataset_root),
            "entity_ids": [entity_id],
        },
        "model": {
            "model_name": model_name,
        },
    }
    if model_name == "thesis_multitask":
        config["two_stage"] = {
            "expected_total_training_epochs": 30,
            "stage_a_multitask_epochs": 25,
            "stage_b_fusion_finetuning_epochs": 5,
        }
    return config


def test_build_comparative_run_plan_dispatches_thesis_and_baseline_without_duplicate_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "data" / "ServerMachineDataset"
    (dataset_root / "train").mkdir(parents=True)
    (dataset_root / "test").mkdir(parents=True)
    (dataset_root / "test_label").mkdir(parents=True)

    thesis_config_path = tmp_path / "configs" / "thesis.yaml"
    baseline_config_path = tmp_path / "configs" / "baseline.yaml"
    _write_placeholder_config(thesis_config_path)
    _write_placeholder_config(baseline_config_path)

    stub_configs = {
        thesis_config_path.resolve(): _build_stub_config(
            experiment_name="thesis_run",
            dataset_root=dataset_root,
            output_dir=tmp_path / "outputs" / "thesis",
            checkpoint_dir=tmp_path / "outputs" / "thesis" / "checkpoints",
            entity_id="machine-3-9",
            seed=6,
            model_name="thesis_multitask",
        ),
        baseline_config_path.resolve(): _build_stub_config(
            experiment_name="baseline_run",
            dataset_root=dataset_root,
            output_dir=tmp_path / "outputs" / "baseline",
            checkpoint_dir=tmp_path / "outputs" / "baseline" / "checkpoints",
            entity_id="machine-1-6",
            seed=36,
            model_name="redlamp_baseline",
        ),
    }

    monkeypatch.setattr(
        "scripts.run_comparative_smd_experiments.load_experiment_config",
        lambda config_path: stub_configs[Path(config_path).resolve()],
    )

    run_plan = build_comparative_run_plan(
        config_paths=[thesis_config_path, baseline_config_path],
        smoke_config_paths=[],
        report_dir=tmp_path / "reports",
    )

    assert len(run_plan["main_runs"]) == 2
    thesis_run = run_plan["main_runs"][0]
    baseline_run = run_plan["main_runs"][1]

    assert thesis_run["stage_family"] == "thesis_two_stage"
    assert len(thesis_run["commands"]) == 1
    assert thesis_run["commands"][0][1].endswith(
        "scripts/run_two_stage_offline_pretraining.py"
    )

    assert baseline_run["stage_family"] == "baseline_single_stage"
    assert len(baseline_run["commands"]) == 2
    assert baseline_run["commands"][0][1].endswith("scripts/train.py")
    assert baseline_run["commands"][1][1].endswith("scripts/evaluate.py")
    assert baseline_run["commands"][1][4] == "--checkpoint-path"
    assert baseline_run["commands"][1][5].endswith("baseline/checkpoints/best.pt")


def test_build_comparative_run_plan_rejects_missing_dataset_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "configs" / "missing-dataset.yaml"
    _write_placeholder_config(config_path)

    monkeypatch.setattr(
        "scripts.run_comparative_smd_experiments.load_experiment_config",
        lambda config_path: _build_stub_config(
            experiment_name="broken_run",
            dataset_root=tmp_path / "missing-dataset-root",
            output_dir=tmp_path / "outputs" / "broken",
            checkpoint_dir=tmp_path / "outputs" / "broken" / "checkpoints",
            entity_id="machine-3-1",
            seed=68,
            model_name="redlamp_baseline",
        ),
    )

    with pytest.raises(FileNotFoundError, match="Dataset root does not exist"):
        build_comparative_run_plan(
            config_paths=[config_path],
            smoke_config_paths=[],
            report_dir=tmp_path / "reports",
        )


def test_execute_comparative_run_plan_dry_run_writes_execution_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "data" / "ServerMachineDataset"
    (dataset_root / "train").mkdir(parents=True)
    (dataset_root / "test").mkdir(parents=True)
    (dataset_root / "test_label").mkdir(parents=True)

    config_path = tmp_path / "configs" / "baseline.yaml"
    _write_placeholder_config(config_path)
    stub_config = _build_stub_config(
        experiment_name="baseline_run",
        dataset_root=dataset_root,
        output_dir=tmp_path / "outputs" / "baseline",
        checkpoint_dir=tmp_path / "outputs" / "baseline" / "checkpoints",
        entity_id="machine-1-6",
        seed=36,
        model_name="redlamp_baseline",
    )
    monkeypatch.setattr(
        "scripts.run_comparative_smd_experiments.load_experiment_config",
        lambda config_path: stub_config,
    )

    run_plan = build_comparative_run_plan(
        config_paths=[config_path],
        smoke_config_paths=[],
        report_dir=tmp_path / "reports",
    )
    execution_report = execute_comparative_run_plan(run_plan, dry_run=True)

    assert execution_report["status"] == "dry_run"
    assert execution_report["completed_run_ids"] == []
    assert Path(execution_report["execution_report_path"]).exists()


def test_execute_comparative_run_plan_records_failed_command_and_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "data" / "ServerMachineDataset"
    (dataset_root / "train").mkdir(parents=True)
    (dataset_root / "test").mkdir(parents=True)
    (dataset_root / "test_label").mkdir(parents=True)

    config_path = tmp_path / "configs" / "baseline.yaml"
    _write_placeholder_config(config_path)
    stub_config = _build_stub_config(
        experiment_name="baseline_run",
        dataset_root=dataset_root,
        output_dir=tmp_path / "outputs" / "baseline",
        checkpoint_dir=tmp_path / "outputs" / "baseline" / "checkpoints",
        entity_id="machine-1-6",
        seed=36,
        model_name="redlamp_baseline",
    )
    monkeypatch.setattr(
        "scripts.run_comparative_smd_experiments.load_experiment_config",
        lambda config_path: stub_config,
    )
    run_plan = build_comparative_run_plan(
        config_paths=[config_path],
        smoke_config_paths=[],
        report_dir=tmp_path / "reports",
    )

    called_commands: list[list[str]] = []

    def _fake_run(command: list[str], cwd: Path, check: bool) -> None:
        called_commands.append(command)
        if command[1].endswith("scripts/evaluate.py"):
            raise subprocess.CalledProcessError(returncode=17, cmd=command)

    monkeypatch.setattr(
        "scripts.run_comparative_smd_experiments.subprocess.run",
        _fake_run,
    )

    with pytest.raises(subprocess.CalledProcessError):
        execute_comparative_run_plan(run_plan, dry_run=False)

    execution_report_path = Path(run_plan["execution_report_path"])
    execution_report = json.loads(execution_report_path.read_text(encoding="utf-8"))
    assert execution_report["status"] == "failed"
    assert execution_report["failed_run_id"] == "main:baseline_run"
    assert execution_report["failed_command"] == called_commands[-1]


def test_build_comparative_run_plan_can_generate_worker_override_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "data" / "ServerMachineDataset"
    (dataset_root / "train").mkdir(parents=True)
    (dataset_root / "test").mkdir(parents=True)
    (dataset_root / "test_label").mkdir(parents=True)

    config_path = tmp_path / "configs" / "baseline.yaml"
    _write_placeholder_config(config_path)
    stub_config = _build_stub_config(
        experiment_name="baseline_run",
        dataset_root=dataset_root,
        output_dir=tmp_path / "outputs" / "baseline",
        checkpoint_dir=tmp_path / "outputs" / "baseline" / "checkpoints",
        entity_id="machine-1-6",
        seed=36,
        model_name="redlamp_baseline",
    )
    stub_config["device"] = "cuda"
    stub_config["data"]["num_workers"] = 16
    monkeypatch.setattr(
        "scripts.run_comparative_smd_experiments.load_experiment_config",
        lambda config_path: stub_config,
    )

    run_plan = build_comparative_run_plan(
        config_paths=[config_path],
        smoke_config_paths=[],
        report_dir=tmp_path / "reports",
        data_num_workers_override=4,
    )

    run_record = run_plan["main_runs"][0]
    generated_config_path = Path(run_record["config_path"])
    assert generated_config_path.exists()
    assert "generated_configs" in str(generated_config_path)

    generated_config = json.loads(
        Path(run_plan["manifest_path"]).read_text(encoding="utf-8")
    )["main_runs"][0]
    assert generated_config["data_num_workers_override"] == 4
    assert generated_config["original_config_path"] == str(config_path.resolve())


def test_execute_comparative_run_plan_skips_completed_runs_with_existing_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "data" / "ServerMachineDataset"
    (dataset_root / "train").mkdir(parents=True)
    (dataset_root / "test").mkdir(parents=True)
    (dataset_root / "test_label").mkdir(parents=True)

    config_path = tmp_path / "configs" / "baseline.yaml"
    _write_placeholder_config(config_path)
    output_dir = tmp_path / "outputs" / "baseline"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "best.pt").write_text("checkpoint", encoding="utf-8")
    (output_dir / "evaluation_metrics.json").write_text("{}", encoding="utf-8")
    (output_dir / "evaluation_records.json").write_text("{}", encoding="utf-8")
    (output_dir / "evaluation_curves.json").write_text("{}", encoding="utf-8")

    stub_config = _build_stub_config(
        experiment_name="baseline_run",
        dataset_root=dataset_root,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        entity_id="machine-1-6",
        seed=36,
        model_name="redlamp_baseline",
    )
    monkeypatch.setattr(
        "scripts.run_comparative_smd_experiments.load_experiment_config",
        lambda config_path: stub_config,
    )
    run_plan = build_comparative_run_plan(
        config_paths=[config_path],
        smoke_config_paths=[],
        report_dir=tmp_path / "reports",
    )
    Path(run_plan["execution_report_path"]).write_text(
        json.dumps(
            {
                "status": "failed",
                "completed_run_ids": ["main:baseline_run"],
            }
        ),
        encoding="utf-8",
    )

    called_commands: list[list[str]] = []

    def _fake_run(command: list[str], cwd: Path, check: bool) -> None:
        called_commands.append(command)

    monkeypatch.setattr(
        "scripts.run_comparative_smd_experiments.subprocess.run",
        _fake_run,
    )

    execution_report = execute_comparative_run_plan(
        run_plan,
        dry_run=False,
        skip_completed=True,
    )

    assert called_commands == []
    assert execution_report["completed_run_ids"] == ["main:baseline_run"]
    assert execution_report["skipped_run_ids"] == ["main:baseline_run"]


def test_generated_worker_override_config_can_be_reloaded_by_real_config_loader(
    tmp_path: Path,
) -> None:
    run_plan = build_comparative_run_plan(
        config_paths=[
            "configs/experiment/comparative/baseline/"
            "smd__redlamp_baseline__comparative-single-stage-machine_1_6"
            "__w20__seed6__main.yaml"
        ],
        smoke_config_paths=[],
        report_dir=tmp_path / "reports",
        data_num_workers_override=4,
    )

    generated_config = load_experiment_config(run_plan["main_runs"][0]["config_path"])
    assert generated_config["data"]["num_workers"] == 4
