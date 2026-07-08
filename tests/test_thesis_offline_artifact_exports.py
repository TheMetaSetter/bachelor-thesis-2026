from __future__ import annotations

from pathlib import Path

import yaml

from scripts.run_thesis_offline_benchmark import run_thesis_offline_benchmark


def _write_experiment_config(path: Path, output_dir: Path) -> None:
    config = {
        "experiment_name": "pytest-thesis-offline",
        "output_dir": str(output_dir),
        "checkpoint_dir": str(output_dir / "checkpoints"),
        "epochs": 30,
        "device": "cpu",
        "data": {"dataset_name": "smd", "window_size": 20},
        "model": {"model_name": "thesis_multitask"},
        "task": {"task_name": "multitask_tsad"},
        "two_stage": {
            "expected_total_training_epochs": 30,
            "stage_a_multitask_epochs": 25,
            "stage_b_fusion_finetuning_epochs": 5,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_thesis_offline_wrapper_writes_dry_run_report(tmp_path, monkeypatch) -> None:
    experiment_config_path = tmp_path / "experiment.yaml"
    output_dir = tmp_path / "outputs"
    _write_experiment_config(experiment_config_path, output_dir)

    def fake_materialize(experiment_config):
        return {
            "manifest_root": str(output_dir / "two_stage"),
            "evaluation": {"checkpoint_path": str(output_dir / "best.pt")},
        }

    def fake_execute(manifest, dry_run, skip_completed):
        return {
            "status": "dry_run" if dry_run else "completed",
            "dry_run": dry_run,
            "skip_completed": skip_completed,
            "manifest_path": str(output_dir / "two_stage" / "two_stage_manifest.json"),
        }

    monkeypatch.setattr(
        "scripts.run_thesis_offline_benchmark.materialize_two_stage_run_manifest",
        fake_materialize,
    )
    monkeypatch.setattr(
        "scripts.run_thesis_offline_benchmark.execute_two_stage_plan",
        fake_execute,
    )
    monkeypatch.setattr(
        "scripts.run_thesis_offline_benchmark.load_experiment_config",
        lambda path: yaml.safe_load(Path(path).read_text(encoding="utf-8")),
    )

    report = run_thesis_offline_benchmark(
        experiment_config_path=str(experiment_config_path),
        protocol_config_path="configs/protocol/smd_window20_cleanval_q99_ewma09.yaml",
        dry_run=True,
        skip_completed=True,
    )

    report_path = output_dir / "benchmark" / "thesis_offline_benchmark_report.json"
    assert report_path.exists()
    assert report["protocol"]["offline_tail_policy"] == "end_align"
    assert report["two_stage_execution"]["status"] == "dry_run"
    assert report["benchmark_status"] == "dry_run"
