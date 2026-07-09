from __future__ import annotations

from pathlib import Path

import json
import numpy as np
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


def test_thesis_offline_wrapper_exports_protocol_artifacts(tmp_path, monkeypatch) -> None:
    experiment_config_path = tmp_path / "experiment.yaml"
    output_dir = tmp_path / "outputs"
    _write_experiment_config(experiment_config_path, output_dir)

    def fake_materialize(experiment_config):
        return {
            "manifest_root": str(output_dir / "two_stage"),
            "evaluation": {"checkpoint_path": str(output_dir / "best.pt")},
        }

    def fake_execute(manifest, dry_run, skip_completed):
        return {"status": "completed", "dry_run": dry_run, "skip_completed": skip_completed}

    def fake_collect(*, experiment_config, protocol_config, manifest, execution_report):
        return {
            "entity_id": "machine-1-6",
            "seed": 6,
            "variant_name": "O0",
            "clean_validation": {
                "point_scores": np.array([0.1, 0.2, 0.3], dtype=float),
                "point_labels": np.array([0, 0, 0], dtype=np.int64),
                "covered_point_mask": np.array([True, True, True]),
            },
            "synthetic_validation": {
                "point_scores": np.array([0.2, 0.8], dtype=float),
                "point_labels": np.array([0, 1], dtype=np.int64),
                "covered_point_mask": np.array([True, True]),
            },
            "test": {
                "point_scores": np.array([0.05, 0.9], dtype=float),
                "point_labels": np.array([0, 1], dtype=np.int64),
                "covered_point_mask": np.array([True, True]),
            },
            "offline_metrics": {"point_f1": 1.0},
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
    monkeypatch.setattr(
        "scripts.run_thesis_offline_benchmark.collect_offline_artifact_inputs",
        fake_collect,
    )

    report = run_thesis_offline_benchmark(
        experiment_config_path=str(experiment_config_path),
        protocol_config_path="configs/protocol/smd_window20_cleanval_q99_ewma09.yaml",
        dry_run=False,
        skip_completed=True,
    )

    assert (output_dir / "thresholds" / "thresholds.json").exists()
    assert (output_dir / "scores" / "clean_validation_point_scores.npz").exists()
    assert (output_dir / "scores" / "synthetic_validation_point_scores.npz").exists()
    assert (output_dir / "scores" / "test_point_scores.npz").exists()
    assert (output_dir / "metrics" / "offline_metrics.json").exists()
    assert (output_dir / "protocol" / "resolved_protocol.json").exists()
    assert report["artifact_paths"]["thresholds"].endswith("thresholds.json")

    threshold_payload = json.loads(
        (output_dir / "thresholds" / "thresholds.json").read_text(encoding="utf-8")
    )
    assert threshold_payload["thresholds"]["offline_point"]["source_split"] == (
        "clean_validation"
    )
    assert threshold_payload["thresholds"]["offline_point"]["value"] == np.quantile(
        np.array([0.1, 0.2, 0.3], dtype=float),
        0.99,
    )
    clean_scores = np.load(output_dir / "scores" / "clean_validation_point_scores.npz")
    assert np.allclose(clean_scores["point_scores"], [0.1, 0.2, 0.3])
