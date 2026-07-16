from __future__ import annotations

from pathlib import Path

import torch
import yaml

from scripts.run_thesis_online_benchmark import run_thesis_online_benchmark


def _write_online_config(path: Path, output_dir: Path) -> None:
    config = {
        "experiment_name": "pytest-thesis-online-a0",
        "output_dir": str(output_dir),
        "checkpoint_dir": str(output_dir / "checkpoints"),
        "seed": 6,
        "device": "cpu",
        "data": {"dataset_name": "smd", "window_size": 20, "stride": 1},
        "model": {"model_name": "online_adaptation"},
        "task": {
            "task_name": "online_adaptation",
            "offline_variant": "O0",
            "entity_id": "machine-1-6",
            "seed": 6,
            "benchmark_mode": "smoke",
            "stage_name": "stage_b_fusion_finetuning",
        },
        "optimizer": {"learning_rate": 0.001, "weight_decay": 0.0},
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_thesis_online_a0_wrapper_writes_protocol_report(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "online.yaml"
    output_dir = tmp_path / "outputs"
    _write_online_config(config_path, output_dir)
    final_checkpoint_path = output_dir / "checkpoints" / "final.pt"
    final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "extra_state": {
                "online_runtime_state": {
                    "entity_id": "machine-1-6",
                    "online_variant": "A0",
                    "threshold_artifact": {"entity_id": "machine-1-6"},
                    "stream_cursor": 1,
                    "previous_ewma_score": 0.1,
                    "signature_history": [{"entity_id": "machine-1-6"}],
                    "recurrent_signatures": [],
                    "verification_entries": [],
                    "verification_history": [{"step": 1}],
                    "hard_old_intervals": [],
                }
            }
        },
        final_checkpoint_path,
    )

    def fake_online_run(*, experiment_config, protocol_config, online_variant, dry_run):
        assert experiment_config["task"]["reference_checkpoint_path"] == str(
            final_checkpoint_path
        )
        return {
            "final_checkpoint_path": str(final_checkpoint_path),
            "metric_history": [{"online/step": 1}],
            "records": [{"step": 1, "entity_ids": ["machine-1-6"]}],
            "threshold_artifact": {
                "entity_id": "machine-1-6",
                "thresholds": {"online_ewma_point": {"value": 0.4}},
            },
        }

    monkeypatch.setattr(
        "scripts.run_thesis_online_benchmark.run_thesis_online_tta_experiment",
        fake_online_run,
    )
    monkeypatch.setattr(
        "scripts.run_thesis_online_benchmark.resolve_stage_b_checkpoint",
        lambda experiment_config: final_checkpoint_path,
    )

    report = run_thesis_online_benchmark(
        experiment_config_path=str(config_path),
        protocol_config_path="configs/protocol/smd_window20_cleanval_q99_ewma09.yaml",
        online_variant="A0",
        dry_run=False,
    )

    report_path = output_dir / "benchmark" / "thesis_online_A0_benchmark_report.json"
    assert report_path.exists()
    assert report["online_variant"] == "A0"
    assert report["protocol"]["online_window_stride"] == 1
    assert report["online_execution"]["records"][0]["did_update"] is False
    assert report["retention_policy"] == "retain_for_eda"
    assert (
        output_dir / "retention" / "machine-1-6" / "A0" / "retention_summary.json"
    ).exists()
    assert (
        output_dir / "retention" / "machine-1-6" / "A0" / "online_runtime_state.json"
    ).exists()
    assert (
        output_dir
        / "retention"
        / "machine-1-6"
        / "A0"
        / "retention_bundle_manifest.json"
    ).exists()


def test_thesis_online_a0_wrapper_can_reduce_retention_to_summary_only(
    tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "online.yaml"
    output_dir = tmp_path / "outputs"
    _write_online_config(config_path, output_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["evaluation"] = {"retention_policy": "summary_only"}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    final_checkpoint_path = output_dir / "checkpoints" / "final.pt"
    final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "extra_state": {
                "online_runtime_state": {
                    "entity_id": "machine-1-6",
                    "online_variant": "A0",
                    "threshold_artifact": {"entity_id": "machine-1-6"},
                    "stream_cursor": 1,
                    "previous_ewma_score": 0.1,
                    "signature_history": [],
                    "recurrent_signatures": [],
                    "verification_entries": [],
                    "verification_history": [],
                    "hard_old_intervals": [],
                }
            }
        },
        final_checkpoint_path,
    )

    def fake_online_run(*, experiment_config, protocol_config, online_variant, dry_run):
        assert experiment_config["task"]["reference_checkpoint_path"] == str(
            final_checkpoint_path
        )
        return {
            "final_checkpoint_path": str(final_checkpoint_path),
            "metric_history": [{"online/step": 1}],
            "records": [{"step": 1, "entity_ids": ["machine-1-6"]}],
            "threshold_artifact": {
                "entity_id": "machine-1-6",
                "thresholds": {"online_ewma_point": {"value": 0.4}},
            },
        }

    monkeypatch.setattr(
        "scripts.run_thesis_online_benchmark.run_thesis_online_tta_experiment",
        fake_online_run,
    )
    monkeypatch.setattr(
        "scripts.run_thesis_online_benchmark.resolve_stage_b_checkpoint",
        lambda experiment_config: final_checkpoint_path,
    )

    report = run_thesis_online_benchmark(
        experiment_config_path=str(config_path),
        protocol_config_path="configs/protocol/smd_window20_cleanval_q99_ewma09.yaml",
        online_variant="A0",
        dry_run=False,
    )

    retention_root = output_dir / "retention" / "machine-1-6" / "A0"
    assert (retention_root / "retention_summary.json").exists()
    assert (retention_root / "retention_bundle_manifest.json").exists()
    assert not (retention_root / "online_metrics.json").exists()
    assert not (retention_root / "online_records.json").exists()
    assert report["retention_policy"] == "summary_only"
