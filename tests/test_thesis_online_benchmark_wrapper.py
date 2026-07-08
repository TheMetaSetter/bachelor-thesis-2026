from __future__ import annotations

from pathlib import Path

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
        "task": {"task_name": "online_adaptation"},
        "optimizer": {"learning_rate": 0.001, "weight_decay": 0.0},
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_thesis_online_a0_wrapper_writes_protocol_report(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "online.yaml"
    output_dir = tmp_path / "outputs"
    _write_online_config(config_path, output_dir)

    def fake_online_run(experiment_config):
        return {
            "final_checkpoint_path": str(output_dir / "checkpoints" / "final.pt"),
            "metric_history": [{"online/step": 1}],
            "records": [{"step": 1, "entity_ids": ["machine-1-6"]}],
        }

    monkeypatch.setattr(
        "scripts.run_thesis_online_benchmark.run_online_adaptation_experiment",
        fake_online_run,
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
