from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.run_offline_benchmark import run_offline_benchmark


class _FakeBaseline:
    def __init__(self) -> None:
        self.fit_calls = 0
        self.calibrate_calls = 0
        self.score_calls = 0

    def fit(self, train_sequence):
        self.fit_calls += 1
        return self

    def calibrate(self, clean_validation_sequence):
        self.calibrate_calls += 1
        return {
            "threshold": 0.5,
            "point_center": 0.0,
            "point_scale": 1.0,
            "validation_point_scores": np.array([0.1, 0.2, 0.3], dtype=float),
            "validation_covered_mask": np.array([True, True, True]),
            "validation_window_scores": np.array([0.1, 0.2], dtype=float),
            "method_metadata": {"name": "fake"},
        }

    def score_sequence(self, query_sequence):
        self.score_calls += 1
        return np.linspace(0.0, 1.0, query_sequence.shape[0], dtype=float)


def _sequence(values: list[float], labels: list[int]) -> dict[str, object]:
    return {
        "x": torch.tensor(values, dtype=torch.float32).unsqueeze(1),
        "point_labels": torch.tensor(labels, dtype=torch.long),
        "meta": {"entity_id": "machine-1-6"},
    }


def test_run_offline_benchmark_writes_shared_artifacts(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "offline_benchmark.yaml"
    output_dir = tmp_path / "outputs"
    protocol_path = tmp_path / "protocol.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "benchmark_name": "fake-benchmark",
                "baseline_name": "fake",
                "baseline_kwargs": {},
                "data_config_path": "configs/data/smd_benchmark_machine_1_6_window20.yaml",
                "output_dir": str(output_dir),
                "protocol_config_path": str(protocol_path),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    protocol_path.write_text(
        yaml.safe_dump(
            {
                "protocol_name": "smd_window20_cleanval_q99_ewma09",
                "window_size": 20,
                "offline_tail_policy": "end_align",
                "offline_threshold_split": "clean_validation",
                "offline_threshold_quantile": 0.99,
                "online_window_stride": 1,
                "online_threshold_split": "clean_validation",
                "online_threshold_quantile": 0.99,
                "online_ewma_current_weight": 0.9,
                "online_ewma_previous_weight": 0.1,
                "test_label_usage": "metrics_only",
                "point_adjustment": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    fake_baseline = _FakeBaseline()
    monkeypatch.setattr(
        "scripts.run_offline_benchmark.BASELINE_BUILDERS",
        {"fake": lambda **kwargs: fake_baseline},
    )
    monkeypatch.setattr(
        "scripts.run_offline_benchmark.register_evaluation_runtime_components",
        lambda: None,
    )
    monkeypatch.setattr(
        "scripts.run_offline_benchmark.load_yaml_config",
        lambda path: yaml.safe_load(Path(path).read_text(encoding="utf-8")),
    )
    monkeypatch.setattr(
        "scripts.run_offline_benchmark.build_dataset",
        lambda name, config: {
            "scaled_sequences": {
                "train": [_sequence([0.0] * 40, [0] * 40)],
                "val": [_sequence([0.0] * 40, [0] * 40)],
                "val_synth": [_sequence([0.0] * 40, [0, 0] * 20)],
                "test": [_sequence([0.0] * 40, [0, 1] * 20)],
            }
        },
    )

    report = run_offline_benchmark(
        benchmark_config_path=str(config_path),
        dry_run=False,
    )

    report_path = output_dir / "benchmark" / "offline_benchmark_report.json"
    assert report_path.exists()
    assert fake_baseline.fit_calls == 1
    assert fake_baseline.calibrate_calls == 1
    assert fake_baseline.score_calls == 3
    assert report["benchmark_status"] == "completed"
    assert report["thresholds"]["offline_point"]["value"] == 0.5

    threshold_payload = json.loads(
        (output_dir / "thresholds" / "thresholds.json").read_text(encoding="utf-8")
    )
    assert threshold_payload["method_name"] == "fake"
    assert threshold_payload["thresholds"]["offline_point"]["source_split"] == (
        "clean_validation"
    )
    assert (output_dir / "scores" / "test_point_scores.npz").exists()
    assert (output_dir / "metrics" / "offline_metrics.json").exists()


def test_run_offline_benchmark_applies_data_overrides(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "offline_benchmark.yaml"
    output_dir = tmp_path / "outputs"
    protocol_path = tmp_path / "protocol.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "benchmark_name": "fake-benchmark",
                "baseline_name": "fake",
                "baseline_kwargs": {},
                "data_config_path": "configs/data/smd_benchmark_machine_1_6_window20.yaml",
                "output_dir": str(output_dir),
                "protocol_config_path": str(protocol_path),
                "data_overrides": {
                    "batch_size": 8,
                    "num_workers": 0,
                    "max_train_windows": 3,
                    "max_val_windows": 2,
                    "max_test_windows": 2,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    protocol_path.write_text(
        yaml.safe_dump(
            {
                "protocol_name": "smd_window20_cleanval_q99_ewma09",
                "window_size": 20,
                "offline_tail_policy": "end_align",
                "offline_threshold_split": "clean_validation",
                "offline_threshold_quantile": 0.99,
                "online_window_stride": 1,
                "online_threshold_split": "clean_validation",
                "online_threshold_quantile": 0.99,
                "online_ewma_current_weight": 0.9,
                "online_ewma_previous_weight": 0.1,
                "test_label_usage": "metrics_only",
                "point_adjustment": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    fake_baseline = _FakeBaseline()
    captured_data_configs: list[dict[str, object]] = []

    def _fake_build_dataset(name, config):
        captured_data_configs.append(dict(config))
        return {
            "scaled_sequences": {
                "train": [_sequence([0.0] * 40, [0] * 40)],
                "val": [_sequence([0.0] * 40, [0] * 40)],
                "val_synth": [_sequence([0.0] * 40, [0, 0] * 20)],
                "test": [_sequence([0.0] * 40, [0, 1] * 20)],
            }
        }

    monkeypatch.setattr(
        "scripts.run_offline_benchmark.BASELINE_BUILDERS",
        {"fake": lambda **kwargs: fake_baseline},
    )
    monkeypatch.setattr(
        "scripts.run_offline_benchmark.register_evaluation_runtime_components",
        lambda: None,
    )
    monkeypatch.setattr(
        "scripts.run_offline_benchmark.load_yaml_config",
        lambda path: yaml.safe_load(Path(path).read_text(encoding="utf-8")),
    )
    monkeypatch.setattr(
        "scripts.run_offline_benchmark.build_dataset",
        _fake_build_dataset,
    )

    run_offline_benchmark(
        benchmark_config_path=str(config_path),
        dry_run=False,
    )

    assert captured_data_configs[0]["batch_size"] == 8
    assert captured_data_configs[0]["num_workers"] == 0
    assert captured_data_configs[0]["max_train_windows"] == 3
    assert captured_data_configs[0]["max_val_windows"] == 2
    assert captured_data_configs[0]["max_test_windows"] == 2
