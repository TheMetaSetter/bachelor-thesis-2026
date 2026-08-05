from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.run_online_streaming_benchmark import run_online_streaming_benchmark


class _FakeOnlineBaseline:
    def __init__(self) -> None:
        self.calibrate_calls = 0
        self.run_calls = 0

    def calibrate(self, clean_validation_sequences, protocol_config, device: str):
        self.calibrate_calls += 1
        return {
            "threshold_artifact": {
                "schema_version": 3,
                "method_name": "THESIS",
                "variant_name": "O0-A0",
                "entity_id": "machine-1-6",
                "seed": 7,
                "window_size": 20,
                "calibration_split": "clean_validation",
                "stochastic_inference": True,
                "monte_carlo_samples": 10,
                "continuous_temperature": 0.9,
                "discrete_temperature": 0.9,
                "score_reduction": "mean",
                "variance_correction": 1,
                "numeric_precision": "fp32",
                "return_mc_samples": False,
                "sample_retention_policy": "none",
                "offline_point_threshold_nonoverlap": 0.3,
                "online_point_threshold_ewma": 0.4,
                "offline_stride": 20,
                "online_stride": 1,
                "ewma_current_weight": 0.9,
                "ewma_previous_weight": 0.1,
                "thresholds": {
                    "offline_point": {
                        "value": 0.3,
                        "source_split": "clean_validation",
                        "score_rule": "nonoverlap_tail_average",
                        "quantile": 0.99,
                    },
                    "online_ewma_point": {
                        "value": 0.4,
                        "source_split": "clean_validation",
                        "score_rule": "stride1_causal_endpoint_ewma",
                        "quantile": 0.99,
                        "ewma_current_weight": 0.9,
                        "ewma_previous_weight": 0.1,
                    },
                },
                "provenance": {
                    "created_by": "tests",
                    "config_path": "tests",
                    "calibration_split": "clean_validation",
                    "threshold_method": "THESIS",
                    "threshold_variant": "O0-A0",
                    "test_label_usage": "metrics_only",
                    "score_reduction": "mean",
                    "variance_correction": 1,
                    "numeric_precision": "fp32",
                },
            },
            "threshold_value": 0.4,
            "threshold_source": "clean_validation_stride1_ewma",
            "method_metadata": {
                "method": "fake",
                "online_variant": "main",
                "checkpoint_role": "pretrained_encoder",
                "checkpoint_sha256": "fixture-sha256",
            },
        }

    def run_sequence(self, sequence, threshold_value, protocol_config, device: str):
        self.run_calls += 1
        return (
            [
                {
                    "online/step": 1,
                    "online/raw_point_score": 0.1,
                    "online/ewma_point_score": 0.1,
                    "online/threshold": threshold_value,
                    "online/prediction": 0,
                    "online/did_update": False,
                    "online/loss_total": None,
                    "online/triage_decision": "hard_old_normality",
                    "online/verification_buffer_size": 0,
                }
            ],
            [
                {
                    "entity_id": sequence["meta"]["entity_id"],
                    "point_index": 19,
                    "window_start_index": 0,
                    "window_end_index": 20,
                    "raw_point_score": 0.1,
                    "ewma_point_score": 0.1,
                    "latent_window_score": 0.1,
                    "threshold": threshold_value,
                    "prediction": 0,
                    "online_variant": "A0",
                    "triage_decision": "hard_old_normality",
                    "did_update": False,
                    "loss_total": None,
                }
            ],
        )


class _RecordingOnlineBaseline(_FakeOnlineBaseline):
    def __init__(self) -> None:
        super().__init__()
        self.calibration_validation_lengths: list[int] = []
        self.run_sequence_lengths: list[int] = []

    def calibrate(self, clean_validation_sequences, protocol_config, device: str):
        self.calibration_validation_lengths.append(len(clean_validation_sequences))
        return super().calibrate(
            clean_validation_sequences=clean_validation_sequences,
            protocol_config=protocol_config,
            device=device,
        )

    def run_sequence(self, sequence, threshold_value, protocol_config, device: str):
        self.run_sequence_lengths.append(int(sequence["x"].shape[0]))
        return super().run_sequence(
            sequence=sequence,
            threshold_value=threshold_value,
            protocol_config=protocol_config,
            device=device,
        )


def _build_sequence(entity_id: str) -> dict[str, object]:
    return {
        "x": torch.randn(40, 38),
        "point_labels": torch.zeros(40, dtype=torch.long),
        "mask": torch.ones(40, 38),
        "timestamps": torch.arange(40),
        "meta": {
            "dataset_name": "smd",
            "entity_id": entity_id,
            "split": "test",
            "sequence_length": 40,
            "source_sequence_length": 40,
        },
    }


def test_online_streaming_benchmark_writes_shared_report(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "online_benchmark.yaml"
    protocol_path = tmp_path / "protocol.yaml"
    output_dir = tmp_path / "outputs"

    config_path.write_text(
        yaml.safe_dump(
            {
                "benchmark_name": "fake-online-benchmark",
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

    fake_baseline = _FakeOnlineBaseline()
    monkeypatch.setattr(
        "scripts.run_online_streaming_benchmark.BASELINE_BUILDERS",
        {"fake": lambda **kwargs: fake_baseline},
    )
    monkeypatch.setattr(
        "scripts.run_online_streaming_benchmark.load_yaml_config",
        lambda path: yaml.safe_load(Path(path).read_text(encoding="utf-8")),
    )
    monkeypatch.setattr(
        "scripts.run_online_streaming_benchmark.build_dataset",
        lambda name, config: {
            "scaled_sequences": {
                "val": [_build_sequence("machine-1-6")],
                "test": [_build_sequence("machine-1-6")],
            }
        },
    )
    monkeypatch.setattr(
        "scripts.run_online_streaming_benchmark.register_evaluation_runtime_components",
        lambda: None,
    )

    report = run_online_streaming_benchmark(
        benchmark_config_path=str(config_path),
        protocol_config_path=str(protocol_path),
        dry_run=False,
    )

    assert fake_baseline.calibrate_calls == 1
    assert fake_baseline.run_calls == 1
    assert report["online_execution"]["threshold_source"] == (
        "clean_validation_stride1_ewma"
    )
    assert report["method_metadata"]["checkpoint_role"] == "pretrained_encoder"
    assert report["online_execution"]["method_metadata"]["checkpoint_sha256"] == (
        "fixture-sha256"
    )
    assert (
        output_dir / "benchmark" / "online_streaming_benchmark_report.json"
    ).exists()
    assert (output_dir / "thresholds" / "online_thresholds.json").exists()
    assert (output_dir / "online_metrics.json").exists()
    assert (output_dir / "online_records.json").exists()
    threshold_payload = json.loads(
        (output_dir / "thresholds" / "online_thresholds.json").read_text(
            encoding="utf-8"
        )
    )
    assert threshold_payload["thresholds"]["online_ewma_point"]["value"] == 0.4


def test_online_streaming_benchmark_applies_overrides_and_max_online_steps(
    tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "online_benchmark.yaml"
    protocol_path = tmp_path / "protocol.yaml"
    output_dir = tmp_path / "outputs"

    config_path.write_text(
        yaml.safe_dump(
            {
                "benchmark_name": "fake-online-benchmark",
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
                "task_overrides": {
                    "absolute_start_index": 5,
                    "absolute_end_index": 30,
                    "max_online_steps": 2,
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

    fake_baseline = _RecordingOnlineBaseline()
    captured_data_configs: list[dict[str, object]] = []

    def _fake_build_dataset(name, config):
        captured_data_configs.append(dict(config))
        return {
            "scaled_sequences": {
                "train": [_build_sequence("machine-1-6")],
                "val": [_build_sequence("machine-1-6")],
                "test": [_build_sequence("machine-1-6")],
            }
        }

    monkeypatch.setattr(
        "scripts.run_online_streaming_benchmark.BASELINE_BUILDERS",
        {"fake": lambda **kwargs: fake_baseline},
    )
    monkeypatch.setattr(
        "scripts.run_online_streaming_benchmark.load_yaml_config",
        lambda path: yaml.safe_load(Path(path).read_text(encoding="utf-8")),
    )
    monkeypatch.setattr(
        "scripts.run_online_streaming_benchmark.build_dataset",
        _fake_build_dataset,
    )
    monkeypatch.setattr(
        "scripts.run_online_streaming_benchmark.register_evaluation_runtime_components",
        lambda: None,
    )

    report = run_online_streaming_benchmark(
        benchmark_config_path=str(config_path),
        protocol_config_path=str(protocol_path),
        dry_run=False,
    )

    assert captured_data_configs[0]["batch_size"] == 8
    assert captured_data_configs[0]["num_workers"] == 0
    assert captured_data_configs[0]["max_train_windows"] == 3
    assert captured_data_configs[0]["max_val_windows"] == 2
    assert captured_data_configs[0]["max_test_windows"] == 2
    assert fake_baseline.calibration_validation_lengths == [1]
    assert fake_baseline.run_sequence_lengths == [21]
    assert report["online_execution"]["stream_selections"] == [
        {
            "absolute_end_index": 26,
            "absolute_start_index": 5,
            "entity_id": "machine-1-6",
            "sequence_length": 21,
            "source_sequence_length": 40,
        }
    ]
