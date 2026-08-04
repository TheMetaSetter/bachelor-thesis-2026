from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from demo.offline_replay import build_offline_replay_state
from demo.online_replay import build_online_replay_state
from demo.online_replay import _thesis_vector_series


def _build_sequence(entity_id: str) -> dict[str, object]:
    return {
        "x": torch.randn(16, 3),
        "point_labels": torch.tensor([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        "mask": torch.ones(16, 3),
        "timestamps": torch.arange(16),
        "meta": {
            "dataset_name": "smd",
            "entity_id": entity_id,
            "split": "test",
            "sequence_length": 16,
            "source_sequence_length": 16,
        },
    }


def test_demo_state_builders_load_offline_and_online_replays(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "outputs"
    data_config_path = tmp_path / "data.yaml"
    data_config_path.write_text(
        yaml.safe_dump(
            {
                "dataset_name": "smd",
                "root_dir": "data/SMD",
                "entity_ids": ["machine-1-6"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    offline_report_path = output_dir / "benchmark" / "offline_benchmark_report.json"
    offline_report_path.parent.mkdir(parents=True, exist_ok=True)
    offline_threshold_path = output_dir / "thresholds" / "thresholds.json"
    offline_threshold_path.parent.mkdir(parents=True, exist_ok=True)
    offline_scores_path = output_dir / "scores" / "test_point_scores.npz"
    offline_scores_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        offline_scores_path,
        point_scores=np.array([0.1, 0.2, 0.9, 1.1], dtype=np.float64),
        point_labels=np.array([0, 0, 1, 1], dtype=np.int64),
        covered_point_mask=np.array([True, True, True, True], dtype=bool),
    )
    offline_threshold_path.write_text(
        json.dumps(
            {
                "method_name": "stumpy",
                "variant_name": "stumpy",
                "entity_id": "machine_1_6",
                "seed": 6,
                "thresholds": {
                    "offline_point": {
                        "value": 0.5,
                        "source_split": "clean_validation",
                        "score_rule": "nonoverlap_tail_average",
                    },
                    "online_ewma_point": {
                        "value": 0.7,
                        "source_split": "clean_validation",
                        "score_rule": "stride1_causal_endpoint_ewma",
                    },
                },
                "provenance": {"test_label_usage": "metrics_only"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    offline_report_path.write_text(
        json.dumps(
            {
                "benchmark_config": {
                    "baseline_name": "stumpy",
                    "seed": 6,
                    "data_config_path": str(data_config_path),
                },
                "protocol": {"test_label_usage": "metrics_only"},
                "artifact_paths": {
                    "thresholds": str(offline_threshold_path),
                    "test_scores": str(offline_scores_path),
                },
                "offline_metrics": {"point_f1": 0.9},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    online_report_path = (
        output_dir / "benchmark" / "online_streaming_benchmark_report.json"
    )
    online_threshold_path = output_dir / "thresholds" / "online_thresholds.json"
    online_threshold_path.parent.mkdir(parents=True, exist_ok=True)
    online_threshold_path.write_text(
        json.dumps(
            {
                "method_name": "candi",
                "variant_name": "A1",
                "entity_id": "machine_1_6",
                "seed": 8,
                "thresholds": {
                    "offline_point": {
                        "value": 0.4,
                        "source_split": "clean_validation",
                        "score_rule": "nonoverlap_tail_average",
                    },
                    "online_ewma_point": {
                        "value": 0.6,
                        "source_split": "clean_validation",
                        "score_rule": "stride1_causal_endpoint_ewma",
                    },
                },
                "provenance": {"test_label_usage": "metrics_only"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    online_report_path.write_text(
        json.dumps(
            {
                "benchmark_config": {
                    "baseline_name": "candi",
                    "online_variant": "A1",
                    "seed": 8,
                    "data_config_path": str(data_config_path),
                },
                "protocol": {"test_label_usage": "metrics_only"},
                "artifact_paths": {
                    "thresholds": str(online_threshold_path),
                },
                "online_execution": {
                    "baseline_name": "candi",
                    "online_variant": "A1",
                    "entity_id": "machine_1_6",
                    "seed": 8,
                    "threshold_source": "clean_validation_stride1_ewma",
                    "threshold_value": 0.6,
                    "metric_history": [{"online/step": 1}, {"online/step": 2}],
                    "records": [
                        {
                            "raw_point_score": 0.1,
                            "ewma_point_score": 0.1,
                            "did_update": True,
                        },
                        {
                            "raw_point_score": 0.3,
                            "ewma_point_score": 0.4,
                            "did_update": False,
                        },
                    ],
                    "threshold_artifact": json.loads(
                        online_threshold_path.read_text(encoding="utf-8")
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "demo.loaders.register_evaluation_runtime_components", lambda: None
    )
    monkeypatch.setattr(
        "demo.loaders.load_experiment_config",
        lambda path: {"data_config_path": str(data_config_path)},
    )
    monkeypatch.setattr(
        "demo.loaders.build_dataset",
        lambda name, config: {
            "scaled_sequences": {"test": [_build_sequence("machine_1_6")]}
        },
    )

    offline_state = build_offline_replay_state(offline_report_path)
    online_state = build_online_replay_state(online_report_path)

    assert offline_state.method == "stumpy"
    assert offline_state.threshold == 0.5
    assert offline_state.predicted_mask.tolist() == [False, False, True, True]
    assert offline_state.metrics["point_f1"] == 0.9
    assert online_state.method == "candi"
    assert online_state.variant == "A1"
    assert online_state.threshold == 0.6
    assert online_state.predicted_mask.tolist() == [False, False]
    assert online_state.metrics["num_updates"] == 1


def test_demo_state_builders_accept_runner_report_shape_without_benchmark_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "outputs"
    data_config_path = tmp_path / "data.yaml"
    experiment_config_path = tmp_path / "experiment.yaml"
    data_config_path.write_text(
        yaml.safe_dump(
            {
                "dataset_name": "smd",
                "root_dir": "data/SMD",
                "entity_ids": ["machine-1-6"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    experiment_config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "pytest-demo",
                "output_dir": str(output_dir),
                "checkpoint_dir": str(output_dir / "checkpoints"),
                "seed": 6,
                "device": "cpu",
                "data_config_path": "configs/data/smd_benchmark_machine_1_6_window20.yaml",
                "model_config_path": "configs/model/thesis_multitask.yaml",
                "task_config_path": "configs/task/multitask_tsad.yaml",
                "data": {"dataset_name": "smd"},
                "model": {"model_name": "thesis_multitask"},
                "task": {"task_name": "multitask_tsad"},
                "optimizer": {"learning_rate": 0.001, "weight_decay": 0.0},
                "epochs": 30,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    offline_report_path = (
        output_dir / "benchmark" / "thesis_offline_benchmark_report.json"
    )
    offline_report_path.parent.mkdir(parents=True, exist_ok=True)
    offline_threshold_path = output_dir / "thresholds" / "thresholds.json"
    offline_threshold_path.parent.mkdir(parents=True, exist_ok=True)
    offline_scores_path = output_dir / "scores" / "test_point_scores.npz"
    offline_scores_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        offline_scores_path,
        point_scores=np.array([0.1, 0.2, 0.9, 1.1], dtype=np.float64),
        point_labels=np.array([0, 0, 1, 1], dtype=np.int64),
        covered_point_mask=np.array([True, True, True, True], dtype=bool),
    )
    offline_threshold_path.write_text(
        json.dumps(
            {
                "method_name": "THESIS",
                "variant_name": "O0",
                "entity_id": "machine_1_6",
                "seed": 6,
                "thresholds": {
                    "offline_point": {
                        "value": 0.5,
                        "source_split": "clean_validation",
                        "score_rule": "nonoverlap_tail_average",
                    },
                    "online_ewma_point": {
                        "value": 0.7,
                        "source_split": "clean_validation",
                        "score_rule": "stride1_causal_endpoint_ewma",
                    },
                },
                "provenance": {"test_label_usage": "metrics_only"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    offline_report_path.write_text(
        json.dumps(
            {
                "experiment_config_path": str(experiment_config_path),
                "protocol": {"test_label_usage": "metrics_only"},
                "artifact_paths": {
                    "thresholds": str(offline_threshold_path),
                    "test_scores": str(offline_scores_path),
                },
                "offline_metrics": {"point_f1": 0.9},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    online_report_path = (
        output_dir / "benchmark" / "thesis_online_A0_benchmark_report.json"
    )
    online_threshold_path = output_dir / "thresholds" / "online_thresholds.json"
    online_threshold_path.parent.mkdir(parents=True, exist_ok=True)
    online_threshold_path.write_text(
        json.dumps(
            {
                "method_name": "THESIS",
                "variant_name": "A0",
                "entity_id": "machine_1_6",
                "seed": 6,
                "thresholds": {
                    "offline_point": {
                        "value": 0.4,
                        "source_split": "clean_validation",
                        "score_rule": "nonoverlap_tail_average",
                    },
                    "online_ewma_point": {
                        "value": 0.6,
                        "source_split": "clean_validation",
                        "score_rule": "stride1_causal_endpoint_ewma",
                    },
                },
                "provenance": {"test_label_usage": "metrics_only"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    online_report_path.write_text(
        json.dumps(
            {
                "experiment_config_path": str(experiment_config_path),
                "protocol": {"test_label_usage": "metrics_only"},
                "artifact_paths": {"thresholds": str(online_threshold_path)},
                "online_execution": {
                    "baseline_name": "THESIS",
                    "online_variant": "A0",
                    "entity_id": "machine_1_6",
                    "seed": 6,
                    "threshold_source": "clean_validation_stride1_ewma",
                    "threshold_value": 0.6,
                    "metric_history": [{"online/step": 1}],
                    "records": [
                        {
                            "causal_window": {"absolute_indices": [0, 1]},
                            "window_point_scores": [0.1, 0.1],
                            "current_window_ewma_point_scores": [0.1, 0.1],
                            "window_point_predictions": [0, 0],
                            "did_update": False,
                        }
                    ],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "demo.loaders.register_evaluation_runtime_components", lambda: None
    )
    monkeypatch.setattr(
        "demo.loaders.load_experiment_config",
        lambda path: {"data_config_path": str(data_config_path)},
    )
    monkeypatch.setattr(
        "demo.loaders.build_dataset",
        lambda name, config: {
            "scaled_sequences": {"test": [_build_sequence("machine_1_6")]}
        },
    )

    offline_state = build_offline_replay_state(offline_report_path)
    online_state = build_online_replay_state(online_report_path)

    assert offline_state.method == "THESIS"
    assert offline_state.variant == "O0"
    assert online_state.method == "THESIS"
    assert online_state.variant == "A0"


def test_thesis_vector_series_keeps_latest_value_for_overlapping_points() -> None:
    indices, raw_scores, ewma_scores, predictions = _thesis_vector_series(
        [
            {
                "causal_window": {"absolute_indices": [10, 11, 12]},
                "window_point_scores": [0.1, 0.2, 0.3],
                "current_window_ewma_point_scores": [0.1, 0.2, 0.3],
                "window_point_predictions": [0, 0, 1],
            },
            {
                "causal_window": {"absolute_indices": [11, 12, 13]},
                "window_point_scores": [0.4, 0.5, 0.6],
                "current_window_ewma_point_scores": [0.35, 0.45, 0.6],
                "window_point_predictions": [0, 1, 1],
            },
        ]
    )

    assert indices.tolist() == [10, 11, 12, 13]
    assert raw_scores.tolist() == [0.1, 0.4, 0.5, 0.6]
    assert ewma_scores.tolist() == [0.1, 0.35, 0.45, 0.6]
    assert predictions.tolist() == [False, False, True, True]
