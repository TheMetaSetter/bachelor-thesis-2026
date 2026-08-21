from __future__ import annotations

from pathlib import Path

import torch
import yaml

from scripts.run_thesis_offline_benchmark import run_thesis_offline_benchmark
from scripts.run_thesis_online_benchmark import run_thesis_online_benchmark


def test_full_spec_runtime_readiness_exports_retention_for_offline_and_online(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Batch 9 focuses on the main experiment gates, so this test checks the two
    # wrapper entry points that a real run would execute: offline benchmarking
    # and online benchmarking.
    offline_output_dir = tmp_path / "offline_outputs"
    online_output_dir = tmp_path / "online_outputs"
    offline_config_path = tmp_path / "offline.yaml"
    online_config_path = tmp_path / "online.yaml"
    protocol_path = tmp_path / "protocol.yaml"

    offline_config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "offline-readiness",
                "output_dir": str(offline_output_dir),
                "checkpoint_dir": str(offline_output_dir / "checkpoints"),
                "epochs": 10,
                "device": "cpu",
                "data": {"dataset_name": "smd", "window_size": 20},
                "model": {"model_name": "thesis_multitask"},
                "task": {"task_name": "multitask_tsad"},
                "evaluation": {"retention_policy": "retain_for_eda"},
                "two_stage": {
                    "expected_total_training_epochs": 10,
                    "stage_a_multitask_epochs": 8,
                    "stage_b_fusion_finetuning_epochs": 2,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    online_config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "online-readiness",
                "output_dir": str(online_output_dir),
                "checkpoint_dir": str(online_output_dir / "checkpoints"),
                "seed": 6,
                "device": "cpu",
                "data": {"dataset_name": "smd", "window_size": 20, "stride": 1},
                "model": {"model_name": "online_adaptation"},
                "task": {
                    "task_name": "online_adaptation",
                    "offline_variant": "O0",
                    "entity_id": "machine-1-6",
                    "seed": 6,
                    "benchmark_mode": "main",
                    "stage_name": "stage_b_fusion_finetuning",
                },
                "evaluation": {"retention_policy": "retain_for_eda"},
                "optimizer": {"learning_rate": 0.001, "weight_decay": 0.0},
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

    monkeypatch.setattr(
        "scripts.run_thesis_offline_benchmark.materialize_two_stage_run_manifest",
        lambda experiment_config: {
            "evaluation": {"checkpoint_path": str(offline_output_dir / "best.pt")}
        },
    )
    monkeypatch.setattr(
        "scripts.run_thesis_offline_benchmark.execute_two_stage_plan",
        lambda manifest, dry_run, skip_completed: {"status": "completed"},
    )
    monkeypatch.setattr(
        "scripts.run_thesis_offline_benchmark.register_evaluation_runtime_components",
        lambda: None,
    )
    monkeypatch.setattr(
        "scripts.run_thesis_offline_benchmark.build_dataset",
        lambda dataset_name, config: {
            "loaders": {
                "val": {"split_name": "val"},
                "val_synth": {"split_name": "val_synth"},
                "test": {"split_name": "test"},
            }
        },
    )
    monkeypatch.setattr(
        "scripts.run_thesis_offline_benchmark.build_model_from_experiment_config",
        lambda experiment_config: torch.nn.Linear(1, 1),
    )
    monkeypatch.setattr(
        "scripts.run_thesis_offline_benchmark.CheckpointManager",
        type(
            "FakeCheckpointManager",
            (),
            {
                "__init__": lambda self, checkpoint_dir: None,
                "_stable_json_digest": staticmethod(lambda value: "stable-json-digest"),
                "load_checkpoint": lambda self, checkpoint_path, model, strict=False: {
                    "scaler_state_dict": {
                        "feature_mean": torch.zeros(1),
                        "feature_std": torch.ones(1),
                    }
                },
            },
        ),
    )
    monkeypatch.setattr(
        "scripts.run_thesis_offline_benchmark.Evaluator",
        type(
            "FakeEvaluator",
            (),
            {
                "__init__": lambda self, device, vus_max_buffer_size=None, vus_num_thresholds=200: (
                    None
                ),
                "evaluate": lambda self, model, data_loader, point_score_threshold=None, threshold_source=None: {
                    "records": [
                        {
                            "entity_id": "machine-1-6",
                            "point_scores": torch.tensor([0.1, 0.2]),
                            "point_labels": torch.tensor([0, 0]),
                            "covered_point_mask": torch.tensor([True, True]),
                        }
                    ],
                    "metrics": {"point_f1": 1.0},
                    "traces": [{"batch_index": 1}],
                },
            },
        ),
    )
    monkeypatch.setattr(
        "scripts.run_thesis_offline_benchmark.collect_offline_artifact_inputs",
        lambda **kwargs: {
            "entity_id": "machine-1-6",
            "seed": 6,
            "variant_name": "O0",
            "clean_validation": {
                "point_scores": torch.tensor([0.1, 0.2]).numpy(),
                "point_labels": torch.tensor([0, 0]).numpy(),
                "covered_point_mask": torch.tensor([True, True]).numpy(),
            },
            "synthetic_validation": {
                "point_scores": torch.tensor([0.3, 0.4]).numpy(),
                "point_labels": torch.tensor([0, 1]).numpy(),
                "covered_point_mask": torch.tensor([True, True]).numpy(),
            },
            "test": {
                "point_scores": torch.tensor([0.5, 0.6]).numpy(),
                "point_labels": torch.tensor([0, 1]).numpy(),
                "covered_point_mask": torch.tensor([True, True]).numpy(),
            },
            "clean_validation_traces": [{"batch_index": 1}],
            "synthetic_validation_traces": [{"batch_index": 2}],
            "test_traces": [{"batch_index": 3}],
            "offline_metrics": {"point_f1": 1.0},
        },
    )
    monkeypatch.setattr(
        "scripts.run_thesis_offline_benchmark.load_experiment_config",
        lambda path: yaml.safe_load(Path(path).read_text(encoding="utf-8")),
    )

    offline_report = run_thesis_offline_benchmark(
        experiment_config_path=str(offline_config_path),
        protocol_config_path=str(protocol_path),
        dry_run=False,
        skip_completed=True,
    )

    assert (
        offline_output_dir
        / "retention"
        / "machine-1-6"
        / "offline"
        / "retention_summary.json"
    ).exists()
    assert offline_report["retention_policy"] == "retain_for_eda"
    assert offline_report["retention_artifact_paths"]["summary"].endswith(
        "retention_summary.json"
    )

    final_checkpoint_path = online_output_dir / "checkpoints" / "final.pt"
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

    monkeypatch.setattr(
        "scripts.run_thesis_online_benchmark.run_thesis_online_tta_experiment",
        lambda **kwargs: {
            "final_checkpoint_path": str(final_checkpoint_path),
            "metric_history": [{"online/step": 1}],
            "records": [{"step": 1, "entity_ids": ["machine-1-6"]}],
            "threshold_artifact": {
                "entity_id": "machine-1-6",
                "thresholds": {"online_ewma_point": {"value": 0.4}},
            },
        },
    )
    monkeypatch.setattr(
        "scripts.run_thesis_online_benchmark.resolve_stage_b_checkpoint",
        lambda experiment_config: final_checkpoint_path,
    )
    monkeypatch.setattr(
        "scripts.run_thesis_online_benchmark.load_experiment_config",
        lambda path: yaml.safe_load(Path(path).read_text(encoding="utf-8")),
    )

    online_report = run_thesis_online_benchmark(
        experiment_config_path=str(online_config_path),
        protocol_config_path=str(protocol_path),
        online_variant="A0",
        dry_run=False,
    )

    assert (
        online_output_dir
        / "retention"
        / "machine-1-6"
        / "A0"
        / "retention_summary.json"
    ).exists()
    assert online_report["retention_policy"] == "retain_for_eda"
    assert online_report["online_execution"]["records"][0]["did_update"] is False
