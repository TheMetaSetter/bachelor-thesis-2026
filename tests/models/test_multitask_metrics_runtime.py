from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.evaluate import run_evaluation_experiment
from scripts.train import run_training_experiment
from src.engine.trainer import Trainer
from src.metrics.pointwise import compute_multiclass_classification_metrics
from src.models.thesis_multitask import ThesisMultitaskModel


def test_multitask_forward_outputs_include_forward_pass_timing() -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        window_size=100,
        encoder_dim=64,
        hidden_dim=16,
        num_classes=2,
        dropout=0.0,
        continuous_enabled=True,
        continuous_num_prototypes=4,
        discrete_enabled=True,
        discrete_codebook_size=8,
        gumbel_temperature=1.5,
        alpha_logit_init=0.0,
        beta_logit_init=0.0,
    )
    batch = {
        "x": torch.randn(4, 100, 38),
        "point_labels": torch.zeros(4, 100, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": f"machine-{index}"} for index in range(4)],
    }

    outputs = model(batch)

    assert "forward_pass_seconds" in outputs["aux"]
    assert outputs["aux"]["forward_pass_seconds"] >= 0.0


def test_compute_multiclass_classification_metrics_reports_classification_scores() -> (
    None
):
    logits = torch.tensor(
        [
            [4.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 2.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 1, 2, 2])

    metrics = compute_multiclass_classification_metrics(logits=logits, labels=labels)

    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] > 0.0
    assert metrics["weighted_f1"] > 0.0
    assert metrics["num_classes_observed"] == 3.0


def test_trainer_dispatches_twelve_class_logits_to_multiclass_metrics() -> None:
    trainer = Trainer.__new__(Trainer)
    logits = torch.eye(12)
    labels = torch.arange(12)

    metrics = trainer._aggregate_multitask_classification_metrics(
        logits_history=[logits],
        label_history=[labels],
        forward_pass_seconds_history=[],
        stage_name="val_synth",
    )

    assert metrics["val_synth_accuracy"] == 1.0
    assert metrics["val_synth_macro_f1"] == 1.0
    assert metrics["val_synth_weighted_f1"] == 1.0
    assert metrics["val_synth_num_classes_observed"] == 12.0
    assert "val_synth_roc_auc" not in metrics
    assert "val_synth_pr_auc" not in metrics


def test_run_training_experiment_logs_multitask_epoch_metrics(tmp_path: Path) -> None:
    experiment_config = {
        "experiment_name": "multitask_epoch_metrics_smoke",
        "seed": 7,
        "device": "cpu",
        "output_dir": str(tmp_path / "outputs"),
        "checkpoint_dir": str(tmp_path / "outputs" / "checkpoints"),
        "epochs": 1,
        "data": {
            "dataset_name": "smd",
            "root_dir": "data/ServerMachineDataset",
            "window_size": 100,
            "stride": 10,
            "batch_size": 2,
            "num_workers": 0,
            "validation_split_ratio": 0.2,
            "shuffle_train": True,
            "max_train_windows": 4,
            "max_val_windows": 2,
            "max_test_windows": 2,
        },
        "model": {
            "model_name": "thesis_multitask",
            "input_dim": 38,
            "encoder_dim": 16,
            "hidden_dim": 8,
            "num_classes": 2,
            "dropout": 0.0,
            "continuous_enabled": True,
            "continuous_num_prototypes": 4,
            "discrete_enabled": True,
            "discrete_codebook_size": 8,
            "gumbel_temperature": 1.5,
            "temperature_start": 1.5,
            "temperature_end": 0.7,
            "temperature_anneal_fraction": 0.8,
            "alpha_logit_init": 0.0,
            "beta_logit_init": 0.0,
            "lambda_cls": 1.0,
            "enable_diversity_loss": False,
            "enable_variance_loss": False,
            "enable_covariance_loss": False,
            "enable_usage_loss": True,
            "enable_gate_loss": False,
            "lambda_div": 0.0,
            "lambda_var": 0.0,
            "lambda_cov": 0.0,
            "lambda_use": 0.01,
            "lambda_gate": 0.0,
            "variance_floor_gamma": 1.0,
            "gate_barrier_margin": 0.25,
        },
        "task": {
            "task_name": "multitask_tsad",
            "use_synthetic_augmentation": True,
            "freeze_fusion_for_epochs": 0,
            "warmup_alpha_value": 0.5,
            "warmup_beta_value": 0.5,
            "anomaly_probability": 0.5,
            "min_segment_fraction": 0.1,
            "max_segment_fraction": 0.2,
            "spike_scale": 3.0,
            "anomaly_families": [
                "spike",
                "flip",
                "speedup",
                "noise",
                "cutoff",
                "average",
                "scale",
                "wander",
                "contextual",
                "upsidedown",
                "mixture",
            ],
        },
        "optimizer": {
            "learning_rate": 0.001,
            "weight_decay": 0.0,
        },
        "logging": {
            "use_wandb": False,
        },
    }

    training_outputs = run_training_experiment(experiment_config)
    epoch_metrics = training_outputs["metric_history"][-1]

    assert "train_precision" in epoch_metrics
    assert "train_recall" in epoch_metrics
    assert "train_roc_auc" in epoch_metrics
    assert "train_pr_auc" in epoch_metrics
    assert "train_fpr" in epoch_metrics
    assert "train_forward_pass_seconds_mean" in epoch_metrics
    assert "val_precision" in epoch_metrics
    assert "val_recall" in epoch_metrics
    assert "val_roc_auc" in epoch_metrics
    assert "val_pr_auc" in epoch_metrics
    assert "val_fpr" in epoch_metrics
    assert "val_forward_pass_seconds_mean" in epoch_metrics


def test_run_evaluation_experiment_writes_curves_and_logs_metrics_to_wandb(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _FakeRun:
        def __init__(self) -> None:
            self.logged_metrics: list[dict[str, object]] = []
            self.logged_artifacts: list[tuple[object, list[str] | None]] = []
            self.summary: dict[str, object] = {}
            self.finished = False

        def log(self, metrics: dict[str, object]) -> None:
            self.logged_metrics.append(metrics)

        def log_artifact(self, artifact, aliases=None) -> None:
            self.logged_artifacts.append((artifact, aliases))

        def finish(self) -> None:
            self.finished = True

    class _FakeArtifact:
        def __init__(self, name: str, type: str, metadata=None) -> None:
            self.name = name
            self.type = type
            self.metadata = metadata
            self.files: list[str] = []

        def add_file(self, file_path: str, name: str | None = None) -> None:
            self.files.append(name or file_path)

    fake_run = _FakeRun()
    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: fake_run,
        Artifact=_FakeArtifact,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr("scripts.evaluate.register_runtime_components", lambda: None)
    monkeypatch.setattr(
        "scripts.evaluate.build_dataset",
        lambda dataset_name, data_config: {
            "datasets": {"test": [1, 2]},
            "loaders": {"test": object()},
        },
    )
    monkeypatch.setattr(
        "scripts.evaluate.build_model_from_experiment_config",
        lambda experiment_config: torch.nn.Linear(1, 1),
    )

    class _FakeCheckpointManager:
        def __init__(self, checkpoint_dir) -> None:
            self.checkpoint_dir = checkpoint_dir

        def load_checkpoint(self, checkpoint_path, model, optimizer, strict=True):
            return {
                "scaler_state_dict": {
                    "feature_mean": torch.zeros(1),
                    "feature_std": torch.ones(1),
                },
            }

    class _FakeEvaluator:
        def __init__(self, device: str = "cpu") -> None:
            self.device = device

        def evaluate(
            self,
            model,
            data_loader,
            point_score_threshold=None,
            threshold_source=None,
        ):
            return {
                "metrics": {
                    "precision": 0.5,
                    "recall": 0.75,
                    "roc_auc": 0.8,
                    "pr_auc": 0.7,
                    "fpr": 0.25,
                    "threshold": 0.12,
                    "forward_pass_seconds_mean": 0.001,
                },
                "records": [
                    {
                        "entity_id": "machine-a",
                        "point_scores": torch.tensor([0.1, 0.9]),
                        "point_labels": torch.tensor([0, 1]),
                        "covered_point_mask": torch.tensor([1, 0], dtype=torch.bool),
                        "num_points": 2,
                    }
                ],
                "curves": {
                    "roc_curve": {
                        "x": [0.0, 1.0],
                        "y": [0.0, 1.0],
                        "thresholds": [1.0, 0.0],
                    },
                    "pr_curve": {"x": [0.0, 1.0], "y": [1.0, 0.5], "thresholds": [0.9]},
                },
            }

    monkeypatch.setattr("scripts.evaluate.CheckpointManager", _FakeCheckpointManager)
    monkeypatch.setattr("scripts.evaluate.Evaluator", _FakeEvaluator)

    experiment_config = {
        "experiment_name": "evaluation-metrics-test",
        "device": "cpu",
        "output_dir": str(tmp_path / "outputs"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "data": {"dataset_name": "smd"},
        "model": {"model_name": "thesis_multitask"},
        "task": {"task_name": "multitask_tsad"},
        "logging": {
            "use_wandb": True,
            "wandb_project": "bachelor-thesis-2026",
            "wandb_mode": "offline",
        },
    }

    outputs = run_evaluation_experiment(
        experiment_config, checkpoint_path=str(tmp_path / "best.pt")
    )

    curves_path = tmp_path / "outputs" / "evaluation_curves.json"
    metrics_path = tmp_path / "outputs" / "evaluation_metrics.json"
    records_path = tmp_path / "outputs" / "evaluation_records.json"
    assert outputs["metrics"]["fpr"] == 0.25
    assert outputs["metrics"]["forward_pass_seconds_mean"] == 0.001
    assert curves_path.exists()
    assert metrics_path.exists()
    saved_curves = json.loads(curves_path.read_text(encoding="utf-8"))
    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    saved_records = json.loads(records_path.read_text(encoding="utf-8"))
    assert "roc_curve" in saved_curves
    assert "pr_curve" in saved_curves
    assert saved_metrics["benchmark_comparability"] == "non_comparable"
    assert saved_metrics["protocol_status"] == "fallback_unknown"
    assert "label_regime" not in saved_metrics
    assert saved_metrics["threshold_source"] == "positive_support_quantile_0.99"
    assert saved_records[0]["covered_point_mask"] == [True, False]
    assert any(
        "evaluation/precision" in logged_metrics
        for logged_metrics in fake_run.logged_metrics
    )
    assert any(
        artifact.name == "evaluation-metrics-test-evaluation-curves"
        for artifact, _ in fake_run.logged_artifacts
    )


def test_run_evaluation_experiment_reuses_checkpoint_threshold_and_rebuilt_loader(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("scripts.evaluate.register_runtime_components", lambda: None)

    fake_bundle = {
        "datasets": {"test": [1]},
        "loaders": {"test": "initial-test-loader"},
        "raw_sequences": {"train": [], "val": [], "test": []},
        "scaled_sequences": {"train": [], "val": [], "test": []},
        "scaler": object(),
    }
    monkeypatch.setattr(
        "scripts.evaluate.build_dataset",
        lambda dataset_name, data_config: fake_bundle,
    )
    monkeypatch.setattr(
        "scripts.evaluate.build_model_from_experiment_config",
        lambda experiment_config: torch.nn.Linear(1, 1),
    )

    class _FakeCheckpointManager:
        def __init__(self, checkpoint_dir) -> None:
            self.checkpoint_dir = checkpoint_dir

        def load_checkpoint(self, checkpoint_path, model, optimizer, strict=True):
            return {
                "scaler_state_dict": {
                    "epsilon": 1.0e-6,
                    "feature_mean": torch.zeros(1),
                    "feature_std": torch.ones(1),
                },
                "extra_state": {
                    "evaluation_threshold": 0.42,
                    "evaluation_threshold_source": "checkpoint::val_synth_threshold",
                },
            }

    rebuilt_bundle = dict(fake_bundle)
    rebuilt_bundle["loaders"] = {"test": "rebuilt-test-loader"}
    monkeypatch.setattr("scripts.evaluate.CheckpointManager", _FakeCheckpointManager)
    monkeypatch.setattr(
        "scripts.evaluate.rebuild_dataset_bundle_with_scaler_state",
        lambda *, data_bundle, data_config, scaler_state_dict: rebuilt_bundle,
    )
    monkeypatch.setattr(
        "scripts.evaluate.build_dataset_protocol_audit_report",
        lambda **kwargs: {
            "dataset_name": "smd",
            "scaler_fit_scope": "train_only_before_windowing",
            "benchmark_comparability": "benchmark_comparable",
            "protocol_status": "benchmark_comparable_full_timeline",
            "splits": {},
            "warnings": [],
            "evaluation": {
                "evaluated_num_points": 0,
                "raw_num_points": 0,
                "is_truncated_evaluation": False,
            },
        },
    )

    captured_evaluator_call: dict[str, object] = {}

    class _FakeEvaluator:
        def __init__(
            self,
            device: str = "cpu",
            vus_max_buffer_size: int | None = None,
            vus_num_thresholds: int = 200,
        ) -> None:
            self.device = device
            self.vus_max_buffer_size = vus_max_buffer_size
            self.vus_num_thresholds = vus_num_thresholds

        def evaluate(
            self,
            model,
            data_loader,
            point_score_threshold=None,
            threshold_source=None,
        ):
            captured_evaluator_call["data_loader"] = data_loader
            captured_evaluator_call["point_score_threshold"] = point_score_threshold
            captured_evaluator_call["threshold_source"] = threshold_source
            return {
                "metrics": {
                    "precision": 0.5,
                    "recall": 0.75,
                    "roc_auc": 0.8,
                    "pr_auc": 0.7,
                    "fpr": 0.25,
                    "threshold": 0.42,
                    "threshold_source": "checkpoint::val_synth_threshold",
                    "forward_pass_seconds_mean": 0.001,
                },
                "records": [],
                "curves": {
                    "roc_curve": {"x": [], "y": [], "thresholds": []},
                    "pr_curve": {"x": [], "y": [], "thresholds": []},
                },
            }

    monkeypatch.setattr("scripts.evaluate.Evaluator", _FakeEvaluator)

    experiment_config = {
        "experiment_name": "evaluation-reuse-test",
        "device": "cpu",
        "output_dir": str(tmp_path / "outputs"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "data": {"dataset_name": "smd"},
        "model": {"model_name": "thesis_multitask"},
        "task": {"task_name": "multitask_tsad"},
        "logging": {"use_wandb": False},
    }

    run_evaluation_experiment(
        experiment_config, checkpoint_path=str(tmp_path / "best.pt")
    )

    assert captured_evaluator_call["data_loader"] == "rebuilt-test-loader"
    assert captured_evaluator_call["point_score_threshold"] == 0.42
    assert (
        captured_evaluator_call["threshold_source"] == "checkpoint::val_synth_threshold"
    )
