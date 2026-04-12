from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.evaluate import run_evaluation_experiment
from scripts.train import run_training_experiment
from src.models.thesis_multitask import ThesisMultitaskModel


def test_multitask_forward_outputs_include_forward_pass_timing() -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
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

        def load_checkpoint(self, checkpoint_path, model, optimizer):
            return {
                "scaler_state_dict": {"feature_mean": torch.zeros(1), "feature_std": torch.ones(1)},
            }

    class _FakeScaler:
        def load_state_dict(self, state_dict) -> None:
            self.state_dict = state_dict

    class _FakeEvaluator:
        def __init__(self, device: str = "cpu") -> None:
            self.device = device

        def evaluate(self, model, data_loader):
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
                        "num_points": 2,
                    }
                ],
                "curves": {
                    "roc_curve": {"x": [0.0, 1.0], "y": [0.0, 1.0], "thresholds": [1.0, 0.0]},
                    "pr_curve": {"x": [0.0, 1.0], "y": [1.0, 0.5], "thresholds": [0.9]},
                },
            }

    monkeypatch.setattr("scripts.evaluate.CheckpointManager", _FakeCheckpointManager)
    monkeypatch.setattr("scripts.evaluate.SequenceStandardScaler", _FakeScaler)
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

    outputs = run_evaluation_experiment(experiment_config, checkpoint_path=str(tmp_path / "best.pt"))

    curves_path = tmp_path / "outputs" / "evaluation_curves.json"
    assert outputs["metrics"]["fpr"] == 0.25
    assert outputs["metrics"]["forward_pass_seconds_mean"] == 0.001
    assert curves_path.exists()
    saved_curves = json.loads(curves_path.read_text(encoding="utf-8"))
    assert "roc_curve" in saved_curves
    assert "pr_curve" in saved_curves
    assert any("evaluation/precision" in logged_metrics for logged_metrics in fake_run.logged_metrics)
    assert any(
        artifact.name == "evaluation-metrics-test-evaluation-curves"
        for artifact, _ in fake_run.logged_artifacts
    )
