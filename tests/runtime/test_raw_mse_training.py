from __future__ import annotations

import pytest
import torch

from src.engine.checkpoint import CheckpointManager
from src.engine.logger import ExperimentLogger
from src.engine.trainer import Trainer
from tests.models.test_raw_reconstruction_loss import build_model, scaler_state


def build_trainer(tmp_path, model=None):
    model = build_model() if model is None else model
    return Trainer(
        model,
        torch.optim.Adam(model.parameters()),
        None,
        None,
        CheckpointManager(tmp_path / "checkpoints"),
        ExperimentLogger(
            tmp_path / "logs",
            {"experiment_name": "raw"},
            logging_config={"use_wandb": False},
        ),
    )


def test_trainer_attaches_raw_train_scaler_and_scores_mc_errors(tmp_path):
    trainer = build_trainer(tmp_path)
    trainer._configure_reconstruction_context(
        {"reconstruction_loss_space": "raw_input"}, scaler_state()
    )
    samples = torch.tensor([[[[1.0, 0.0]], [[-1.0, 0.0]]]])
    step = {
        "batch": {"x": torch.zeros(1, 1, 2)},
        "outputs": {
            "point_scores": torch.tensor([[999.0]]),
            "recon": samples.mean(dim=1),
            "aux": {"stochastic_query": {"reconstruction_samples": samples}},
        },
    }
    torch.testing.assert_close(
        trainer._validation_point_scores(step), torch.tensor([[2.0]])
    )
    assert trainer.model.reconstruction_loss_space == "raw_input"


def test_synthetic_metrics_use_supplied_clean_threshold(tmp_path):
    trainer = build_trainer(tmp_path)
    trainer._configure_reconstruction_context(
        {"reconstruction_loss_space": "raw_input"}, scaler_state()
    )

    class Dataset:
        sequences = [
            {
                "x": torch.zeros(4, 2),
                "point_labels": torch.zeros(4),
                "meta": {"entity_id": "toy"},
            }
        ]

    class Loader:
        dataset = Dataset()

    payloads = [
        {
            "meta": [{"entity_id": "toy", "start_index": 0, "end_index": 4}],
            "point_scores": torch.tensor([[1.0, 2.0, 8.0, 9.0]]),
            "point_labels": torch.tensor([[0, 0, 1, 1]]),
        }
    ]
    metrics = trainer._aggregate_reconstructed_pointwise_metrics(
        data_loader=Loader(),
        batch_payloads=payloads,
        stage_name="val_synth",
        threshold=3.0,
    )
    assert metrics["val_synth_threshold"] == 3.0
    assert metrics["val_synth_f1_pointwise"] == 1.0


def test_raw_training_rejects_sigmoid_evaluation_config(tmp_path):
    trainer = build_trainer(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        trainer._configure_reconstruction_context(
            {
                "reconstruction_loss_space": "raw_input",
                "evaluation": {"point_score_transform": "sigmoid"},
            },
            scaler_state(),
        )


def test_cli_raw_evaluation_fits_threshold_on_clean_loader():
    from scripts.cli import evaluate
    from src.engine.evaluator import Evaluator
    from tests.evaluation.test_raw_input_mse_scores import (
        _RawEvaluationModel,
        _RawEvaluationLoader,
        _fit_one_feature_scaler,
    )

    result = evaluate.evaluate_raw_checkpoint(
        Evaluator(device="cpu"),
        _RawEvaluationModel(),
        {
            "scaler": _fit_one_feature_scaler(),
            "loaders": {"val": _RawEvaluationLoader(), "test": _RawEvaluationLoader()},
        },
    )
    assert result["metrics"]["score_space"] == "raw_input"
    assert result["metrics"]["threshold"] == pytest.approx(0.25)
    assert result["metrics"]["threshold_source"] == "clean_validation_quantile"
    assert result["records"][0]["point_scores"].tolist() == [0.25, 0.0, 0.25]


def test_raw_checkpoint_threshold_metadata_names_clean_validation():
    from src.engine.thresholding import build_checkpoint_evaluation_metadata

    metadata = build_checkpoint_evaluation_metadata(
        checkpoint_monitor_metric="val_synth_vus_pr",
        epoch_metrics={
            "val_score_space": "raw_input",
            "val_threshold": 3.0,
            "val_synth_threshold": 9.0,
            "val_synth_vus_pr": 0.8,
        },
        base_extra_state=None,
    )
    assert metadata["evaluation_threshold"] == 3.0
    assert metadata["evaluation_threshold_source"] == "clean_validation_quantile"
    assert metadata["checkpoint_monitor_value"] == 0.8
