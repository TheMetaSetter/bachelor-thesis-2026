from __future__ import annotations

from pathlib import Path

import torch

from src.engine.checkpoint import CheckpointManager
from src.engine.trainer import Trainer
from src.models.thesis_multitask import ThesisMultitaskModel


class _NoOpExperimentLogger:
    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics = metrics


def _build_batch(batch_size: int = 2) -> dict[str, object]:
    classification_labels = torch.zeros(batch_size, dtype=torch.long)
    if batch_size > 1:
        classification_labels[1::2] = 1
    return {
        "x": torch.randn(batch_size, 100, 38),
        "point_labels": torch.zeros(batch_size, 100, dtype=torch.long),
        "classification_labels": classification_labels,
        "synthetic_anomaly_mask": torch.zeros(batch_size, 100, dtype=torch.long),
        "augmentation_metadata": [
            {
                "is_synthetic_anomaly": bool(classification_labels[index].item()),
                "anomaly_family": "test",
                "anomaly_family_index": None,
                "start_index": None,
                "end_index": None,
                "affected_channels": [],
                "family_parameters_by_channel": {},
            }
            for index in range(batch_size)
        ],
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": f"machine-{index}"} for index in range(batch_size)],
    }


def _build_model(*, bootstrap_encoder_epochs: int) -> ThesisMultitaskModel:
    return ThesisMultitaskModel(
        input_dim=38,
        encoder_dim=32,
        hidden_dim=16,
        mlp_num_linear_layers=3,
        num_classes=2,
        dropout=0.0,
        continuous_enabled=True,
        continuous_num_prototypes=4,
        discrete_enabled=True,
        discrete_codebook_size=8,
        gumbel_temperature=1.0,
        temperature_start=1.0,
        temperature_end=1.0,
        temperature_anneal_fraction=1.0,
        alpha_logit_init=0.0,
        beta_logit_init=0.0,
        lambda_cls=1.0,
        lambda_div=0.0,
        lambda_var=0.0,
        lambda_cov=0.0,
        lambda_use=0.0,
        lambda_gate=0.0,
        bootstrap_encoder_epochs=bootstrap_encoder_epochs,
        use_synthetic_augmentation=False,
        use_synthetic_validation=False,
        anomaly_probability=0.5,
        min_segment_fraction=0.1,
        max_segment_fraction=0.2,
        spike_scale=3.0,
    )


def test_bootstrap_epochs_bypass_memory_and_keep_state_unchanged() -> None:
    model = _build_model(bootstrap_encoder_epochs=2)
    model.set_epoch_context(epoch_index=0, total_epochs=3)
    batch = _build_batch()
    memory_before = model.get_memory_tensor_state()

    step_output = model.training_step(batch)
    memory_after = model.get_memory_tensor_state()

    assert step_output["loss"].item() >= 0.0
    assert step_output["outputs"]["aux"]["continuous_branch"]["aux"][
        "memory_bypass_active"
    ]
    assert step_output["outputs"]["aux"]["discrete_branch"]["aux"][
        "memory_bypass_active"
    ]
    assert step_output["outputs"]["aux"]["memory"]["memory_initialized"] is False
    assert step_output["outputs"]["aux"]["memory"]["memory_training_enabled"] is False
    assert torch.equal(
        memory_before["continuous_prototype_bank"],
        memory_after["continuous_prototype_bank"],
    )
    assert torch.equal(
        memory_before["discrete_codebook"],
        memory_after["discrete_codebook"],
    )


def test_trainer_reaches_memory_initialization_boundary_after_bootstrap(
    tmp_path: Path,
) -> None:
    model = _build_model(bootstrap_encoder_epochs=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scheduler_monitor_metric=None,
        checkpoint_manager=CheckpointManager(tmp_path / "checkpoints"),
        experiment_logger=_NoOpExperimentLogger(),
        device="cpu",
    )
    train_loader = [_build_batch()]
    val_loader = [_build_batch()]

    training_outputs = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        scaler_state={"feature_mean": torch.zeros(38), "feature_std": torch.ones(38)},
        config={"model": {"model_name": "thesis_multitask"}},
        epochs=2,
    )
    best_checkpoint = torch.load(training_outputs["best_checkpoint_path"])

    assert training_outputs["metric_history"][0]["train_memory_mode"] == 0.0
    assert training_outputs["metric_history"][1]["train_memory_ready_for_initialization"] == 1.0
    assert model.memory_ready_for_initialization is True
    assert model.memory_initialized is False
    assert best_checkpoint["extra_state"]["memory_ready_for_initialization"] is True
