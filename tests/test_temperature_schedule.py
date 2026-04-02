from __future__ import annotations

from pathlib import Path

import torch

from src.engine.checkpoint import CheckpointManager
from src.engine.logger import ExperimentLogger
from src.engine.trainer import Trainer
from src.models.thesis_multitask import ThesisMultitaskModel


def _build_batch() -> dict[str, object]:
    return {
        "x": torch.randn(2, 100, 38),
        "point_labels": torch.zeros(2, 100, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }


def test_temperature_schedule_is_monotonic() -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        encoder_dim=64,
        hidden_dim=16,
        use_synthetic_augmentation=False,
        temperature_start=1.5,
        temperature_end=0.3,
        temperature_anneal_fraction=0.5,
    )

    observed_temperatures = []
    for epoch_index in range(4):
        model.set_epoch_context(epoch_index=epoch_index, total_epochs=4)
        observed_temperatures.append(model.get_schedule_state()["temperature"])

    assert observed_temperatures == sorted(observed_temperatures, reverse=True)
    assert observed_temperatures[0] == 1.5
    assert observed_temperatures[-1] == 0.3


def test_trainer_keeps_warmup_alpha_and_beta_fixed_for_configured_epochs(tmp_path: Path) -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        encoder_dim=64,
        hidden_dim=16,
        use_synthetic_augmentation=False,
        freeze_fusion_for_epochs=1,
        warmup_alpha_value=0.0,
        warmup_beta_value=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    experiment_logger = ExperimentLogger(tmp_path / "logs")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        checkpoint_manager=CheckpointManager(tmp_path / "checkpoints"),
        experiment_logger=experiment_logger,
        device="cpu",
    )
    batch = _build_batch()

    try:
        outputs = trainer.train(
            train_loader=[batch],
            val_loader=[batch],
            scaler_state={"feature_mean": torch.zeros(38), "feature_std": torch.ones(38)},
            config={"experiment_name": "schedule-test"},
            epochs=2,
        )
    finally:
        experiment_logger.close()

    assert outputs["metric_history"][0]["train_warmup_active"] == 1.0
    assert outputs["metric_history"][0]["train_alpha"] == 0.0
    assert outputs["metric_history"][0]["train_beta"] == 0.0
    assert outputs["metric_history"][1]["train_warmup_active"] == 0.0
    assert outputs["metric_history"][1]["train_alpha"] != 0.0
    assert outputs["metric_history"][1]["train_beta"] != 0.0
