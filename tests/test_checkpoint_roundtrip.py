from __future__ import annotations

from pathlib import Path

import torch

from src.data.scalers import SequenceStandardScaler
from src.engine.checkpoint import CheckpointManager
from src.models.reconstruction_mlp_ae import ReconstructionMLPAutoencoder


def test_checkpoint_roundtrip_restores_model_optimizer_scaler_and_config(tmp_path: Path) -> None:
    model = ReconstructionMLPAutoencoder(input_dim=38, encoder_dim=64, hidden_dim=16, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = SequenceStandardScaler()
    scaler.feature_mean = torch.zeros(38)
    scaler.feature_std = torch.ones(38)
    config = {"experiment_name": "unit-test"}

    checkpoint_manager = CheckpointManager(tmp_path)
    checkpoint_path = checkpoint_manager.save_checkpoint(
        checkpoint_name="roundtrip.pt",
        model=model,
        optimizer=optimizer,
        scaler_state=scaler.state_dict(),
        config=config,
        epoch=3,
        metric_history=[{"val_loss": 1.0}],
    )

    reloaded_model = ReconstructionMLPAutoencoder(input_dim=38, encoder_dim=64, hidden_dim=16, dropout=0.0)
    reloaded_optimizer = torch.optim.Adam(reloaded_model.parameters(), lr=1e-3)
    loaded_checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path, reloaded_model, reloaded_optimizer)

    assert loaded_checkpoint["config"] == config
    assert loaded_checkpoint["epoch"] == 3
    assert torch.equal(loaded_checkpoint["scaler_state_dict"]["feature_mean"], scaler.feature_mean)
    for parameter, reloaded_parameter in zip(model.parameters(), reloaded_model.parameters()):
        assert torch.allclose(parameter, reloaded_parameter)
