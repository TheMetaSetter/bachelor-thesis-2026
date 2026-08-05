from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.baselines.online.redlamp_encoder_checkpoint import (
    load_redlamp_encoder_checkpoint,
)
from src.models.simple_window_cnn_autoencoder import SimpleWindowCnnAutoencoder


def _model() -> SimpleWindowCnnAutoencoder:
    return SimpleWindowCnnAutoencoder(
        input_dim=3,
        latent_dim=128,
        hidden_channels=64,
        kernel_size=3,
        num_layers=3,
        dropout=0.1,
    )


def test_loader_reads_encoder_and_ignores_redlamp_heads(tmp_path: Path) -> None:
    source_model = _model()
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": source_model.state_dict(),
            "classification_head.fake": torch.tensor(1),
            "epoch": 100,
        },
        checkpoint_path,
    )
    target_model = _model()

    identity = load_redlamp_encoder_checkpoint(
        encoder=target_model.encoder,
        checkpoint_path=checkpoint_path,
    )

    assert identity.checkpoint_role == "pretrained_encoder"
    assert identity.epoch == 100
    assert identity.checkpoint_sha256
    for key, tensor in source_model.encoder.state_dict().items():
        assert torch.equal(target_model.encoder.state_dict()[key], tensor)


def test_loader_rejects_missing_encoder_key(tmp_path: Path) -> None:
    source_model = _model()
    state_dict = {
        key: value
        for key, value in source_model.state_dict().items()
        if not key.startswith("encoder.network.1")
    }
    checkpoint_path = tmp_path / "missing_encoder_key.pt"
    torch.save({"model_state_dict": state_dict}, checkpoint_path)

    with pytest.raises(ValueError, match="encoder keys"):
        load_redlamp_encoder_checkpoint(
            encoder=_model().encoder,
            checkpoint_path=checkpoint_path,
        )
