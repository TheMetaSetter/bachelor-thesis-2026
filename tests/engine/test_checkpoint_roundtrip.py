from __future__ import annotations

from pathlib import Path

import torch

from src.engine.checkpoint import CheckpointManager
from src.models.reconstruction_mlp_ae import ReconstructionMLPAutoencoder


def test_checkpoint_metadata_includes_provenance_hashes(tmp_path: Path) -> None:
    # Ở đây mình kiểm tra đúng lớp metadata mà batch 9 yêu cầu:
    # hash của resolved config và hash của extra_state phải đi theo checkpoint.
    model = ReconstructionMLPAutoencoder(
        input_dim=38, encoder_dim=64, hidden_dim=16, dropout=0.0
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_manager = CheckpointManager(tmp_path)
    config = {"experiment_name": "engine-checkpoint-provenance", "seed": 7}
    extra_state = {"online_variant": "A0", "stream_cursor": 3}

    checkpoint_path = checkpoint_manager.save_checkpoint(
        checkpoint_name="checkpoint.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler_state={"feature_mean": torch.zeros(38), "feature_std": torch.ones(38)},
        config=config,
        epoch=2,
        metric_history=[{"val_loss": 1.0}],
        extra_state=extra_state,
    )
    loaded = checkpoint_manager.load_checkpoint(checkpoint_path, model, optimizer)

    assert (
        loaded["checkpoint_metadata"]["resolved_config_sha256"]
        == (loaded["checkpoint_metadata"]["config_sha256"])
    )
    assert loaded["checkpoint_metadata"]["extra_state_sha256"] is not None
    assert loaded["extra_state"] == extra_state
