from __future__ import annotations

from pathlib import Path

import torch
import pytest

from src.engine.checkpoint import CheckpointManager
from src.models.online_adaptation import OnlineAdaptationModel
from src.models.reconstruction_mlp_ae import ReconstructionMLPAutoencoder


def test_online_model_rejects_reconstruction_reference_checkpoint(tmp_path: Path) -> None:
    model = ReconstructionMLPAutoencoder(
        input_dim=38,
        encoder_dim=64,
        hidden_dim=32,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_manager = CheckpointManager(tmp_path)
    checkpoint_path = checkpoint_manager.save_checkpoint(
        checkpoint_name="baseline.pt",
        model=model,
        optimizer=optimizer,
        scaler_state={"feature_mean": torch.zeros(38), "feature_std": torch.ones(38)},
        config={
            "model": {
                "model_name": "reconstruction_mlp_ae",
                "input_dim": 38,
                "encoder_dim": 64,
                "hidden_dim": 32,
                "dropout": 0.0,
            },
            "task": {"task_name": "reconstruction"},
        },
        epoch=1,
        metric_history=[],
    )

    with pytest.raises(ValueError, match="thesis_multitask checkpoint"):
        OnlineAdaptationModel(
            input_dim=38,
            encoder_dim=64,
            hidden_dim=32,
            projector_hidden_dim=48,
            projector_dropout=0.0,
            enable_prototype_alignment=False,
            lambda_align=1.0,
            lambda_proto=0.1,
            lambda_anchor=0.001,
            score_source="projected_hidden",
            reference_checkpoint_path=str(checkpoint_path),
            warm_start_projector=False,
            target_param_group="projector_params",
            clean_stream_only=True,
        )
