from __future__ import annotations

from pathlib import Path

import torch
import pytest

from src.engine.checkpoint import CheckpointManager
from src.models.online_adaptation import OnlineAdaptationModel
from src.models.reconstruction_mlp_ae import ReconstructionMLPAutoencoder
from src.models.thesis_multitask import ThesisMultitaskModel


def test_online_model_rejects_reconstruction_reference_checkpoint(
    tmp_path: Path,
) -> None:
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
        scheduler=None,
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


def test_online_model_accepts_multitask_reference_checkpoint_with_memory_extra_state(
    tmp_path: Path,
) -> None:
    reference_model = ThesisMultitaskModel(
        input_dim=38,
        window_size=100,
        encoder_dim=64,
        hidden_dim=32,
        num_classes=2,
        dropout=0.0,
        bootstrap_encoder_epochs=0,
        use_synthetic_augmentation=False,
    )
    reference_model.mark_memories_initialized(initialization_epoch=3)
    optimizer = torch.optim.Adam(reference_model.parameters(), lr=1e-3)
    checkpoint_manager = CheckpointManager(tmp_path)
    checkpoint_path = checkpoint_manager.save_checkpoint(
        checkpoint_name="multitask_memory.pt",
        model=reference_model,
        optimizer=optimizer,
        scheduler=None,
        scaler_state={"feature_mean": torch.zeros(38), "feature_std": torch.ones(38)},
        config={
            "model": {
                "model_name": "thesis_multitask",
                "input_dim": 38,
                "window_size": 100,
                "encoder_dim": 64,
                "hidden_dim": 32,
                "num_classes": 2,
                "dropout": 0.0,
                "continuous_enabled": True,
                "continuous_num_prototypes": 8,
                "discrete_enabled": True,
                "discrete_codebook_size": 16,
                "gumbel_temperature": 1.0,
                "temperature_start": 1.0,
                "temperature_end": 1.0,
                "temperature_anneal_fraction": 1.0,
                "alpha_logit_init": 0.0,
                "beta_logit_init": 0.0,
                "lambda_cls": 1.0,
                "lambda_div": 0.0,
                "lambda_var": 0.0,
                "lambda_cov": 0.0,
                "lambda_use": 0.0,
                "lambda_gate": 0.0,
                "bootstrap_encoder_epochs": 0,
            },
            "task": {
                "task_name": "multitask_tsad",
                "use_synthetic_augmentation": False,
                "anomaly_probability": 0.5,
                "min_segment_fraction": 0.1,
                "max_segment_fraction": 0.2,
                "spike_scale": 3.0,
            },
        },
        epoch=3,
        metric_history=[],
        extra_state=reference_model.get_checkpoint_extra_state(),
    )

    online_model = OnlineAdaptationModel(
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

    assert online_model.reference_encoder.model.memory_initialized is True
    assert online_model.reference_encoder.model.memory_training_enabled is True
