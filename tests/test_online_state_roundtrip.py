from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.data.stream import OnlineWindowBatcher, SMDOnlineStream
from src.engine.checkpoint import CheckpointManager
from src.models.online_adaptation import OnlineAdaptationModel
from src.models.thesis_multitask import ThesisMultitaskModel


def _build_reference_checkpoint(tmp_path: Path) -> Path:
    model = ThesisMultitaskModel(
        input_dim=38,
        encoder_dim=64,
        hidden_dim=32,
        num_classes=2,
        dropout=0.0,
        use_synthetic_augmentation=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_manager = CheckpointManager(tmp_path)
    config = {
        "model": {
            "model_name": "thesis_multitask",
            "input_dim": 38,
            "encoder_dim": 64,
            "hidden_dim": 32,
            "num_classes": 2,
            "dropout": 0.0,
            "continuous_enabled": True,
            "continuous_num_prototypes": 8,
            "discrete_enabled": True,
            "discrete_codebook_size": 16,
            "gumbel_temperature": 1.5,
            "alpha_logit_init": 0.0,
            "beta_logit_init": 0.0,
            "lambda_cls": 1.0,
            "enable_diversity_loss": False,
            "enable_variance_loss": False,
            "enable_covariance_loss": False,
            "enable_usage_loss": False,
            "enable_gate_loss": False,
            "lambda_div": 0.01,
            "lambda_var": 0.01,
            "lambda_cov": 0.01,
            "lambda_use": 0.01,
            "lambda_gate": 0.01,
            "variance_floor_gamma": 1.0,
            "gate_barrier_margin": 0.25,
        },
        "task": {
            "task_name": "multitask_tsad",
            "use_synthetic_augmentation": False,
            "anomaly_probability": 0.5,
            "min_segment_fraction": 0.1,
            "max_segment_fraction": 0.2,
            "spike_scale": 3.0,
        },
    }
    return checkpoint_manager.save_checkpoint(
        checkpoint_name="reference.pt",
        model=model,
        optimizer=optimizer,
        scaler_state={"feature_mean": torch.zeros(38), "feature_std": torch.ones(38)},
        config=config,
        epoch=1,
        metric_history=[],
    )


def _build_sequence(entity_id: str, sequence_length: int = 130, num_channels: int = 38) -> dict[str, Any]:
    return {
        "x": torch.randn(sequence_length, num_channels),
        "point_labels": torch.zeros(sequence_length, dtype=torch.long),
        "mask": torch.ones(sequence_length, num_channels),
        "timestamps": torch.arange(sequence_length),
        "meta": {
            "dataset_name": "smd",
            "entity_id": entity_id,
            "split": "test",
            "num_channels": num_channels,
            "sequence_length": sequence_length,
        },
    }


def test_online_checkpoint_roundtrip_restores_extra_state(tmp_path: Path) -> None:
    reference_checkpoint_path = _build_reference_checkpoint(tmp_path)
    model = OnlineAdaptationModel(
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
        reference_checkpoint_path=str(reference_checkpoint_path),
        warm_start_projector=False,
        target_param_group="projector_params",
        clean_stream_only=True,
    )
    optimizer = torch.optim.Adam(model.get_parameter_group("projector_params"), lr=1e-3)

    stream = SMDOnlineStream(
        sequences=[_build_sequence("machine-1"), _build_sequence("machine-2")],
        window_size=100,
        stride=10,
        clean_stream_only=True,
    )
    batcher = OnlineWindowBatcher(
        stream=stream,
        batch_size=2,
        view_noise_std=0.0,
        view_dropout_probability=0.0,
    )
    _ = batcher.next_batch()

    checkpoint_manager = CheckpointManager(tmp_path / "online")
    checkpoint_path = checkpoint_manager.save_checkpoint(
        checkpoint_name="online_roundtrip.pt",
        model=model,
        optimizer=optimizer,
        scaler_state={"feature_mean": torch.zeros(38), "feature_std": torch.ones(38)},
        config={"experiment_name": "online-roundtrip"},
        epoch=2,
        metric_history=[{"online/alignment_loss": 1.0}],
        extra_state={
            "stream_state_dict": batcher.state_dict(),
            "projector_anchor_state_dict": model.get_projector_anchor_state_dict(),
            "target_param_group": "projector_params",
            "online_metric_history": [{"online/alignment_loss": 1.0}],
            "reset_policy_state": {"reset_policy": "disabled", "reset_alignment_threshold": 0.0},
        },
    )

    reloaded_model = OnlineAdaptationModel(
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
        reference_checkpoint_path=str(reference_checkpoint_path),
        warm_start_projector=False,
        target_param_group="projector_params",
        clean_stream_only=True,
    )
    reloaded_optimizer = torch.optim.Adam(reloaded_model.get_parameter_group("projector_params"), lr=1e-3)
    loaded_checkpoint = checkpoint_manager.load_checkpoint(
        checkpoint_path,
        reloaded_model,
        reloaded_optimizer,
    )
    reloaded_model.load_projector_anchor_state_dict(
        loaded_checkpoint["extra_state"]["projector_anchor_state_dict"]
    )

    assert loaded_checkpoint["extra_state"]["target_param_group"] == "projector_params"
    assert "stream_state_dict" in loaded_checkpoint["extra_state"]
    assert loaded_checkpoint["extra_state"]["online_metric_history"][0]["online/alignment_loss"] == 1.0
    assert reloaded_model.get_projector_anchor_state_dict().keys() == model.get_projector_anchor_state_dict().keys()
