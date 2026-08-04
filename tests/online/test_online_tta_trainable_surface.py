from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.engine.checkpoint import CheckpointManager
from src.engine.online_tta.online_optimizer import (
    assert_only_projector_is_trainable,
    collect_projector_parameters,
)
from src.models.online_adaptation import OnlineAdaptationModel
from src.models.thesis_multitask import ThesisMultitaskModel


def _build_reference_checkpoint(tmp_path: Path) -> Path:
    model = ThesisMultitaskModel(
        input_dim=38,
        window_size=100,
        encoder_dim=64,
        hidden_dim=32,
        num_classes=2,
        dropout=0.0,
        use_synthetic_augmentation=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_manager = CheckpointManager(tmp_path)
    return checkpoint_manager.save_checkpoint(
        checkpoint_name="reference.pt",
        model=model,
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
        },
        epoch=1,
        metric_history=[],
    )


def test_online_projector_uses_near_identity_residual_init(tmp_path: Path) -> None:
    checkpoint_path = _build_reference_checkpoint(tmp_path)
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
        reference_checkpoint_path=str(checkpoint_path),
        warm_start_projector=False,
        target_param_group="projector_params",
        clean_stream_only=True,
    )

    assert hasattr(model, "online_mlp_projector")
    assert model.projector is model.online_mlp_projector
    assert float(model.online_mlp_projector.alpha.detach().cpu()) == pytest.approx(
        1.0e-3
    )

    projector_parameters = collect_projector_parameters(model)
    expected_parameters = list(model.online_mlp_projector.parameters())

    assert len(projector_parameters) == len(expected_parameters)
    assert [id(parameter) for parameter in projector_parameters] == [
        id(parameter) for parameter in expected_parameters
    ]
    assert_only_projector_is_trainable(model)


def test_online_projector_forward_shape_is_preserved(tmp_path: Path) -> None:
    checkpoint_path = _build_reference_checkpoint(tmp_path)
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
        reference_checkpoint_path=str(checkpoint_path),
        warm_start_projector=False,
        target_param_group="projector_params",
        clean_stream_only=True,
    )

    z = torch.randn(2, 5, 32)
    projected = model.online_mlp_projector(z)

    assert projected.shape == z.shape
    assert torch.max(torch.abs(projected - z)).item() < 1.0e-1


def test_a0_has_no_projector_or_trainable_online_parameters(tmp_path: Path) -> None:
    checkpoint_path = _build_reference_checkpoint(tmp_path)
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
        reference_checkpoint_path=str(checkpoint_path),
        warm_start_projector=False,
        target_param_group="projector_params",
        clean_stream_only=True,
        online_variant="A0",
    )

    assert not hasattr(model, "online_mlp_projector")
    assert not any(parameter.requires_grad for parameter in model.parameters())
