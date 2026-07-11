from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.engine.checkpoint import CheckpointManager
from src.engine.online_tta.online_engine import execute_online_tta_step
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


def _build_batch(batch_size: int = 1) -> dict[str, Any]:
    window_size = 100
    num_channels = 38
    x_tensor = torch.randn(batch_size, window_size, num_channels)
    return {
        "x": x_tensor,
        "view_a": x_tensor.clone(),
        "view_b": x_tensor.clone(),
        "point_labels": torch.zeros(batch_size, window_size, dtype=torch.long),
        "mask": torch.ones(batch_size, window_size, num_channels),
        "timestamps": torch.arange(window_size).repeat(batch_size, 1),
        "meta": [
            {
                "dataset_name": "smd",
                "entity_id": "machine-1-6",
                "split": "test",
                "start_index": 0,
                "end_index": window_size,
                "window_size": window_size,
                "stream_step": 0,
            }
            for _ in range(batch_size)
        ],
    }


def _build_model(tmp_path: Path) -> OnlineAdaptationModel:
    checkpoint_path = _build_reference_checkpoint(tmp_path)
    model = OnlineAdaptationModel(
        input_dim=38,
        encoder_dim=64,
        hidden_dim=32,
        projector_hidden_dim=48,
        projector_dropout=0.0,
        enable_prototype_alignment=True,
        lambda_align=1.0,
        lambda_proto=0.1,
        lambda_anchor=0.001,
        score_source="projected_hidden",
        reference_checkpoint_path=str(checkpoint_path),
        warm_start_projector=False,
        target_param_group="projector_params",
        clean_stream_only=True,
    )
    reference = model.reference_encoder.model
    reference.verification_metadata_source = "pytest_training_memory"
    reference.anomalous_codeword_mask = torch.zeros(
        reference.discrete_codebook.shape[0], dtype=torch.bool
    )
    reference.anomalous_codeword_mask[reference.discrete_codebook.shape[0] // 2 :] = True
    reference.anomaly_radii = torch.ones(reference.discrete_codebook.shape[0])
    return model


def test_execute_online_tta_step_updates_only_projector_for_a1_and_a2(
    tmp_path: Path,
) -> None:
    model_a1 = _build_model(tmp_path)
    optimizer_a1 = torch.optim.Adam(
        model_a1.get_parameter_group("projector_params"), lr=1e-2
    )
    batch = _build_batch()
    projector_before = [
        parameter.detach().clone()
        for parameter in model_a1.online_mlp_projector.parameters()
    ]

    a1_record = execute_online_tta_step(
        model=model_a1,
        optimizer=optimizer_a1,
        batch=batch,
        online_variant="A1",
        threshold_value=0.0,
        triage_decision="pnn_verified",
    )

    assert a1_record["did_update"] is True
    assert a1_record["online_variant"] == "A1"
    assert a1_record["record"]["reconstruction_loss"] >= 0.0
    assert a1_record["record"]["projector_grad_norm"] >= 0.0
    assert any(
        not torch.allclose(before, after)
        for before, after in zip(
            projector_before, model_a1.online_mlp_projector.parameters()
        )
    )

    model_a2 = _build_model(tmp_path)
    optimizer_a2 = torch.optim.Adam(
        model_a2.get_parameter_group("projector_params"), lr=1e-2
    )
    projector_before_a2 = [
        parameter.detach().clone()
        for parameter in model_a2.online_mlp_projector.parameters()
    ]

    a2_record = execute_online_tta_step(
        model=model_a2,
        optimizer=optimizer_a2,
        batch=batch,
        online_variant="A2",
        threshold_value=0.0,
        triage_decision="hard_old_normality",
    )

    assert a2_record["did_update"] is True
    assert a2_record["online_variant"] == "A2"
    assert a2_record["record"]["contrastive_loss"] >= 0.0
    assert any(
        not torch.allclose(before, after)
        for before, after in zip(
            projector_before_a2, model_a2.online_mlp_projector.parameters()
        )
    )


def test_execute_online_tta_step_keeps_a0_frozen(tmp_path: Path) -> None:
    model = _build_model(tmp_path)
    optimizer = torch.optim.Adam(model.get_parameter_group("projector_params"), lr=1e-2)
    projector_before = [
        parameter.detach().clone()
        for parameter in model.online_mlp_projector.parameters()
    ]

    record = execute_online_tta_step(
        model=model,
        optimizer=optimizer,
        batch=_build_batch(),
        online_variant="A0",
        threshold_value=0.0,
        triage_decision=None,
    )

    assert record["did_update"] is False
    assert all(
        torch.allclose(before, after)
        for before, after in zip(
            projector_before, model.online_mlp_projector.parameters()
        )
    )
