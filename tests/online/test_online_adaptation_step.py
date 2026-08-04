from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.engine.checkpoint import CheckpointManager
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
    config = {
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
    }
    return checkpoint_manager.save_checkpoint(
        checkpoint_name="reference.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler_state={"feature_mean": torch.zeros(38), "feature_std": torch.ones(38)},
        config=config,
        epoch=1,
        metric_history=[],
    )


def _build_online_batch(batch_size: int = 2) -> dict[str, Any]:
    window_size = 100
    num_channels = 38
    x_tensor = torch.randn(batch_size, window_size, num_channels)
    return {
        "x": x_tensor,
        "absolute_indices": torch.arange(window_size).repeat(batch_size, 1),
        "view_a": x_tensor.clone(),
        "view_b": x_tensor.clone() + 0.01,
        "point_labels": torch.zeros(batch_size, window_size, dtype=torch.long),
        "mask": torch.ones(batch_size, window_size, num_channels),
        "timestamps": torch.arange(window_size).repeat(batch_size, 1),
        "meta": [
            {
                "dataset_name": "smd",
                "entity_id": f"machine-{batch_index}",
                "split": "test",
                "start_index": batch_index * 10,
                "end_index": batch_index * 10 + window_size,
                "window_size": window_size,
                "stream_step": batch_index,
            }
            for batch_index in range(batch_size)
        ],
    }


def test_online_adaptation_step_updates_only_projector_parameters(
    tmp_path: Path,
) -> None:
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
    optimizer = torch.optim.Adam(model.get_parameter_group("projector_params"), lr=1e-2)
    batch = _build_online_batch()

    reference_before = [
        parameter.detach().clone() for parameter in model.reference_encoder.parameters()
    ]
    online_encoder_before = [
        parameter.detach().clone()
        for parameter in model.online_encoder.encoder_parameters()
    ]
    projector_before = [
        parameter.detach().clone() for parameter in model.projector.parameters()
    ]

    step_output = model.training_step(batch)
    optimizer.zero_grad()
    step_output["loss"].backward()
    optimizer.step()

    assert step_output["outputs"]["hidden"].shape == (2, 100, 32)
    assert step_output["outputs"]["window_scores"].shape == (2,)
    assert any(
        not torch.allclose(before_parameter, after_parameter)
        for before_parameter, after_parameter in zip(
            projector_before, model.projector.parameters()
        )
    )
    assert all(
        torch.allclose(before_parameter, after_parameter)
        for before_parameter, after_parameter in zip(
            reference_before, model.reference_encoder.parameters()
        )
    )
    assert all(
        torch.allclose(before_parameter, after_parameter)
        for before_parameter, after_parameter in zip(
            online_encoder_before,
            model.online_encoder.encoder_parameters(),
        )
    )


def test_online_adapter_exposes_frozen_prototype_metadata(
    tmp_path: Path,
) -> None:
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
    reference_model = model.reference_encoder.model
    reference_model.verification_metadata_source = "pytest_training_memory"
    reference_model.verification_metadata_split = "synthetic_train"
    reference_model.verification_metadata_schema_version = 1
    reference_model.verification_metadata_initialization_seed = 7
    reference_model.verification_codeword_class_ids = torch.zeros(
        reference_model.discrete_codebook.shape[0], dtype=torch.long
    )
    reference_model.verification_contributing_token_counts = torch.zeros(
        reference_model.discrete_codebook.shape[0], dtype=torch.float32
    )
    reference_model.anomalous_codeword_mask = torch.zeros(
        reference_model.discrete_codebook.shape[0], dtype=torch.bool
    )
    reference_model.anomaly_radii = torch.ones(
        reference_model.discrete_codebook.shape[0], dtype=torch.float32
    )
    metadata = model.reference_encoder.prototype_verification_metadata()
    assert metadata.schema_version == 1
    assert metadata.source_split == "synthetic_train"
    assert metadata.codebook.shape[0] == metadata.anomalous_codeword_mask.shape[0]
