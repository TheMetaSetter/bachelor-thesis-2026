from __future__ import annotations

from typing import Any

import pytest
import torch

from src.models.thesis_multitask import ThesisMultitaskModel


def _build_phase_model(
    training_phase: str,
    **overrides: Any,
) -> ThesisMultitaskModel:
    model_kwargs = {
        "input_dim": 4,
        "window_size": 3,
        "encoder_dim": 8,
        "hidden_dim": 6,
        "mlp_num_linear_layers": 3,
        "num_classes": 12,
        "dropout": 0.0,
        "continuous_enabled": True,
        "continuous_num_prototypes": 2,
        "discrete_enabled": True,
        "discrete_codebook_size": 60,
        "gumbel_temperature": 1.0,
        "temperature_start": 1.0,
        "temperature_end": 1.0,
        "temperature_anneal_fraction": 1.0,
        "alpha_logit_init": 0.0,
        "beta_logit_init": 0.0,
        "lambda_recon": 0.9,
        "lambda_cls": 1.1,
        "enable_two_view_contrastive": True,
        "lambda_contrastive": 0.1,
        "lambda_div": 0.0,
        "lambda_var": 0.0,
        "lambda_cov": 0.0,
        "lambda_use": 0.0,
        "lambda_gate": 0.0,
        "bootstrap_encoder_epochs": 0,
        "use_synthetic_augmentation": False,
        "use_synthetic_validation": False,
        "classification_label_mode": "redlamp_multiclass",
        "anomaly_probability": 0.5,
        "min_segment_fraction": 0.1,
        "max_segment_fraction": 0.2,
        "spike_scale": 3.0,
        "fusion_mode": "task_specific_concat_projection",
        "discrete_query_mode": "cosine_topk",
        "discrete_topk": 3,
        "discrete_query_temperature": 0.1,
        "training_phase": training_phase,
        "freeze_memories_after_initialization": True,
        "freeze_recovered_zipped_encoder_during_warmup": True,
        "discrete_memory_label_source": "synthetic_train_labels",
    }
    model_kwargs.update(overrides)
    model = ThesisMultitaskModel(
        **model_kwargs,
    )
    return model


def _build_preaugmented_batch() -> dict[str, Any]:
    return {
        "x": torch.randn(2, 3, 4),
        "point_labels": torch.zeros(2, 3, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
        "classification_labels": torch.tensor([0, 1], dtype=torch.long),
        "classification_class_names": tuple(str(index) for index in range(12)),
        "synthetic_anomaly_mask": torch.tensor(
            [[0, 0, 0], [0, 1, 1]],
            dtype=torch.long,
        ),
        "augmentation_metadata": [
            {"is_synthetic_anomaly": False, "anomaly_family": "clean"},
            {"is_synthetic_anomaly": True, "anomaly_family": "spike"},
        ],
    }


@pytest.mark.parametrize(
    ("training_phase", "expected_bypass", "expected_update"),
    [
        ("stage1_classification", True, False),
        ("stage1_reconstruction", True, False),
        ("stage2_recovery", True, False),
        ("stage3_prototype_warmup", False, False),
        ("multitask_pretraining", False, False),
    ],
)
def test_memory_lifecycle_switches_with_three_stage_phase(
    training_phase: str,
    expected_bypass: bool,
    expected_update: bool,
) -> None:
    model = _build_phase_model(training_phase)
    model.mark_memories_initialized(initialization_epoch=1)

    assert model._should_bypass_memory_for_stage("train") is expected_bypass
    assert model._should_update_memory("train") is expected_update


def test_stage3_warmup_freezes_encoder_but_keeps_heads_and_concat_projections_trainable() -> None:
    model = _build_phase_model("stage3_prototype_warmup")

    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(
        parameter.requires_grad for parameter in model.reconstruction_head.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.classification_head.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.reconstruction_concat_projection.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.classification_concat_projection.parameters()
    )


@pytest.mark.parametrize(
    ("training_phase", "expected_reconstruction_weight", "expected_classification_weight"),
    [
        ("stage1_classification", 0.0, 1.1),
        ("stage1_reconstruction", 0.9, 0.0),
        ("stage2_recovery", 0.9, 1.1),
        ("stage3_prototype_warmup", 0.9, 1.1),
    ],
)
def test_loss_weighting_changes_with_three_stage_phase(
    training_phase: str,
    expected_reconstruction_weight: float,
    expected_classification_weight: float,
) -> None:
    model = _build_phase_model(
        training_phase,
        enable_two_view_contrastive=False,
    )

    step_output = model.training_step(_build_preaugmented_batch())

    expected_loss = (
        expected_reconstruction_weight * step_output["loss_terms"]["reconstruction_loss"]
        + expected_classification_weight
        * step_output["loss_terms"]["classification_loss"]
    )
    assert torch.allclose(step_output["loss"], expected_loss, atol=1.0e-6)


@pytest.mark.parametrize(
    ("training_phase", "should_build_pair"),
    [
        ("stage1_classification", True),
        ("stage1_reconstruction", True),
        ("stage2_recovery", False),
        ("stage3_prototype_warmup", False),
        ("multitask_pretraining", True),
    ],
)
def test_contrastive_pairing_is_phase_aware(
    training_phase: str,
    should_build_pair: bool,
) -> None:
    model = _build_phase_model(training_phase)
    contrastive_pair = model._prepare_contrastive_pair_batches(
        _build_preaugmented_batch(),
        stage_name="train",
    )

    assert (contrastive_pair is not None) is should_build_pair
