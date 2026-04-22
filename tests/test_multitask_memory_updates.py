from __future__ import annotations

import torch

from src.models.thesis_multitask import ThesisMultitaskModel


def _build_batch() -> dict[str, object]:
    return {
        "x": torch.randn(2, 100, 38),
        "point_labels": torch.zeros(2, 100, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }


def _build_initialized_model() -> ThesisMultitaskModel:
    model = ThesisMultitaskModel(
        input_dim=38,
        encoder_dim=64,
        hidden_dim=16,
        num_classes=2,
        dropout=0.0,
        continuous_enabled=True,
        continuous_num_prototypes=4,
        discrete_enabled=True,
        discrete_codebook_size=8,
        gumbel_temperature=1.0,
        temperature_start=1.0,
        temperature_end=1.0,
        temperature_anneal_fraction=1.0,
        alpha_logit_init=0.0,
        beta_logit_init=0.0,
        lambda_cls=1.0,
        lambda_div=0.0,
        lambda_var=0.0,
        lambda_cov=0.0,
        lambda_use=0.0,
        lambda_gate=0.0,
        bootstrap_encoder_epochs=1,
        use_synthetic_augmentation=False,
        use_synthetic_validation=False,
        anomaly_probability=0.5,
        min_segment_fraction=0.1,
        max_segment_fraction=0.2,
        spike_scale=3.0,
    )
    initialization_batch = _build_batch()
    model.set_epoch_context(epoch_index=1, total_epochs=2)
    model.maybe_initialize_memories_from_loader([initialization_batch], device="cpu")
    model.set_epoch_context(epoch_index=1, total_epochs=2)
    return model


def test_training_step_updates_continuous_memory_bank() -> None:
    model = _build_initialized_model()
    continuous_before = model.continuous_prototype_bank.clone()

    model.training_step(_build_batch())

    assert not torch.allclose(model.continuous_prototype_bank, continuous_before)


def test_validation_step_does_not_update_continuous_memory_bank() -> None:
    model = _build_initialized_model()
    continuous_before = model.continuous_prototype_bank.clone()

    model.validation_step(_build_batch())

    assert torch.allclose(model.continuous_prototype_bank, continuous_before)


def test_continuous_memory_bank_rows_keep_controlled_norm() -> None:
    model = _build_initialized_model()

    model.training_step(_build_batch())
    row_norms = model.continuous_prototype_bank.norm(dim=-1)

    assert torch.all(row_norms > 0.0)
    assert torch.max(row_norms) - torch.min(row_norms) < 1.0e-3
