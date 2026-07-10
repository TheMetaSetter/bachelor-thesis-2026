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


def test_continuous_only_fusion_is_exact_limiting_case() -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        window_size=100,
        encoder_dim=64,
        hidden_dim=16,
        use_synthetic_augmentation=False,
        freeze_fusion_for_epochs=1,
        warmup_alpha_value=0.0,
        warmup_beta_value=0.0,
    )
    model.set_epoch_context(epoch_index=0, total_epochs=1)
    outputs = model.forward(_build_batch())

    assert torch.allclose(
        outputs["aux"]["hidden_reconstruction"],
        outputs["aux"]["continuous_branch"]["prototype_context"],
    )
    assert torch.allclose(
        outputs["aux"]["hidden_classification"],
        outputs["aux"]["continuous_branch"]["prototype_context"],
    )


def test_discrete_only_fusion_is_exact_limiting_case() -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        window_size=100,
        encoder_dim=64,
        hidden_dim=16,
        use_synthetic_augmentation=False,
        freeze_fusion_for_epochs=1,
        warmup_alpha_value=1.0,
        warmup_beta_value=1.0,
    )
    model.set_epoch_context(epoch_index=0, total_epochs=1)
    outputs = model.forward(_build_batch())

    assert torch.allclose(
        outputs["aux"]["hidden_reconstruction"],
        outputs["aux"]["discrete_branch"]["quantized_hidden"],
    )
    assert torch.allclose(
        outputs["aux"]["hidden_classification"],
        outputs["aux"]["discrete_branch"]["quantized_hidden"],
    )
