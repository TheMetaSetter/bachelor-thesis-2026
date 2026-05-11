from __future__ import annotations

import torch

from src.models.thesis_multitask import ThesisMultitaskModel


def test_multitask_model_returns_documented_shapes() -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        encoder_dim=64,
        hidden_dim=16,
        mlp_num_linear_layers=3,
        num_classes=2,
        dropout=0.0,
        continuous_enabled=True,
        continuous_num_prototypes=4,
        discrete_enabled=True,
        discrete_codebook_size=8,
        gumbel_temperature=1.5,
        alpha_logit_init=0.0,
        beta_logit_init=0.0,
    )
    batch = {
        "x": torch.randn(4, 100, 38),
        "point_labels": torch.zeros(4, 100, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": f"machine-{index}"} for index in range(4)],
    }

    outputs = model(batch)

    assert outputs["hidden"].shape == (4, 100, 16)
    assert outputs["recon"].shape == (4, 100, 38)
    assert outputs["logits"].shape == (4, 2)
    assert outputs["aux"]["class_probabilities"].shape == (4, 2)
    assert torch.allclose(
        outputs["aux"]["class_probabilities"].sum(dim=-1),
        torch.ones(4),
        atol=1e-6,
    )
    assert outputs["point_scores"].shape == (4, 100)
    assert outputs["window_scores"].shape == (4,)
    assert "continuous_branch" in outputs["aux"]
    assert "discrete_branch" in outputs["aux"]
    assert "fusion" in outputs["aux"]
    assert "alpha" in outputs["aux"]
    assert "beta" in outputs["aux"]
    assert outputs["aux"]["fusion"]["fusion_mode"] == "learnable_sigmoid_scalars"
    assert outputs["aux"]["fusion"]["fusion_mode"] not in {"identity", "average"}


def test_multitask_model_uses_shared_three_layer_mlp_depth() -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        encoder_dim=64,
        hidden_dim=16,
        mlp_num_linear_layers=3,
        num_classes=2,
        dropout=0.0,
        continuous_enabled=True,
        continuous_num_prototypes=4,
        discrete_enabled=True,
        discrete_codebook_size=8,
        gumbel_temperature=1.5,
        alpha_logit_init=0.0,
        beta_logit_init=0.0,
    )

    encoder_linear_layers = [
        layer for layer in model.encoder.network if isinstance(layer, torch.nn.Linear)
    ]
    reconstruction_linear_layers = [
        layer
        for layer in model.reconstruction_head
        if isinstance(layer, torch.nn.Linear)
    ]
    classification_linear_layers = [
        layer
        for layer in model.classification_head
        if isinstance(layer, torch.nn.Linear)
    ]

    assert len(encoder_linear_layers) == 3
    assert len(reconstruction_linear_layers) == 3
    assert len(classification_linear_layers) == 3


def test_multitask_model_supports_redlamp_multiclass_logits_and_probabilities() -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        encoder_dim=64,
        hidden_dim=16,
        mlp_num_linear_layers=3,
        num_classes=12,
        dropout=0.0,
        continuous_enabled=True,
        continuous_num_prototypes=4,
        discrete_enabled=True,
        discrete_codebook_size=8,
        gumbel_temperature=1.5,
        alpha_logit_init=0.0,
        beta_logit_init=0.0,
        use_label_refurbishment=True,
        refurbishment_alpha=0.1,
        refurbishment_beta=0.01,
        classification_label_mode="redlamp_multiclass",
    )
    batch = {
        "x": torch.randn(3, 20, 38),
        "point_labels": torch.zeros(3, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": f"machine-{index}"} for index in range(3)],
    }

    outputs = model(batch)
    refurbished_targets = model._build_refurbished_classification_targets(
        torch.tensor([0, 1, 11]),
        outputs["logits"].dtype,
    )

    assert outputs["logits"].shape == (3, 12)
    assert outputs["aux"]["class_probabilities"].shape == (3, 12)
    assert torch.allclose(
        outputs["aux"]["class_probabilities"].sum(dim=-1),
        torch.ones(3),
        atol=1e-6,
    )
    assert refurbished_targets.shape == (3, 12)
    assert torch.allclose(refurbished_targets.sum(dim=-1), torch.ones(3), atol=1e-6)
