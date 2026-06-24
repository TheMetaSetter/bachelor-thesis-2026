from __future__ import annotations

import torch

from src.data.augment import REDLAMP_MULTICLASS_CLASS_NAMES
from src.models.thesis_multitask import ThesisMultitaskModel


def test_multitask_model_defaults_to_redlamp_multiclass_configuration() -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        window_size=20,
        encoder_dim=64,
        hidden_dim=16,
    )

    assert model.num_classes == len(REDLAMP_MULTICLASS_CLASS_NAMES)
    assert model.classification_label_mode == "redlamp_multiclass"
    assert model.synthetic_anomaly_injector.train_balance_classes is True


def test_multitask_model_returns_documented_shapes() -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        window_size=100,
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
    assert outputs["pooled"].shape == (4, 100 * 16)
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
    assert model.classification_head[0].in_features == 100 * 16


def test_multitask_model_supports_task_specific_concat_projection_fusion_mode() -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        window_size=20,
        encoder_dim=64,
        hidden_dim=16,
        num_classes=12,
        fusion_mode="task_specific_concat_projection",
    )
    batch = {
        "x": torch.randn(2, 20, 38),
        "point_labels": torch.zeros(2, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": f"machine-{index}"} for index in range(2)],
    }

    outputs = model(batch)

    assert outputs["hidden"].shape == (2, 20, 16)
    assert outputs["aux"]["fusion"]["fusion_mode"] == "task_specific_concat_projection"
    assert outputs["aux"]["hidden_reconstruction"].shape == (2, 20, 16)
    assert outputs["aux"]["hidden_classification"].shape == (2, 20, 16)


def test_multitask_model_supports_cosine_topk_discrete_query_mode() -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        window_size=20,
        encoder_dim=64,
        hidden_dim=16,
        num_classes=12,
        discrete_query_mode="cosine_topk",
        discrete_topk=3,
        discrete_query_temperature=0.1,
    )
    batch = {
        "x": torch.randn(2, 20, 38),
        "point_labels": torch.zeros(2, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": f"machine-{index}"} for index in range(2)],
    }

    outputs = model(batch)

    assert outputs["aux"]["discrete_branch"]["aux"]["query_mode"] == "cosine_topk"
    assert outputs["aux"]["discrete_branch"]["aux"]["topk"] == 3
    assert outputs["aux"]["discrete_branch"]["aux"]["query_temperature"] == 0.1


def test_multitask_model_uses_shared_three_layer_mlp_depth() -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        window_size=20,
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
        window_size=20,
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


def test_multitask_model_flattens_hidden_classification_before_classifier() -> None:
    model = ThesisMultitaskModel(
        input_dim=4,
        window_size=3,
        encoder_dim=8,
        hidden_dim=6,
        mlp_num_linear_layers=3,
        num_classes=2,
        dropout=0.0,
        continuous_enabled=True,
        continuous_num_prototypes=2,
        discrete_enabled=True,
        discrete_codebook_size=3,
        gumbel_temperature=1.0,
        use_synthetic_augmentation=False,
        use_synthetic_validation=False,
        anomaly_probability=0.0,
        min_segment_fraction=0.1,
        max_segment_fraction=0.2,
        spike_scale=3.0,
        lambda_cls=1.0,
        lambda_div=0.0,
        lambda_var=0.0,
        lambda_cov=0.0,
        lambda_use=0.0,
        lambda_gate=0.0,
        usage_lambda_start=0.0,
        usage_lambda_end=0.0,
        usage_lambda_schedule_fraction=1.0,
        variance_floor_gamma=1.0,
        gate_barrier_margin=0.25,
        bootstrap_encoder_epochs=0,
        discrete_ema_decay=0.99,
        memory_norm_epsilon=1.0e-6,
        memory_initialization_batches=1,
        memory_initialization_with_synthetic_windows=False,
        classification_label_mode="binary",
    )
    batch = {
        "x": torch.randn(2, 3, 4),
        "point_labels": torch.zeros(2, 3, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "unit-test"}, {"entity_id": "unit-test"}],
    }

    outputs = model(batch, stage_name="test")

    assert outputs["hidden"].shape == (2, 3, 6)
    assert outputs["pooled"].shape == (2, 3 * 6)
    assert torch.allclose(
        outputs["pooled"],
        outputs["aux"]["hidden_classification"].reshape(2, 3 * 6),
    )


def test_multitask_gate_mlps_use_shared_bias_zero_initialization_policy() -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        window_size=20,
        encoder_dim=64,
        hidden_dim=16,
        mlp_num_linear_layers=3,
        num_classes=12,
        dropout=0.0,
    )

    gate_modules = [
        model.continuous_update_gate,
        model.classification_fusion_gate,
        model.reconstruction_fusion_gate,
    ]
    for gate_module in gate_modules:
        gate_linear_layers = [
            layer for layer in gate_module if isinstance(layer, torch.nn.Linear)
        ]
        assert gate_linear_layers
        for linear_layer in gate_linear_layers:
            assert linear_layer.bias is not None
            assert torch.allclose(
                linear_layer.bias.detach(),
                torch.zeros_like(linear_layer.bias.detach()),
            )
