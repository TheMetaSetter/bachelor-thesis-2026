from __future__ import annotations

import torch

from src.models.thesis_multitask import ThesisMultitaskModel


def _build_cnn_model() -> ThesisMultitaskModel:
    return ThesisMultitaskModel(
        input_dim=4,
        window_size=20,
        encoder_dim=8,
        hidden_dim=16,
        mlp_num_linear_layers=3,
        num_classes=2,
        dropout=0.0,
        encoder_family="cnn_simple",
        cnn_num_layers=2,
        cnn_kernel_size=3,
        cnn_hidden_channels=8,
        continuous_enabled=True,
        continuous_num_prototypes=2,
        discrete_enabled=True,
        discrete_codebook_size=4,
        gumbel_temperature=1.0,
        temperature_start=1.0,
        temperature_end=1.0,
        temperature_anneal_fraction=1.0,
        temperature_hold_fraction=0.0,
        alpha_logit_init=0.0,
        beta_logit_init=0.0,
        use_label_refurbishment=False,
        refurbishment_alpha=0.0,
        refurbishment_beta=0.0,
        reconstruction_normal_only=False,
        lambda_cls=1.0,
        lambda_div=0.0,
        lambda_var=0.0,
        lambda_cov=0.0,
        lambda_use=0.0,
        lambda_gate=0.0,
        bootstrap_encoder_epochs=0,
        discrete_ema_decay=0.99,
        memory_norm_epsilon=1.0e-6,
        memory_initialization_batches=2,
        memory_initialization_with_synthetic_windows=True,
        use_synthetic_augmentation=False,
        use_synthetic_validation=True,
        synthetic_validation_seed=7,
        freeze_fusion_for_epochs=0,
        warmup_alpha_value=0.5,
        warmup_beta_value=0.5,
        anomaly_probability=1.0,
        min_segment_fraction=0.1,
        max_segment_fraction=0.2,
        spike_scale=3.0,
        train_balance_classes=False,
        anomaly_families=["spike", "noise"],
    )


def test_thesis_multitask_cnn_encoder_and_forward_shapes() -> None:
    model = _build_cnn_model()
    batch = {
        "x": torch.randn(2, 20, 4),
        "point_labels": torch.zeros(2, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "unit-test"}, {"entity_id": "unit-test"}],
    }

    encoder_outputs = model.encoder({"x": batch["x"]})
    outputs = model(batch)

    assert encoder_outputs["hidden"].shape == (2, 20, 16)
    assert encoder_outputs["pooled"].shape == (2, 16)
    assert outputs["hidden"].shape == (2, 20, 16)
    assert outputs["recon"].shape == (2, 20, 4)
    assert outputs["logits"].shape == (2, 2)
    assert outputs["point_scores"].shape == (2, 20)
    assert outputs["window_scores"].shape == (2,)
