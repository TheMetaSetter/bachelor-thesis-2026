from __future__ import annotations

import torch

from src.models.thesis_multitask import ThesisMultitaskModel


def test_one_multitask_forward_and_backward_step_runs_with_optional_losses_enabled() -> None:
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
        gumbel_temperature=1.5,
        alpha_logit_init=0.0,
        beta_logit_init=0.0,
        lambda_cls=1.0,
        enable_diversity_loss=True,
        enable_variance_loss=True,
        enable_covariance_loss=True,
        enable_usage_loss=True,
        enable_gate_loss=True,
        lambda_div=0.01,
        lambda_var=0.01,
        lambda_cov=0.01,
        lambda_use=0.01,
        lambda_gate=0.01,
        use_synthetic_augmentation=True,
        anomaly_probability=1.0,
        min_segment_fraction=0.1,
        max_segment_fraction=0.2,
        spike_scale=3.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = {
        "x": torch.randn(2, 100, 38),
        "point_labels": torch.zeros(2, 100, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }

    step_output = model.training_step(batch)
    optimizer.zero_grad()
    step_output["loss"].backward()
    optimizer.step()

    assert step_output["loss"].item() >= 0.0
    assert step_output["loss_terms"]["classification_loss"].item() >= 0.0
    assert step_output["batch"]["classification_labels"].sum().item() == 2
    assert model.alpha_logit.grad is not None
    assert model.beta_logit.grad is not None


def test_one_multitask_train_step_runs_with_optional_losses_disabled() -> None:
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
        gumbel_temperature=1.5,
        alpha_logit_init=0.0,
        beta_logit_init=0.0,
        lambda_cls=1.0,
        enable_diversity_loss=False,
        enable_variance_loss=False,
        enable_covariance_loss=False,
        enable_usage_loss=False,
        enable_gate_loss=False,
        lambda_div=0.0,
        lambda_var=0.0,
        lambda_cov=0.0,
        lambda_use=0.0,
        lambda_gate=0.0,
        use_synthetic_augmentation=True,
        anomaly_probability=1.0,
        min_segment_fraction=0.1,
        max_segment_fraction=0.2,
        spike_scale=3.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = {
        "x": torch.randn(2, 100, 38),
        "point_labels": torch.zeros(2, 100, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }

    step_output = model.training_step(batch)
    optimizer.zero_grad()
    step_output["loss"].backward()
    optimizer.step()

    expected_baseline_loss = (
        step_output["loss_terms"]["reconstruction_loss"]
        + step_output["loss_terms"]["classification_loss"]
    )

    assert step_output["loss_terms"]["diversity_loss"].item() == 0.0
    assert step_output["loss_terms"]["variance_loss"].item() == 0.0
    assert step_output["loss_terms"]["covariance_loss"].item() == 0.0
    assert step_output["loss_terms"]["usage_loss"].item() == 0.0
    assert step_output["loss_terms"]["gate_loss"].item() == 0.0
    assert torch.isclose(step_output["loss"], expected_baseline_loss)
    assert model.alpha_logit.grad is not None
    assert model.beta_logit.grad is not None
