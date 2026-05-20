from __future__ import annotations

import torch

from src.models.thesis_multitask import ThesisMultitaskModel


def test_one_multitask_forward_and_backward_step_runs_with_optional_losses_enabled() -> (
    None
):
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
        classification_label_mode="binary",
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
        classification_label_mode="binary",
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


def test_usage_loss_schedule_weight_contributes_to_total_loss() -> None:
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
        lambda_cls=1.0,
        enable_diversity_loss=False,
        enable_variance_loss=False,
        enable_covariance_loss=False,
        enable_usage_loss=True,
        enable_gate_loss=False,
        lambda_div=0.0,
        lambda_var=0.0,
        lambda_cov=0.0,
        lambda_use=0.0,
        usage_lambda_start=0.2,
        usage_lambda_end=0.2,
        usage_lambda_schedule_fraction=1.0,
        lambda_gate=0.0,
        use_synthetic_augmentation=True,
        anomaly_probability=1.0,
        min_segment_fraction=0.1,
        max_segment_fraction=0.2,
        spike_scale=3.0,
        classification_label_mode="binary",
    )
    model.set_epoch_context(epoch_index=0, total_epochs=1)
    batch = {
        "x": torch.randn(2, 100, 38),
        "point_labels": torch.zeros(2, 100, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }

    step_output = model.training_step(batch)
    expected_loss = (
        step_output["loss_terms"]["reconstruction_loss"]
        + step_output["loss_terms"]["classification_loss"]
        + 0.2 * step_output["loss_terms"]["usage_loss"]
    )

    assert step_output["loss_terms"]["usage_loss"].item() > 0.0
    assert torch.isclose(step_output["loss"], expected_loss)


def test_one_multitask_train_step_with_exp2_logs_contrastive_and_gate_stats() -> None:
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
        lambda_cls=1.0,
        lambda_contrastive=1.0,
        enable_two_view_contrastive=True,
        enable_cka_gated_fusion=True,
        use_synthetic_augmentation=True,
        anomaly_probability=1.0,
        min_segment_fraction=0.1,
        max_segment_fraction=0.2,
        spike_scale=3.0,
        classification_label_mode="binary",
    )
    batch = {
        "x": torch.randn(2, 20, 38),
        "point_labels": torch.zeros(2, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }

    step_output = model.training_step(batch)

    assert "contrastive_loss" in step_output["loss_terms"]
    assert step_output["log"]["train_contrastive_loss"] >= 0.0
    assert "train_alpha_std" in step_output["log"]
    assert "train_beta_std" in step_output["log"]
    assert "train_cka_reconstruction_mean" in step_output["log"]
    assert "train_cka_reconstruction_std" in step_output["log"]
    assert "train_cka_classification_mean" in step_output["log"]
    assert "train_cka_classification_std" in step_output["log"]


def test_exp2_synthetic_validation_step_logs_cka_metrics() -> None:
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
        lambda_cls=1.0,
        lambda_contrastive=1.0,
        enable_two_view_contrastive=True,
        enable_cka_gated_fusion=True,
        use_synthetic_validation=True,
        anomaly_probability=1.0,
        min_segment_fraction=0.1,
        max_segment_fraction=0.2,
        spike_scale=3.0,
        classification_label_mode="binary",
    )
    batch = {
        "x": torch.randn(2, 20, 38),
        "point_labels": torch.zeros(2, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }

    step_output = model.synthetic_validation_step(batch)

    assert "val_synth_cka_reconstruction_mean" in step_output["log"]
    assert "val_synth_cka_reconstruction_std" in step_output["log"]
    assert "val_synth_cka_classification_mean" in step_output["log"]
    assert "val_synth_cka_classification_std" in step_output["log"]


def test_exp2_cka_log_keys_exist_with_zero_fallback_when_gate_disabled() -> None:
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
        lambda_cls=1.0,
        enable_two_view_contrastive=True,
        enable_cka_gated_fusion=False,
        use_synthetic_augmentation=True,
        anomaly_probability=1.0,
        min_segment_fraction=0.1,
        max_segment_fraction=0.2,
        spike_scale=3.0,
        classification_label_mode="binary",
    )
    batch = {
        "x": torch.randn(2, 20, 38),
        "point_labels": torch.zeros(2, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }

    step_output = model.training_step(batch)

    assert step_output["log"]["train_cka_reconstruction_mean"] == 0.0
    assert step_output["log"]["train_cka_reconstruction_std"] == 0.0
    assert step_output["log"]["train_cka_classification_mean"] == 0.0
    assert step_output["log"]["train_cka_classification_std"] == 0.0
