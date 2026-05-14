from __future__ import annotations

import torch

from src.models.thesis_multitask import ThesisMultitaskModel


def _build_model(**overrides: object) -> ThesisMultitaskModel:
    model_kwargs: dict[str, object] = {
        "input_dim": 38,
        "window_size": 100,
        "encoder_dim": 64,
        "hidden_dim": 16,
        "mlp_num_linear_layers": 3,
        "num_classes": 2,
        "dropout": 0.0,
        "continuous_enabled": True,
        "continuous_num_prototypes": 4,
        "discrete_enabled": True,
        "discrete_codebook_size": 8,
        "gumbel_temperature": 1.5,
        "alpha_logit_init": 0.0,
        "beta_logit_init": 0.0,
        "use_label_refurbishment": False,
        "refurbishment_alpha": 0.0,
        "refurbishment_beta": 0.0,
        "reconstruction_normal_only": False,
        "lambda_cls": 1.0,
        "enable_diversity_loss": False,
        "enable_variance_loss": False,
        "enable_covariance_loss": False,
        "enable_usage_loss": False,
        "enable_gate_loss": False,
        "lambda_div": 0.0,
        "lambda_var": 0.0,
        "lambda_cov": 0.0,
        "lambda_use": 0.0,
        "lambda_gate": 0.0,
        "use_synthetic_augmentation": False,
        "use_synthetic_validation": True,
        "synthetic_validation_seed": 7,
        "freeze_fusion_for_epochs": 0,
        "warmup_alpha_value": 0.5,
        "warmup_beta_value": 0.5,
        "anomaly_probability": 1.0,
        "min_segment_fraction": 0.1,
        "max_segment_fraction": 0.2,
        "spike_scale": 3.0,
    }
    model_kwargs.update(overrides)
    return ThesisMultitaskModel(**model_kwargs)


def _build_batch(batch_size: int = 2) -> dict[str, object]:
    return {
        "x": torch.randn(batch_size, 100, 38),
        "point_labels": torch.zeros(batch_size, 100, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": f"machine-{index}"} for index in range(batch_size)],
    }


def test_refurbished_binary_targets_match_configured_alpha_and_beta() -> None:
    model = _build_model(
        use_label_refurbishment=True,
        refurbishment_alpha=0.2,
        refurbishment_beta=0.1,
    )
    hard_labels = torch.tensor([0, 1], dtype=torch.long)

    refurbished_targets = model._build_refurbished_binary_targets(
        hard_labels, torch.float32
    )

    expected_targets = torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float32)
    assert torch.allclose(refurbished_targets, expected_targets)


def test_masked_reconstruction_loss_ignores_anomalous_time_steps() -> None:
    model = _build_model(reconstruction_normal_only=True)
    recon = torch.tensor([[[0.0], [2.0], [0.0], [2.0]]], dtype=torch.float32)
    batch = {
        "x": torch.zeros(1, 4, 1, dtype=torch.float32),
        "synthetic_anomaly_mask": torch.tensor([[0, 1, 0, 1]], dtype=torch.long),
    }

    reconstruction_loss = model._compute_reconstruction_loss({"recon": recon}, batch)

    assert reconstruction_loss.item() == 0.0


def test_masked_reconstruction_loss_falls_back_to_full_mse_when_disabled() -> None:
    model = _build_model(reconstruction_normal_only=False)
    recon = torch.tensor([[[0.0], [2.0], [0.0], [2.0]]], dtype=torch.float32)
    batch = {
        "x": torch.zeros(1, 4, 1, dtype=torch.float32),
        "synthetic_anomaly_mask": torch.tensor([[0, 1, 0, 1]], dtype=torch.long),
    }

    reconstruction_loss = model._compute_reconstruction_loss({"recon": recon}, batch)

    assert reconstruction_loss.item() == 2.0


def test_training_step_runs_with_refurbishment_and_normal_only_masking_enabled() -> (
    None
):
    model = _build_model(
        use_label_refurbishment=True,
        refurbishment_alpha=0.2,
        refurbishment_beta=0.1,
        reconstruction_normal_only=True,
        use_synthetic_augmentation=True,
        anomaly_probability=1.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    batch = _build_batch()

    step_output = model.training_step(batch)
    optimizer.zero_grad()
    step_output["loss"].backward()
    optimizer.step()

    squared_reconstruction_error = (
        step_output["outputs"]["recon"] - step_output["batch"]["x"]
    ) ** 2
    normal_time_step_mask = step_output["batch"]["synthetic_anomaly_mask"] == 0
    expanded_normal_mask = normal_time_step_mask.unsqueeze(-1).expand_as(
        squared_reconstruction_error
    )
    expected_normal_only_reconstruction_loss = (
        torch.sum(squared_reconstruction_error * expanded_normal_mask)
        / expanded_normal_mask.sum()
    )
    assert step_output["loss"].item() >= 0.0
    assert step_output["loss_terms"]["classification_loss"].item() >= 0.0
    assert step_output["loss_terms"]["reconstruction_loss"].item() >= 0.0
    assert step_output["batch"]["classification_labels"].dtype == torch.long
    assert step_output["batch"]["classification_labels"].shape == (2,)
    assert torch.all(
        step_output["batch"]["classification_labels"] == torch.tensor([1, 1])
    )
    assert torch.allclose(
        step_output["loss_terms"]["reconstruction_loss"],
        expected_normal_only_reconstruction_loss,
    )
