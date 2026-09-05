from __future__ import annotations

import pytest
import torch

from src.data.scalers import SequenceStandardScaler
from src.engine.checkpoint import CheckpointManager
from src.models.thesis_multitask import ThesisMultitaskModel


def build_model():
    return ThesisMultitaskModel(
        input_dim=2,
        window_size=4,
        encoder_dim=8,
        hidden_dim=4,
        num_classes=2,
        continuous_num_prototypes=2,
        discrete_codebook_size=2,
        training_phase="stage_a_multitask_pretraining",
        use_synthetic_augmentation=False,
        use_synthetic_validation=False,
        reconstruction_normal_only=True,
        enable_score_loss=True,
        score_loss_type="pointwise_balanced_bce_logits",
    )


def scaler_state():
    return {
        "epsilon": 0.001,
        "feature_mean": torch.tensor([10.0, 5.0]),
        "feature_std": torch.tensor([2.0, 0.0]),
        "feature_active_mask": torch.tensor([True, False]),
    }


def test_raw_masked_loss_and_gradient_preserve_inactive_sensor_units():
    model = build_model()
    model.configure_reconstruction_loss("raw_input", scaler_state())
    reconstruction = torch.tensor([[[1.0, 2.0], [9.0, 9.0]]], requires_grad=True)
    batch = {
        "x": torch.zeros_like(reconstruction),
        "synthetic_anomaly_mask": torch.tensor([[0, 1]]),
    }
    loss = model._compute_reconstruction_loss({"recon": reconstruction}, batch)
    assert loss.item() == pytest.approx(4.0)  # (2**2 + 2**2) / 2
    loss.backward()
    torch.testing.assert_close(
        reconstruction.grad, torch.tensor([[[4.0, 2.0], [0.0, 0.0]]])
    )
    diagnostics = model._compute_reconstruction_diagnostics(
        {"recon": reconstruction}, batch
    )
    assert diagnostics["recon_mse_mean"] == pytest.approx(103.25)
    assert diagnostics["normalized_input_recon_mse_mean"] == pytest.approx(41.75)


def test_raw_loss_averages_sample_errors_before_mc_reduction():
    model = build_model()
    model.configure_reconstruction_loss("raw_input", scaler_state())
    samples = torch.tensor([[[[1.0, 0.0]], [[-1.0, 0.0]]]])
    outputs = {
        "recon": samples.mean(dim=1),
        "aux": {"stochastic_query": {"reconstruction_samples": samples}},
    }
    assert (
        model._compute_reconstruction_loss(outputs, {"x": torch.zeros(1, 1, 2)}).item()
        == 2.0
    )
    assert (
        model._compute_reconstruction_diagnostics(outputs, {"x": torch.zeros(1, 1, 2)})[
            "normalized_input_recon_mse_mean"
        ]
        == 0.5
    )


def test_raw_point_supervision_uses_sensor_weighted_mse():
    model = build_model()
    model.configure_reconstruction_loss("raw_input", scaler_state())
    outputs = {
        "recon": torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]])
    }
    batch = {
        "x": torch.zeros(1, 4, 2),
        "synthetic_anomaly_mask": torch.tensor([[0, 0, 1, 1]]),
    }
    loss, _ = model._compute_point_score_loss(outputs, batch)
    # Raw point MSE=[2,.5,2.5,8]; normal mean=1.25, std=.75.
    logits = torch.tensor([1.0, -1.0, 5.0 / 3.0, 9.0])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([0.0, 0.0, 1.0, 1.0])
    )
    torch.testing.assert_close(loss, expected)


def test_checkpoint_restores_raw_loss_context(tmp_path):
    model = build_model()
    model.configure_reconstruction_loss("raw_input", scaler_state())
    manager = CheckpointManager(tmp_path)
    path = manager.save_checkpoint(
        "best.pt",
        model,
        None,
        None,
        scaler_state(),
        {"reconstruction_loss_space": "raw_input"},
        1,
        [],
    )
    restored = build_model()
    manager.load_checkpoint(path, restored)
    assert restored.reconstruction_loss_space == "raw_input"
    assert (
        restored._compute_reconstruction_loss(
            {"recon": torch.ones(1, 1, 2)}, {"x": torch.zeros(1, 1, 2)}
        ).item()
        == 2.5
    )


def test_inverse_transform_retains_float64_autograd():
    scaler = SequenceStandardScaler()
    scaler.load_state_dict(scaler_state())
    values = torch.ones(1, 2, dtype=torch.float64, requires_grad=True)
    restored = scaler.inverse_transform_tensor(values)
    assert restored.dtype == torch.float64
    restored.sum().backward()
    torch.testing.assert_close(
        values.grad, torch.tensor([[2.0, 1.0]], dtype=torch.float64)
    )


@pytest.mark.parametrize("enable_score_loss", [False, True])
def test_raw_stage_a_variants_forward_backward(enable_score_loss):
    from tests.models.test_thesis_multitask_point_score_loss import (
        _build_model,
        _build_batch,
    )

    model = _build_model(enable_score_loss)
    model.configure_reconstruction_loss(
        "raw_input",
        {
            "epsilon": 0.001,
            "feature_mean": torch.zeros(38),
            "feature_std": torch.full((38,), 2.0),
            "feature_active_mask": torch.ones(38, dtype=torch.bool),
        },
    )
    step = model.training_step(_build_batch())
    prepared = step["batch"]
    errors = 4 * (step["outputs"]["recon"] - prepared["x"]).square()
    normal = ~prepared["synthetic_anomaly_mask"].bool()
    expected = (
        errors[normal].mean() if model.reconstruction_normal_only else errors.mean()
    )
    torch.testing.assert_close(step["loss_terms"]["reconstruction_loss"], expected)
    step["loss"].backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
