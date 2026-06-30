from __future__ import annotations

import torch

from src.data.augment import REDLAMP_MULTICLASS_CLASS_NAMES
from src.models.redlamp_baseline import RedLampBaseline


def test_redlamp_cnn_baseline_forward_backward_and_gradient_profiling() -> None:
    model = RedLampBaseline(
        input_dim=4,
        window_size=20,
        latent_dim=16,
        encoder_family="cnn_simple",
        cnn_num_layers=2,
        cnn_kernel_size=3,
        cnn_hidden_channels=8,
        mlp_num_linear_layers=3,
        classifier_dim=8,
        num_classes=len(REDLAMP_MULTICLASS_CLASS_NAMES),
        dropout=0.0,
        anomaly_probability=1.0,
        enable_gradient_conflict_profiling=True,
        gradient_log_every_n_steps=1,
        gradient_focus_layer_name="encoder_last_affine",
    )
    batch = {
        "x": torch.randn(2, 20, 4),
        "point_labels": torch.zeros(2, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "unit-test"}, {"entity_id": "unit-test"}],
    }

    step_output = model.training_step(batch)
    outputs = step_output["outputs"]
    loss = step_output["loss"]
    loss.backward()

    assert outputs["hidden"].shape == (2, 20, 16)
    assert outputs["pooled"].shape == (2, 20 * 16)
    assert outputs["recon"].shape == (2, 20, 4)
    assert outputs["logits"].shape == (2, len(REDLAMP_MULTICLASS_CLASS_NAMES))
    assert outputs["point_scores"].shape == (2, 20)
    assert outputs["window_scores"].shape == (2,)
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())
    assert any(key.startswith("train_gradconf_raw/") for key in step_output["log"])
