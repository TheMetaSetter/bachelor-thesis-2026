from __future__ import annotations

import math

import torch

from src.models.thesis_multitask import ThesisMultitaskModel


def test_thesis_multitask_train_step_logs_gradient_conflict_metrics() -> None:
    model = ThesisMultitaskModel(
        input_dim=4,
        window_size=20,
        encoder_dim=16,
        hidden_dim=8,
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
        lambda_recon=0.9,
        lambda_cls=0.1,
        use_synthetic_augmentation=True,
        anomaly_probability=1.0,
        min_segment_fraction=0.1,
        max_segment_fraction=0.2,
        spike_scale=3.0,
        classification_label_mode="redlamp_multiclass",
        enable_gradient_conflict_profiling=True,
        gradient_log_every_n_steps=1,
        gradient_focus_layer_name="encoder_last_affine",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    batch = {
        "x": torch.randn(2, 20, 4),
        "point_labels": torch.zeros(2, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "unit-test"}, {"entity_id": "unit-test"}],
    }

    optimizer.zero_grad(set_to_none=True)
    step_output = model.training_step(batch)
    step_output["loss"].backward()
    optimizer.step()

    required_log_keys = [
        "train_gradconf_raw/focus/cosine_sim",
        "train_gradconf_raw/focus/r_ratio",
        "train_gradconf_ema/focus/cosine_sim",
        "train_gradconf_sma/focus/cosine_sim",
    ]
    for metric_key in required_log_keys:
        assert metric_key in step_output["log"]
        assert math.isfinite(step_output["log"][metric_key])
