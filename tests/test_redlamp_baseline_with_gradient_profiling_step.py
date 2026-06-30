from __future__ import annotations

import math

import torch

from src.data.augment import REDLAMP_MULTICLASS_CLASS_NAMES
from src.models.redlamp_baseline import RedLampBaseline


def test_redlamp_baseline_train_step_logs_gradient_conflict_metrics() -> None:
    model = RedLampBaseline(
        input_dim=4,
        window_size=20,
        latent_dim=16,
        mlp_num_linear_layers=3,
        classifier_dim=8,
        num_classes=len(REDLAMP_MULTICLASS_CLASS_NAMES),
        anomaly_probability=1.0,
        enable_gradient_conflict_profiling=True,
        gradient_log_every_n_steps=1,
        gradient_ema_alpha=0.1,
        gradient_sma_window=50,
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
