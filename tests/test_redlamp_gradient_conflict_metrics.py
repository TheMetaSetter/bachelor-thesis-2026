from __future__ import annotations

import math

import torch

from src.data.augment import REDLAMP_MULTICLASS_CLASS_NAMES
from src.models.redlamp_mlp_baseline import RedLampMLPBaseline


def _build_model_for_gradient_metrics() -> RedLampMLPBaseline:
    return RedLampMLPBaseline(
        input_dim=4,
        window_size=20,
        latent_dim=16,
        mlp_num_linear_layers=3,
        classifier_dim=8,
        num_classes=len(REDLAMP_MULTICLASS_CLASS_NAMES),
        dropout=0.0,
        anomaly_probability=1.0,
    )


def test_gradient_conflict_metric_helpers_match_expected_values() -> None:
    model = _build_model_for_gradient_metrics()
    gradient_ce = torch.tensor([1.0, 0.0], dtype=torch.float32)
    gradient_mse = torch.tensor([0.0, 1.0], dtype=torch.float32)
    gradient_total = gradient_ce + gradient_mse

    cosine_similarity = model._compute_cosine_similarity(gradient_ce, gradient_mse)
    preservation_ratio = model._compute_preservation_ratio(
        gradient_ce=gradient_ce,
        gradient_mse=gradient_mse,
        gradient_total=gradient_total,
    )

    assert math.isclose(cosine_similarity, 0.0, abs_tol=1e-6)
    assert math.isclose(preservation_ratio, math.sqrt(2.0) / 2.0, rel_tol=1e-6)
    assert -1.0 <= cosine_similarity <= 1.0
    assert 0.0 <= preservation_ratio <= 1.0


def test_gradient_conflict_smoothing_helpers_behave_as_expected() -> None:
    model = _build_model_for_gradient_metrics()
    model.gradient_ema_alpha = 0.1
    model.gradient_sma_window = 3

    ema_1 = model._update_ema("metric", 1.0)
    ema_2 = model._update_ema("metric", 0.0)
    ema_3 = model._update_ema("metric", 0.0)
    sma_1 = model._update_sma("metric", 1.0)
    sma_2 = model._update_sma("metric", 2.0)
    sma_3 = model._update_sma("metric", 3.0)
    sma_4 = model._update_sma("metric", 6.0)

    assert math.isclose(ema_1, 1.0, rel_tol=1e-6)
    assert math.isclose(ema_2, 0.9, rel_tol=1e-6)
    assert math.isclose(ema_3, 0.81, rel_tol=1e-6)
    assert math.isclose(sma_1, 1.0, rel_tol=1e-6)
    assert math.isclose(sma_2, 1.5, rel_tol=1e-6)
    assert math.isclose(sma_3, 2.0, rel_tol=1e-6)
    assert math.isclose(sma_4, (2.0 + 3.0 + 6.0) / 3.0, rel_tol=1e-6)
