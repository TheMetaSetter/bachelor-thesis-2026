from __future__ import annotations

import torch

from src.models.thesis_multitask import ThesisMultitaskModel


def _build_model() -> ThesisMultitaskModel:
    return ThesisMultitaskModel(
        input_dim=38,
        window_size=20,
        encoder_dim=32,
        hidden_dim=8,
        num_classes=2,
        continuous_enabled=True,
        continuous_num_prototypes=4,
        discrete_enabled=True,
        discrete_codebook_size=8,
        use_synthetic_augmentation=True,
        use_synthetic_validation=True,
        enable_two_view_contrastive=True,
        enable_cka_gated_fusion=True,
        classification_label_mode="binary",
    )


def test_linear_cka_score_is_finite_and_high_for_identical_inputs() -> None:
    model = _build_model()
    hidden = torch.randn(20, 8)
    cka_score = model._compute_linear_cka_score(hidden, hidden)
    assert torch.isfinite(cka_score)
    assert float(cka_score.item()) > 0.99


def test_linear_cka_batch_shape_and_range() -> None:
    model = _build_model()
    lhs = torch.randn(3, 20, 8)
    rhs = torch.randn(3, 20, 8)
    cka_scores = model._compute_batch_linear_cka_scores(lhs, rhs)
    assert cka_scores.shape == (3,)
    assert torch.all(torch.isfinite(cka_scores))
    assert torch.all(cka_scores >= -1.0)
    assert torch.all(cka_scores <= 1.0)


def test_two_view_contrastive_loss_uses_normal_tokens_and_is_finite() -> None:
    model = _build_model()
    anchors = torch.randn(2, 20, 8)
    positives = torch.randn(2, 20, 8)
    synthetic_anomaly_mask = torch.zeros(2, 20, dtype=torch.long)
    synthetic_anomaly_mask[:, :5] = 1
    contrastive_loss = model._compute_two_view_contrastive_loss(
        anchor_hidden=anchors,
        positive_hidden=positives,
        synthetic_anomaly_mask=synthetic_anomaly_mask,
    )
    assert torch.isfinite(contrastive_loss)
    assert float(contrastive_loss.item()) >= 0.0
