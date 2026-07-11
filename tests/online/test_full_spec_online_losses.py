from __future__ import annotations

import pytest
import torch

from src.engine.online_tta.online_losses import (
    compute_hard_old_hinge_loss,
    compute_token_multi_positive_info_nce,
)


def test_hard_old_hinge_is_zero_at_threshold() -> None:
    score = torch.tensor(2.0, requires_grad=True)
    loss = compute_hard_old_hinge_loss(score, 2.0)
    assert loss.item() == 0.0


def test_token_info_nce_uses_anomalous_codeword_negatives() -> None:
    projected = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    source = projected.detach().clone()
    far_negative = torch.tensor([[-1.0, 0.0]])
    close_negative = torch.tensor([[1.0, 0.0]])
    far_loss = compute_token_multi_positive_info_nce(projected, source, far_negative)
    close_loss = compute_token_multi_positive_info_nce(projected, source, close_negative)
    assert close_loss > far_loss
    far_loss.backward()
    assert projected.grad is not None


def test_token_info_nce_requires_anomalous_codeword() -> None:
    hidden = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="at least one"):
        compute_token_multi_positive_info_nce(hidden, hidden, torch.empty(0, 3))


def test_empty_pnn_mask_skips_with_differentiable_zero() -> None:
    projected = torch.randn(1, 2, 3, requires_grad=True)
    source = torch.randn(1, 2, 3)
    loss = compute_token_multi_positive_info_nce(
        projected,
        source,
        torch.randn(2, 3),
        pnn_mask=torch.zeros(1, 2, dtype=torch.bool),
    )
    assert loss.item() == 0.0
    loss.backward()
    assert projected.grad is not None
