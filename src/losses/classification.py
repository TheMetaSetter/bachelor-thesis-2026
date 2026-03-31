from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [B, C]")
    if labels.ndim != 1:
        raise ValueError("labels must have shape [B]")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels batch sizes must match")
    return F.cross_entropy(logits, labels.long())
