from __future__ import annotations

from typing import Any

import torch


def compute_hard_prediction_ratio(
    logits: torch.Tensor, class_names: tuple[str, ...] | list[str]
) -> dict[str, float]:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [N, C]")
    predicted_labels = torch.argmax(logits, dim=-1)
    num_predictions = max(int(predicted_labels.shape[0]), 1)
    ratios: dict[str, float] = {}
    for class_index, class_name in enumerate(class_names):
        ratios[class_name] = float((predicted_labels == class_index).sum().item()) / float(
            num_predictions
        )
    return ratios


def compute_row_normalized_confusion_matrix(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [N, C]")
    if labels.ndim != 1:
        raise ValueError("labels must have shape [N]")
    num_classes = len(class_names)
    predicted_labels = torch.argmax(logits, dim=-1).long()
    labels = labels.long()

    confusion_counts = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for true_label, predicted_label in zip(labels, predicted_labels):
        if 0 <= int(true_label) < num_classes and 0 <= int(predicted_label) < num_classes:
            confusion_counts[int(true_label), int(predicted_label)] += 1

    row_sums = confusion_counts.sum(dim=1, keepdim=True).to(dtype=torch.float32)
    normalized = torch.where(
        row_sums > 0.0,
        confusion_counts.to(dtype=torch.float32) / row_sums.clamp_min(1.0),
        torch.zeros_like(confusion_counts, dtype=torch.float32),
    )
    return {
        "class_names": list(class_names),
        "counts": confusion_counts.tolist(),
        "row_normalized": normalized.tolist(),
        "support": confusion_counts.sum(dim=1).tolist(),
    }
