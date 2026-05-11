from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_curve,
    roc_auc_score,
)


def _safe_metric(metric_function: Any, *args: Any, **kwargs: Any) -> float:
    try:
        return float(metric_function(*args, **kwargs))
    except ValueError:
        return float("nan")


def _safe_curve(
    metric_function: Any, *args: Any, **kwargs: Any
) -> dict[str, list[float]]:
    try:
        curve_outputs = metric_function(*args, **kwargs)
    except ValueError:
        return {"x": [], "y": [], "thresholds": []}

    if metric_function is precision_recall_curve:
        precision_values, recall_values, thresholds = curve_outputs
        return {
            "x": [float(value) for value in recall_values.tolist()],
            "y": [float(value) for value in precision_values.tolist()],
            "thresholds": [float(value) for value in thresholds.tolist()],
        }

    false_positive_rates, true_positive_rates, thresholds = curve_outputs
    return {
        "x": [float(value) for value in false_positive_rates.tolist()],
        "y": [float(value) for value in true_positive_rates.tolist()],
        "thresholds": [float(value) for value in thresholds.tolist()],
    }


def _compute_false_positive_rate(
    labels: np.ndarray,
    binary_predictions: np.ndarray,
) -> float:
    try:
        true_negative, false_positive, _, _ = confusion_matrix(
            labels,
            binary_predictions,
            labels=[0, 1],
        ).ravel()
    except ValueError:
        return float("nan")
    denominator = true_negative + false_positive
    if denominator == 0:
        return float("nan")
    return float(false_positive / denominator)


def compute_binary_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    label_array = labels.detach().cpu().numpy().astype(np.int64)
    probabilities = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
    binary_predictions = (probabilities > 0.5).astype(np.int64)
    return {
        "roc_auc": _safe_metric(roc_auc_score, label_array, probabilities),
        "pr_auc": _safe_metric(average_precision_score, label_array, probabilities),
        "precision": _safe_metric(
            precision_score, label_array, binary_predictions, zero_division=0
        ),
        "recall": _safe_metric(
            recall_score, label_array, binary_predictions, zero_division=0
        ),
        "fpr": _compute_false_positive_rate(label_array, binary_predictions),
    }


def compute_multiclass_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    label_array = labels.detach().cpu().numpy().astype(np.int64)
    prediction_array = torch.argmax(logits.detach().cpu(), dim=-1).numpy().astype(
        np.int64
    )
    return {
        "accuracy": _safe_metric(accuracy_score, label_array, prediction_array),
        "macro_f1": _safe_metric(
            f1_score,
            label_array,
            prediction_array,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": _safe_metric(
            f1_score,
            label_array,
            prediction_array,
            average="weighted",
            zero_division=0,
        ),
        "num_classes_observed": float(len(np.unique(label_array))),
    }


def compute_pointwise_metrics(
    point_labels: np.ndarray,
    point_scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    # Use a strict comparison so a collapsed zero threshold does not mark every
    # zero-valued point as anomalous.

    # TODO: Không được phép sử dụng các point-adjusted metrics
    # TODO: Đọc, hiểu bản chất và sử dụng thêm metric VUS-PR.

    # Dựa vào threshold đã tính toán ra dựa trên
    # anomlay score của tất cả timestep trong validation set
    # để convert score sang dự đoán nhị phân.
    binary_predictions = (point_scores > threshold).astype(np.int64)

    return {
        "roc_auc": _safe_metric(roc_auc_score, point_labels, point_scores),
        "pr_auc": _safe_metric(average_precision_score, point_labels, point_scores),
        "precision": _safe_metric(
            precision_score, point_labels, binary_predictions, zero_division=0
        ),
        "recall": _safe_metric(
            recall_score, point_labels, binary_predictions, zero_division=0
        ),
        "f1": _safe_metric(f1_score, point_labels, binary_predictions, zero_division=0),
        "fpr": _compute_false_positive_rate(point_labels, binary_predictions),
    }


def compute_pointwise_curve_payload(
    point_labels: np.ndarray,
    point_scores: np.ndarray,
) -> dict[str, dict[str, list[float]]]:
    return {
        "roc_curve": _safe_curve(roc_curve, point_labels, point_scores),
        "pr_curve": _safe_curve(precision_recall_curve, point_labels, point_scores),
    }
