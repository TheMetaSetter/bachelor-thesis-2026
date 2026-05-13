from __future__ import annotations

import math
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


def extract_binary_anomaly_ranges(point_labels: np.ndarray) -> list[tuple[int, int]]:
    label_array = np.asarray(point_labels).astype(np.int64).reshape(-1)
    ranges: list[tuple[int, int]] = []
    range_start: int | None = None

    for index, label_value in enumerate(label_array):
        if label_value == 1 and range_start is None:
            range_start = index
        elif label_value == 0 and range_start is not None:
            ranges.append((range_start, index))
            range_start = None

    if range_start is not None:
        ranges.append((range_start, len(label_array)))

    return ranges


def _prediction_exists(binary_predictions: np.ndarray, start: int, end: int) -> bool:
    if start >= end:
        return False
    return bool(np.any(binary_predictions[start:end] == 1))


def build_threshold_aware_range_labels(
    point_labels: np.ndarray,
    binary_predictions: np.ndarray,
    buffer_size: int,
) -> np.ndarray:
    label_array = np.asarray(point_labels).astype(np.int64).reshape(-1)
    prediction_array = np.asarray(binary_predictions).astype(np.int64).reshape(-1)
    if label_array.shape != prediction_array.shape:
        raise ValueError("point_labels and binary_predictions must have the same shape")
    if buffer_size < 0:
        raise ValueError("buffer_size must be non-negative")

    range_labels = label_array.astype(np.float64)
    if buffer_size == 0:
        return range_labels

    for anomaly_start, anomaly_end in extract_binary_anomaly_ranges(label_array):
        left_start = max(0, anomaly_start - buffer_size)
        left_end = anomaly_start
        right_start = anomaly_end
        right_end = min(len(label_array), anomaly_end + buffer_size)

        if _prediction_exists(prediction_array, left_start, left_end):
            for index in range(left_start, left_end):
                distance_to_boundary = anomaly_start - index
                range_labels[index] = max(
                    range_labels[index],
                    math.sqrt(max(0.0, 1.0 - distance_to_boundary / buffer_size)),
                )

        if _prediction_exists(prediction_array, right_start, right_end):
            for index in range(right_start, right_end):
                distance_to_boundary = index - anomaly_end + 1
                range_labels[index] = max(
                    range_labels[index],
                    math.sqrt(max(0.0, 1.0 - distance_to_boundary / buffer_size)),
                )

    return range_labels


def _build_score_thresholds(point_scores: np.ndarray, num_thresholds: int) -> np.ndarray:
    score_array = np.asarray(point_scores).astype(np.float64).reshape(-1)
    if num_thresholds <= 0:
        raise ValueError("num_thresholds must be positive")
    if score_array.size == 0:
        return np.array([], dtype=np.float64)

    unique_scores = np.unique(score_array)
    if unique_scores.size <= num_thresholds:
        thresholds = unique_scores
    else:
        thresholds = np.linspace(
            float(np.min(score_array)),
            float(np.max(score_array)),
            num=num_thresholds,
            dtype=np.float64,
        )
    return np.concatenate(
        [
            np.array([float(np.min(score_array)) - 1.0e-12], dtype=np.float64),
            thresholds,
            np.array([float(np.max(score_array)) + 1.0e-12], dtype=np.float64),
        ]
    )


def _compute_existence_reward(
    point_labels: np.ndarray,
    binary_predictions: np.ndarray,
) -> float:
    anomaly_ranges = extract_binary_anomaly_ranges(point_labels)
    if not anomaly_ranges:
        return float("nan")

    detected_ranges = 0
    for anomaly_start, anomaly_end in anomaly_ranges:
        if _prediction_exists(binary_predictions, anomaly_start, anomaly_end):
            detected_ranges += 1
    return float(detected_ranges / len(anomaly_ranges))


def _compute_range_precision_recall(
    point_labels: np.ndarray,
    point_scores: np.ndarray,
    threshold: float,
    buffer_size: int,
) -> tuple[float, float]:
    binary_predictions = (point_scores > threshold).astype(np.int64)
    range_labels = build_threshold_aware_range_labels(
        point_labels=point_labels,
        binary_predictions=binary_predictions,
        buffer_size=buffer_size,
    )
    true_positive = float(np.dot(range_labels, binary_predictions))
    false_positive = float(np.dot(1.0 - range_labels, binary_predictions))
    predicted_positive = true_positive + false_positive
    precision = 1.0 if predicted_positive == 0.0 else true_positive / predicted_positive

    positive_mass = float(np.sum((point_labels.astype(np.float64) + range_labels) / 2.0))
    if positive_mass == 0.0:
        recall = float("nan")
    else:
        existence_reward = _compute_existence_reward(point_labels, binary_predictions)
        recall = (true_positive / positive_mass) * existence_reward
    precision = min(1.0, max(0.0, precision))
    if np.isfinite(recall):
        recall = min(1.0, max(0.0, recall))
    return precision, recall


def _compute_pr_area_from_points(
    precision_values: list[float],
    recall_values: list[float],
) -> float:
    best_precision_by_recall: dict[float, float] = {}
    for recall, precision in zip(recall_values, precision_values):
        if not np.isfinite(recall) or not np.isfinite(precision):
            continue
        clipped_recall = min(1.0, max(0.0, float(recall)))
        clipped_precision = min(1.0, max(0.0, float(precision)))
        best_precision_by_recall[clipped_recall] = max(
            clipped_precision,
            best_precision_by_recall.get(clipped_recall, 0.0),
        )

    finite_points = [
        (recall, precision)
        for recall, precision in best_precision_by_recall.items()
        if np.isfinite(recall) and np.isfinite(precision)
    ]
    if not finite_points:
        return float("nan")

    finite_points.append((0.0, 1.0))
    finite_points.sort(key=lambda pair: pair[0])
    sorted_recalls = np.array([pair[0] for pair in finite_points], dtype=np.float64)
    sorted_precisions = np.array([pair[1] for pair in finite_points], dtype=np.float64)
    for index in range(sorted_precisions.size - 2, -1, -1):
        sorted_precisions[index] = max(
            sorted_precisions[index],
            sorted_precisions[index + 1],
        )
    pr_area = float(np.trapezoid(sorted_precisions, sorted_recalls))
    return min(1.0, max(0.0, pr_area))


def compute_vus_pr_exact_naive(
    point_labels: np.ndarray,
    point_scores: np.ndarray,
    max_buffer_size: int,
    num_thresholds: int = 200,
) -> float:
    label_array = np.asarray(point_labels).astype(np.int64).reshape(-1)
    score_array = np.asarray(point_scores).astype(np.float64).reshape(-1)
    if label_array.shape != score_array.shape:
        raise ValueError("point_labels and point_scores must have the same shape")
    if max_buffer_size < 0:
        raise ValueError("max_buffer_size must be non-negative")
    if len(np.unique(label_array)) < 2:
        return float("nan")

    thresholds = _build_score_thresholds(score_array, num_thresholds)
    average_precision_values: list[float] = []
    for buffer_size in range(max_buffer_size + 1):
        precision_values: list[float] = []
        recall_values: list[float] = []
        for threshold in thresholds:
            precision, recall = _compute_range_precision_recall(
                point_labels=label_array,
                point_scores=score_array,
                threshold=float(threshold),
                buffer_size=buffer_size,
            )
            precision_values.append(precision)
            recall_values.append(recall)
        average_precision_values.append(
            _compute_pr_area_from_points(
                precision_values=precision_values,
                recall_values=recall_values,
            )
        )

    finite_average_precision_values = [
        value for value in average_precision_values if np.isfinite(value)
    ]
    if not finite_average_precision_values:
        return float("nan")
    return float(np.mean(finite_average_precision_values))


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
    vus_max_buffer_size: int | None = None,
    vus_num_thresholds: int = 200,
) -> dict[str, float]:
    # Use a strict comparison so a collapsed zero threshold does not mark every
    # zero-valued point as anomalous.

    # TODO: Không được phép sử dụng các point-adjusted metrics
    # TODO: Đọc, hiểu bản chất và sử dụng thêm metric VUS-PR.

    # Dựa vào threshold đã tính toán ra dựa trên
    # anomlay score của tất cả timestep trong validation set
    # để convert score sang dự đoán nhị phân.
    binary_predictions = (point_scores > threshold).astype(np.int64)

    metrics = {
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
    if vus_max_buffer_size is not None:
        metrics["vus_pr"] = compute_vus_pr_exact_naive(
            point_labels=point_labels,
            point_scores=point_scores,
            max_buffer_size=vus_max_buffer_size,
            num_thresholds=vus_num_thresholds,
        )
    return metrics


def compute_pointwise_curve_payload(
    point_labels: np.ndarray,
    point_scores: np.ndarray,
) -> dict[str, dict[str, list[float]]]:
    return {
        "roc_curve": _safe_curve(roc_curve, point_labels, point_scores),
        "pr_curve": _safe_curve(precision_recall_curve, point_labels, point_scores),
    }
