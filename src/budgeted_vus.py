"""Compute VUS at fixed false positive rate limits."""

import math

import numpy as np


FPR_BUDGETS = (0.001, 0.005, 0.01)


def _validate_pointwise_array_shapes(
    point_labels: np.ndarray,
    point_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    label_array = np.asarray(point_labels).astype(np.int64).reshape(-1)
    score_array = np.asarray(point_scores).astype(np.float64).reshape(-1)
    if label_array.shape != score_array.shape:
        raise ValueError("point_labels and point_scores must have the same shape")
    return label_array, score_array


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


def _build_score_thresholds(
    point_scores: np.ndarray, num_thresholds: int
) -> np.ndarray:
    score_array = np.asarray(point_scores).astype(np.float64).reshape(-1)

    # debug
    print("score_array w/ reshape(-1):", score_array)
    print("score_array w/o reshape(-1):", np.asarray(point_scores).astype(np.float64))

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

    positive_mass = float(
        np.sum((point_labels.astype(np.float64) + range_labels) / 2.0)
    )
    if positive_mass == 0.0:
        recall = float("nan")
    else:
        existence_reward = _compute_existence_reward(point_labels, binary_predictions)
        recall = (true_positive / positive_mass) * existence_reward
    precision = min(1.0, max(0.0, precision))
    if np.isfinite(recall):
        recall = min(1.0, max(0.0, recall))
    return precision, recall


def _compute_range_roc_rates(
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
    positive_mass = float(np.sum(range_labels))
    negative_mass = float(np.sum(1.0 - range_labels))
    if positive_mass == 0.0:
        true_positive_rate = float("nan")
    else:
        true_positive_rate = float(
            np.dot(range_labels, binary_predictions) / positive_mass
        )
    if negative_mass == 0.0:
        false_positive_rate = float("nan")
    else:
        false_positive_rate = float(
            np.dot(1.0 - range_labels, binary_predictions) / negative_mass
        )
    if np.isfinite(true_positive_rate):
        true_positive_rate = min(1.0, max(0.0, true_positive_rate))
    if np.isfinite(false_positive_rate):
        false_positive_rate = min(1.0, max(0.0, false_positive_rate))
    return false_positive_rate, true_positive_rate


def _normalised_partial_roc_area(
    false_positive_rates: list[float],
    true_positive_rates: list[float],
    budget: float,
) -> float:
    curve = np.asarray(
        [
            (false_positive_rate, true_positive_rate)
            for false_positive_rate, true_positive_rate in zip(
                false_positive_rates,
                true_positive_rates,
                strict=True,
            )
            if np.isfinite(false_positive_rate) and np.isfinite(true_positive_rate)
        ],
        dtype=np.float64,
    )
    if curve.size == 0:
        return float("nan")
    curve = curve[np.argsort(curve[:, 0], kind="stable")]
    unique_fprs, inverse = np.unique(curve[:, 0], return_inverse=True)
    curve = np.column_stack(
        [
            unique_fprs,
            np.asarray(
                [
                    np.max(curve[inverse == index, 1])
                    for index in range(unique_fprs.size)
                ],
                dtype=np.float64,
            ),
        ]
    )
    curve[:, 1] = np.maximum.accumulate(curve[:, 1])
    allowed_curve = curve[curve[:, 0] <= budget]
    if allowed_curve.size == 0:
        return float("nan")
    if float(allowed_curve[-1, 0]) < budget:
        boundary_tpr = float(np.interp(budget, curve[:, 0], curve[:, 1]))
        allowed_curve = np.vstack(
            [allowed_curve, np.asarray([[budget, boundary_tpr]], dtype=np.float64)]
        )
    trapezoid_function = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area = float(trapezoid_function(allowed_curve[:, 1], allowed_curve[:, 0]))
    return min(1.0, max(0.0, area / budget))


def _constrained_pr_area(
    precision_values: list[float],
    recall_values: list[float],
    false_positive_rates: list[float],
    budget: float,
) -> float:
    curve = np.asarray(
        [
            (precision, recall)
            for precision, recall, false_positive_rate in zip(
                precision_values,
                recall_values,
                false_positive_rates,
                strict=True,
            )
            if (
                np.isfinite(precision)
                and np.isfinite(recall)
                and np.isfinite(false_positive_rate)
                and false_positive_rate <= budget
            )
        ],
        dtype=np.float64,
    )
    if curve.size == 0:
        return float("nan")
    curve = curve[np.argsort(curve[:, 1], kind="stable")]
    area = 0.0
    for index in range(1, curve.shape[0]):
        recall_change = float(curve[index, 1] - curve[index - 1, 1])
        if recall_change > 0.0:
            area += float(curve[index, 0]) * recall_change
    return min(1.0, max(0.0, area))


def _threshold_rates(labels, scores, thresholds, buffer_size):
    precision_values: list[float] = []
    recall_values: list[float] = []
    false_positive_rates: list[float] = []
    true_positive_rates: list[float] = []
    for threshold in thresholds:
        precision, recall = _compute_range_precision_recall(
            point_labels=labels,
            point_scores=scores,
            threshold=float(threshold),
            buffer_size=buffer_size,
        )
        false_positive_rate, true_positive_rate = _compute_range_roc_rates(
            point_labels=labels,
            point_scores=scores,
            threshold=float(threshold),
            buffer_size=buffer_size,
        )
        precision_values.append(precision)
        recall_values.append(recall)
        false_positive_rates.append(false_positive_rate)
        true_positive_rates.append(true_positive_rate)
    return precision_values, recall_values, false_positive_rates, true_positive_rates


def _aggregate_over_buffers(values: list[float], max_buffer_size: int) -> float:
    finite_values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(finite_values)):
        return float("nan")
    if max_buffer_size == 0:
        return float(finite_values[0])
    trapezoid_function = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(
        trapezoid_function(finite_values, np.arange(max_buffer_size + 1))
        / max_buffer_size
    )


def compute_budgeted_vus_metrics(
    point_labels: np.ndarray,
    point_scores: np.ndarray,
    max_buffer_size: int,
    num_thresholds: int = 200,
) -> dict[str, dict[str, float]]:
    label_array, score_array = _validate_pointwise_array_shapes(
        point_labels=point_labels, point_scores=point_scores,
    )
    if max_buffer_size < 0:
        raise ValueError("max_buffer_size must be non-negative")
    if len(np.unique(label_array)) < 2:
        return {
            "vus_pr": {str(budget): float("nan") for budget in FPR_BUDGETS},
            "vus_roc": {str(budget): float("nan") for budget in FPR_BUDGETS},
        }
    thresholds = _build_score_thresholds(score_array, num_thresholds)
    pr_values_by_budget = {budget: [] for budget in FPR_BUDGETS}
    roc_values_by_budget = {budget: [] for budget in FPR_BUDGETS}
    for buffer_size in range(max_buffer_size + 1):
        precision_values, recall_values, false_positive_rates, true_positive_rates = _threshold_rates(
            label_array, score_array, thresholds, buffer_size,
        )
        for budget in FPR_BUDGETS:
            pr_values_by_budget[budget].append(
                _constrained_pr_area(
                    precision_values=precision_values,
                    recall_values=recall_values,
                    false_positive_rates=false_positive_rates,
                    budget=budget,
                )
            )
            roc_values_by_budget[budget].append(
                _normalised_partial_roc_area(
                    false_positive_rates=false_positive_rates,
                    true_positive_rates=true_positive_rates,
                    budget=budget,
                )
            )

    return {
        "vus_pr": {
            str(budget): _aggregate_over_buffers(pr_values_by_budget[budget], max_buffer_size)
            for budget in FPR_BUDGETS
        },
        "vus_roc": {
            str(budget): _aggregate_over_buffers(roc_values_by_budget[budget], max_buffer_size)
            for budget in FPR_BUDGETS
        },
    }
