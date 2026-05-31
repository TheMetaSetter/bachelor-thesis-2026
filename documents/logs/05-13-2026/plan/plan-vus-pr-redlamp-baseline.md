---
date: 2026-05-13 13:31:34 +07 +0700
researcher: TheMetaSetter
git_commit: afba6086047157d72eb96249544402b1d72bc699
branch: dev
repository: bachelor-thesis-2026
topic: "VUS-PR exact-naive metric and RedLamp baseline setup"
tags: [plan, time-series, anomaly-detection, metrics, vus-pr, redlamp, baseline]
status: complete
last_updated: 2026-05-13
last_updated_by: TheMetaSetter
---

# VUS-PR Exact-Naive Metric and RedLamp Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an exact-naive VUS-PR metric to the offline evaluation path and prepare a reproducible RedLamp baseline comparison against the prototype-based multi-task thesis model.

**Architecture:** Keep metric computation in `src/metrics/pointwise.py` so evaluation remains model-agnostic. Let `src/engine/evaluator.py` pass explicit VUS configuration after it merges overlapping window scores back to original timelines. Reuse the existing RedLamp MLP baseline and thesis RedLamp multi-class configs, then extend the ablation summary so baseline runs report `pr_auc`, `vus_pr`, thresholded metrics, and training statistics in one artifact.

**Tech Stack:** Python, NumPy, PyTorch, scikit-learn, YAML configs, Pytest, existing registry, trainer, evaluator, and logger infrastructure.

---

## Current State

- `documents/logs/05-12-2026/research/research-vus-pr-metric-and-redlamp-baseline-planning.md` documents that VUS-PR is not implemented.
- `src/metrics/pointwise.py:117` defines `compute_pointwise_metrics`, which currently returns `roc_auc`, `pr_auc`, thresholded precision, recall, F1, and false positive rate.
- `src/metrics/pointwise.py:125` contains the explicit implementation note that point-adjusted metrics must not be used and VUS-PR should be added.
- `src/engine/evaluator.py:218` concatenates point scores after merging overlapping windows back to entity timelines.
- `src/engine/evaluator.py:229` calls `compute_pointwise_metrics`, so VUS-PR can be added without changing model files.
- `src/models/redlamp_mlp_baseline.py` already defines the RedLamp-inspired MLP baseline with reconstruction and twelve-class synthetic anomaly classification.
- `configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml` and `configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20__w20__seed11__default.yaml` already define the two comparison experiments.
- `scripts/run_ablation.py:31` builds compact comparison rows, but it does not yet include `pr_auc` or `vus_pr`.

## Design Options

- **Option A: Add only `vus_pr` inside `compute_pointwise_metrics`.** This is the smallest change, but it hides important VUS parameters such as maximum buffer length and number of thresholds.
- **Option B: Add a dedicated exact-naive VUS helper plus evaluator-level configuration.** This is the recommended option because it keeps mathematical logic in the metric module while keeping experiment-specific defaults in evaluation.
- **Option C: Implement the optimized VUS algorithm from the paper immediately.** This is premature for this repository because the exact-naive version is easier to test, easier to read, and sufficient for baseline planning before scaling to many seeds.

## Recommended Design

Implement Option B.

The first version computes a global `vus_pr` value on `concatenated_labels` and `concatenated_scores`, matching the current global style of `pr_auc`. The default `max_buffer_size` is the experiment data window size, because the current RedLamp comparison uses window-20 configs and the metric should tolerate boundary lag up to the scoring window. A later ablation can compare `max_buffer_size = window_size // 2`.

The metric function should be exact-naive in this sense:

1. Iterate through every integer buffer length from `0` to `max_buffer_size`.
2. Iterate through every threshold in a deterministic threshold grid.
3. Build thresholded predictions.
4. Build the threshold-aware continuous range label for that buffer.
5. Compute range precision and range recall.
6. Compute the area under the PR curve for that buffer.
7. Average the AP values across all buffer lengths.

## Contract Decisions

- Batch contract remains unchanged: `batch["x"]: Tensor[B, L, D]`, optional `point_labels`, `mask`, `timestamps`, and `meta`.
- Model output contract remains unchanged: each model exposes `outputs["point_scores"]: Tensor[B, L]`.
- Encoder contract remains unchanged: thesis-facing models expose `outputs["hidden"]: Tensor[B, L, H]`.
- VUS-PR is an evaluation-only metric and must not affect training losses or model forward paths.
- VUS-PR must consume one-dimensional NumPy arrays: `point_labels: np.ndarray` and `point_scores: np.ndarray`.
- VUS-PR must return `float("nan")` when labels contain only one class, matching the repository's safe metric style.

## Risk and Mitigation

- **Risk: VUS-PR is confused with point-adjusted F1.** Mitigation: keep VUS-PR threshold-independent and do not alter predictions after detection.
- **Risk: The implementation silently depends on the test threshold.** Mitigation: compute VUS-PR from score thresholds internal to the metric and do not use `select_point_score_threshold`.
- **Risk: The metric is too slow on long timelines.** Mitigation: start exact-naive for correctness, expose `num_thresholds`, and add an optimized implementation only after tests lock the reference behavior.
- **Risk: RedLamp and thesis summaries omit the new metric.** Mitigation: add `vus_pr` and `pr_auc` to `scripts/run_ablation.py`.
- **Risk: Baseline runs are not comparable.** Mitigation: run RedLamp MLP and thesis RedLamp multi-class configs through the same train, evaluate, and ablation summary codepath.

## Files To Create Or Modify

- Modify `src/metrics/pointwise.py`: add exact-naive VUS-PR helper functions and include `vus_pr` in `compute_pointwise_metrics`.
- Modify `src/engine/evaluator.py`: pass `vus_max_buffer_size` and `vus_num_thresholds` into `compute_pointwise_metrics`.
- Modify `scripts/evaluate.py`: allow optional VUS settings to flow from experiment config to evaluator.
- Modify `scripts/run_ablation.py`: include `pr_auc` and `vus_pr` in summary rows.
- Modify `tests/test_evaluator_thresholding.py`: add VUS-PR integration tests.
- Create `tests/test_vus_pr_metric.py`: unit tests for exact-naive VUS-PR helpers.
- Optionally modify `configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml`: add `evaluation.vus_max_buffer_size: 20` and `evaluation.vus_num_thresholds: 200`.
- Optionally modify `configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20__w20__seed11__default.yaml`: add the same evaluation block.

---

### Task 1: Add Exact-Naive VUS-PR Unit Tests

**Files:**
- Create: `tests/test_vus_pr_metric.py`
- Modify: `src/metrics/pointwise.py`

- [ ] **Step 1: Write failing tests for anomaly-range extraction**

Create `tests/test_vus_pr_metric.py`:

```python
from __future__ import annotations

import math

import numpy as np

from src.metrics.pointwise import (
    compute_vus_pr_exact_naive,
    extract_binary_anomaly_ranges,
)


def test_extract_binary_anomaly_ranges_returns_end_exclusive_ranges() -> None:
    labels = np.array([0, 1, 1, 0, 1, 0, 1, 1], dtype=np.int64)

    ranges = extract_binary_anomaly_ranges(labels)

    assert ranges == [(1, 3), (4, 5), (6, 8)]
```

- [ ] **Step 2: Run the test and verify failure**

Run:

```bash
pytest -q tests/test_vus_pr_metric.py::test_extract_binary_anomaly_ranges_returns_end_exclusive_ranges
```

Expected result: fail with `ImportError` because `extract_binary_anomaly_ranges` does not exist.

- [ ] **Step 3: Implement `extract_binary_anomaly_ranges`**

Add this function to `src/metrics/pointwise.py` above `compute_pointwise_metrics`:

```python
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
```

- [ ] **Step 4: Run the test and verify pass**

Run:

```bash
pytest -q tests/test_vus_pr_metric.py::test_extract_binary_anomaly_ranges_returns_end_exclusive_ranges
```

Expected result: pass.

---

### Task 2: Implement Threshold-Aware Continuous Range Labels

**Files:**
- Modify: `src/metrics/pointwise.py`
- Modify: `tests/test_vus_pr_metric.py`

- [ ] **Step 1: Write failing tests for zero-buffer and positive-buffer labels**

Append these tests to `tests/test_vus_pr_metric.py`:

```python
from src.metrics.pointwise import build_threshold_aware_range_labels


def test_build_threshold_aware_range_labels_keeps_original_labels_for_zero_buffer() -> None:
    labels = np.array([0, 1, 1, 0], dtype=np.int64)
    predictions = np.array([0, 0, 1, 0], dtype=np.int64)

    range_labels = build_threshold_aware_range_labels(
        point_labels=labels,
        binary_predictions=predictions,
        buffer_size=0,
    )

    assert np.array_equal(range_labels, labels.astype(np.float64))


def test_build_threshold_aware_range_labels_extends_only_predicted_buffer_side() -> None:
    labels = np.array([0, 0, 1, 1, 0, 0], dtype=np.int64)
    predictions = np.array([0, 1, 0, 0, 0, 0], dtype=np.int64)

    range_labels = build_threshold_aware_range_labels(
        point_labels=labels,
        binary_predictions=predictions,
        buffer_size=2,
    )

    assert range_labels[2] == 1.0
    assert range_labels[3] == 1.0
    assert range_labels[1] > 0.0
    assert range_labels[0] == 0.0
    assert range_labels[4] == 0.0
    assert range_labels[5] == 0.0
```

- [ ] **Step 2: Run the new tests and verify failure**

Run:

```bash
pytest -q tests/test_vus_pr_metric.py::test_build_threshold_aware_range_labels_keeps_original_labels_for_zero_buffer tests/test_vus_pr_metric.py::test_build_threshold_aware_range_labels_extends_only_predicted_buffer_side
```

Expected result: fail with `ImportError` because `build_threshold_aware_range_labels` does not exist.

- [ ] **Step 3: Implement the range-label helper**

Add these functions to `src/metrics/pointwise.py`:

```python
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
```

Also add `import math` near the top of `src/metrics/pointwise.py`.

- [ ] **Step 4: Run the range-label tests and verify pass**

Run:

```bash
pytest -q tests/test_vus_pr_metric.py::test_build_threshold_aware_range_labels_keeps_original_labels_for_zero_buffer tests/test_vus_pr_metric.py::test_build_threshold_aware_range_labels_extends_only_predicted_buffer_side
```

Expected result: pass.

---

### Task 3: Implement Range Precision, Range Recall, and Exact-Naive VUS-PR

**Files:**
- Modify: `src/metrics/pointwise.py`
- Modify: `tests/test_vus_pr_metric.py`

- [ ] **Step 1: Write failing tests for exact-naive VUS-PR behavior**

Append these tests to `tests/test_vus_pr_metric.py`:

```python
def test_compute_vus_pr_exact_naive_returns_one_for_perfect_scores() -> None:
    labels = np.array([0, 1, 1, 0, 0, 1], dtype=np.int64)
    scores = np.array([0.0, 0.9, 0.8, 0.1, 0.2, 0.95], dtype=np.float64)

    vus_pr = compute_vus_pr_exact_naive(
        point_labels=labels,
        point_scores=scores,
        max_buffer_size=2,
        num_thresholds=20,
    )

    assert math.isclose(vus_pr, 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_compute_vus_pr_exact_naive_returns_nan_for_single_class_labels() -> None:
    labels = np.array([0, 0, 0, 0], dtype=np.int64)
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    vus_pr = compute_vus_pr_exact_naive(
        point_labels=labels,
        point_scores=scores,
        max_buffer_size=2,
        num_thresholds=20,
    )

    assert math.isnan(vus_pr)
```

- [ ] **Step 2: Run the tests and verify failure**

Run:

```bash
pytest -q tests/test_vus_pr_metric.py::test_compute_vus_pr_exact_naive_returns_one_for_perfect_scores tests/test_vus_pr_metric.py::test_compute_vus_pr_exact_naive_returns_nan_for_single_class_labels
```

Expected result: fail with `ImportError` or `NameError` because `compute_vus_pr_exact_naive` is incomplete.

- [ ] **Step 3: Implement exact-naive VUS-PR**

Add these functions to `src/metrics/pointwise.py`:

```python
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
    if predicted_positive == 0.0:
        precision = 1.0
    else:
        precision = true_positive / predicted_positive

    positive_mass = float(np.sum((point_labels.astype(np.float64) + range_labels) / 2.0))
    if positive_mass == 0.0:
        recall = float("nan")
    else:
        existence_reward = _compute_existence_reward(point_labels, binary_predictions)
        recall = (true_positive / positive_mass) * existence_reward
    return precision, recall


def _compute_pr_area_from_points(
    precision_values: list[float],
    recall_values: list[float],
) -> float:
    finite_points = [
        (recall, precision)
        for recall, precision in zip(recall_values, precision_values)
        if np.isfinite(recall) and np.isfinite(precision)
    ]
    if not finite_points:
        return float("nan")

    finite_points.sort(key=lambda pair: pair[0])
    sorted_recalls = np.array([pair[0] for pair in finite_points], dtype=np.float64)
    sorted_precisions = np.array([pair[1] for pair in finite_points], dtype=np.float64)
    return float(np.trapz(sorted_precisions, sorted_recalls))


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
```

- [ ] **Step 4: Run VUS-PR unit tests**

Run:

```bash
pytest -q tests/test_vus_pr_metric.py
```

Expected result: all tests in `tests/test_vus_pr_metric.py` pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/metrics/pointwise.py tests/test_vus_pr_metric.py
git commit -m "Add exact naive VUS PR metric"
```

---

### Task 4: Add VUS-PR To Pointwise Metrics and Evaluator Configuration

**Files:**
- Modify: `src/metrics/pointwise.py`
- Modify: `src/engine/evaluator.py`
- Modify: `tests/test_evaluator_thresholding.py`

- [ ] **Step 1: Write failing integration test for `compute_pointwise_metrics`**

Append to `tests/test_evaluator_thresholding.py`:

```python
def test_compute_pointwise_metrics_includes_vus_pr() -> None:
    point_labels = np.array([0, 1, 1, 0, 0, 1], dtype=np.int64)
    point_scores = np.array([0.0, 0.9, 0.8, 0.1, 0.2, 0.95], dtype=np.float32)

    metrics = compute_pointwise_metrics(
        point_labels=point_labels,
        point_scores=point_scores,
        threshold=0.5,
        vus_max_buffer_size=2,
        vus_num_thresholds=20,
    )

    assert "vus_pr" in metrics
    assert metrics["vus_pr"] >= 0.0
    assert metrics["vus_pr"] <= 1.0
```

- [ ] **Step 2: Run the test and verify failure**

Run:

```bash
pytest -q tests/test_evaluator_thresholding.py::test_compute_pointwise_metrics_includes_vus_pr
```

Expected result: fail because `compute_pointwise_metrics` does not accept VUS arguments.

- [ ] **Step 3: Extend `compute_pointwise_metrics` signature and return value**

Modify `compute_pointwise_metrics` in `src/metrics/pointwise.py`:

```python
def compute_pointwise_metrics(
    point_labels: np.ndarray,
    point_scores: np.ndarray,
    threshold: float,
    vus_max_buffer_size: int | None = None,
    vus_num_thresholds: int = 200,
) -> dict[str, float]:
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
```

Remove the stale VUS implementation comments once the metric is implemented.

- [ ] **Step 4: Add evaluator configuration fields**

Modify `Evaluator.__init__` in `src/engine/evaluator.py`:

```python
class Evaluator:
    def __init__(
        self,
        device: str = "cpu",
        vus_max_buffer_size: int | None = None,
        vus_num_thresholds: int = 200,
    ) -> None:
        self.device = device
        self.vus_max_buffer_size = vus_max_buffer_size
        self.vus_num_thresholds = vus_num_thresholds
```

Modify the `compute_pointwise_metrics` call:

```python
metrics = compute_pointwise_metrics(
    point_labels=concatenated_labels,
    point_scores=concatenated_scores,
    threshold=threshold,
    vus_max_buffer_size=self.vus_max_buffer_size,
    vus_num_thresholds=self.vus_num_thresholds,
)
```

- [ ] **Step 5: Run evaluator tests**

Run:

```bash
pytest -q tests/test_evaluator_thresholding.py tests/test_vus_pr_metric.py
```

Expected result: all tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/metrics/pointwise.py src/engine/evaluator.py tests/test_evaluator_thresholding.py tests/test_vus_pr_metric.py
git commit -m "Report VUS PR in evaluation metrics"
```

---

### Task 5: Wire VUS Settings Through Evaluation Entrypoint and Experiment Configs

**Files:**
- Modify: `scripts/evaluate.py`
- Modify: `configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml`
- Modify: `configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20__w20__seed11__default.yaml`
- Modify: `tests/test_evaluator_thresholding.py`

- [ ] **Step 1: Write failing test for evaluator config construction**

Append to `tests/test_evaluator_thresholding.py`:

```python
def test_evaluator_accepts_vus_configuration() -> None:
    evaluator = Evaluator(
        device="cpu",
        vus_max_buffer_size=20,
        vus_num_thresholds=50,
    )

    assert evaluator.vus_max_buffer_size == 20
    assert evaluator.vus_num_thresholds == 50
```

- [ ] **Step 2: Run the test and verify pass or targeted failure**

Run:

```bash
pytest -q tests/test_evaluator_thresholding.py::test_evaluator_accepts_vus_configuration
```

Expected result: pass if Task 4 is complete.

- [ ] **Step 3: Read evaluation config inside `scripts/evaluate.py`**

Modify `run_evaluation_experiment` in `scripts/evaluate.py` before evaluator creation:

```python
    evaluation_config = dict(experiment_config.get("evaluation", {}))
    vus_max_buffer_size = evaluation_config.get(
        "vus_max_buffer_size",
        experiment_config["data"].get("window_size"),
    )
    vus_num_thresholds = int(evaluation_config.get("vus_num_thresholds", 200))
    evaluator = Evaluator(
        device=experiment_config["device"],
        vus_max_buffer_size=vus_max_buffer_size,
        vus_num_thresholds=vus_num_thresholds,
    )
```

Replace the existing `Evaluator(device=experiment_config["device"])` construction.

- [ ] **Step 4: Add explicit evaluation blocks to the two comparison configs**

Add to `configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml`:

```yaml
evaluation:
  vus_max_buffer_size: 20
  vus_num_thresholds: 200
```

Add the same block to `configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20__w20__seed11__default.yaml`.

- [ ] **Step 5: Run focused tests**

Run:

```bash
pytest -q tests/test_evaluator_thresholding.py tests/test_config_loading.py
```

Expected result: all tests pass and config loading accepts the new `evaluation` block.

- [ ] **Step 6: Commit**

Run:

```bash
git add scripts/evaluate.py configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20__w20__seed11__default.yaml tests/test_evaluator_thresholding.py
git commit -m "Configure VUS PR evaluation settings"
```

---

### Task 6: Add VUS-PR To Ablation and Baseline Summary Artifacts

**Files:**
- Modify: `scripts/run_ablation.py`
- Modify: `tests/test_ablation_runner.py`

- [ ] **Step 1: Write failing ablation summary test**

Modify the fake evaluation metrics in `tests/test_ablation_runner.py`:

```python
    def fake_evaluate(
        experiment_config: dict[str, object], checkpoint_path: str
    ) -> dict[str, object]:
        return {
            "metrics": {
                "roc_auc": 0.75,
                "pr_auc": 0.65,
                "vus_pr": 0.61,
                "f1": 0.5,
                "threshold": 0.12,
            }
        }
```

Add assertions:

```python
    assert outputs["summary_rows"][0]["pr_auc"] == 0.65
    assert outputs["summary_rows"][0]["vus_pr"] == 0.61
```

- [ ] **Step 2: Run the test and verify failure**

Run:

```bash
pytest -q tests/test_ablation_runner.py::test_run_ablation_suite_writes_compact_summary_artifacts
```

Expected result: fail because summary rows do not include `pr_auc` or `vus_pr`.

- [ ] **Step 3: Extend `_build_summary_row`**

Modify `scripts/run_ablation.py` inside the returned dictionary:

```python
        "pr_auc": evaluation_outputs["metrics"].get("pr_auc"),
        "vus_pr": evaluation_outputs["metrics"].get("vus_pr"),
```

Place these fields next to `roc_auc` and `f1`.

- [ ] **Step 4: Run ablation test**

Run:

```bash
pytest -q tests/test_ablation_runner.py::test_run_ablation_suite_writes_compact_summary_artifacts
```

Expected result: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add scripts/run_ablation.py tests/test_ablation_runner.py
git commit -m "Summarize VUS PR in ablation runs"
```

---

### Task 7: Baseline Preflight Commands for RedLamp and Thesis Comparison

**Files:**
- No source code changes if earlier tasks are complete.

- [ ] **Step 1: Validate the RedLamp baseline config**

Run:

```bash
python scripts/run_multiseed_experiments.py \
  --config-paths configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml \
  --preflight-only
```

Expected result: command exits successfully after validating config paths, dataset root, output directory, and checkpoint directory.

- [ ] **Step 2: Validate the thesis multi-task config**

Run:

```bash
python scripts/run_multiseed_experiments.py \
  --config-paths configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20__w20__seed11__default.yaml \
  --preflight-only
```

Expected result: command exits successfully.

- [ ] **Step 3: Run one smoke-level evaluation test suite before launching long training**

Run:

```bash
pytest -q tests/test_vus_pr_metric.py tests/test_evaluator_thresholding.py tests/test_redlamp_mlp_baseline.py tests/test_multitask_shapes.py tests/test_one_redlamp_mlp_train_step.py tests/test_one_multitask_train_step.py
```

Expected result: all tests pass.

- [ ] **Step 4: Launch RedLamp baseline training**

Run:

```bash
python scripts/train.py \
  --experiment-config configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml
```

Expected result: training completes and writes `outputs/smd_redlamp_mlp_baseline_window20/checkpoints/best.pt`.

- [ ] **Step 5: Evaluate RedLamp baseline with VUS-PR**

Run:

```bash
python scripts/evaluate.py \
  --experiment-config configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml \
  --checkpoint-path outputs/smd_redlamp_mlp_baseline_window20/checkpoints/best.pt
```

Expected result: `outputs/smd_redlamp_mlp_baseline_window20/evaluation_metrics.json` contains `pr_auc` and `vus_pr`.

- [ ] **Step 6: Launch thesis multi-task training**

Run:

```bash
python scripts/train.py \
  --experiment-config configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20__w20__seed11__default.yaml
```

Expected result: training completes and writes `outputs/smd_thesis_multitask_redlamp_multiclass_window20/checkpoints/best.pt`.

- [ ] **Step 7: Evaluate thesis multi-task model with VUS-PR**

Run:

```bash
python scripts/evaluate.py \
  --experiment-config configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20__w20__seed11__default.yaml \
  --checkpoint-path outputs/smd_thesis_multitask_redlamp_multiclass_window20/checkpoints/best.pt
```

Expected result: `outputs/smd_thesis_multitask_redlamp_multiclass_window20/evaluation_metrics.json` contains `pr_auc` and `vus_pr`.

- [ ] **Step 8: Produce a compact comparison summary**

Run:

```bash
python scripts/run_ablation.py \
  --experiment-config configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml \
  --experiment-config configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20__w20__seed11__default.yaml \
  --summary-output-dir outputs/redlamp_vs_thesis_vus_pr_summary
```

Expected result: `outputs/redlamp_vs_thesis_vus_pr_summary/ablation_summary.json` and `.csv` contain rows with `roc_auc`, `pr_auc`, `vus_pr`, `f1`, `threshold`, and training statistics.

---

## Validation Procedure

Run the focused metric and evaluator tests:

```bash
pytest -q tests/test_vus_pr_metric.py tests/test_evaluator_thresholding.py tests/test_ablation_runner.py
```

Run the baseline and thesis smoke-relevant tests:

```bash
pytest -q tests/test_redlamp_mlp_baseline.py tests/test_multitask_shapes.py tests/test_one_redlamp_mlp_train_step.py tests/test_one_multitask_train_step.py
```

Run config preflight before long experiments:

```bash
python scripts/run_multiseed_experiments.py \
  --config-paths \
  configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml \
  configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20__w20__seed11__default.yaml \
  --preflight-only
```

## Self-Review

- Spec coverage: The plan covers exact-naive VUS-PR implementation, evaluator integration, ablation summary reporting, RedLamp baseline setup, and thesis multi-task comparison.
- Placeholder scan: No implementation step depends on unspecified placeholders or unspecified file paths.
- Type consistency: VUS-PR helpers consistently consume one-dimensional NumPy arrays and return Python floats. Evaluator configuration uses `int | None` for `vus_max_buffer_size` and `int` for `vus_num_thresholds`.
- Scope check: The plan intentionally excludes optimized VUS algorithms and threshold calibration refactoring. Those should be separate plans after exact-naive VUS-PR is validated.
