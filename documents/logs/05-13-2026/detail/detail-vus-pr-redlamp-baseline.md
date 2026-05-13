---
date: 2026-05-13 13:38:43 +07 +0700
researcher: TheMetaSetter
git_commit: afba6086047157d72eb96249544402b1d72bc699
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed programming plan for VUS-PR exact-naive metric and RedLamp baseline setup"
tags: [detail, time-series, anomaly-detection, metrics, vus-pr, redlamp, baseline]
status: complete
last_updated: 2026-05-13
last_updated_by: TheMetaSetter
source_plan: documents/logs/05-13-2026/plan/plan-vus-pr-redlamp-baseline.md
---

# Detail: VUS-PR Exact-Naive Metric and RedLamp Baseline Setup

## Objective

This detailed plan implements the plan in `documents/logs/05-13-2026/plan/plan-vus-pr-redlamp-baseline.md`. The goal is to add a threshold-independent VUS-PR metric to the offline anomaly detection evaluation path and use the existing RedLamp MLP baseline as a reproducible comparison point for the thesis model that combines continuous prototypes, discrete prototypes, reconstruction, and multi-class synthetic anomaly classification.

The implementation must preserve the repository's central contracts:

- Data batches expose `batch["x"]: Tensor[B, L, D]`.
- Models expose `outputs["hidden"]: Tensor[B, L, H]` when a thesis-facing hidden state exists.
- Models expose `outputs["point_scores"]: Tensor[B, L]` for offline pointwise anomaly evaluation.
- Evaluation metrics consume merged one-dimensional timeline arrays, not raw overlapping windows.
- Model-specific training and inference logic remains inside the owning model file.

The VUS-PR implementation is evaluation-only. It must not change the data loader, encoder, model forward path, reconstruction objective, classification objective, prototype modules, or online adaptation modules.

## Phase 1: Define Exact-Naive VUS-PR Metric Core

### Phase Summary

This phase adds the mathematical core for exact-naive VUS-PR in `src/metrics/pointwise.py`. This supports the thesis objective by providing a range-aware, threshold-independent metric for comparing RedLamp and the proposed prototype multi-task method on the same SMD timelines.

The implementation follows composition over inheritance: VUS-PR is a set of pure metric helpers composed by `compute_vus_pr_exact_naive`. No new class hierarchy is introduced.

### File-Level Edits

Create `tests/test_vus_pr_metric.py`.

Add:

```python
from __future__ import annotations

import math

import numpy as np

from src.metrics.pointwise import (
    build_threshold_aware_range_labels,
    compute_vus_pr_exact_naive,
    extract_binary_anomaly_ranges,
)
```

Add anomaly range extraction test:

```python
def test_extract_binary_anomaly_ranges_returns_end_exclusive_ranges() -> None:
    labels = np.array([0, 1, 1, 0, 1, 0, 1, 1], dtype=np.int64)

    ranges = extract_binary_anomaly_ranges(labels)

    assert ranges == [(1, 3), (4, 5), (6, 8)]
```

Add threshold-aware label tests:

```python
def test_build_threshold_aware_range_labels_keeps_original_labels_for_zero_buffer() -> None:
    labels = np.array([0, 1, 1, 0], dtype=np.int64)
    predictions = np.array([0, 0, 1, 0], dtype=np.int64)

    range_labels = build_threshold_aware_range_labels(
        point_labels=labels,
        binary_predictions=predictions,
        buffer_size=0,
    )

    assert np.array_equal(range_labels, labels.astype(np.float64))
```

```python
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

Add VUS-PR behavior tests:

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
```

```python
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

Modify `src/metrics/pointwise.py`.

Add import:

```python
import math
```

Add anomaly range extraction above `compute_pointwise_metrics`:

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

Add threshold-aware continuous labels:

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

Add exact-naive VUS-PR:

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
```

```python
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
```

```python
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
    return precision, recall
```

```python
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
```

```python
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

### Interfaces and Contracts

- `extract_binary_anomaly_ranges(point_labels)` accepts any array-like one-dimensional or flattenable label input and returns end-exclusive anomaly ranges.
- `build_threshold_aware_range_labels(point_labels, binary_predictions, buffer_size)` accepts binary labels and binary predictions with identical shapes and returns a continuous `np.float64` label sequence.
- `compute_vus_pr_exact_naive(point_labels, point_scores, max_buffer_size, num_thresholds)` accepts one-dimensional arrays and returns one float.
- These helpers are pure functions. They do not read configs, mutate global state, or depend on PyTorch tensors.

### Design Pattern Application

- Composition over inheritance: small helpers compose into one metric function.
- Strategy pattern: the metric remains a selectable evaluation behavior through evaluator configuration in later phases.
- Registry/factory: no changes are made to the model or dataset registries in this phase.
- Adapter pattern: no encoder adapter changes are needed because the metric consumes evaluator-normalized arrays.

### Risk Mitigation

- Evaluation metric inflation is mitigated by not using point-adjusted metrics and not changing binary predictions after thresholding.
- Prototype redundancy and fusion collapse are not modified in this phase. Existing thesis model diagnostics remain untouched.
- Adaptation contamination and projector drift are not modified because VUS-PR is offline evaluation only.

### Validation

Run:

```bash
pytest -q tests/test_vus_pr_metric.py
```

Acceptance criteria:

- All tests in `tests/test_vus_pr_metric.py` pass.
- Perfect ranking returns `vus_pr == 1.0` within `1e-6`.
- Single-class labels return `nan`.
- The metric code does not import or depend on model classes.

## Phase 2: Integrate VUS-PR Into Pointwise Evaluation

### Phase Summary

This phase adds `vus_pr` to the existing pointwise metric dictionary and exposes evaluator-level parameters. It keeps the engine model-agnostic: models continue to return `point_scores`, and the evaluator continues to own timeline aggregation and metric calls.

### File-Level Edits

Modify `src/metrics/pointwise.py`.

Change `compute_pointwise_metrics` signature:

```python
def compute_pointwise_metrics(
    point_labels: np.ndarray,
    point_scores: np.ndarray,
    threshold: float,
    vus_max_buffer_size: int | None = None,
    vus_num_thresholds: int = 200,
) -> dict[str, float]:
```

Replace the return statement with a local dictionary:

```python
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

Modify `src/engine/evaluator.py`.

Change constructor:

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

Change the metric call:

```python
        metrics = compute_pointwise_metrics(
            point_labels=concatenated_labels,
            point_scores=concatenated_scores,
            threshold=threshold,
            vus_max_buffer_size=self.vus_max_buffer_size,
            vus_num_thresholds=self.vus_num_thresholds,
        )
```

Modify `tests/test_evaluator_thresholding.py`.

Add:

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

Add:

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

### Interfaces and Contracts

- `Evaluator.evaluate(model, data_loader)` remains unchanged for callers.
- `Evaluator.__init__` gains optional VUS settings without breaking existing construction sites.
- `compute_pointwise_metrics` remains backward-compatible because VUS parameters default to disabled.

### Risk Mitigation

- Evaluation metric inflation is mitigated because `vus_pr` is computed independently of the 95th-percentile threshold.
- Existing thresholded metrics remain available for continuity, but `vus_pr` and `pr_auc` become the preferred threshold-independent comparison fields.

### Validation

Run:

```bash
pytest -q tests/test_vus_pr_metric.py tests/test_evaluator_thresholding.py
```

Acceptance criteria:

- Existing evaluator overlap averaging tests still pass.
- `compute_pointwise_metrics` returns `vus_pr` when `vus_max_buffer_size` is provided.
- Existing callers that omit VUS parameters still receive the old metric keys.

## Phase 3: Wire VUS Configuration Through Scripts and Experiment YAML

### Phase Summary

This phase makes VUS-PR reproducible through experiment configs. It applies a strategy-pattern-style configuration: evaluation behavior is selected by an `evaluation` block rather than hard-coded in the metric or model.

### File-Level Edits

Modify `scripts/evaluate.py`.

Replace:

```python
    evaluator = Evaluator(device=experiment_config["device"])
```

with:

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

Modify `configs/experiment/smd_redlamp_mlp_baseline_window20.yaml`.

Add:

```yaml
evaluation:
  vus_max_buffer_size: 20
  vus_num_thresholds: 200
```

Modify `configs/experiment/smd_thesis_multitask_redlamp_multiclass_window20.yaml`.

Add:

```yaml
evaluation:
  vus_max_buffer_size: 20
  vus_num_thresholds: 200
```

### Interfaces and Contracts

- Experiment configs may include an optional `evaluation` dictionary.
- If `evaluation.vus_max_buffer_size` is absent, `scripts/evaluate.py` uses `experiment_config["data"]["window_size"]`.
- If `evaluation.vus_num_thresholds` is absent, `scripts/evaluate.py` uses `200`.
- This keeps the registry/factory pattern intact: datasets and models are still built through `build_dataset` and `build_model`.

### Risk Mitigation

- Baseline comparability is protected because both RedLamp and thesis configs specify the same VUS parameters.
- Metric inflation from inconsistent buffer sizes is reduced by storing the chosen values in the resolved experiment config.

### Validation

Run:

```bash
pytest -q tests/test_config_loading.py tests/test_evaluator_thresholding.py
```

Acceptance criteria:

- Config loading accepts both updated experiment files.
- `scripts/evaluate.py` constructs `Evaluator` with VUS settings.
- Resolved configs written by evaluation include the `evaluation` block.

## Phase 4: Extend Ablation and Baseline Summary Artifacts

### Phase Summary

This phase ensures RedLamp and thesis comparison artifacts expose `vus_pr` in a compact table. It supports thesis reporting by making VUS-PR visible next to ROC-AUC, PR-AUC, F1, threshold, and training statistics.

### File-Level Edits

Modify `tests/test_ablation_runner.py`.

Change `fake_evaluate` to:

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

Add:

```python
    assert outputs["summary_rows"][0]["pr_auc"] == 0.65
    assert outputs["summary_rows"][0]["vus_pr"] == 0.61
```

Modify `scripts/run_ablation.py`.

Add fields near `roc_auc`:

```python
        "pr_auc": evaluation_outputs["metrics"].get("pr_auc"),
        "vus_pr": evaluation_outputs["metrics"].get("vus_pr"),
```

Keep the existing fallback logic for `roc_auc` and `f1`.

### Interfaces and Contracts

- `run_ablation_suite` still returns `summary_rows`, `summary_json_path`, and `summary_csv_path`.
- Summary rows gain two optional fields: `pr_auc` and `vus_pr`.
- Existing tests and downstream consumers remain compatible because fields are additive.

### Risk Mitigation

- Evaluation metric inflation is mitigated by showing both thresholded and threshold-independent metrics.
- Prototype redundancy and fusion collapse are monitored through existing thesis metrics such as `final_train_alpha`, `final_train_beta`, and `final_train_discrete_usage_concentration`.

### Validation

Run:

```bash
pytest -q tests/test_ablation_runner.py::test_run_ablation_suite_writes_compact_summary_artifacts
```

Acceptance criteria:

- `ablation_summary.json` and `ablation_summary.csv` include `pr_auc` and `vus_pr`.
- Existing summary fields remain present.

## Phase 5: Verify RedLamp Baseline and Thesis Multi-Task Protocol

### Phase Summary

This phase does not add new model code. It validates that the existing RedLamp MLP baseline and thesis multi-task model can be compared under one evaluation protocol with VUS-PR enabled.

### File-Level Edits

No source file edits are required if Phases 1 through 4 are complete.

### Interfaces and Contracts

The RedLamp baseline contract is:

- `src/models/redlamp_mlp_baseline.py` accepts `batch["x"]: Tensor[B, L, D]`.
- It exposes `outputs["hidden"]: Tensor[B, L, latent_dim]` by expanding the latent representation across timesteps.
- It exposes `outputs["recon"]: Tensor[B, L, D]`.
- It exposes `outputs["logits"]: Tensor[B, 12]`.
- It exposes `outputs["point_scores"]: Tensor[B, L]`.

The thesis multi-task contract is:

- `src/models/thesis_multitask.py` accepts `batch["x"]: Tensor[B, L, D]`.
- It exposes the thesis hidden state `outputs["hidden"]: Tensor[B, L, H]`.
- It uses task-specific fusion for reconstruction and classification.
- It exposes `outputs["point_scores"]: Tensor[B, L]` for evaluation.

The data contract is:

- SMD windows are produced by the config-driven data stack.
- Evaluation merges overlapping scores back to entity timelines before computing metrics.

### Commands

Run RedLamp preflight:

```bash
python scripts/run_multiseed_experiments.py \
  --config-paths configs/experiment/smd_redlamp_mlp_baseline_window20.yaml \
  --preflight-only
```

Run thesis preflight:

```bash
python scripts/run_multiseed_experiments.py \
  --config-paths configs/experiment/smd_thesis_multitask_redlamp_multiclass_window20.yaml \
  --preflight-only
```

Run smoke-relevant tests:

```bash
pytest -q tests/test_vus_pr_metric.py tests/test_evaluator_thresholding.py tests/test_redlamp_mlp_baseline.py tests/test_multitask_shapes.py tests/test_one_redlamp_mlp_train_step.py tests/test_one_multitask_train_step.py
```

Train RedLamp:

```bash
python scripts/train.py \
  --experiment-config configs/experiment/smd_redlamp_mlp_baseline_window20.yaml
```

Evaluate RedLamp:

```bash
python scripts/evaluate.py \
  --experiment-config configs/experiment/smd_redlamp_mlp_baseline_window20.yaml \
  --checkpoint-path outputs/smd_redlamp_mlp_baseline_window20/checkpoints/best.pt
```

Train thesis multi-task:

```bash
python scripts/train.py \
  --experiment-config configs/experiment/smd_thesis_multitask_redlamp_multiclass_window20.yaml
```

Evaluate thesis multi-task:

```bash
python scripts/evaluate.py \
  --experiment-config configs/experiment/smd_thesis_multitask_redlamp_multiclass_window20.yaml \
  --checkpoint-path outputs/smd_thesis_multitask_redlamp_multiclass_window20/checkpoints/best.pt
```

Produce comparison summary:

```bash
python scripts/run_ablation.py \
  --experiment-config configs/experiment/smd_redlamp_mlp_baseline_window20.yaml \
  --experiment-config configs/experiment/smd_thesis_multitask_redlamp_multiclass_window20.yaml \
  --summary-output-dir outputs/redlamp_vs_thesis_vus_pr_summary
```

### Risk Mitigation

- Prototype redundancy: compare RedLamp against thesis logs containing prototype usage concentration, fusion values, and validation metrics.
- Fusion collapse: inspect `final_train_alpha`, `final_train_beta`, and prototype usage fields in the ablation summary and raw `metrics.jsonl`.
- Adaptation contamination: not applicable to this offline baseline phase; do not run online adaptation as part of this task.
- Projector drift: not applicable to this offline baseline phase; the online projector remains untouched.
- Evaluation metric inflation: rely on `vus_pr` and `pr_auc` for threshold-independent reporting, and label thresholded F1/precision/recall as thresholded metrics.

### Acceptance Criteria

- `outputs/smd_redlamp_mlp_baseline_window20/evaluation_metrics.json` contains `vus_pr`.
- `outputs/smd_thesis_multitask_redlamp_multiclass_window20/evaluation_metrics.json` contains `vus_pr`.
- `outputs/redlamp_vs_thesis_vus_pr_summary/ablation_summary.json` contains two rows.
- Each row contains `experiment_name`, `roc_auc`, `pr_auc`, `vus_pr`, `f1`, `threshold`, and relevant training statistics.
- Both models are evaluated through `scripts/evaluate.py`, not separate notebooks.

## Phase 6: Documentation and Reproducibility Checks

### Phase Summary

This phase records exactly what was implemented and confirms that all artifacts needed for thesis reporting are generated by the codebase rather than ad hoc notebook state.

### File-Level Edits

If implementation changes are made, update this detail file or create a follow-up detail note under `documents/logs/05-13-2026/detail/` with:

- final test commands;
- final output paths;
- observed metric keys;
- any deviation from exact-naive behavior.

Do not update design documents unless the directory tree or core contracts change.

### Validation

Run:

```bash
pytest -q tests/test_vus_pr_metric.py tests/test_evaluator_thresholding.py tests/test_ablation_runner.py
```

Run:

```bash
pytest -q tests/test_redlamp_mlp_baseline.py tests/test_multitask_shapes.py tests/test_one_redlamp_mlp_train_step.py tests/test_one_multitask_train_step.py
```

Run:

```bash
python scripts/run_multiseed_experiments.py \
  --config-paths \
  configs/experiment/smd_redlamp_mlp_baseline_window20.yaml \
  configs/experiment/smd_thesis_multitask_redlamp_multiclass_window20.yaml \
  --preflight-only
```

### Acceptance Criteria

- The test suite above passes.
- The implementation remains within the existing `src/metrics`, `src/engine`, `scripts`, `configs`, and `tests` ownership boundaries.
- No model file is modified solely for VUS-PR.
- No point-adjusted metric is introduced.
- The evaluation outputs are reproducible from YAML configs and checkpoint paths.

## Overall Acceptance Criteria

- `compute_vus_pr_exact_naive` exists in `src/metrics/pointwise.py`.
- `compute_pointwise_metrics` can emit `vus_pr`.
- `Evaluator` accepts VUS configuration without breaking existing evaluator usage.
- `scripts/evaluate.py` passes VUS settings from `experiment_config["evaluation"]`.
- RedLamp and thesis RedLamp multi-class experiment configs explicitly define VUS settings.
- `scripts/run_ablation.py` includes `vus_pr` and `pr_auc` in summary rows.
- Unit tests cover anomaly range extraction, continuous range labels, perfect-score VUS-PR, single-class `nan`, evaluator integration, and ablation summary fields.
- Long baseline commands use the same training and evaluation scripts for RedLamp and the thesis model.

## Explicit Non-Goals

- Do not implement optimized VUS algorithms in this task.
- Do not refactor threshold calibration in this task.
- Do not change RedLamp synthetic anomaly taxonomy.
- Do not alter prototype memory updates, continuous prototypes, discrete prototypes, or fusion logic.
- Do not run online adaptation or modify projector behavior.
- Do not introduce a new metrics package or external dependency.

## Commit Sequence

Use short imperative commit messages consistent with repository history:

1. `Add exact naive VUS PR metric`
2. `Report VUS PR in evaluation metrics`
3. `Configure VUS PR evaluation settings`
4. `Summarize VUS PR in ablation runs`

If the implementation is performed in one session without intermediate commits, still keep changes grouped by the same conceptual boundaries for review.
