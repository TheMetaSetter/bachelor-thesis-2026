---
date: 2026-05-17 21:30:58 +07+0700
researcher: TheMetaSetter
git_commit: 5012240e27ce4ad2b1ee613e3c328a4b26b685a2
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed implementation plan for VUS-PR computation on synthetic anomalous validation windows"
tags: [detail, time-series, anomaly-detection, multi-class, validation]
status: draft
last_updated: 2026-05-17
last_updated_by: TheMetaSetter
research_source: documents/logs/05-17-2026/research/research-codebase-state-before-planning-training-optimization-runtime-support.md
plan_source: documents/logs/05-17-2026/plan/plan-synthetic-validation-vus-pr-runtime-support.md
---

# Detailed Plan: Synthetic Validation VUS-PR Runtime Support

## Objective

Implement a readable, configuration-compatible, and thesis-aligned runtime extension that computes `val_synth_vus_pr` on validation windows after synthetic anomaly injection. The implementation must preserve the current offline training contracts, reuse the repository's existing overlap-aware point-score reconstruction semantics, and allow future checkpoint selection from `val_synth_vus_pr` without changing cosine learning-rate scheduling behavior.

The completed implementation must make it possible to:

1. compute synthetic validation pointwise metrics from the same validation windows already used by `synthetic_validation_step()`;
2. reconstruct overlapping synthetic validation windows back to an entity-level synthetic validation timeline;
3. log `val_synth_vus_pr` and related pointwise metrics on every epoch when synthetic validation is enabled;
4. support `checkpoint_monitor_metric: val_synth_vus_pr`;
5. preserve the existing distinction that cosine scheduling requires no monitored metric.

## Scope Boundaries

### In scope

- Synthetic validation pointwise metric computation.
- Reuse of overlap-aware reconstruction for synthetic validation windows.
- A stable batch-level field for synthetic pointwise anomaly targets if needed.
- Epoch-level logging of `val_synth_vus_pr` and supporting `val_synth_*` pointwise metrics.
- Checkpoint monitor support for `val_synth_vus_pr`.
- Focused tests and smoke validation for the synthetic validation VUS path.

### Out of scope

- Changing the semantics of the clean SMD validation split.
- Replacing or removing `val_vus_pr` on clean validation.
- Adding metric-driven scheduler logic to cosine scheduling.
- Changing model architectures, prototype mechanisms, or online adaptation logic.
- Refactoring the entire evaluation stack into a generalized plugin framework.

## Stable Interfaces and Design Rules

### Batch contract

The batch contract remains a dictionary with the established offline shape contract:

```python
batch = {
    "x": Tensor[B, L, D],
    "point_labels": Tensor[B, L] | None,
    "mask": Tensor[B, L, D] | None,
    "timestamps": Tensor[B, L] | None,
    "meta": list[dict[str, object]],
}
```

For synthetic validation support, the detailed plan allows one additional explicit field when needed:

```python
batch["synthetic_anomaly_mask"]: Tensor[B, L]
```

This field must represent the pointwise synthetic anomaly target used for pointwise synthetic validation metrics. It must not replace `point_labels`; it is a complementary explicit field whose purpose is to preserve the existing clean-versus-synthetic distinction.

### Encoder contract

No change is permitted to the thesis-facing hidden-state contract. Models continue to expose their existing `hidden` representation and associated outputs. Synthetic validation VUS computation must consume already available `point_scores` and pointwise labels rather than introducing a second hidden-state or scoring contract.

### Model output contract

The public model output schema remains unchanged:

- `hidden`
- `pooled`
- `recon`
- `logits`
- `point_scores`
- `window_scores`
- `aux`

If additional synthetic-validation metadata is required, it should be attached to the returned batch dictionary rather than by expanding the model output contract with another synthetic-only score structure.

### Engine contract

The training engine remains responsible for:

- moving batches to the target device;
- invoking model stage methods;
- aggregating scalar metrics;
- coordinating evaluator-style metric computation;
- performing checkpoint selection.

The engine must not duplicate model-specific anomaly generation logic. Synthetic anomaly injection remains model-owned or injector-owned; synthetic VUS computation is engine-owned.

## Design Pattern Application

### Composition over inheritance

The synthetic validation VUS feature must be built through composition. The trainer should call reusable reconstruction and metric helpers rather than subclassing `Evaluator` or introducing a second trainer hierarchy.

### Adapter pattern

The repository already treats evaluation as a model-agnostic consumer of the standard `point_scores` contract. Synthetic validation VUS computation should preserve this adapter-like behavior by converting stage outputs and synthetic labels into the same pointwise metric core used by the clean evaluator.

### Strategy pattern

The code already has distinct runtime strategies:

- clean validation through `validation_step()`;
- synthetic validation through `synthetic_validation_step()`;
- plateau scheduling through `ReduceLROnPlateau`;
- cosine scheduling through arithmetic per-batch learning-rate updates.

This feature should follow the same strategy approach. Synthetic VUS computation is an additional validation strategy layered on top of the synthetic validation phase, not a replacement for clean validation.

### Registry and factory principles

No change is required to dataset and model registration. This work remains at the engine, model-output, and configuration-validation layers. Registry paths in `scripts/train.py` and `scripts/evaluate.py` must remain intact.

## Phase 1: Standardize Synthetic Validation Pointwise Targets

### Phase summary

This phase makes the synthetic validation path expressible in pointwise-metric terms without changing the clean-validation semantics. It serves the thesis objective of evaluating anomaly localization behavior on synthetic validation windows while preserving the readability-first design of the model files.

### File-level edits

#### `src/models/redlamp_mlp_baseline.py`

Inspect the prepared batch returned by `_prepare_batch()` and the batch returned from `_shared_step()` during `synthetic_validation_step()`.

If `synthetic_anomaly_mask` is already present and stable in the returned batch, document that behavior with a brief explanatory comment. If it is not guaranteed, make the synthetic validation path explicitly populate:

```python
prepared_batch["synthetic_anomaly_mask"] = synthetic_anomaly_mask
```

The mask must have shape `[B, L]` and must align with the `point_scores` emitted by the model forward path.

#### `src/models/thesis_multitask.py`

Apply the same rule as above. The synthetic validation batch returned by `synthetic_validation_step()` must expose one stable pointwise synthetic target field that the engine can trust.

If `synthetic_anomaly_mask` already exists in the prepared synthetic batch, preserve the existing name and semantics. Do not invent a second alias.

### Interface requirements

- `synthetic_anomaly_mask` must be pointwise and window-aligned.
- It must travel through the returned `step_output["batch"]`.
- Clean validation batches are not required to include it.

### Risk mitigation

- Prototype redundancy: not directly changed; this phase must not touch prototype logic.
- Fusion collapse: not directly changed; this phase must not alter fusion weighting or gating.
- Adaptation contamination: not directly changed; this feature remains offline-only.
- Projector drift: not applicable to this phase.
- Evaluation metric inflation: explicit synthetic pointwise labels prevent accidental reuse of clean labels for synthetic metrics.

### Tests

Add or extend tests to confirm that `synthetic_validation_step()` returns a batch whose `synthetic_anomaly_mask` exists and has the same first two dimensions as `x`.

Preferred test homes:

- `tests/test_multitask_validation_alignment.py`
- `tests/test_one_redlamp_mlp_train_step.py`

### Acceptance criteria

- Both active offline model families expose a stable synthetic pointwise target field during synthetic validation.
- The synthetic pointwise target aligns shape-wise with `point_scores`.
- No clean validation or training-step contract is broken.

## Phase 2: Extract Reusable Overlap-Aware Reconstruction Helpers

### Phase summary

This phase turns the evaluator's current overlap-aware timeline merge into a reusable engine-level utility. It directly supports the thesis objective that pointwise anomaly metrics should be computed on reconstructed timelines rather than on disconnected windows.

### File-level edits

#### `src/engine/evaluator.py`

Extract the overlap-aware reconstruction path into one or more helpers that can be reused by both the public evaluator and the trainer synthetic validation path. A readable target shape is:

```python
def reconstruct_pointwise_records_from_window_payload(
    *,
    sequences_by_entity: dict[str, dict[str, Any]],
    batch_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ...
```

or a closely related helper pair such as:

```python
def accumulate_pointwise_window_payload(...)
def finalize_pointwise_records(...)
```

Each payload should include:

- batch metadata with `entity_id`, `start_index`, `end_index`;
- per-window `point_scores`;
- per-window pointwise labels.

The public `Evaluator.evaluate()` method should be rewritten to use the new helpers internally without changing its return contract:

```python
{
    "metrics": ...,
    "records": ...,
    "curves": ...,
}
```

### Interface and contract definitions

The reusable helper must enforce:

- per-window `point_scores` shape `[B, L]`;
- per-window pointwise labels shape `[B, L]`;
- metadata length equal to batch size;
- output records with `entity_id`, `point_scores`, `point_labels`, and `num_points`.

### Design pattern note

This is the core composition step. The trainer should become a consumer of the same reconstruction helper rather than reimplementing entity accumulation in a second codepath.

### Risk mitigation

- Evaluation metric inflation: shared reconstruction helpers guarantee that clean and synthetic pointwise metrics use the same overlap-aware semantics.
- Codepath drift: one shared helper limits silent semantic divergence between test evaluation and synthetic validation.

### Tests

Add deterministic tests that:

- reconstruct overlapping synthetic windows into the expected entity-level pointwise sequence;
- preserve the existing evaluator behavior for clean evaluation.

Preferred test home:

- `tests/test_evaluator_thresholding.py`

### Acceptance criteria

- `Evaluator.evaluate()` still passes its current regression tests unchanged.
- The overlap-aware reconstruction logic can be called independently of `model.test_step()`.
- Synthetic validation can consume the same helper without duplicating merge logic.

## Phase 3: Compute Synthetic Validation Pointwise Metrics in the Trainer

### Phase summary

This phase integrates `val_synth_vus_pr` into the epoch-level validation runtime. It translates synthetic validation from a classification-only monitoring path into a path that also supports thesis-facing pointwise anomaly localization metrics.

### File-level edits

#### `src/engine/trainer.py`

Inside the synthetic validation loop, retain the information needed for post-loop pointwise metric computation. A readable approach is to collect a list of payload objects such as:

```python
synthetic_validation_pointwise_payloads: list[dict[str, Any]] = []
```

For each synthetic validation batch, append:

```python
{
    "meta": step_output["batch"]["meta"],
    "point_scores": step_output["outputs"]["point_scores"].detach().cpu(),
    "point_labels": step_output["batch"]["synthetic_anomaly_mask"].detach().cpu(),
}
```

After the `val_synth` loop:

1. resolve `sequences_by_entity` from the validation loader dataset;
2. reconstruct entity-level synthetic validation records using the shared helper from `src/engine/evaluator.py`;
3. concatenate reconstructed synthetic scores and labels;
4. compute pointwise metrics with the same `compute_pointwise_metrics(...)` path used by the evaluator;
5. prefix the resulting metrics with `val_synth_`.

Recommended emitted metrics:

- `val_synth_roc_auc_pointwise`
- `val_synth_pr_auc_pointwise`
- `val_synth_precision_pointwise`
- `val_synth_recall_pointwise`
- `val_synth_f1_pointwise`
- `val_synth_fpr_pointwise`
- `val_synth_vus_pr`
- `val_synth_threshold`

If the repository prefers not to rename the existing classification metrics, keep the existing `val_synth_pr_auc` and `val_synth_roc_auc` for classification and use explicit `_pointwise` suffixes for the pointwise versions, while reserving `val_synth_vus_pr` as the new thesis-facing metric.

### Contract and naming rules

- Existing synthetic classification metrics must remain intact for backward compatibility.
- New pointwise synthetic metrics must be distinguishable from classification metrics by name.
- `val_synth_vus_pr` must refer only to the pointwise synthetic validation path, never to classification logits.

### Risk mitigation

- Evaluation metric inflation: explicit naming separation prevents confusion between classification `pr_auc` and pointwise synthetic `vus_pr`.
- Fusion collapse and prototype redundancy: unchanged by this phase; the trainer must stay model-agnostic.

### Tests

Add trainer-level tests proving:

- `val_synth_vus_pr` is present in `epoch_metrics` after one epoch when synthetic validation is enabled;
- `val_synth_vus_pr` is computed from `point_scores` and `synthetic_anomaly_mask`, not from `classification_labels`;
- existing `val_synth_pr_auc` classification metrics remain present.

Preferred test homes:

- `tests/test_learning_rate_scheduler.py`
- `tests/test_multitask_validation_alignment.py`

### Acceptance criteria

- One synthetic validation epoch emits `val_synth_vus_pr`.
- Existing synthetic classification metrics remain behaviorally unchanged.
- Pointwise synthetic metrics and classification synthetic metrics are not conflated.

## Phase 4: Extend Configuration and Checkpoint-Monitor Semantics

### Phase summary

This phase makes the new synthetic validation VUS metric first-class at the experiment configuration layer. It supports thesis experiments that want best-checkpoint selection from a pointwise synthetic validation metric while preserving the runtime distinction between checkpoint monitoring and scheduler stepping.

### File-level edits

#### `src/core/config.py`

Extend `checkpoint_monitor_metric` validation to accept:

```python
"val_synth_vus_pr"
```

Do not change scheduler monitor validation for `reduce_on_plateau` in this phase unless a specific experiment requires it. The metric-driven scheduler and checkpoint monitor remain separate concerns.

#### `src/engine/trainer.py`

Extend `_resolve_best_checkpoint_monitor()` so that:

```python
"val_synth_vus_pr": "max"
```

is a supported checkpoint monitor mode.

### Contract rules

- `checkpoint_monitor_metric` may differ from any scheduler monitor metric.
- Cosine scheduling must continue to function with no scheduler monitor metric.
- Plateau scheduling must preserve existing monitor validation behavior.

### Risk mitigation

- Evaluation metric inflation: only metrics with explicit runtime semantics may become checkpoint monitors.
- Scheduler confusion: separate monitor surfaces prevent accidental coupling between cosine arithmetic scheduling and checkpoint selection.

### Tests

Add tests proving:

- config validation accepts `checkpoint_monitor_metric: val_synth_vus_pr`;
- checkpoint selection chooses the epoch with the highest `val_synth_vus_pr`;
- cosine scheduling still functions with `val_synth_vus_pr` as checkpoint monitor and no scheduler monitor.

Preferred test homes:

- `tests/test_config_loading.py`
- `tests/test_learning_rate_scheduler.py`

### Acceptance criteria

- `checkpoint_monitor_metric: val_synth_vus_pr` is valid.
- Trainer saves `best.pt` using `val_synth_vus_pr` when configured.
- Cosine scheduling remains metric-free.

## Phase 5: Add Experiment-Surface Support

### Phase summary

This phase exposes the new metric through explicit experiment configuration, making it reproducible and visible to later thesis runs without requiring source-code inspection.

### File-level edits

#### `configs/experiment/`

Create or update explicit experiment YAML files for baseline and multitask synthetic-VUS monitoring runs. Use file names that clearly describe the monitor choice, for example:

```text
configs/experiment/smd_redlamp_mlp_baseline_machine_2_1_window20_adamw_cosine_lr1e-3_val_synth_vus_pr.yaml
configs/experiment/smd_redlamp_mlp_baseline_machine_2_1_window20_adamw_cosine_lr1e-4_val_synth_vus_pr.yaml
```

or, if duplication is undesirable, update the current synthetic-validation-oriented experiment family to point at:

```yaml
checkpoint_monitor_metric: val_synth_vus_pr
```

while keeping the old monitor family in separate files for comparison.

### Design guidance

The filename should reveal:

- dataset or entity surface;
- model family;
- scheduler family;
- learning rate;
- checkpoint monitor metric.

This follows the repository's reproducibility and readability preferences.

### Risk mitigation

- Experiment ambiguity: explicit file naming prevents confusion between clean-VUS, synthetic-classification, and synthetic-VUS monitor surfaces.
- Ablation friendliness: separate YAML files keep experiment comparisons explicit and reproducible.

### Tests

Extend config-loading coverage to ensure the new experiment files load successfully and resolve to the intended checkpoint monitor metric.

### Acceptance criteria

- At least one explicit experiment config requests `checkpoint_monitor_metric: val_synth_vus_pr`.
- The config filename and contents make the monitoring semantics obvious to a future reader.

## Phase 6: Validation and Regression Pass

### Phase summary

This phase verifies that the new synthetic validation VUS metric works without regressing clean evaluation, checkpoint roundtrip, or one-step training behavior.

### Required commands

Run at minimum:

```bash
pytest -q tests/test_config_loading.py tests/test_learning_rate_scheduler.py
pytest -q tests/test_evaluator_thresholding.py tests/test_vus_pr_metric.py
pytest -q tests/test_multitask_validation_alignment.py
pytest -q tests/test_one_train_step.py tests/test_one_multitask_train_step.py tests/test_one_redlamp_mlp_train_step.py
pytest -q tests/test_checkpoint_roundtrip.py
```

### Manual validation

Run one short smoke experiment with reduced epochs and reduced window limits on an SMD machine-specific experiment that enables synthetic validation. Verify:

- `val_synth_pr_auc` is still present as a classification metric;
- `val_synth_vus_pr` is present as a pointwise synthetic validation metric;
- `best.pt` is written when `checkpoint_monitor_metric: val_synth_vus_pr` is used;
- cosine learning-rate summaries still appear at the epoch level and do not require a scheduler monitor;
- no existing clean validation metrics disappear unexpectedly.

### Acceptance criteria

- Focused regression tests pass.
- The synthetic validation path now emits `val_synth_vus_pr`.
- Best-checkpoint selection can use `val_synth_vus_pr`.
- Existing clean evaluator behavior remains intact.

## Cross-Cutting Risk Mitigation

| Repository risk from thesis context | Relevance to this feature | Mitigation in this plan |
|---|---|---|
| Prototype redundancy | Indirect | Do not alter prototype modules; constrain work to synthetic validation metrics only. |
| Fusion collapse | Indirect | Preserve current model logic and avoid changing fusion paths while adding metric computation. |
| Adaptation contamination | Not directly relevant | Keep this work offline-only and separate from online adaptation. |
| Projector drift | Not relevant to current scope | Do not touch online adaptation model or projector codepaths. |
| Evaluation metric inflation | Directly relevant | Separate classification synthetic metrics from pointwise synthetic metrics by explicit naming and shared evaluator semantics. |
| Monitor ambiguity | Directly relevant | Keep checkpoint monitor semantics distinct from scheduler semantics and make config names explicit. |

## Detailed Test Inventory

| Test area | File |
|---|---|
| Synthetic anomaly mask availability | `tests/test_multitask_validation_alignment.py` |
| Overlap-aware synthetic reconstruction | `tests/test_evaluator_thresholding.py` |
| Synthetic VUS epoch logging | `tests/test_learning_rate_scheduler.py` |
| Checkpoint selection from synthetic VUS | `tests/test_learning_rate_scheduler.py` |
| Config acceptance for `val_synth_vus_pr` | `tests/test_config_loading.py` |
| Clean evaluation regression | `tests/test_vus_pr_metric.py`, `tests/test_evaluator_thresholding.py` |
| One-step offline regression | `tests/test_one_train_step.py`, `tests/test_one_multitask_train_step.py`, `tests/test_one_redlamp_mlp_train_step.py` |
| Checkpoint persistence regression | `tests/test_checkpoint_roundtrip.py` |

## Final Acceptance Criteria

The feature is complete when all of the following are true:

1. Synthetic validation batches expose a stable pointwise synthetic target field.
2. The overlap-aware reconstruction logic can be reused outside the public test evaluator.
3. `val_synth_vus_pr` is computed from synthetic validation point scores and synthetic pointwise labels.
4. Existing synthetic classification metrics remain unchanged and still log under `val_synth_*`.
5. `checkpoint_monitor_metric: val_synth_vus_pr` is accepted by config validation.
6. Best-checkpoint selection can use `val_synth_vus_pr`.
7. Cosine learning-rate scheduling remains arithmetic and metric-free.
8. Clean `val_vus_pr` behavior remains available and unchanged for experiments that still request it.
9. Focused tests and regression tests pass.
10. At least one explicit experiment configuration makes the new synthetic-VUS checkpoint-monitor path reproducible.
