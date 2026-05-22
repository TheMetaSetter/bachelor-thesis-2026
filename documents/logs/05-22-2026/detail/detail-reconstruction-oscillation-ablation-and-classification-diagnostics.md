---
date: 2026-05-22T17:42:00+07:00
author: Artificial Intelligence Agent
git_commit: acf12e9b1708a6832426f2ffe01768a0d5eacbee
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed implementation plan for reconstruction-oscillation ablation and classification diagnostics"
tags: [detail, plan, multitask, diagnostics, testing]
status: proposed
last_updated: 2026-05-22
last_updated_by: Artificial Intelligence Agent
source_plan: documents/logs/05-22-2026/plan/plan-reconstruction-oscillation-ablation-and-classification-diagnostics.md
---

# Detailed Plan: Reconstruction-Oscillation Ablation and Classification Diagnostics

## 1. Objective and Scope
This detailed plan operationalizes the approved plan for two offline pre-training experiments in the current codebase:
1. Experiment 1 isolates reconstruction by disabling classification forward/loss/logging while preserving baseline synthetic augmentation behavior.
2. Experiment 2 preserves the full multitask path and adds stage-scoped (`train`, `val_synth`) classification diagnostics: hard prediction ratio and row-normalized confusion matrix.

This document defines phase sequencing, exact file-level edits, interface-level contracts, risk controls, and measurable acceptance criteria.

## 2. Stable Contracts and Design Rules

### 2.1 Batch contract
- Keep runtime batch dictionary unchanged.
- Required key: `x` with shape `[B, L, D]`.
- Optional keys used by current multitask path remain unchanged (`classification_labels`, `synthetic_anomaly_mask`, `meta`, and existing point-label fields).

### 2.2 Encoder contract
- No change to encoder output semantics.
- `hidden` remains `[B, L, H]` and continues to be the thesis-facing representation.

### 2.3 Model output contract
- Maintain existing output keys and optionality.
- In reconstruction-isolated mode, classification outputs become explicit optional nulls (`logits=None`, `aux.class_probabilities=None` or omitted under a guarded contract path), while reconstruction outputs remain mandatory.

### 2.4 Engine contract
- `Trainer` continues to be model-agnostic and only consumes stage outputs.
- Classification aggregation must remain conditional on `logits is not None`.

### 2.5 Applied design patterns
- Composition over inheritance: retain current small engine + model-owned stage logic.
- Adapter pattern for encoders: unchanged and out of scope for this task.
- Strategy pattern for tasks/stages: preserved via model stage methods (`training_step`, `validation_step`, `synthetic_validation_step`).
- Registry/factory: unchanged (`register_model`, `register_dataset`, `build_model`, `build_dataset`).

## 3. Phase-by-Phase Execution Plan

## Phase 0: Pre-Implementation Baseline Lock
### Phase summary
Establish deterministic baseline references so subsequent ablation and diagnostic behavior is attributable to approved changes only.

### File-level edits
- No source-code edits in this phase.

### Actions
1. Resolve and snapshot baseline config from:
   - `configs/experiment/smd_thesis_multitask_redlamp_multiclass_window20_recon_diag_quick_100ep.yaml`
2. Record pre-change behavior expectations:
   - classification scalar metrics exist in epoch logs for `train` and `val_synth`.
   - confusion matrix and class-ratio artifacts do not exist yet.

### Acceptance criteria
- A written baseline checklist exists in working notes before any code modification.
- Resolved config path and experiment command are fixed and unambiguous.

## Phase 1: Configuration and Schema Extension
### Phase summary
Add explicit, ablation-friendly switches and diagnostic controls while preserving backward compatibility.

### File-level edits
- `configs/model/thesis_multitask_redlamp_multiclass.yaml`
- `configs/experiment/smd_thesis_multitask_redlamp_multiclass_window20_recon_diag_quick_100ep.yaml`
- `src/core/config.py`

### Explicit edit content
1. Add model flag:
   - `enable_classification_path: true` (default for existing behavior).
2. Add logging controls in experiment scope:
   - `diagnostics_stages_for_classification: [train, val_synth]`
   - `log_hard_prediction_ratio: true`
   - `log_row_normalized_confusion_matrix: true`
   - `focus_metrics: [...]` (experiment-specific curated list).
3. Add focused metrics stream control:
   - `log_focused_metrics_jsonl: true`
   - `focused_metrics_filename: focused_metrics.jsonl`.
4. Extend schema checks in `src/core/config.py`:
   - boolean validation for new booleans,
   - list validation for stage keys with allowed set `{train, val, val_synth, test}`,
   - list-of-string validation for `focus_metrics`.

### Acceptance criteria
- Invalid stage names are rejected with clear errors.
- Missing optional new keys preserve old behavior.
- Existing experiment configs outside this scope still load successfully.

## Phase 2: Reconstruction-Isolated Mode in Model Path (Experiment 1)
### Phase summary
Implement strict classification-path deactivation while preserving reconstruction mechanics and synthetic augmentation behavior.

### File-level edits
- `src/models/thesis_multitask.py`

### Explicit edit content
1. Parse and store `enable_classification_path` in model config initialization.
2. In forward/stage flow:
   - guard classification head execution behind `enable_classification_path`.
   - when disabled, avoid computing logits/probabilities.
3. In `_shared_step` path:
   - skip classification loss term when classification path disabled.
   - keep reconstruction loss and optional diagnostics unchanged.
4. In step logs:
   - suppress classification keys when classification path disabled.
5. Ensure compatibility with existing `Trainer` conditionals (`logits is not None`).

### Acceptance criteria
- With `enable_classification_path=false`, one training step runs without classification tensors and without classification log keys.
- Reconstruction loss remains finite and backpropagates.
- Synthetic augmentation still executes as baseline (no changes to injector path).

## Phase 3: Classification Diagnostics Aggregation (Experiment 2)
### Phase summary
Add deterministic per-epoch diagnostics for class bias and class confusion in selected stages.

### File-level edits
- `src/engine/trainer.py`
- `src/metrics/classification_diagnostics.py` (new file, preferred for separation)
- Optional import wiring in `src/metrics/__init__.py` if present.

### Explicit edit content
1. Add helper functions in new metrics module:
   - `compute_hard_prediction_ratio(logits, class_names) -> dict[str, float]`
   - `compute_row_normalized_confusion_matrix(logits, labels, class_names) -> dict`
2. In trainer epoch aggregation:
   - for configured stages only (`train`, `val_synth`), compute diagnostics from concatenated logits/labels.
   - add scalar summaries to `epoch_metrics`:
     - per-class hard ratio metrics,
     - normalized confusion diagonal per class.
3. Artifact persistence:
   - write full confusion matrices to
     `output_dir/classification_diagnostics/epoch_{epoch:04d}_{stage}.json`.

### Acceptance criteria
- Diagnostics are absent for non-configured stages.
- Row-normalized confusion rows sum to 1.0 for rows with support > 0.
- Artifacts include class names, raw counts matrix, normalized matrix, and epoch metadata.

## Phase 4: Focused Metrics Stream + W&B Mirroring
### Phase summary
Provide fast inspection path while preserving complete metrics history.

### File-level edits
- `src/engine/logger.py`
- `src/engine/trainer.py` (call-site integration only)

### Explicit edit content
1. Extend logger with focused-metrics writer:
   - initialize `focused_metrics_path` when enabled.
   - add `log_focused_metrics(metrics, focus_metric_names)` method.
2. At each epoch:
   - filter epoch metrics by `focus_metrics` list,
   - write filtered record to separate JSONL stream,
   - mirror same filtered dictionary to W&B using epoch step.
3. Preserve existing `metrics.jsonl` behavior unchanged.

### Acceptance criteria
- `focused_metrics.jsonl` is created and contains only selected keys + epoch index.
- Focused metrics appear in W&B for each epoch where full metrics are logged.
- Full metrics stream remains intact and unchanged in schema.

## Phase 5: Test Implementation and Verification
### Phase summary
Enforce adversarial-quality validation via unit and integration tests to prevent silent regressions.

### Unit tests
- `tests/test_thesis_multitask_classification_path_toggle.py`
  - classification disabled path returns `logits is None`.
  - no classification log keys are emitted.
  - reconstruction loss remains finite.
- `tests/test_classification_diagnostics_metrics.py`
  - hard-ratio sums to ~1.0.
  - row normalization properties hold.
  - edge handling for empty-support rows is explicit and deterministic.
- `tests/test_config_loading.py` extensions
  - invalid diagnostic stage list fails.
  - invalid focused metric config types fail.

### Integration tests
- `tests/test_reconstruction_only_ablation_pipeline.py`
  - short run with classification path off, synthetic augmentation unchanged.
  - no classification metrics; reconstruction metrics and checkpoint exist.
- `tests/test_trainer_epoch_diagnostics_train_and_val_synth.py`
  - short full-model run.
  - diagnostics artifacts appear exactly for `train` and `val_synth`.
  - focused metrics file exists and is non-empty.

### Validation command set (to be executed in implementation phase)
1. Targeted new tests.
2. Regression subset for multitask shape/one-step/checkpoint behavior.
3. Two short dry-run experiments (Exp 1 and Exp 2).

### Acceptance criteria
- All newly added tests pass.
- Existing critical regression subset passes.
- Runtime artifacts and logs match experiment-specific expectations.

## 4. Risk Mitigation Matrix

- Prototype redundancy risk:
  - Out of direct scope for this implementation.
  - Mitigation continuity: preserve existing branch-level diagnostics and avoid altering branch architecture.
- Fusion collapse risk:
  - Do not modify fusion logic.
  - Continue logging alpha/beta and CKA-related diagnostics already present.
- Adaptation contamination risk:
  - Online adaptation is out of scope; no changes to adaptation modules.
- Projector drift risk:
  - Online projector path is out of scope; no behavioral changes introduced.
- Evaluation metric inflation risk:
  - Add transparent confusion and ratio diagnostics with explicit definitions.
  - Keep existing pointwise and VUS metrics unchanged for comparability.

## 5. Deliverables and Completion Gate

### Deliverables
1. Updated configs and schema validation for new ablation/diagnostic controls.
2. Model-level classification-path gating for Experiment 1.
3. Trainer-level diagnostics and artifact emission for Experiment 2.
4. Focused metrics JSONL stream and W&B focused metric mirroring.
5. Unit and integration tests with execution evidence.
6. Post-implementation verification note under `documents/logs/05-22-2026/detail/`.

### Completion gate
Implementation is considered complete only when:
- all acceptance criteria across Phases 1–5 are satisfied,
- tests pass with evidence,
- and artifact/log outputs are verified for both experiment modes.
