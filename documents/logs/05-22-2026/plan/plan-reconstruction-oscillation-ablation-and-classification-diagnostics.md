---
date: 2026-05-22T17:31:00+07:00
planner: Artificial Intelligence Agent
git_commit: acf12e9b1708a6832426f2ffe01768a0d5eacbee
branch: dev
repository: bachelor-thesis-2026
topic: "Implementation plan for reconstruction-oscillation ablation and classification diagnostics"
tags: [plan, time-series, anomaly-detection, multitask, diagnostics]
status: proposed
last_updated: 2026-05-22
last_updated_by: Artificial Intelligence Agent
source_research: documents/logs/05-22-2026/research/research-current-codebase-status-for-reconstruction-oscillation-ablation-and-classification-diagnostics.md
---

# Plan: Implementation Plan for Reconstruction-Oscillation Ablation and Classification Diagnostics

## Planning Objective
This plan defines a minimal, codebase-aligned implementation sequence for two offline pre-training experiments:
1. Experiment 1: isolate reconstruction learning by disabling classification learning and classification forward/logging paths.
2. Experiment 2: retain the full model and add focused diagnostics for `train` and `val_synth`, specifically hard prediction ratio and row-normalized confusion matrix.

The plan preserves current architecture and contracts, modifies the smallest stable surfaces, and introduces rigorous unit and integration tests for behavioral guarantees.

## Current State
- The runtime is configuration-driven via `scripts/train.py` and resolved YAML composition.
- `ThesisMultitaskModel` already co-locates model architecture, losses, and stage logic in one file (`src/models/thesis_multitask.py`), consistent with repository preferences.
- `Trainer` already aggregates epoch-level logits and labels and computes scalar classification metrics, but it does not emit confusion-matrix artifacts or class-ratio summaries.
- `ExperimentLogger` logs scalar epoch metrics to `metrics.jsonl` and optionally mirrors them to Weights and Biases.

## Fixed Decisions (Resolved Before Planning)
- Experiment 1 must disable both classification learning and classification forward/logging paths to maximize reconstruction isolation.
- Experiment 1 keeps synthetic augmentation exactly as the baseline configuration.
- Experiment 2 diagnostics are required for `train` and `val_synth` only.
- Primary class-ratio diagnostic uses hard predictions (`argmax(logits)`).
- Confusion matrices are normalized by row (ground-truth class).
- Focused metrics are written to a separate JSONL stream for fast inspection and must also be logged to Weights and Biases.

## Design Options
### Option A (Recommended): Minimal surgical extension of existing multitask and trainer paths
- Add explicit configuration switches to control classification-path execution and stage-specific diagnostics.
- Reuse existing epoch-level logits/labels aggregation in `Trainer`.
- Add deterministic confusion-matrix and prediction-ratio computation in a dedicated metrics helper.
- Keep all model computations within `thesis_multitask.py` and all epoch aggregation in `trainer.py`.

Rationale: This option preserves stable interfaces, minimizes codepath divergence, and supports ablations directly from configuration.

### Option B: Separate dedicated diagnostic runner script
- Introduce a parallel training/diagnostic script that post-processes batch outputs independently.

Rationale against selection: This duplicates control flow, increases maintenance burden, and violates the least-codepath preference.

### Option C: External analysis-only notebook pipeline over stored logits
- Persist logits/labels per epoch to disk and compute diagnostics in notebooks.

Rationale against selection: This delays feedback during training and weakens reproducibility of online experiment monitoring.

## Selected Approach
Option A is selected.

## Scope of Code Changes

### 1) Configuration Surface
#### Files to modify
- `configs/model/thesis_multitask_redlamp_multiclass.yaml`
- `configs/experiment/smd_thesis_multitask_redlamp_multiclass_window20_recon_diag_quick_100ep.yaml`
- `src/core/config.py`

#### Planned additions
- Add explicit model-level boolean flag: `enable_classification_path`.
  - Semantics: when `false`, classification head forward and classification-related outputs are skipped by model stage paths.
- Add logging-level diagnostic control object (or flat keys if codebase style prefers flat):
  - `diagnostics_stages_for_classification: [train, val_synth]`
  - `log_hard_prediction_ratio: true|false`
  - `log_row_normalized_confusion_matrix: true|false`

#### Validation requirements
- Extend `src/core/config.py` schema validation for newly introduced booleans/list choices.
- Enforce allowed stage values subset from `{train, val, val_synth, test}`.

### 2) Model Path for Experiment 1 Isolation
#### File to modify
- `src/models/thesis_multitask.py`

#### Planned behavior
- In shared stage logic (`_shared_step`-anchored path), gate classification computations by `enable_classification_path`.
- When classification path is disabled:
  - Do not run classification head forward.
  - Do not compute classification loss.
  - Do not emit classification logits/probabilities/classification metrics in step logs.
  - Keep reconstruction path unchanged.
- Preserve batch contract and model output contract by using explicit `None` for unavailable classification outputs where downstream code expects optional fields.

#### Contract handling
- Batch contract remains unchanged (`batch` dictionary with `x` and optional labels/masks).
- Encoder contract remains unchanged (`hidden` with shape `[B, L, H]`).
- Model output contract remains stable:
  - `recon` and `point_scores` remain active.
  - `logits` becomes `None` when classification path is disabled.

### 3) Epoch-Level Diagnostics for Experiment 2
#### Files to modify
- `src/engine/trainer.py`
- `src/metrics/pointwise.py` (or create `src/metrics/classification_diagnostics.py` if clearer separation is preferred)

#### Planned behavior
- Reuse existing per-stage logits/labels histories.
- For configured stages (`train`, `val_synth`), compute:
  - hard prediction ratio per class (from argmax predictions)
  - row-normalized confusion matrix
- Emit compact scalar summary metrics directly to `epoch_metrics` (for filtering and dashboards):
  - per-class prediction ratio keys
  - per-class diagonal recall from normalized confusion matrix
- Persist full confusion matrix as structured artifact per epoch (JSON) under output directory for post hoc analysis.

#### Data structure conventions
- Class index to name mapping must follow model-provided class names when available (`classification_class_names`) and fallback to deterministic index names otherwise.
- Confusion matrix artifact naming pattern:
  - `classification_diagnostics/epoch_{epoch:04d}_{stage}.json`

### 4) Focused Metric Views per Experiment
#### Files to modify
- `src/engine/logger.py` (if a focused-metric summary helper is needed)
- Possibly `scripts/visualize_training_metrics.py` if it exists in current codebase path and is actively used.

#### Planned behavior
- Add experiment-level `focus_metrics` list support in config.
- At epoch end, produce a compact filtered record that only includes focus metrics for rapid inspection.
- Persist focused metrics to a separate JSONL file (for example `focused_metrics.jsonl`) under the run output directory.
- Mirror the same focused metric keys to Weights and Biases in the same epoch step.
- Keep full metric logging unchanged for reproducibility.

## Test Strategy (Unit + Integration, High-Rigor)

### Unit tests
#### New/updated tests
- `tests/test_thesis_multitask_classification_path_toggle.py`
  - Verify `enable_classification_path=false` leads to `logits is None` and zero classification-related log keys.
  - Verify reconstruction loss path still executes and returns finite loss.
- `tests/test_classification_diagnostics_metrics.py`
  - Validate hard prediction ratio computation sums to 1.0 within tolerance.
  - Validate row-normalized confusion matrix row sums are 1.0 for non-empty rows.
  - Validate deterministic behavior on fixed logits/labels tensors.
- `tests/test_config_loading.py` (extend)
  - Validate new config keys and reject invalid stage names or invalid value types.

### Integration tests
#### New tests
- `tests/test_trainer_epoch_diagnostics_train_and_val_synth.py`
  - Run a short deterministic training loop.
  - Assert that diagnostic artifacts are produced for `train` and `val_synth` only.
  - Assert that logged epoch metrics include configured focus metrics.
- `tests/test_reconstruction_only_ablation_pipeline.py`
  - Run a short reconstruction-only ablation config with classification path disabled.
  - Assert training loop completes, checkpoints save, and no classification metrics are logged.
  - Assert reconstruction metrics are present and finite.

### Test strictness principles
- Use realistic tensor shapes aligned with active SMD window setup.
- Include failure-case assertions (invalid config, empty/degenerate class rows).
- Avoid superficial “smoke-only” assertions; check semantic invariants explicitly.

## Validation and Execution Procedure
1. Run targeted tests for new modules and modified paths.
2. Run a broader regression subset including existing multitask shape/one-step/checkpoint tests.
3. Execute one short dry-run for each experiment mode:
   - reconstruction-isolated mode
   - full-model diagnostics mode
4. Confirm outputs:
   - metrics JSONL contains focus metrics
   - confusion-matrix artifacts exist for required stages only
   - classification outputs are absent in Experiment 1 as designed

## Minimal Vertical Slice Sequence
1. Implement config schema and default fields.
2. Implement Experiment 1 classification-path gating in model with unit tests.
3. Implement Experiment 2 diagnostics aggregation and artifact emission with unit tests.
4. Implement focused metric view and integration tests.
5. Run validation procedure and document outcomes.

## Risks and Mitigations
- Risk: introducing classification-path gating may break existing trainer assumptions.
  - Mitigation: keep `logits` optional and preserve existing branching checks (`if logits is not None`).
- Risk: diagnostic logging increases metric volume and clutter.
  - Mitigation: separate full metrics from focused metric subsets and persist large structures as artifacts.
- Risk: confusion-matrix interpretation drift across class-order definitions.
  - Mitigation: always persist class names alongside matrix payloads.
- Risk: ablation diverges from baseline due to unintended config changes.
  - Mitigation: restrict ablation changes to explicitly enumerated keys and verify resolved config diffs before running.

## Plan Approval Status
All previously open planning questions are resolved and incorporated into this document.

## Proposed Deliverables After Implementation
- Updated configuration files for both experiment modes.
- Model and trainer updates for classification-path gating and diagnostics.
- Diagnostic artifact schema for confusion matrices and class ratios.
- New unit/integration tests with execution evidence.
- Short implementation report in `documents/logs/05-22-2026/detail/` summarizing verification outcomes.
