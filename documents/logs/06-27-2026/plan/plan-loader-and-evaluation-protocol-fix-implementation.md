---
date: 2026-06-27 16:20:00 +07 +0700
researcher: Artificial Intelligence Agent
topic: "Implementation plan for loader and evaluation protocol fixes"
status: draft
---

# Plan: Loader and Evaluation Protocol Fixes

## Current State

Based on the current repository state and the forensic audit already completed, the active runtime path for evaluation is concentrated in:

- `src/data/datasets/smd.py`
- `src/data/datasets/anomaly_archive.py`
- `src/data/loaders.py`
- `src/engine/evaluator.py`
- `src/metrics/pointwise.py`
- `scripts/evaluate.py`
- `scripts/visualize_evaluation_results.py`
- `src/analysis/evaluation_protocol_audit.py`

The codebase already preserves one important contract correctly:

1. raw split sequences are parsed first,
2. `SequenceStandardScaler` is fit on train only,
3. train, val, and test are transformed,
4. transformed sequences are then cut into overlapping windows.

That means the normalization order is already acceptable and should be preserved.

The confirmed problems are not mainly in scaling or in the core overlap-averaging formula.

The confirmed problems are in test-split semantics and evaluation interpretation:

1. `anomaly_archive pre_vs_anomaly` creates an all-positive test set,
2. `anomaly_archive pre_vs_post` creates an all-negative or almost all-negative test set,
3. `max_test_windows` in smoke runs truncates test coverage to an early prefix,
4. several metric outputs become mathematically valid but scientifically misleading under those regimes.

The current code also already contains useful hardening that should be preserved:

- evaluation records now keep coverage metadata such as `evaluated_num_points`,
- audit reports already expose `label_regime`, `positive_ratio`, and truncation warnings,
- visualization already prefers raw parser labels over reconstructed labels when plotting ground truth.

## Design Options

### Option A: Minimal protocol repair with a new benchmark mode

Keep `pre_vs_anomaly` and `pre_vs_post` in `anomaly_archive`, but downgrade them to research-only modes. Add one new benchmark-oriented split mode, likely `full_future_segment`, that builds a continuous future test segment containing both normal and anomalous timesteps.

This option changes the least amount of architecture and best respects the current codebase style.

This is the recommended option.

### Option B: Replace current comparison modes entirely

Remove `pre_vs_anomaly` and `pre_vs_post` from the public data config path and support only one benchmark mode.

This option is cleaner in principle, but it removes currently useful research-only slices that still have explanatory value in a thesis or analysis note.

This option is not recommended as a first step.

### Option C: Keep current loader semantics and only harden warnings

Leave all split logic as it is, but add louder warnings and stronger audit reports.

This option is too weak. It documents the protocol problem but does not fix the protocol.

This option is not recommended except as a temporary guardrail.

## Recommended Approach

The implementation should follow Option A.

The main idea is simple:

1. preserve the existing loader architecture,
2. preserve research-only comparison modes,
3. introduce one proper benchmark mode for `AnomalyArchive/UCR`-style single-series datasets,
4. fail loudly when test regimes are unsuitable for benchmark-style evaluation,
5. keep smoke runs possible but never let them look like full evaluations.

## Proposed Module-Level Changes

### 1. `src/data/datasets/anomaly_archive.py`

Add one new benchmark-oriented comparison mode.

Recommended name:

- `full_future_segment`

Its job is:

1. choose train and val from earlier timesteps only,
2. choose one continuous future test segment,
3. ensure the test segment contains both normal and anomalous timesteps whenever the raw sequence allows it,
4. fail loudly when a mixed-label benchmark test segment cannot be formed cleanly.

The current `pre_vs_anomaly` and `pre_vs_post` logic should remain available, but only as protocol-specific research slices.

### 2. `src/data/loaders.py`

Do not redesign the builder path.

Keep the current split flow and scaling order.

Only extend the loader metadata so that downstream code can distinguish:

1. benchmark evaluation,
2. research-only comparison-mode evaluation,
3. truncated smoke evaluation.

This should stay additive rather than architectural.

### 3. `src/analysis/evaluation_protocol_audit.py`

Extend the audit report so it becomes the main reader-facing truth source for protocol status.

The report should clearly classify a run into statuses such as:

- full benchmark coverage,
- truncated smoke evaluation,
- all-positive research-only test regime,
- all-negative research-only test regime.

The explanation must remain short and plain.

### 4. `src/engine/evaluator.py`

Preserve the current overlap reconstruction logic.

Do not change the chronological reconstruction direction.

Only strengthen the saved metadata and warnings around:

- raw versus evaluated point count,
- single-class evaluated labels,
- truncated evaluation coverage.

### 5. `src/metrics/pointwise.py`

Do not change the formulas first.

The formulas are not the primary problem.

The work here should be limited to:

1. preserving diagnostic fields,
2. exposing clearer metric-status fields for undefined cases,
3. making downstream interpretation easier.

### 6. `scripts/evaluate.py`

Make sure every saved evaluation artifact carries enough information so a reader does not need to inspect source code.

This script should remain the single runtime path for:

- evaluation metrics,
- evaluation records,
- audit JSON,
- audit Markdown.

### 7. `scripts/visualize_evaluation_results.py`

Keep raw parser labels as the primary truth overlay.

Strengthen the visual explanation of:

- evaluated coverage,
- uncovered suffix or uncovered future region,
- protocol-special `AnomalyArchive` modes,
- truncated smoke runs.

## Stable Contracts to Preserve

The following contracts should remain stable during this work.

### Batch contract

Windows should continue to flow through the trainer and evaluator as dictionaries containing:

- `x`
- `point_labels`
- `mask`
- `timestamps`
- `meta`

### Loader contract

Dataset builders should continue to return a bundle containing:

- parser
- scaler
- raw sequences
- scaled sequences
- datasets
- loaders

### Evaluator contract

The evaluator should continue to:

1. score windows,
2. reconstruct pointwise timeline scores,
3. compute pointwise metrics on reconstructed timelines,
4. save per-entity records and global metrics.

## Risk and Mitigation

### Risk 1: The new benchmark mode still produces single-class test labels in short sequences

Mitigation:

- add explicit validation in the parser,
- fail loudly if mixed-label benchmark test construction is impossible,
- keep this failure close to the dataset parser, not hidden later in the evaluator.

### Risk 2: Smoke runs remain easy to misuse in reports

Mitigation:

- add stronger protocol-status fields,
- stamp smoke-truncated runs clearly in audit Markdown,
- keep `is_truncated_evaluation` in `evaluation_metrics.json`.

### Risk 3: Changing parser semantics breaks old configs silently

Mitigation:

- keep existing modes available,
- add the new benchmark mode additively,
- require explicit config choice for benchmark-oriented `AnomalyArchive` evaluation.

### Risk 4: Readers still confuse mathematically undefined metrics with code bugs

Mitigation:

- add explicit metric-status fields,
- keep plain-English explanations in audit reports,
- make visualization captions mention the label regime directly.

### Risk 5: Evaluation coverage and label regime differ between raw split and reconstructed evaluated region

Mitigation:

- keep both views visible,
- store both raw split coverage and evaluated coverage,
- flag when reconstructed evaluated labels are single-class even if raw test labels are mixed.

## Preliminary Implementation Phases

### Phase 1: Lock protocol semantics in tests

Before changing behavior, add tests that describe the desired benchmark contract and the known dangerous cases.

Priority test areas:

1. `AnomalyArchive` benchmark mode yields mixed-label test segments,
2. `pre_vs_anomaly` remains all-positive and is flagged as protocol-special,
3. `pre_vs_post` remains all-negative and is flagged as protocol-special,
4. smoke configs with `max_test_windows` are marked truncated,
5. reconstructed evaluated coverage is distinguishable from raw test length.

### Phase 2: Add the new benchmark mode

Implement the new `AnomalyArchive` split mode in the parser only.

Do not spread split logic across multiple files.

Keep all split semantics readable in one place.

### Phase 3: Harden audit and metric-status outputs

After the parser is corrected, extend audit and evaluation artifacts so they clearly describe:

- benchmark vs protocol-special mode,
- raw vs evaluated coverage,
- mixed-label vs single-class regime.

### Phase 4: Harden visualization

Make the visualization path show protocol conditions so a human can verify them by eye without reading the JSON artifacts.

### Phase 5: Run focused validation

Validate separately on:

1. one `AnomalyArchive pre_vs_anomaly` case,
2. one `AnomalyArchive pre_vs_post` case,
3. one `AnomalyArchive` benchmark-mode case,
4. one SMD smoke case,
5. one SMD full case.

## File-Level Worklist

Expected primary files to modify:

- `src/data/datasets/anomaly_archive.py`
- `src/analysis/evaluation_protocol_audit.py`
- `src/engine/evaluator.py`
- `src/metrics/pointwise.py`
- `scripts/evaluate.py`
- `scripts/visualize_evaluation_results.py`
- `tests/test_anomaly_archive_dataset_loader.py`
- `tests/test_evaluation_protocol_audit.py`
- `tests/test_evaluator_thresholding.py`
- `tests/test_evaluation_visualization.py`

Possible secondary file:

- `src/data/loaders.py`

This secondary file should only change if additional loader-level protocol metadata is truly needed.

## Validation Plan

Validation should not rely only on shape tests.

The most important checks are behavioral.

Minimum validation scenarios:

1. `AnomalyArchive` benchmark mode produces a mixed-label test timeline.
2. `pre_vs_anomaly` produces all-one labels and gets a protocol-special warning.
3. `pre_vs_post` produces all-zero labels and gets a protocol-special warning.
4. SMD smoke marks truncated evaluation and stores the correct raw/evaluated point counts.
5. SMD full run shows non-truncated full test coverage.
6. Visualization uses raw parser labels as primary truth, even when reconstructed labels are degenerate.

## Open Questions That Need User Decisions

These are low-level decisions that should be confirmed before moving to the next planning stage.

1. For the new benchmark mode in `AnomalyArchive`, should the test segment be defined as:
   - a segment that starts before `anomaly_start` and continues past `anomaly_end`,
   - or a segment that starts somewhere before `anomaly_start` and ends exactly at sequence end?

2. If a raw sequence is too short to create a mixed-label benchmark test split cleanly, should the loader:
   - raise an error immediately,
   - or skip that sequence with a loud warning?

3. Do anh want `pre_vs_anomaly` and `pre_vs_post` to stay available in normal config files, or should they be moved into explicitly named research-only configs later?

4. For smoke runs, should `scripts/evaluate.py` only warn, or should it also stamp an explicit field such as `protocol_status: truncated_smoke_not_comparable`?

5. For metric artifacts, do anh want undefined cases to keep numeric `NaN` only, or also add text fields such as `roc_auc_status: undefined_single_class`?

## Final Recommendation

The best next move is not to start coding immediately across many files.

The best next move is:

1. confirm the low-level split policy for the new `AnomalyArchive` benchmark mode,
2. confirm how strict failure should be for short or impossible sequences,
3. then move to the structure stage with test-first implementation planning.

This keeps the work aligned with the codebase workflow and reduces the chance of patching symptoms before the split contract is truly fixed.
