---
date: 2026-06-27 16:05:00 +07 +0700
researcher: Artificial Intelligence Agent
topic: "Specification for loader and evaluation protocol fixes after forensic audit"
status: draft
---

# Specification: Loader and Evaluation Protocol Fixes After Forensic Audit

## 1. Purpose

This specification defines how the active dataset and evaluation pipeline should be corrected after the forensic audit.

The goal is not to redesign the whole codebase.

The goal is to fix the parts that currently produce misleading evaluation results:

1. `anomaly_archive` comparison modes that create single-class test sets,
2. smoke runs that truncate the test timeline without loud enough warnings,
3. evaluation artifacts that do not clearly state how much of the raw test timeline was truly evaluated,
4. metric outputs that can be mathematically valid but scientifically misleading.

The implementation should stay close to the current architecture:

- shared builder path in `src/data/loaders.py`
- one parser per dataset family
- pointwise evaluator in `src/engine/evaluator.py`
- pointwise metric helpers in `src/metrics/pointwise.py`

No model API change is required for `redlamp_mlp_baseline` or `thesis_multitask`.

## 2. Problem Statement

### 2.1 Confirmed current problems

The audit has already confirmed three different failure modes.

First, under `anomaly_archive` with `comparison_mode: pre_vs_anomaly`, the test split is built only from the annotated anomaly span. Therefore every timestep in test is labeled `1`.

Second, under `anomaly_archive` with `comparison_mode: pre_vs_post`, the test split is built only from the segment after the anomaly. Therefore the test split becomes all normal or almost all normal, so labels become all `0`.

Third, in SMD smoke configurations, `max_test_windows` truncates the evaluated region to an early test prefix. This means the raw test sequence may contain anomalies later, but the evaluator never reaches them.

These are three different protocol problems:

1. all-positive test set,
2. all-negative test set,
3. partial-prefix test evaluation.

### 2.2 Why this is serious

When the test labels contain only one class, metrics such as `ROC-AUC`, `VUS-PR`, `VUS-ROC`, and `affiliation_f1` become undefined or misleading.

When smoke evaluation covers only an early prefix, the metric output no longer describes the full raw test timeline.

In both cases, a reader can easily think the model is good or bad because of the metric values, while the real issue is that the test protocol itself is unsuitable.

## 3. Target Behavior

### 3.1 General rule for anomaly-detection test splits

For benchmark-style anomaly detection, the test split should be a future segment of the timeline that contains both:

1. normal timesteps,
2. anomalous timesteps.

The test split must be a continuous segment on the time axis.

The train and validation splits must come strictly from earlier timesteps than the test split.

This is the minimum condition for a fair pointwise anomaly-detection evaluation.

### 3.2 Normalization rule

The normalization contract remains:

1. parse full split sequences,
2. fit `SequenceStandardScaler` on train sequences only,
3. transform train, val, and test sequences,
4. slice transformed sequences into windows.

This part of the current code is already correct and should be preserved.

### 3.3 Smoke-evaluation rule

Smoke runs may still truncate training, validation, and test windows for speed.

However, smoke runs must never be presented as ordinary full-timeline evaluations.

Any smoke-truncated test evaluation must be explicitly labeled as:

- truncated,
- partial coverage,
- not directly comparable to full evaluation.

## 4. Required Dataset Semantics

### 4.1 SMD

SMD should keep the current core split semantics:

- `train` and `val` come from the official train series,
- `test` comes from the official test series,
- point labels are attached to the test series.

This part is already acceptable.

The required change is not the SMD parser itself.

The required change is stronger handling of truncated smoke evaluation.

### 4.2 AnomalyArchive and UCR-like single-series datasets

The current `pre_vs_anomaly` and `pre_vs_post` modes must not be treated as normal benchmark modes.

They may remain available as research-only protocol slices, but they must be clearly marked as such.

For benchmark-style evaluation, a new mode must be introduced.

Recommended mode name:

- `full_future_segment`

Acceptable alternative:

- `contextual_test_segment`

The exact name matters less than the behavior.

The behavior must be:

1. `train` and `val` come from earlier timesteps only,
2. `test` is one continuous future segment,
3. `test` contains both normal and anomalous timesteps whenever the source sequence allows it,
4. no data leakage across time.

### 4.3 Required semantics of the new AnomalyArchive benchmark mode

For a sequence with:

- total length `T`,
- anomaly start `a_start`,
- anomaly end `a_end`,

the new benchmark mode must choose test so that:

1. test is not just `[a_start, a_end)`,
2. test is not just `[a_end, T)`,
3. test includes a normal region plus the anomaly region, or the anomaly region plus a normal region, inside one continuous segment.

The exact boundary policy can be configurable, but the mode must always try to preserve mixed labels in test.

Minimum valid outcome:

- `test` has at least one `0`,
- `test` has at least one `1`.

If the raw sequence is too short to satisfy this cleanly, the loader must fail loudly instead of silently building a single-class benchmark test set.

### 4.4 Allowed research-only modes

`pre_vs_anomaly` and `pre_vs_post` may remain in the codebase only under these conditions:

1. they are documented as protocol-specific research slices,
2. they are not silently presented as standard anomaly-detection benchmarks,
3. their audit report must explicitly explain why their metrics are degenerate or not comparable.

## 5. Window and Label Semantics

### 5.1 Pointwise labels remain primary

The evaluator in this codebase reconstructs overlapping window scores back into pointwise timeline scores.

Therefore the main ground truth must remain pointwise labels on the original timeline.

Window-level labels may still exist for training convenience, but they are not the primary truth object for final evaluation.

### 5.2 Requirement on test windows

The test timeline should be cut into overlapping windows after the test segment is defined.

It is acceptable that some windows lie entirely inside an anomaly span and therefore have all-ones label vectors.

This is not automatically wrong.

What matters is the full test timeline.

If the full test timeline still contains both normal and anomalous timesteps, then some all-one windows inside it are normal and expected.

The real problem is when the entire test timeline becomes all-one or all-zero.

### 5.3 Coverage reconstruction rule

The evaluator must continue to reconstruct scores by placing each window score sequence back onto its original timeline indices and averaging overlaps per timestep.

This reconstruction direction must remain chronological:

- earlier timesteps stay earlier,
- later timesteps stay later.

No shuffling or label-aware reordering is allowed in evaluation reconstruction.

## 6. Required Evaluation and Audit Outputs

Every evaluation run must save enough metadata to answer these questions immediately:

1. How long was the raw test timeline?
2. How many points were actually evaluated?
3. Was evaluation truncated?
4. Did the test labels contain both classes?
5. Was this a protocol-specific AnomalyArchive mode?

The saved outputs must include at least:

- `raw_num_points`
- `evaluated_num_points`
- `is_truncated_evaluation`
- `label_regime`
- `n_pos`
- `n_neg`
- `positive_ratio`
- `comparison_mode` when relevant
- `comparison_mode_explanation` when relevant

These fields should be visible in:

1. `evaluation_metrics.json`
2. `evaluation_protocol_audit.json`
3. `evaluation_protocol_audit.md`

## 7. Required Runtime Warnings and Failures

### 7.1 Single-class test labels

If the raw test labels contain only one class, the pipeline must raise a strong warning.

If the caller is using a benchmark-evaluation mode, the pipeline should fail loudly by default.

If the caller explicitly requested a research-only mode such as `pre_vs_anomaly` or `pre_vs_post`, the pipeline may continue but must mark the run as protocol-special.

### 7.2 Reconstructed single-class evaluated labels

If the reconstructed pointwise evaluated labels contain only one class, the evaluator must flag this explicitly in the saved artifact.

This is different from raw split labels.

It can happen because of:

1. single-class raw test design,
2. truncated evaluated coverage,
3. both.

### 7.3 Truncated smoke evaluation

If `max_test_windows` is set and test coverage is shorter than the raw test timeline, the run must be marked as truncated.

This should never be silent.

The wording should clearly say:

- only part of the test timeline was evaluated,
- later test points were not scored,
- this run is not directly comparable to full evaluation.

### 7.4 Visualization from truncated artifacts

If a visualization is requested from a truncated artifact, the visualization must show the covered region clearly.

If strict full-coverage visualization is requested, the command should fail.

## 8. Required Human-Readable Explanation

The report must explain `pre_vs_anomaly` and `pre_vs_post` in plain English.

The explanation must be short and direct.

Required message for `pre_vs_anomaly`:

- train and val come from before the anomaly,
- test is only the anomaly segment,
- therefore all test labels become `1`,
- this is not a standard benchmark test.

Required message for `pre_vs_post`:

- train and val come from before the anomaly,
- test is only the segment after the anomaly,
- therefore test labels become `0` or almost all `0`,
- this is not a standard benchmark test.

This explanation must appear both in:

1. the Markdown audit report,
2. visualization metadata for AnomalyArchive runs.

## 9. Implementation Scope

### 9.1 Files expected to change

The minimal implementation is expected to focus on:

- `src/data/datasets/anomaly_archive.py`
- `src/data/loaders.py`
- `src/analysis/evaluation_protocol_audit.py`
- `src/engine/evaluator.py`
- `src/metrics/pointwise.py`
- `scripts/evaluate.py`
- `scripts/visualize_evaluation_results.py`
- tests covering loader semantics, coverage metadata, and degenerate metric regimes

### 9.2 Files not expected to change

These fixes should not require behavior changes in:

- `src/models/redlamp_mlp_baseline.py`
- `src/models/thesis_multitask.py`

## 10. Acceptance Criteria

The protocol-fix work is accepted only if all of the following become true.

### 10.1 SMD full evaluation

For a normal SMD full config:

- raw test labels remain sparse and mixed,
- evaluated coverage matches the full raw test timeline,
- audit report says coverage is not truncated.

### 10.2 SMD smoke evaluation

For an SMD smoke config with `max_test_windows`:

- audit report says coverage is truncated,
- evaluation metadata stores raw and evaluated point counts,
- visualization shows only the covered prefix as evaluated.

### 10.3 AnomalyArchive benchmark mode

For the new benchmark mode:

- test contains both normal and anomalous timesteps,
- train and val remain strictly earlier than test,
- no single-class benchmark test is silently produced.

### 10.4 AnomalyArchive research-only modes

For `pre_vs_anomaly`:

- audit report says test is all-positive,
- report explains why metrics are degenerate.

For `pre_vs_post`:

- audit report says test is all-negative or nearly all-negative,
- report explains why metrics are degenerate.

### 10.5 Metric artifact clarity

Every evaluation artifact must let a reader answer, without opening source code:

1. Was this full coverage or partial coverage?
2. Was the label regime mixed, all-zero, or all-one?
3. Was this a benchmark mode or a protocol-special mode?

## 11. Recommended Implementation Order

The safest implementation order is:

1. finalize the new benchmark split contract for `anomaly_archive`,
2. add tests for mixed-label benchmark test construction,
3. keep `pre_vs_anomaly` and `pre_vs_post` but downgrade them to clearly marked research-only modes,
4. harden evaluator and audit artifacts for truncation and single-class regimes,
5. harden visualization so raw truth and evaluated coverage are always visible,
6. verify smoke and full runs separately.

This order reduces the chance of fixing only the symptoms while leaving the protocol problem untouched.

## 12. Final Position

The current main issue is not that the formulas for recall, precision, PR-AUC, ROC-AUC, or VUS are broken.

The current main issue is that some dataset configurations feed those formulas unsuitable test regimes.

So the correct repair is not “change the metric formula first.”

The correct repair is:

1. define proper test timeline semantics,
2. fail loudly on degenerate regimes,
3. label truncated smoke runs honestly,
4. only then compare model quality.
