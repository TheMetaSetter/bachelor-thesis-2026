---
date: 2026-09-04T13:48:30+07:00
planner: OpenAI Codex
topic: "Use raw-input-space MSE for thresholding and prediction"
status: ready
revision: 974af2b3a3d075f5cd4f3368f2cb584a5a8a3720
branch: dev
related_research: documents/logs/2026-09-04/research/research-raw-input-space-mse-score-change-surface.md
---

# Implementation Plan: Use raw-input-space MSE for thresholding and prediction

## Summary

Change the operational anomaly score from normalized/calibrated output to simple MSE computed in the original sensor-value space. Use this raw-input-space MSE for offline thresholds, online thresholds, EWMA, triage, and point/window predictions. Keep normalized MSE only as an explicit diagnostic field.

## Request

Use raw-input-space MSE as the anomaly score for thresholding and prediction. Do not use the calibrated sigmoid score. Support both point-level and window-level scores, preserve the MC per-sample-MSE averaging rule, and validate the result on synthetic validation data.

## Current state

- `SequenceStandardScaler` transforms the raw sequence before `WindowDataset` emits `batch["x"]` (`src/data/scalers.py:17-58`; `src/data/loaders.py:231-298`).
- The model computes point and window reconstruction MSE against this scaled `batch["x"]` (`src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:186-196`; `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py:241-260`).
- The model can transform point scores with the shifted-and-scaled logistic sigmoid through `ThesisMultitaskStateMixin` (`src/models/thesis_multitask_impl/thesis_multitask_state_mixin.py:29-44`).
- Offline evaluation consumes one `point_scores` timeline and thresholds it (`src/engine/evaluator.py:415-502`).
- Online calibration and EWMA also consume the model point scores; its `input_window_score` is normalized-space reconstruction MSE (`src/engine/online_tta/online_calibration.py:61-133`; `src/engine/online_tta/online_engine_window_metrics.py:150-171`).
- Threshold schema version 4 requires sigmoid calibration fields for THESIS (`src/protocols/threshold_artifact.py:64-93`, `:322-339`).

## Desired end state

For input `x_scaled` and reconstruction `reconstruction_scaled`, use the fitted training scaler to obtain `x_raw` and `reconstruction_raw`:

```text
raw_point_mse[t] = mean_feature((x_raw[t] - reconstruction_raw[t])²)
raw_window_mse = mean_time(raw_point_mse)
```

For stochastic inference, compute these values for every reconstruction sample and then average the MSE values over MC samples. Do not compute MSE from the averaged reconstruction.

The operational path must use raw-input-space MSE for:

- offline point threshold and point prediction;
- online stride-1 point threshold and EWMA point prediction;
- input-window threshold and window triage;
- online event records and persisted prediction state.

The normalized MSE remains available only under explicit diagnostic names such as `normalized_input_point_mse` and `normalized_input_window_mse`. New threshold artifacts must identify `score_space: raw_input` and an identity transform. Old sigmoid artifacts remain historical and must not be silently reused.

## Scope

### In scope

- Raw-space inverse transformation and point/window MSE calculation.
- Offline clean-validation threshold calibration and synthetic/test prediction.
- Online threshold calibration, EWMA, triage, prediction, and event records.
- Threshold-artifact schema/provenance and protocol configuration.
- Tests, synthetic validation, histogram outputs, and score-space documentation.

### Out of scope

- Changing the reconstruction training loss; it remains computed on the model's training input representation.
- Reweighting channels or inventing a normalized raw-space variant; simple MSE uses equal feature weights.
- Changing RedLamp or other baselines unless a later request explicitly includes them.
- Deleting historical sigmoid artifacts or rewriting historical benchmark results.

## Evidence

- `src/data/scalers.py:36-58` — transform exists, inverse transform does not.
- `src/data/loaders.py:150-193`, `:231-298` — both raw and scaled sequences exist, but windows expose scaled `x`.
- `src/engine/checkpoint.py:185-215` — checkpoint already stores scaler state.
- `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:186-250` — MC reconstruction and normalized MSE are available.
- `src/engine/evaluator.py:464-502` — offline metrics use one point-score payload and one threshold.
- `src/engine/online_tta/online_calibration.py:61-133` — online calibration computes normalized input-window MSE and EWMA input.
- `src/engine/online_tta/online_engine_window_metrics.py:110-171`, `:207-277` — online event fields and predictions consume the current point/window scores.
- `src/protocols/threshold_artifact.py:64-93`, `:282-430` — current THESIS artifact requires sigmoid-specific calibration fields.
- `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml:1-16` — current protocol defines window, quantile, stride, and EWMA values but no score-space selector.
- `documents/spec/full-spec-v3.md:517-620`, `:899-909`, `:1013` — current specification defines MC raw-MSE semantics but still assigns official point score to the transformed field and requires schema version 4 online artifacts.

## Implementation approach

Use one scorer at the evaluator/online boundary instead of placing the scaler inside the model. The checkpoint already owns scaler state, while the model currently owns reconstruction and the downstream boundaries own thresholding and prediction. The scorer will consume the scaled batch and reconstruction samples, inverse-transform both, and return raw point/window MSE.

The model's existing top-level outputs remain available for compatibility and diagnostics. The evaluator and online engine will stop using those fields for operational decisions and will use the explicit raw-input score returned by the scorer. The new protocol will not load or apply sigmoid calibration.

## Phase 1: Lock the raw-input score contract

### Goal

Create one normative definition for the new operational score and remove ambiguity between “raw” meaning uncalibrated and “raw” meaning unscaled sensor units.

### Changes

#### 1. Define the score contract

- **Files:** proposed new `documents/spec/full-spec-v4.md`; modify `documents/spec/anomaly-score-designs-and-notation.md`, `documents/spec/online_tta_terminology_ontology.md`, and `documents/spec/offline_pretraining_terminology_ontology.md`.
- **Change:** Define `raw_input_point_mse`, `raw_input_window_mse`, `normalized_input_point_mse`, and `normalized_input_window_mse`. State that `point_scores` and window/triage operational scores refer to raw-input MSE in the new protocol, with identity transformation and no sigmoid.
- **Reason:** `full-spec-v3` currently gives `point_scores` sigmoid semantics while the requested protocol requires simple raw MSE.
- **Dependencies:** Compare v1–v3 terminology before creating v4. Add a terminology-change table marking renamed, unchanged, and newly introduced objects.

#### 2. Add protocol selection and score identity

- **File:** `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml`.
- **Change:** Add explicit raw-input score settings, including `score_space: raw_input` and `point_score_transform: identity`; keep existing window size, quantiles, strides, and EWMA weights unchanged.
- **Reason:** Threshold calibration and prediction must be auditable and must not depend on implicit defaults.
- **Dependencies:** The threshold artifact must copy the resolved score identity.

### Verification

#### Automated

- [ ] Run `.venv/bin/python -m pytest -q tests/benchmarks/test_benchmark_protocol_config.py tests/engine/test_threshold_artifact.py tests/online/test_threshold_artifact.py` and confirm the protocol and artifact tests pass after the contract change.
- [ ] Add a terminology/config assertion that the new protocol resolves to `raw_input` and `identity`.

#### Manual

- [ ] Read the v4 score table and confirm that no threshold or prediction definition refers to calibrated sigmoid output.

### Risks

- **Risk:** Historical v3 terminology is overwritten and old results become ambiguous. **Mitigation:** create v4 and preserve v3 unchanged; verify that historical artifacts remain readable only under their original schema.

## Phase 2: Add scaler-aware raw MSE primitives

### Goal

Produce tested raw point/window MSE tensors from normalized tensors without changing the model training objective.

### Changes

#### 1. Add inverse transformation

- **File:** `src/data/scalers.py`, `SequenceStandardScaler`.
- **Change:** Add a tensor inverse-transform operation that applies `raw = scaled * clamp(std, epsilon) + mean` only to active features; inactive features remain unchanged because `transform_sequence` leaves them unchanged.
- **Reason:** The dataset exposes scaled windows, and the checkpoint already provides the fitted scaler state.
- **Dependencies:** Preserve device and dtype behavior; reject use before `load_state_dict`/`fit`.

#### 2. Add the shared reconstruction scorer

- **File:** proposed `src/protocols/reconstruction_scores.py`.
- **Interface:** A scorer consumes `input_scaled [B,L,D]`, `reconstruction_scaled [B,L,D]` or `reconstruction_samples_scaled [B,M,L,D]`, and a fitted `SequenceStandardScaler`; it produces point scores `[B,L]` and window scores `[B]` in the requested input space.
- **Change:** Implement raw-input scoring by inverse-transforming both operands before feature-wise MSE. For MC samples, return the mean of per-sample point/window MSE. Keep normalized scoring available for diagnostics through the same explicit interface.
- **Reason:** `src/protocols/point_scores.py` owns timeline aggregation, not reconstruction math; a focused scorer prevents duplicated formulas in offline and online paths.
- **Dependencies:** The scorer must accept post-injection `batch["x"]`, so synthetic anomalies remain part of the measured input.

### Verification

#### Automated

- [ ] Add scaler round-trip tests for active and inactive features.
- [ ] Add hand-computed tests proving raw point MSE and raw window MSE use the inverse-transformed values.
- [ ] Add MC tests proving `mean(MSE(sample_m))` differs correctly from `MSE(mean(sample_m))`.
- [ ] Run the scaler and model score contract tests with `.venv/bin/python -m pytest -q tests/data tests/models`.

#### Manual

- [ ] Inspect one batch and verify that raw input and raw reconstruction are in sensor units before subtraction.

### Risks

- **Risk:** Applying inverse transform twice changes score scale. **Mitigation:** pass an explicit `input_space` to the scorer and test normalized/raw values on the same toy tensor.

## Phase 3: Make offline evaluation operationally raw-space

### Goal

Use raw-input point/window MSE for offline threshold calibration, predictions, metrics, and exported score artifacts.

### Changes

#### 1. Extend evaluator score payloads

- **File:** `src/engine/evaluator.py`, payload validation, overlap reconstruction, `evaluate`.
- **Change:** Pass the fitted scaler or a scorer context into evaluation. Compute raw point/window scores from the post-injection batch and model reconstruction samples. Use raw point scores for `resolve_evaluation_threshold` and metrics. Preserve normalized scores under explicit diagnostic fields.
- **Reason:** The evaluator currently thresholds only `outputs["point_scores"]`, which can be calibrated or normalized.
- **Dependencies:** `accumulate_pointwise_window_payload`, `reconstruct_pointwise_records_from_window_payload`, and `extract_covered_pointwise_arrays` must aggregate raw and diagnostic timelines with identical overlap rules.

#### 2. Update offline benchmark orchestration

- **File:** `scripts/benchmarks/run_thesis_offline_benchmark.py`, artifact-input collection, evaluation conversion, threshold building, and export.
- **Change:** Retain the checkpoint scaler in `artifact_inputs`; calibrate offline and online thresholds from raw clean-validation scores; evaluate clean, synthetic validation, and test with raw scores; export raw point/window scores and `score_space` provenance separately from normalized diagnostics.
- **Reason:** Current orchestration stores scaled validation sequences and exports only one point-score file.
- **Dependencies:** Synthetic injection must happen before the scorer inverse-transforms `batch["x"]`.

### Verification

#### Automated

- [ ] Add evaluator tests for raw-score overlap averaging and raw threshold selection.
- [ ] Add benchmark export tests for raw score files, raw threshold metadata, and retained normalized diagnostics.
- [ ] Run the existing offline evaluator and benchmark test modules with `.venv/bin/python -m pytest -q tests/evaluation tests/benchmarks`.

#### Manual

- [ ] Inspect one clean and one synthetic window; confirm the threshold is compared with raw MSE and that normal/anomalous labels remain separate from score values.

### Risks

- **Risk:** Synthetic anomalies are scored against the pre-injection raw sequence. **Mitigation:** calculate raw input from the actual batch after injection and add a regression test with a known perturbation.

## Phase 4: Replace threshold-artifact sigmoid requirements for the new protocol

### Goal

Make raw-input score identity and thresholds explicit in persisted artifacts, while preserving historical artifact compatibility.

### Changes

#### 1. Add a raw-score artifact schema

- **File:** `src/protocols/threshold_artifact.py`.
- **Change:** Add a new schema version for raw-input MSE artifacts. Persist `score_space: raw_input`, `point_score_transform: identity`, point/window score definitions, threshold source splits, and checkpoint/config provenance. Do not require `point_score_c`, `point_score_tau`, or sigmoid estimator fields for the raw schema.
- **Reason:** Current schema version 4 requires sigmoid fields and cannot prove that a threshold was fitted in raw units.
- **Dependencies:** Keep v3/v4 validation for historical artifacts; the new raw protocol must reject an artifact whose score space or transform does not match.

#### 2. Stop loading sigmoid calibration for raw runs

- **Files:** `src/engine/online_tta/online_engine_run.py:103-194`; model calibration setup and threshold-artifact resolution.
- **Change:** Require a matching raw-score artifact, leave point-score calibration unset for raw runs, and fail with a clear mismatch error when a legacy sigmoid artifact is supplied.
- **Reason:** The selected operational score must not pass through calibrated sigmoid.
- **Dependencies:** Historical benchmark replay may continue to use its original artifact/schema outside the new raw protocol.

### Verification

#### Automated

- [ ] Add artifact round-trip tests for raw schema fields.
- [ ] Add rejection tests for normalized-space thresholds, sigmoid transforms, missing score identity, and artifact/checkpoint mismatch.
- [ ] Run `tests/engine/test_threshold_artifact.py`, `tests/online/test_threshold_artifact.py`, and related benchmark artifact tests with `.venv/bin/python -m pytest -q`.

#### Manual

- [ ] Open a generated artifact and verify that it contains raw score identity and no required sigmoid calibration values.

### Risks

- **Risk:** Existing online configs point to old schema-4 artifacts. **Mitigation:** regenerate raw artifacts under new output directories and reject old artifacts instead of silently reusing them.

## Phase 5: Make online EWMA, triage, and prediction raw-space

### Goal

Ensure every online operational decision uses the same raw-input MSE definition and matching raw thresholds.

### Changes

#### 1. Add raw scores to online calibration

- **File:** `src/engine/online_tta/online_calibration.py`.
- **Change:** Pass scaler context into batch-score collection; compute raw point scores and raw input-window scores from MC reconstruction samples; fit offline point, online EWMA, and input-window thresholds from raw clean-validation values.
- **Reason:** Current `input_scores` and model point scores are normalized/calibrated values.
- **Dependencies:** Reuse the Phase 2 scorer and keep causal stride-1 indexing unchanged.

#### 2. Select raw scores in event construction

- **Files:** `src/engine/online_tta/online_engine_window_metrics.py`, `online_engine_window_core.py`, `online_engine_step.py`.
- **Change:** Feed raw point MSE into `update_window_point_ewma`; use raw window MSE for input-window admission/triage; write raw operational scores and explicit normalized diagnostic fields to events and step records; update fallback paths so they cannot silently use calibrated `outputs["point_scores"]`.
- **Reason:** The current event path directly consumes `outputs["point_scores"]` and `input_window_score`.
- **Dependencies:** Preserve absolute-index EWMA state and prediction shapes; only the score values and score identity change.

#### 3. Carry scaler and score identity through runtime context

- **Files:** `src/engine/online_tta/online_engine_run.py`, `src/data/stream.py` if a stream contract must carry context.
- **Change:** Load scaler state from the checkpoint/data bundle, pass it to the online scorer, and persist score identity with runtime/threshold state where required.
- **Reason:** Online streams intentionally expose scaled model inputs, while raw scoring needs the fitted scaler.
- **Dependencies:** Do not add raw tensors to model batches unless a concrete caller requires them.

### Verification

#### Automated

- [ ] Add online calibration tests proving raw values are used for point threshold, EWMA threshold, and input-window threshold.
- [ ] Add online event tests proving point/window predictions change according to raw thresholds and that normalized diagnostics do not control decisions.
- [ ] Add runtime-state tests for score-identity mismatch and continuation with a matching raw artifact.
- [ ] Run the online test modules with `.venv/bin/python -m pytest -q tests/online`.

#### Manual

- [ ] Trace one online event from model output to EWMA and prediction; confirm every comparison uses raw-input score units.

### Risks

- **Risk:** A fallback path bypasses the new scorer and uses normalized `outputs["point_scores"]`. **Mitigation:** centralize operational score selection and add a test that fails when only the normalized field is available.
- **Risk:** Raw channels have different physical magnitudes. **Mitigation:** document equal-feature raw MSE explicitly and report this limitation; do not add weighting without a new score definition.

## Phase 6: Validate on synthetic windows and update visualization/reporting

### Goal

Demonstrate that raw-input MSE separates normal/anomalous points and windows, and that the histogram uses the operational score rather than sigmoid output.

### Changes

#### 1. Add end-to-end regression coverage

- **Files:** proposed `tests/evaluation/test_raw_input_mse_scores.py`; modify `tests/evaluation/test_evaluator_thresholding.py`, `tests/benchmarks/test_thesis_offline_artifact_exports.py`, `tests/online/test_online_calibration_contract.py`, `tests/online/test_online_ewma_threshold.py`, and `tests/online/test_online_tta_triage.py`.
- **Change:** Verify score-space metadata, anomalous/normal labels, threshold comparisons, point/window outputs, and artifact exports.
- **Reason:** Unit tests cannot prove that synthetic injection, inverse transformation, thresholding, and prediction use the same score.

#### 2. Re-run the validation set

- **Data:** `machine-1-6`, `machine-3-4`, and `machine-3-9`.
- **Change:** First run one complete smoke combination, then run the three requested synthetic validation windows using the raw protocol. Save only report-ready score summaries, thresholds, metrics, provenance, and required plots under the canonical output hierarchy.
- **Reason:** The repository requires one end-to-end combination before expanding a benchmark run.

#### 3. Produce histograms

- **Change:** Plot separate histograms for raw point MSE and raw window MSE, marking anomalous and normal observations separately and showing the fitted threshold. Do not plot calibrated sigmoid values as the operational score.
- **Dependencies:** Use exported raw score artifacts and labels from the same run.

### Verification

#### Automated

- [ ] Run the focused score, evaluator, benchmark, and online test set after all phases with `.venv/bin/python -m pytest -q` and record the exact command/output.
- [ ] Validate that generated score arrays are finite, point/window shapes match the contract, and threshold artifacts report `raw_input`.

#### Manual

- [ ] Inspect histograms for all three machines and confirm normal/anomalous distributions and threshold annotations use raw MSE values.
- [ ] Confirm the final report distinguishes anomalous point/normal point and anomalous window/normal window.

### Risks

- **Risk:** Machine-specific raw units produce incomparable score magnitudes. **Mitigation:** calibrate and report thresholds per entity, retain entity/checkpoint provenance, and do not compare raw threshold values across machines without an explicit normalization rule.

## Testing strategy

Use narrow tests first, then integration tests:

1. Scaler inverse-transform and score arithmetic.
2. MC aggregation and synthetic post-injection behavior.
3. Offline overlap aggregation and thresholding.
4. Threshold artifact schema and mismatch rejection.
5. Online calibration, EWMA, triage, prediction, and runtime continuation.
6. One end-to-end smoke combination, followed by the three requested machines.

The tests must assert both numeric values and score-space identity. Shape-only tests are insufficient.

## Migration and rollback

- Keep v3/v4 threshold artifacts readable for historical experiments, but do not use them for the new raw protocol.
- Generate new raw artifacts under new stage/output paths with a new schema version and provenance.
- If raw validation fails, roll back by selecting the previous protocol/artifact explicitly; do not overwrite old score files or threshold files.
- Existing normalized diagnostic arrays remain useful for comparison but must not be used for raw prediction decisions.

## Documentation

- Add the new score contract and terminology mapping in the next specification version.
- Update protocol and online ontology documents so `raw_input` means original sensor units and `normalized_input` means post-standardization values.
- Update benchmark artifact/plot documentation with score-space metadata and the four normal/anomaly categories.
- Record the exact protocol, checkpoint scaler provenance, threshold values, and raw-score histogram paths in the experiment log.

## Final verification

- [ ] A newly generated threshold artifact declares `score_space: raw_input` and `point_score_transform: identity`.
- [ ] Offline point/window thresholding and prediction use raw-input MSE.
- [ ] Online EWMA, triage, and prediction use raw-input MSE.
- [ ] No raw-protocol execution loads calibrated sigmoid parameters.
- [ ] Synthetic validation for `machine-1-6`, `machine-3-4`, and `machine-3-9` exports separate raw point/window scores and histograms with normal/anomalous labels.
- [ ] Existing historical artifacts remain unchanged and are not silently reused.

## Assumptions and non-blocking uncertainties

- Raw input space means the original sensor-value units before standardization.
- Simple MSE averages all feature channels with equal weight.
- The model continues to emit its current fields for compatibility; the evaluator and online engine own operational raw-score selection.
- The exact new schema integer and final field spelling must follow the repository's schema-version convention when implementation begins; the plan requires a new schema rather than modifying old artifacts in place.
