---
date: 2026-09-05T17:09:13+07:00
topic: "Synthetic-validation normal-score q99 threshold rerun"
status: ready
revision: 5529a0c3f1ab9f4b7f013543aa7e61922b74b56d
source_structure: /Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/documents/logs/2026-09-05/structure/structure-synthetic-validation-threshold-rerun.md
related_documents:
  - /Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/documents/logs/2026-09-05/research/research-synthetic-validation-threshold-change-surface.md
  - /Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/documents/logs/2026-09-05/plan/plan-synthetic-validation-threshold-rerun.md
---

## Execution status

Implemented the local code, protocol, artifact, output-root, and regression-test
steps in this document. The remote checkpoint matrix remains unexecuted; its
read-only preflight and evaluation commands are in
`detail-synthetic-validation-threshold-rerun-command.md`.

# Detailed Implementation: Synthetic-validation normal-score q99 threshold rerun

## Summary

The implementation will add one opt-in raw-input protocol. That protocol will compute one point threshold from finite, covered synthetic-normal scores at q99, reuse it for synthetic and test point metrics, preserve the existing clean-validation default, and write artifacts below a new output root.

## Source structure

The approved structure has five phases: establish the opt-in contract; implement synthetic-normal q99 selection; isolate artifacts; add regression coverage; and prepare the cloud command. The structure is approved because the user requested this detail document immediately after requesting the structure.

## Current state

`/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_thesis_offline_benchmark.py:452-544` selects the offline point threshold from clean-validation scores. `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/evaluator.py:421-433` and `:537-669` already support synthetic labels, covered points, and an explicit threshold. `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_thesis_offline_benchmark.py:923-970` writes all current artifacts below the experiment-configured output directory.

## Desired end state

The new protocol performs this sequence for the active raw-input path:

```text
val_synth scores
  -> covered synthetic normal points
  -> finite scores
  -> q99 point threshold
  -> thresholded val_synth metrics + thresholded test metrics
```

The existing protocol keeps its current clean-validation sequence. The top-level artifact calibration contract remains `clean_validation`; the offline point threshold record carries the separate source `synthetic_validation_normal` for the new protocol.

## Scope

### In scope

- Optional protocol source field `offline_point_threshold_source_split`.
- Synthetic-normal q99 selection for raw-input offline evaluation.
- One fixed point threshold per run.
- Optional evaluation-only output-directory override.
- Threshold-source provenance and focused tests.
- A command plan for all 18 current Stage-B best checkpoints.

### Out of scope

- Model, optimizer, Stage-A, or Stage-B training changes.
- Online EWMA threshold recalibration.
- Changing the existing `smd_window20_cleanval_q99_ewma09.yaml` behavior.
- Remote execution during implementation planning.

## Evidence

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_thesis_offline_benchmark.py:452-544` — current clean-validation threshold flow.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_thesis_offline_benchmark.py:576-631` — covered point payload construction.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_thesis_offline_benchmark.py:655-762` — threshold artifact construction.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_thesis_offline_benchmark.py:863-994` — evaluation-only output and CLI path.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/protocols/smd_benchmark_protocol.py:39-57` — locked default protocol rules.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/protocols/threshold_artifact.py:345-505` — threshold-artifact builder and schema fields.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/tests/benchmarks/test_thesis_offline_artifact_exports.py:503-601` — current evaluation call-order test.

## Phase 1: The opt-in contract is explicit

### Goal

Allow the new source and output root without changing the old protocol or old callers.

### Dependencies

- Research note status `complete`.
- Current protocol and artifact validators.

### Detailed changes

#### 1. Validate the optional threshold-source field

- **File:** `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/protocols/smd_benchmark_protocol.py`
- **Symbol:** `validate_protocol_config`
- **Current responsibility:** Validates locked SMD protocol values and requires `offline_threshold_split == clean_validation`.
- **Change:** Read optional `offline_point_threshold_source_split`; accept only `clean_validation` or `synthetic_validation_normal`; treat an absent field as `clean_validation`.
- **Reason:** The legacy field remains unchanged while the new point-threshold source gets an explicit name.
- **Inputs:** Mapping with the existing required keys and, optionally, the new source key.
- **Outputs:** Validation success or a `ValueError` for an unsupported source.
- **Errors:** Reject empty or unsupported source values before evaluation starts.
- **Dependencies:** Existing protocol tests and all callers of `validate_protocol_config`.
- **Compatibility:** Existing configs without the new key validate exactly as before.

#### 2. Add the new protocol file

- **File:** **Proposed new file:** `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/configs/protocol/smd_window20_synthnormal_q99_ewma09.yaml`
- **Symbol:** YAML protocol mapping.
- **Current responsibility:** No file currently exists at this path.
- **Change:** Copy the active raw-input protocol values and add `offline_point_threshold_source_split: synthetic_validation_normal`; use a distinct `protocol_name`.
- **Reason:** The existing clean-validation YAML remains a stable historical/default entry point.
- **Inputs:** Values from `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/configs/protocol/smd_window20_cleanval_q99_ewma09.yaml`.
- **Outputs:** A parseable opt-in protocol mapping.
- **Errors:** The validator rejects missing locked values or an invalid source.
- **Dependencies:** Phase 1 step 1.
- **Compatibility:** Do not edit the existing clean-validation YAML.

#### 3. Extend threshold-artifact provenance without changing schema version

- **File:** `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/protocols/threshold_artifact.py`
- **Symbol:** `build_threshold_artifact` and `validate_threshold_artifact`.
- **Current responsibility:** Uses one `calibration_split` for top-level metadata and every threshold record; raw artifacts use schema version 5.
- **Change:** Add optional `offline_point_threshold_source_split`, defaulting to `calibration_split`; write it to the offline point threshold record and, when present, validate that it is a non-empty supported source. Keep top-level `calibration_split` unchanged.
- **Reason:** The new offline point source differs from the existing online calibration source, but existing artifact readers and schema version 5 must remain valid.
- **Inputs:** Existing builder arguments plus an optional source string.
- **Outputs:** An artifact whose `thresholds.offline_point.source_split` identifies the actual point source.
- **Errors:** Reject unsupported or inconsistent optional source metadata.
- **Dependencies:** Phase 1 step 1 and runner threshold construction.
- **Compatibility:** Existing callers omit the argument and produce byte-compatible field values and schema version 5 behavior.

#### 4. Add an optional evaluation-only output root

- **File:** `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_thesis_offline_benchmark.py`
- **Symbol:** `run_thesis_offline_benchmark`, `_build_evaluation_only_run`, and `main`.
- **Current responsibility:** Uses `experiment_config["output_dir"]` for evaluation-only manifests, exports, retention, reports, and has no output override.
- **Change:** Add optional `output_dir` to the Python API and `--output-dir` to the CLI; allow the override only with `--evaluation-only`; resolve the configured directory when the option is absent.
- **Reason:** The requested rerun must not overwrite existing benchmark outputs.
- **Inputs:** Optional path string and existing experiment config.
- **Outputs:** Evaluation-only manifest and artifacts below the effective path.
- **Errors:** Reject `--output-dir` for training flow or an empty path.
- **Dependencies:** Existing artifact and retention writers.
- **Compatibility:** Existing calls and commands omit the option and use the configured directory.

### Tests

#### Protocol and artifact compatibility

- **Location:** `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/tests/benchmarks/test_benchmark_protocol_config.py` and `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/tests/engine/test_threshold_artifact.py`
- **Level:** Unit/contract.
- **Setup:** Load the legacy YAML and construct a raw schema version 5 artifact.
- **Action:** Validate with no new field, then validate with `synthetic_validation_normal` and an unsupported source.
- **Expected result:** Legacy and supported opt-in inputs pass; unsupported input fails; top-level calibration remains `clean_validation`.
- **Edge cases:** Missing optional field and empty source.

#### Output override

- **Location:** `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/tests/benchmarks/test_thesis_offline_artifact_exports.py`
- **Level:** Integration-style wrapper test.
- **Setup:** Use the existing fake manifest/collector and two temporary directories.
- **Action:** Call evaluation-only with an explicit output root.
- **Expected result:** Report, scores, thresholds, and retention files appear only below the override root.
- **Edge cases:** Omitted override uses the experiment-configured root; non-evaluation flow rejects the override.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest tests/benchmarks/test_benchmark_protocol_config.py tests/engine/test_threshold_artifact.py tests/benchmarks/test_thesis_offline_artifact_exports.py -q` — legacy and contract tests pass.

#### Manual

- [ ] Compare the resolved old and new protocol files — only the new source identity and protocol name differ.

### Risks and recovery

- **Risk:** The optional source changes legacy artifacts. **Mitigation:** default to `calibration_split` and keep schema version 5. **Verification:** load and compare a legacy fixture.
- **Risk:** Output override is applied to training accidentally. **Mitigation:** reject it outside evaluation-only. **Recovery:** rerun without the option using the original config.

### Complete when

- Both protocol files validate.
- Existing artifact tests pass.
- The new optional output root is accepted only for evaluation-only runs.

## Phase 2: Synthetic-normal q99 drives offline point metrics

### Goal

Select the requested point threshold from covered, normal, finite `val_synth` scores and reuse it for synthetic/test point metrics.

### Dependencies

- Phase 1 source field and raw-input opt-in protocol.

### Detailed changes

#### 1. Add a finite synthetic-normal threshold helper

- **File:** `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/thresholding.py`
- **Symbol:** Proposed `select_synthetic_validation_normal_point_threshold`.
- **Current responsibility:** `select_clean_validation_point_threshold` selects a q-quantile from clean scores.
- **Change:** Add a small helper that receives synthetic point scores and synthetic point labels, keeps labels equal to zero, keeps finite scores, validates the quantile, and returns q99 or the requested quantile.
- **Reason:** The filtering rule becomes a named, testable contract rather than inline masking.
- **Inputs:** Equal-length numeric score and label arrays; quantile in `(0, 1]`.
- **Outputs:** One finite `float` threshold.
- **Errors:** Raise `ValueError` when array lengths differ or no covered normal finite score remains.
- **Dependencies:** NumPy and existing quantile validation.
- **Compatibility:** Existing clean and generic helper behavior remains unchanged.

#### 2. Permit an unthresholded synthetic collection pass

- **File:** `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_thesis_offline_benchmark.py`
- **Symbol:** `_evaluate_named_split`.
- **Current responsibility:** Requires a `float` point threshold and always labels it `clean_validation_quantile`.
- **Change:** Accept `point_score_threshold: float | None` and an optional `threshold_source`; include those evaluator keyword arguments only when supplied. Keep `evaluation_stage="val_synth"` for raw synthetic evaluation.
- **Reason:** The first synthetic pass needs scores before the new threshold exists.
- **Inputs:** Existing split arguments plus an optional threshold and source.
- **Outputs:** Existing evaluator output shape.
- **Errors:** Preserve evaluator validation for missing labels, score-space mismatch, and missing scaler.
- **Dependencies:** Existing `Evaluator.evaluate` interface.
- **Compatibility:** Existing callers pass a float and receive the same threshold source.

#### 3. Select the synthetic threshold in the raw split flow

- **File:** `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_thesis_offline_benchmark.py`
- **Symbol:** `_evaluate_offline_benchmark_splits`.
- **Current responsibility:** Selects `clean_threshold`, then evaluates `val_synth` and test with it.
- **Change:** Read the optional source field. If it is `synthetic_validation_normal`, run `val_synth` once without a fixed point threshold, convert records with `_evaluation_outputs_to_score_payload`, select q99 from its normal covered scores, run `val_synth` again with the fixed threshold, and evaluate test with the same threshold and source label. Leave the current clean branch unchanged.
- **Reason:** The threshold must be based on synthetic normal scores before synthetic/test metrics are computed.
- **Inputs:** Raw-input evaluator, `val_synth` loader, quantile, and scaler.
- **Outputs:** Final clean, synthetic, and test outputs plus the selected point threshold and source for artifact export.
- **Errors:** Reject synthetic source when raw-input scoring or the required scaler is unavailable; propagate the no-normal-score error.
- **Dependencies:** Phase 2 steps 1–2 and the evaluator's existing `val_synth` path.
- **Compatibility:** The absent-field and `clean_validation` branches retain their current call order and thresholds.

#### 4. Keep the window threshold separate

- **File:** `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_thesis_offline_benchmark.py`
- **Symbol:** `_evaluate_offline_benchmark_splits` raw-input branch.
- **Current responsibility:** Computes `clean_window_threshold` from raw input window MSE.
- **Change:** Continue using the existing window threshold for window predictions; change only the point threshold source.
- **Reason:** Point and window scores have different units and the request asks for one point threshold.
- **Inputs:** Existing raw window score arrays and B-window quantile.
- **Outputs:** Existing window threshold and predictions.
- **Errors:** Preserve existing NumPy quantile behavior.
- **Dependencies:** Existing raw reconstruction score path.
- **Compatibility:** Window metrics and window artifact fields remain unchanged.

#### 5. Carry the selected point threshold to artifact inputs

- **File:** `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_thesis_offline_benchmark.py`
- **Symbol:** `collect_offline_artifact_inputs`.
- **Current responsibility:** Returns split payloads and test metrics but not an explicit selected offline point threshold.
- **Change:** Copy the threshold and source returned by `_evaluate_offline_benchmark_splits` into `artifact_inputs`.
- **Reason:** Export must use the exact threshold selected for metrics instead of recalculating it from another split.
- **Inputs:** Split-output mapping.
- **Outputs:** Existing artifact inputs plus `offline_point_threshold` and `offline_point_threshold_source`.
- **Errors:** Fail if the split flow does not return a finite threshold.
- **Dependencies:** Phase 2 step 3.
- **Compatibility:** Existing artifact fields remain present.

### Tests

#### Filtering and q99

- **Location:** `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/tests/evaluation/test_thresholding_helpers.py`
- **Level:** Unit.
- **Setup:** Use score arrays containing normal, anomalous, NaN, infinity, and covered-only values.
- **Action:** Call the proposed helper with q99.
- **Expected result:** The threshold equals the q99 of finite normal values only.
- **Edge cases:** No normal values, no finite values, and mismatched lengths raise `ValueError`.

#### Two-pass benchmark flow

- **Location:** `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/tests/benchmarks/test_thesis_offline_artifact_exports.py`
- **Level:** Integration-style unit test with fakes.
- **Setup:** Return distinct synthetic scores and labels from each fake evaluator call.
- **Action:** Run `_evaluate_offline_benchmark_splits` with the opt-in source.
- **Expected result:** Calls occur as `val`, unthresholded `val_synth`, thresholded `val_synth`, `test`; test receives the synthetic threshold and source.
- **Edge cases:** Anomaly scores do not enter the selected q99; the window threshold remains distinct.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest tests/evaluation/test_thresholding_helpers.py tests/benchmarks/test_thesis_offline_artifact_exports.py -q` — filtering and two-pass flow pass.
- [ ] `.venv/bin/python -m compileall src scripts` — modified Python files compile.

#### Manual

- [ ] Inspect a test fixture's recorded threshold and source — the value matches the normal covered synthetic score calculation.

### Risks and recovery

- **Risk:** Synthetic anomaly labels are not used after overlap reconstruction. **Mitigation:** assert the fake call labels and the filtered score set. **Recovery:** stop before artifact export if counts disagree.
- **Risk:** The first pass writes misleading metrics. **Mitigation:** discard its metrics and retain only its scores for calibration. **Recovery:** use the second thresholded pass for exported metrics.

### Complete when

- The opt-in flow selects one finite q99 point threshold from synthetic normal covered scores.
- Synthetic and test point metrics use that same threshold.
- The default clean-validation flow remains unchanged.

## Phase 3: New rerun artifacts are isolated and auditable

### Goal

Make the selected source, threshold, checkpoint, and output root visible without invalidating existing artifacts.

### Dependencies

- Phase 1 artifact extension.
- Phase 2 selected threshold in `artifact_inputs`.

### Detailed changes

#### 1. Use the selected threshold in `_build_thresholds`

- **File:** `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_thesis_offline_benchmark.py`
- **Symbol:** `_build_thresholds`.
- **Current responsibility:** Recomputes the offline point threshold from clean-validation scores at lines `664-718`.
- **Change:** Use `artifact_inputs["offline_point_threshold"]` and its source; pass the optional source to `build_threshold_artifact`; retain clean-validation values for legacy inputs that lack the new fields.
- **Reason:** Artifact and metrics must use one selected threshold.
- **Inputs:** Selected threshold, source, online calibration arrays, checkpoint hash, and protocol config.
- **Outputs:** Existing raw schema version 5 artifact with the new offline source record.
- **Errors:** Reject a missing or non-finite selected threshold.
- **Dependencies:** Phase 2 step 5 and `build_threshold_artifact`.
- **Compatibility:** Fallback to the current clean-score calculation for old callers and fixtures.

#### 2. Route every evaluation-only writer to the effective output root

- **File:** `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_thesis_offline_benchmark.py`
- **Symbol:** `run_thesis_offline_benchmark` lines `923-970`.
- **Current responsibility:** Sends `experiment_config["output_dir"]` to artifact export, retention export, and report writing.
- **Change:** Compute one `effective_output_dir` and pass it to manifest creation, `_export_offline_artifacts`, `_export_offline_retention_bundle`, and `_write_report`.
- **Reason:** A single root prevents mixed old/new artifact trees.
- **Inputs:** Optional CLI/API override or configured output directory.
- **Outputs:** All evaluation-only artifacts under one root.
- **Errors:** Fail before evaluation if the root is empty or resolves to an existing protected legacy root when the new protocol requires a rerun root.
- **Dependencies:** Phase 1 output override.
- **Compatibility:** No override preserves the current configured path.

#### 3. Preserve online calibration metadata

- **File:** `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/protocols/threshold_artifact.py`
- **Symbol:** Artifact top-level fields and `thresholds.online_ewma_point`.
- **Current responsibility:** Uses `calibration_split` for online threshold metadata.
- **Change:** Leave online source and top-level calibration unchanged when the offline point source is synthetic.
- **Reason:** The request changes offline point calibration only.
- **Inputs:** Existing online stride-1 calibration arrays.
- **Outputs:** Existing online threshold records with `source_split: clean_validation`.
- **Errors:** Existing online validation errors remain active.
- **Dependencies:** Existing online artifact consumers.
- **Compatibility:** Existing online readers see the same fields and values.

### Tests

#### Artifact source and output isolation

- **Location:** `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/tests/benchmarks/test_thesis_offline_artifact_exports.py` and `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/tests/engine/test_threshold_artifact.py`
- **Level:** Integration/contract.
- **Setup:** Provide selected synthetic threshold and two temporary output roots.
- **Action:** Export artifacts with the opt-in protocol and explicit output root.
- **Expected result:** `thresholds.offline_point.source_split` is synthetic; top-level `calibration_split` and online source remain clean; old root has no new files.
- **Edge cases:** Legacy artifact input without selected-threshold metadata falls back to clean behavior.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest tests/benchmarks/test_thesis_offline_artifact_exports.py tests/engine/test_threshold_artifact.py -q` — export and schema tests pass.
- [ ] `git diff --check` — no whitespace errors after implementation.

#### Manual

- [ ] Inspect the new `thresholds.json`, `resolved_protocol.json`, and benchmark report — each names the new protocol and output root.

### Risks and recovery

- **Risk:** Offline and online source fields become conflated. **Mitigation:** keep separate field names and assertions. **Recovery:** regenerate the rerun artifact with the legacy protocol if source metadata is ambiguous.
- **Risk:** Output root points at an existing benchmark. **Mitigation:** preflight exact path comparison. **Recovery:** stop before evaluation and choose a new run identifier.

### Complete when

- The new artifact identifies synthetic-normal q99 as the offline point source.
- Online and top-level legacy calibration metadata remain valid.
- Every new output is isolated below the new rerun root.

## Phase 4: Regression tests prove both paths

### Goal

Prove the opt-in path and the legacy path with minimal focused tests.

### Dependencies

- Phases 1–3.

### Detailed changes

#### 1. Preserve the existing default call order

- **File:** `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/tests/benchmarks/test_thesis_offline_artifact_exports.py`
- **Symbol:** `test_collect_offline_artifact_inputs_uses_checkpoint_and_three_splits`.
- **Current responsibility:** Asserts the legacy non-raw evaluator call order.
- **Change:** Keep the assertion for the default protocol and assert the legacy threshold source remains clean.
- **Reason:** This is the primary backward-compatibility guard.
- **Inputs:** Existing fake evaluator and legacy protocol mapping.
- **Outputs:** Stable call-order evidence.
- **Errors:** Any extra synthetic pass in the legacy branch fails the test.
- **Dependencies:** Phase 2 flow branching.
- **Compatibility:** Existing test semantics remain explicit.

#### 2. Test the new protocol path separately

- **File:** `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/tests/benchmarks/test_thesis_offline_artifact_exports.py`
- **Symbol:** Proposed focused opt-in benchmark test.
- **Current responsibility:** No test currently asserts synthetic-normal calibration.
- **Change:** Add a fixture with mixed synthetic labels, covered masks, and finite/non-finite scores; assert the selected q99 and second-pass threshold.
- **Reason:** The requested behavior needs direct evidence.
- **Inputs:** Fake loaders and evaluator records.
- **Outputs:** Selected threshold, source, and call order.
- **Errors:** Empty normal set raises the expected error.
- **Dependencies:** Phase 2 helper and runner changes.
- **Compatibility:** Test uses a new protocol mapping and does not alter legacy fixtures.

#### 3. Verify command/API argument compatibility

- **File:** `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/tests/benchmarks/test_thesis_offline_artifact_exports.py`
- **Symbol:** Wrapper tests calling `run_thesis_offline_benchmark`.
- **Current responsibility:** Calls the API without an output override.
- **Change:** Add one call with `output_dir` and retain one call without it.
- **Reason:** Optional parameters must not break old callers.
- **Inputs:** Temporary experiment config and output paths.
- **Outputs:** Expected report locations.
- **Errors:** Training flow with an override raises the documented error.
- **Dependencies:** Phase 1 step 4.
- **Compatibility:** Existing call signatures remain valid.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest tests/evaluation/test_thresholding_helpers.py tests/evaluation/test_evaluator_thresholding.py tests/benchmarks/test_benchmark_protocol_config.py tests/engine/test_threshold_artifact.py tests/benchmarks/test_thesis_offline_artifact_exports.py -q` — all focused tests pass.
- [ ] `.venv/bin/python -m compileall src scripts` — source compiles.
- [ ] `git diff --check` — patch is clean.

#### Manual

- [ ] Review the test output and confirm the new tests are opt-in while legacy tests remain unchanged.

### Risks and recovery

- **Risk:** Fakes accept fewer evaluator keywords than the real evaluator. **Mitigation:** update only the focused fake signatures and assert the exact kwargs. **Recovery:** restore the prior fake behavior for legacy tests.

### Complete when

- The focused suite passes for both legacy and opt-in paths.
- The new source and output contracts have direct assertions.

## Phase 5: The cloud rerun command is ready

### Goal

Prepare an inspectable command for all 18 Stage-B best checkpoints without executing remote jobs in this task.

### Dependencies

- Phases 1–4 implemented and verified.
- Current remote endpoint and checkpoint inventory.

### Detailed changes

#### 1. Read the current remote endpoint

- **File:** `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/ssh-gpu.txt` or the repository's current SSH instruction file.
- **Symbol:** Active host, port, username, and remote repository path.
- **Current responsibility:** Operational configuration for the shared GPU host.
- **Change:** Read values immediately before execution; do not copy stale endpoint values into a permanent script.
- **Reason:** Remote endpoints can change.
- **Inputs:** Current local SSH instruction file.
- **Outputs:** Validated connection parameters.
- **Errors:** Stop if the endpoint or remote repository path is missing.
- **Dependencies:** Remote access authorization.
- **Compatibility:** No local code or remote state changes.

#### 2. Read the Stage-B checkpoint inventory

- **File:** Remote Stage-B checkpoint tree under the inventory's current paths.
- **Symbol:** Each `best.pt` for O0/O1, three entities, and seeds 6/8/36.
- **Current responsibility:** Defines the requested 18 evaluation inputs.
- **Change:** Perform a read-only existence check immediately before writing the command.
- **Reason:** The command must target current checkpoints only.
- **Inputs:** Remote inventory and new protocol path.
- **Outputs:** Exactly 18 validated checkpoint/config pairs.
- **Errors:** Stop and report missing or duplicate checkpoints.
- **Dependencies:** Phase 5 step 1.
- **Compatibility:** Do not modify remote outputs.

#### 3. Write one evaluation-only command template

- **File:** **Proposed new document section:** a command section in the next approved CLI detail log.
- **Symbol:** `.venv/bin/python -m scripts.benchmarks.run_thesis_offline_benchmark`.
- **Current responsibility:** Existing CLI evaluates one experiment config and checkpoint.
- **Change:** Use `--evaluation-only`, `--checkpoint-path`, the proposed synthetic-normal protocol, and a unique per-checkpoint `--output-dir` below a new rerun root.
- **Reason:** Each checkpoint must produce isolated artifacts with the selected threshold.
- **Inputs:** Validated remote config/checkpoint pairs.
- **Outputs:** One inspectable command per checkpoint.
- **Errors:** Stop command generation for a missing checkpoint or duplicate output path.
- **Dependencies:** Phase 5 steps 1–2.
- **Compatibility:** Do not pass new flags to old commands unless the new code is deployed.

#### 4. Define the one-combination preflight

- **File:** Same command document.
- **Symbol:** First selected checkpoint command.
- **Current responsibility:** Project instructions require one full concrete run before a matrix.
- **Change:** Run one authorized evaluation in `tmux`, then inspect its threshold, source, score, metric, and report artifacts.
- **Reason:** Detect protocol, checkpoint, environment, or output-root errors before 18 runs.
- **Inputs:** One current Stage-B checkpoint and its config.
- **Outputs:** One verified rerun artifact bundle.
- **Errors:** Stop the matrix when preflight fails.
- **Dependencies:** Phase 5 step 3 and explicit remote authorization.
- **Compatibility:** Keep the preflight output in the new rerun root.

### Verification

#### Automated

- [ ] Run the existing command parser/dry-run path with one new protocol/config pair — it resolves the new output root and source.
- [ ] Read the preflight `thresholds.json` and benchmark report — source is synthetic-normal and point threshold is finite.

#### Manual

- [ ] Confirm the remote inventory contains exactly 18 intended Stage-B `best.pt` files.
- [ ] Confirm the new rerun root does not equal any existing benchmark output root.

### Risks and recovery

- **Risk:** A stale inventory targets a deleted checkpoint. **Mitigation:** read-only existence check immediately before command generation. **Recovery:** regenerate the command.
- **Risk:** Matrix execution hides a first-run failure. **Mitigation:** require one preflight and artifact inspection. **Recovery:** stop before the remaining 17 runs.
- **Risk:** Remote command overwrites historical results. **Mitigation:** unique output root and exact path comparison. **Recovery:** stop the exact job; do not delete broad directories.

### Complete when

- The command document names the new protocol, unique output-root rule, preflight, and all 18 current checkpoint/config pairs.
- No remote job has started without separate authorization.

## Interface and data changes

- Existing protocol field `offline_threshold_split` remains `clean_validation`.
- New optional protocol field: `offline_point_threshold_source_split`, with `synthetic_validation_normal` as the opt-in value.
- Existing top-level artifact `calibration_split` remains `clean_validation`.
- Existing `thresholds.offline_point.source_split` can identify the opt-in source.
- Existing CLI/API calls remain valid when `--output-dir`/`output_dir` is absent.
- No schema version increase is planned; optional fields must be accepted by existing schema readers.

## Deployment and rollout

1. Run local focused tests.
2. Deploy the source/config/test changes to the remote checkout using the approved synchronization method.
3. Run one read-only environment and checkpoint preflight.
4. Run one authorized evaluation-only combination in `tmux`.
5. Inspect its artifacts.
6. Run the remaining 17 combinations only after preflight success.

Mixed-version behavior: do not use the new protocol or flags against a remote checkout that lacks the corresponding code. Rollback is to omit the new protocol field and use the existing clean-validation command and output root.

## Documentation changes

- Keep the research note unchanged as the source of the discovered change surface.
- Keep this detail document as the implementation instructions.
- Add the final CLI command and preflight evidence to a dated detail log after execution is authorized.

## Final verification

- [ ] Legacy protocol and existing focused tests pass.
- [ ] New synthetic-normal q99 tests pass.
- [ ] New output-root tests pass.
- [ ] `thresholds.json` records the synthetic-normal offline point source while preserving clean-validation online metadata.
- [ ] The command targets 18 current Stage-B best checkpoints and writes to a new rerun root.

## Assumptions and non-blocking uncertainties

- The remote endpoint, repository path, and checkpoint inventory are runtime facts and must be refreshed before execution.
- The active requested protocol is raw-input; the new synthetic-normal source is not planned for the existing model-output branch.
- The exact rerun root name is intentionally left to command-generation time so it can include a fresh run identifier.
