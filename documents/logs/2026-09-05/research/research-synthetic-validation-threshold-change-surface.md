---
date: 2026-09-05T17:09:13+07:00
researcher: OpenAI Codex
topic: "Detect lines of code that need modification for a synthetic-validation normal-score q99 threshold"
status: complete
revision: 5529a0c3f1ab9f4b7f013543aa7e61922b74b56d
branch: dev
---

# Research: Detect lines of code that need modification for a synthetic-validation normal-score q99 threshold

## Summary

The active evaluator already supports the required data flow: `val_synth` selects the synthetic-validation step, reconstructs covered point timelines, and accepts an externally supplied point threshold. The current threshold is still selected from clean validation in the benchmark runner. The main modification surface is therefore the benchmark runner, followed by the threshold-artifact/protocol contract, CLI output-directory plumbing, and focused tests.

No source code was changed during this research pass.

## Research question

Use `prompts/1_research_prompt.md` to detect the lines of code that need modification so one q99 threshold is selected from anomaly scores of normal samples in the synthetic validation set, while preserving backward compatibility and writing results to a new re-run directory.

## System context

The active entry point is `scripts/benchmarks/run_thesis_offline_benchmark.py`. It loads an experiment config and a protocol config, optionally loads a Stage-B checkpoint in evaluation-only mode, evaluates clean validation, synthetic validation, and test, then exports scores, metrics, threshold artifacts, traces, and a report.

The evaluator is `src/engine/evaluator.py`. It maps `evaluation_stage="val_synth"` to `synthetic_validation_step`, replaces point labels with `synthetic_anomaly_mask`, reconstructs pointwise timelines, and computes metrics only on covered points.

## Execution path

1. `run_thesis_offline_benchmark` loads configs and validates the protocol.
2. `collect_offline_artifact_inputs` builds the data bundle, model, checkpoint state, and evaluator.
3. `_evaluate_offline_benchmark_splits` evaluates validation data and selects the current offline point threshold from clean-validation scores.
4. The runner passes that fixed threshold to synthetic validation and test.
5. `_build_thresholds` writes the same clean-validation source into the threshold artifact.
6. `_export_offline_artifacts` and the retention helper write under `experiment_config["output_dir"]`.

## Detailed findings

### Implemented evaluator behavior: no direct evaluator change identified

`src/engine/evaluator.py:537-548` exposes `point_score_threshold`, `threshold_source`, `score_space`, `scaler`, `window_score_threshold`, and `evaluation_stage`.

`src/engine/evaluator.py:421-433` maps `val_synth` to `synthetic_validation_step`.

`src/engine/evaluator.py:579-585` uses `synthetic_anomaly_mask` as labels for `val_synth`.

`src/engine/evaluator.py:646-669` reconstructs entity timelines, extracts covered points, resolves the supplied threshold, and creates raw-input point predictions.

These lines already provide the required inputs and threshold injection point. The available evidence does not show a need to modify this file for the requested source split.

### Primary modification surface: offline split evaluation

`scripts/benchmarks/run_thesis_offline_benchmark.py:452-544` is the active threshold-selection path.

- Lines `472-475` select `clean_threshold` from clean-validation scores.
- Lines `482-501` pass that threshold to synthetic validation and test in the raw-input protocol.
- Lines `519-536` do the same in the calibrated model-output path.

These ranges must be inspected in any implementation because they currently make clean validation the single source for the offline point threshold. The synthetic path would also need a first evaluation without a fixed threshold to obtain scores before a threshold can be selected, then a thresholded evaluation for metrics. The current tests confirm the existing call order at `tests/benchmarks/test_thesis_offline_artifact_exports.py:503-601`.

### Score payload and normal-sample filtering surface

`scripts/benchmarks/run_thesis_offline_benchmark.py:576-631` converts reconstructed records to NumPy payloads and keeps only `covered_point_mask` positions at lines `585-606`.

The payload currently keeps `point_labels` but does not distinguish synthetic normal points from synthetic anomaly points. This is the nearest active location for verifying that threshold calibration uses both `synthetic_anomaly_mask == 0` and covered points. The code must also exclude non-finite scores before a q99 calculation; the current generic helper only removes NaN values.

### Threshold helper surface

`src/engine/thresholding.py:15-44` contains the existing point-score quantile helpers. `select_clean_validation_point_threshold` is named and documented specifically for clean validation at lines `37-44`.

This is a candidate modification surface only if the implementation introduces a named synthetic-normal helper. The current generic quantile behavior is otherwise reusable after the runner supplies the filtered scores. No call site currently selects a synthetic-normal threshold.

### Threshold-artifact contract surface

`scripts/benchmarks/run_thesis_offline_benchmark.py:655-762` builds the exported threshold artifact.

- Lines `664-671` choose the score array and offline quantile.
- Lines `715-718` select the artifact's offline point threshold from clean scores.
- Lines `723-750` populate artifact provenance and score-space fields.

`src/protocols/threshold_artifact.py:345-383` accepts `calibration_split`, and lines `414-429` copy that value into threshold records. `src/protocols/threshold_artifact.py:148-150` currently rejects any artifact whose top-level `calibration_split` is not `clean_validation`.

This is a compatibility boundary. Existing schema validation and existing artifacts assume `calibration_split=clean_validation`. A synthetic-validation rerun therefore cannot silently replace that contract without touching the validator and its tests. The available code does not establish whether the intended new metadata should replace the top-level calibration split or be recorded as a separate offline source field.

### Protocol configuration surface

`configs/protocol/smd_window20_cleanval_q99_ewma09.yaml:6-7` selects `offline_threshold_split: clean_validation` and `offline_threshold_quantile: 0.99`.

`src/protocols/smd_benchmark_protocol.py:39-57` rejects `test` threshold selection and requires `offline_threshold_split == clean_validation`.

The current protocol file is used by existing commands and tests. Backward compatibility requires preserving its current behavior; the evidence supports creating or selecting a separate protocol/config path for the synthetic-validation source rather than changing this file in place. Whether that separate protocol is allowed by the locked validator remains an unresolved contract question.

### New re-run output directory surface

`scripts/benchmarks/run_thesis_offline_benchmark.py:863-887` creates the evaluation-only manifest under `experiment_config["output_dir"]`.

`scripts/benchmarks/run_thesis_offline_benchmark.py:890-898` exposes the Python API, but it has no output-directory override.

`scripts/benchmarks/run_thesis_offline_benchmark.py:923-970` sends `experiment_config["output_dir"]` to artifact export, retention export, and report writing.

`scripts/benchmarks/run_thesis_offline_benchmark.py:975-994` defines the CLI and currently has no `--output-dir` argument.

These ranges are the modification surface for a new re-run directory. Existing callers that omit a new option must continue using the configured output directory.

### Tests that establish current behavior

`tests/benchmarks/test_thesis_offline_artifact_exports.py:503-601` verifies checkpoint loading, evaluator construction, and the current non-raw call order (`val`, `val`, `val_synth`, `test`).

`tests/benchmarks/test_thesis_offline_artifact_exports.py:169-235` and `tests/benchmarks/test_full_spec_runtime_readiness.py:199-216` verify that artifacts are written below the configured output directory.

`tests/benchmarks/test_benchmark_protocol_config.py:20-33` verifies the locked clean-validation protocol, and `tests/engine/test_threshold_artifact.py:54-85` verifies raw schema version 5 and identity score fields.

Any implementation must preserve these existing assertions for the default protocol and add separate coverage for synthetic-normal q99 selection and the new output directory.

## Evidence

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/prompts/1_research_prompt.md:1-352` — requires evidence-based research, execution-path tracing, uncertainty labels, and a saved Markdown report.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/evaluator.py:421-433` — maps `val_synth` to the synthetic evaluation method.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/evaluator.py:537-669` — accepts an explicit threshold, reconstructs timelines, and filters metric inputs to covered points.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_thesis_offline_benchmark.py:452-544` — selects the current clean-validation threshold and evaluates synthetic/test with it.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_thesis_offline_benchmark.py:576-631` — builds covered point-score payloads.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_thesis_offline_benchmark.py:655-762` — builds the threshold artifact from clean scores.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_thesis_offline_benchmark.py:863-994` — controls evaluation-only output paths and CLI arguments.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/thresholding.py:15-44` — defines existing point quantile helpers.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/protocols/smd_benchmark_protocol.py:39-57` — locks the active offline threshold split to clean validation.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/protocols/threshold_artifact.py:22-150` — validates the existing threshold-artifact schema and clean-validation calibration split.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/configs/protocol/smd_window20_cleanval_q99_ewma09.yaml:1-17` — defines the active q99 clean-validation protocol.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/tests/benchmarks/test_thesis_offline_artifact_exports.py:503-601` — verifies current evaluator call order and checkpoint use.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/tests/benchmarks/test_benchmark_protocol_config.py:20-73` — verifies locked protocol validation.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/tests/engine/test_threshold_artifact.py:54-85` — verifies raw threshold-artifact compatibility.

## Configuration observed

| Setting | Active value | Evidence | Scope |
| --- | --- | --- | --- |
| `offline_threshold_split` | `clean_validation` | `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/configs/protocol/smd_window20_cleanval_q99_ewma09.yaml:6` | Active offline protocol |
| `offline_threshold_quantile` | `0.99` | `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/configs/protocol/smd_window20_cleanval_q99_ewma09.yaml:7` | Active offline point threshold |
| `score_space` | `raw_input` | `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/configs/protocol/smd_window20_cleanval_q99_ewma09.yaml:2` | Active raw-MSE protocol |
| `output_dir` source | experiment config | `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_thesis_offline_benchmark.py:935-970` | Evaluation artifacts and report |

## Conflicts and uncertainties

- The active protocol validator requires `offline_threshold_split=clean_validation`, while the requested threshold source is synthetic validation. The code does not define a compatible synthetic source value.
- The threshold artifact has one top-level `calibration_split` field and validates it as `clean_validation`. It is unknown whether the requested rerun should change that field or add separate offline-source metadata.
- `scripts/benchmarks/_internal/run_thesis_offline_benchmark_helpers.py` contains overlapping helper code, but the active wrapper imports only `_export_offline_retention_bundle` from it at `scripts/benchmarks/run_thesis_offline_benchmark.py:858-860`. The main runner remains the confirmed threshold-selection owner.
- The remote `cloud-gpu` checkpoint inventory was not inspected in this research-only pass because the question was limited to the local modification surface.

## Open questions

- Which artifact field should identify the synthetic-normal source while preserving existing clean-validation artifacts?
- Should the new synthetic-validation protocol be a separate YAML file, or should the validator accept an optional source field while keeping the current default unchanged?
- Should the new re-run directory be supplied through a new optional CLI argument, generated experiment configs, or both?
