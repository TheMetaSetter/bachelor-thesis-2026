# Pro-reconstruction code modification surfaces

Date: 2026-09-06

Scope: identify the active code and configuration lines that would need modification for the agreed `pro-reconstruction` offline contract in `documents/notes/pro-reconstruction-offline-fairness-decisions.md`. This report is research only and does not change source code or configurations.

## Current execution path

The main-method path is generated experiment YAML → two-stage runner → `Evaluator` → THESIS offline benchmark runner → threshold artifact and metric files.

The traditional-baseline path is baseline YAML → `scripts/benchmarks/run_offline_benchmark.py` → fit on train → calibrate on clean validation → point-score artifact and metrics.

The standalone checkpoint path, including RedLamp checkpoints, is experiment YAML → `scripts/cli/evaluate.py` → `Evaluator`.

## Implemented already

- `src/protocols/reconstruction_scores.py:53-63` calculates normalized-input point and window MSE from scaled input and reconstruction samples, then averages MSE over Monte-Carlo samples.
- `scripts/benchmarks/generate_full_direct_recon075_cls025_matrix.py:48-67` defines O0/O1, loss weights `0.75` and `0.25`, direct branch routing, and the O1 balanced point-score-loss settings.
- `scripts/benchmarks/generate_full_direct_recon075_cls025_matrix.py:91-93` already sets `reconstruction_loss_space: normalized_input` and 12 data-loading workers for the 18-combination main-method matrix.
- `src/engine/thresholding.py:47-60` already selects q99 from finite synthetic-validation point scores with point label `0`; because it filters only point labels, it includes normal points from anomalous windows.
- `src/metrics/pointwise.py:23,404-485,749-778` already calculates and persists VUS-PR and VUS-ROC for all three FPR budgets `0.001`, `0.005`, and `0.01` in one metric call.
- `tests/evaluation/test_budgeted_vus_metrics.py:13-34` checks the locked budgets and both budgeted metric mappings. This test file is currently uncommitted, so its status is user-owned working-tree state.

## Required modification surfaces

| Priority | File and lines | Current implemented behavior | Why this surface must change for the agreed contract |
| --- | --- | --- | --- |
| 1 | `configs/protocol/smd_window20_synthnormal_q99_ewma09.yaml:2,6-8` | Declares `score_space: raw_input`, chooses the offline window threshold from clean validation, and changes only the point-threshold source to synthetic validation. | The contract requires normalized-input MSE for main method and RedLamp, and both offline thresholds from synthetic validation. |
| 1 | `src/protocols/smd_benchmark_protocol.py:48-64` | Rejects every score space except `raw_input`, requires `offline_threshold_split: clean_validation`, and recognizes a synthetic source only for the point threshold. | It blocks the intended protocol before training or evaluation starts. |
| 1 | `src/engine/evaluator.py:359-384,537-610,665-722` | Accepts only `model_output` or `raw_input`; raw-input evaluation computes both MSE spaces but selects raw MSE for point scores, point predictions, window predictions, metric score space, and window threshold. | Normalized MSE exists as auxiliary output only, so it cannot currently become the operational anomaly score. |
| 1 | `scripts/benchmarks/run_thesis_offline_benchmark.py:465-557,627-682` | The raw protocol derives the point threshold from synthetic normal points when configured, but it always derives `clean_window_threshold` from clean validation raw-input MSE. | This is the direct mismatch with the required q99 synthetic-normal window threshold and normalized score space. The payload builder already exposes normalized point and window MSE, which is evidence that no new MSE formula is required. |
| 1 | `src/engine/thresholding.py:47-60` | Contains the needed point-threshold selector but has no corresponding synthetic-normal window-threshold selector. | The contract needs an independently selected q99 window threshold using finite window scores whose synthetic window label is normal. |
| 1 | `src/protocols/threshold_artifact.py:357-552` | Allows only `model_output` and `raw_input`, persists only `offline_point`, and names the optional window field `input_window` with raw-MSE or generic window-MSE semantics. | The output schema cannot unambiguously record the new offline normalized-MSE window threshold and its synthetic-validation source. |
| 1 | `scripts/cli/evaluate.py:41-59,205-217` | The standalone checkpoint path selects a raw-input helper only when `reconstruction_loss_space` is `raw_input`; that helper calibrates clean validation raw MSE for both thresholds. | RedLamp is evaluated through this entry point, so it cannot yet receive the agreed normalized-MSE and synthetic-validation calibration protocol. |
| 1 | `scripts/benchmarks/run_offline_benchmark.py:136-183,275-350` | Fits each traditional baseline on train, calibrates it on clean validation, emits transformed point scores, selects only an offline point threshold, and has no native window-score artifact or window prediction. | Traditional baselines must keep native scoring but need the same synthetic-validation point and window q99 calibration rules for a fair comparison. |
| 1 | `src/baselines/traditional/iforest.py:77-142`, `src/baselines/traditional/kmeans_ad.py:79-146`, and `src/baselines/traditional/stumpy_channel_ab.py:181-267` | Each baseline exposes internal native window scores, but `calibrate` fits clean-validation score normalization and `score_sequence` depends on that calibration. | The current public flow cannot calibrate from synthetic normal values or retain native window scores without altering these method boundaries or their runner use. |
| 2 | `tests/evaluation/test_evaluator_thresholding.py`, `tests/evaluation/test_thresholding_helpers.py:13-32`, `tests/engine/test_threshold_artifact.py`, and `tests/benchmarks/test_run_offline_benchmark.py:13-133` | Tests cover raw-input evaluation, finite synthetic-normal point selection, raw threshold artifacts, and clean-validation baseline calibration. | These tests would otherwise preserve the old contract and do not establish the new normalized-score, dual-threshold, or native-baseline behavior. |

## No modification required for budgeted VUS calculation

`src/metrics/pointwise.py:404-485` iterates all three budgets and returns both curves, while `src/metrics/pointwise.py:749-778` inserts the two mappings into ordinary evaluation metrics. The current evidence does not show a required algorithm change in this module for the stated evaluation contract.

## Conditional checkpoint-selection surface

`src/engine/trainer.py:811-859` runs validation evaluation and copies its metric dictionary into epoch metrics, while `src/engine/trainer.py:896-943` converts the configured monitor value to one float before saving `best.pt`.

The budgeted VUS values are nested mappings, so none is directly usable as the monitor value at `src/engine/trainer.py:901-903` without a scalar extraction path and a matching monitor name in generated experiment configurations.

This change is conditional because the agreed note explicitly leaves the best-checkpoint scalar metric and FPR budget undecided. The current source cannot determine which of the three budgets or which curve type should select a checkpoint.

## Terminology and evidence limits

`pro-reconstruction` is documented as the target experiment label, but the active generator uses `two_stage_base_recon075_cls025_direct_v1` and `two_stage_point_score_supervised_recon075_cls025_direct_v1` at `scripts/benchmarks/generate_full_direct_recon075_cls025_matrix.py:83-87`.

The exact canonical runtime name is therefore unknown from executable code, matching the unresolved item in the decision note. No source-based rename can be selected safely yet.

## Version context

The inspected checkout is branch `dev` at `cc42b4e3faa433d9d81e33719dfab0cbbe1cee66`.

The working tree contains user changes in `src/metrics/pointwise.py` and untracked research, notes, and `tests/evaluation/test_budgeted_vus_metrics.py`. This report treats those files as current local evidence but does not claim they are committed baseline behavior.
