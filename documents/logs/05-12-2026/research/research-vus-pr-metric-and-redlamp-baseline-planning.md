---
date: 2026-05-12 12:22:48 +07 +0700
researcher: TheMetaSetter
git_commit: b028706b5e57fb259cdd1ed26466753bb479b8de
branch: dev
repository: bachelor-thesis-2026
topic: "VUS-PR metric planning for RedLamp baseline and prototype multi-task comparison"
tags: [research, time-series, anomaly-detection, metrics, vus-pr, redlamp, multi-class]
status: complete
last_updated: 2026-05-12
last_updated_by: TheMetaSetter
---

# Research: VUS-PR Metric Planning for RedLamp Baseline and Prototype Multi-Task Comparison

**Date**: 2026-05-12 12:22:48 +07 +0700
**Researcher**: TheMetaSetter
**Git Commit**: b028706b5e57fb259cdd1ed26466753bb479b8de
**Branch**: dev

## Research Question

The user wants to plan the implementation of the VUS-PR metric based on `papers/Paparrizos et al. - 2022 - Volume under the surface a new accuracy evaluation measure for time-series anomaly detection.pdf`. The metric will support baseline statistics from RedLamp and comparison with the proposed prototype-based multi-task model that combines discrete prototypes and continuous prototypes for reconstruction and multi-class classification.

## Summary

The active repository does not currently implement VUS-PR. The only direct VUS reference in source code is a TODO in `src/metrics/pointwise.py` that explicitly says not to use point-adjusted metrics and to read and add VUS-PR. Current offline evaluation computes pointwise `roc_auc`, `pr_auc`, thresholded precision, recall, F1, and false positive rate after merging overlapping window scores back to each entity timeline.

The most compatible implementation location is `src/metrics/pointwise.py`, because the evaluator already supplies one-dimensional point labels and point scores after timeline reconstruction. `src/engine/evaluator.py` should call the VUS-PR helper after `concatenated_labels` and `concatenated_scores` are built. The metric should be stored in `evaluation_metrics.json`, logged to Weights and Biases through the existing evaluation logger, and included in ablation summaries.

The active repository already contains a RedLamp-inspired MLP baseline in `src/models/redlamp_mlp_baseline.py`, with a window-20 experiment config at `configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml`. A matching thesis multi-task RedLamp multi-class config exists at `configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20__w20__seed11__default.yaml`.

## Detailed Findings

### Data Preparation

The project batch contract is `batch["x"]: Tensor[B, L, D]`. SMD data is loaded through the config-driven data layer and windows are later merged back to per-entity timelines in `src/engine/evaluator.py`.

For the RedLamp-aligned comparison, the active synthetic taxonomy is the eleven RedLamp anomaly families:

- `spike`
- `flip`
- `speedup`
- `noise`
- `cutoff`
- `average`
- `scale`
- `wander`
- `contextual`
- `upsidedown`
- `mixture`

The multi-class label space used by the active RedLamp baseline is twelve classes: `normal` followed by the eleven synthetic anomaly families. The window-20 RedLamp baseline config uses `configs/data/smd_rtx3090_machine_2_1_20.yaml`.

### Modeling and Training

The RedLamp-inspired baseline is self-contained in `src/models/redlamp_mlp_baseline.py`. It flattens each window, encodes it with a configurable MLP, reconstructs the flattened window, and predicts a multi-class synthetic anomaly label from the latent representation. The default baseline config sets `mlp_num_linear_layers: 3`, `latent_dim: 128`, `classifier_dim: 32`, `num_classes: 12`, and `lambda_cls: 0.1`.

The proposed thesis model config for the same RedLamp multi-class setting is `configs/model/thesis_multitask_redlamp_multiclass.yaml`. It uses the thesis multi-task model with continuous and discrete prototype branches enabled, twelve classification classes, and a reconstruction plus classification objective with optional regularizers disabled by default.

The trainer already aggregates binary classification metrics for two-class outputs and multi-class metrics for outputs with more than two classes. Multi-class metrics currently include accuracy, macro F1, weighted F1, and the number of observed classes.

### Evaluation

`src/engine/evaluator.py` evaluates a checkpoint by running `model.test_step`, collecting `outputs["point_scores"]`, merging overlapping windows back to entity timelines, concatenating all entity point scores and labels, selecting a 95th-quantile threshold from the same evaluated score distribution, and calling `compute_pointwise_metrics`.

`src/metrics/pointwise.py` currently computes:

- `roc_auc`
- `pr_auc`
- `precision`
- `recall`
- `f1`
- `fpr`

The file contains a TODO near `compute_pointwise_metrics` stating that point-adjusted metrics must not be used and that VUS-PR should be added.

The threshold calibration detail note at `documents/logs/04-29-2026/detail/detail-threshold-calibration-design-alignment.md` records that thresholded metrics are currently weak because the threshold is selected from the same test score distribution. VUS-PR is useful here because it is threshold-independent. It does not remove the need to later fix threshold calibration for F1, precision, and recall.

### VUS-PR From the Paper

The provided PDF describes Range-AUC and VUS as threshold-independent extensions for time-series anomaly detection with range-based anomalies. Range-AUC first creates a continuous label extension around anomaly boundaries using a buffer length. VUS then computes Range-AUC over multiple buffer lengths instead of requiring one fixed buffer length.

For VUS-PR, the computation varies two dimensions:

- anomaly-score thresholds;
- buffer lengths from zero to a maximum buffer length.

For each buffer length, the algorithm builds modified labels, computes range precision and range recall over thresholds, computes the area under the precision-recall curve for that buffer length, and averages those areas across buffer lengths. The paper also describes optimized exact implementations, but a naive exact implementation is easier to validate first in this repository because the existing evaluator already works on one-dimensional label and score arrays.

## Code References

- `src/metrics/pointwise.py:73` defines `compute_binary_classification_metrics`.
- `src/metrics/pointwise.py:95` defines `compute_multiclass_classification_metrics`.
- `src/metrics/pointwise.py:117` defines `compute_pointwise_metrics` and contains the VUS-PR TODO.
- `src/engine/evaluator.py:153` documents that window scores are merged back to original entity timelines.
- `src/engine/evaluator.py:220` concatenates all point scores and point labels.
- `src/engine/evaluator.py:229` calls `compute_pointwise_metrics`.
- `src/models/redlamp_mlp_baseline.py:1` defines the self-contained RedLamp-inspired MLP baseline.
- `configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml:1` defines the active RedLamp MLP baseline experiment.
- `configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20__w20__seed11__default.yaml:1` defines the matching thesis multi-task RedLamp multi-class experiment.
- `documents/logs/05-11-2026/research/research-redlamp-baseline-synthetic-anomaly-alignment.md:1` documents the RedLamp multi-class alignment research.
- `documents/logs/04-29-2026/detail/detail-threshold-calibration-design-alignment.md:1` documents the unresolved threshold calibration issue.

## Pipeline Documentation

The offline evaluation pipeline is currently:

1. Load an experiment config and checkpoint through `scripts/evaluate.py`.
2. Build the SMD test loader and the model from the resolved config.
3. Run `Evaluator.evaluate`.
4. Compute point scores per window through `model.test_step`.
5. Merge overlapping window point scores back to original entity timelines by averaging.
6. Concatenate all entity point scores and labels.
7. Compute pointwise metrics and curves.
8. Write `evaluation_records.json`, `evaluation_metrics.json`, `evaluation_curves.json`, and `resolved_experiment_config.json`.

The planned VUS-PR metric fits after step 6 and before step 7. The metric should consume the same `concatenated_labels` and `concatenated_scores` arrays used for ordinary AUC-PR. For entity-level reporting, it can also be computed per evaluation record and then macro-averaged, but the first implementation should match the current global metric style unless the user chooses otherwise.

## Historical Context

`documents/design/idea.md` fixes the thesis goal of a multi-task anomaly detection system with continuous and discrete prototypes, reconstruction, and anomaly-type classification. `documents/design/design_starter.md` fixes the standardized batch and model output contracts. `documents/logs/05-11-2026/research/research-redlamp-baseline-synthetic-anomaly-alignment.md` records that the active codebase now has RedLamp multi-class alignment and a RedLamp MLP baseline path.

## Open Questions

- Should VUS-PR be implemented first as an exact naive reference implementation for readability and testability, or should the first implementation include one of the optimized algorithms from the paper?
- Should the first reported `vus_pr` value be computed globally on concatenated SMD timelines, matching current `pr_auc`, or per entity with a macro average?
- What maximum buffer length should be used by default for SMD window-20 experiments: the model window length, half the window length, or an explicit config field?
- Should ablation summaries be extended to include both `pr_auc` and `vus_pr` before running the RedLamp baseline and thesis multi-task comparison?
