---
date: 2026-05-22T17:03:30+07:00
researcher: Artificial Intelligence Agent
git_commit: acf12e9b1708a6832426f2ffe01768a0d5eacbee
branch: dev
repository: bachelor-thesis-2026
topic: "Current codebase status before reconstruction-oscillation ablations and classification diagnostics experiments"
tags: [research, time-series, anomaly-detection, multi-class]
status: complete
last_updated: 2026-05-22
last_updated_by: Artificial Intelligence Agent
---

# Research: Current codebase status before reconstruction-oscillation ablations and classification diagnostics experiments

**Date**: 2026-05-22T17:03:30+07:00  
**Researcher**: Artificial Intelligence Agent  
**Git Commit**: acf12e9b1708a6832426f2ffe01768a0d5eacbee  
**Branch**: dev

## Research Question
For the baseline command:
`python scripts/train.py --experiment-config configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20-recon-diag-quick-100ep__w20__seed11__default.yaml`,
document the current implementation status for:
1. An ablation that disables the classification learning objective to test whether reconstruction can learn stably on its own.
2. A full-model run that logs classification prediction ratios and epoch-level confusion matrices to inspect class confusion and class bias.

## Summary
The current pipeline is configuration-driven and already logs epoch-level scalar metrics for reconstruction, classification accuracy/F1, and VUS-PR, with optional reconstruction diagnostics. The reconstruction loss for `thesis_multitask` is computed in `_compute_reconstruction_loss(...)`, with optional masking by `synthetic_anomaly_mask` when `reconstruction_normal_only` is active. The training engine aggregates logits and labels across an epoch and computes multiclass scalar metrics, but it does not currently emit confusion matrices or predicted-class distribution histograms as first-class epoch outputs. The model forward path already exposes `class_probabilities` in `outputs["aux"]`, yet trainer aggregation currently consumes logits/labels for scalar metrics only.

## Detailed Findings

### Data Preparation
- Dataset source and bundle construction are initiated from `scripts/train.py` through registry-based dataset construction (`build_dataset(...)`) using resolved `data` config.
- Synthetic anomaly injection for multitask supervision is owned by `src/data/augment.py` via `SyntheticAnomalyInjector`.
- For each batch, anomaly-injection decisions are sampled per window. When `balance_binary_classes_within_batch` is `false`, injection uses Bernoulli sampling per sample (`_sample_injection_decisions`, `return rand < anomaly_probability`).
- For each selected anomalous window, anomaly family is sampled uniformly over configured `anomaly_families` using `_randint(0, len(anomaly_families))`.
- Classification labels are generated from augmentation metadata; in multiclass mode, labels map to `("normal", *REDLAMP_ANOMALY_FAMILIES)`.

### Modeling and Training
- Baseline experiment config: `configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20-recon-diag-quick-100ep__w20__seed11__default.yaml`.
- The model is built from merged `model` and `task` sections in `scripts/train.py` (`build_model_from_experiment_config`).
- `ThesisMultitaskModel.forward(...)` produces:
  - `recon` for reconstruction,
  - `logits` for classification,
  - `aux.class_probabilities` (softmax over logits).
- Reconstruction loss:
  - base error tensor: `(outputs["recon"] - batch["x"]) ** 2`.
  - if `reconstruction_normal_only` is enabled and `synthetic_anomaly_mask` exists, loss is computed over normal timesteps only via `_build_normal_time_step_mask(...)`; otherwise full MSE mean is used.
- Classification loss:
  - default is cross-entropy on `outputs["logits"]` vs `batch["classification_labels"]`.
  - label-refurbishment branch exists when `use_label_refurbishment` is enabled.
- Trainer behavior:
  - training loop calls `model.training_step(...)`, then `loss.backward()` and optimizer step.
  - per-epoch metrics include aggregated train/val/val_synth logs and optional reconstruction diagnostics.
  - logits and labels are collected through the epoch for train, val, val_synth; metrics are computed via `compute_binary_classification_metrics` or `compute_multiclass_classification_metrics`.

### Evaluation
- Epoch-level multiclass classification metrics currently include `accuracy`, `macro_f1`, `weighted_f1`, and `num_classes_observed`.
- Pointwise anomaly metrics include `roc_auc`, `pr_auc`, `precision`, `recall`, `f1`, `fpr`, and `vus_pr` (for configured VUS settings).
- Logging outputs:
  - `metrics.jsonl` and optional W&B logging through `ExperimentLogger.log_metrics(...)`.
  - current logger writes scalar metric dictionaries; no dedicated confusion-matrix artifact or class-ratio artifact is currently emitted by trainer.

## Code References
- `scripts/train.py:50` - model/task merge and model construction.
- `scripts/train.py:229` - dataset bundle construction.
- `src/core/config.py:383` - boolean config validation including `balance_binary_classes_within_batch`.
- `src/models/thesis_multitask.py:1686` - classification logits and probabilities.
- `src/models/thesis_multitask.py:1763` - reconstruction loss computation logic.
- `src/models/thesis_multitask.py:1827` - classification loss computation.
- `src/models/thesis_multitask.py:2281` - training step uses `classification_weight=self.lambda_cls`.
- `src/models/thesis_multitask.py:2289` - validation step path with `classification_weight=0.0` and no classification metrics.
- `src/models/thesis_multitask.py:2297` - synthetic validation step with classification metrics enabled.
- `src/data/augment.py:713` - Bernoulli per-sample injection branch when balancing is disabled.
- `src/data/augment.py:657` - uniform anomaly-family sampling by index.
- `src/engine/trainer.py:157` - epoch-level aggregation of classification scalar metrics from concatenated logits/labels.
- `src/metrics/pointwise.py:310` - multiclass metric computation (accuracy, macro_f1, weighted_f1).
- `src/engine/logger.py:130` - scalar metric JSONL and W&B logging.

## Pipeline Documentation
- Training pipeline currently follows:
  1. Resolve experiment, model, task, data config.
  2. Build dataset loaders and model.
  3. Run multitask training with reconstruction and classification outputs.
  4. Aggregate scalar epoch metrics for train, val, val_synth.
  5. Save checkpoints using configured monitor metric.
- For this baseline family, window length in task naming and active design context is `L=20`.
- Classification taxonomy in multiclass synthetic mode is `normal + 11 anomaly families` from `REDLAMP_ANOMALY_FAMILIES`.

## Historical Context (from documents/)
- `documents/design/idea.md` and `documents/design/design_starter.md` state that the active thesis direction keeps multitask reconstruction/classification on fused task representations and emphasizes objective modularity with ablation-friendly switches.
- The design context explicitly positions offline training as the current core phase and retains synthetic anomaly injection as a central supervision mechanism for auxiliary classification.

## Open Questions
- For the intended ablation in Experiment 1, should "disable classification head" be interpreted strictly as zeroing classification loss weight only, or also suppressing classification forward/output logging during train time?
- For Experiment 2 diagnostics, should confusion matrices and class prediction ratios be required for all stages (`train`, `val`, `val_synth`) or only for `train` and `val_synth`?
- Should class prediction ratio be computed from hard argmax predictions only, or from mean softmax probability mass per class as an additional diagnostic view?
