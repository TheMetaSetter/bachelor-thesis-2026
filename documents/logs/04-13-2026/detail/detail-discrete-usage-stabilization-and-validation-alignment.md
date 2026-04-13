# Discrete Usage Stabilization and Validation Alignment

## Summary

This change addresses two observed problems in the offline `thesis_multitask` path.

1. The discrete branch was sharpening too early. Discrete usage entropy dropped quickly, top-1 usage rose early, and code usage concentration increased before the codebook had time to explore broadly.
2. The old validation classification metrics were semantically weak. Training classification uses synthetic clean-versus-corrupted labels, but the clean validation split contains only clean windows, so `val_classification_loss` and `val_classification_accuracy` were not measuring the same task.

The implementation keeps the current binary classifier and conservative loss design, then makes two targeted changes:

- add scheduled discrete-usage stabilization with a temperature hold phase and epoch-dependent usage-loss weight
- split validation into a clean path and a deterministic synthetic classification path

## Implemented Changes

### 1. Scheduled early discrete exploration

The multitask model now supports:

- `temperature_hold_fraction`
- `usage_lambda_start`
- `usage_lambda_end`
- `usage_lambda_schedule_fraction`

The effective objective becomes:

\[
L_{\text{total}}
=
L_{\text{recon}}
+
\lambda_{\text{cls}} L_{\text{cls}}
+
\lambda_{\text{use}}(e) L_{\text{use}}
\]

where `lambda_use(e)` is epoch-dependent. This keeps the weighted-sum objective explicit while allowing stronger early pressure against codebook collapse.

The temperature schedule now has two phases:

- hold `temperature = temperature_start` during an explicit early exploration phase
- anneal toward `temperature_end` only after that hold period

This is a deterministic schedule, not an adaptive controller. The goal is to preserve early exploration without adding multiple regularizers at once.

### 2. Clean validation and synthetic validation are now separate

Validation is now split into two surfaces:

- `val_*`
  - clean validation only
  - no synthetic augmentation
  - used for reconstruction and branch-health diagnostics aligned with the real clean validation path

- `val_synth_*`
  - deterministic synthetic validation
  - uses the same binary classification task as training
  - used for classification diagnostics and discrete-usage diagnostics under synthetic corruption

The synthetic validation injector uses a fixed seed and resets its RNG each validation epoch, so curves are comparable across epochs and runs.

## Why this design was chosen

This repository should stay readability-first and keep the active thesis path understandable from the model file and YAML alone.

The chosen design therefore:

- keeps one model per file
- keeps the classifier binary
- avoids adding multiple regularizers at the same time
- keeps clean validation aligned with the real test setting
- adds a second deterministic diagnostic surface instead of overloading one validation stream with two incompatible meanings

The design does **not** introduce:

- anomaly-family multiclass supervision
- dynamic entropy feedback control
- additional branch regularizers such as diversity, variance, covariance, or gate regularization in the same step

## Logging Contract

The active logging interpretation is now:

- `train_*`
  - synthetic multitask training metrics
- `val_*`
  - clean validation metrics
- `val_synth_*`
  - deterministic synthetic validation classification metrics

This removes the previous ambiguity where clean-only validation classification metrics could be misread as if they reflected the synthetic classification task used during training.

## Test Coverage Added

Strict targeted tests were added to lock this behavior:

- temperature hold before annealing
- epoch-dependent usage-lambda schedule
- usage-loss contribution to total loss when scheduled
- clean validation omits classification metrics
- synthetic validation is deterministic after RNG reset
- trainer logs `val_*` and `val_synth_*` separately in one epoch

The regression suite covering config loading, multitask shapes, synthetic anomaly injection, and ablation execution remains green after this change.

## Current Research Interpretation

This change does not claim the discrete branch is solved. It makes the next comparison runs interpretable.

The immediate experimental questions after this implementation are:

- does the early entropy curve stay flatter than before
- does top-1 usage stop spiking early
- does clean validation remain stable
- do `val_synth_classification_*` curves become interpretable enough to support diagnosis

The next comparison should still be conservative:

- same binary classifier
- same main reconstruction path
- same clean test protocol
- targeted changes only around early discrete exploration and validation semantics
