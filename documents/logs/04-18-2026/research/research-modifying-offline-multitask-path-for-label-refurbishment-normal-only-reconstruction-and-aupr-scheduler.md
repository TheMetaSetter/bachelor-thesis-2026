---
date: 2026-04-18 17:55:33 +07 +0700
researcher: TheMetaSetter
git_commit: c70a3dbe70bf60b74cda147917fde24a418ee79d
branch: dev
repository: bachelor-thesis-2026
topic: "How to modify the current offline multitask codepath to support label refurbishment, reconstruction loss on normal time steps only, and learning-rate scheduling from AUC-PR"
tags: [research, multitask, anomaly-detection, label-refurbishment, anomaly-mask, scheduler]
status: complete
last_updated: 2026-04-18
last_updated_by: TheMetaSetter
---

# Research: Modifying the Offline Multitask Path for Label Refurbishment, Normal-Only Reconstruction, and AUC-PR Scheduler Monitoring

**Date**: 2026-04-18 17:55:33 +07 +0700  
**Researcher**: TheMetaSetter  
**Git Commit**: `c70a3dbe70bf60b74cda147917fde24a418ee79d`  
**Branch**: `dev`

## Research Question

With the current status of the codebase, how should the offline multitask training path be modified so that:

1. it supports label refurbishment in the spirit of `bsc-thesis-ref-codebases/RedLamp/models/meta.py`,
2. reconstruction loss is computed only on normal time steps, and
3. the learning-rate scheduler monitors AUC-PR instead of AUC-ROC?

## Summary

The current thesis multitask implementation already contains three of the four runtime surfaces needed for this request:

- the model owns the multitask loss assembly in one file, namely `src/models/thesis_multitask.py`,
- synthetic augmentation already emits a per-window anomaly mask as `synthetic_anomaly_mask`,
- the trainer already aggregates `val_synth_pr_auc`, and
- the scheduler builder already treats every non-`val_loss` monitor as a maximization target.

The main missing pieces are not in the trainer loop. They are in the label contract and in the loss definitions. At present, the multitask classification path is binary clean-versus-anomalous classification, not anomaly-family classification. The current model uses integer labels with `F.cross_entropy`, while RedLamp-style label refurbishment expects a dense class-probability target. The current reconstruction loss is also an unconditional mean squared error over all time steps, even when synthetic anomalies are present.

Therefore, the smallest coherent modification set is:

- extend the thesis multitask model so it can build and consume refurbished probability targets while still preserving hard labels for metrics,
- compute reconstruction loss with a normal-time-step mask derived from the existing anomaly mask surface,
- expose these behaviors through explicit YAML fields, and
- allow `val_synth_pr_auc` as a scheduler monitor in config validation and experiment YAML.

## Detailed Findings

### Data Preparation and Label Surfaces

The current synthetic multitask augmentation path is implemented in `src/data/augment.py`. It creates:

- `classification_labels` with shape `[B]` and values in `{0, 1}`,
- `synthetic_anomaly_mask` with shape `[B, L]`, where `1` denotes anomalous time steps introduced by augmentation,
- `augmentation_metadata`, which records the selected anomaly family, and
- `point_labels`, which are updated by `torch.maximum(original_point_labels, anomaly_masks)`.

This behavior is visible in `src/data/augment.py:552-607`. The decisive detail is that the code always writes `classification_labels[batch_index] = 1` whenever any anomaly family is injected, regardless of which family was chosen (`src/data/augment.py:568-578`). The current task therefore collapses all synthetic anomaly families into one anomalous class.

The model configuration also confirms that the active classifier is binary. The default model config sets `num_classes: 2` in `configs/model/thesis_multitask.yaml:1-32`, especially line 6. The task config lists the eleven RedLamp-inspired anomaly families, but those families currently affect augmentation mechanics and metadata, not classifier targets (`configs/task/multitask_tsad.yaml:1-23`).

This point matters because the RedLamp refurbishment formula in `bsc-thesis-ref-codebases/RedLamp/models/meta.py:41-80` assumes a one-hot class vector with one normal class and multiple anomaly classes. In the current thesis model, that class structure does not yet exist.

### Modeling and Training

The thesis multitask model remains the correct ownership point for these changes, because the repository explicitly keeps one model in one file and keeps task-specific loss logic colocated with the model. The current loss assembly is in `src/models/thesis_multitask.py:684-924`.

The current behavior is:

- `_compute_reconstruction_loss` returns a plain mean squared error over all batch elements and all time steps (`src/models/thesis_multitask.py:684-690`).
- `_compute_classification_loss` uses `F.cross_entropy(outputs["logits"], batch["classification_labels"].long())` (`src/models/thesis_multitask.py:691-696`).
- `_shared_step` always prepares the batch, runs the forward pass, computes reconstruction and classification loss, then adds any optional regularizers (`src/models/thesis_multitask.py:865-924`).

The batch preparation path is also important. For training, `_prepare_clean_batch` delegates to `self.synthetic_anomaly_injector.augment_batch(batch)` when synthetic augmentation is enabled (`src/models/thesis_multitask.py:529-581`). For clean validation and test, it creates zero `classification_labels` and zero `synthetic_anomaly_mask` tensors (`src/models/thesis_multitask.py:548-580`). For synthetic validation, `_prepare_batch` delegates to `self.synthetic_validation_injector.augment_batch(batch)` (`src/models/thesis_multitask.py:583-586`).

This means the model already sees the anomaly mask during the same stage where it computes reconstruction and classification losses. No trainer-level redesign is required for masking reconstruction loss.

### Validation Metrics and Scheduler Path

The trainer already aggregates binary classification metrics from logits and hard labels through `compute_binary_classification_metrics` (`src/engine/trainer.py:59-84`). That metric function already computes both ROC-AUC and PR-AUC (`src/metrics/pointwise.py:66-79`).

During training, the trainer aggregates:

- `train_*` classification metrics from training batches,
- `val_*` metrics only from clean validation logs, and
- `val_synth_*` classification metrics from synthetic validation batches.

This is visible in `src/engine/trainer.py:237-283`. Because the synthetic validation metrics are aggregated with `stage_name="val_synth"`, the epoch metrics already include `val_synth_pr_auc` whenever `synthetic_validation_step` is available.

The scheduler stepping logic is generic. It reads whichever metric name is configured in `self.scheduler_monitor_metric` and calls `self.scheduler.step(monitor_value)` (`src/engine/trainer.py:89-127`). The builder in `scripts/train.py:65-104` already sets the scheduler to `"max"` mode for every monitor other than `val_loss`. Therefore, if `val_synth_pr_auc` is accepted by configuration, the runtime scheduler behavior is already aligned with maximization.

The current limitation is the config validator. `src/core/config.py:157-186` only allows `monitor_metric` to be either `val_loss` or `val_synth_roc_auc`. The active experiment YAML also still points to `val_synth_roc_auc` in `configs/experiment/smd_multitask_rtx3090_seed11_machine_2_1_val_synth_roc_auc.yaml:14-25`.

## Modification Map

### 1. Add label refurbishment to the thesis multitask model

The smallest codebase-aligned implementation is to keep hard labels for metrics and introduce a second, loss-only probability target inside `src/models/thesis_multitask.py`.

Concretely:

- Add explicit model or task config fields such as `use_label_refurbishment`, `refurbishment_alpha`, `refurbishment_beta`, and `normal_class_index`.
- Keep `batch["classification_labels"]` as the hard integer labels used by accuracy, PR-AUC, and ROC-AUC logging.
- Inside `_compute_classification_loss`, derive a dense target distribution from the hard labels when refurbishment is enabled.
- Compute classification loss against that dense target, either by an explicit `log_softmax` formulation or another probability-target cross-entropy implementation.

This separation is important because the current logging path compares `argmax(logits)` to `batch["classification_labels"]` in `src/models/thesis_multitask.py:854-863`, and the trainer metric aggregation path in `src/engine/trainer.py:59-84` expects integer labels.

There are two implementation depths:

- Minimal binary-compatible path: keep the current binary taxonomy and refurbish a two-class target `[normal, anomalous]`. This is the least invasive change and preserves the current metric pipeline.
- Full RedLamp-consistent path: change the classifier target space to one normal class plus one class per anomaly family. This would require the injector to emit family-aware class indices rather than a binary anomaly flag.

The current repository state supports the first path immediately. The second path is more faithful to the RedLamp formulation, but it is not a drop-in change because the current taxonomy is binary.

### 2. Compute reconstruction loss only on normal time steps

The current `synthetic_anomaly_mask` marks anomalous time steps with `1` (`src/data/augment.py:564-587`). Your requested behavior is the complement of RedLamp's direct multiplication example. In RedLamp, multiplying by `anomaly_mask` keeps the masked region defined by that reference implementation (`bsc-thesis-ref-codebases/RedLamp/models/meta.py:45-52`). In the thesis repository, using the current mask directly would keep anomalous time steps, not normal ones.

Therefore, the reconstruction mask in `src/models/thesis_multitask.py:684-690` should be built as:

- `normal_time_mask = 1 - synthetic_anomaly_mask`,
- expanded to `[B, L, 1]` before applying it to `[B, L, D]` tensors,
- converted to the reconstruction dtype, and
- normalized by the number of active normal cells so the loss scale does not shrink simply because more anomaly points were injected.

The cleanest implementation is:

- add a config flag such as `reconstruction_normal_only`,
- in `_compute_reconstruction_loss`, when enabled and when `synthetic_anomaly_mask` exists, compute a masked mean squared error over normal positions only,
- fall back to the current full-window mean squared error if the batch is clean or if the denominator would be zero.

If future training datasets contain real anomaly point labels, a slightly broader variant is to derive the normal mask from `point_labels` instead of only `synthetic_anomaly_mask`, because `point_labels` already merges original labels with injected anomalies in `src/data/augment.py:580-584`. That broader choice would exclude both real and synthetic anomaly time steps from reconstruction supervision.

### 3. Switch the scheduler to AUC-PR

This is the narrowest modification in the request because most of the runtime path already exists.

Required changes:

- Allow `val_synth_pr_auc` in `src/core/config.py:157-186`.
- Update experiment YAML files that currently use `val_synth_roc_auc`, for example `configs/experiment/smd_multitask_rtx3090_seed11_machine_2_1_val_synth_roc_auc.yaml:14-25`.
- Update scheduler-related tests that currently validate only `val_loss` and `val_synth_roc_auc`.

No trainer change is required for this part, because:

- `val_synth_pr_auc` is already produced by the existing metric aggregation path (`src/engine/trainer.py:275-283`, `src/metrics/pointwise.py:66-79`),
- the scheduler already reads the configured monitor by name (`src/engine/trainer.py:89-127`), and
- the scheduler builder already uses `"max"` mode for non-`val_loss` metrics (`scripts/train.py:78-80`).

### 4. Decide whether checkpoint selection should remain on `val_loss`

The current trainer still chooses the best checkpoint only from `val_loss` (`src/engine/trainer.py:288-301`), even if the scheduler monitors a different metric. That is not a bug in the current implementation; it is simply a separate selection rule.

If the intended experiment policy is:

- scheduler follows AUC-PR, but best checkpoint remains reconstruction-driven clean validation loss,

then no checkpoint logic change is needed.

If the intended experiment policy is:

- scheduler and best-checkpoint selection should both follow AUC-PR,

then checkpoint selection must be modified separately in `src/engine/trainer.py:288-301`.

## Required Configuration Surface

The current configs do not expose any label-refurbishment or reconstruction-masking switches. `configs/model/thesis_multitask.yaml:1-32` and `configs/task/multitask_tsad.yaml:1-23` contain no fields for these behaviors.

To keep the codebase ablation-friendly and consistent with `codebase_preferences.md`, the following fields should be added explicitly through YAML and validated in `src/core/config.py`:

- `use_label_refurbishment: bool`
- `refurbishment_alpha: float`
- `refurbishment_beta: float`
- `reconstruction_normal_only: bool`
- `scheduler.monitor_metric: val_synth_pr_auc`

These changes preserve the current single-model, explicit-config, minimal-codepath design.

## Testing Surface That Must Change

The current tests assume the old contracts in several places:

- `tests/test_one_multitask_train_step.py:56` assumes `classification_labels` is a hard binary tensor whose sum equals the number of anomalous windows.
- `tests/test_learning_rate_scheduler.py:132-142` and `tests/test_learning_rate_scheduler.py:271-309` currently validate the `val_synth_roc_auc` monitor path.
- `tests/test_config_loading.py` currently validates only `val_loss` and `val_synth_roc_auc` as accepted scheduler monitors.

Minimal additional tests should cover:

- refurbished classification loss still backpropagates while hard labels remain available for metrics,
- masked reconstruction loss ignores anomalous time steps and preserves the full-loss behavior on clean batches,
- config loading accepts the new refurbishment and masking flags,
- scheduler config accepts `val_synth_pr_auc`,
- trainer metric history records `scheduler_monitor_val_synth_pr_auc` when that monitor is selected.

## Code References

- `src/data/augment.py:552-607` - synthetic augmentation writes binary `classification_labels`, `synthetic_anomaly_mask`, and merged `point_labels`
- `src/models/thesis_multitask.py:529-586` - stage-aware batch preparation for train, clean validation, and synthetic validation
- `src/models/thesis_multitask.py:684-696` - current reconstruction and classification loss definitions
- `src/models/thesis_multitask.py:854-863` - hard-label accuracy logging path
- `src/models/thesis_multitask.py:865-924` - shared multitask loss assembly
- `src/engine/trainer.py:59-84` - binary classification metric aggregation
- `src/engine/trainer.py:89-127` - scheduler stepping from any named epoch metric
- `src/engine/trainer.py:288-301` - best checkpoint remains tied to `val_loss`
- `src/core/config.py:157-186` - scheduler monitor validation currently rejects `val_synth_pr_auc`
- `scripts/train.py:65-104` - scheduler builder already supports maximization monitors
- `src/metrics/pointwise.py:66-79` - binary classification metrics already include PR-AUC
- `configs/model/thesis_multitask.yaml:1-32` - current classifier has `num_classes: 2`
- `configs/task/multitask_tsad.yaml:1-23` - anomaly families exist at augmentation level
- `bsc-thesis-ref-codebases/RedLamp/models/meta.py:41-80` - reference loss path for anomaly masking and label refurbishment

## Historical Context (from documents/)

The design documents emphasize three constraints that support this modification approach:

- `documents/design/idea.md` keeps the offline objective modular and colocated with the thesis model.
- `documents/design/design_starter.md` favors a thin contract between data, model, and engine layers.
- `codebase_preferences.md` requires readability-first design, one model per file, and explicit ablation-friendly configuration.

The requested changes fit those constraints best when implemented as additional explicit switches inside `src/models/thesis_multitask.py`, not as a second multitask model or a trainer-side special case.

## Open Questions

1. Should label refurbishment be implemented only for the current binary clean-versus-anomalous classifier, or should the classifier be expanded to one normal class plus one class per anomaly family?
2. Should normal-only reconstruction masking use only `synthetic_anomaly_mask`, or should it use `point_labels` so that real anomaly labels are excluded as well when available?
3. Should best-checkpoint selection remain tied to clean `val_loss`, or should it also move to `val_synth_pr_auc` once the scheduler monitor changes?
