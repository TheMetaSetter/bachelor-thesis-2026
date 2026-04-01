---
date: 2026-03-31 22:27:21 +0700
researcher: TheMetaSetter
git_commit: 87c0e9b2a092b3e3b5a5b6f6ea5b54b4b948555d
branch: dev
repository: bachelor-thesis-2026
topic: "How the current repository would absorb the revised phase 1, 2, and 3 changes before phase 4"
tags: [research, time-series, anomaly-detection, multi-class]
status: complete
last_updated: 2026-03-31
last_updated_by: TheMetaSetter
---

# Research: How the current repository would absorb the revised phase 1, 2, and 3 changes before phase 4

**Date**: 2026-03-31 22:27:21 +0700
**Researcher**: TheMetaSetter
**Git Commit**: 87c0e9b2a092b3e3b5a5b6f6ea5b54b4b948555d
**Branch**: dev

## Research Question

According to the revised design and planning documents, research how the current repository implementation would need to absorb the revised Phase 1, Phase 2, and Phase 3 changes so that those phases are closed before any work proceeds into Phase 4.

## Summary

The repository already contains a runnable offline vertical slice for SMD together with a reconstruction baseline, a multitask model, synthetic anomaly injection, and associated tests. However, the current implementation still reflects the older staged plan rather than the revised plan. The present codebase separates model architecture and training logic across `src/models/`, `src/tasks/`, and `src/losses/`; the training and evaluation scripts register the SMD dataset builder but instantiate data through the concrete loader directly; the synthetic anomaly injector uses three local perturbation types rather than the CARLA-style subsequence anomaly families described in the revised documents; and the repository does not contain a maintained user-facing anomaly-visualization script. The files that would absorb the revised Phase 1 to Phase 3 changes are therefore already identifiable in the current codebase. Phase 1 would center on the existing data, registry, script, baseline model, trainer, evaluator, and checkpoint files. Phase 2 would center on `src/models/thesis_multitask.py` and related registry and contract tests. Phase 3 would center on the current augmentation, multitask task, and prototype-related files, plus a new visualization surface. No active Phase 4 implementation was found in `src`, `scripts`, `tests`, or `configs`.

## Detailed Findings

### Data Preparation

- The active dataset pipeline already centers on SMD. `src/data/datasets/smd.py` provides the parser, `src/data/scalers.py` provides scaling, `src/data/loaders.py` constructs `WindowDataset` and the three dataloaders, and `src/data/collate.py` stacks windows into the batch contract.
- Windowing is implemented inside `src/data/loaders.py` by `WindowDataset`, not through a separate `Windowizer` utility. The dataset returns windows with `x`, optional `point_labels`, optional `mask`, optional `timestamps`, and `meta` containing dataset name, entity identifier, split, start index, end index, and window size.
- `src/data/loaders.py` returns a `data_bundle` with parser, scaler, raw sequences, scaled sequences, datasets, and dataloaders. This existing `data_bundle` is the data path consumed by both `scripts/train.py` and `scripts/evaluate.py`.
- The synthetic augmentation path is not part of the parser or loader. It is invoked later inside the multitask task layer through `src/tasks/multitask_tsad_task.py`.
- `src/data/augment.py` currently implements a `SyntheticAnomalyInjector` that clones a batch and injects one contiguous anomaly segment into one randomly chosen channel. The supported anomaly types are `spike`, `dropout`, and `level_shift`. The injector returns `classification_labels`, `synthetic_anomaly_mask`, and `augmentation_metadata`.
- The repository search did not find a visualization script, plotting helper, or plotting dependency invocation in `src`, `scripts`, or `tests`. This means the present repository has synthetic anomaly generation and anomaly-injection testing, but no maintained user-facing export or inspection surface for injected anomalies.

### Modeling and Training

Terminology normalized on 2026-04-02. Current design target: gate entropy regularization. Current implementation status: the code still uses a barrier-style gate term and should be updated separately.

- The reconstruction baseline is currently split across `src/models/reconstruction_mlp_ae.py` and `src/tasks/reconstruction_task.py`. The model file defines the architecture and forward pass, while the task file computes the reconstruction loss and stage outputs.
- `src/models/reconstruction_mlp_ae.py` already exposes outputs with `hidden`, `pooled`, `recon`, `point_scores`, and `window_scores`, and it validates the batch and output contracts. However, it does not implement `training_step`, `validation_step`, or `test_step` itself.
- `src/models/thesis_multitask.py` currently composes `MultitaskWindowEncoder`, `ContinuousPrototypeLookup`, `DiscretePrototypeLookup`, `TaskFusion`, a reconstruction head, and a classification head. This file already reflects the later thesis architecture at the forward-pass level, but not as a self-contained model file in the revised sense.
- The multitask training logic is located in `src/tasks/multitask_tsad_task.py`, not in `src/models/thesis_multitask.py`. The task file prepares batches, applies synthetic anomaly injection during training, computes reconstruction and cross-entropy classification losses, computes prototype regularization, and calculates classification accuracy.
- Prototype support is split across `src/models/modules/continuous_prototypes.py`, `src/models/modules/discrete_prototypes.py`, and `src/models/modules/fusion.py`. The current continuous branch performs softmax prototype lookup when enabled, the discrete branch performs nearest-codebook quantization when enabled, and the fusion module either keeps the base hidden state or averages branch outputs.
- Loss functions are further separated into `src/losses/classification.py` and `src/losses/prototype.py`. This confirms that the present repository still follows the older multi-file architecture rather than the revised one-model-one-file interpretation.
- `src/engine/trainer.py` currently remains task-driven. It stores both `model` and `task`, moves batches to the selected device, and calls `self.task.training_step(self.model, batch_on_device)` and `self.task.validation_step(self.model, batch_on_device)`.
- `scripts/train.py` and `scripts/evaluate.py` register datasets, models, and tasks in the registry. However, both scripts instantiate the data bundle by calling `build_smd_dataloaders(...)` directly instead of calling `build_dataset("smd", ...)`. This preserves two data-construction paths at once.
- The experiment configuration surface still reflects the older staging. The repository contains `configs/task/reconstruction.yaml` and `configs/task/multitask_tsad.yaml`, and the default experiment file is `configs/experiment/smd_vertical_slice.yaml`, not the revised `smd_reconstruction.yaml` and `smd_multitask.yaml` structure described in the updated documents.

### Evaluation

- `src/engine/evaluator.py` is also task-driven. It calls `self.task.test_step(model, batch_on_device)` and extracts `point_scores` from `step_output["outputs"]`.
- The evaluator accumulates window-level point scores back onto full entity timelines by summing overlapping windows and dividing by overlap counts. It then computes a global threshold as the 95th percentile of all point scores and passes the concatenated labels and scores to `src/metrics/pointwise.py`.
- The evaluation record schema currently used in code is smaller than the revised documentation schema. The current evaluator records only `entity_id`, `point_scores`, `point_labels`, and `num_points`.
- `scripts/evaluate.py` writes `evaluation_records.json` and `evaluation_metrics.json` under the configured output directory. The serialized record keeps only `entity_id`, `point_scores`, `point_labels`, and `num_points`.
- No eventwise metrics, uncertainty reporting, ablation runner, exported synthetic-anomaly visualization, or data-versioning workflow were found in the active repository files inspected during this pass.

## Code References

- `scripts/train.py:18` - train script imports registry helpers and direct SMD dataloader builder
- `scripts/train.py:23` - train script registers the dataset builder
- `scripts/train.py:41` - train script instantiates data through `build_smd_dataloaders(...)` directly
- `scripts/evaluate.py:14` - evaluate script imports registry helpers and direct SMD dataloader builder
- `scripts/evaluate.py:22` - evaluate script registers the dataset builder
- `scripts/evaluate.py:40` - evaluate script instantiates data through `build_smd_dataloaders(...)` directly
- `src/core/registry.py:7` - registry contains dataset, model, task, and encoder builders
- `src/data/loaders.py:11` - `WindowDataset` is implemented in the loader file
- `src/data/loaders.py:56` - `build_smd_dataloaders` constructs parser, scaler, datasets, and loaders
- `src/models/reconstruction_mlp_ae.py:11` - reconstruction baseline model definition
- `src/models/reconstruction_mlp_ae.py:30` - reconstruction model forward path and score computation
- `src/tasks/reconstruction_task.py:10` - reconstruction loss and stage logic
- `src/models/thesis_multitask.py:14` - multitask window encoder definition
- `src/models/thesis_multitask.py:40` - multitask model definition
- `src/models/thesis_multitask.py:79` - multitask model forward path
- `src/tasks/multitask_tsad_task.py:13` - multitask task definition
- `src/tasks/multitask_tsad_task.py:31` - synthetic augmentation enters through `_prepare_batch`
- `src/tasks/multitask_tsad_task.py:48` - multitask loss computation and stage output
- `src/data/augment.py:8` - synthetic anomaly injector definition
- `src/data/augment.py:31` - single-window anomaly injection logic
- `src/data/augment.py:73` - batch-level augmentation and label creation
- `src/models/modules/continuous_prototypes.py:9` - continuous prototype lookup
- `src/models/modules/discrete_prototypes.py:9` - discrete prototype lookup
- `src/models/modules/fusion.py:9` - task fusion module
- `src/engine/trainer.py:12` - trainer stores both model and task
- `src/engine/trainer.py:45` - trainer calls task-owned training logic
- `src/engine/evaluator.py:13` - evaluator stores a task object
- `src/engine/evaluator.py:28` - evaluator calls task-owned test logic
- `tests/test_synthetic_anomaly_injection.py:8` - current synthetic anomaly injection test coverage
- `tests/test_registry.py:31` - registry tests still cover dataset, model, task, and encoder builders

## Pipeline Documentation

As implemented today, the offline repository pipeline is:

```text
raw SMD files
-> SMDDatasetParser
-> SequenceStandardScaler
-> WindowDataset inside src/data/loaders.py
-> collate_windows
-> scripts/train.py or scripts/evaluate.py
-> registry registration for model and task
-> direct build_smd_dataloaders(...) call for data
-> model forward pass
-> task-owned loss computation and stage output
-> Trainer or Evaluator
```

Within that pipeline, the current repository already exposes three distinct surfaces that correspond to the revised phase closure work:

1. Revised Phase 1 maps onto the current files that define the offline vertical slice:
   `src/data/datasets/smd.py`, `src/data/scalers.py`, `src/data/loaders.py`, `src/core/registry.py`, `src/models/reconstruction_mlp_ae.py`, `src/tasks/reconstruction_task.py`, `src/engine/trainer.py`, `src/engine/evaluator.py`, `scripts/train.py`, and `scripts/evaluate.py`.
2. Revised Phase 2 maps onto the current multitask architecture boundary:
   `src/models/thesis_multitask.py`, `src/models/base_encoder.py`, `src/models/modules/continuous_prototypes.py`, `src/models/modules/discrete_prototypes.py`, `src/models/modules/fusion.py`, `configs/model/thesis_multitask.yaml`, and the model and registry tests.
3. Revised Phase 3 maps onto the current multitask augmentation and supervision surface:
   `src/data/augment.py`, `src/tasks/multitask_tsad_task.py`, `src/losses/classification.py`, `src/losses/prototype.py`, `tests/test_synthetic_anomaly_injection.py`, `tests/test_multitask_shapes.py`, and `tests/test_one_multitask_train_step.py`.

The revised documents move the closure criteria earlier, but the code paths that would absorb those changes are already visible in the present repository. By contrast, the repository does not expose an active Phase 4 code path. Searches across `src`, `scripts`, `tests`, and `configs` did not find `src/models/online_adaptation.py`, `src/engine/online_loop.py`, projector code, or online adaptation tests.

## Historical Context (from documents/)

The revised `documents/design/design_starter.md` now defines a four-layer runtime architecture with one-model-one-file as the governing implementation rule. The revised `documents/design/idea.md` adds the same rule to the thesis-facing design and states that a pre-Phase-4 gate should block online adaptation until registry-only construction, CARLA-aligned synthetic anomaly injection, and anomaly inspection support are closed. The revised `documents/logs/03-31-2026/plans/detail-smd-loader-windowing.md` updates the staged plan accordingly: Phase 1 is the vertical slice with registry cleanup and a self-contained baseline model, Phase 2 reserves a self-contained multitask model boundary, Phase 3 covers CARLA-aligned augmentation and visualization support, and only after those are closed may Phase 4 begin.

The current repository state still mirrors the older planning documents more closely than the revised ones. It already contains phase-like implementations for the baseline model, multitask model, prototype branches, and augmentation, but these remain distributed across model, task, module, and loss files. The repository therefore already contains most of the code surfaces that the revised Phase 1 to Phase 3 work would act upon, but it has not yet been reorganized to match the revised document set.

## Open Questions

- The revised documents now call for one-model-one-file as the implementation rule, but the current repository still has active `src/tasks/` and `src/losses/` code. This research pass documents that mismatch, but it does not determine the exact migration sequence within those files.
- The revised Phase 3 language requires CARLA-aligned subsequence anomaly families. The present repository exposes only `spike`, `dropout`, and `level_shift`, so the exact intended naming correspondence between the current anomaly taxonomy and the CARLA taxonomy is not yet encoded in code.
- The revised documents call for user-visible anomaly inspection. The present repository does not contain a maintained visualization script, so the exact expected artifact format is not yet reflected in implementation.
- The revised Phase 4 documents describe an online adaptation path with a residual projector and a stream-oriented loop. No active Phase 4 implementation was found in the repository, so this research note can only document its absence rather than its operational details.
