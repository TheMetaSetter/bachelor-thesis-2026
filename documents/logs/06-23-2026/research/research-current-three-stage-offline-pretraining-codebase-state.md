---
date: 2026-06-23 12:19:48 +0700
researcher: TheMetaSetter
git_commit: be3ef38b1ef8ad8677991e9fbd25bd2414c86d7a
branch: dev
repository: bachelor-thesis-2026
topic: "Current codebase state for the three-stage offline pre-training implementation before further coding"
tags: [research, offline-pretraining, three-stage, smd, thesis_multitask]
status: complete
last_updated: 2026-06-23
last_updated_by: TheMetaSetter
---

# Research: Current codebase state for the three-stage offline pre-training implementation before further coding

**Date**: 2026-06-23 12:19:48 +0700
**Researcher**: TheMetaSetter
**Git Commit**: `be3ef38b1ef8ad8677991e9fbd25bd2414c86d7a`
**Branch**: `dev`

## Research Question

What is the actual current state of the repository around the three-stage offline pre-training experiment for SMD machine `3-4`, and where does the implementation already match or still diverge from the finalized wording and contracts before any further coding is done?

## Summary

The repository already contains an executable three-stage offline pre-training orchestration path for SMD machine `3-4` with `window_size=20`, `stride=1`, and an exact configured training budget of `300` epochs. The active experiment is defined in `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml:7-51`, and the orchestrator is implemented in `scripts/run_three_stage_offline_pretraining.py:29-245`.

The data path is concrete and runnable today. SMD raw entity files are parsed in `src/data/datasets/smd.py:14-181`, the train split is further divided into train and validation by sequence tail split, normalization is fit on the cleaned training sequences in `src/data/loaders.py:150-155`, and windows are then materialized with overlap through `WindowDataset` in `src/data/loaders.py:177-244`. The active experiment for this run uses only `machine-3-4`, `window_size=20`, `stride=1`, and `batch_size=256` from `configs/data/smd_rtx3090_machine_3_4_20_stride1.yaml:1-10`.

The model path is also concrete. The active model is `ThesisMultitaskModel` with a `cnn_simple` encoder, continuous memory, discrete memory, `cosine_topk` discrete querying, and `task_specific_concat_projection` fusion as configured in `configs/model/thesis_multitask_three_stage_window20.yaml:1-58`. Runtime phase behavior is implemented inside `src/models/thesis_multitask.py`, especially phase gating and trainability control in `src/models/thesis_multitask.py:749-862`, memory initialization in `src/models/thesis_multitask.py:1272-1560`, and stage-step loss assembly in `src/models/thesis_multitask.py:3012-3069`.

The main mismatch is semantic rather than infrastructural. The finalized wording of Stage 3 is a single stage called `Memory Initialization and Fusion Warm-Up`, but the current codebase still exposes the schedule as five execution phases: `stage1_classification`, `stage1_reconstruction`, `stage2_recovery`, `stage3_prototype_warmup`, and `multitask_pretraining` in `scripts/run_three_stage_offline_pretraining.py:29-35`. In addition, the current Stage 2 implementation is not true layer-wise Multi-Task Zipping. It is explicitly an approximation that averages matching encoder parameters and reuses task heads, as shown in `scripts/run_three_stage_offline_pretraining.py:280-349`.

## Detailed Findings

### Data Preparation

The active data configuration for the target run is fully specified in `configs/data/smd_rtx3090_machine_3_4_20_stride1.yaml:1-10`. The dataset is `smd`, the root directory is `data/ServerMachineDataset`, and the selected entity list contains only `machine-3-4`.

Raw SMD parsing is implemented in `src/data/datasets/smd.py:14-181`. For each selected entity, the parser loads:

- training features from `train/<entity>.txt`,
- test features from `test/<entity>.txt`,
- test labels from `test_label/<entity>.txt`.

The parser then splits the original SMD train sequence into:

- a train subsequence,
- a validation subsequence,

using `validation_split_ratio` from config, with the validation subsequence taken from the tail of the original training series in `src/data/datasets/smd.py:134-157`. Both derived train and validation subsequences are assigned all-zero point labels, while the test split keeps the real test labels from file in `src/data/datasets/smd.py:159-165`.

The loader stack in `src/data/loaders.py:135-174` then applies:

1. sequence cleaning through `SequenceCleaningPipeline`,
2. standardization through `SequenceStandardScaler.fit(train)` followed by `transform_sequences(...)`,
3. window construction through `WindowDataset`.

This means the current preprocessing order is:

`parse raw sequences -> clean sequences -> fit scaler on cleaned train sequences -> transform train/val/test -> build overlapping windows`.

Window generation is overlap-based. `WindowDataset` iterates from `start_index=0` to `sequence_length - window_size` with the configured stride and stores index records only, then slices windows lazily in `__getitem__` in `src/data/loaders.py:177-244`. The older helper `slice_sequence_into_windows` in `src/data/window.py:16-69` follows the same contract, but the active SMD loader path uses `WindowDataset` directly.

The batch contract exposed to the model is the standard dictionary contract:

- `x`
- `point_labels`
- `mask`
- `timestamps`
- `meta`

as returned by `WindowDataset.__getitem__` in `src/data/loaders.py:204-244`.

### Modeling and Training

The active experiment config locks the exact training budget at `300` epochs in `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml:27,42-48`. The configured split is:

- `stage1_classification_epochs = 50`
- `stage1_reconstruction_epochs = 70`
- `stage2_recovery_epochs = 20`
- `stage3_prototype_warmup_epochs = 20`
- `multitask_pretraining_epochs = 140`

The orchestrator validates that the sum is exact in `scripts/run_three_stage_offline_pretraining.py:53-82`, then expands the run into five training phases in `scripts/run_three_stage_offline_pretraining.py:85-102`.

The active model config in `configs/model/thesis_multitask_three_stage_window20.yaml:1-58` sets:

- `encoder_family: cnn_simple`
- `window_size: 20`
- `input_dim: 38`
- `hidden_dim: 32`
- `num_classes: 12`
- `continuous_num_prototypes: 16`
- `discrete_codebook_size: 60`
- `enable_two_view_contrastive: true`
- `lambda_contrastive: 0.1`
- `fusion_mode: task_specific_concat_projection`
- `discrete_query_mode: cosine_topk`
- `discrete_topk: 3`
- `discrete_query_temperature: 0.1`
- `freeze_memories_after_initialization: true`
- `freeze_recovered_zipped_encoder_during_warmup: true`
- `discrete_memory_label_source: synthetic_train_labels`

`scripts/train.py:55-82` merges model config and task config into one owning model file, then `run_training_experiment(...)` executes the current runtime path:

`seed -> register components -> build dataset bundle -> build model -> optional checkpoint init -> optimizer -> logger -> trainer`

as shown in `scripts/train.py:224-320`.

Inside `ThesisMultitaskModel`, the current runtime semantics are phase-dependent:

- Prototype branches are only active in `stage3_prototype_warmup` and `multitask_pretraining` through `_phase_uses_prototype_path()` in `src/models/thesis_multitask.py:749-753`.
- Contrastive loss is only active in `stage1_classification`, `stage1_reconstruction`, and `multitask_pretraining` through `_phase_uses_contrastive_objective()` in `src/models/thesis_multitask.py:755-760`.
- The encoder is frozen only when the phase is `stage3_prototype_warmup` and the warm-up freeze flag is true, through `_phase_freezes_encoder()` in `src/models/thesis_multitask.py:774-778`.

The trainable-parameter gate is controlled by `_configure_trainable_parameters_for_phase()` in `src/models/thesis_multitask.py:791-862`. This implementation already does several important things correctly:

- Stage 1 classification disables the reconstruction head.
- Stage 1 reconstruction disables the classification head.
- Non-prototype phases disable concat-projection fusion layers and prototype-side trainable modules.
- Stage 3 warm-up freezes the encoder.

However, the current code does not yet strictly enforce the finalized Stage 3 wording "train only the task heads and task-specific concat-projection fusion layers." The code freezes the encoder in Stage 3, but its module-level trainability control is broader and still expressed through the older phase abstraction in `src/models/thesis_multitask.py:791-862`.

Memory initialization is implemented inside the model and triggered by the trainer hook. In `src/engine/trainer.py:595-611`, the trainer calls `set_epoch_context(...)` and then `maybe_initialize_memories_from_loader(...)` at the start of each epoch. The actual initialization logic is in `src/models/thesis_multitask.py:1286-1560`.

The current memory initialization behavior is:

- only prototype phases may initialize memories,
- initialization runs only when memories are not already initialized,
- initialization reads a limited number of train batches according to `memory_initialization_batches`,
- continuous memory seeds are selected from normal-only hidden tokens,
- discrete memory seeds are selected class-stratified by synthetic training labels,
- both memory banks are normalized and then frozen after initialization if `freeze_memories_after_initialization` is true.

The continuous memory token pool is built from normal positions in synthetically augmented training windows in `src/models/thesis_multitask.py:1452-1465`. The discrete memory token pool is built from the same recovered training batch path but grouped by synthetic class label in `src/models/thesis_multitask.py:1466-1476`. The actual class-stratified covering selection is implemented in `src/models/thesis_multitask.py:1538-1556`.

This means the repository already implements the training-split-only rule for synthetic anomaly generation and discrete-memory labels, but it currently does so on the first `memory_initialization_batches` train batches rather than exhaustively over the full recovered training feature population.

The Stage 2 initialization path is implemented in `scripts/run_three_stage_offline_pretraining.py:280-349`. This script loads the best Stage 1 classification and Stage 1 reconstruction checkpoints, then constructs a Stage 2 initialization checkpoint by:

- averaging matching encoder parameters,
- copying the classification head from the classification checkpoint,
- copying the reconstruction head from the reconstruction checkpoint.

The script itself records that this is `zipping_approximation="parameter_average_identity_matching"` in `scripts/run_three_stage_offline_pretraining.py:341-348`. Therefore, the current codebase does not yet implement the finalized sequential layer-wise Multi-Task Zipping interpretation from the design note.

### Evaluation

The trainer performs normal validation and realistic validation during training. In `src/engine/trainer.py:679-735`, the validation flow supports:

- `validation_step` for the ordinary validation path,
- `realistic_validation_step` when realistic validation is available,
- fallback to synthetic validation otherwise.

For realistic validation anomaly rate estimation, the trainer uses the real SMD test window anomaly rate through `compute_smd_test_window_anomaly_rate(...)` in `src/engine/trainer.py:536-558` and `src/data/datasets/smd.py:184-241`.

The trainer also computes pointwise metrics, classification diagnostics, and optional evaluator metrics during training in `src/engine/trainer.py:737-835`. Checkpoint behavior is robust:

- best checkpoint is saved when the monitored metric improves,
- final checkpoint is always saved,
- if best checkpoint was never improved, the trainer falls back to the final state,
- if memory becomes initialized after the best checkpoint snapshot, the best checkpoint is refreshed with initialized memory state.

These behaviors are implemented in `src/engine/trainer.py:849-940`.

At the orchestration level, `scripts/run_three_stage_offline_pretraining.py:195-245` writes a three-stage manifest with one generated config per training phase and one evaluation reference config. The execution path then runs the phases sequentially and records an execution report in `scripts/run_three_stage_offline_pretraining.py:352-430`.

## Code References

- `documents/logs/06-17-2026/detail/detail-offline-pretraining-three-stage-discussion-context.md:1-211` - finalized design context and stage terminology
- `documents/design/idea.md:1-220` - broader design context that still contains older offline semantics
- `documents/design/design_starter.md:1-220` - current repository contract philosophy and folder-level architecture
- `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml:7-51` - active RTX 3090 experiment config and exact `300`-epoch schedule
- `configs/data/smd_rtx3090_machine_3_4_20_stride1.yaml:1-10` - active SMD machine `3-4` data config
- `configs/model/thesis_multitask_three_stage_window20.yaml:1-58` - active model/runtime config for the experiment
- `src/data/datasets/smd.py:14-181` - SMD raw sequence parser and train/val/test split creation
- `src/data/datasets/smd.py:184-241` - SMD test-window anomaly-rate computation
- `src/data/loaders.py:135-174` - scaler fit/transform and loader bundle construction
- `src/data/loaders.py:177-244` - lazy overlapping `WindowDataset` implementation
- `src/data/window.py:16-69` - shared sequence-to-window slicing helper
- `scripts/train.py:55-82` - model build contract from experiment config
- `scripts/train.py:224-320` - training experiment runtime graph
- `src/models/thesis_multitask.py:749-862` - phase semantics and trainability control
- `src/models/thesis_multitask.py:1272-1560` - memory initialization and covering selection
- `src/models/thesis_multitask.py:1770-1939` - active continuous/discrete lookup and concat-projection fusion
- `src/models/thesis_multitask.py:3012-3069` - phase-aware loss assembly
- `src/engine/trainer.py:560-940` - epoch loop, memory-init hook, validation, and checkpoint policy
- `scripts/run_three_stage_offline_pretraining.py:29-245` - phase plan and manifest generation
- `scripts/run_three_stage_offline_pretraining.py:280-349` - Stage 2 initialization approximation
- `scripts/run_three_stage_offline_pretraining.py:352-430` - sequential orchestration execution and report writing

## Pipeline Documentation

The current repository path for the target experiment is:

`raw SMD entity files -> parser split into train/val/test sequences -> cleaning -> scaler fit on train sequences -> transform all splits -> overlapping windows with size 20 and stride 1 -> batched window dictionaries -> phase-aware ThesisMultitaskModel -> trainer-managed epoch loop -> phase checkpoints -> sequential three-stage orchestration -> final evaluation`

Within that pipeline, the current execution semantics are not literally "three training stages" in runtime form. The repository currently realizes the workflow as five concrete phases:

1. `stage1_classification`
2. `stage1_reconstruction`
3. `stage2_recovery`
4. `stage3_prototype_warmup`
5. `multitask_pretraining`

This five-phase runtime still aims to approximate the conceptual three-stage design, but the naming and execution surfaces are not yet fully aligned with the finalized wording of Stage 3.

## Historical Context (from documents/)

`documents/design/idea.md` and `documents/design/design_starter.md` still encode the broader thesis architecture and the general repository contracts, especially the hidden-state contract `H in R^{B x L x d_h}` and the one-model-one-file rule. For the current first implementation, however, the more specific contract is the locked discussion note in `documents/logs/06-17-2026/detail/detail-offline-pretraining-three-stage-discussion-context.md`.

That note records several decisions that matter directly for present code reading:

- exact total offline pre-training budget is `300` epochs,
- Stage 1 uses two separate task-specific models,
- Stage 2 is meant to be Multi-Task Zipping,
- Stage 3 is a single semantic stage combining memory initialization and short fusion warm-up,
- both memory banks are frozen after initialization,
- memory initialization must be derived from the training split only,
- forward fusion for the first implementation should be `task_specific_concat_projection`,
- CKA is diagnostic-only, not part of the forward path.

The codebase already matches some of these decisions well, especially the frozen-memory policy, the train-split-only memory source, the `cosine_topk` discrete query, the `task_specific_concat_projection` fusion mode, and the exact `300`-epoch config budget. The remaining gaps are concentrated in execution semantics and Stage 2/Stage 3 interpretation.

## Open Questions

1. Should the runtime surface continue to expose five execution phases while treating them as an implementation detail of the conceptual three-stage design, or should the naming and reporting be revised so Stage 3 is represented explicitly as one semantic stage with two substeps?
2. Should memory initialization continue to sample only the first `memory_initialization_batches` training batches, or must the first implementation exhaustively cover the full recovered training feature pool to match the exact wording more strictly?
3. Should Stage 2 remain an explicitly documented approximation for now, or must it be replaced before further experimentation because parameter averaging is not semantically equivalent to the intended layer-wise Multi-Task Zipping procedure?
