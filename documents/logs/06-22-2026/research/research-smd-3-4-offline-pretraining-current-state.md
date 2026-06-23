---
date: 2026-06-22 16:17:09 +0700
researcher: TheMetaSetter
git_commit: d310f2c3b3f36cd260870504ac6811c2720c9952
branch: dev
repository: bachelor-thesis-2026
topic: "Current-state audit before implementing automated offline pre-training for SMD 3-4 on an RTX 3090 server"
tags: [research, offline-pretraining, smd, smd-3-4, server-automation, multitask]
status: complete
last_updated: 2026-06-22
last_updated_by: Codex
---

# Research: Current-state audit before implementing automated offline pre-training for SMD 3-4 on an RTX 3090 server

**Date**: 2026-06-22 16:17:09 +0700  
**Researcher**: TheMetaSetter  
**Git Commit**: `d310f2c3b3f36cd260870504ac6811c2720c9952`  
**Branch**: `dev`

## Research Question

Audit the current repository state before planning code changes for an automated offline pre-training experiment on SMD `machine-3-4`, with careful data preparation, server-side execution on one RTX 3090 via `tmux`, and final train/test behavior that explicitly uses windows cut from the test sequence.

## Summary

The repository already has a solid SMD parsing and preprocessing backbone for this work. The current offline data path is: parse full SMD sequences, split the training sequence into train/val at the raw-sequence level, fit one feature-wise standard scaler on the full train sequences only, transform train/val/test sequences with that scaler, then cut windows and build PyTorch data loaders. This part is already reusable for `machine-3-4`; the dataset file exists, and the parser already supports `entity_ids` filtering.

The main problem is not the loader foundation. The main problem is that the current `thesis_multitask` implementation and its experiment configs still represent an older single-loop prototype-fusion design, not the three-stage first-implementation contract that was locked in the June 22 discussion note. In particular, the current model still uses a learned discrete assignment layer plus Gumbel-Softmax, EMA-updated discrete memory, adaptive continuous memory updates, scalar fusion weights with optional CKA-gated forward fusion, and one shared end-to-end multitask training loop. There is no current runtime support for: separate Stage 1 task-specific encoders, Stage 2 zipping, Stage 2 recovery, Stage 3 frozen-memory warm-up, `cosine_topk` discrete querying, or `task_specific_concat_projection`.

There is also a configuration mismatch with the requested target run. The active window-20 thesis configs still point to `machine-2-1`, not `machine-3-4`, and the current `machine-2-1` window-20 data config uses `stride: 20`, which means non-overlapping windows. That is materially different from the overlap-aware semantics discussed in the design notes around same-source-timestep positives and overlap-aware reasoning.

## Detailed Findings

### Data Preparation

The current SMD parser reads raw train, test, and test-label files per machine entity, validates entity membership, and returns full-length sequences with metadata such as `dataset_name`, `entity_id`, `split`, `num_channels`, and `sequence_length` (`src/data/datasets/smd.py:14-179`). When `entity_ids` is provided, the parser filters to the requested entities and validates that the matching train/test/label files exist (`src/data/datasets/smd.py:85-105`). This means `machine-3-4` can already be targeted by configuration alone, assuming the config file is added.

The current train/val split is created from the original train sequence itself, not from the test sequence. For each entity, the parser cuts the last `validation_split_ratio` fraction from the raw train tensor into the validation split, while the entire original test tensor remains the test split (`src/data/datasets/smd.py:132-163`). Therefore, the repository already distinguishes three sequence scopes clearly: train tail-split validation, full test sequence, and test labels.

The current preprocessing order is exactly normalize-first then window-after. `SequenceStandardScaler.fit` concatenates all train-sequence points and computes one feature-wise mean and standard deviation on the train split only (`src/data/scalers.py:16-27`). `src/data/loaders.py` then applies that fitted scaler to `train`, `val`, and `test` full sequences before building `WindowDataset` objects (`src/data/loaders.py:135-174`). This is consistent with the previously established thesis preference that normalization must happen on full sequences before window slicing.

Window slicing is index-based and happens after scaling. `WindowDataset` stores `(sequence_index, start_index, end_index)` triples and materializes each window on demand, while preserving `entity_id`, `split`, `start_index`, and `end_index` metadata per window (`src/data/loaders.py:177-228`). The actual window loop is `range(0, sequence_length - window_size + 1, stride)` (`src/data/loaders.py:191-199`). So overlap or non-overlap is fully controlled by the data config.

For the active thesis `exp2` config, the current data file is still `configs/data/smd_rtx3090_machine_2_1_20.yaml`, which pins the run to `machine-2-1`, `window_size: 20`, and `stride: 20` (`configs/data/smd_rtx3090_machine_2_1_20.yaml:1-10`). This is currently a non-overlapping window setup. There is no existing `machine-3-4` counterpart under `configs/data/`.

### Modeling and Training

The active thesis `exp2` experiment is still wired to the old window-20 single-entity run on `machine-2-1`. The experiment YAML points to the `machine-2-1` data config, uses `epochs: 300`, and enables `enable_two_view_contrastive: true` plus `enable_cka_gated_fusion: true` through model overrides (`configs/experiment/thesis/exp2/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20-exp2__w20__seed11__default.yaml:7-50`).

The model config still encodes the older design assumptions. It enables both continuous and discrete branches, uses a `discrete_codebook_size` with `gumbel_temperature`, and still exposes the old temperature-annealing and memory-initialization controls (`configs/model/thesis_multitask_redlamp_multiclass.yaml:1-49`). This config does not express any of the new first-implementation concepts such as frozen memories, `cosine_topk`, class-stratified discrete covering selection, or concat-projection fusion.

The current `ThesisMultitaskModel` is still a one-loop prototype-fusion model. The discrete branch is explicitly built around a learned `self.discrete_assignment = nn.Linear(...)` and a persistent `discrete_codebook` with EMA state buffers (`src/models/thesis_multitask.py:703-732`). The continuous branch also has a learnable update gate for online bank updates (`src/models/thesis_multitask.py:734-740`).

The current memory initialization path is still generic “covering from hidden tokens,” not the finalized class-aware/frozen-memory protocol from the June 22 note. The model collects hidden tokens from the train loader, optionally also from synthetic windows, then seeds both continuous and discrete memories from that single token pool (`src/models/thesis_multitask.py:1201-1289`). This directly conflicts with the newer design lock that says memory initialization must be train-derived but split by role: continuous memory from normal-only covering, discrete memory from class-stratified covering, both frozen after initialization.

The current discrete update mechanism is not `cosine_topk`. Instead, it computes assignment logits with the learned `discrete_assignment` layer, applies `F.gumbel_softmax(...)`, and updates the codebook through EMA counts and EMA sums (`src/models/thesis_multitask.py:1360-1424`). The read path also reconstructs the discrete hidden state through a full weighted sum over the codebook using those Gumbel-Softmax assignment probabilities (`src/models/thesis_multitask.py:1485-1538`). This is an old semantics branch, not the finalized first-pass discrete-query design.

The current fusion mechanism is also older than the note-finalized contract. The forward path computes task-specific fused states with scalar `alpha` and `beta`, optionally replacing those scalars with CKA-gated values, then mixes continuous and discrete branch outputs through convex interpolation (`src/models/thesis_multitask.py:1540-1634`). The auxiliary payload even records `fusion_mode: "learnable_sigmoid_scalars"` (`src/models/thesis_multitask.py:1606-1634`). This does not match the newer requirement that forward-path fusion be `task_specific_concat_projection` and that CKA remain diagnostic-only.

The current two-view contrastive loss is a simple one-positive formulation on normal timesteps only. It keeps only tokens where `synthetic_anomaly_mask == 0`, forms a full similarity matrix between anchor and positive tokens, and uses `cross_entropy(logits, arange(...))` as the loss (`src/models/thesis_multitask.py:1669-1692`). There is no current support for metadata-defined multi-positive sets, same-source-timestep positives across overlapping windows, or the richer batching contract discussed in the design notes.

The core training loop is still a single shared multitask step. `_shared_step` optionally builds one clean/augmented pair for contrastive learning, computes one multitask forward pass, and then assembles reconstruction loss, classification loss, optional regularizers, and contrastive loss into one total objective (`src/models/thesis_multitask.py:2608-2659`). There is no existing runtime separation for:

- Stage 1 classification-only encoder training,
- Stage 1 reconstruction-only encoder training,
- Stage 2 zipping,
- Stage 2 recovery without prototypes,
- Stage 3 frozen-memory warm-up,
- or a later main multitask run with frozen memories.

At the engine level, the trainer already knows how to run epoch loops, call a model-owned `training_step`, and trigger a memory-initialization hook, but it still assumes one model instance and one training loop (`src/engine/trainer.py:560-610`). This means the current engine can be reused, but the three-stage procedure does not exist yet as an orchestration layer.

### Evaluation

The evaluation path is overlap-aware on the test timeline. `Evaluator` accumulates each window’s `point_scores` back into the original entity sequence by `start_index:end_index`, averages overlapping contributions, and preserves point labels on the original timeline (`src/engine/evaluator.py:44-128`). This already matches the requirement that test-time metrics be interpreted on windows cut from the test sequence and then mapped back to the test timeline.

The current training-time “realistic validation” is not true test evaluation. During training, the trainer computes an anomaly rate from SMD test windows (`src/engine/trainer.py:540-558`), but then it still runs `realistic_validation_step` on `val_loader`, not on the true `test` loader (`src/engine/trainer.py:678-718`). So the current training loop does not checkpoint directly on real `test` windows. It only uses the test split indirectly to estimate an anomaly probability for synthetic corruption. That is a major mismatch relative to the new request, which explicitly wants train and test behavior tied to windows from the real test sequence.

The standalone evaluation script does evaluate on the true `test` loader. It rebuilds the dataset bundle, loads the checkpoint, and passes `data_bundle["loaders"]["test"]` into the evaluator (`scripts/evaluate.py:79-138`). So the repository already has a post-training test entrypoint; what is missing is end-to-end automation that couples the right experiment config, the right target entity, and the server launch flow.

One cautionary implementation detail is that `scripts/evaluate.py` rebuilds and scales the dataset bundle before loading the saved scaler state from the checkpoint (`scripts/evaluate.py:96-113`). In the current deterministic SMD path this is likely redundant rather than immediately wrong, because the scaler is recomputed from the same train split. But architecturally, the loaded scaler is not actually applied back onto the already-built `data_bundle`.

## Implementation-Critical Mismatches

The following mismatches are the concrete blockers between the current codebase and the requested `SMD 3-4` server experiment:

1. The active thesis exp2 config still points to `machine-2-1`, not `machine-3-4` (`configs/experiment/thesis/exp2/...:7-14`, `configs/data/smd_rtx3090_machine_2_1_20.yaml:1-10`).
2. The current window-20 single-entity data config uses `stride: 20`, so it is non-overlapping (`configs/data/smd_rtx3090_machine_2_1_20.yaml:5-6`). This is inconsistent with overlap-aware contrastive semantics discussed in the detail note.
3. The current model still uses Gumbel-Softmax discrete assignment and EMA-updated codebooks (`src/models/thesis_multitask.py:1360-1424`, `src/models/thesis_multitask.py:1485-1538`), not frozen `cosine_topk` querying.
4. The current fusion path is scalar interpolation with optional CKA-gated forward routing (`src/models/thesis_multitask.py:1540-1634`), not `task_specific_concat_projection` with diagnostic-only CKA.
5. The current contrastive loss is one-positive-only (`src/models/thesis_multitask.py:1669-1692`), not the multi-positive overlap-aware contract from the June discussion.
6. The current training loop is one shared multitask loop (`src/models/thesis_multitask.py:2608-2659`, `src/engine/trainer.py:560-610`), not the requested three-stage offline pre-training pipeline.
7. The current realistic validation path still evaluates on the validation loader, not on true test windows (`src/engine/trainer.py:678-718`).
8. The current launcher is local subprocess-based (`scripts/run_multiseed_experiments.py:125-222`) and has no `tmux` session orchestration, no server-oriented resume protocol, and no experiment-specific shell wrapper for an RTX 3090 machine.

## Code References

- `src/data/datasets/smd.py:14` - SMD parser with entity filtering and raw split construction.
- `src/data/datasets/smd.py:132` - Raw train tail-split into train/val; full test sequence preserved.
- `src/data/scalers.py:16` - Train-only feature normalization on full sequences.
- `src/data/loaders.py:150` - Scaler fit on train sequences before window creation.
- `src/data/loaders.py:177` - WindowDataset index-based slicing with `start_index` / `end_index`.
- `src/engine/evaluator.py:44` - Overlap-aware test-time score aggregation back to entity timelines.
- `src/engine/trainer.py:550` - Realistic-validation anomaly-rate lookup from SMD test windows.
- `src/engine/trainer.py:697` - Realistic validation still runs on `val_loader`.
- `src/models/thesis_multitask.py:703` - Learned discrete assignment layer and codebook buffers.
- `src/models/thesis_multitask.py:1201` - Generic memory initialization from train-loader hidden tokens.
- `src/models/thesis_multitask.py:1360` - EMA-updated discrete codebook logic.
- `src/models/thesis_multitask.py:1540` - Scalar and optional CKA-gated fusion path.
- `src/models/thesis_multitask.py:1669` - Current one-positive two-view contrastive loss.
- `src/models/thesis_multitask.py:2608` - One-loop multitask stage assembly.
- `scripts/run_multiseed_experiments.py:125` - Local subprocess launcher without `tmux`.
- `scripts/evaluate.py:96` - Standalone evaluation on the true test loader.
- `configs/data/smd_rtx3090_machine_2_1_20.yaml:1` - Current single-entity window-20 data config.
- `configs/model/thesis_multitask_redlamp_multiclass.yaml:1` - Current exp2 model-surface config.
- `configs/experiment/thesis/exp2/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20-exp2__w20__seed11__default.yaml:7` - Current active thesis exp2 experiment wiring.

## Historical Context (from documents/)

The current code still aligns most closely with `documents/design/offline_pretraining_phase_two_view_contrastive_design.md`, which documents the older “two-view contrastive + CKA-gated fusion” experiment family. However, the latest implementation contract now lives in `documents/logs/06-17-2026/detail/detail-offline-pretraining-three-stage-discussion-context.md`, especially its June 22 lock section, where the first implementation was finalized as:

- three-stage preparation before the final multitask run,
- frozen memories after initialization,
- `cosine_topk` discrete querying,
- `task_specific_concat_projection` as the forward fusion mechanism,
- and train-only statistics for all synthetic generation and memory initialization.

So the current codebase should be treated as the old experimental branch that must now be brought into alignment with the newer first-implementation note.

## Open Questions

1. Should the first runnable implementation checkpoint on a validation split derived from train as today, then run a separate final test pass on real test windows, or should the orchestration explicitly include a dedicated post-train test stage as a first-class required step?
2. For the first `machine-3-4` run, should overlap be `stride=1` to honor the richer contrastive metadata semantics immediately, or should the first automation pass preserve the smallest runtime surface first and add overlap-aware contrastive batching as the next increment?
3. The design note fixes the target method, but the repo still has many tests that lock the older exp2 behavior. Those tests will need to be updated or replaced once the new three-stage pipeline becomes the active contract.
