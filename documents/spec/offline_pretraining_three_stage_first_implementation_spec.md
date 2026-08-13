---
title: "Offline Pre-Training Three-Stage First Implementation Spec"
date: 2026-06-22
status: approved
owners:
  - TheMetaSetter
  - Codex
tags:
  - design
  - offline-pretraining
  - smd
  - machine-3-4
  - three-stage
---

# Offline Pre-Training Three-Stage First Implementation Spec

> **Notation authority:** Khi đối chiếu anomaly score mức điểm, tài liệu lịch sử này dùng mapping trong [Thiết kế anomaly score mức điểm và bộ ký hiệu chuẩn](anomaly-score-designs-and-notation.md). Tên runtime và ngữ nghĩa lịch sử trong thân tài liệu được giữ nguyên.


## Purpose

This spec defines the first real implementation of the offline pre-training
pipeline described in:

- `documents/logs/06-17-2026/detail/detail-offline-pretraining-three-stage-discussion-context.md`

The immediate target is a reproducible experiment on SMD `machine-3-4` with:

- `window_size = 20`,
- `stride = 1`,
- careful train-only preprocessing and memory initialization,
- automatic server execution through `tmux`,
- final evaluation on windows cut from the test sequence and aggregated back to
  the test timeline by the existing evaluator path.

This spec is intentionally implementation-facing. It states what the code must
do now, what existing code can be reused, and what parts of the current repo
must be replaced or extended because they still reflect the older Exp2 design.

## Current Codebase Truth

The existing SMD data path is already mostly correct for the new experiment and
should be preserved unless a narrow contract gap blocks the new method:

1. `src/data/datasets/smd.py`
   - loads full train and test sequences per machine,
   - splits train tail into validation,
   - keeps test labels aligned with the full test sequence.

2. `src/data/loaders.py`
   - fits `SequenceStandardScaler` on the training split only,
   - transforms train, val, and test sequences after fitting,
   - then slices scaled sequences into windows.

3. `src/engine/evaluator.py`
   - already supports overlap-aware aggregation from window scores back to the
     full test timeline.

So the primary data risk is not the coarse preprocessing order. The primary
risk is that the current window metadata contract is too weak for the finalized
contrastive semantics.

The current model/runtime truth is different:

1. `src/models/thesis_multitask.py`
   - still contains legacy Exp2 semantics such as:
     - Gumbel-softmax discrete assignment,
     - EMA-updated memories,
     - forward-path CKA-gated fusion,
     - older two-view contrastive assumptions.

2. `configs/experiment/thesis/exp2/...`
   - still points to `machine-2-1`,
   - still uses `stride: 20`,
   - still encodes the older model behavior rather than the finalized
     three-stage contract.

3. `scripts/run_multiseed_experiments.py`
   - is a local subprocess launcher,
   - does not encode three-stage orchestration,
   - does not launch detached `tmux` sessions for server runs.

Therefore the implementation should preserve the current SMD loader foundation,
but must replace or extend the experiment orchestration and thesis model
semantics.

## Scope

This first implementation includes:

1. exact SMD `machine-3-4` data preparation with `window_size = 20` and
   `stride = 1`,
2. Stage 1 separate task-specific training,
3. Stage 2 encoder zipping and optional short recovery stage,
4. Stage 3 memory initialization and fusion warm-up,
5. main multitask offline pre-training with frozen memories,
6. final checkpoint evaluation on test windows,
7. server automation through `tmux`,
8. smoke tests and contract tests covering the new data/orchestration path.

This first implementation does **not** include:

1. custom overlap-aware sampler construction,
2. Jacobian-based ERF estimation during training,
3. KMeans prototype initialization,
4. EMA-updated memory banks,
5. forward-path CKA gating.

## Data Contract

### Non-Negotiable Data Rules

1. All normalization statistics must come from the training split only.
2. All memory initialization pools must come from the training split only.
3. All synthetic anomaly generation must be derived from the training split
   only.
4. No test labels, no test windows, and no test-derived statistics may enter
   Stage 1, Stage 2, Stage 3 initialization, or the main training loop.
5. Test windows are for final evaluation only.

### Required Window Metadata Extension

The current `WindowDataset` metadata includes:

- `dataset_name`,
- `entity_id`,
- `split`,
- `start_index`,
- `end_index`,
- `window_size`.

That is insufficient for the finalized overlap-aware contrastive rule. The
window contract must be extended so a token can be traced back to its original
absolute timestep in the source sequence.

The minimum first-pass metadata extension should expose enough information to
recover:

1. which source sequence the window came from,
2. what absolute source range the window covers,
3. and therefore what absolute timestep corresponds to local token position
   `t` inside the window.

The first implementation should prefer the smallest contract change that makes
this exact mapping deterministic. It should not refactor the whole loader stack.

## Stage Schedule

The implementation target follows the finalized three-stage contract.

### Stage 1: Separate Task-Specific Training

Train two separate models:

1. classification model:
   - shared encoder path for classification only,
   - loss = classification + contrastive,
   - no prototype branches,
   - no memory querying,
   - no multitask fusion.

2. reconstruction model:
   - shared encoder path for reconstruction only,
   - loss = reconstruction + contrastive,
   - no prototype branches,
   - no memory querying,
   - no multitask fusion.

### Stage 2: Zipping and Short Recovery

1. zip the two Stage 1 encoders into one shared encoder,
2. initialize multitask encoder from the zipped result,
3. initialize task-specific heads from their corresponding Stage 1 heads,
4. optionally run a short recovery stage without prototypes to stabilize the
   zipped encoder before memory initialization.

The first implementation should make this recovery stage explicit in the runner
even if its epoch budget is short.

### Stage 3: Memory Initialization and Fusion Warm-Up

1. initialize continuous memory from normal-only recovered training features,
2. initialize discrete memory from class-stratified recovered training
   features using synthetic anomaly labels generated from the training split,
3. freeze both memory banks after initialization,
4. freeze the recovered zipped encoder during the short warm-up,
5. train only the task heads and task-specific concat-projection fusion
   layers.

### Main Multitask Offline Pre-Training

Use:

- frozen memories,
- zipped-and-recovered encoder initialization,
- task-specific concat projection fusion,
- cosine top-k memory querying,
- low-compute RF-aware InfoNCE,
- multitask loss:
  - `L_rec + L_cls + 0.1 * L_RF-InfoNCE`.

## Model Changes Required

The current `src/models/thesis_multitask.py` still reflects the older active
repository design more than the finalized first implementation.

The following semantics must be changed:

1. remove dependence on Gumbel-softmax discrete assignment in the active path,
2. remove EMA-updated memory behavior from the active path,
3. replace forward fusion with `task_specific_concat_projection`,
4. keep CKA as diagnostic-only if retained at all,
5. support frozen memory banks after initialization,
6. support cosine top-k querying with normalized matrix multiplication and
   without `torch.cdist`,
7. support stage-aware behavior so Stage 1, Stage 2 recovery, Stage 3 warm-up,
   and main multitask training do not silently share the wrong forward/loss
   logic.

This is the largest code change in the implementation and the one most likely
to require surgical refactoring rather than small patches.

## Config Changes Required

New config files should be added rather than mutating the old Exp2 experiment
into something semantically different.

The first implementation should introduce dedicated configs for:

1. SMD `machine-3-4` with `window_size = 20`, `stride = 1`, and RTX 3090
   runtime assumptions,
2. the new thesis multitask model surface for the three-stage method,
3. one main experiment config,
4. one smoke experiment config,
5. stage-specific budgets or overrides needed by the orchestration script.

The old Exp2 config files should remain as historical experiments, not be
quietly repurposed.

## Orchestration Changes Required

The existing `scripts/train.py` and `scripts/evaluate.py` are still useful as
runtime building blocks, but they do not encode the new multi-stage pipeline.

The first implementation therefore needs a dedicated orchestration script that:

1. runs Stage 1 classification training,
2. runs Stage 1 reconstruction training,
3. zips the encoders,
4. optionally runs short recovery,
5. initializes frozen memories from recovered train features,
6. runs Stage 3 memory initialization and fusion warm-up,
7. runs main multitask offline pre-training,
8. runs final evaluation on the test windows,
9. writes stage artifacts and resolved configs clearly enough for later thesis
   analysis.

Separately, the repo needs a server launcher script that wraps the experiment in
`tmux` for detached GPU execution on the RTX 3090 server.

## Testing Requirements

Before claiming this work implemented, the code must have fresh verification for
at least these contracts:

1. SMD `machine-3-4` config loads successfully.
2. Train-only preprocessing remains true after the new orchestration path is
   added.
3. Window metadata exposes enough information for overlap-aware positive
   recovery.
4. The three-stage orchestration script passes a smoke run or dry-run preflight.
5. Final evaluation still consumes test windows through the existing evaluator
   path.

The tests should be small and realistic. The goal is to catch real contract
breaks in data flow and stage orchestration, not just mock-only unit behavior.

## Must-Fix List Before Server Execution

The implementation is not considered ready for the actual RTX 3090 run until
all of the following are true:

1. `machine-3-4` config exists and loads cleanly.
2. `stride = 1` is active for the target experiment.
3. window metadata can recover same-source timestep identity.
4. no training-stage data path can accidentally consume test-derived
   information.
5. legacy active-path Gumbel/EMA/CKA-forward semantics are removed or bypassed
   for the three-stage run.
6. the orchestration runner can execute the full stage sequence.
7. the `tmux` launcher can start the experiment with a single command on the
   server.
8. final evaluation writes metrics from the real test-window path.

## Implementation Strategy

The implementation should proceed in this order:

1. lock the new data contract with tests,
2. extend data metadata minimally,
3. add dedicated configs,
4. add the orchestration runner and `tmux` launcher,
5. then refactor the thesis model active path to match the finalized
   three-stage semantics,
6. then verify with smoke tests and config-load checks,
7. only after that prepare the real server command.

This ordering keeps the highest-risk thesis correctness issue first: the data
contract and stage semantics, not cosmetic runtime wrapping.

## Approval Gate

This spec is the approval gate for implementation.

No code implementation should start until this spec is accepted as the working
contract for the first pass.
