---
date: 2026-06-23 12:19:48 +0700
researcher: TheMetaSetter
git_commit: be3ef38b1ef8ad8677991e9fbd25bd2414c86d7a
branch: dev
repository: bachelor-thesis-2026
topic: "Preliminary implementation plan for semantic correction of the three-stage offline pre-training pipeline"
tags: [plan, offline-pretraining, three-stage, smd, thesis_multitask]
status: complete
last_updated: 2026-06-23
last_updated_by: TheMetaSetter
---

# Plan: Preliminary implementation plan for semantic correction of the three-stage offline pre-training pipeline

## Planning Goal

The goal of this plan is to define a safe programming direction for the current SMD `machine-3-4`, `window_size=20`, `stride=1` offline pre-training pipeline so that future implementation work aligns with the finalized three-stage wording, preserves the exact `300`-epoch budget, and avoids further semantic drift between documentation and code.

This is a preliminary implementation plan. It is intentionally focused on the minimum corrective path required before further training runs on the RTX 3090 server are treated as thesis-facing evidence.

## Current State

- The repository already contains a runnable orchestration path for the target experiment in `scripts/run_three_stage_offline_pretraining.py`.
- The data pipeline for SMD `machine-3-4` is already concrete and usable through `src/data/datasets/smd.py`, `src/data/loaders.py`, and `configs/data/smd_rtx3090_machine_3_4_20_stride1.yaml`.
- The model contract is already centralized in `src/models/thesis_multitask.py`, which is desirable because it respects the repository rule that one model remains self-contained in one file.
- The experiment config already enforces the exact budget `50 + 70 + 20 + 20 + 140 = 300` in `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml`.
- Continuous memory, discrete memory, `cosine_topk` discrete querying, and `task_specific_concat_projection` fusion are already implemented and active in the current model path.
- The main misalignment is semantic: the repository currently realizes the design as five execution phases, while the design note now treats Stage 3 as one unified semantic stage, namely `Memory Initialization and Fusion Warm-Up`.
- A second major misalignment is algorithmic: the current Stage 2 path is not genuine Multi-Task Zipping, but an approximation by matching-parameter averaging.

## Established Contracts That Must Be Preserved

### Batch Contract

The active data path already exposes batch dictionaries with:

- `x`
- `point_labels`
- `mask`
- `timestamps`
- `meta`

This contract should remain unchanged. It is already stable and is consumed consistently by the trainer and the multitask model.

### Encoder Contract

The thesis-facing encoder output contract remains:

`hidden: Tensor[B, L, H]`

with the current implementation provided by `MultitaskWindowEncoder` inside `src/models/thesis_multitask.py`. Any Stage 2 correction must preserve this public contract. Stage 2 may change parameter initialization semantics, but must not change the external encoder output shape or key naming.

### Model Output Contract

The current model output contract already supports:

- reconstruction output,
- classification logits,
- auxiliary branch and fusion diagnostics,
- phase-aware loss assembly.

Future corrections should preserve this external contract. The goal is to change semantic correctness, not to destabilize the trainer or evaluator interface.

## Existing Modules That Should Be Preserved

The following modules already provide correct or mostly correct infrastructure and should be preserved rather than refactored broadly:

- `src/data/datasets/smd.py`
- `src/data/loaders.py`
- `src/data/window.py`
- `scripts/train.py`
- `src/engine/trainer.py`
- `src/models/thesis_multitask.py`
- `configs/data/smd_rtx3090_machine_3_4_20_stride1.yaml`
- `configs/model/thesis_multitask_three_stage_window20.yaml`
- `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml`

The repository already has the right structural ownership boundaries. The next work should therefore be correction-oriented rather than architecture-oriented.

## Design Options

### Option A: Documentation-Only Clarification While Preserving the Current Runtime

Under this option, the repository would keep the current five execution phases exactly as they are, and only the wording in notes, logs, and launcher output would be clarified so that users understand the five phases as an implementation detail of the conceptual three-stage method.

This is the lowest-risk option for code churn, but it is not the best scientific option. It leaves the current Stage 2 approximation and the Stage 3 trainability mismatch intact. It therefore reduces wording confusion but does not actually correct the experimental semantics.

### Option B: Partial Semantic Correction on the Existing Runtime Skeleton

Under this option, the current data pipeline, trainer, and orchestration skeleton would be preserved, but the implementation would be tightened so that:

- Stage 3 is represented consistently as a single semantic stage with two substeps,
- Stage 3 trainability is restricted exactly to task heads and task-specific concat-projection fusion layers,
- memory initialization is clearly separated from warm-up in code comments, reporting, and runtime state,
- Stage 2 remains approximate for the moment, but the approximation is made explicit everywhere.

This option is pragmatic and low-risk, but it still leaves the most important algorithmic gap unresolved, namely the absence of true Multi-Task Zipping.

### Option C: Full Semantic Correction Before Further Thesis-Facing Runs

Under this option, the next implementation cycle would correct both major semantic gaps:

1. replace the current Stage 2 approximation with a true first-pass implementation of the finalized zipping interpretation;
2. tighten Stage 3 so that it exactly follows the wording:
   - initialize continuous memory from normal-only recovered training features;
   - initialize discrete memory from class-stratified recovered training features;
   - freeze both memory banks after initialization;
   - freeze the recovered zipped encoder during the short warm-up;
   - train only the task heads and task-specific concat-projection fusion layers.

This is the most rigorous option and the one most aligned with the finalized design note. It also creates the cleanest basis for later server runs, because training evidence would then be attached to a pipeline whose semantics match the thesis wording rather than an acknowledged approximation.

## Recommended Approach

The recommended approach is **Option C**.

The reason is straightforward. The current repository is already beyond the stage of needing broad infrastructure work. The main remaining issue is correctness of experimental semantics. A partial fix would reduce wording confusion, but would still leave the code executing a materially different Stage 2 and a still-imprecise Stage 3 warm-up contract. Since the user has already identified terminology drift as a serious failure mode, the next implementation cycle should prioritize semantic correction rather than incremental convenience.

However, Option C should still be executed conservatively. The correction should be staged in this order:

1. correct Stage 3 semantics first because they are now explicitly finalized;
2. then correct Stage 2 semantics in a minimal, thesis-facing first implementation;
3. only after both are stable, rerun the three-stage orchestration and server launch path.

## Preliminary Programming Scope

### Scope Included in the Next Implementation Cycle

- Correct Stage 3 semantic ownership in runtime naming and control flow.
- Restrict Stage 3 trainable parameters exactly to task heads and task-specific concat-projection fusion layers.
- Ensure memory initialization uses recovered training features in a way consistent with the finalized wording.
- Replace Stage 2 parameter-averaging initialization with a first real zipping implementation that preserves the existing encoder contract.
- Update tests so they validate semantic behavior rather than only phase-name existence.
- Update launcher, preflight, and execution-report wording so user-visible artifacts stop reinforcing the old ambiguity.

### Scope Excluded from the Next Implementation Cycle

- Broad refactoring of the SMD loader stack.
- Changes to the batch contract, evaluator contract, or checkpoint format unless strictly required by semantic correction.
- Online adaptation work.
- New datasets.
- New architecture families beyond the current `cnn_simple` target run.

## File-Level Implementation Map

### Primary Files to Modify

- `src/models/thesis_multitask.py`
  - This file owns the phase semantics, memory initialization rules, trainability control, and the fusion path. It is the main correction surface for Stage 3.

- `scripts/run_three_stage_offline_pretraining.py`
  - This file owns the runtime expansion of the experiment into phases and the Stage 2 initialization path. It is the main correction surface for Stage 2 semantics and Stage 3 naming/reporting semantics.

- `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml`
  - This file owns the visible stage schedule and the exact `300`-epoch accounting. It may require wording cleanup or additional explicit fields, but its total budget must remain unchanged.

- `configs/model/thesis_multitask_three_stage_window20.yaml`
  - This file owns runtime semantic toggles. It may need new explicit switches if Stage 3 initialization and warm-up control need to be separated more clearly without increasing ambiguous codepaths.

- `src/engine/trainer.py`
  - This file should be changed only if the Stage 3 initialization trigger or Stage 2 non-epoch semantics require a trainer-level distinction between parameter/statistical procedures and optimizer-training epochs.

### Secondary Files to Modify

- `scripts/preflight_three_stage_server.py`
- `scripts/launch_tmux_three_stage_experiment.sh`
- `scripts/verify_three_stage_run.py`

These files should be updated only so that their terminology, readiness checks, and verification summaries match the corrected semantics.

### Test Files to Modify or Add

- `tests/test_three_stage_phase_runtime.py`
- `tests/test_three_stage_orchestration_smoke.py`
- `tests/test_smd_machine_3_4_three_stage_config_loading.py`
- `tests/test_three_stage_run_verifier.py`

Potential additional focused tests:

- `tests/test_three_stage_stage2_zipping_runtime.py`
- `tests/test_three_stage_stage3_memory_warmup_contract.py`

## Planned Programming Sequence

### Phase 1: Correct Stage 3 Semantics First

The first implementation priority should be Stage 3 because its wording is now exact and locked. This phase should:

- make Stage 3 explicitly represent one semantic stage with two internal substeps,
- ensure both memory banks are initialized from recovered training features only,
- freeze both memory banks immediately after initialization,
- freeze the recovered zipped encoder during warm-up,
- restrict trainable modules during warm-up to task heads and task-specific concat-projection fusion layers only.

This work should remain mostly inside `src/models/thesis_multitask.py` and the reporting layer of `scripts/run_three_stage_offline_pretraining.py`.

### Phase 2: Replace Stage 2 Approximation With a First Real Zipping Path

The second implementation priority should be Stage 2. The current parameter-average initialization should be removed or downgraded to an explicitly named fallback path. The primary path should implement the first thesis-facing version of the intended zipping procedure while preserving:

- the current `cnn_simple` encoder interface,
- the current task-head initialization rule,
- the downstream checkpoint and trainer contracts.

The key design principle here is minimal semantic correction without introducing a new abstract subsystem. The zipping logic should be readable, local, and directly traceable from the design note to the orchestration code.

### Phase 3: Reconcile Epoch Accounting and Reporting

After Stage 2 and Stage 3 semantics are corrected, the orchestration layer should be reviewed to ensure that:

- the exact `300`-epoch budget is still preserved,
- non-training statistical procedures are not described as ordinary optimizer-training epochs if that wording is misleading,
- manifests, reports, preflight summaries, and launcher output use consistent terminology.

The plan does not currently require reducing the visible `300`-epoch training total. Instead, it requires that reporting and control flow stop suggesting a conceptually incorrect stage structure.

### Phase 4: Revalidate End-to-End Local Orchestration

Before any server run, the corrected pipeline should pass:

- config-loading tests,
- phase-semantics tests,
- orchestration smoke tests,
- run-verification tests.

Only then should the `tmux` launcher and server preflight path be used again for the main RTX 3090 experiment.

## Risk and Mitigation

### Risk: Stage 2 Correction Breaks Checkpoint Compatibility

If Stage 2 zipping changes parameter layout assumptions, checkpoint loading into the later phases may fail.

Mitigation:

- preserve the encoder module names and tensor shapes,
- keep task-head ownership unchanged,
- add dedicated checkpoint round-trip tests for Stage 2 output into Stage 3 input.

### Risk: Stage 3 Exact Freezing Contract Is Only Partially Enforced

It is possible to freeze the encoder and memories correctly while still leaving unrelated trainable modules enabled.

Mitigation:

- add explicit tests that enumerate which modules are trainable in Stage 3 warm-up,
- assert that only task heads and concat-projection layers retain `requires_grad=True`.

### Risk: Memory Initialization Still Uses an Incomplete Feature Pool

The current implementation samples the first `memory_initialization_batches` batches only, which may not match the intended reading of recovered training features.

Mitigation:

- make the sampling policy explicit,
- if exhaustive initialization is adopted, test it on the SMD machine `3-4` path specifically,
- keep the policy visible in config and in logs.

### Risk: Further Terminology Drift Reappears in User-Facing Artifacts

If launcher output, preflight summaries, and notes keep old names, future confusion will return even after code semantics are improved.

Mitigation:

- update artifact wording in the same implementation cycle,
- treat terminology consistency as a testable acceptance criterion for the corrected path.

### Risk: Broad Refactoring Delays the Main Experiment

A large cleanup could consume time without improving thesis-facing correctness.

Mitigation:

- preserve the current data and trainer skeleton,
- keep edits local to the Stage 2 and Stage 3 ownership surfaces,
- avoid unrelated abstraction work.

## Validation and Test Plan

The next implementation cycle should validate at four levels.

### Unit-Level Validation

- Stage 3 trainable-parameter mask is exact.
- Stage 3 memory state transitions are exact.
- Stage 2 zipping output preserves expected tensor shapes and state-dict compatibility.
- Memory initialization uses the intended class and normal-token selection rules.

### Integration-Level Validation

- Config loading succeeds for the main RTX 3090 config and the smoke config.
- Orchestrator manifest generation reflects the corrected semantics.
- Stage-to-stage checkpoint handoff remains valid.
- Run verification still reports success on complete smoke runs.

### Smoke Runtime Validation

- The smoke three-stage experiment still executes end-to-end locally.
- The corrected Stage 3 behavior is visible in logs and checkpoint extra state.

### Server Launch Validation

- `scripts/preflight_three_stage_server.py` still recognizes the main config as launch-ready when the target machine has `tmux` and an RTX 3090.
- `scripts/launch_tmux_three_stage_experiment.sh` still produces a one-command launch path for the corrected orchestration.

## Preliminary Acceptance Criteria

The pipeline should be considered ready for the next implementation phase only when all of the following are true:

1. Stage 3 is represented and documented as `Memory Initialization and Fusion Warm-Up` with exact behavior matching the finalized wording.
2. The current Stage 2 parameter-average approximation is either replaced by the intended first zipping implementation or explicitly demoted to a clearly named fallback path.
3. The exact `300`-epoch budget is still preserved.
4. The SMD `machine-3-4`, `window_size=20`, `stride=1` path still loads, trains in smoke mode, and verifies successfully.
5. User-facing orchestration artifacts no longer reinforce the earlier Stage 3 ambiguity.

## Open Questions

1. Should the corrected Stage 3 memory initialization use the full recovered training feature population, or is a configurable capped-batch approximation still acceptable for the first implementation?
2. Should the repository retain the visible five-phase runtime as an implementation detail after semantic correction, or should reporting be collapsed more aggressively around conceptual Stage 1, Stage 2, and Stage 3 naming?
3. Does the first Stage 2 correction need to implement full channel sharing only, as previously preferred in the discussion note, or does it need a broader configurable sharing policy in the very first pass?

## Recommended Next Action

The next action should be a detail-phase note or implementation checklist focused only on **Stage 3 exact contract correction first**, because that is the most finalized and least ambiguous corrective surface. Once Stage 3 is locked in code and tests, the Stage 2 correction can be implemented with less risk of reintroducing terminology drift.
