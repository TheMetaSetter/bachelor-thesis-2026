---
date: 2026-06-22 16:17:09 +0700
planner: Codex
git_commit: d310f2c3b3f36cd260870504ac6811c2720c9952
branch: dev
repository: bachelor-thesis-2026
topic: "Implementation plan for automated SMD 3-4 offline pre-training with the finalized three-stage method on an RTX 3090 server"
tags: [plan, offline-pretraining, smd, smd-3-4, three-stage, tmux, server-automation]
status: complete
last_updated: 2026-06-22
last_updated_by: Codex
source_research_note: documents/logs/06-22-2026/research/research-smd-3-4-offline-pretraining-current-state.md
---

# Plan: Automated SMD 3-4 Offline Pre-Training with the Finalized Three-Stage Method

**Date**: 2026-06-22 16:17:09 +0700  
**Planner**: Codex  
**Git Commit**: `d310f2c3b3f36cd260870504ac6811c2720c9952`  
**Branch**: `dev`

## Plan Objective

This plan specifies the implementation work required to make the repository automatically run the finalized offline pre-training method described in `documents/logs/06-17-2026/detail/detail-offline-pretraining-three-stage-discussion-context.md` for SMD `machine-3-4`, on a GPU server with one RTX 3090, using `tmux` for resilient execution. The target end state is not the legacy `Exp2` method. The target end state is the newer first implementation with:

- careful SMD `machine-3-4` data preparation,
- overlapping windows with `window_size = 20` and `stride = 1`,
- stage-separated offline pre-training,
- frozen memories after initialization,
- `cosine_topk` discrete querying,
- `task_specific_concat_projection` forward-path fusion,
- and explicit train followed by true test evaluation on windows cut from the test sequence.

## Current State (Grounded)

1. The current SMD pipeline already parses raw sequences, filters entities, splits train into train/val, fits one scaler on train sequences only, transforms full sequences, and then windows them (`src/data/datasets/smd.py`, `src/data/scalers.py`, `src/data/loaders.py`).
2. The evaluator already reconstructs overlap-aware pointwise scores back onto full entity timelines, which is compatible with `stride = 1` test evaluation (`src/engine/evaluator.py`).
3. The current window-20 thesis configuration is still hard-wired to `machine-2-1` with `stride = 20`, so it does not match the new target experiment (`configs/data/smd_rtx3090_machine_2_1_20.yaml`).
4. The current `thesis_multitask` implementation still represents the older single-loop prototype-fusion method:
   - learned discrete assignment,
   - Gumbel-Softmax read path,
   - EMA-updated codebook,
   - adaptive continuous memory updates,
   - scalar fusion with optional CKA-gated forward routing,
   - and one shared multitask loop.
5. The current training engine can already run config-driven offline experiments and checkpoint them, but it does not yet orchestrate the new multi-stage offline procedure.
6. The current launcher is local subprocess-based and has no `tmux` server execution wrapper.

## Selected Approach

The implementation should follow one integrated approach that preserves the strongest parts of the existing repository while replacing the outdated offline method surface:

- **Keep the current SMD parser, scaler, window dataset, collate path, evaluator, and registry-driven train/evaluate scripts where they remain correct.**
- **Add a new orchestration layer for the finalized multi-stage offline procedure instead of trying to fake the new method inside one old Exp2 loop.**
- **Refactor `src/models/thesis_multitask.py` only where the old method semantics directly contradict the new contract.**
- **Add one server-facing launch surface that uses `tmux` and exact resolved configs, rather than inventing a separate ad hoc runtime.**

This approach is the best fit because it respects the user’s instruction to reuse code aggressively, avoids unnecessary loader rewrites, and still moves the codebase toward the actual target method rather than preserving the old Exp2 behavior.

## Design Options Considered

### Option A — Patch the old Exp2 path directly

This option would keep the current `Exp2` runtime mostly intact, only switch to `machine-3-4`, add `tmux`, and minimally tweak some flags.

This option is rejected because it would leave the core method wrong. The current code still implements old discrete-memory, old fusion, and old training-loop semantics.

### Option B — Reuse the loader and engine foundation, but add a new three-stage orchestration path

This option keeps the existing SMD data path and evaluator, introduces new configuration and orchestration for Stage 1A, Stage 1B, Stage 2 recovery, Stage 3 warm-up, and main multitask training, and updates `thesis_multitask.py` to the new forward-path semantics.

This is the recommended option because it maximizes reuse while still aligning to the actual method.

### Option C — Build an almost entirely separate implementation stack

This option would create many new files and isolate the three-stage method completely from the current multitask model and scripts.

This option is rejected for the first implementation because it introduces too much duplication, broadens the edit surface, and conflicts with the repository’s readability-first constraint unless there is strong evidence that reuse is impossible.

## File Structure and Responsibilities

The following file-level decomposition should be used.

### Files to create

1. `configs/data/smd_rtx3090_machine_3_4_20_stride1.yaml`
- SMD `machine-3-4` data config for the finalized run.
- Owns `entity_ids`, `window_size`, `stride`, batch size, and worker choices for this experiment family.

2. `configs/model/thesis_multitask_three_stage_window20.yaml`
- Model-surface config for the finalized first implementation.
- Owns architecture, frozen-memory settings, discrete query mode, fusion mode, and stage-specific toggles.

3. `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml`
- Top-level resolved experiment family for the final three-stage run on `machine-3-4`.

4. `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml`
- Smoke config for one small verification run before long server execution.

5. `scripts/run_three_stage_offline_pretraining.py`
- New orchestration entrypoint for the finalized three-stage method.
- Owns stage sequencing, checkpoint handoff, and final true-test evaluation call.

6. `scripts/launch_tmux_three_stage_experiment.sh`
- Server-facing wrapper that launches the experiment in `tmux`, sets clear log paths, and uses the resolved experiment config.

7. `tests/test_smd_machine_3_4_three_stage_config_loading.py`
- Verifies new config files resolve and preserve required contracts.

8. `tests/test_smd_overlap_metadata_contract.py`
- Verifies `machine-3-4` windows and metadata required for same-source-timestep tracking.

9. `tests/test_three_stage_orchestration_smoke.py`
- Verifies the new orchestration layer can execute a minimal stage sequence without contract breakage.

### Files to modify

1. `src/data/loaders.py`
- Add the missing overlap-aware metadata needed by the finalized contrastive semantics.
- Preserve the current parse -> scale -> window order.

2. `src/data/datasets/smd.py`
- Keep core parser behavior.
- Add any metadata helpers needed for sequence identity and stricter per-entity audit logging.

3. `src/data/api.py`
- Update public helper defaults or add helper entrypoints so window-20 overlap usage is explicit and not accidentally inherited from old `window_size=100` defaults.

4. `src/core/config.py`
- Extend validation to accept new three-stage and server-launch config fields.
- Keep strict validation behavior.

5. `scripts/train.py`
- Keep existing registry-driven training intact, but allow it to be called as a sub-stage executor from the new three-stage orchestration script if that design path is selected.

6. `scripts/evaluate.py`
- Ensure it can be called cleanly from the new orchestration pipeline for final true-test evaluation.

7. `src/models/thesis_multitask.py`
- This is the main implementation file that must be refactored to align with the finalized method.
- Replace or gate the old method semantics that contradict the June 22 contract.

8. `tests/test_one_multitask_train_step.py`
- Update or extend for the new stage-specific runtime contracts.

9. `tests/test_multitask_memory_updates.py`
- Replace old EMA-update assumptions with frozen-memory expectations where the new method becomes authoritative.

10. `tests/test_smd_dataset_shapes.py`
- Extend to cover `machine-3-4` filtering and overlap-specific metadata.

## Batch, Encoder, and Output Contracts

### Batch contract

The current batch contract must remain a dictionary consumed by the model:

- `x: Tensor[B, L, D]`
- `point_labels: Optional[Tensor[B, L]]`
- `mask: Optional[Tensor[B, L, D]]`
- `timestamps: Optional[Tensor[B, L]]`
- `meta: list[dict]`

For the finalized overlap-aware contrastive design, `meta` must be enriched so each window carries enough information to recover source identity without inventing a second batch schema. At minimum the implementation should expose:

- `dataset_name`
- `entity_id`
- `split`
- `start_index`
- `end_index`
- `window_size`
- `series_id` or a stable equivalent
- local timestep identity reconstruction support
- absolute source-timestep identity reconstruction support

The design goal is to keep the external batch type stable while making `meta` rich enough for same-source-timestep positives when they naturally appear in a batch.

### Encoder contract

Every stage-specific encoder path must still expose:

- `hidden: Tensor[B, L, H]`

This must remain true for:

- Stage 1 classification encoder,
- Stage 1 reconstruction encoder,
- zipped encoder after Stage 2,
- and the multitask model after Stage 3.

### Model output contract

The top-level output contract should stay stable for the engine:

- `hidden`
- `pooled`
- `recon`
- `logits`
- `point_scores`
- `window_scores`
- `aux`

Any new stage-specific diagnostics must stay inside `aux` and step logs rather than breaking `validate_model_outputs`.

## Implementation Plan

### Phase 1 — Lock the SMD 3-4 data path and overlap-aware metadata

#### Phase summary

This phase prepares the exact data path for the requested experiment without redesigning the loader foundation. The purpose is to make `machine-3-4` a first-class, audited, reproducible data target with `window_size = 20` and `stride = 1`.

#### File-level edits

1. Create `configs/data/smd_rtx3090_machine_3_4_20_stride1.yaml`.
2. Modify `src/data/loaders.py` so `WindowDataset` emits richer window metadata required by overlap-aware contrastive semantics.
3. Modify `src/data/datasets/smd.py` only if needed to expose a stable per-sequence identifier beyond `entity_id`.
4. Extend `tests/test_smd_dataset_shapes.py`.
5. Add `tests/test_smd_overlap_metadata_contract.py`.

#### Required behavior

1. The scaler must still fit on train sequences only, before any window slicing.
2. `machine-3-4` must be selected only through configuration, not through hard-coded script edits.
3. Window metadata must allow reconstruction of absolute source-timestep identity.
4. The test split must remain the real full test sequence from SMD.

#### Risk and mitigation

- **Risk**: over-refactoring the loader path.
  **Mitigation**: keep the current parser/scaler/window builder and only extend metadata.
- **Risk**: false-negative contrastive logic later because metadata is incomplete.
  **Mitigation**: add precise metadata tests in this phase before touching contrastive logic.

#### Acceptance criteria

1. `machine-3-4` config resolves and builds train/val/test windows.
2. `stride = 1` produces overlapping windows and preserves exact `start_index` / `end_index`.
3. Unit tests prove that metadata is sufficient for same-source-timestep bookkeeping.

### Phase 2 — Introduce stage-aware offline orchestration

#### Phase summary

This phase adds a real orchestration layer for the finalized three-stage procedure. The objective is to stop treating the new method as a single old-style multitask loop.

#### File-level edits

1. Create `scripts/run_three_stage_offline_pretraining.py`.
2. Reuse `scripts/train.py` and `scripts/evaluate.py` as subroutines where practical.
3. Extend `src/core/config.py` to validate three-stage experiment config fields.
4. Add `tests/test_three_stage_orchestration_smoke.py`.

#### Required behavior

The orchestration path must support:

1. Stage 1A classification pre-training.
2. Stage 1B reconstruction pre-training.
3. Stage 2 zipping.
4. Short Stage 2 recovery without prototypes.
5. Stage 3 memory initialization from zip-recovered encoder space.
6. Stage 3 prototype warm-up with encoder frozen.
7. Main multitask pre-training with frozen memories.
8. Final true-test evaluation on windows cut from the real test sequence.

#### Design pattern application

- **Composition over inheritance**: the orchestration script composes existing train/evaluate/model/config surfaces instead of subclassing the trainer.
- **Strategy pattern**: stage behavior is driven by explicit config and stage mode rather than hidden if-else sprawl across the engine.

#### Risk and mitigation

- **Risk**: trying to force all stages through one opaque monolithic train step.
  **Mitigation**: isolate stage transitions in the orchestration layer.
- **Risk**: stage artifacts become ambiguous.
  **Mitigation**: each stage must write explicit checkpoints and resolved configs.

#### Acceptance criteria

1. A smoke orchestration run can execute stage transitions end to end.
2. Each stage emits a named checkpoint or artifact required by the next stage.
3. Final evaluation runs on `data_bundle["loaders"]["test"]`, not on validation windows.

### Phase 3 — Refactor `thesis_multitask.py` from old Exp2 semantics to the finalized first implementation

#### Phase summary

This phase updates the model file that currently conflicts most strongly with the new method. The objective is not cosmetic cleanup. The objective is semantic replacement where the old implementation is wrong for the target method.

#### File-level edits

1. Modify `src/models/thesis_multitask.py`.
2. Create or update `configs/model/thesis_multitask_three_stage_window20.yaml`.
3. Extend `tests/test_one_multitask_train_step.py`.
4. Update `tests/test_thesis_multitask_config_refactor.py`.

#### Required changes

1. Remove the current method default that depends on learned `discrete_assignment` plus `Gumbel-Softmax` for the finalized run.
2. Introduce a discrete query mode that supports `cosine_topk` and freezes the discrete memory after initialization.
3. Replace forward-path scalar fusion as the primary method with `task_specific_concat_projection`.
4. Keep CKA as diagnostic-only for this first implementation.
5. Support stage-aware behavior:
   - prototype-free Stage 1 task-specific operation,
   - prototype-enabled warm-up and final multitask operation after Stage 3.

#### Risk and mitigation

- **Risk**: breaking old tests that encode obsolete Exp2 behavior.
  **Mitigation**: explicitly decide which tests remain legacy coverage and which must be rewritten because the method contract changed.
- **Risk**: over-fragmenting the model file.
  **Mitigation**: keep one-model-one-file readability and add focused helper sections only where necessary.

#### Acceptance criteria

1. The finalized model config can build a model with the new surface.
2. The forward path for the target experiment no longer depends on old scalar-fusion semantics as the main method.
3. The discrete query path for the finalized run is `cosine_topk`, not Gumbel-Softmax assignment.

### Phase 4 — Implement frozen-memory initialization and usage rules

#### Phase summary

This phase makes memory behavior align with the June 22 note. The core rule is that memories are initialized from train-derived latent tokens and then frozen for the main first implementation.

#### File-level edits

1. Modify `src/models/thesis_multitask.py`.
2. Extend `tests/test_multitask_memory_updates.py`.
3. Add targeted tests for covering selection and frozen read behavior.

#### Required behavior

1. Continuous memory initialization:
   - normal-only covering selection.
2. Discrete memory initialization:
   - class-stratified covering selection across 12 classes.
3. Both memory banks:
   - initialized from the zip-recovered encoder space,
   - train-derived only,
   - frozen after initialization in the first implementation.
4. No test-derived statistics or tokens may influence initialization.

#### Risk and mitigation

- **Risk**: accidental leakage from test-derived windows or labels.
  **Mitigation**: confine all initialization inputs to train-split artifacts and add explicit test assertions.
- **Risk**: keeping old EMA codepath active by accident.
  **Mitigation**: gate legacy behavior out of the finalized experiment config and tests.

#### Acceptance criteria

1. Memory initialization tests prove train-only sourcing.
2. Finalized run no longer updates memory during later train steps.
3. Covering-selection logic is tested for expected output shape and determinism under fixed seeds.

### Phase 5 — Implement the finalized contrastive semantics at a minimal correct scope

#### Phase summary

This phase introduces the contrastive objective that is actually compatible with the finalized note. The goal is not to overbuild batching machinery. The goal is to make the minimal correct overlap-aware version.

#### File-level edits

1. Modify `src/models/thesis_multitask.py`.
2. Potentially add a small helper in the same file for contrastive-positive construction.
3. Add tests covering overlap metadata usage.

#### Required behavior

1. Use `stride = 1` overlapping windows in the target experiment.
2. Use aligned-view positives.
3. Use same-source-timestep positives from other overlapping windows when they naturally appear in a batch.
4. Fall back cleanly when no overlap-positive is present.
5. Log overlap-positive availability explicitly.
6. Do not make custom overlap-aware batch sampling a prerequisite for the first implementation.

#### Risk and mitigation

- **Risk**: false negatives from naive one-positive InfoNCE.
  **Mitigation**: use metadata-aware positive-set construction and explicit filtering.
- **Risk**: overengineering custom batchers too early.
  **Mitigation**: keep natural-batch overlap support first; defer custom batching.

#### Acceptance criteria

1. The target run no longer uses the old one-positive-only contrastive helper as its authoritative implementation.
2. Tests verify that overlap positives can be recovered from metadata.
3. Logs expose whether overlap positives were available during a run.

### Phase 6 — Add server automation with `tmux` and exact experiment preflight

#### Phase summary

This phase makes the experiment runnable on the actual target environment: one RTX 3090 server with long-running execution protected by `tmux`.

#### File-level edits

1. Create `scripts/launch_tmux_three_stage_experiment.sh`.
2. Optionally extend `scripts/run_multiseed_experiments.py` only if reuse is clean; otherwise keep the new server launcher separate.
3. Add config or documentation notes for server command usage.

#### Required behavior

1. The launcher must:
   - accept one resolved experiment config,
   - create or reuse a named `tmux` session,
   - write stdout/stderr to a stable log file,
   - run the three-stage orchestration script,
   - and print the exact attach command.
2. Preflight checks must fail fast when:
   - the config path is invalid,
   - dataset root is missing,
   - required output paths collide,
   - or the final stage artifacts are misconfigured.

#### Risk and mitigation

- **Risk**: server runs die on disconnect.
  **Mitigation**: enforce `tmux` launch surface rather than raw shell commands.
- **Risk**: silent config drift on the server.
  **Mitigation**: save resolved config and stage manifests with each run.

#### Acceptance criteria

1. A dry-run server launch prints the exact `tmux` command and resolved paths.
2. A smoke launch can start inside `tmux` without path errors.
3. Logs and attach instructions are deterministic and user-readable.

### Phase 7 — Final true-test evaluation and reproducibility audit

#### Phase summary

This phase ensures that the requested end state is actually satisfied: after training, the system must test on windows cut from the real SMD test sequence and preserve enough artifacts to reproduce the run.

#### File-level edits

1. Modify `scripts/evaluate.py` only where needed for smoother orchestration.
2. Ensure the three-stage orchestration script calls the true test evaluation path.
3. Add or extend tests around final evaluation artifacts.

#### Required behavior

1. Final evaluation must use the true `test` loader.
2. Output artifacts must include:
   - metrics,
   - pointwise records,
   - curves,
   - resolved config,
   - and stage checkpoints or manifests.
3. The saved scaler and resolved config must be enough to reproduce the same train/test preprocessing path.

#### Risk and mitigation

- **Risk**: claiming test-time behavior while still checkpointing and validating only on `val_loader`.
  **Mitigation**: make post-train test evaluation an explicit orchestration stage, not an optional side effect.
- **Risk**: artifact incompleteness.
  **Mitigation**: require resolved config and evaluation files as acceptance artifacts.

#### Acceptance criteria

1. The orchestration pipeline ends with a real test evaluation run.
2. Evaluation artifacts are written under the experiment output directory.
3. The run can be audited afterward from config, checkpoints, and evaluation outputs alone.

## Test Plan and Validation Procedures

### Unit tests

1. `tests/test_smd_dataset_shapes.py`
- add `machine-3-4` filtering coverage.

2. `tests/test_smd_overlap_metadata_contract.py`
- verify overlap metadata and absolute timestep reconstruction support.

3. `tests/test_multitask_memory_updates.py`
- replace legacy EMA assumptions with finalized frozen-memory expectations for the new config surface.

4. `tests/test_one_multitask_train_step.py`
- add one-stage smoke cases for the finalized model path.

5. `tests/test_smd_machine_3_4_three_stage_config_loading.py`
- verify new configs resolve and respect validation rules.

### Integration tests

1. Three-stage smoke orchestration test:
- run a minimal stage sequence with small caps or smoke windows.

2. Final-evaluation smoke test:
- ensure test loader evaluation happens after stage execution.

3. Server-launch dry-run test:
- verify generated `tmux` command and log-path behavior.

### Manual validation commands

The final implementation should support the following kinds of validation:

1. Config-only resolution for the new `machine-3-4` smoke config.
2. One small smoke run of the three-stage orchestration on local CPU or minimal GPU settings.
3. One `tmux`-wrapped dry run on the server-facing script.
4. One real final evaluation call on the test loader using the produced checkpoint.

## Risk and Mitigation Summary

1. **Risk: preserving the wrong Exp2 behavior because tests currently encode it.**  
   **Mitigation:** explicitly migrate target tests to the June 22 contract instead of treating old behavior as mandatory truth.

2. **Risk: data leakage from test-derived statistics.**  
   **Mitigation:** keep scaler fit, synthetic generation, and memory initialization confined to train-derived artifacts only.

3. **Risk: overlap-aware contrastive logic fails because metadata is incomplete.**  
   **Mitigation:** complete metadata enrichment before contrastive refactor.

4. **Risk: fusion collapses or the wrong fusion path remains active.**  
   **Mitigation:** make `task_specific_concat_projection` explicit in config and logs; keep CKA diagnostic-only.

5. **Risk: server automation is brittle.**  
   **Mitigation:** use a dedicated `tmux` launcher with preflight validation and deterministic log paths.

## Measurable Completion Criteria

The work should only be considered complete when all of the following are true:

1. SMD `machine-3-4` has a dedicated `window_size = 20`, `stride = 1` config and passes config validation.
2. The repository can run the finalized three-stage offline pre-training pipeline, not the old Exp2 approximation.
3. The finalized model path uses frozen memories, `cosine_topk`, and `task_specific_concat_projection` for the target experiment.
4. The orchestration path performs final evaluation on windows cut from the real test sequence.
5. A server-facing `tmux` launch surface exists and can start the experiment reproducibly on one RTX 3090 machine.
6. Tests and smoke checks prove the data path, stage orchestration, and final evaluation contracts.

## Recommended Next Step

The next execution step should be to write the detail artifact for this plan, with exact edit order, file-by-file contracts, and verification commands, before touching implementation code. After that, implementation should proceed in the same order as the phases above, starting with the SMD `machine-3-4` data path and metadata contract.
