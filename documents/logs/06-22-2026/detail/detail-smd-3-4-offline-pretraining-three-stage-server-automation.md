---
date: 2026-06-22 16:17:09 +0700
author: Codex
git_commit: d310f2c3b3f36cd260870504ac6811c2720c9952
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed implementation plan for automated SMD 3-4 offline pre-training with the finalized three-stage method on an RTX 3090 server"
tags: [detail-plan, offline-pretraining, smd, smd-3-4, three-stage, tmux, server-automation]
status: complete
last_updated: 2026-06-22
last_updated_by: Codex
source_plan: documents/logs/06-22-2026/plan/plan-smd-3-4-offline-pretraining-three-stage-server-automation.md
source_research: documents/logs/06-22-2026/research/research-smd-3-4-offline-pretraining-current-state.md
---

# Detailed Plan: Automated SMD 3-4 Offline Pre-Training with the Finalized Three-Stage Method

## Objective

The objective is to implement a reproducible, server-runnable offline pre-training pipeline for SMD `machine-3-4` that follows the finalized three-stage method recorded in `documents/logs/06-17-2026/detail/detail-offline-pretraining-three-stage-discussion-context.md`. The implementation shall preserve the strongest existing repository contracts, especially the current SMD parsing, scaling, windowing, evaluation, and registry-driven training surfaces, while replacing only the outdated method semantics that contradict the finalized design.

The required end state is:

1. SMD `machine-3-4` is prepared through the existing loader pipeline with `window_size = 20` and `stride = 1`.
2. The batch metadata is rich enough for overlap-aware same-source-timestep bookkeeping.
3. The finalized multi-stage offline procedure is runnable end to end.
4. The target model path uses frozen memories, `cosine_topk`, and `task_specific_concat_projection`.
5. Final testing runs on windows cut from the real test sequence.
6. The experiment can be launched on an RTX 3090 server via `tmux`.

## Non-goals

1. Do not redesign the whole data layer.
2. Do not preserve the old Exp2 method as the main target path just because tests already encode it.
3. Do not introduce a parallel abstraction-heavy training framework when the current engine can be reused.
4. Do not make custom overlap-aware batch sampling a prerequisite for the first runnable implementation.

## Phase 1 — Establish the exact SMD 3-4 data contract

### Phase summary

This phase creates the exact data contract for the target experiment. The purpose is to ensure that data preparation for `machine-3-4` is correct before any model semantics are changed. Because the user emphasized that this step is extremely important, the implementation should bias toward visibility, auditability, and minimal loader refactoring.

### Files

- Create: `configs/data/smd_rtx3090_machine_3_4_20_stride1.yaml`
- Modify: `src/data/loaders.py`
- Modify: `src/data/datasets/smd.py`
- Modify: `src/data/api.py`
- Modify: `tests/test_smd_dataset_shapes.py`
- Create: `tests/test_smd_overlap_metadata_contract.py`

### Required edits

#### 1. Add the dedicated target data config

Create `configs/data/smd_rtx3090_machine_3_4_20_stride1.yaml` with:

- `dataset_name: smd`
- `root_dir: data/ServerMachineDataset`
- `entity_ids: [machine-3-4]`
- `window_size: 20`
- `stride: 1`
- an explicit batch size appropriate for RTX 3090 experiments
- explicit worker policy
- `validation_split_ratio: 0.2`
- `shuffle_train: true`

This file becomes the SSOT data config for the finalized target run.

#### 2. Enrich per-window metadata without replacing the batch schema

Modify `WindowDataset.__getitem__` in `src/data/loaders.py` so each window still returns the current keys, but `meta` becomes richer. The implementation should preserve:

- `dataset_name`
- `entity_id`
- `split`
- `start_index`
- `end_index`
- `window_size`

and add the minimum extra fields needed for overlap-aware contrastive bookkeeping, for example:

- `series_id`
- `sequence_length`
- `window_index`
- `absolute_timestep_offset_base`

The exact field names can be chosen to fit existing style, but they must let later code recover:

1. which long sequence the window came from,
2. which local timestep belongs to which absolute timestep,
3. and whether two tokens from different windows map to the same original timestep.

#### 3. Keep the parser stable, but add stricter sequence identity support if needed

`src/data/datasets/smd.py` should stay close to its current behavior. Only make edits if `entity_id` alone is not enough for unambiguous sequence identity under the finalized contrastive contract. If a stable `series_id` or equivalent is needed, compute it in the raw-sequence metadata there rather than inventing it later in model code.

#### 4. Make public data helpers less misleading for the window-20 thesis path

`src/data/api.py` still defaults to `window_size=100` and `stride=10`. Do not break generic helpers, but add or adjust public entrypoints so the thesis-target SMD path can be called explicitly without accidentally inheriting the old defaults.

### Acceptance criteria

1. The new `machine-3-4` config resolves through `load_experiment_config`.
2. Building the SMD dataset bundle for that config produces non-empty train/val/test windows.
3. Test windows are overlapping because `stride=1`.
4. Unit tests verify that metadata can reconstruct absolute source timestep identity.
5. The scaler still fits only on train sequences before window slicing.

### Validation commands

Run:

```bash
.venv/bin/python -m pytest -q tests/test_smd_dataset_shapes.py tests/test_smd_overlap_metadata_contract.py
```

## Phase 2 — Add configuration and orchestration surfaces for the finalized method

### Phase summary

This phase creates the runtime surfaces needed to express the finalized three-stage procedure. The repository already has a config-driven train/evaluate path, so the main job here is to add a new orchestration layer rather than rewrite the engine.

### Files

- Create: `configs/model/thesis_multitask_three_stage_window20.yaml`
- Create: `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml`
- Create: `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml`
- Modify: `src/core/config.py`
- Create: `scripts/run_three_stage_offline_pretraining.py`
- Create: `tests/test_smd_machine_3_4_three_stage_config_loading.py`
- Create: `tests/test_three_stage_orchestration_smoke.py`

### Required edits

#### 1. Define a dedicated finalized model config

Create `configs/model/thesis_multitask_three_stage_window20.yaml` so the target experiment does not reuse the old Exp2 model config. This file should encode the first-implementation contract, not the legacy one. It should include:

- encoder family and dimensions,
- `num_classes: 12`,
- frozen-memory behavior,
- discrete query mode set to `cosine_topk`,
- fusion mode set to `task_specific_concat_projection`,
- stage-related flags needed by the new orchestration path,
- any remaining diagnostic-only CKA switches,
- and explicit disabling or exclusion of old Exp2-only semantics when possible.

#### 2. Define one smoke config and one real RTX 3090 config

The smoke config should:

- reuse the `machine-3-4` data config,
- use smaller epoch counts or bounded windows,
- write to isolated output and checkpoint directories,
- and exist only to verify orchestration and contract correctness.

The real config should:

- use the same target data config,
- encode the exact 300-epoch budget split required by the June 22 note,
- and expose the stage schedule clearly in resolved configuration.

#### 3. Extend config validation strictly

Modify `src/core/config.py` so the new experiment files fail fast when:

- stage budgets are missing or inconsistent,
- discrete query mode is unsupported,
- fusion mode is unsupported,
- server-launch-specific paths or stage outputs are malformed,
- or the new configs silently fall back to old behavior.

#### 4. Add a new orchestration script instead of abusing the old single-loop entrypoint

Create `scripts/run_three_stage_offline_pretraining.py` as the owning script for the finalized multi-stage method. This script should orchestrate:

1. Stage 1A classification pre-training,
2. Stage 1B reconstruction pre-training,
3. Stage 2 zipping,
4. Stage 2 recovery,
5. Stage 3 memory initialization,
6. Stage 3 prototype warm-up,
7. main multitask pre-training,
8. final evaluation on the test loader.

This script may reuse `scripts/train.py` and `scripts/evaluate.py` internally, but it must own the stage sequencing explicitly.

### Acceptance criteria

1. New configs resolve without unknown-key errors.
2. Stage budget accounting can be validated from config alone.
3. The orchestration script supports at least a smoke run through all declared stage transitions.

### Validation commands

Run:

```bash
.venv/bin/python -m pytest -q tests/test_config_loading.py tests/test_smd_machine_3_4_three_stage_config_loading.py tests/test_three_stage_orchestration_smoke.py
```

## Phase 3 — Replace the outdated `thesis_multitask.py` target semantics

### Phase summary

This is the highest-risk phase because `src/models/thesis_multitask.py` is the main place where current code disagrees with the finalized method. The implementation should remain self-contained and readable, but the target run must stop behaving like the old Exp2 path.

### Files

- Modify: `src/models/thesis_multitask.py`
- Modify: `tests/test_thesis_multitask_config_refactor.py`
- Modify: `tests/test_one_multitask_train_step.py`

### Required edits

#### 1. Introduce explicit method modes instead of implicit legacy defaults

The model config and runtime should make it explicit whether the path is:

- legacy Exp2-compatible behavior,
- or finalized three-stage behavior.

For the finalized path, the old defaults must not remain silently active.

#### 2. Replace the discrete-query path for the finalized run

The current path uses:

- learned `discrete_assignment`,
- `F.gumbel_softmax(...)`,
- and full weighted codebook reconstruction.

For the finalized run, implement a discrete-query mode based on normalized cosine similarity and top-k selection. The implementation should:

- normalize hidden tokens and codewords,
- compute similarity through matrix multiplication,
- select top-k indices,
- soft-weight only that subset,
- aggregate into the discrete branch representation,
- and keep the chosen mode explicitly logged.

The first target setting should support:

- `query_mode: cosine_topk`
- `k=3`
- `tau_q=0.1`

#### 3. Replace forward-path fusion for the finalized run

The current path fuses branches through scalar `alpha` and `beta`, optionally derived from CKA features. For the finalized run, replace that as the main forward-path behavior with `task_specific_concat_projection`.

This should be implemented so:

- reconstruction receives its own concat-projection path,
- classification receives its own concat-projection path,
- and CKA, if retained, is diagnostic-only rather than a forward-routing gate.

#### 4. Preserve output contract stability

Even though internal semantics change, the model must still emit:

- `hidden`
- `pooled`
- `recon`
- `logits`
- `point_scores`
- `window_scores`
- `aux`

### Acceptance criteria

1. The finalized model config constructs successfully.
2. The target run logs that it is using `cosine_topk`.
3. The target run logs that it is using concat-projection fusion.
4. The forward path for the target run does not depend on legacy scalar-gated fusion as its main method.

### Validation commands

Run:

```bash
.venv/bin/python -m pytest -q tests/test_thesis_multitask_config_refactor.py tests/test_one_multitask_train_step.py
```

## Phase 4 — Implement true frozen-memory lifecycle and train-only initialization

### Phase summary

This phase aligns the memory lifecycle with the finalized note. The current code still updates memories during training. The target implementation must initialize from train-derived latent tokens and then freeze.

### Files

- Modify: `src/models/thesis_multitask.py`
- Modify: `tests/test_multitask_memory_updates.py`
- Add or extend targeted memory-initialization tests

### Required edits

#### 1. Split continuous and discrete initialization by role

Continuous memory initialization must use:

- normal-only covering selection.

Discrete memory initialization must use:

- class-stratified covering selection across 12 classes.

#### 2. Initialize from zip-recovered encoder space only

Memory initialization should happen after:

1. Stage 1A,
2. Stage 1B,
3. Stage 2 zipping,
4. Stage 2 recovery.

Only then may the latent token pool be harvested for memory construction.

#### 3. Freeze memories after initialization

For the finalized first implementation:

- continuous memory is read-only after initialization,
- discrete memory is read-only after initialization,
- no EMA update path should remain active for the target method.

#### 4. Enforce train-only sourcing

Memory initialization and any synthetic anomaly generation used for that initialization must not use:

- test windows,
- test labels,
- or test-derived statistics.

### Acceptance criteria

1. Tests prove that memory initialization only uses train-derived artifacts.
2. Tests prove that finalized-run memories do not update during later train steps.
3. Legacy EMA-update expectations are removed or isolated away from the finalized path.

### Validation commands

Run:

```bash
.venv/bin/python -m pytest -q tests/test_multitask_memory_updates.py tests/test_multitask_memory_initialization.py tests/test_multitask_memory_bootstrap.py
```

## Phase 5 — Implement minimal correct overlap-aware contrastive semantics

### Phase summary

This phase replaces the old one-positive-only contrastive behavior with the minimal correct semantics required by the finalized note, while deliberately avoiding premature custom batch-sampler complexity.

### Files

- Modify: `src/models/thesis_multitask.py`
- Possibly modify: `src/data/loaders.py`
- Extend tests around contrastive logic and metadata usage

### Required edits

#### 1. Use overlap-aware positive construction

The finalized first implementation should:

- use aligned-view positives,
- use same-source-timestep positives from other overlapping windows when naturally available in a batch,
- fall back to aligned-view positives when overlap positives are absent,
- and log overlap-positive availability explicitly.

#### 2. Do not require a custom batcher in pass one

The implementation may rely on natural overlap occurrence under `stride=1` for the first version. If later optimization is needed, that should be treated as an extension, not a prerequisite.

#### 3. Keep the batching contract external shape unchanged

Do not invent a new batch tensor structure. Use the enriched metadata contract to resolve source-timestep identity.

### Acceptance criteria

1. The target contrastive helper no longer reduces to the old one-positive-only logic.
2. Tests verify that overlap positives can be discovered from metadata.
3. The target run logs overlap-positive availability statistics.

### Validation commands

Run:

```bash
.venv/bin/python -m pytest -q tests/test_one_multitask_train_step.py tests/test_exp2_two_view_cka.py
```

If old Exp2-specific tests become semantically obsolete, replace them with finalized-method tests under new filenames rather than keeping misleading legacy assertions.

## Phase 6 — Add `tmux` server automation for RTX 3090 runs

### Phase summary

This phase makes the experiment runnable on the actual target environment. The repository currently lacks a server-oriented launch surface. The new one must be explicit and resilient.

### Files

- Create: `scripts/launch_tmux_three_stage_experiment.sh`
- Possibly modify: `scripts/run_multiseed_experiments.py` only if reuse is truly clean
- Add a server-launch dry-run test or config validation test

### Required edits

#### 1. Add a dedicated `tmux` launcher

The launcher should:

- accept the experiment config path,
- derive a stable session name,
- create or replace a `tmux` session deliberately,
- route stdout/stderr to a log file,
- run `scripts/run_three_stage_offline_pretraining.py`,
- and print the exact `tmux attach -t ...` command.

#### 2. Add preflight checks

Before launching, the script should verify:

- config file exists,
- dataset root exists,
- output path and checkpoint path are coherent,
- and the smoke or real run mode is explicit.

#### 3. Keep server commands reproducible

The launcher should save:

- resolved config,
- exact command,
- log path,
- and stage artifact locations.

### Acceptance criteria

1. The launcher can dry-run successfully.
2. The launcher produces deterministic session and log naming.
3. The launched command uses the finalized three-stage orchestration script, not the old single-loop `train.py` path directly.

### Validation commands

Run:

```bash
bash scripts/launch_tmux_three_stage_experiment.sh --help
```

and a dry-run variant once implemented.

## Phase 7 — Wire final true-test evaluation into the orchestration end state

### Phase summary

This phase ensures the user’s requested end state is literally true: after training, the system must test on windows cut from the true test sequence.

### Files

- Modify: `scripts/run_three_stage_offline_pretraining.py`
- Modify if needed: `scripts/evaluate.py`
- Extend test/evaluation artifact checks

### Required edits

#### 1. Make final test evaluation an explicit orchestration phase

The orchestration script must not stop after training. It must call the evaluation path on the true `test` loader.

#### 2. Persist full evaluation artifacts

The final run should save:

- evaluation metrics,
- evaluation records,
- evaluation curves,
- resolved config,
- and the final chosen checkpoint path.

#### 3. Preserve overlap-aware evaluation semantics

Because the evaluator already averages overlapping contributions back to the original timeline, do not replace that logic. Use it as the official target behavior for `stride=1` test windows.

### Acceptance criteria

1. A finalized run ends with a true test evaluation stage.
2. Evaluation artifacts are written to disk under the run output directory.
3. The test path is auditable from output files alone.

### Validation commands

Run:

```bash
.venv/bin/python scripts/evaluate.py --experiment-config <resolved-config> --checkpoint-path <best-checkpoint>
```

using the finalized `machine-3-4` experiment output once available.

## Edit Order

The implementation should follow this exact order:

1. Data config and metadata contract.
2. Config validation and orchestration surface.
3. `thesis_multitask.py` semantic refactor for finalized method mode.
4. Frozen-memory initialization lifecycle.
5. Overlap-aware contrastive semantics.
6. `tmux` launcher.
7. Final evaluation wiring.
8. Smoke validation.
9. Broader regression tests.

This order minimizes the risk of debugging model semantics before the data and stage boundaries are trustworthy.

## Measurable Final Acceptance

The implementation is acceptable only when all of the following are true:

1. `machine-3-4` run configuration exists and uses `window_size=20`, `stride=1`.
2. The SMD loader path stays normalize-before-windowing and is tested.
3. The target run uses the finalized three-stage procedure, not the old Exp2 shortcut.
4. The target run uses frozen memories, `cosine_topk`, and `task_specific_concat_projection`.
5. The experiment can be launched via `tmux` on an RTX 3090 server.
6. The run ends with test evaluation on windows cut from the real test sequence.
7. Tests and smoke checks prove the contracts above.
