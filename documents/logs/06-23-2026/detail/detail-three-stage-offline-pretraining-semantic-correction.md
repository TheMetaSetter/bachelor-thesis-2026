---
date: 2026-06-23 12:19:48 +0700
researcher: TheMetaSetter
git_commit: be3ef38b1ef8ad8677991e9fbd25bd2414c86d7a
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed implementation checklist for semantic correction of the three-stage offline pre-training pipeline"
tags: [detail, offline-pretraining, three-stage, smd, thesis_multitask]
status: complete
last_updated: 2026-06-23
last_updated_by: TheMetaSetter
---

# Detail: Detailed implementation checklist for semantic correction of the three-stage offline pre-training pipeline

## Purpose

This note expands the preliminary plan into a programming-facing checklist for the current SMD `machine-3-4`, `window_size=20`, `stride=1` three-stage offline pre-training pipeline.

The immediate thesis-facing objective is not to redesign the repository. The objective is to correct the semantic mismatch between:

- the finalized three-stage wording in `documents/logs/06-17-2026/detail/detail-offline-pretraining-three-stage-discussion-context.md`,
- and the actual executable code path in:
  - `src/models/thesis_multitask.py`,
  - `scripts/run_three_stage_offline_pretraining.py`,
  - and the associated configs and runtime tests.

The detail here is intentionally implementation-oriented. It specifies the phases of corrective work, the exact ownership surface of each change, the validation steps, and measurable acceptance criteria.

## Locked Semantic Targets

The following targets must be treated as fixed for the next implementation cycle.

### Exact Stage 3 Wording

`Stage 3: Memory Initialization and Fusion Warm-Up`

- initialize continuous memory from normal-only recovered training features;
- initialize discrete memory from class-stratified recovered training features;
- freeze both memory banks after initialization;
- freeze the recovered zipped encoder during the short warm-up;
- train only the task heads and task-specific concat-projection fusion layers.

This wording is not merely descriptive. It is an implementation contract.

### Exact Budget Constraint

The total configured training budget for the main run must remain exactly:

`300 epochs`

No code change in this cycle is allowed to increase the configured total beyond `300`.

### Current Batch and Encoder Contracts

The following contracts must remain stable:

- batch dictionary with `x`, `point_labels`, `mask`, `timestamps`, `meta`;
- encoder public representation `hidden: Tensor[B, L, H]`;
- self-contained multitask model ownership in `src/models/thesis_multitask.py`.

## Phase 1: Stage 3 Semantic Correction

### Phase Summary

The first corrective phase must align the model runtime with the finalized Stage 3 wording. This is the least ambiguous part of the corrective work and should be completed before any Stage 2 algorithmic replacement.

The main goal is to ensure that the runtime behavior of Stage 3 is exactly:

1. initialize memories from recovered training features;
2. freeze memories;
3. freeze recovered zipped encoder;
4. train only task heads and task-specific concat-projection fusion layers.

### Files to Edit

- `src/models/thesis_multitask.py`
- `scripts/run_three_stage_offline_pretraining.py`
- `configs/model/thesis_multitask_three_stage_window20.yaml`
- `tests/test_three_stage_phase_runtime.py`
- `tests/test_three_stage_orchestration_smoke.py`

### Required Edits

#### 1. Tighten Stage 3 trainable-parameter masking

In `src/models/thesis_multitask.py`, revise `_configure_trainable_parameters_for_phase()` so that when the effective runtime phase is the Stage 3 warm-up:

- `encoder` parameters have `requires_grad=False`;
- continuous memory buffers remain frozen;
- discrete codebook buffers remain frozen;
- only:
  - `classification_head`,
  - `reconstruction_head`,
  - `classification_concat_projection`,
  - `reconstruction_concat_projection`
  remain trainable.

The following modules must not stay spuriously trainable during Stage 3:

- `classification_fusion_gate`,
- `reconstruction_fusion_gate`,
- `continuous_update_gate`,
- `discrete_assignment`,
- any other trainable module not named in the locked Stage 3 wording.

This correction should be local and explicit. It should not rely on indirect side effects of prototype-path toggles.

#### 2. Make Stage 3 substeps explicit in runtime state

In `src/models/thesis_multitask.py`, separate these concepts clearly in comments, helper names, or lifecycle state:

- memory initialization step,
- warm-up step after initialization.

The repository does not need a new model file or a new trainer abstraction for this. The goal is only to stop the code from presenting Stage 3 as a vague monolithic phase internally.

At minimum, the runtime state should make it easy to inspect:

- whether memory initialization has already happened,
- whether warm-up is now running with frozen memories and frozen encoder,
- which modules are currently trainable.

#### 3. Make memory-source semantics explicit

The current implementation already sources discrete labels from synthetic train labels and continuous tokens from normal positions. That logic should remain, but the code comments and runtime logs in `src/models/thesis_multitask.py` should say explicitly that:

- both memory banks are initialized from recovered training features;
- no test split statistics, labels, or windows are used.

If the code keeps a capped-batch initialization policy for now, that policy must be described honestly in logs and comments rather than silently presented as if it were exhaustive.

#### 4. Reconcile Stage 3 naming in orchestration artifacts

In `scripts/run_three_stage_offline_pretraining.py`, keep the existing generated-config and execution path stable if needed, but update manifest/report wording so that user-facing artifacts no longer imply that Stage 3 is only "prototype warm-up" in the older narrow sense.

The preferred behavior is:

- preserve existing runtime compatibility;
- add clearer labels, comments, or manifest metadata that Stage 3 is the combined `Memory Initialization and Fusion Warm-Up` stage.

### Interface and Contract Rules for Phase 1

- Do not change batch keys.
- Do not change output tensor shapes.
- Do not split the multitask model into multiple files.
- Do not move Stage 3 semantics into the trainer unless the trainer change is strictly needed to expose lifecycle state cleanly.

### Test Plan for Phase 1

Add or update tests so they verify:

1. Stage 3 warm-up freezes the encoder.
2. Stage 3 warm-up leaves only task heads and concat-projection layers trainable.
3. Stage 3 memory banks are frozen after initialization.
4. Stage 3 memory initialization still sources labels from the synthetic training split only.
5. Smoke orchestration still completes with the corrected Stage 3 semantics.

Recommended commands:

```bash
.venv/bin/python -m pytest -q \
  tests/test_three_stage_phase_runtime.py \
  tests/test_three_stage_orchestration_smoke.py \
  tests/test_smd_machine_3_4_three_stage_config_loading.py
```

### Acceptance Criteria for Phase 1

Phase 1 is complete only if all of the following are true:

1. The Stage 3 runtime behavior matches the locked wording exactly.
2. No module outside task heads and concat-projection layers remains trainable during Stage 3 warm-up.
3. Stage 3 still works with the current smoke orchestration path.
4. The exact total configured budget remains `300`.

## Phase 2: Stage 2 MTZ Replacement

### Phase Summary

The second corrective phase must replace the current parameter-average initialization with a first real implementation of the intended Stage 2 Multi-Task Zipping behavior.

The goal is not to build a large generic zipping framework. The goal is to implement the first thesis-facing zipping path for the current `cnn_simple` encoder while keeping the downstream contracts stable.

### Files to Edit

- `scripts/run_three_stage_offline_pretraining.py`
- `src/models/thesis_multitask.py` only if helper code is genuinely needed
- `tests/test_three_stage_orchestration_smoke.py`
- `tests/test_three_stage_stage2_zipping_runtime.py` or equivalent new focused test
- `tests/test_three_stage_run_verifier.py`

### Required Edits

#### 1. Remove Stage 2 as pure parameter averaging

In `scripts/run_three_stage_offline_pretraining.py`, replace the current logic in `_prepare_stage2_recovery_initialization_checkpoint(...)` that does:

- `encoder = 0.5 * (classification + reconstruction)`.

That logic may remain only as an explicitly named fallback if necessary for temporary compatibility, but it must no longer be presented as the main Stage 2 path.

#### 2. Implement the first zipped encoder initialization path

The replacement Stage 2 path should:

- read both Stage 1 checkpoints;
- construct one shared encoder initialization according to the intended first zipping interpretation;
- preserve the existing `cnn_simple` encoder module layout and state-dict compatibility;
- initialize task heads from the corresponding Stage 1 heads as already intended in the design note.

This implementation should stay local to the current orchestration script unless a very small helper in the model file makes the code significantly clearer.

#### 3. Keep downstream checkpoint handoff stable

The Stage 2 output checkpoint must still be loadable by the Stage 2 recovery run and by the later Stage 3 path without requiring invasive checkpoint-manager changes.

The safest approach is:

- preserve state-dict key names,
- preserve tensor shapes,
- preserve checkpoint payload structure.

### Interface and Contract Rules for Phase 2

- Preserve `hidden: Tensor[B, L, H]` as the public encoder contract.
- Preserve the current task-head ownership.
- Preserve checkpoint payload keys used by the trainer and verifier.
- Do not add a broad abstraction layer for generic zipping policies unless a later cycle truly requires it.

### Test Plan for Phase 2

Add or update tests so they verify:

1. Stage 2 no longer reports the old averaging approximation as the primary path.
2. The Stage 2 initialization checkpoint contains a valid encoder state dict.
3. Stage 3 can load the new Stage 2 output checkpoint.
4. Smoke orchestration still completes end-to-end.

Recommended commands:

```bash
.venv/bin/python -m pytest -q \
  tests/test_three_stage_orchestration_smoke.py \
  tests/test_three_stage_run_verifier.py \
  tests/test_three_stage_stage2_zipping_runtime.py
```

### Acceptance Criteria for Phase 2

Phase 2 is complete only if all of the following are true:

1. The primary Stage 2 path is no longer simple parameter averaging.
2. The new Stage 2 initialization checkpoint remains compatible with later phases.
3. The smoke three-stage run still completes successfully.
4. User-facing logs no longer mislabel the primary Stage 2 path.

## Phase 3: Epoch Accounting and Reporting Reconciliation

### Phase Summary

After Stage 3 and Stage 2 are corrected semantically, the orchestration and reporting layer must be cleaned so that it no longer reinforces the old ambiguity.

This phase is not about changing the total budget. It is about making the budget and stage semantics legible and honest.

### Files to Edit

- `scripts/run_three_stage_offline_pretraining.py`
- `scripts/preflight_three_stage_server.py`
- `scripts/launch_tmux_three_stage_experiment.sh`
- `scripts/verify_three_stage_run.py`
- `tests/test_three_stage_server_preflight.py`
- `tests/test_three_stage_server_launcher.py`
- `tests/test_three_stage_run_verifier.py`

### Required Edits

#### 1. Preserve exact total budget

The following configured training sum must stay unchanged:

`50 + 70 + 20 + 20 + 140 = 300`

Any wording cleanup must not accidentally change `epochs`, `expected_total_training_epochs`, or the per-phase fields in the main config.

#### 2. Make reports semantically cleaner

Execution report, manifest, preflight summary, and launcher output should:

- use consistent terms for Stage 3;
- stop implying that Stage 2 and Stage 3 statistical procedures are ordinary optimizer-training stages if that is conceptually misleading;
- retain enough runtime detail for debugging and verification.

#### 3. Keep server tooling stable

The `tmux` launcher and preflight scripts should remain one-command usable on the target RTX 3090 server after these semantic corrections.

### Test Plan for Phase 3

Recommended commands:

```bash
.venv/bin/python -m pytest -q \
  tests/test_three_stage_server_preflight.py \
  tests/test_three_stage_server_launcher.py \
  tests/test_three_stage_run_verifier.py
```

And one dry-run orchestration check:

```bash
bash scripts/launch_tmux_three_stage_experiment.sh --dry-run \
  --experiment-config configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml \
  --session-name detail-semantic-check
```

### Acceptance Criteria for Phase 3

1. Main config still totals exactly `300`.
2. Preflight, launcher, and verifier outputs use consistent terminology.
3. Dry-run launch path still renders a complete command chain.

## Phase 4: Full Local Validation Before Server Reuse

### Phase Summary

The final phase of this cycle is not new coding. It is a disciplined local validation gate before the corrected path is reused on the remote server.

### Files Involved

- all corrected source files from Phases 1 to 3
- smoke experiment config
- main RTX 3090 config

### Validation Checklist

Run the consolidated local suite:

```bash
.venv/bin/python -m pytest -q \
  tests/test_smd_machine_3_4_three_stage_config_loading.py \
  tests/test_three_stage_phase_runtime.py \
  tests/test_three_stage_orchestration_smoke.py \
  tests/test_three_stage_server_preflight.py \
  tests/test_three_stage_server_launcher.py \
  tests/test_three_stage_run_verifier.py
```

Then rerun the smoke orchestration:

```bash
.venv/bin/python scripts/run_three_stage_offline_pretraining.py \
  --experiment-config configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml
```

Then verify smoke artifacts:

```bash
.venv/bin/python scripts/verify_three_stage_run.py \
  --output-dir outputs/smd_offline_pretraining_three_stage_machine_3_4_window20_smoke_seed11
```

### Acceptance Criteria for Phase 4

1. All targeted tests pass.
2. Smoke orchestration completes end-to-end.
3. Run verification reports success.
4. No change in the main config violates the exact `300`-epoch budget.

## Risk Mitigation Notes

### Prototype Redundancy

This cycle does not change the continuous/discrete dual-branch architecture itself. The mitigation here is to keep branch behavior observable in logs and tests, not to redesign the branch structure.

### Fusion Collapse

Because Stage 3 warm-up is being tightened around the concat-projection layers, tests should confirm that the warm-up still uses the intended fusion path rather than silently bypassing it.

### Adaptation Contamination

Online adaptation is out of scope for this cycle. No corrective work in this note should reach into online adaptation modules.

### Evaluation Metric Inflation

This cycle must preserve the current evaluator path and should not alter metric definitions unless a direct semantic bug is found. The aim is to keep training semantics honest without silently moving the evaluation target.

## Final Acceptance Gate for the Whole Detail Note

The corrective cycle described in this note should be considered complete only when:

1. Stage 3 behavior exactly matches the locked wording.
2. Stage 2 no longer relies primarily on simple parameter averaging.
3. The visible main budget remains exactly `300`.
4. Smoke orchestration and verification still pass locally.
5. User-facing runtime artifacts no longer reinforce the earlier Stage 3 ambiguity.

## Immediate Next Step After This Detail Note

The immediate next action after this note is to implement **Phase 1 only** first, namely Stage 3 semantic correction, because:

- it is the least ambiguous surface,
- it is localized mainly to `src/models/thesis_multitask.py`,
- and it reduces the risk of compounding confusion while Stage 2 is still being corrected.
