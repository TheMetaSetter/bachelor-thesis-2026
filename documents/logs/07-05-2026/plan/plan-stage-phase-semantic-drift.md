---
date: 2026-07-05 14:04:14 +07
researcher: TheMetaSetter
git_commit: 757d9480d72ee0a1925b0b7194b05b599b3b2f0f
branch: dev
repository: bachelor-thesis-2026
topic: "Plan to eliminate stage/phase semantic drift"
tags:
  - plan
  - stage
  - phase
  - semantics
  - thesis-multitask
status: draft
last_updated: 2026-07-05
last_updated_by: TheMetaSetter
---

# Plan: Eliminate stage/phase semantic drift

## Current State

- The active two-stage offline-pretraining rerun is already implemented and verified by focused tests.
- The current runner in `scripts/run_two_stage_offline_pretraining.py` treats Stage A and Stage B as the two execution units of the offline-pretraining phase, but the public manifest still uses `phase_name` and `two_stage_phase`.
- The main thesis model still carries historical runtime labels in `src/models/thesis_multitask_components.py`, `src/models/thesis_multitask_setup_mixin.py`, and `src/models/thesis_multitask_state_mixin.py`.
- The config loader in `src/core/config.py` still normalizes legacy Stage 3 aliases for compatibility.
- The repository-wide grep shows that most remaining `phase` usages are historical three-stage artifacts, while the active two-stage surfaces are concentrated in the runner, model runtime metadata, and config validation.

## Design Options

### Option A: Compatibility-first cleanup

- Keep all serialized keys and legacy class names stable.
- Rewrite only comments, docstrings, and internal variable names where the meaning is currently ambiguous.
- Add narrow tests that assert the cleaned wording and preserve the current runtime contract.

This option is the lowest risk. It improves readability without forcing downstream consumers to update immediately.

### Option B: Canonical stage-first naming with compatibility shims

- Introduce stage-first public names in the two-stage runner and model runtime metadata.
- Keep compatibility aliases for older three-stage and older manifest consumers.
- Update tests to assert canonical stage-first behavior while preserving backward compatibility coverage.

This option is the best fit for the user's intent because it creates one dominant meaning for the active two-stage rerun without breaking the historical path immediately.

### Option C: Full terminology migration

- Rename the active public two-stage surfaces all the way through manifests, config metadata, runtime state, and test names.
- Remove or isolate most legacy phase terminology into explicit compatibility wrappers.

This option is the cleanest conceptually, but it is broader than needed for a short refactor pass and should be treated as a separate migration if selected.

## Recommended Direction

Choose Option B.

The repository should keep the active two-stage behavior stable, but the active two-stage surfaces should be rewritten so the reader sees one meaning only:

- `offline pre-training` remains the large phase.
- `Stage A` and `Stage B` are the sub-stages inside that phase.
- Historical three-stage names stay available only where compatibility requires them.

## Proposed Refactor Passes

### Pass 1: Make the active two-stage runner stage-first in its internal vocabulary

- Current behavior: `scripts/run_two_stage_offline_pretraining.py` generates the correct plan, manifest, and Stage B bootstrap checkpoint, but several public keys still say `phase`.
- Structural improvement: rename internal variables to `stage_*`, keep output schema stable for now, and update comments so the runner explains the offline-pretraining phase and its two stages clearly.
- Validation check: `pytest -q tests/test_offline_pretraining_two_stage_runner.py tests/test_offline_pretraining_two_stage_config_loading.py`.

### Pass 2: Clean model runtime metadata so active stage labels read clearly

- Current behavior: `src/models/thesis_multitask_state_mixin.py` exposes `semantic_stage_label`, substep flags, and trainable-module snapshots, but still lives beside historical stage-family names.
- Structural improvement: keep the runtime logic intact, but rewrite labels and comments so Stage A / Stage B semantics are explicit and historical Stage 3 compatibility is clearly marked as such.
- Validation check: focused tests for memory lifecycle state, phase switching, and trainable surface remain green.

### Pass 3: Separate active two-stage semantics from legacy three-stage compatibility in config validation

- Current behavior: `src/core/config.py` still normalizes legacy Stage 3 aliases and preserves three-stage validation logic beside two-stage validation logic.
- Structural improvement: make the active two-stage contract explicit in comments and error messages, and keep legacy normalization only where a compatibility shim is genuinely required.
- Validation check: config-loading tests for both two-stage and three-stage families continue to pass.

### Pass 4: Rename or isolate the confusing runtime container names

- Current behavior: `ThreeStageRuntimeConfig` still accepts two-stage labels and therefore does not advertise the active contract clearly.
- Structural improvement: either rename the active runtime container in a compatibility-safe way or isolate the legacy class into a clearly historical compatibility boundary.
- Validation check: model config construction tests, three-stage compatibility tests, and two-stage runner tests remain stable.

### Pass 5: Rewrite user-facing docs and logs so they teach one meaning only

- Current behavior: `documents/logs/` contains historical three-stage notes, while the active two-stage design lives in `documents/design/offline_pretraining_two_stage_kmeans_memory_design.md`.
- Structural improvement: update only the active docs and comments that a new reader will see first; leave historical logs intact but explicitly label them historical.
- Validation check: a fresh grep from the repository root should show that the active two-stage surfaces no longer read as if `phase` and `stage` are interchangeable.

## Risk and Mitigation

- Risk: renaming public keys too aggressively may break downstream consumers.
  - Mitigation: keep a compatibility boundary and migrate one surface at a time.
- Risk: cleaning comments without changing names may leave the reader still confused.
  - Mitigation: change the internal variable names first, then rewrite the comments and docs around them.
- Risk: historical three-stage notes can drown out the active contract in a repository-wide search.
  - Mitigation: label historical notes explicitly and keep the active two-stage design in the SSOT files.
- Risk: the codebase can regain ambiguity if future refactors reintroduce mixed terminology.
  - Mitigation: treat `minimalistic`, `easy to comprehend`, and `single-meaning` as mandatory repository constraints and check them in review.

## Open Questions

- Should the active two-stage manifest keep `phase_name` as a compatibility field, or should a later migration switch it to a stage-first public name?
- Should `ThreeStageRuntimeConfig` remain as a compatibility type, or should the active two-stage runtime get its own clearly named class?
- Should the historical three-stage files stay untouched except for labels, or should they be explicitly fenced off as archived context in docs?

## Minimal Vertical Slice

The first implementation slice should be small:

1. keep the two-stage execution behavior unchanged,
2. make the runner vocabulary stage-first internally,
3. clarify model runtime labels,
4. keep compatibility aliases working,
5. verify with the existing two-stage config and runner tests.

That slice is enough to make the active contract easier to read without turning the work into a migration project.
