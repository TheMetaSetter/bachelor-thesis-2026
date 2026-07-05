---
date: 2026-07-05 14:04:14 +07
researcher: TheMetaSetter
git_commit: 757d9480d72ee0a1925b0b7194b05b599b3b2f0f
branch: dev
repository: bachelor-thesis-2026
topic: "Scan for semantic drift between stage and phase"
tags:
  - research
  - stage
  - phase
  - semantics
  - thesis-multitask
status: complete
last_updated: 2026-07-05
last_updated_by: TheMetaSetter
---

# Research: Scan for semantic drift between stage and phase

**Date**: 2026-07-05 14:04:14 +07
**Researcher**: TheMetaSetter
**Git Commit**: 757d9480d72ee0a1925b0b7194b05b599b3b2f0f
**Branch**: dev

## Research Question
Quet trong repo xem con cho nao nham lan ngua nghia cua `stage` va `phase`.

## Summary
The active two-stage offline pre-training design now treats offline pre-training as the large phase and Stage A / Stage B as the sub-stages. The runtime behavior is mostly aligned with that contract, but several public names still preserve older `phase` terminology for compatibility. Those names are the main source of reader confusion, even when the execution logic is correct.

## Detailed Findings

### 1. Runner logic is correct, but the public field names still say `phase`
- The runner now builds a two-stage plan with `stage_name` and `stage_record` internally, but it still writes `phase_name` into the generated plan and manifest records for compatibility.
- `scripts/run_two_stage_offline_pretraining.py:74-149` builds the plan, stage-specific config, and stage-specific output directory. The internal variables now use `stage_*`, but the serialized keys still use `phase_name` and `two_stage_phase`.
- `scripts/run_two_stage_offline_pretraining.py:156-307` materializes the manifest and execution report with `training_stages`, but each record still carries `phase_name`.

Interpretation:

- This is not a runtime bug.
- It is naming drift in a public artifact, which can still mislead readers into thinking the runner is modeling phases instead of stages.

### 2. The model runtime config still uses a legacy `ThreeStageRuntimeConfig` name
- `src/models/thesis_multitask_components.py:309-337` defines `ThreeStageRuntimeConfig`, but its accepted `training_phase` values include:
  - `stage1_classification`
  - `stage1_reconstruction`
  - `stage2_recovery`
  - `stage_a_multitask_pretraining`
  - `stage_b_fusion_finetuning`
- The class name therefore no longer matches the full runtime surface.

Interpretation:

- This is compatibility-oriented naming drift.
- A new reader can easily infer the wrong contract from the class name alone.

### 3. Stage labels inside model state are mostly compatibility labels, not current phase semantics
- `src/models/thesis_multitask_state_mixin.py:161-230` maps `training_phase` into `semantic_stage_label`, `memory_initialization_substep`, and `fusion_warmup_substep`.
- The code treats Stage A and Stage B as stage labels, which is consistent with the current two-stage design.
- However, the same file still preserves the Stage 3 compatibility path, so readers must distinguish:
  - active two-stage semantics
  - historical Stage 3 compatibility

Interpretation:

- The behavior is coherent, but the mixed historical labels make the file easy to misread.

### 4. Config validation still preserves legacy Stage 3 wording
- `src/core/config.py:228-272` validates the two-stage config surface.
- `src/core/config.py:129-153` keeps the legacy Stage 3 alias normalization logic for backward compatibility.
- `src/core/config.py:310-317` exposes top-level `two_stage_phase` and stage-related metadata keys in the merged experiment config.

Interpretation:

- This is intentional compatibility support.
- The legacy alias logic is useful, but it can make Stage 3 sound more canonical than it really is for the current two-stage rerun.

## Code References
- `scripts/run_two_stage_offline_pretraining.py:74-149` - builds the two-stage plan and stage configs
- `scripts/run_two_stage_offline_pretraining.py:156-307` - writes the manifest and execution report
- `src/models/thesis_multitask_components.py:309-337` - legacy `ThreeStageRuntimeConfig` still accepts two-stage labels
- `src/models/thesis_multitask_state_mixin.py:161-230` - stage labels and memory lifecycle state
- `src/core/config.py:129-153` - legacy Stage 3 alias normalization
- `src/core/config.py:228-272` - two-stage config validation

## Pipeline Documentation

```text
offline pre-training phase
   |
   +--> Stage A: multitask pretraining
   |
   +--> Stage B: fusion finetuning
```

The code path is mostly aligned with this structure. The confusing parts are the public field names and legacy compatibility labels that still say `phase` in places where the current runtime is really talking about a stage.

## Historical Context (from documents/)
- `documents/design/offline_pretraining_two_stage_kmeans_memory_design.md` defines the active two-stage contract and explicitly says the old three-stage intent is superseded for this rerun.
- `documents/design/idea.md` and `documents/design/design_starter.md` still contain broader thesis terminology and are useful for understanding the older naming history.

## Open Questions
- Should the public manifest/config keys stay `phase_name` / `two_stage_phase` for compatibility, or should a later cleanup introduce stage-first names and keep only a compatibility shim?
- Should `ThreeStageRuntimeConfig` be renamed in a future compatibility-aware migration, or should it remain as a legacy container because too much downstream code already depends on it?
