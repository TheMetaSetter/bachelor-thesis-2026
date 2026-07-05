---
date: 2026-07-05 15:19:31 +07
researcher: Artificial Intelligence Agent
git_commit: 79031c28ad7bfa61e53b366676683d04f5863981
branch: dev
repository: bachelor-thesis-2026
topic: "Classify phase/stage hits from a root grep"
tags: [research, stage, phase, semantic-drift]
status: complete
last_updated: 2026-07-05
last_updated_by: Artificial Intelligence Agent
---

# Research: Classify phase/stage hits from a root grep

**Date**: 2026-07-05 15:19:31 +07
**Researcher**: Artificial Intelligence Agent
**Git Commit**: 79031c28ad7bfa61e53b366676683d04f5863981
**Branch**: dev

## Research Question
Quet tiep tu root folder va phan loai cac hit co keyword `phase` va `stage` thanh 3 nhom:
active two-stage, legacy three-stage, va runtime `stage_name` binh thuong nhu `train`, `val`, `test`.

## Summary
The repository contains three distinct uses of the words `phase` and `stage`.

1. The active two-stage offline pre-training path uses stage semantics for the current rerun contract, even though some public keys still retain `phase_name` and `two_stage_phase` for compatibility.
2. The legacy three-stage offline pre-training path remains present as an archived or compatibility-oriented runtime, with its own runner, tests, and config files.
3. Ordinary runtime `stage_name` usage appears throughout model step methods, the trainer, and visualization scripts, where the word means batch or execution split names such as `train`, `val`, `val_synth`, and `test`.

The grep results are therefore not one uniform semantic problem. Most of the ambiguity comes from the active two-stage runner and from legacy three-stage compatibility surface, while ordinary `stage_name` usage in step methods is not semantic drift.

## Detailed Findings

### Active Two-Stage

The active two-stage contract is concentrated in the offline two-stage runner, the two-stage config validation surface, and the model state/setup helpers that interpret Stage A and Stage B as the active rerun contract.

Relevant code:
- [`scripts/run_two_stage_offline_pretraining.py`](../../../scripts/run_two_stage_offline_pretraining.py)
  - `build_two_stage_training_plan()` uses `TWO_STAGE_A_PHASE_NAME` and `TWO_STAGE_B_PHASE_NAME` to construct the current offline plan.
  - The runtime still serializes `phase_name`, `two_stage_phase`, and `training_phase`, so the public schema is compatibility-shaped even though the contract is stage-first.
- [`src/core/config.py`](../../../src/core/config.py)
  - `_validate_two_stage_config()` defines the active rerun contract.
  - The validator explicitly separates `two_stage` from `three_stage`.
- [`src/models/thesis_multitask_state_mixin.py`](../../../src/models/thesis_multitask_state_mixin.py)
  - `_semantic_stage_label()` returns `Stage A` or `Stage B` for active two-stage runs.
  - `get_memory_lifecycle_state()` explicitly says the lifecycle state is stage-facing for active two-stage runs.
- [`src/models/thesis_multitask_setup_mixin.py`](../../../src/models/thesis_multitask_setup_mixin.py)
  - `_phase_uses_prototype_path()` and `_phase_freezes_encoder()` mark Stage B as the active freeze point.

Evidence lines:
- [`scripts/run_two_stage_offline_pretraining.py:77-155`](../../../scripts/run_two_stage_offline_pretraining.py#L77-L155)
- [`src/core/config.py:232-281`](../../../src/core/config.py#L232-L281)
- [`src/models/thesis_multitask_state_mixin.py:161-236`](../../../src/models/thesis_multitask_state_mixin.py#L161-L236)
- [`src/models/thesis_multitask_setup_mixin.py:189-233`](../../../src/models/thesis_multitask_setup_mixin.py#L189-L233)

Interpretation:
- `offline pre-training` is the large phase.
- Stage A and Stage B are the only active stages inside that phase for the two-stage rerun.
- Public field names still expose some `phase` words, but the active meaning is stage-first.

### Legacy Three-Stage

The legacy three-stage path is still present and is clearly separate from the active two-stage rerun. It is not the current active design for the new rerun, but it is still supported as compatibility and historical material.

Relevant code:
- [`scripts/run_three_stage_offline_pretraining.py`](../../../scripts/run_three_stage_offline_pretraining.py)
  - The file header says it is a preflight and plan builder for three-stage offline pre-training.
  - The runner builds `THREE_STAGE_PHASE_FIELD_ORDER` and produces `phase_name` records intentionally.
- [`src/core/config.py`](../../../src/core/config.py)
  - `_normalize_three_stage_config_keys()` and `_validate_three_stage_config()` keep legacy Stage 3 alias support.
- [`src/models/thesis_multitask_components.py`](../../../src/models/thesis_multitask_components.py)
  - `ThreeStageRuntimeConfig` still accepts both the canonical Stage 3 label and the active two-stage labels.
- [`tests/test_three_stage_orchestration_smoke.py`](../../../tests/test_three_stage_orchestration_smoke.py)
  - The smoke test asserts the five-phase three-stage schedule and the Stage 3 semantics.
- [`tests/test_three_stage_server_preflight.py`](../../../tests/test_three_stage_server_preflight.py)
  - The preflight tests exercise the three-stage offline pre-training configs.

Evidence lines:
- [`scripts/run_three_stage_offline_pretraining.py:1-37`](../../../scripts/run_three_stage_offline_pretraining.py#L1-L37)
- [`scripts/run_three_stage_offline_pretraining.py:41-47`](../../../scripts/run_three_stage_offline_pretraining.py#L41-L47)
- [`scripts/run_three_stage_offline_pretraining.py:126-220`](../../../scripts/run_three_stage_offline_pretraining.py#L126-L220)
- [`src/core/config.py:129-162`](../../../src/core/config.py#L129-L162)
- [`src/models/thesis_multitask_components.py:309-339`](../../../src/models/thesis_multitask_components.py#L309-L339)
- [`tests/test_three_stage_orchestration_smoke.py:22-35`](../../../tests/test_three_stage_orchestration_smoke.py#L22-L35)

Interpretation:
- The three-stage path is legacy or compatibility-oriented.
- It should not be read as the active meaning of the current two-stage rerun.
- Grep hits in this family are expected and should be treated as historical unless they sit in active two-stage files.

### Runtime `stage_name`

The keyword `stage_name` appears widely in ordinary runtime code. In these files, it means the execution split or loop step name, not the offline pre-training phase/stage taxonomy.

Relevant code:
- [`src/engine/trainer.py`](../../../src/engine/trainer.py)
  - `stage_name` is used for `train`, `val`, `val_synth`, and `test`.
  - The trainer logs metrics under stage-prefixed names.
- [`src/models/reconstruction_mlp_ae.py`](../../../src/models/reconstruction_mlp_ae.py)
  - The baseline model uses `stage_name` in its shared step methods for `train`, `val`, and `test`.
- [`src/models/thesis_multitask_loss_mixin.py`](../../../src/models/thesis_multitask_loss_mixin.py)
  - `stage_name` is the runtime step name used inside shared training and evaluation logic.
- [`scripts/visualize_classification_diagnostics.py`](../../../scripts/visualize_classification_diagnostics.py)
  - The script groups records by `stage` for plotting and diagnostics.
- [`src/models/online_adaptation.py`](../../../src/models/online_adaptation.py)
  - The online path logs completion of an online adaptation stage step.

Evidence lines:
- [`src/engine/trainer.py:85-135`](../../../src/engine/trainer.py#L85-L135)
- [`src/engine/trainer.py:170-210`](../../../src/engine/trainer.py#L170-L210)
- [`src/engine/trainer.py:689-833`](../../../src/engine/trainer.py#L689-L833)
- [`src/models/reconstruction_mlp_ae.py:120-160`](../../../src/models/reconstruction_mlp_ae.py#L120-L160)
- [`src/models/thesis_multitask_loss_mixin.py:689-833`](../../../src/models/thesis_multitask_loss_mixin.py#L689-L833)
- [`scripts/visualize_classification_diagnostics.py:18-132`](../../../scripts/visualize_classification_diagnostics.py#L18-L132)
- [`src/models/online_adaptation.py:497`](../../../src/models/online_adaptation.py#L497)

Interpretation:
- This usage is normal runtime step naming.
- It does not imply the offline phase/stage terminology is wrong.
- These hits should not be treated as semantic drift unless they appear in a file that is already part of the active phase/stage contract.

## Classification Table

| Group | What it means | Typical files |
|---|---|---|
| Active two-stage | Current offline pre-training rerun contract with Stage A and Stage B | `scripts/run_two_stage_offline_pretraining.py`, `src/core/config.py`, `src/models/thesis_multitask_state_mixin.py`, `src/models/thesis_multitask_setup_mixin.py` |
| Legacy three-stage | Historical or compatibility-supported three-stage offline pre-training path | `scripts/run_three_stage_offline_pretraining.py`, `src/core/config.py`, `src/models/thesis_multitask_components.py`, `tests/test_three_stage_*` |
| Runtime `stage_name` | Ordinary execution-step naming such as `train`, `val`, `test` | `src/engine/trainer.py`, `src/models/reconstruction_mlp_ae.py`, `src/models/thesis_multitask_loss_mixin.py`, `scripts/visualize_classification_diagnostics.py` |

## Pipeline Documentation

The semantic picture is:

```text
offline pre-training phase
  -> active two-stage rerun
     -> Stage A
     -> Stage B

offline pre-training phase
  -> legacy three-stage path
     -> stage1
     -> stage2
     -> stage3
     -> multitask pretraining

runtime step naming
  -> train
  -> val
  -> val_synth
  -> test
```

The key distinction is that the first two lines belong to the offline pre-training phase design taxonomy, while the last line is ordinary runtime step naming.

## Historical Context (from documents/)

- `documents/design/offline_pretraining_two_stage_kmeans_memory_design.md` defines the current active two-stage rerun contract and explicitly treats the older three-stage material as historical context.
- `documents/design/offline_pretraining_phase_two_view_contrastive_design.md` uses phase terminology for the offline pre-training design and stage terminology for sub-steps inside a phase.
- `documents/logs/07-05-2026/research/research-stage-phase-semantic-drift.md` already established that the active two-stage path is the main source of reader confusion, while runtime `stage_name` usage is generally benign.

## Open Questions

- Should the public two-stage manifest keep `phase_name` and `two_stage_phase` for compatibility, or should a later migration rename them to stage-first public keys?
- Should `ThreeStageRuntimeConfig` remain as a compatibility class, or should the active two-stage runtime get its own clearer runtime type?
- Should legacy three-stage files remain unchanged except for comments, or should they be fenced off more explicitly as archived context in docs?

