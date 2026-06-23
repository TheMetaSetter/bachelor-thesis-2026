---
date: 2026-06-23 19:46:32 +0700
researcher: TheMetaSetter
git_commit: be3ef38b1ef8ad8677991e9fbd25bd2414c86d7a
branch: dev
repository: bachelor-thesis-2026
topic: "Audit of the current three-stage offline pre-training implementation after recent semantic-correction patches"
tags: [research, offline-pretraining, three-stage, smd, thesis_multitask]
status: complete
last_updated: 2026-06-23
last_updated_by: TheMetaSetter
---

# Research: Audit of the current three-stage offline pre-training implementation after recent semantic-correction patches

**Date**: 2026-06-23 19:46:32 +0700  
**Researcher**: TheMetaSetter  
**Git Commit**: `be3ef38b1ef8ad8677991e9fbd25bd2414c86d7a`  
**Branch**: `dev`

## Research Question

After the recent semantic-correction patches for the SMD `machine-3-4` three-stage offline pre-training path, what mismatches or ambiguities still remain in the repository, especially around user-facing configuration, runtime artifacts, and stage terminology?

## Summary

The current implementation is functionally consistent enough to run locally. The focused three-stage verification suite completed successfully with `44 passed in 6.48s`. The exact configured budget remains `300` epochs in the active RTX 3090 experiment configuration, and the code now records cleaner semantic metadata for Stage 2 statistical zipping and Stage 3 memory initialization.

However, two user-facing ambiguities still remain in the current repository state.

First, the newly introduced canonical Stage 3 epoch key `stage3_memory_initialization_and_fusion_warmup_epochs` is normalized into memory together with the legacy alias `stage3_prototype_warmup_epochs` in `src/core/config.py:100-127`. Because `scripts/run_three_stage_offline_pretraining.py:162-214` deep-copies the already-normalized experiment configuration and writes it back out as generated per-phase YAML, the generated stage configs currently serialize both keys simultaneously. This means the main experiment file is now cleaner, but the generated runtime artifact still reintroduces the older ambiguous field name.

Second, the generated per-phase YAML still keeps `model.training_phase: multitask_pretraining` from the base model config in `configs/model/thesis_multitask_three_stage_window20.yaml:51`, then applies the actual phase through `model_overrides["training_phase"]` in `scripts/run_three_stage_offline_pretraining.py:192-203`. This is functionally correct after `load_experiment_config(...)` merges overrides, but the raw generated YAML is semantically misleading when inspected directly because the `model` section and `model_overrides` section disagree until the loader resolves them.

In addition, the runtime still exposes the concrete phase name `stage3_prototype_warmup` in filenames, output directories, generated config names, manifest stage records, preflight summaries, verifier outputs, tests, and launcher dry-run text, even though the semantic label has been corrected to `Stage 3: Memory Initialization and Fusion Warm-Up`. This is not a functional bug, but it remains a terminology gap between the conceptual wording and the concrete runtime surface.

## Detailed Findings

### Data Preparation

The target experiment remains scoped to SMD entity `machine-3-4`, `window_size=20`, and `stride=1` through `configs/data/smd_rtx3090_machine_3_4_20_stride1.yaml:1-10`. The underlying parser, split creation, scaling, and overlapping window generation remain unchanged from the previously documented active path:

- SMD parser: `src/data/datasets/smd.py`
- scaler fit and transform: `src/data/loaders.py`
- lazy overlapping windows: `src/data/loaders.py`

No new audit issue was found in the data pipeline during this pass. The batch contract remains the standardized dictionary contract consumed by the trainer and model.

### Modeling and Training

The active RTX 3090 experiment file now uses the clearer Stage 3 field name:

- `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml:42-48`

The exact budget remains:

- `50 + 70 + 20 + 20 + 140 = 300`

The orchestrator still expands the training run into five optimizer-training phases through `scripts/run_three_stage_offline_pretraining.py:27-39`, while separately exposing statistical procedures through `STATISTICAL_PROCEDURE_NAMES` in `scripts/run_three_stage_offline_pretraining.py:42-45`.

The recent Stage 2 correction is visible in `scripts/run_three_stage_offline_pretraining.py:299-514` and `scripts/run_three_stage_offline_pretraining.py:517-578`. The main path is no longer the old identity-based parameter average. It now performs:

- activation capture on the training split,
- cosine-similarity channel matching,
- matched channel merging for the CNN encoder,
- task-head reuse from the corresponding Stage 1 checkpoints,
- and metadata emission under `stage2_zip_metadata`.

This aligns materially better with the locked first-pass MTZ-inspired approximation described in the design note.

### Evaluation and Reporting

The execution report, preflight summary, verifier summary, and launcher dry-run output now distinguish:

- optimizer-training phases,
- statistical procedures,
- and the exact `300`-epoch total.

Relevant files are:

- `scripts/run_three_stage_offline_pretraining.py`
- `scripts/preflight_three_stage_server.py`
- `scripts/verify_three_stage_run.py`
- `scripts/launch_tmux_three_stage_experiment.sh`

The remaining issue is not missing metadata, but conflicting raw artifact wording. In particular, the raw generated stage YAML is still easy to misread because it contains both:

- the canonical Stage 3 epoch key and the legacy alias,
- the base model training phase and the override training phase.

## Code References

- `src/core/config.py:100-127` - normalization that keeps both canonical and legacy Stage 3 epoch keys in memory
- `src/core/config.py:130-186` - three-stage validation over the normalized config mapping
- `scripts/run_three_stage_offline_pretraining.py:27-39` - concrete optimizer-training phase list that still uses `stage3_prototype_warmup`
- `scripts/run_three_stage_offline_pretraining.py:131-150` - output directory and checkpoint path generation that still use `stage3_prototype_warmup`
- `scripts/run_three_stage_offline_pretraining.py:162-214` - generated stage-config builder that deep-copies the already-normalized experiment config and writes `model_overrides`
- `scripts/run_three_stage_offline_pretraining.py:220-241` - manifest generation with semantic metadata
- `configs/model/thesis_multitask_three_stage_window20.yaml:51-58` - base model config still defaults `training_phase` to `multitask_pretraining`
- `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml:42-48` - main config now uses the canonical Stage 3 epoch field
- `scripts/preflight_three_stage_server.py:226-236` - preflight summary exposes both optimizer-training phases and statistical procedures
- `scripts/verify_three_stage_run.py:62-83` - verifier summary now preserves optimizer-training phases and statistical procedures
- `scripts/launch_tmux_three_stage_experiment.sh:190-191` - launcher dry-run prints both phase families but still uses `stage3_prototype_warmup` in the optimizer-training phase list

## Pipeline Documentation

The current concrete runtime surface is still:

1. `stage1_classification`
2. `stage1_reconstruction`
3. `stage2_recovery`
4. `stage3_prototype_warmup`
5. `multitask_pretraining`

while the semantic interpretation layered on top of that runtime is:

- `stage2_mtz_parameter_zipping` as a statistical procedure before Stage 2 recovery,
- `Stage 3: Memory Initialization and Fusion Warm-Up` as the semantic label for the Stage 3 runtime segment,
- `stage3_memory_initialization` as a statistical procedure distinct from optimizer-training epochs.

Therefore, the repository is now in a partially cleaned state: semantic metadata has improved, but the concrete runtime identifiers still preserve older names.

## Historical Context (from documents/)

The locked source of truth remains `documents/logs/06-17-2026/detail/detail-offline-pretraining-three-stage-discussion-context.md`, especially the final wording:

`Stage 3: Memory Initialization and Fusion Warm-Up`

The newer detail and plan notes under `documents/logs/06-23-2026/` explicitly accepted a conservative transition strategy in which semantic metadata could be cleaned before a larger runtime renaming pass. The current codebase state matches that intermediate position: semantics have been clarified in metadata and tests, but concrete runtime names have not been fully migrated.

## Open Questions

1. Should the repository keep `stage3_prototype_warmup` as the concrete internal runtime identifier, or should a later migration rename the concrete phase itself to reduce remaining user-facing ambiguity?
2. Should the config loader continue materializing both the canonical Stage 3 epoch key and the legacy alias into runtime dictionaries, or should the legacy key be kept read-compatible but omitted from generated artifacts?
3. Should generated stage YAML continue to rely on `model_overrides.training_phase`, or should the generated `model.training_phase` field itself be rewritten per stage so the raw artifact is self-consistent before the loader runs?
