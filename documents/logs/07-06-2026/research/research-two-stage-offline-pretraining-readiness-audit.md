---
date: 2026-07-06 13:40:52 +07:00
researcher: TheMetaSetter
git_commit: 2872741520d313df4ed4e9f3d7efee9997d4cbe0
branch: dev
repository: bachelor-thesis-2026
topic: "Current readiness of the two-stage offline pretraining codebase"
tags: [research, time-series, anomaly-detection, two-stage, offline-pretraining, thesis-multitask]
status: complete
last_updated: 2026-07-06
last_updated_by: TheMetaSetter
---

# Research: Current readiness of the two-stage offline pretraining codebase

**Date**: 2026-07-06 13:40:52 +07:00  
**Researcher**: TheMetaSetter  
**Git Commit**: 2872741520d313df4ed4e9f3d7efee9997d4cbe0  
**Branch**: dev

## Research Question
Is the current codebase ready to run the experiment described in `documents/design/two-stage-offline-pretraining-spec.md`?

## Summary
The model/runtime implementation is mostly in place, but the checkout is **not yet ready to run the experiment end-to-end** because the active two-stage experiment YAMLs referenced by the tests and runner are missing from the current tree.

The code already contains:
- two-stage config validation in `src/core/config.py`,
- experiment-level mutual exclusion between `three_stage` and `two_stage` in `src/core/config_experiment_validation.py`,
- a dedicated orchestration script in `scripts/run_two_stage_offline_pretraining.py`,
- `stage_name` / `training_phase` compatibility in the model config path,
- Stage A / Stage B phase switching and freeze logic in the model mixins.

However, the current repository snapshot does not contain the exp4 two-stage experiment configs under `configs/experiment/thesis/exp4/`, and the two-stage tests fail immediately with `FileNotFoundError` when they try to load those YAMLs.

## Detailed Findings

### What Is Already Implemented
- `src/core/config.py` accepts `two_stage`, validates the epoch split, checks `discrete_memory_label_source`, and rejects configs that define both `three_stage` and `two_stage`.
- `src/models/thesis_multitask_components.py` and `src/models/thesis_multitask_setup_mixin.py` already accept `stage_name`, normalize it into `training_phase`, and switch trainable surfaces for Stage A and Stage B.
- `src/models/thesis_multitask_state_mixin.py` already contains `maybe_initialize_memories_from_loader(...)`, which is the memory bootstrap hook used after Stage A.
- `scripts/run_two_stage_offline_pretraining.py` already materializes per-stage configs, writes a manifest, prepares `stage_b_init.pt`, and builds the train/evaluate command plan.

### What Is Missing For A Real Run
- The current checkout does not contain the exp4 two-stage experiment YAMLs that the two-stage tests and runner expect:
  - `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-two-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml`
  - `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-two-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml`
- Because those files are absent, `load_experiment_config(...)` fails before the runner can build the plan.

### Test Evidence
- `./.venv/bin/python -m pytest -q tests/test_offline_pretraining_two_stage_config_loading.py tests/test_offline_pretraining_two_stage_runner.py`
- Result: 5 failed, 1 passed.
- The failures are all `FileNotFoundError` from `src/core/config.py:57` while loading the missing exp4 two-stage YAMLs.

## Code References
- `src/core/config.py:279-327` - two-stage config validator
- `src/core/config_experiment_validation.py:1-60` - top-level experiment validation and `three_stage` vs `two_stage` exclusion
- `src/models/thesis_multitask_components.py:310-447` - runtime phase normalization and `stage_name` aliasing
- `src/models/thesis_multitask_setup_mixin.py:135-230` - stage-aware config storage, prototype-path toggles, and freeze behavior
- `src/models/thesis_multitask_setup_mixin.py:260-360` - two-stage memory bank construction and Stage B `cosine_topk` handling
- `src/models/thesis_multitask_state_mixin.py:277-325` - memory bootstrap hook from the training loader
- `src/models/thesis_multitask_routing_mixin.py:385-548` - batch preparation, synthetic augmentation, and shared step routing
- `src/models/thesis_multitask_loss_mixin.py:531-830` - stage loss assembly and metrics
- `scripts/run_two_stage_offline_pretraining.py:45-327` - epoch budget, manifest creation, Stage B init checkpoint, and execution plan
- `tests/test_offline_pretraining_two_stage_config_loading.py:6-43` - exp4 config loading expectations
- `tests/test_offline_pretraining_two_stage_runner.py:6-171` - orchestration and phase-surface expectations

## Pipeline Documentation
The implementation is structurally ready, but the runnable experiment contract is incomplete because the specific experiment YAML entrypoints referenced by the two-stage flow are not present in this checkout.

In practical terms:
1. The code knows how to run two stages.
2. The model knows how to switch phase and freeze surfaces.
3. The repo does not currently ship the exp4 two-stage experiment configs that the runner/tests expect.

## Open Questions
- Are the missing exp4 two-stage YAMLs intentionally removed from this checkout, or should they be reintroduced before marking the experiment runnable?
- If the configs are meant to live elsewhere, which path should become the active SSOT for the two-stage rerun?

## Follow-up Note
The Stage A loss stack is only partially complete relative to the spec quoted by the user.

What is already implemented:
- reconstruction loss via `_compute_reconstruction_loss(...)`
- classification loss via `_compute_classification_loss(...)`
- two-view contrastive loss via `_compute_two_view_contrastive_loss(...)`

What is not implemented yet:
- the optional point-wise balanced reconstruction-score loss

Important distinction:
- `src/models/thesis_multitask_routing_mixin.py` does compute `point_scores` and `window_scores` as outputs for diagnostics / downstream use.
- `src/models/thesis_multitask_loss_mixin.py` does **not** currently turn those point scores into a fourth supervised loss term inside `_shared_step(...)` or `_compute_total_loss(...)`.

So, for the base run, the code covers the first three Stage A losses only. The fourth loss named in the spec is still missing from the training objective and should be treated as not implemented in the current checkout.

Two additional spec items are also not implemented as written:

- The proposed score-loss config surface does not exist in the validator or model config path yet. The current config schema accepts `training_phase`, `discrete_query_mode`, `freeze_memories_after_initialization`, and `discrete_memory_label_source`, but not the new fields `enable_score_loss`, `score_loss_granularity`, `score_loss_type`, `score_loss_target`, or `score_loss_normalization`.
- The batch contract in the spec (`x_clean`, `x_input`, `class_labels`, `is_synthetic`) is not the current runtime contract. The code still uses the existing thesis-multitask keys (`x`, `classification_labels`, `synthetic_anomaly_mask`, `point_labels`), so the spec’s input naming and the code’s input naming are not aligned yet.

One thing that is already implemented and should not be counted as missing:
- timeline-compatible point scores, validation thresholding, VUS-PR/VUS-ROC computation, and affiliation F1 already exist in the evaluation stack. The missing part is the training-side supervision for the point-wise score loss, not the downstream score usage itself.
