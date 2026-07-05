# Stage/Phase Semantic Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `phase` and `stage` single-meaning across the codebase so readers can infer the intended contract from names, comments, and config keys without changing the active two-stage behavior.

**Architecture:** Keep `offline pre-training` as the only large phase. Treat active two-stage and legacy three-stage as separate design surfaces inside that phase, but do not mix their public contracts. Preserve runtime `stage_name` for ordinary execution splits such as `train`, `val`, and `test`. Perform renames in a compatibility-first order: internal names and comments first, public schema changes only in isolated migration steps.

**Tech Stack:** Python, Pytest, YAML config loading, repository design docs under `documents/design/`, local runner scripts.

---

## Context From Research

The research note `documents/logs/07-05-2026/research/research-stage-phase-hit-classification.md` classifies grep hits into three groups:

1. **Active two-stage offline pre-training**
   - Current active rerun contract.
   - Uses Stage A and Stage B inside the `offline pre-training` phase.
   - Still exposes some compatibility-shaped public keys such as `phase_name` and `two_stage_phase`.

2. **Legacy three-stage offline pre-training**
   - Historical or compatibility-supported path.
   - Should remain readable as legacy, not as the active contract.

3. **Runtime `stage_name`**
   - Ordinary execution-step naming in trainer, model step methods, and visualization code.
   - Means `train`, `val`, `val_synth`, `test`, or similar runtime splits.

The main risk is semantic drift caused by one word carrying multiple meanings in different files. The plan below narrows each meaning to one contract and keeps compatibility boundaries explicit.

## Recommended Approach

### Option A: Compatibility-first rename, recommended
- Rename local variables, helper names, and comments first.
- Keep public config and manifest keys stable until the migration boundary is isolated.
- Treat legacy three-stage terminology as historical context.
- Preserve runtime `stage_name` as-is when it refers to execution splits.

Why this is preferred:
- Lowest risk.
- Gives immediate readability gains.
- Does not break configs, tests, or experiment history.

### Option B: Public-contract rename with alias wrappers
- Rename public keys as well, but add alias translation layers.
- More complete, but riskier and more invasive.
- Should be split into a separate migration if needed.

### Option C: Full terminology migration in one pass
- Rename all active and legacy surfaces together.
- Highest risk.
- Not recommended because the repo still intentionally supports legacy three-stage compatibility.

---

## File Map

- `scripts/run_two_stage_offline_pretraining.py`
- `src/core/config.py`
- `src/models/thesis_multitask_state_mixin.py`
- `src/models/thesis_multitask_setup_mixin.py`
- `src/models/thesis_multitask_components.py`
- `scripts/run_three_stage_offline_pretraining.py`
- `src/engine/trainer.py`
- `src/models/reconstruction_mlp_ae.py`
- `src/models/thesis_multitask_loss_mixin.py`
- `scripts/visualize_classification_diagnostics.py`
- `src/models/online_adaptation.py`
- `tests/test_offline_pretraining_two_stage_runner.py`
- `tests/test_offline_pretraining_two_stage_config_loading.py`
- `tests/test_three_stage_phase_runtime.py`
- `tests/test_config_loading.py`

---

## Pass 1: Lock terminology and compatibility boundary

### Current behavior
The research note already shows that `phase` and `stage` are used in three distinct ways, but the codebase still mixes them in comments, helper names, and some compatibility fields.

### Structural improvement
Write a short terminology block into the active SSOT design doc so future renames are guided by a single rule:

- `offline pre-training` is the large phase.
- Stage A and Stage B are stages inside that phase.
- Legacy three-stage is historical or compatibility-only.
- Runtime `stage_name` remains ordinary execution naming.

### Files
- Modify: `documents/design/offline_pretraining_two_stage_kmeans_memory_design.md`
- Modify: `documents/logs/07-05-2026/research/research-stage-phase-hit-classification.md` only if the classification table needs one more clarification sentence

### Validation check
- Manual read-through of the terminology section.
- Verify that the wording does not claim a behavior change.
- Verify that the doc still distinguishes active two-stage, legacy three-stage, and runtime `stage_name`.

---

## Pass 2: Rename internal vocabulary in the active two-stage runner

### Current behavior
`scripts/run_two_stage_offline_pretraining.py` still uses naming such as `phase_name`, `phase_record`, and `training_stages` in places where the code actually handles stages inside the offline pre-training phase.

### Structural improvement
Rename local variables and comments so the file reads stage-first internally while keeping public manifest keys stable. This makes the runner self-explanatory without changing the experiment schema that downstream tools may already consume.

### Files
- Modify: `scripts/run_two_stage_offline_pretraining.py`

### Target rename rules
- `phase_name` local variable -> `stage_name`
- `phase_record` -> `stage_record`
- `training_stages` local name -> `stage_records`
- Comments that say “phase” when the code means Stage A/B -> rewrite to “stage”
- Keep manifest keys such as `phase_name`, `two_stage_phase`, and `training_phase` unchanged in this pass

### Validation check
- `pytest -q tests/test_offline_pretraining_two_stage_runner.py`
- Confirm the output schema keys are unchanged.
- Confirm the log text and comments now reflect stage-first internal meaning.

---

## Pass 3: Make the active two-stage contract explicit in model helpers

### Current behavior
`src/models/thesis_multitask_state_mixin.py` and `src/models/thesis_multitask_setup_mixin.py` already implement the active Stage A / Stage B contract, but some helper names and comments still carry phase-oriented wording.

### Structural improvement
Make the helper names and comments describe what they actually do:

- stage-facing lifecycle state
- active Stage B freeze point
- active two-stage semantic labels

This keeps the model file readable from top to bottom and reduces the chance that a maintainer confuses offline phase taxonomy with runtime step naming.

### Files
- Modify: `src/models/thesis_multitask_state_mixin.py`
- Modify: `src/models/thesis_multitask_setup_mixin.py`
- Modify: `src/models/thesis_multitask_components.py` only if a helper or dataclass name can be made clearer without changing the public interface

### Validation check
- `pytest -q tests/test_offline_pretraining_two_stage_runner.py tests/test_offline_pretraining_two_stage_config_loading.py`
- Confirm Stage A/B labels still resolve exactly as before.
- Confirm the model state serialization stays stable.

---

## Pass 4: Fence off legacy three-stage compatibility

### Current behavior
Legacy three-stage support remains in `scripts/run_three_stage_offline_pretraining.py`, `src/core/config.py`, `src/models/thesis_multitask_components.py`, and three-stage tests. The code is correct but the naming can make active two-stage readers think the two systems are the same thing.

### Structural improvement
Keep legacy three-stage support, but make its compatibility boundary unmistakable:

- say “legacy three-stage” in comments and docstrings
- avoid using legacy wording in active two-stage codepaths
- preserve alias support only where old configs truly need it

### Files
- Modify: `scripts/run_three_stage_offline_pretraining.py`
- Modify: `src/core/config.py`
- Modify: `src/models/thesis_multitask_components.py`
- Modify: `tests/test_three_stage_phase_runtime.py`
- Modify: `tests/test_three_stage_orchestration_smoke.py` if test names still teach the wrong mental model

### Validation check
- `pytest -q tests/test_three_stage_phase_runtime.py tests/test_three_stage_orchestration_smoke.py tests/test_three_stage_server_preflight.py`
- Confirm old three-stage configs still load.
- Confirm active two-stage docs and code do not borrow three-stage labels unless explicitly marked legacy.

---

## Pass 5: Preserve ordinary runtime `stage_name` usage and avoid over-renaming

### Current behavior
Files such as `src/engine/trainer.py`, `src/models/reconstruction_mlp_ae.py`, `src/models/thesis_multitask_loss_mixin.py`, `scripts/visualize_classification_diagnostics.py`, and `src/models/online_adaptation.py` use `stage_name` in the ordinary runtime sense.

### Structural improvement
Do not rename these away from `stage_name` unless the file is explicitly part of the offline pre-training phase taxonomy. This avoids creating fake symmetry where none exists.

### Files
- Review only:
  - `src/engine/trainer.py`
  - `src/models/reconstruction_mlp_ae.py`
  - `src/models/thesis_multitask_loss_mixin.py`
  - `scripts/visualize_classification_diagnostics.py`
  - `src/models/online_adaptation.py`

### Validation check
- Grep for `stage_name` after the rename passes.
- Confirm the remaining occurrences are runtime splits, not offline pre-training taxonomy.
- No code change is required if the meaning is already correct.

---

## Pass 6: Tighten tests and docs so they teach one meaning only

### Current behavior
Some tests and docs still use legacy naming in titles or helper names even when the assertions are already correct.

### Structural improvement
Update names and prose so the tests themselves teach the right contract:

- active two-stage tests should say stage, not phase, when they mean Stage A/B
- legacy three-stage tests should say legacy or compatibility
- runtime step tests should keep `stage_name` where that is the actual runtime contract

### Files
- Modify: `tests/test_offline_pretraining_two_stage_runner.py`
- Modify: `tests/test_offline_pretraining_two_stage_config_loading.py`
- Modify: `tests/test_three_stage_phase_runtime.py`
- Modify: `documents/logs/07-05-2026/research/research-stage-phase-hit-classification.md` only if the classification table needs a final wording pass

### Validation check
- `pytest -q tests/test_offline_pretraining_two_stage_runner.py tests/test_offline_pretraining_two_stage_config_loading.py tests/test_three_stage_phase_runtime.py tests/test_config_loading.py`
- Run a final grep classification check:
  - `rg -n -w "phase_name|two_stage_phase|training_phase|stage_name|phase|stage" scripts src tests documents/design`
- Review the remaining hits by the three research buckets:
  - active two-stage
  - legacy three-stage
  - runtime `stage_name`

---

## Risk and Mitigation

- Risk: renaming public keys breaks saved configs or downstream scripts.
  - Mitigation: keep public schema stable in the first pass and isolate any public-key migration into a separate task.

- Risk: active two-stage and legacy three-stage get conflated again.
  - Mitigation: make the terminology block explicit in the SSOT design doc and keep legacy wording fenced off.

- Risk: runtime `stage_name` gets renamed unnecessarily.
  - Mitigation: treat runtime `stage_name` as a separate semantic bucket and do not rename it unless the file is part of the offline pre-training taxonomy.

- Risk: comments change but the reader still cannot infer the contract.
  - Mitigation: keep test names, helper names, and doc sections aligned with the same terminology.

---

## Definition of Done

- Active two-stage files read stage-first internally.
- Legacy three-stage support is clearly labeled as historical or compatibility-only.
- Runtime `stage_name` usage remains untouched where it means `train`, `val`, or `test`.
- No public schema changes are made without a separate migration step.
- The docs, tests, and code all point to the same meaning for each term.
- The remaining grep hits can be classified cleanly into the three research buckets without ambiguity.

