# Codebase Modernization Plan

Date: 2026-07-04

## Goal

Make the codebase simpler, easier to read, and easier to extend without changing behavior unless a later task explicitly asks for a functional change.

The main rule is to preserve the current benchmark and thesis experiment flows, especially the newest ones documented under `documents/`.

## Error Handling Rule

The refactor should make failure modes clearer, not quieter.

- validate inputs early
- raise explicit errors with specific field names
- avoid silent fallback when a config, path, or registry name is invalid
- keep compatibility boundaries narrow so errors are easier to localize

## Scope

This plan focuses on refactors that reduce cognitive load:

- delete dead code
- remove duplicated paths
- split oversized modules
- replace stale compatibility layers with explicit boundaries
- simplify control flow
- extract small helpers with one responsibility

This plan does not include:

- framework migration
- dependency upgrades
- public API changes unless required by a refactor
- architecture moves that need a separate migration task

## Current Shape

The current runtime is still organized around a few large ownership points:

- `src/core/config.py` is the main config validation choke point
- `src/models/thesis_multitask.py` is the largest model owner file
- `src/engine/trainer.py` and `src/engine/evaluator.py` contain overlapping evaluation and threshold logic
- `src/models/redlamp_mlp_baseline.py` is now a compatibility shim, while `src/models/redlamp_baseline.py` is the canonical baseline implementation

The repository already has a stable contract surface, so the refactor should keep those contracts intact and only simplify the internal structure.

## Target Shape

The target is a small set of obvious ownership boundaries:

```text
config load / validation
    -> dataset / task / model resolution
    -> model forward + loss logic
    -> trainer step orchestration
    -> evaluator metric assembly
```

```text
public entrypoints
    -> canonical helper functions
    -> thin compatibility wrappers only where required
```

```text
model file
    -> inputs and config checks
    -> encoder / branch / head setup
    -> one-step training logic
    -> evaluation / scoring helpers
```

The target is not more abstraction. The target is smaller, clearer modules with fewer hidden codepaths.

## Refactor Passes

### Pass 1: Delete dead code and isolate compatibility shims

Current behavior:

- legacy names and compatibility files still exist alongside canonical ones
- some helpers still teach the old mental model even when the canonical path already exists

Structural improvement:

- keep one canonical owner for each active concept
- move legacy support into narrow compatibility-only wrappers
- delete code that is no longer referenced by any active path

Validation check:

- run focused import tests for canonical and compatibility names
- run config loading checks for active experiment files
- run the baseline smoke tests that cover the current public entrypoints

### Pass 2: Simplify public entrypoints

Current behavior:

- `train`, `evaluate`, and online-adaptation entrypoints each repeat model registration and setup patterns

Structural improvement:

- extract shared registration and resolution helpers
- keep entrypoints thin and linear
- preserve CLI flags and output behavior

Validation check:

- run the current CLI help and config-help tests
- run one smoke command for each affected entrypoint
- compare resolved model / dataset names before and after refactor

### Pass 3: Split oversized config logic into small validators

Current behavior:

- `src/core/config.py` owns many unrelated validation concerns in one long path

Structural improvement:

- split validation into small helpers by concern
- keep the public loader API stable
- make each validation branch easy to inspect in isolation

Validation check:

- load all active experiment configs
- run strict config validation tests, including duplicate-key rejection
- run any test that checks normalization of three-stage and benchmark configs

### Pass 4: Reduce duplication in trainer and evaluator metrics

Current behavior:

- threshold selection and pointwise metric assembly are spread across trainer and evaluator

Structural improvement:

- extract shared helpers for threshold resolution, metric naming, and checkpoint metadata
- keep the pointwise metric definitions unchanged
- keep trainer and evaluator semantics aligned

Validation check:

- compare metric dictionaries against the current test fixtures
- verify checkpoint metadata tests still pass
- run evaluator thresholding and VUS-PR tests

### Pass 5: Make model owner files easier to scan

Current behavior:

- `src/models/thesis_multitask.py` contains many dataclasses, helpers, and model paths in one large file

Structural improvement:

- group the file into clear sections with one responsibility per helper
- remove stale branches and unused internal helpers
- keep the model public API stable

Validation check:

- run model shape tests
- run one forward/backward step test
- run checkpoint roundtrip tests
- run gradient profiling tests that cover the active thesis path

### Pass 6: Tighten docs and parity checks

Current behavior:

- helper docs and example paths can lag behind the canonical runtime surface

Structural improvement:

- update docs and helper text to point at canonical paths
- keep legacy mentions only where compatibility is intentionally preserved
- add a short parity checklist for future refactors

Validation check:

- re-read the updated docs against the active configs
- run the smoke and config-load checks from the docs

## What Must Stay Stable

- public APIs that are already used by active tests and experiment scripts
- tensor contracts
- config field names that are already part of the active benchmark flow
- checkpoint format and reload behavior
- current evaluation semantics

## ASCII Architecture

### Before

```text
                +----------------------+
                |    scripts/train.py  |
                +----------+-----------+
                           |
                           v
                    +-------------+
                    | config.py   |
                    | large mixed |
                    | validation  |
                    +------+------+ 
                           |
            +--------------+--------------+
            |                             |
            v                             v
 +----------------------+      +--------------------------+
 | thesis_multitask.py  |      | redlamp_baseline.py     |
 | large mixed model    |      | + compatibility shim    |
 +----------+-----------+      +------------+-------------+
            |                               |
            v                               v
     +-----------+                    +-------------+
     | trainer   |<------------------>| evaluator   |
     | + metrics |    duplicated      | + metrics   |
     +-----------+                    +-------------+
```

### After

```text
                 +---------------------+
                 | public entrypoints  |
                 +----------+----------+
                            |
                            v
                  +-------------------+
                  | small resolvers   |
                  | + shared helpers  |
                  +---------+---------+
                            |
         +------------------+------------------+
         |                                     |
         v                                     v
 +-------------------+              +------------------------+
 | config validators  |              | model owner file(s)    |
 | by concern         |              | clear sections         |
 +---------+---------+              +-----------+------------+
           |                                     |
           v                                     v
    +--------------+                    +--------------------+
    | trainer step  |<------------------>| evaluator metrics  |
    | orchestration |    shared helpers  | and thresholds     |
    +--------------+                    +--------------------+
```

### Current

```text
      thesis_multitask.py
             |
             +--> thesis_multitask_components.py
             +--> thesis_multitask_setup_mixin.py
             +--> thesis_multitask_state_mixin.py
             +--> thesis_multitask_routing_mixin.py
             +--> thesis_multitask_loss_mixin.py

      tests/test_config_loading.py
             +--> tests/test_config_loading_additional.py

      tests/test_learning_rate_scheduler.py
             +--> tests/test_learning_rate_scheduler_additional.py
```

## Parity Rule

Any refactor pass is only acceptable if it preserves the current behavior for:

- active experiment configs
- canonical model names
- checkpoint save/load
- evaluation metrics
- smoke training and evaluation paths

If a change would alter one of those, it must be split into a separate migration task.

## Refactor Exit Criteria

The refactor is considered complete only when all of the following are true:

1. The active benchmark and thesis experiment families documented under `documents/` still load and run with the same public config entrypoints.
2. Canonical runtime helpers exist for the shared surfaces that were duplicated before, and compatibility shims are limited to explicit legacy boundaries.
3. `src/core/config.py` reads as an orchestration file, not a monolithic wall of validation logic.
4. `src/models/thesis_multitask.py` keeps the one-file ownership rule but is easier to scan from top to bottom.
5. Shared runtime behavior that was duplicated across `trainer` and `evaluator` stays routed through one helper boundary.
6. The architecture note contains the current ASCII diagrams, the pass list, and the verification commands that passed.
7. A focused verification bundle covering config loading, registry, thresholding, checkpoint fallback, and the active multitask paths passes green.
8. Any remaining large surface still in the repo is justified by ownership, not by leftover migration debt.
9. Every Python code file under `src/`, `scripts/`, and `tests/` stays below 1000 lines.

## Verified So Far

The following simplifications have already been applied and verified:

- shared runtime registration was extracted into `src/core/runtime_components.py`
- CLI help now points to the canonical `redlamp_baseline` experiment path
- `src/core/config.py` now shares small validation helpers for non-negative integers and booleans
- `src/core/config.py` now separates top-level, data, optimizer, and logging validation
- `src/models/thesis_multitask.py` now uses a shared two-stage phase-name set for the new flow
- threshold resolution and checkpoint threshold metadata now live in `src/engine/thresholding.py`

Verification commands that passed:

- `pytest -q tests/test_registry.py`
- `pytest -q tests/test_config_loading.py`
- `pytest -q tests/test_offline_pretraining_two_stage_config_loading.py tests/test_offline_pretraining_two_stage_runner.py`
- `pytest -q tests/test_thesis_multitask_config_refactor.py tests/test_multitask_shapes.py tests/test_one_multitask_train_step.py`
- `pytest -q tests/test_thresholding_helpers.py tests/test_evaluator_thresholding.py tests/test_trainer_checkpoint_fallback.py`
- `pytest -q tests/test_config_loading.py tests/test_offline_pretraining_two_stage_config_loading.py tests/test_offline_pretraining_two_stage_runner.py tests/test_thresholding_helpers.py tests/test_evaluator_thresholding.py tests/test_trainer_checkpoint_fallback.py`
- `pytest -q tests/test_config_loading.py tests/test_config_loading_additional.py tests/test_multitask_shapes.py tests/test_one_multitask_train_step.py`
- `pytest -q tests/test_learning_rate_scheduler.py tests/test_learning_rate_scheduler_additional.py tests/test_checkpoint_roundtrip.py tests/test_offline_pretraining_two_stage_runner.py tests/test_thesis_multitask_gradient_profiling.py tests/test_multitask_memory_initialization.py`
