---
date: 2026-06-29 23:16:54 +07 +0700
researcher: Codex
git_commit: e9380c0396ab57b7b6d564593cde701cb8773d06
branch: dev
repository: bachelor-thesis-2026
topic: "Implementation plan for simplifying the active benchmark runtime and removing legacy validation surfaces"
tags: [plan, time-series, anomaly-detection, benchmark, simplification]
status: draft
last_updated: 2026-06-29
last_updated_by: Codex
---

# Plan: Implementation plan for simplifying the active benchmark runtime and removing legacy validation surfaces

**Date**: 2026-06-29 23:16:54 +07 +0700  
**Researcher**: Codex  
**Git Commit**: `e9380c0396ab57b7b6d564593cde701cb8773d06`  
**Branch**: `dev`

## Research Basis

This plan is grounded in the current repository state documented in:

- [research-codebase-simplification-hotspots.md](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/documents/logs/06-29-2026/research/research-codebase-simplification-hotspots.md)
- `documents/design/idea.md`
- `documents/design/design_starter.md`
- `codebase_preferences.md`

The active benchmark intent is already visible in the new SMD benchmark configurations:

- `epochs = 100`
- `train_stride = 10`
- `val_stride = 1`
- `test_stride = 1`
- `checkpoint_monitor_metric = val_synth_vus_pr`
- `task.val_realistic = false`

The main problem is no longer the benchmark contract itself. The main problem is that the active benchmark runtime still coexists with a broad legacy surface made of:

- `val_realistic` runtime branches
- `comparative` naming across scripts, tests, manifests, and configs
- legacy balancing aliases
- documentation and tests that still treat the old surface as primary

## Current State

- The data layer already follows a stable batch contract with `x`, `point_labels`, `mask`, `timestamps`, and `meta`.
- The evaluation path already reconstructs pointwise timelines from overlapping windows and supports the active benchmark split-specific stride policy.
- The active benchmark task path is effectively `train -> val -> val_synth -> test`.
- The trainer, config validator, model classes, scheduler monitor logic, and test suite still treat `val_realistic` as a first-class runtime branch.
- The launcher and comparative orchestration layer already point their main configs at the new benchmark configs, but the script and artifact vocabulary still say `comparative`.
- The active baseline already supports multiple encoder families through `encoder_family`, so the name `RedLampMLPBaseline` is now misleading and no longer matches the intended encoder-agnostic direction of the codebase.
- The benchmark path must preserve the existing model files and the existing dataset builder architecture. The objective is simplification by subtraction, not redesign by expansion.

## Stable Contracts That Must Be Preserved

### Batch Contract

The active runtime should continue to accept batches of the form:

```python
batch = {
    "x": Tensor[B, L, D],
    "point_labels": Optional[Tensor[B, L]],
    "mask": Optional[Tensor[B, L, D]],
    "timestamps": Optional[Tensor[B, L]],
    "meta": list[dict[str, object]],
}
```

This contract is already enforced by:

- `src/core/contracts.py`
- `src/data/collate.py`
- `src/data/loaders.py`

No simplification step should weaken or bypass this contract.

### Encoder Contract

The thesis-facing hidden-state contract must remain stable:

```python
hidden: Tensor[B, L, H]
```

This is required by:

- `documents/design/idea.md`
- `documents/design/design_starter.md`
- `src/models/redlamp_mlp_baseline.py`
- `src/models/thesis_multitask.py`

The simplification work must not alter model tensor contracts.

### Model Output Contract

The active runtime should continue to rely on:

- `recon`
- `logits`
- `point_scores`
- `window_scores`
- `aux`

The evaluator and trainer already depend on this output surface. Simplification should remove legacy stage branches, not change the model output schema.

## Design Options

### Option A: Hard removal of legacy runtime surfaces

Remove `val_realistic` immediately from:

- `src/core/config.py`
- `src/engine/trainer.py`
- `scripts/train.py`
- `src/models/redlamp_mlp_baseline.py`
- `src/models/thesis_multitask.py`
- related tests and configs

Advantages:

- Produces the smallest active runtime.
- Eliminates the most confusing branch immediately.
- Aligns best with the “least amount of codepaths” principle.

Disadvantages:

- Causes broad test and config churn in one pass.
- Removes easy replay of historical experiments unless archived carefully.
- Increases short-term change risk right before benchmark runs.

### Option B: Soft deprecation with warnings and legacy compatibility

Keep `val_realistic` codepaths but:

- make benchmark path the documented default
- add comments or warnings
- keep all runtime paths functional

Advantages:

- Lowest immediate breakage risk.
- Historical runs remain directly reproducible without config migration.

Disadvantages:

- Keeps the confusing branch alive.
- Does not materially reduce codepath count.
- Fails the user’s simplification goal in practice.

### Option C: Hybrid simplification with active-runtime cleanup and legacy quarantine

Define one active runtime and one historical surface:

- Active runtime: `train`, `val`, `val_synth`, `test`
- Historical surface: archived `comparative` and `val_realistic` assets kept only where needed for historical inspection

Under this option:

- remove `val_realistic` from active config validation, trainer stage logic, scheduler monitor logic, and active tests
- stop using `comparative` terminology in active benchmark launch and reports
- keep historical configs or notes only if they are outside the active runtime path

Advantages:

- Matches the user’s benchmark intent closely.
- Reduces active complexity without rewriting the whole repository.
- Preserves reproducibility better than a blind hard delete.

Disadvantages:

- Requires a clean boundary between active and legacy assets.
- Some historical tests and scripts must be retired or moved.

## Recommended Approach

The recommended approach is **Option C: Hybrid simplification with active-runtime cleanup and legacy quarantine**.

This is the most practical choice for the current thesis stage because it balances four constraints:

1. The active benchmark runtime must become easier to trust quickly.
2. The codebase must not grow additional abstraction layers.
3. Historical experiment traces should not be mixed into the active path.
4. The repository is close to real benchmark execution, so unnecessary churn should be avoided.

## Implementation Scope

### Scope In

- Remove `val_realistic` from the active offline benchmark runtime.
- Align active launch, manifests, and tests with `benchmark` vocabulary.
- Reduce legacy balancing aliases where they are not needed anymore.
- Simplify logging noise in hot data paths.
- Update active documentation and config help.
- Evaluate and likely adopt an encoder-agnostic baseline name, with `RedLampBaseline` as the simplest recommended canonical name.

### Scope Out

- No redesign of the data loader architecture.
- No change to tensor contracts.
- No change to scoring formulas or benchmark metrics.
- No change to the SMD benchmark entity list, stride contract, or 100-epoch budget.
- No addition of new dataset abstractions.

## Additional Naming Simplification

The active baseline naming surface should be made encoder-agnostic.

Today the class name, module name, registry name, config file names, and many experiment identifiers still say `redlamp_mlp_baseline`, even though the implementation already supports at least:

- `encoder_family: mlp`
- `encoder_family: cnn_simple`

This creates an avoidable source of confusion. A reader can easily think the baseline is permanently tied to MLP, while the code already allows other encoders.

Recommended direction:

- canonical class name: `RedLampBaseline`
- canonical registry name: `redlamp_baseline`
- canonical model config names should also drop `mlp` where the file is meant to describe the general baseline rather than one specific encoder instantiation

Recommended migration timing:

- do this after the `val_realistic` cleanup and config-surface reduction
- do it before the final benchmark-launch surface is frozen for the thesis runs

This keeps the rename auditable while avoiding a long-lived misleading public API.

## Planned File-Level Changes

### 1. Remove `val_realistic` from active runtime configuration

Modify:

- `src/core/config.py`
- `src/core/config_help.py`

Actions:

- Remove `val_realistic`, `val_realistic_source`, and `val_anomaly_rate_override` from the active `multitask_tsad` task keys.
- Remove `val_realistic_*` from allowed checkpoint monitor metrics.
- Remove `val_realistic_*` from allowed scheduler monitor metrics.
- Restrict active classification diagnostic stages to `train`, `val`, `val_synth`, and `test` if synthetic validation diagnostics are still desired.
- Update CLI help text so the synthetic validation contract is explained directly.

Expected result:

- Config validation will fail loudly if a new active benchmark config tries to reintroduce `val_realistic`.

### 2. Remove `val_realistic` from the trainer

Modify:

- `src/engine/trainer.py`
- `scripts/train.py`

Actions:

- Delete `_resolve_realistic_validation_anomaly_rate`.
- Remove `use_val_realistic` branching.
- Collapse auxiliary validation logic to a single synthetic validation branch:
  - `prepare_synthetic_validation_epoch`
  - `synthetic_validation_step`
  - `stage_name = "val_synth"`
- Remove `val_realistic_*` metric mode maps from checkpoint and scheduler logic.
- Rename any remaining threshold or reconstruction-diagnostic surfaces to match `val_synth`.

Expected result:

- The trainer will expose one clean auxiliary validation path only.

### 3. Remove `val_realistic` from models

Modify:

- `src/models/redlamp_mlp_baseline.py`
- `src/models/thesis_multitask.py`

Actions for `redlamp_mlp_baseline.py`:

- Remove `prepare_realistic_validation_epoch`.
- Remove `realistic_validation_step`.
- Restrict synthetic validation preparation to `prepare_synthetic_validation_epoch`.
- Restrict valid synthetic-validation stage names to `val_synth`.

Actions for `thesis_multitask.py`:

- Remove `SyntheticAnomalyConfig.val_realistic`.
- Remove `SyntheticAnomalyConfig.val_realistic_source`.
- Remove `SyntheticAnomalyConfig.val_anomaly_rate_override`.
- Remove matching flat kwargs and model attributes.
- Replace the current wrapper relationship with a direct synthetic-validation method.
- Restrict stage handling and contrastive pair preparation to `train`, `val_synth`, and any genuinely needed clean stage.

Expected result:

- Both offline models will expose the same active stage surface.

### 4. Simplify synthetic balancing API without over-expanding churn

Modify:

- `src/data/augment.py`
- `src/models/redlamp_mlp_baseline.py`
- `src/models/thesis_multitask.py`
- active task YAMLs
- related tests

Recommended design choice:

- Keep `train_balance_classes` as the active field for the immediate implementation.
- Remove or retire legacy aliases `balance_binary_classes_within_batch` and `balance_classes_within_batch` from active runtime paths if they are no longer needed by current configs.
- Explain clearly in code comments and config help that the field controls synthetic class balancing for both training and synthetic validation injectors.

Rationale:

- Renaming the canonical field now would cause wide churn without enough practical gain before benchmark execution.
- Removing extra aliases still simplifies the public surface materially.

### 5. Clean orchestration naming

Modify:

- `scripts/launch_tmux_comparative_smd_experiment.sh`
- `scripts/run_comparative_smd_experiments.py`
- `scripts/preflight_comparative_smd_server.py`
- related tests

Recommended design choice:

- Promote `benchmark` naming to the active orchestration vocabulary.

Possible implementation pattern:

- Either rename the active scripts and tests to `benchmark_*`, or keep file names temporarily but change:
  - manifest names
  - report names
  - status strings
  - help text
  - comments

The simpler long-term state is a full rename. The safer short-term state is to keep filenames but align all runtime-facing strings and artifacts with `benchmark`.

### 6. Reduce logging noise in hot paths

Modify:

- `src/data/collate.py`

Actions:

- Remove per-batch `console_print` calls from hot collation unless a debug flag is explicitly enabled.

Expected result:

- Cleaner logs during long benchmark runs.

### 7. Update documentation and historical boundaries

Modify:

- `documents/design/experiment_config_organization_guideline.md`
- any active notes that still describe `val_realistic` as the primary validation namespace

Actions:

- Mark `comparative` and `val_realistic` as historical if retained.
- State clearly that the active benchmark path uses `val_synth`.
- Keep historical references only where they explain old experiments, not where they define active runtime behavior.

## Test Plan

### Priority 1: Active runtime protection tests

Add or update tests to ensure:

- active benchmark configs no longer accept `val_realistic`
- active benchmark configs no longer accept `val_realistic_*` checkpoint or scheduler monitors
- the trainer always logs `val_synth_*` and never `val_realistic_*`
- both offline models expose synthetic validation only through `val_synth`

Target files:

- `tests/test_config_loading.py`
- `tests/test_multitask_validation_alignment.py`
- `tests/test_learning_rate_scheduler.py`
- `tests/test_redlamp_mlp_baseline.py`
- `tests/test_thesis_multitask_config_refactor.py`

### Priority 2: Legacy quarantine tests

If historical assets are kept, add narrow tests that verify:

- archived or legacy configs are not accidentally treated as active benchmark configs
- old scripts are either removed from active CI or clearly marked legacy

### Priority 3: Regression tests for unchanged behavior

Preserve or add tests for:

- current batch contract
- current window reconstruction behavior
- current benchmark config loading
- current checkpoint save/load behavior
- current synthetic anomaly injector deterministic behavior with fixed seeds

## Validation Procedures

After implementation, run at least:

1. Targeted config and validation tests
2. Synthetic validation alignment tests
3. Scheduler monitor tests
4. One offline train-step smoke for baseline
5. One offline train-step smoke for thesis multitask
6. Benchmark config load validation over all active SMD benchmark configs
7. Launcher dry-run and preflight checks for the active benchmark launcher

Recommended commands will be decided in the detail phase, but the validation philosophy is:

- fail fast on config drift
- verify the benchmark launcher still resolves all active configs
- verify metric namespaces are clean and unambiguous

## Risk and Mitigation

### Risk 1: Breaking historical config replay

Mitigation:

- quarantine legacy assets explicitly instead of silently reusing them in the active runtime
- keep notes or archived configs if needed for thesis traceability

### Risk 2: Removing a branch that some tests still rely on indirectly

Mitigation:

- rewrite the tests first or in parallel with runtime removal
- run focused suites before any broad smoke pass

### Risk 3: Over-renaming surfaces right before experiments

Mitigation:

- prioritize semantic cleanup in runtime behavior first
- apply file renames only where they clearly reduce confusion without creating extra wrappers

### Risk 4: Confusing canonical and historical benchmark vocabulary

Mitigation:

- define one active vocabulary: `benchmark`
- define one active auxiliary validation namespace: `val_synth`

### Risk 5: Introducing new abstraction instead of removing complexity

Mitigation:

- do not add new framework layers
- work by deleting branches, deleting keys, and tightening interfaces

## Minimal Vertical Slice for This Work

Before any broader cleanup, the first completed vertical slice should be:

1. Active benchmark config loads without any `val_realistic` fields.
2. Trainer runs `train -> val -> val_synth`.
3. Baseline model supports `val_synth` only.
4. Thesis multitask model supports `val_synth` only.
5. Active benchmark tests pass.
6. Active launcher dry-run still works.

Only after this slice is stable should the implementation continue into:

- orchestration renaming
- documentation cleanup
- optional alias removal beyond the active path

## Open Questions

1. Should the old `comparative` scripts and configs be kept as archived historical assets inside the repository, or removed entirely from the active working tree?
2. Should the active launcher file itself be renamed now to a `benchmark` name, or should only its runtime-facing strings and artifact names be updated first?
3. Should the legacy balancing aliases be removed immediately, or should they remain temporarily until after the benchmark runs are complete?

## Recommended Decisions

The most pragmatic decisions for the next implementation step are:

1. Remove `val_realistic` completely from the active runtime now.
2. Keep historical `comparative` assets only if they are clearly quarantined from the active path.
3. Keep `train_balance_classes` as the canonical field for now, but remove extra alias surfaces where they are not required by active configs.
4. Align all active launcher outputs and reports with `benchmark` vocabulary.

## Which Approach Aligns Best

The approach that aligns best with the current thesis direction is:

**Option C with immediate active-runtime cleanup and minimal legacy quarantine.**

This gives the repository a smaller active benchmark surface, preserves current tensor and loader contracts, and avoids introducing new architecture or framework machinery immediately before benchmark execution.
