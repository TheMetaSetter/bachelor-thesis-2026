---
date: 2026-06-30 13:21:29 +0700
planner: Codex
git_commit: ddd20afb2f45c83a17fa93d54624789b783ca29d
branch: dev
repository: bachelor-thesis-2026
topic: "Preliminary implementation plan to simplify the active runtime surface and remove misleading legacy paths"
tags: [plan, readability, runtime, configs, baseline, benchmark]
status: draft
last_updated: 2026-06-30
last_updated_by: Codex
source_research: documents/logs/06-30-2026/research/research-code-paths-not-yet-simplified-and-easy-to-misunderstand.md
---

# Plan: Simplify the active runtime surface and remove misleading legacy paths

## Current State

- The active RedLamp baseline runtime has already been renamed conceptually to `RedLampBaseline` and `redlamp_baseline`, but backward-compatible aliases remain in multiple layers.
- The current code already preserves the important existing contracts:
  - batch contract remains centered on `{"x", "point_labels", "mask", "timestamps", "meta"}`
  - encoder contract remains sequence-first, especially for the thesis path with hidden states shaped `[B, L, H]`
  - model output and trainer interfaces already run through the current `scripts/train.py`, `scripts/evaluate.py`, and `src/engine/trainer.py` paths
- The current repository state must preserve:
  - the benchmark-safe loader path in `src/data/loaders.py`
  - the benchmark-safe anomaly archive parsing path in `src/data/datasets/anomaly_archive.py`
  - the current evaluator and audit behavior
  - the current one-model-one-file discipline for `src/models/redlamp_baseline.py` and `src/models/thesis_multitask.py`
- The main readability debt is transitional duplication, not a broken core architecture.

## Design Options

### Option A: Minimal compatibility-first cleanup

Keep all legacy file names and most legacy aliases, but clean only the most visible user-facing surfaces:

- help text in `src/core/config_help.py`
- launcher scripts such as `scripts/launch_tmux_comparative_smd_experiment.sh`
- active config internals and active test expectations
- one small helper extraction for realistic validation anomaly-rate estimation

Advantages:

- Lowest risk before benchmark execution.
- Smallest code churn.
- Fastest path to a less misleading active runtime.

Disadvantages:

- Legacy identity still remains deeply present.
- New readers may still encounter both old and new names in tests and compatibility modules.
- The cleanup would be visibly incomplete.

### Option B: Balanced simplification with one temporary legacy shim

Treat `redlamp_baseline` as the only canonical runtime identity, keep exactly one temporary compatibility shim, and simplify all active user-facing surfaces around that single identity.

This includes:

- keeping `src/models/redlamp_mlp_baseline.py` only as a short shim
- removing old naming from active docs, launcher scripts, config-help text, and most tests
- keeping legacy experiment file paths temporarily if needed, but making all runtime-loaded content canonical
- isolating SMD-specific realistic-validation anomaly-rate logic into a small helper module or a clearly named SMD-only helper path

Advantages:

- Best balance between simplicity and runtime safety.
- Greatly reduces mental-model duplication.
- Avoids a risky full-path rename immediately before experiments.
- Matches the repository preference for small extensions over disruptive rewrites.

Disadvantages:

- Some old file names may still exist temporarily.
- A small amount of compatibility debt remains until a later final cleanup pass.

### Option C: Full hard cleanup now

Delete the legacy shim and rename all remaining legacy file paths and config file names immediately.

Advantages:

- Cleanest final naming surface.
- Lowest long-term conceptual debt.

Disadvantages:

- Highest operational risk before experiments.
- Largest change set.
- Most likely to break auxiliary scripts, old notes, or external references.

## Recommended Approach

Option B aligns best with the current thesis situation.

The codebase is already in a transition state. The practical objective now is not aesthetic perfection. The practical objective is to reach one stable, easy-to-read active runtime surface before benchmark execution while keeping the number of moving parts low. Option B does that.

It keeps the current runtime stable, preserves reproducibility, and removes the most misleading duplication first. It also leaves one safe escape hatch for backward compatibility while the benchmark campaign is still close.

## Proposed Implementation Scope

### Scope 1: Canonicalize the active user-facing naming surface

Modify:

- `src/core/config_help.py`
- `scripts/launch_tmux_comparative_smd_experiment.sh`
- `scripts/run_comparative_smd_experiments.py`
- active tests that still teach the old baseline identity

Target behavior:

- `redlamp_baseline` becomes the only canonical name shown to a new reader in active runtime docs and launcher surfaces.
- The old identity is treated as compatibility-only, not as a co-equal public surface.

Important boundary:

- This scope should not redesign the trainer, evaluator, or loader contracts.

### Scope 2: Separate SMD-specific realistic-validation prior logic from the generic trainer surface

Modify:

- `src/engine/trainer.py`
- possibly `src/data/datasets/smd.py`
- possibly one new helper module, for example `src/data/realistic_validation.py` or a similarly small file
- `src/core/config.py`
- `src/models/thesis_multitask.py`

Target behavior:

- Generic trainer code should not carry SMD-specific naming such as `test_smd_all` more than absolutely necessary.
- SMD-specific anomaly-rate estimation should be clearly isolated.
- Non-SMD datasets should follow an explicit fallback or explicit override path rather than implicit SMD assumptions.

Important boundary:

- Do not redesign realistic validation itself in this step.
- Only simplify responsibility boundaries and naming semantics.

### Scope 3: Reduce legacy-test mental-model duplication

Modify:

- tests that still import `RedLampMLPBaseline` from the compatibility shim
- tests that still teach `model_name="redlamp_mlp_baseline"` when they are not specifically testing compatibility

Representative files:

- `tests/test_cnn_encoder_config_loading.py`
- `tests/test_redlamp_gradient_conflict_metrics.py`
- `tests/test_redlamp_cnn_baseline_shapes.py`
- `tests/test_one_redlamp_mlp_train_step.py`
- `tests/test_redlamp_cnn_rerun_configs.py`
- `tests/test_comparative_runner.py`
- `tests/test_comparative_preflight.py`

Target behavior:

- Tests should distinguish clearly between:
  - canonical runtime behavior
  - legacy compatibility behavior

Important boundary:

- Keep a very small number of explicit compatibility tests.
- Do not preserve the old naming model everywhere.

### Scope 4: Clarify actual active dataset support

Modify:

- `src/core/config.py`
- possibly `src/data/api.py`
- possibly one small documentation note or config-help text

Target behavior:

- The codebase should clearly communicate that the currently validated runtime supports only the active dataset families that are truly runnable now.
- The presence of directories such as `data/SWaT`, `data/IOPS`, `data/NASA`, and `data/ibm-cloud-console-anomaly-dataset-iccad` should not be implicitly interpreted as active runtime support.

Important boundary:

- This scope should not onboard new datasets yet.
- It should only make the active support boundary easier to understand.

## File-Level Programming Plan

### `src/core/config.py`

Role in this plan:

- Remains the single validation entry point for experiment configs.

Changes to plan:

- Preserve the current validator architecture for now.
- Reduce misleading public semantics by narrowing or explicitly labeling compatibility-only names.
- Keep `supported_model_names` and task semantics simple and explicit.
- Remove or isolate generic acceptance of names that are no longer meant to be public-first identities.

Reason:

- This file is already the semantic choke point. The correct short-term move is to simplify what it accepts and how it names things, not to split it aggressively during benchmark preparation.

### `src/engine/trainer.py`

Role in this plan:

- Remains the generic epoch loop and validation orchestrator.

Changes to plan:

- Move the SMD-specific realistic-validation anomaly-rate computation decision out of the middle of the generic training loop path as much as possible.
- Keep one thin helper boundary so the trainer asks for an anomaly-rate estimate without embedding dataset-family semantics directly.

Reason:

- This improves separation of concerns without changing training behavior broadly.

### `src/models/redlamp_baseline.py`

Role in this plan:

- Stays the canonical active baseline model file.

Changes to plan:

- No architectural redesign is needed.
- Only maintain canonical imports and canonical references.

Reason:

- The model file itself is not the main simplification problem now.

### `src/models/redlamp_mlp_baseline.py`

Role in this plan:

- Compatibility shim only.

Changes to plan:

- Keep it extremely small if retained.
- Make it obvious that it is temporary and compatibility-only.

Reason:

- If the shim remains, it must not behave like a second real model file.

### `scripts/train.py`, `scripts/evaluate.py`, `scripts/run_online_adaptation.py`

Role in this plan:

- Entry points for active runtime construction.

Changes to plan:

- Keep their current explicit wiring style.
- Simplify naming and alias treatment where possible.
- Avoid expanding parallel alias logic.

Reason:

- These scripts are already readable. The problem is duplicated naming surface, not the basic pattern itself.

### `scripts/run_comparative_smd_experiments.py`

Role in this plan:

- Comparative-run launcher and family resolver.

Changes to plan:

- Simplify supported baseline naming so it reflects the canonical runtime identity more directly.
- Keep compatibility logic only if it is still required by real active configs.

Reason:

- Comparative runs are benchmark-facing and should not teach an outdated baseline identity.

## Contract Enforcement Plan

### Batch contract

The batch contract should remain enforced exactly where it is now:

- dataset building in `src/data/loaders.py`
- batch movement and logging in `src/engine/trainer.py`
- model consumption in `src/models/redlamp_baseline.py` and `src/models/thesis_multitask.py`

No batch-shape redesign is needed in this plan.

### Encoder contract

The encoder contract should remain:

- `RedLampBaseline` stays window-based and consistent with the current baseline path.
- `ThesisMultitaskModel` continues to expose thesis-facing sequence representations shaped `[B, L, H]`.

This plan should not change encoder tensor contracts.

### Model output contract

The model output contract should remain under the current model-owned stage-step logic.

This plan only reduces naming confusion and responsibility leakage. It should not alter:

- forward-path output semantics
- checkpoint-loading semantics
- evaluator-facing score reconstruction interfaces

## Risk and Mitigation

### Risk 1: Breaking benchmark execution shortly before experiments

Mitigation:

- Prefer compatibility-preserving simplification.
- Keep one shim if needed.
- Use targeted smoke runs through `scripts/train.py`, not only unit tests.

### Risk 2: Over-cleaning and deleting useful compatibility too early

Mitigation:

- Keep compatibility tests, but reduce them to a small explicit compatibility layer.
- Separate canonical tests from legacy tests.

### Risk 3: Spreading SMD-specific semantics into even more places

Mitigation:

- Move SMD-specific realistic-validation prior logic toward a clearly named helper boundary instead of adding more `if dataset_name == ...` logic across generic layers.

### Risk 4: Creating a large refactor under time pressure

Mitigation:

- Keep this plan as a narrow simplification pass.
- Do not restructure `src/models/thesis_multitask.py` in this step.
- Do not onboard new datasets in this step.

### Risk 5: Making documentation and launchers disagree again

Mitigation:

- Treat `src/core/config_help.py`, shell launchers, and active plan/benchmark scripts as first-class user-facing runtime surfaces.
- Verify them with grep-backed checks after edits.

## Test Plan

The test strategy should stay realistic and pressure-test runtime behavior.

### Canonical naming tests

Add or update tests that assert:

- canonical active configs resolve to `model_name == "redlamp_baseline"`
- active launcher/help examples use canonical baseline naming
- compatibility-only tests remain isolated

### Runtime-construction tests

Run targeted tests for:

- `scripts/train.py` model registration and construction
- `scripts/evaluate.py` model registration and construction
- comparative runner stage-family resolution

### Realistic-validation tests

Add or keep tests that assert:

- SMD-specific realistic anomaly-rate estimation works only on the intended path
- non-SMD datasets fall back cleanly without crashing

### Smoke validation

Run at least:

- one anomaly-archive smoke train command through `scripts/train.py`
- one SMD comparative smoke train command through `scripts/train.py`

This is important because the recent anomaly-archive realistic-validation bug was exposed only by real smoke execution.

## Validation Procedure

After implementation, validate in this order:

1. grep for legacy public-facing names in active runtime surfaces
2. run targeted pytest for baseline/config/runtime paths
3. run targeted pytest for realistic-validation and baseline behavior
4. run one or two smoke train commands through real CLI paths
5. review launcher and config-help text manually

## Minimal Vertical Slice

The minimal vertical slice for this simplification pass is:

1. one canonical baseline identity in active docs, launchers, and active runtime tests
2. one isolated SMD-specific realistic-validation helper path
3. one small explicit compatibility shim, if still needed
4. no changes to the core batch, encoder, or model-output contracts

This is the smallest useful slice that reduces confusion without increasing benchmark risk.

## Open Questions

1. Should physical experiment file names such as `configs/experiment/...redlamp_mlp_baseline...yaml` be renamed now, or should that be deferred until after the benchmark run?
2. Should the compatibility shim `src/models/redlamp_mlp_baseline.py` remain until the benchmark campaign finishes, or should it be removed in the next cleanup pass?
3. Should `test_smd_all` remain as an explicit SMD-only option in the task/config surface for now, or should it be removed from generic surfaces immediately and replaced by a more local mechanism?

## Recommendation

Proceed with Option B.

That means:

- simplify active user-facing naming first,
- isolate SMD-specific realistic-validation semantics second,
- preserve one temporary compatibility shim only if still operationally useful,
- and defer any fully destructive rename sweep until after the benchmark run.
