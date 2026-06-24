---
date: 2026-06-24 19:08:00 +0700
researcher: Codex
git_commit: c1c3065ee611bab9b0d5c1071e7a58f62b99d6c7
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed implementation plan for cleaning contradictory semantics and improving reader-facing clarity in the three-stage offline pretraining codepath"
tags: [plan, codebase-audit, semantics, clarity, three-stage]
status: complete
last_updated: 2026-06-24
last_updated_by: Codex
---

# Plan: Detailed implementation plan for cleaning contradictory semantics and improving reader-facing clarity in the three-stage offline pretraining codepath

**Date**: 2026-06-24 19:08:00 +0700
**Researcher**: Codex
**Git Commit**: `c1c3065ee611bab9b0d5c1071e7a58f62b99d6c7`
**Branch**: `dev`

## Planning Goal

The immediate goal is not to redesign the thesis pipeline. The goal is to remove contradictory defaults, reduce reader confusion, and make user-facing experiment semantics match the active implementation contract more closely. This plan therefore treats the current three-stage offline pretraining flow as the baseline contract and focuses on semantic cleanup, configuration alignment, and validation clarity.

The work must preserve the active runtime guarantees already established:

- exact `300`-epoch three-stage budget;
- canonical Stage 3 wording: `stage3_memory_initialization_and_fusion_warmup`;
- balanced `12`-class RedLamp multiclass synthetic labeling as the default intended behavior;
- clean `val` loss remaining separate from synthetic or realistic auxiliary validation;
- post-training evaluation remaining part of the three-stage runner.

## Current State

- The repository already has a stable experiment-driven runtime path through `src/core/config.py`, `scripts/train.py`, `src/engine/trainer.py`, `src/models/thesis_multitask.py`, `src/models/redlamp_mlp_baseline.py`, and `src/data/augment.py`.
- The active three-stage experiment config already enforces the exact `300`-epoch budget and uses the canonical Stage 3 name through `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml`.
- The synthetic anomaly runtime has already moved toward a default balanced RedLamp multiclass contract in `SyntheticAnomalyInjector`, `ThesisMultitaskModel`, and `RedLampMLPBaseline`.
- The trainer already separates `validation_step()`, `synthetic_validation_step()`, and `realistic_validation_step()`, but the naming of `val_realistic` still risks implying a separate validation split instead of a synthetic validation pass calibrated by test-window anomaly priors.
- The codebase still preserves several legacy or transitional semantics that are now readability liabilities:
  - config-layer fallback defaults still imply binary or unbalanced behavior;
  - `anomaly_probability` remains prominent in task YAMLs even when balanced multiclass overrides its practical role during training;
  - legacy Stage 3 alias support is still reflected back into normalized configs and tests;
  - one baseline argument name still refers to binary balancing while the active path is multiclass.

## Design Options

### Option A: Minimal semantic cleanup with no runtime behavior changes

This option changes comments, config-help text, docstrings, and research notes only. It does not modify defaults or field names in executable code.

This option is low-risk but inadequate. It leaves contradictory fallback behavior in `src/core/config.py`, which means future configs may silently encode outdated semantics.

### Option B: Semantic alignment cleanup with backward-compatible runtime preservation

This option updates configuration fallbacks, clarifies naming, improves comments and validation messages, and narrows legacy alias visibility while preserving current experiment behavior and backward compatibility where practical.

This is the preferred option. It addresses the main confusion sources without destabilizing the working pipeline.

### Option C: Full API cleanup with aggressive deprecation and breaking renames

This option would remove legacy alias support immediately, rename several public arguments, and potentially remove fields such as `anomaly_probability` from active task YAMLs.

This option is too aggressive for the current repo state. It increases short-term breakage risk across tests, historical configs, and research notes.

## Recommended Approach

Option B aligns best with the current repository state and the thesis workflow. It preserves the active experimental path while making the code easier to read, easier to configure correctly, and less likely to drift back toward obsolete semantics.

## Risk and Mitigation

- Risk: changing config fallbacks may unintentionally affect older experiments that relied on omitted fields.
  Mitigation: update config-loading tests first, then apply fallback changes, then run targeted config and model-construction tests across both baseline and thesis experiment families.

- Risk: reducing the visibility of Stage 3 legacy aliases may break historical YAMLs or tests.
  Mitigation: keep legacy input acceptance for now, but stop treating the legacy alias as a first-class normalized output where possible; adjust tests to assert canonical behavior.

- Risk: renaming user-facing parameters such as `balance_binary_classes_within_batch` may break instantiation call sites.
  Mitigation: preserve the old parameter temporarily as a compatibility alias, but introduce a clearer canonical parameter name and document precedence explicitly.

- Risk: clarifying `val_realistic` semantics without changing runtime may still leave users confused if naming remains too broad.
  Mitigation: add narrow docstrings, trainer comments, config-help text, and a focused plan to optionally rename the preparation hook in a later cleanup.

- Risk: removing or downplaying `anomaly_probability` may obscure its continued role in unbalanced or realistic validation flows.
  Mitigation: retain the field, but constrain its documentation precisely: it controls Bernoulli anomaly injection only when class balancing is disabled and controls auxiliary realistic-validation prior when explicitly passed into the validation injector.

## Open Questions

- Should the repository preserve `anomaly_probability` as a visible task-level field in balanced multiclass experiment YAMLs, or should it be moved into an advanced or commented section to signal its reduced training role?
- Should `prepare_realistic_validation_epoch(anomaly_probability)` be renamed in this implementation round, or should the first pass be limited to docstrings and comments to minimize churn?
- Should legacy Stage 3 alias normalization continue to populate both keys in-memory, or should canonical-only normalized output become the new standard immediately?

## Implementation Workstreams

### Workstream 1: Align config-layer fallbacks with active balanced multiclass defaults

**Objective**

Remove the contradiction between config validation defaults and model or injector defaults.

**Files to modify**

- `src/core/config.py`
- `tests/test_config_loading.py`
- `tests/test_smd_machine_3_4_three_stage_config_loading.py`
- optionally `tests/test_redlamp_aligned_configs.py`

**Required changes**

1. Change `task_config.get("train_balance_classes", False)` to a fallback consistent with the active intended contract.
2. Change `task_config.get("classification_label_mode", "binary")` to a fallback consistent with the active intended contract for multitask RedLamp-style experiments.
3. Review whether these new fallbacks should apply only under `task_name == "multitask_tsad"` or more narrowly under explicit multiclass-capable experiments. The narrowest stable scope should be preferred if older binary experiments still exist.
4. Update validation error messages so they describe the true active contract more clearly when `num_classes != 12`.

**Interface impact**

- Batch contract remains unchanged.
- Encoder contract remains unchanged.
- Model output contract remains unchanged.
- Only config interpretation and validation semantics change.

**Tests**

- Add tests showing omitted `classification_label_mode` and omitted `train_balance_classes` resolve to the intended modern defaults for multitask RedLamp-aligned configs.
- Keep or add a test proving explicitly binary experiments still validate correctly when they opt in intentionally.

### Workstream 2: Clarify the role of `anomaly_probability` under balanced 12-class training

**Objective**

Prevent users from misreading `anomaly_probability: 0.5` as “balanced 12 classes.”

**Files to modify**

- `src/data/augment.py`
- `configs/task/multitask_tsad_redlamp_multiclass_window20.yaml`
- `configs/task/multitask_tsad.yaml`
- `src/models/thesis_multitask.py`
- `src/models/redlamp_mlp_baseline.py`
- `tests/test_synthetic_anomaly_injection.py`
- `tests/test_multitask_shapes.py`
- `tests/test_redlamp_mlp_baseline.py`

**Required changes**

1. Add explicit comments and docstrings in `SyntheticAnomalyInjector` explaining the branch split:
   - `train_balance_classes=True` means balanced quotas over the active class taxonomy;
   - `train_balance_classes=False` means `anomaly_probability` drives Bernoulli anomaly injection.
2. In active task YAMLs, add a short comment directly above `anomaly_probability` clarifying that it is not the balancing mechanism when `train_balance_classes: true`.
3. In `ThesisMultitaskModel.from_flat_kwargs`, add symmetric default clarification for the `num_classes == 12` path so the flat-kwargs path does not feel implicitly binary-first.
4. In `RedLampMLPBaseline`, replace or supplement `balance_binary_classes_within_batch` with a clearer canonical argument name, such as `balance_classes_within_batch`, while keeping the old name as a compatibility alias for now.

**Interface impact**

- Batch contract remains unchanged.
- Synthetic label generation contract becomes more explicit, not functionally larger.
- Constructor readability improves.

**Tests**

- Keep existing behavioral tests for balanced multiclass sampling.
- Add constructor-level tests for the new canonical argument name if it is introduced.
- Add one regression test proving the compatibility alias still works if that alias is preserved.

### Workstream 3: Make `val_realistic` semantics explicit and user-readable

**Objective**

Ensure that users understand that `val_realistic` is an auxiliary synthetic validation pass calibrated by test anomaly priors, not a separate validation split.

**Files to modify**

- `src/engine/trainer.py`
- `src/models/thesis_multitask.py`
- `src/models/redlamp_mlp_baseline.py`
- `src/core/config.py`
- `documents/logs/06-24-2026/research/research-codebase-audit-three-stage-semantics-and-user-facing-clarity.md`
- optionally `scripts/train.py` config help text if exposed there
- `tests/test_multitask_validation_alignment.py`
- `tests/test_redlamp_realistic_validation_alignment.py`

**Required changes**

1. Add precise docstrings to `prepare_realistic_validation_epoch()` in both models, stating that the method only re-parameterizes the synthetic validation injector for the upcoming auxiliary validation epoch.
2. Add an explanatory trainer comment near the `val_realistic` branch in `src/engine/trainer.py` stating that the same `val_loader` is reused and only the synthetic injection prior changes.
3. Improve config validation or help text for `val_realistic_source` so that `test_same_scope` and `test_smd_all` are described as sources for anomaly-rate estimation, not alternative validation loaders.
4. Decide whether to keep the method name `prepare_realistic_validation_epoch()` for compatibility in this round. The preferred first pass is to keep the method name but tighten its documentation.

**Interface impact**

- Batch contract remains unchanged.
- Validation-step contract remains unchanged.
- Trainer loop mechanics remain unchanged.

**Tests**

- Keep the current validation alignment tests.
- Add or strengthen tests that assert `val_realistic` continues to use the validation loader while producing separate auxiliary metrics.

### Workstream 4: Reduce Stage 3 legacy wording as a reader-facing concept

**Objective**

Keep backward compatibility for historical configs while making the canonical Stage 3 name the only reader-facing primary contract.

**Files to modify**

- `src/core/config.py`
- `scripts/run_three_stage_offline_pretraining.py`
- `src/models/thesis_multitask.py`
- `tests/test_smd_machine_3_4_three_stage_config_loading.py`
- `tests/test_three_stage_orchestration_smoke.py`
- `documents/logs/06-17-2026/detail/detail-offline-pretraining-three-stage-discussion-context.md`
- any other `documents/` note that still presents the legacy Stage 3 name as current truth

**Required changes**

1. Preserve acceptance of `stage3_prototype_warmup_epochs` as an input alias, but stop amplifying it as an equally primary normalized field if the surrounding code allows that cleanup safely.
2. Update tests so the canonical field is the asserted public contract. Legacy alias tests should move from “canonical and legacy both appear” toward “legacy input is still accepted and normalized into canonical semantics.”
3. Review `scripts/run_three_stage_offline_pretraining.py` and `src/models/thesis_multitask.py` for user-facing comments or metadata still mentioning the legacy label.
4. Update the existing detailed discussion note so the surviving wording no longer presents `stage3_prototype_warmup` as the preferred logging contract.

**Interface impact**

- Three-stage runner phase order remains unchanged.
- Canonical Stage 3 label becomes clearer to users.
- Backward compatibility for old YAML inputs can remain intact.

**Tests**

- Maintain rejection of conflicting alias values.
- Add or update tests so canonical-only configs remain the normative success case.

## Execution Order

The implementation should proceed in the following order:

1. Update tests for config-default semantics and legacy Stage 3 alias expectations.
2. Apply config fallback alignment in `src/core/config.py`.
3. Apply synthetic anomaly semantics clarification in `src/data/augment.py`, `src/models/thesis_multitask.py`, `src/models/redlamp_mlp_baseline.py`, and the active task YAMLs.
4. Apply `val_realistic` docstring and trainer-comment clarification.
5. Apply Stage 3 wording cleanup across tests and `documents/`.
6. Re-run targeted test suites, then broader regression suites for config loading, validation alignment, three-stage orchestration, and synthetic anomaly injection.

This order follows a minimal-risk path. It starts with the highest-leverage semantic contradiction in the config layer, then moves outward to user-facing names and documentation.

## Test Plan

At minimum, the following targeted test commands should pass after implementation:

```bash
pytest -q tests/test_config_loading.py tests/test_smd_machine_3_4_three_stage_config_loading.py
pytest -q tests/test_synthetic_anomaly_injection.py tests/test_multitask_shapes.py tests/test_redlamp_mlp_baseline.py
pytest -q tests/test_multitask_validation_alignment.py tests/test_redlamp_realistic_validation_alignment.py
pytest -q tests/test_three_stage_orchestration_smoke.py tests/test_three_stage_phase_runtime.py
```

If the Stage 3 alias normalization behavior changes materially, additional targeted tests around `load_experiment_config()` and `validate_experiment_config()` must be added before claiming completion.

## Validation Procedures

Validation should be performed at three levels.

First, perform static validation:

- load the active three-stage experiment config;
- load the active task configs;
- verify no duplicate or unknown keys are introduced;
- verify canonical Stage 3 keys remain accepted and visible.

Second, perform behavioral validation:

- confirm balanced multiclass defaults are preserved when fields are omitted where intended;
- confirm explicit binary opt-in still works when requested;
- confirm `val_loss` remains clean-only and `val_realistic_*` remains auxiliary;
- confirm the three-stage runner still materializes the same `300`-epoch plan.

Third, perform user-facing validation:

- inspect config YAML comments for clarity;
- inspect error messages for misleading binary-first wording;
- inspect the research and detail documents for stale Stage 3 terminology.

## Scope Boundaries

This implementation plan does not include:

- redesigning the multitask architecture;
- changing the exact stage epoch allocation;
- changing `lambda_recon` or `lambda_cls`;
- altering clean `val_loss` semantics;
- replacing the current `val_realistic` mechanism with a separate validation dataset;
- removing backward compatibility for old YAMLs outright unless tests and historical config coverage are updated accordingly.

## Recommended First Slice

The smallest valuable implementation slice is:

1. align `src/core/config.py` fallback defaults;
2. add precise comments/docstrings around `anomaly_probability` and `val_realistic`;
3. convert Stage 3 tests from legacy-visible assertions to canonical-first assertions.

That slice provides the highest clarity gain with the lowest runtime risk and should be completed before any broader cleanup.
