---
date: 2026-06-24 15:10:00 +0700
author: Codex
repository: bachelor-thesis-2026
topic: "Detailed implementation plan for strict validation-semantics synchronization between RedLamp baseline and thesis multitask models"
tags: [detail-plan, validation, val_realistic, val_synth, redlamp, thesis_multitask]
status: draft
last_updated: 2026-06-24
last_updated_by: Codex
source_research:
  - documents/logs/06-22-2026/research/research-smd-3-4-offline-pretraining-current-state.md
  - documents/logs/06-23-2026/research/research-current-three-stage-offline-pretraining-codebase-state.md
source_detail:
  - documents/logs/05-31-2026/detail/detail-sampling-rules-train-balance-val-realistic-implementation.md
  - documents/logs/06-23-2026/detail/detail-three-stage-offline-pretraining-semantic-correction.md
---

# Detailed Plan: Strict Validation-Semantics Synchronization Between `redlamp_mlp_baseline.py` and `thesis_multitask.py`

## Objective

The objective of this implementation cycle is to remove the remaining semantic drift between the active RedLamp baseline path in `src/models/redlamp_mlp_baseline.py` and the active thesis multitask path in `src/models/thesis_multitask.py` with respect to validation behavior.

The immediate target is not to redesign the entire evaluation engine. The target is to make the two model families follow the same validation contract as closely as their architectural differences permit, while preserving the current repository interfaces and the user-locked experiment semantics:

1. both models must expose clean validation on `val`, synthetic validation on `val_synth`, and realistic synthetic validation on `val_realistic`;
2. `val_loss` must remain a clean-window reconstruction-oriented quantity and must not add classification loss;
3. `lambda_recon=0.9` and `lambda_cls=0.1` must remain synchronized across both models;
4. metrics on `val_realistic`, especially `val_realistic_vus_pr`, must remain the highest-priority validation signal during training;
5. the baseline must stop relying on the trainer fallback path and must expose explicit `prepare_realistic_validation_epoch(...)` and `realistic_validation_step(...)` methods.

## Locked Semantic Decisions

The following semantics are already decided and must be treated as fixed during implementation.

### Clean validation semantics

For both models, `validation_step` must operate on clean validation windows. The resulting `val_loss` must remain based only on clean-window reconstruction loss, scaled by `lambda_recon`, without adding the classification-loss term.

This means:

- `val_loss = lambda_recon * reconstruction_loss`;
- classification metrics may still be logged for inspection if the model emits them;
- clean validation must not be contaminated by synthetic corruption.

### Synthetic and realistic validation semantics

For both models, the auxiliary validation path must be explicit and stage-named:

- `val_synth` means deterministic synthetic validation with the model's configured validation injector;
- `val_realistic` means deterministic synthetic validation whose anomaly probability is prepared from the trainer's realistic-validation prior logic.

This detail note does not redefine `val_realistic` as true test-set evaluation. In the current engine, `val_realistic` still runs on `val_loader` after configuring the synthetic validation injector from a test-derived anomaly-rate prior. That behavior should be documented honestly rather than implied to be real test evaluation.

### Shared optimization weighting

Both models must keep:

- `lambda_recon = 0.9`;
- `lambda_cls = 0.1`.

This synchronization applies to:

- model constructor defaults;
- config-loader fallbacks;
- active aligned experiment configurations;
- and tests that assert parity.

### Label refurbishment default

Both models must continue to default to:

- `use_label_refurbishment = True`.

No change in this cycle should silently desynchronize that default.

## Current Repository State That Drives This Plan

The plan is grounded in the current codebase behavior.

1. `src/engine/trainer.py` already prefers `realistic_validation_step` and `prepare_realistic_validation_epoch(...)` when the model exposes them.
2. `src/models/thesis_multitask.py` already exposes:
   - `prepare_realistic_validation_epoch(...)`;
   - `prepare_synthetic_validation_epoch(...)`;
   - `realistic_validation_step(...)`;
   - `synthetic_validation_step(...)`;
   - and a clean `validation_step(...)` that excludes the classification term from `val_loss`.
3. `src/models/redlamp_mlp_baseline.py` already exposes:
   - `validation_step(...)` with clean `val_loss` semantics;
   - `synthetic_validation_step(...)`;
   - `prepare_synthetic_validation_epoch(...)`;
   - but it does not yet expose:
     - `prepare_realistic_validation_epoch(...)`;
     - `realistic_validation_step(...)`.
4. Because of that gap, the trainer currently routes the baseline through a fallback path in which the auxiliary validation stage is named `val_realistic` at the engine level but still uses `synthetic_validation_step(...)` under the hood.

This hybrid behavior is the main source of confusion and should be removed.

## Stable Interfaces and Contracts

### Batch contract

Both models must continue to accept the current batch dictionary contract:

- `x`
- `point_labels`
- `mask`
- `timestamps`
- `meta`

When synthetic corruption is added, both models must continue to populate:

- `classification_labels`
- `classification_class_names`
- `synthetic_anomaly_mask`
- `augmentation_metadata`

No trainer-level batch-schema change is required for this synchronization task.

### Encoder contract

No change is planned to the encoder hidden-state contract:

- baseline and thesis paths must continue to produce latent states with stable model-specific internal shapes;
- no adapter or registry rewrite is required for this task.

### Model output contract

Both models must continue to return the repository-standard output fields:

- `recon`
- `logits`
- `point_scores`
- `window_scores`
- `hidden`
- `pooled`
- `aux`

No evaluation code should need a new output key merely to support this synchronization.

## Phase 1: Remove the Baseline Fallback Ambiguity

### Phase summary

This phase makes the RedLamp baseline explicitly support the same realistic-validation lifecycle hooks as the thesis model. The purpose is to eliminate the current state in which the trainer believes it is running `val_realistic`, while the baseline still serves that path through `synthetic_validation_step(...)`.

### File-level edits

1. `src/models/redlamp_mlp_baseline.py`
2. `tests/test_one_redlamp_mlp_train_step.py`
3. `tests/test_redlamp_aligned_configs.py`
4. `tests/test_multitask_validation_alignment.py` only if a shared helper or trainer-facing assertion is expanded

### Required edits

#### 1. Add explicit realistic-validation preparation for the baseline

In `src/models/redlamp_mlp_baseline.py`, add:

- `prepare_realistic_validation_epoch(self, anomaly_probability: float) -> None`

This method should:

1. validate that `anomaly_probability` is within `[0.0, 1.0]`;
2. assign the value to the baseline synthetic validation injector;
3. reset the validation injector RNG so the path remains deterministic.

The behavior should mirror the current thesis implementation closely enough that the trainer can treat both models uniformly.

#### 2. Add explicit realistic-validation execution for the baseline

In `src/models/redlamp_mlp_baseline.py`, add:

- `realistic_validation_step(self, batch: dict[str, Any]) -> dict[str, Any]`

This method should call the same internal shared step used by `synthetic_validation_step(...)`, but the logged namespace must be `val_realistic_*`, not `val_synth_*`.

The recommended design is to keep `_shared_step(...)` as the single owner of batch preparation, forward pass, loss computation, and log construction, then expose thin public wrappers:

- `validation_step(...)`
- `synthetic_validation_step(...)`
- `realistic_validation_step(...)`

That preserves composition over duplicated logic and keeps one stable implementation surface for the baseline model.

#### 3. Preserve current clean-validation semantics

While adding the two new hooks, do not change the clean `validation_step(...)` contract that was just synchronized:

- clean `val_loss` must still exclude the classification term;
- clean `validation_step(...)` must still operate on non-synthetic windows.

### Test plan

Add or update tests to verify:

1. calling `prepare_realistic_validation_epoch(...)` twice with the same probability makes baseline realistic validation deterministic across resets;
2. `realistic_validation_step(...)` emits `val_realistic_*` metrics;
3. `synthetic_anomaly_mask` is still present in the returned batch;
4. `validation_step(...)` continues to keep `val_loss` clean-only.

### Acceptance criteria

Phase 1 is complete only if:

1. the baseline exposes both missing public hooks;
2. the trainer no longer needs the synthetic-step fallback for baseline realistic validation;
3. the baseline logs `val_realistic_*` under its own explicit method;
4. existing clean-validation loss semantics remain unchanged.

## Phase 2: Tighten Validation-Semantics Parity Across Both Models

### Phase summary

This phase audits and synchronizes the remaining semantics that can still drift even after the baseline receives explicit realistic-validation hooks. The main objective is to make the two models comparable at the experiment-reporting level.

### File-level edits

1. `src/models/redlamp_mlp_baseline.py`
2. `src/models/thesis_multitask.py`
3. `src/engine/trainer.py`
4. `tests/test_multitask_validation_alignment.py`
5. optionally a new focused baseline-validation-alignment test if the current thesis-only file becomes too semantically overloaded

### Required edits

#### 1. Align stage naming and metric namespaces

Audit all public step methods in both models so that:

- `validation_step(...)` logs `val_*`;
- `synthetic_validation_step(...)` logs `val_synth_*`;
- `realistic_validation_step(...)` logs `val_realistic_*`.

No method should emit a misleading namespace that does not match the stage name the trainer is orchestrating.

#### 2. Align auxiliary-label semantics

Audit `_prepare_batch(...)` and related helpers in both models so that:

- `classification_labels` are present whenever synthetic or realistic corruption is injected;
- `synthetic_anomaly_mask` remains the pointwise label source for `val_synth` and `val_realistic`;
- clean validation continues to preserve all-zero synthetic labels.

The baseline and thesis model need not share a literal helper function, but they should share the same observable batch semantics.

#### 3. Align which losses contribute to which logged quantities

Keep the following semantics synchronized:

- `train_loss` includes both weighted terms;
- `val_loss` remains clean reconstruction-only;
- `val_synth_loss` and `val_realistic_loss` may include the weighted classification term when synthetic labels are active;
- classification metrics on `val_realistic` remain enabled because the user identified `val_realistic_vus_pr` as the most important validation metric family.

If either model diverges from that table, fix the divergence or document it explicitly in tests and comments.

#### 4. Align trainer aggregation expectations

Audit `src/engine/trainer.py` to ensure epoch aggregation behavior does not accidentally rely on thesis-only fields when baseline is used.

The key checks are:

- whether `val_realistic_*` metrics aggregate correctly for both models;
- whether pointwise metrics on `synthetic_anomaly_mask` remain computed for both models;
- whether no thesis-specific metric such as `usage_lambda` is assumed to exist in the baseline path.

### Risk mitigation

1. Evaluation metric inflation
   Keep `val_loss` clean-only and separate from `val_realistic_*` metrics so checkpoint selection and reporting do not mix incompatible semantics.
2. Silent namespace drift
   Prefer tests that assert metric-key presence explicitly rather than relying on informal naming conventions.
3. False parity
   Do not force architectural fields that only exist in the thesis model into the baseline merely to make logs look identical.

### Acceptance criteria

Phase 2 is complete only if:

1. both models expose the same three validation stage names publicly;
2. both models emit stage-consistent metric namespaces;
3. both models compute pointwise auxiliary-validation metrics from `synthetic_anomaly_mask`;
4. no trainer aggregation path depends on thesis-only metrics when the baseline runs.

## Phase 3: Lock Shared Config Semantics and Monitor Priorities

### Phase summary

This phase ensures that synchronized runtime semantics are also reflected in config defaults, experiment configs, and metric-monitoring behavior. The purpose is to avoid reintroducing semantic drift from YAML files after code parity is fixed.

### File-level edits

1. `src/core/config.py`
2. `configs/model/redlamp_mlp_baseline.yaml`
3. `configs/model/redlamp_cnn_baseline.yaml`
4. `configs/model/thesis_multitask.yaml`
5. aligned experiment YAMLs under `configs/experiment/`
6. `tests/test_config_loading.py`
7. `tests/test_redlamp_aligned_configs.py`

### Required edits

#### 1. Preserve synchronized defaults

Keep the following synchronized in loader defaults and active model YAMLs:

- `lambda_recon: 0.9`
- `lambda_cls: 0.1`
- `use_label_refurbishment: true`

#### 2. Audit monitor metrics in aligned experiment configs

For aligned baseline and thesis experiments, verify that the chosen checkpoint monitor remains compatible with the user's priority ordering. The intended outcome is:

- `val_realistic_vus_pr` is the primary high-value validation metric where realistic validation is enabled;
- `val_loss` remains a clean diagnostic rather than the main checkpoint-selection criterion for those aligned experiments.

This phase is mainly an audit unless a config still points to an outdated metric namespace.

#### 3. Remove misleading comments or names

If any config comment or field name still implies that:

- `val_loss` contains classification loss;
- or `val_realistic` is identical to true test evaluation;

rewrite the wording so the YAML surface does not mislead future runs.

### Acceptance criteria

Phase 3 is complete only if:

1. constructor defaults, config-loader defaults, and aligned model YAMLs all match on `0.9/0.1`;
2. label refurbishment defaults remain synchronized;
3. aligned experiment configs monitor the intended realistic-validation metric namespace without stale comments.

## Phase 4: Regression Tests and Stress Cases

### Phase summary

This phase adds the minimum high-value regression coverage needed to ensure the synchronization survives future refactors. Because the user explicitly asked for stress testing and “case khó”, the test plan should include edge cases that are more adversarial than a single happy-path step.

### File-level edits

1. `tests/test_one_redlamp_mlp_train_step.py`
2. `tests/test_multitask_validation_alignment.py`
3. `tests/test_redlamp_aligned_configs.py`
4. optionally:
   - `tests/test_redlamp_realistic_validation_alignment.py`
   - `tests/test_trainer_checkpoint_fallback.py`

### Required stress cases

#### 1. Determinism under repeated realistic-validation resets

For baseline and thesis models separately, verify that:

1. `prepare_realistic_validation_epoch(...)` is called;
2. `realistic_validation_step(...)` is run;
3. the call sequence is repeated with the same input batch;
4. injected windows, class labels, and synthetic masks are identical across resets.

#### 2. Clean-versus-realistic loss separation

Verify for the baseline and thesis model that:

- clean `val_loss` stays on the clean-window path;
- realistic validation emits classification-aware metrics under `val_realistic_*`;
- those two stages cannot be confused by metric-key overlap.

#### 3. Trainer-path stress test

Construct a one-epoch trainer smoke test in which:

- the model exposes `realistic_validation_step(...)`;
- `val_realistic` is enabled;
- pointwise payloads are aggregated;
- and the resulting epoch metrics contain:
  - `val_loss`,
  - `val_realistic_loss`,
  - `val_realistic_roc_auc`,
  - `val_realistic_pr_auc`,
  - `val_realistic_vus_pr`.

The important property is that this test must pass for the baseline path as well, not only for the thesis path.

#### 4. Fallback-path protection

If the baseline now exposes explicit realistic-validation hooks, add a test that would fail if the trainer silently dropped back to the older `synthetic_validation_step(...)` branch. This prevents future regressions that accidentally remove or rename the new public hooks.

### Acceptance criteria

Phase 4 is complete only if:

1. targeted regression tests pass for both model families;
2. realistic-validation determinism is verified explicitly;
3. trainer smoke coverage proves the baseline no longer depends on the old fallback path.

## Implementation Order

The recommended order is:

1. add `prepare_realistic_validation_epoch(...)` and `realistic_validation_step(...)` to the baseline;
2. lock baseline tests around deterministic realistic validation and clean-loss separation;
3. audit trainer aggregation behavior with the baseline active;
4. run aligned-config regression checks;
5. only then proceed to broader experiment or server-run work.

This ordering minimizes semantic ambiguity early and avoids mixing runtime-contract fixes with unrelated offline-pretraining changes.

## Validation Commands

Run the following focused bundle after implementation:

```bash
.venv/bin/python -m pytest -q \
  tests/test_one_redlamp_mlp_train_step.py \
  tests/test_multitask_validation_alignment.py \
  tests/test_redlamp_aligned_configs.py
```

If a dedicated baseline realistic-validation test file is added, extend the bundle with that file as well.

Then run the broader regression bundle most relevant to current aligned semantics:

```bash
.venv/bin/python -m pytest -q \
  tests/test_config_loading.py \
  tests/test_one_redlamp_mlp_train_step.py \
  tests/test_multitask_validation_alignment.py \
  tests/test_redlamp_aligned_configs.py \
  tests/test_redlamp_cnn_baseline_shapes.py
```

## Final Exit Criteria

This synchronization task is complete only if all of the following are true:

1. the RedLamp baseline has explicit `prepare_realistic_validation_epoch(...)` and `realistic_validation_step(...)` methods;
2. `val`, `val_synth`, and `val_realistic` semantics are stage-consistent across baseline and thesis models;
3. `val_loss` remains clean-only for both models;
4. `lambda_recon=0.9` and `lambda_cls=0.1` remain synchronized across code and configs;
5. `val_realistic_vus_pr` remains available as the key realistic-validation metric family for aligned experiments;
6. the trainer no longer depends on a misleading fallback path when the baseline model is used.
