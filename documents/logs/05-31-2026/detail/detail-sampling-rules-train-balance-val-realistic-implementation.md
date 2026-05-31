# Detail Plan: Implement Strict Sampling Rules (Train Balanced + Val Realistic)

Date: 2026-05-31
Status: Ready for implementation
Inputs:
- `documents/logs/05-30-2026/detail/detail-sampling-rules-train-balanced-val-realistic.md`
- `documents/logs/05-30-2026/research/research-current-state-sampling-rules-train-balanced-val-realistic.md`
- `codebase_preferences.md`

## Thesis-Oriented Objective
This plan operationalizes the confirmed sampling semantics into the current multitask thesis pipeline while preserving readability, reproducibility, strict configuration semantics, and minimal codepath branching. The implementation preserves the current one-model-one-file discipline and keeps synthetic anomaly generation explicit, testable, and ablation-friendly.

## Scope and Decision Lock
1. `task.train_balance_classes` is active for both `classification_label_mode=binary` and `classification_label_mode=redlamp_multiclass`.
2. `task.val_realistic_source=test_smd_all` derives the anomaly-window prior from the complete SMD test split across all 28 entities.
3. If `task.val_anomaly_rate_override` is provided, that value overrides the derived prior while anomaly-family assignment remains uniform across configured anomaly families.
4. Remove all incompatible legacy task and experiment configurations.

## Target Interfaces and Contracts

### Task configuration contract (`multitask_tsad`)
The task YAML schema must expose exactly these validation-related fields:
- `train_balance_classes: bool`
- `val_realistic: bool`
- `val_realistic_source: test_same_scope|test_smd_all`
- `val_anomaly_rate_override: float|null`

Legacy field removal:
- Remove `balance_binary_classes_within_batch` from the multitask task schema and all active multitask experiment configs.

### Injector contract
The augmentation component must provide:
- deterministic per-batch class quotas when `train_balance_classes=true`;
- remainder distribution by round-robin over class indices;
- rotating coverage behavior when `batch_size < num_classes`;
- uniform anomaly-family sampling within anomaly classes.

### Validation contract
The training engine must expose:
- clean validation (`val`) unchanged;
- realistic validation (`val_realistic`) using target anomaly rate from configured source or override;
- pointwise and classification metrics for realistic validation without contaminating the clean validation stage.

## Phase 1: Configuration Schema Migration and Strict Validation

### Phase summary
This phase establishes strict and explicit task semantics in configuration loading so all downstream logic receives valid and complete settings.

### File-level edits
1. `src/core/config.py`
- Update allowed task keys for `multitask_tsad`:
  - add `train_balance_classes`, `val_realistic`, `val_realistic_source`, `val_anomaly_rate_override`;
  - remove `balance_binary_classes_within_batch`.
- Add explicit validation rules:
  - `train_balance_classes` and `val_realistic` must be boolean;
  - `val_realistic_source` must be one of `test_same_scope` or `test_smd_all`;
  - `val_anomaly_rate_override` must be null or float in `[0.0, 1.0]`.
- Preserve fail-fast error text with clear fix instructions.

2. `configs/task/*.yaml` (multitask task files only)
- Replace legacy key with new keys and defaults:
  - `train_balance_classes: false` (or true in explicitly balanced presets),
  - `val_realistic: true`,
  - `val_realistic_source: test_same_scope` by default,
  - `val_anomaly_rate_override: null`.

3. `configs/experiment/*.yaml` (multitask-related)
- Remove references to incompatible legacy task semantics.
- Update scheduler/checkpoint monitor keys from `val_synth_*` to `val_realistic_*` where appropriate.

### CLI execution steps
```bash
rg -n "balance_binary_classes_within_batch|val_synth" configs src tests -S
pytest -q tests/test_config_loading.py tests/test_config_stress_cases.py
```

### Acceptance criteria
- Unknown/legacy keys fail validation immediately.
- All active multitask task configs parse successfully with new schema.
- Config tests pass with updated monitor and field semantics.

## Phase 2: Train Batch Class-Balancing Implementation

### Phase summary
This phase replaces binary-only balancing with class-aware balancing that supports both binary and multiclass modes under one explicit codepath.

### File-level edits
1. `src/data/augment.py`
- Extend injector constructor signature:
  - replace `balance_binary_classes_within_batch` with `train_balance_classes`.
- Add helper methods:
  - class-quota computation from `batch_size` and `num_classes`;
  - round-robin remainder allocation over class indices;
  - rotating class coverage when `batch_size < num_classes` across consecutive batches.
- Keep anomaly-family selection uniform among anomaly families.
- Maintain classification label contract (`classification_labels`, `classification_class_names`, `synthetic_anomaly_mask`).

2. `src/models/thesis_multitask.py`
- Wire new synthetic config field `train_balance_classes`.
- Ensure both training injector and realistic-validation injector receive consistent balancing semantics when enabled.

### CLI execution steps
```bash
pytest -q tests/test_synthetic_anomaly_injection.py tests/test_multitask_shapes.py tests/test_one_multitask_train_step.py
```

### Acceptance criteria
- With `train_balance_classes=true`, label distribution follows class-quota policy.
- Remainder allocation uses round-robin as specified.
- Behavior remains deterministic when deterministic seeds are configured.

## Phase 3: Realistic Validation Prior Source and Override

### Phase summary
This phase introduces realistic validation with source-controlled anomaly-rate priors and override behavior, while preserving clean validation as a separate measurement channel.

### File-level edits
1. `src/data/datasets/smd.py`
- Add parser utility for deriving test-window anomaly prior from:
  - same scope entities (`test_same_scope`),
  - all 28 SMD entities (`test_smd_all`).
- Compute anomaly-window rate using current `window_size` and `stride`, with window anomalous iff any point label is anomalous.

2. `src/data/loaders.py` or new helper under `src/data/`
- Add thin helper to request prior derivation payload for trainer/model stage without duplicating full loader creation logic.

3. `src/models/thesis_multitask.py`
- Add `realistic_validation_step` and a stage preparation routine that configures injector anomaly probability according to:
  - override value if provided,
  - else derived source prior.
- Keep anomaly-family assignment uniform under realistic validation.

4. `src/engine/trainer.py`
- Replace `val_synth` stage orchestration with `val_realistic` when enabled.
- Keep `val` clean pass unchanged.
- Route pointwise labels for realistic stage through `synthetic_anomaly_mask`.

### CLI execution steps
```bash
pytest -q tests/test_multitask_validation_alignment.py tests/test_evaluator_thresholding.py tests/test_multitask_metrics_runtime.py
```

### Acceptance criteria
- `val_realistic_source=test_smd_all` uses all 28 SMD test entities for prior derivation.
- `val_anomaly_rate_override` supersedes source-derived prior when set.
- Family distribution remains uniform under realistic validation.
- Metrics are logged under `val_realistic_*` namespace.

## Phase 4: Remove Incompatible Configurations and Migrate Monitoring

### Phase summary
This phase enforces repository consistency by removing or rewriting obsolete multitask experiment/task configs that still encode incompatible semantics.

### File-level edits
1. `configs/task/`
- Delete or rewrite all multitask task files with legacy balancing field.

2. `configs/experiment/`
- Delete or rewrite multitask experiment presets that monitor `val_synth_*` or assume legacy field names.
- Preserve only configurations that pass strict schema checks.

3. `tests/test_config_loading.py`
- Update fixture snippets and assertions to new task fields and realistic-validation metric names.

### CLI execution steps
```bash
rg -n "balance_binary_classes_within_batch|val_synth" configs -S
pytest -q tests/test_config_loading.py
```

### Acceptance criteria
- No active config file contains legacy key names.
- All surviving multitask experiment presets pass config load + validation.

## Phase 5: Verification, Regression Safety, and Reproducibility Evidence

### Phase summary
This phase validates runtime integrity and documents reproducibility-oriented evidence required by the codebase standards.

### File-level edits
1. `tests/` (new or updated)
- Add unit tests for:
  - class quota and round-robin remainder logic,
  - rotating coverage for `batch_size < num_classes`,
  - source prior derivation for `test_same_scope` vs `test_smd_all`,
  - override precedence over source prior.
- Add integration smoke test:
  - one epoch with `val_realistic` enabled and metrics emitted.

2. `documents/logs/05-31-2026/research/` (optional post-run evidence note)
- Summarize final metric namespaces and schema migration outcomes.

### CLI execution steps
```bash
pytest -q \
  tests/test_config_loading.py \
  tests/test_synthetic_anomaly_injection.py \
  tests/test_multitask_validation_alignment.py \
  tests/test_multitask_metrics_runtime.py \
  tests/test_one_multitask_train_step.py
```

### Acceptance criteria
- All targeted tests pass.
- No schema regressions in config loading.
- Runtime stages remain explicit: `train`, `val`, `val_realistic`, `test`.

## Risk Mitigation Matrix
1. Prototype redundancy
- Preserve optional-loss toggles and avoid introducing hidden coupling in the new validation path.

2. Fusion collapse
- Keep clean validation (`val`) untouched for baseline stability; compare against `val_realistic` separately.

3. Adaptation contamination
- Do not reuse online adaptation components for offline realistic validation semantics.

4. Projector drift
- No change to online projector path in this migration.

5. Evaluation metric inflation
- Enforce source-derived/override-controlled realistic anomaly-rate semantics and explicit metric namespaces.

## End-to-End CLI Playbook (Ordered)
```bash
# 1) Locate legacy usage
rg -n "balance_binary_classes_within_batch|val_synth|val_realistic|train_balance_classes|val_anomaly_rate_override|val_realistic_source" src configs tests -S

# 2) Migrate config schema + code wiring
pytest -q tests/test_config_loading.py tests/test_config_stress_cases.py

# 3) Verify injector balancing behavior
pytest -q tests/test_synthetic_anomaly_injection.py tests/test_multitask_shapes.py tests/test_one_multitask_train_step.py

# 4) Verify realistic validation path
pytest -q tests/test_multitask_validation_alignment.py tests/test_multitask_metrics_runtime.py tests/test_evaluator_thresholding.py

# 5) Full targeted regression bundle
pytest -q \
  tests/test_config_loading.py \
  tests/test_synthetic_anomaly_injection.py \
  tests/test_multitask_validation_alignment.py \
  tests/test_multitask_metrics_runtime.py \
  tests/test_one_multitask_train_step.py
```

## Final Exit Criteria
1. New task schema is strict, validated, and fully documented.
2. Train balancing semantics are class-aware and identical across binary/multiclass modes.
3. Realistic validation prior source supports both same-scope and full-SMD (28-machine) modes.
4. Override semantics preserve uniform anomaly-family assignment.
5. Incompatible legacy configs are removed or migrated, and all designated tests pass.
