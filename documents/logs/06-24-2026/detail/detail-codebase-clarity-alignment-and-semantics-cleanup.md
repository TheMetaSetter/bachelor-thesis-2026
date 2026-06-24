---
date: 2026-06-24 19:35:34 +0700
researcher: Codex
git_commit: c1c3065ee611bab9b0d5c1071e7a58f62b99d6c7
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed implementation plan for line-level cleanup of contradictory semantics and reader-facing clarity in the three-stage offline pretraining codepath"
tags: [detail, codebase-audit, semantics, clarity, three-stage]
status: complete
last_updated: 2026-06-24
last_updated_by: Codex
---

# Detail: Detailed implementation plan for line-level cleanup of contradictory semantics and reader-facing clarity in the three-stage offline pretraining codepath

**Date**: 2026-06-24 19:35:34 +0700
**Researcher**: Codex
**Git Commit**: `c1c3065ee611bab9b0d5c1071e7a58f62b99d6c7`
**Branch**: `dev`

## Objective

This detailed plan converts the approved cleanup plan into concrete programming edits. The purpose is to remove contradictory defaults, reduce semantic ambiguity, and make the three-stage offline pretraining path easier to read and safer to configure, without changing the established training budget, phase order, clean-validation semantics, or post-training evaluation flow.

The implementation is intentionally conservative. It preserves stable runtime interfaces wherever possible and limits changes to configuration interpretation, constructor naming clarity, docstrings, comments, tests, and canonical naming behavior.

## Stable Contracts to Preserve

### Batch Contract

The active batch contract must remain unchanged. Batches continue to be dictionaries centered around `batch["x"]` with optional supervision and metadata fields added by loaders or synthetic augmentation. No phase in this cleanup introduces a new batch key requirement.

### Encoder Contract

The encoder contract remains the same for both `ThesisMultitaskModel` and `RedLampMLPBaseline`. This cleanup must not alter tensor shapes, hidden-state production, or encoder-facing module composition.

### Model Output Contract

`training_step()`, `validation_step()`, `synthetic_validation_step()`, `realistic_validation_step()`, and `test_step()` must keep returning dictionaries compatible with the current training engine. This cleanup must not change the key structure expected by `src/engine/trainer.py`.

### Training Engine Contract

`src/engine/trainer.py` remains the owner of loop mechanics, checkpoint selection, metric aggregation, and auxiliary validation scheduling. No change in this plan should move loss semantics out of the owning model file or move data semantics out of the data layer.

## Design Pattern Interpretation for This Cleanup

The prompt asks for design-pattern treatment. In this task, the correct interpretation is conservative rather than expansive.

- Composition over inheritance remains unchanged: the trainer still orchestrates models, injectors, and configs without introducing new inheritance layers.
- The dataset and model registry or factory path remains unchanged: `src/core/config.py` and the existing builders continue to materialize runtime components.
- The strategy pattern already exists implicitly through `validation_step()`, `synthetic_validation_step()`, and `realistic_validation_step()`. This cleanup only clarifies the semantics of those strategies.
- No new adapter abstraction is needed. The cleanup only preserves the existing interfaces and makes their meaning clearer.

## Phase Structure

The implementation should be executed in six phases.

1. Phase 0: Baseline safety capture through tests.
2. Phase 1: Config fallback alignment.
3. Phase 2: Synthetic multiclass semantics clarity.
4. Phase 3: `val_realistic` semantics clarity.
5. Phase 4: Canonical Stage 3 naming cleanup.
6. Phase 5: Regression validation and documentation sweep.

## Phase 0: Baseline Safety Capture Through Tests

### Phase Summary

The thesis objective here is reproducibility and safe cleanup. Before changing semantics or wording, the current behavior that must survive should be pinned down in tests. This phase reduces the chance that clarity-driven edits unintentionally alter the actual runtime pipeline.

### File-Level Edits

#### 1. `tests/test_config_loading.py`

Add or revise tests that explicitly separate:

- default balanced multiclass expectations for the active multitask path;
- explicit binary opt-in behavior for configs that request it;
- acceptable fallback behavior when fields are omitted.

Concrete edit content:

- Add a test function similar to `test_multitask_default_classification_mode_and_class_balance_are_redlamp_aligned`.
- Load a minimal or copied multitask config with `classification_label_mode` and `train_balance_classes` removed.
- Assert the resolved validation path or model-construction path now uses the intended aligned defaults after the later Phase 1 edits.
- Keep a second test showing that explicitly setting `classification_label_mode: binary` and `train_balance_classes: false` still validates successfully.

#### 2. `tests/test_smd_machine_3_4_three_stage_config_loading.py`

Revise the Stage 3 alias expectations.

Concrete edit content:

- Replace assertions that treat `loaded_config["three_stage"]["stage3_prototype_warmup_epochs"]` as a normative loaded output field.
- Keep a compatibility test that passes a legacy-only alias into validation and asserts canonical normalization behavior.
- Keep the conflicting-alias rejection test intact.

#### 3. `tests/test_multitask_validation_alignment.py`

Strengthen the semantics around clean `val` versus `val_realistic`.

Concrete edit content:

- Add an assertion or helper test that proves the auxiliary validation path is separate in metrics but not a separate loader contract.
- If the current test harness allows, assert that `val_loss` remains tied to the clean `validation_step()` path and that `val_realistic_*` metrics are produced through the auxiliary path.

#### 4. `tests/test_redlamp_realistic_validation_alignment.py`

Mirror the same semantic assertion for the baseline model.

Concrete edit content:

- Ensure the test description and asserts explicitly document that `realistic_validation_step()` is an auxiliary validation mode, not a new dataset split.

### Acceptance Criteria

- Tests clearly define what behavior is intentional before any runtime code is changed.
- Stage 3 tests become canonical-first instead of legacy-visible-first.
- No new test introduces a requirement that contradicts current clean `val_loss` semantics.

## Phase 1: Config Fallback Alignment

### Phase Summary

This phase fixes the highest-leverage contradiction: active runtime defaults already imply balanced multiclass behavior, but `src/core/config.py` still carries older binary or unbalanced fallbacks. The thesis objective here is to make configuration semantics match the active training contract so future experiments do not silently regress.

### File-Level Edits

#### 1. `src/core/config.py`

Primary target blocks:

- the boolean fallback block around `train_balance_classes`;
- the `classification_label_mode` fallback block in the multitask validation section;
- the nearby validation error messaging for class-mode mismatches.

Concrete edit content:

1. In the `boolean_fields` mapping for `multitask_tsad`, change:

```python
"train_balance_classes": task_config.get("train_balance_classes", False),
```

to a fallback that matches the intended current contract for the active multitask path.

Preferred implementation shape:

```python
"train_balance_classes": task_config.get("train_balance_classes", True),
```

but only if repo-wide binary experiments do not depend on omission implying `False`. If a narrower scope is required, make the fallback conditional on `task_name == "multitask_tsad"` and document that explicitly in a short comment.

2. In the `classification_label_mode` resolution block, change:

```python
classification_label_mode = task_config.get("classification_label_mode", "binary")
```

to a fallback aligned with the active multiclass contract for multitask RedLamp-style runs.

Preferred implementation shape:

```python
classification_label_mode = task_config.get(
    "classification_label_mode", "redlamp_multiclass"
)
```

Again, if repo-wide compatibility requires narrower scoping, restrict the fallback based on the multitask task family and comment it explicitly.

3. Tighten the nearby validation message:

Current message:

```python
"classification_label_mode='redlamp_multiclass' requires num_classes == 12"
```

Recommended clarification:

```python
"classification_label_mode='redlamp_multiclass' requires num_classes == 12 "
"for the active RedLamp-aligned multitask taxonomy"
```

This is not a behavior change, but it explains why the condition exists.

4. Add a short explanatory comment immediately above the fallback block stating that the active multitask codepath is now RedLamp-aligned by default unless a config explicitly opts into binary mode.

#### 2. `tests/test_config_loading.py`

Update existing asserts or add new ones so the test suite matches the aligned fallback semantics.

Concrete edit content:

- If a test currently assumes missing `classification_label_mode` means binary, replace that assumption with an explicit binary config in the fixture.
- Add a new positive test for the aligned multitask fallback.

#### 3. `tests/test_redlamp_aligned_configs.py` if necessary

If this test file already checks baseline or thesis alignment assumptions, extend it so that fallback-based construction paths are also covered.

### Interface and Contract Notes

- No change to datasets.
- No change to encoders.
- No change to training engine interfaces.
- This phase only changes config interpretation and validation defaults.

### Risks and Mitigation

- Risk: an older config without explicit fields may now resolve differently.
  Mitigation: add an explicit binary opt-in test and, if needed, update older binary config fixtures to be explicit.

### Acceptance Criteria

- Omitted `classification_label_mode` for the active multitask path resolves to the intended modern default.
- Omitted `train_balance_classes` for the active multitask path resolves to the intended modern default.
- Explicit binary configs still validate and construct cleanly.

## Phase 2: Synthetic Multiclass Semantics Clarity

### Phase Summary

This phase addresses the biggest reader-facing confusion in the synthetic data path: the simultaneous presence of `anomaly_probability: 0.5` and `train_balance_classes: true`. The thesis objective here is not to change the active sampling mechanism, but to make the mechanism self-explanatory directly in code and config.

### File-Level Edits

#### 1. `src/data/augment.py`

Primary target blocks:

- `SyntheticAnomalyInjector.__init__()`;
- `_sample_class_labels()`;
- `_balanced_class_quota()`.

Concrete edit content:

1. Add a concise class-level or constructor-level comment near:

```python
anomaly_probability: float = 0.5,
train_balance_classes: bool = True,
classification_label_mode: str = "redlamp_multiclass",
```

Recommended wording:

```python
# In the active balanced multiclass path, class quotas come from
# train_balance_classes and the active taxonomy. anomaly_probability only
# controls Bernoulli anomaly injection when class balancing is disabled.
```

2. Add an inline comment above the `if not self.train_balance_classes:` branch in `_sample_class_labels()` stating that this branch is the only branch where `anomaly_probability` directly determines anomaly occurrence.

3. Add an inline comment above `class_quota = self._balanced_class_quota(batch_size)` stating that balanced training ignores Bernoulli sampling and instead constructs a near-uniform class allocation across the active class set.

These comments should be short and technical. They should not become tutorial prose.

#### 2. `configs/task/multitask_tsad_redlamp_multiclass_window20.yaml`

Add a YAML comment immediately above `anomaly_probability: 0.5`.

Recommended wording:

```yaml
# When train_balance_classes is true, anomaly_probability is not the class-balancing mechanism.
# It only matters for unbalanced sampling paths and for realistic auxiliary validation when reused there.
anomaly_probability: 0.5
```

Do the same in the common task config below.

#### 3. `configs/task/multitask_tsad.yaml`

Apply the same comment treatment as above so the general task config does not remain ambiguous.

#### 4. `src/models/thesis_multitask.py`

Primary target block:

- `from_flat_kwargs()` near the `classification_label_mode` fallback.

Concrete edit content:

1. Keep the existing binary fallback for `num_classes == 2`.
2. Add the symmetric positive path:

```python
if (
    "classification_label_mode" not in synthetic_values
    and architecture_values.get("num_classes") == 12
):
    synthetic_values["classification_label_mode"] = "redlamp_multiclass"
```

3. Place this logic adjacent to the existing binary fallback, with a short comment indicating that the flat-kwargs path should not remain implicitly binary-first when the architecture already declares the 12-class taxonomy.

#### 5. `src/models/redlamp_mlp_baseline.py`

Primary target blocks:

- constructor signature;
- `effective_balance_binary_classes_within_batch` resolution block;
- injector construction calls.

Concrete edit content:

1. Introduce a clearer canonical parameter name in the signature, for example:

```python
balance_classes_within_batch: bool | None = None,
balance_binary_classes_within_batch: bool = False,
```

2. Resolve precedence explicitly:

- if `balance_classes_within_batch is not None`, use it;
- else fall back to `balance_binary_classes_within_batch or train_balance_classes`.

3. Add a short comment explaining that the old name is preserved only as a compatibility alias from earlier binary-oriented semantics.

4. Replace the local variable name `effective_balance_binary_classes_within_batch` with a neutral name such as `effective_balance_classes_within_batch`.

5. Pass that neutral variable into both injector constructors.

#### 6. `tests/test_redlamp_mlp_baseline.py`

Add tests for the new canonical constructor argument and alias compatibility.

Concrete edit content:

- one test that constructs the baseline with `balance_classes_within_batch=True`;
- one test that constructs it with only `balance_binary_classes_within_batch=True`;
- assert both produce an injector with `train_balance_classes is True`.

### Interface and Contract Notes

- Dataset contract unchanged.
- Model contract unchanged.
- Constructor surface for baseline expands slightly but remains backward-compatible.

### Risks and Mitigation

- Risk: alias precedence becomes ambiguous.
  Mitigation: write the precedence rule explicitly in code and tests.

### Acceptance Criteria

- A new reader can tell from code and YAML comments that balanced multiclass training is not driven by `anomaly_probability`.
- `ThesisMultitaskModel.from_flat_kwargs()` no longer looks implicitly binary-first for the 12-class case.
- `RedLampMLPBaseline` exposes a clearer balancing parameter without breaking older call sites.

## Phase 3: `val_realistic` Semantics Clarity

### Phase Summary

This phase clarifies a subtle but important validation semantic. The thesis objective is to preserve the current metric protocol while removing the false impression that `val_realistic` is a separate validation dataset split.

### File-Level Edits

#### 1. `src/models/thesis_multitask.py`

Primary target block:

- `prepare_realistic_validation_epoch()`.

Concrete edit content:

1. Add a docstring directly under the function definition.

Recommended content:

```python
"""Configure the synthetic validation injector for the upcoming auxiliary
validation epoch using a target anomaly prior. This does not switch loaders or
create a separate validation split; it only adjusts injection behavior for the
existing validation loader."""
```

2. Keep implementation logic unchanged.

#### 2. `src/models/redlamp_mlp_baseline.py`

Apply the same docstring pattern to the baseline implementation.

#### 3. `src/engine/trainer.py`

Primary target block:

- the `if use_val_realistic and hasattr(self.model, "realistic_validation_step"):` branch.

Concrete edit content:

Add a short comment immediately above or inside this branch:

```python
# The auxiliary realistic-validation pass reuses the existing val_loader.
# "Realistic" here means the synthetic injection prior is calibrated from
# test-window anomaly statistics, not that a different validation split is loaded.
```

Keep loop logic unchanged.

#### 4. `src/core/config.py`

Primary target block:

- validation of `val_realistic_source`.

Concrete edit content:

Add a brief comment above the accepted values set:

```python
# val_realistic_source selects the source used to estimate the anomaly prior
# for auxiliary realistic validation. It does not select a different loader.
```

If there is a centralized config-help printer or validation message path nearby, update wording there as well so the semantics are consistent.

#### 5. `tests/test_multitask_validation_alignment.py`

Add or revise assertions to mirror the clarified semantics.

Concrete edit content:

- Ensure the test name and assertion text clearly state that `val_realistic` is an auxiliary validation mode.
- If practical, assert that `val_loss` and `val_realistic_loss` can coexist in epoch metrics without implying the same semantics.

#### 6. `tests/test_redlamp_realistic_validation_alignment.py`

Mirror the same semantic test wording and assertions for the baseline path.

### Interface and Contract Notes

- Validation-step method names remain unchanged for compatibility.
- Loader ownership remains in the trainer.
- Model-owned semantics remain in model files.

### Risks and Mitigation

- Risk: users still infer too much from the name `val_realistic`.
  Mitigation: in this round, documentation and comments are sufficient; renaming can be postponed to a later explicit API cleanup if still needed.

### Acceptance Criteria

- Model docstrings and trainer comments make the real meaning of `val_realistic` unambiguous.
- Tests document the auxiliary nature of the realistic-validation pass.
- No change to `val_loss` semantics occurs.

## Phase 4: Canonical Stage 3 Naming Cleanup

### Phase Summary

This phase reduces the visibility of old Stage 3 language while preserving backward compatibility for old YAML inputs. The thesis objective is to keep the current canonical wording stable across config, orchestration, tests, and notes.

### File-Level Edits

#### 1. `src/core/config.py`

Primary target block:

- `_normalize_three_stage_config_keys()`;
- `_validate_three_stage_config()`.

Concrete edit content:

1. Preserve acceptance of `stage3_prototype_warmup_epochs` as an input alias.
2. If compatible with the existing call graph, stop re-inserting the legacy alias into the normalized in-memory config when the canonical key is already present.

Current shape:

```python
if has_canonical_key:
    three_stage_config[STAGE3_WARMUP_EPOCHS_LEGACY_KEY] = ...
```

Preferred cleanup shape:

- do not write back the legacy key when canonical input already exists;
- only map legacy-only input into the canonical key;
- keep the conflicting-alias rejection path.

3. If other parts of the repo still require the legacy key to exist in normalized output, do not force this change yet. In that case, Phase 4 should instead limit itself to test and documentation cleanup plus a TODO comment that canonical-only normalization is the next step.

#### 2. `tests/test_smd_machine_3_4_three_stage_config_loading.py`

Concrete edit content:

- remove the expectation that `stage3_prototype_warmup_epochs` always exists after loading a canonical config;
- keep a dedicated legacy-input test;
- keep the mismatch rejection test.

#### 3. `tests/test_three_stage_orchestration_smoke.py`

Review asserts for any explicit legacy name mentions in materialized configs or manifests.

Concrete edit content:

- ensure generated config paths, phase names, and manifest checks remain canonical-first.
- legacy alias should only be exercised in dedicated compatibility coverage.

#### 4. `scripts/run_three_stage_offline_pretraining.py`

Primary target block:

- constants and semantic metadata helpers.

Concrete edit content:

- review for any user-facing messages, comments, or metadata strings that still present `stage3_prototype_warmup` as current truth;
- leave the legacy constant only where needed for compatibility parsing.

#### 5. `src/models/thesis_multitask.py`

Primary target block:

- runtime normalization around `STAGE3_PHASE_LEGACY_NAME`.

Concrete edit content:

- retain compatibility normalization from legacy phase name to canonical phase name;
- review comments or error messages so canonical naming is the only reader-facing preferred wording.

#### 6. `documents/logs/06-17-2026/detail/detail-offline-pretraining-three-stage-discussion-context.md`

Concrete edit content:

- replace the logging contract lines that still list `stage3_prototype_warmup` with the canonical wording;
- preserve historical notes only if they are clearly labeled as historical rather than active truth.

### Interface and Contract Notes

- Phase order remains unchanged.
- Budget remains unchanged.
- Canonical naming becomes the only primary user-facing contract.

### Risks and Mitigation

- Risk: hidden dependencies on the legacy key still exist.
  Mitigation: search the repo for `stage3_prototype_warmup_epochs` and rerun the three-stage test suite after edits.

### Acceptance Criteria

- Canonical Stage 3 wording is the primary visible contract in code and docs.
- Legacy alias remains accepted only as a compatibility path if still needed.
- Three-stage tests continue to pass.

## Phase 5: Regression Validation and Documentation Sweep

### Phase Summary

This phase confirms that semantic cleanup did not change the intended runtime pipeline. The thesis objective is safe convergence between code truth, config truth, and user-facing wording.

### File-Level Edits

#### 1. `documents/logs/06-24-2026/research/research-codebase-audit-three-stage-semantics-and-user-facing-clarity.md`

Append a short follow-up section recording which ambiguity classes were actually fixed and which were intentionally deferred.

#### 2. `documents/logs/06-24-2026/plan/plan-codebase-clarity-alignment-and-semantics-cleanup.md`

If implementation diverges slightly from the original plan, append a short implementation note clarifying the final chosen scope.

### Validation Commands

Run these suites at minimum:

```bash
pytest -q tests/test_config_loading.py tests/test_smd_machine_3_4_three_stage_config_loading.py
pytest -q tests/test_synthetic_anomaly_injection.py tests/test_multitask_shapes.py tests/test_redlamp_mlp_baseline.py
pytest -q tests/test_multitask_validation_alignment.py tests/test_redlamp_realistic_validation_alignment.py
pytest -q tests/test_three_stage_orchestration_smoke.py tests/test_three_stage_phase_runtime.py
```

If constructor argument handling changes in the baseline model, also run:

```bash
pytest -q tests/test_one_redlamp_mlp_train_step.py
```

If Stage 3 normalization behavior changes materially, also run:

```bash
pytest -q tests/test_three_stage_run_verifier.py tests/test_three_stage_server_preflight.py tests/test_three_stage_server_launcher.py
```

### Acceptance Criteria

- All targeted suites above pass.
- The active exp4 three-stage config still resolves to the exact `300`-epoch plan.
- The balanced 12-class contract is clearer in code and config than before.
- `val_realistic` is no longer described in a way that implies a separate validation loader.
- Canonical Stage 3 wording dominates user-facing code and notes.

## Exact Recommended Edit Order

The implementation should be performed in this exact order:

1. Update tests that currently encode outdated assumptions.
2. Change config-layer fallbacks in `src/core/config.py`.
3. Update flat-kwargs defaulting in `src/models/thesis_multitask.py`.
4. Introduce clearer baseline balancing parameter naming in `src/models/redlamp_mlp_baseline.py`.
5. Add comments to `src/data/augment.py` and the active task YAMLs.
6. Add `val_realistic` docstrings and trainer comments.
7. Reduce Stage 3 legacy wording in tests, config normalization behavior if safe, and `documents/`.
8. Run targeted tests.
9. Append implementation notes to the dated research or planning artifacts if the final behavior differs from the pre-implementation expectation.

## Non-Goals

This detail plan still explicitly excludes:

- changing any training loss weights;
- changing stage epoch counts;
- changing phase ordering;
- changing checkpoint monitor semantics;
- introducing a separate realistic validation loader;
- redesigning the augmentation engine;
- redesigning the multitask architecture or fusion mechanism.

## Final Acceptance Gate

The cleanup is complete only if all of the following are simultaneously true:

- there is no contradictory default between the config layer and the active model or injector defaults for the multitask RedLamp-aligned path;
- a reader can see from code and YAML comments that `anomaly_probability` is not the balancing mechanism when `train_balance_classes=True`;
- `val_realistic` is documented as an auxiliary synthetic validation mode calibrated by test priors;
- canonical Stage 3 wording is the primary reader-facing truth across code, tests, and notes;
- the active three-stage experiment path remains operational and budget-accurate.
