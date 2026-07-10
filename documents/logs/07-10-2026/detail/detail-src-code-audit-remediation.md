---
date: 2026-07-10T16:10:35+0700
researcher: Codex
git_commit: 8e9b208c7ef485eb5d74b5128a97e68b2a8dcdb1
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed implementation plan for src code-audit remediation"
tags: [detail, source-audit, readability, refactor, compatibility]
status: needs_refresh_after_full_spec_v2
last_updated: 2026-07-10
last_updated_by: Codex
---

## Post-implementation refresh note

This plan was written before the `full-spec-v2` remediation batches. The
current tree now also contains online-TTA helpers for entity thresholds,
signature filtering, verification-cycle TTL, hard-old hinge loss, fresh
projector optimizers, non-overlap guarding, and a demo stream queue. Therefore
the original baseline counts and file ownership assumptions must not be used
as current measurements. A new research pass must recount `src/**/*.py`,
retrace `src/engine/online_tta/online_engine.py`, and classify tests into
source-audit, online-contract, integration, and demo groups before any further
refactor batch is scheduled.

# Detail: `src/` code-audit remediation

## 1. Objective and current baseline

This document is the implementation contract for removing every current
`codebase_preferences.md` size violation from `src/` without changing the
thesis runtime semantics. Implementation must proceed in the batch order below.
Each batch ends with focused tests and may be reviewed or reverted independently
before the next batch begins.

Original baseline at commit `8e9b208c7ef485eb5d74b5128a97e68b2a8dcdb1`:

- 13 Python files under `src/` exceed 500 lines.
- 69 functions or methods exceed 50 lines.
- The repository owns 125 test files and 440 directly declared test functions.
- `pytest.ini` restricts collection to `tests/` and excludes reference
  codebases.
- The active runtime registry names are `thesis_multitask`,
  `redlamp_baseline`, `reconstruction_mlp_ae`, and `online_adaptation`.

Post-full-spec-v2 refresh (2026-07-10) reports 12 files over 500 lines and 71
callables over 50 lines. The current violation list is recorded in
`documents/logs/07-10-2026/research/research-src-code-audit-post-full-spec-v2.md`.
The online remediation also added `threshold_calibration.py`,
`signature_verification.py`, `non_overlap_guard.py`, and
`demo/stream_queue.py`; these files must be treated as existing code during
the next refactor.

The refactor does not change algorithms, experiment defaults, metric formulas,
training schedules, or dataset semantics. A behavior change discovered during
implementation must be logged separately and excluded from the refactor unless
the user explicitly approves it.

## 2. Immutable public contracts

### 2.1 Dataset and batch contract

The data layer continues to return dictionaries whose primary input is
`batch["x"]: Tensor[B, L, D]`. Active thesis experiments use `L = 20`.
Optional keys such as `point_labels`, `mask`, `timestamps`, classification
labels, synthetic masks, and `meta` retain their current names, dtypes, shapes,
and missing-value behavior.

Normalization order remains:

```text
raw sequence -> cleaning -> train-fitted scaler -> transformed sequence
             -> half-open overlapping windows -> collated batch
```

No helper may window before normalization or combine windows from different
entities.

### 2.2 Encoder and model-output contract

The thesis-facing encoder output remains `hidden: Tensor[B, L, H]`, with an
optional `pooled: Tensor[B, H]`. Model outputs retain the current dictionary
surface validated by `validate_model_outputs()`, including `recon`, `logits`,
`point_scores`, `window_scores`, and `aux` when the active model/stage produces
them.

`ThesisMultitaskModel`, `RedLampBaseline`, and `OnlineAdaptationModel` remain
the registry-facing classes. Their public constructors, `forward()`,
`training_step()`, `validation_step()`, `test_step()`, schedule/state methods,
and checkpoint hooks retain their current call signatures.

### 2.3 Training-engine contract

`Trainer.train()` remains the public offline training entrypoint. Models remain
responsible for stage semantics and loss construction; the trainer remains
responsible for iteration, optimizer/scheduler steps, validation invocation,
checkpoint decisions, logging, and artifact persistence.

`Evaluator.evaluate()` remains the public evaluation entrypoint. Overlapping
window contributions continue to be averaged back onto each entity timeline
before pointwise metrics are calculated.

### 2.4 Configuration, registry, and checkpoint contract

`load_yaml_config()` and `load_experiment_config()` remain importable from
`src.core.config`. Duplicate YAML keys, unknown strict keys, legacy aliases,
and cross-section semantic checks preserve their current validation order and
error text.

`src.core.runtime_components` continues to register the same model and dataset
names. The refactor does not add a second registry or bypass the existing
factory functions.

Trainable modules, parameters, and buffers remain assigned under their existing
top-level model attribute names. This is mandatory because wrapping them inside
new `nn.Module` collaborators would change state-dict keys. New collaborators
therefore use pure functions, immutable dataclasses, or non-module controllers
unless a key migration is explicitly approved.

### 2.5 Metrics and reporting contract

Public imports from `src.metrics.pointwise` remain valid. Metric dictionary
keys, NaN behavior, threshold fields, benchmark-comparability fields, protocol
status, and report JSON schemas remain unchanged.

## 3. Design-pattern decisions

1. **Composition over inheritance:** Remove thesis lifecycle mixins. The public
   model composes pure operations and immutable policies while keeping stage
   orchestration visible in `thesis_multitask.py`.
2. **Adapter pattern:** Move the thesis-to-online encoder view into
   `src/adapters/thesis_multitask.py`. The adapter exposes `hidden`, scoring,
   prototype-target, and encoder-parameter operations without changing the
   offline model.
3. **Strategy pattern:** Preserve `BaseModel.training_step()`,
   `validation_step()`, and `test_step()` as the task strategy consumed by the
   engine. Represent online A0/A1/A2 update choices as an explicit mapping of
   variant names to small update callables instead of adding subclasses.
4. **Registry/factory:** Continue selecting datasets and models through
   `src.core.registry` and `src.core.runtime_components`. No direct constructor
   branch is added to the trainer.
5. **Facade:** Keep existing high-traffic modules (`core.config`,
   `data.augment`, `metrics.pointwise`, and public model files) as stable import
   facades even when implementation helpers move.

## 4. Execution rules for every edit batch

Each batch follows this sequence:

1. Run the listed focused tests before editing and record the result.
2. Add or strengthen characterization assertions before moving implementation.
3. Move only one responsibility at a time using `apply_patch`.
4. Preserve original call order, mutation order, random-number consumption,
   and error order.
5. Run the same focused tests after editing.
6. Run the AST inventory for touched files. Target at most 450 lines per new
   helper/facade and at most 45 lines per new function, leaving safety margin
   below the hard limits of 500 and 50.
7. Stop the batch if a compatibility assertion changes. Do not compensate by
   weakening or deleting the assertion.

The rollback boundary is the changed-file set named by each batch. No batch
depends on unverified changes from the following batch.

## 5. Phase 0 — Freeze baseline evidence

### Batch 0.1 — Add the reusable AST audit helper

**Add:** `tests/codebase_compliance.py`.

Implement `scan_source_size_violations(source_root: Path) -> AuditResult`, where
`AuditResult` is an immutable dataclass containing file violations and callable
violations. Count callable length using
`end_lineno - lineno + 1`; include standalone, nested, async, and class methods.
The helper scans only `src/**/*.py`.

**Add:** `tests/test_codebase_compliance_scanner.py`.

Test the scanner on temporary miniature files for the exact boundaries 50/51
function lines and 500/501 file lines. This test validates the scanner itself;
it does not yet fail on known repository violations.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_codebase_compliance_scanner.py
```

**Acceptance:** Scanner fixture results are exact and deterministic. Both new
files are below 300 lines; every callable is below 45 lines.

### Batch 0.2 — Snapshot public compatibility surfaces

**Add:** `tests/test_src_refactor_contracts.py` and
`tests/fixtures/src_refactor_contracts.json`.

The JSON fixture stores only stable textual surfaces: registry names,
representative resolved-config key trees, model output key sets, state-dict key
sets, checkpoint top-level keys, checkpoint extra-state keys, metric key sets,
and online report key sets. Do not store weights, data samples, temporary paths,
or floating-point metric values in this fixture.

Use these active smoke configurations:

- Thesis offline:
  `configs/experiment/benchmark_smoke/thesis/smd__thesis_multitask__benchmark-two-stage-machine_1_6__w20__seed6__smoke.yaml`.
- RedLamp:
  `configs/experiment/comparative/baseline/smd__redlamp_baseline__comparative-single-stage-machine_1_6__w20__seed6__smoke.yaml`.
- Thesis online:
  `configs/experiment/online_benchmark/thesis/smd__thesis__online__O0_A0__machine_1_6__w20__seed6__smoke.yaml`.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_src_refactor_contracts.py tests/test_registry.py tests/test_checkpoint_roundtrip.py
```

**Acceptance:** Snapshot generation is a one-time explicit test-fixture update;
ordinary test runs only compare against it. No environment-specific path enters
the fixture.

## 6. Phase 1 — Lock the minimal offline vertical slice

### Batch 1.1 — Strengthen data/model boundary tests

**Modify:** `tests/test_windowizer.py`, `tests/test_multitask_shapes.py`,
`tests/test_one_multitask_train_step.py`, and
`tests/test_evaluator_thresholding.py`.

Add assertions for `[B, 20, D]`, half-open `start_index/end_index`, non-crossing
entity metadata, `hidden/recon/logits/point_scores/window_scores/aux` presence by
stage, and overlap-averaged point scores. Reuse existing small tensors; do not
load the full SMD dataset.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_windowizer.py tests/test_multitask_shapes.py tests/test_one_multitask_train_step.py tests/test_checkpoint_roundtrip.py tests/test_evaluator_thresholding.py
```

**Acceptance:** One forward/backward step and checkpoint roundtrip preserve the
snapshot surfaces from Batch 0.2.

### Batch 1.2 — Lock stage and ablation observability

**Modify only if coverage is missing:**
`tests/test_multitask_objective_controls.py`,
`tests/test_fusion_ablation_modes.py`, and
`tests/test_three_stage_phase_runtime.py`.

Assert that continuous-only, discrete-only, and fused modes retain distinguishable
outputs/log keys; fusion gates remain observable; disabled losses do not enter
the total loss; and phase-specific trainable parameter sets remain explicit.
These assertions mitigate prototype redundancy and fusion collapse during a
structural refactor without adding a new loss or model behavior.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_multitask_objective_controls.py tests/test_fusion_ablation_modes.py tests/test_three_stage_phase_runtime.py
```

**Acceptance:** All branch and phase combinations used by active YAMLs remain
covered before lifecycle code moves.

## 7. Phase 2 — Configuration loading and validation

### Batch 2.1 — Extract stage, data, optimizer, and logging validators

**Keep as facade:** `src/core/config.py`.

**Add:**

- `src/core/config_stage_validation.py`
- `src/core/config_data_validation.py`
- `src/core/config_optimizer_validation.py`
- `src/core/config_logging_validation.py`

Move existing validation bodies without rewriting conditions. `config.py`
retains YAML loading, config-reference resolution, merging, alias normalization,
and orchestration. Validator functions accept explicit section dictionaries and
return `None`; they do not mutate unrelated sections. Alias normalization occurs
before validation exactly as it does now.

Split `_validate_optimizer_config()` into named checks for optimizer fields,
scheduler fields, warmup/cosine fields, and monitor coupling. Split
`_validate_logging_config()` into destination, W&B, focus-metric, diagnostic,
and artifact-policy checks. Each check is called in original source order.

**Line targets:** `config.py <= 450`; each new module `<= 400`; every function
`<= 45`.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_config_loading.py tests/test_config_loading_additional.py tests/test_config_stress_cases.py tests/test_learning_rate_scheduler.py tests/test_learning_rate_scheduler_additional.py tests/test_kaggle_config_validation.py
```

**Acceptance:** Resolved dictionaries and all tested error messages remain
identical; duplicate YAML keys still fail at parse time.

### Batch 2.2 — Split model schema and semantic validation

**Keep as compatibility facade:** `src/core/config_model_validation.py`.

**Add:**

- `src/core/config_model_schema_validation.py`
- `src/core/config_model_semantic_validation.py`

The facade re-exports `_validate_model_and_task_config`,
`_validate_data_runtime_config`, and `_validate_model_and_task_semantics` so
`config_experiment_validation.py` does not change import behavior. Schema
validation checks field presence, accepted keys, primitive types, and local
ranges. Semantic validation checks cross-field and model/task/data coupling.
Use small literal field groups rather than a class hierarchy or schema library.

**Line targets:** facade `<= 100`; each implementation module `<= 450`; every
function `<= 45`.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_config_loading.py tests/test_config_loading_additional.py tests/test_redlamp_baseline_config_surface.py tests/test_comparative_config_loading.py tests/test_benchmark_protocol_config.py
```

**Acceptance:** The original three private entrypoints remain callable and the
complete active experiment-config inventory loads with no key drift.

## 8. Phase 3 — Deterministic synthetic anomaly injection

### Batch 3.1 — Extract immutable settings and pure segment helpers

**Keep as public facade:** `src/data/augment.py`.

**Add:**

- `src/data/synthetic_anomaly_config.py`
- `src/data/synthetic_anomaly_transforms.py`

`SyntheticAnomalyConfig` validates constructor values but does not replace the
existing public `SyntheticAnomalyInjector(...)` signature. The injector builds
the immutable settings object, then stores the same public attributes expected
by current tests and callers.

Move interpolation, scale calculation, visibility adjustment, segment update,
and family-specific tensor transformations into
`synthetic_anomaly_transforms.py`. Transformation functions receive sampled
bounds/channels and random tensors as inputs; they do not create or own a
generator. Return a small immutable `SyntheticTransformResult` containing the
changed window, mask delta, and metadata additions.

**State ownership:** The injector exclusively owns train/validation generators,
pickle state, reset behavior, balanced-class cursor, and RNG call order.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_synthetic_anomaly_injection.py tests/test_synthetic_anomaly_visibility_profile.py tests/test_synthetic_profile_comparison_plot.py
```

**Acceptance:** Every family yields identical fixed-seed classification labels,
masks, segment bounds, affected channels, and metadata; transform/config modules
are each `<= 450` lines.

### Batch 3.2 — Shorten batch orchestration

**Modify:** `src/data/augment.py`.

Split `augment_batch()` into class-label scheduling, per-window injection,
tensor stacking, metadata attachment, and final batch validation. Keep the
balanced remainder rotation and generator calls in original order. Split the
constructor into validation/config construction, public-attribute storage,
generator construction, and family-registry construction.

`REDLAMP_ANOMALY_FAMILIES`, `REDLAMP_MULTICLASS_CLASS_NAMES`, and
`SyntheticAnomalyInjector` remain re-exported from `augment.py`.

**Line target:** `augment.py <= 450`; every method `<= 45`.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_synthetic_anomaly_injection.py tests/test_synthetic_anomaly_visualization.py tests/test_redlamp_realistic_validation_alignment.py tests/test_one_redlamp_mlp_train_step.py
```

**Acceptance:** Determinism also survives `__getstate__/__setstate__` and
DataLoader worker serialization.

## 9. Phase 4 — Offline engine and evaluation

### Batch 4.1 — Extract explicit epoch state and pure metric aggregation

**Add:**

- `src/engine/training_state.py`
- `src/engine/training_metrics.py`

Define immutable or locally mutable dataclasses for `TrainingBatchHistory`,
`ValidationEpochResult`, and `BestCheckpointState`. Only the owning epoch helper
mutates its local history. Move log aggregation, reconstruction diagnostics,
classification diagnostics, and reconstructed-pointwise aggregation into pure
functions in `training_metrics.py`.

**Modify:** `src/engine/trainer.py` to call these functions without changing
metric key prefixes.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_multitask_metrics_runtime.py tests/test_multitask_validation_alignment.py tests/test_redlamp_realistic_validation_alignment.py
```

**Acceptance:** Aggregate dictionaries are identical for the same fixture logs;
new modules are `<= 400` lines and all callables `<= 45`.

### Batch 4.2 — Split training and validation epoch execution

**Add:** `src/engine/validation_epoch.py`.

Keep optimizer mutation inside `Trainer`; do not create a second training
engine. Refactor `Trainer.train()` into short methods for epoch context,
training batch execution, clean validation, synthetic validation, scheduler
step, checkpoint evaluation, artifact persistence, and final result assembly.
Move only model-agnostic validation iteration into `validation_epoch.py`.

Preserve exact operation order:

```text
set phase/epoch -> optional memory initialization -> train batches
-> clean validation -> synthetic validation -> aggregate metrics
-> scheduler -> best/final checkpoint -> logging/artifacts
```

**Line targets:** `trainer.py <= 480`; `validation_epoch.py <= 350`; every
method/function `<= 45`.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_one_multitask_train_step.py tests/test_temperature_schedule.py tests/test_learning_rate_scheduler.py tests/test_learning_rate_scheduler_additional.py tests/test_trainer_checkpoint_fallback.py tests/test_multitask_memory_bootstrap.py
```

**Acceptance:** Optimizer-step count, scheduler timing, warmup values,
best-checkpoint decision, NaN fallback, and memory-initialization timing are
unchanged.

### Batch 4.3 — Shorten evaluator, checkpoint, and logger methods

**Modify:** `src/engine/evaluator.py`, `src/engine/checkpoint.py`, and
`src/engine/logger.py`.

Refactor in place because these files are already below 500 lines. Split
evaluator execution into payload collection, record reconstruction, metric
calculation, benchmark-comparability labeling, and result assembly. Split
checkpoint save payload construction from serialization. Split logger
construction into local/output/W&B/artifact setup helpers.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_evaluator_thresholding.py tests/test_evaluation_protocol_audit.py tests/test_checkpoint_roundtrip.py tests/test_logger_wandb.py tests/test_artifact_sink_selection.py
```

**Acceptance:** All three files remain below 500 lines, every callable below 50
lines, and persisted payload/report fields remain unchanged.

## 10. Phase 5A — Thesis model configuration and reusable primitives

### Batch 5A.1 — Separate immutable model configuration

**Add:** `src/models/thesis_multitask_config.py`.

Move immutable config dataclasses from `thesis_multitask_components.py`.
Refactor `ThesisMultitaskModelConfig.from_flat_kwargs()` into one dispatcher and
small group builders for architecture, prototypes, schedule, objectives,
memory, runtime, profiling, and synthetic settings. Unknown-key rejection and
fallback rules remain unchanged.

**Modify:** `src/models/thesis_multitask.py` to import and re-export all current
config class names. This preserves `getattr(thesis_multitask,
"ThesisMultitaskModelConfig")` and existing import expectations.

**Line targets:** config module `<= 450`; components module `<= 350`; each
builder `<= 45`.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_thesis_multitask_config_refactor.py tests/test_cnn_encoder_config_loading.py tests/test_config_loading_additional.py
```

**Acceptance:** Config-object and flat-keyword construction produce equivalent
runtime attributes and identical state-dict key sets.

### Batch 5A.2 — Add model-independent operation modules

**Keep/review:** `src/models/neural_blocks.py`.

**Add:**

- `src/models/prototype_operations.py`
- `src/models/fusion_operations.py`
- `src/models/gradient_diagnostics.py`
- `src/models/training_phase_policy.py`
- `src/models/multitask_objectives.py`

These modules contain pure tensor functions, immutable result dataclasses, and
non-module policies. They do not register trainable layers or buffers. The
public thesis model continues to own encoder, memory tensors, fusion gates,
projections, and task heads under their current attribute names.

Move shared gradient flattening, cosine similarity, preservation ratio,
layerwise extraction, EMA/SMA updates, and log assembly into
`gradient_diagnostics.py`. Reuse it later from RedLamp.

Move continuous/discrete lookup, top-k assignment, deterministic k-means,
memory update calculations, fusion calculations, CKA, and generic optional
loss formulas into their corresponding modules. Functions return values;
model-owned mutation occurs only in the model orchestration method.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_multitask_shapes.py tests/test_fusion_ablation_modes.py tests/test_multitask_memory_initialization.py tests/test_multitask_memory_updates.py tests/test_thesis_multitask_gradient_profiling.py tests/test_thesis_multitask_point_score_loss.py
```

**Acceptance:** Primitive extraction changes no parameter/buffer name, tensor
shape, gradient reachability, branch log, or fixed-seed memory value.

## 11. Phase 5B — Thesis public model composition

### Batch 5B.1 — Move setup and state orchestration into the public model

**Modify:** `src/models/thesis_multitask.py`.

Rebuild `ThesisMultitaskModel` to inherit only `BaseModel`. Its constructor calls
short top-to-bottom helpers for config storage, encoder construction, memory
registration, fusion/head construction, synthetic injector construction,
optional diagnostics, trainable-surface configuration, and epoch initialization.

Move phase-policy decisions and memory orchestration from setup/state mixins
into short public-model methods that call pure operations from Phase 5A. Preserve
checkpoint hooks, memory lifecycle state, semantic stage labels, and all public
schedule methods.

**Do not delete mixins in this batch.** Keep them unused until routing/loss
equivalence passes so the rollback is one import/base-class change.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_thesis_multitask_config_refactor.py tests/test_temperature_schedule.py tests/test_multitask_memory_bootstrap.py tests/test_multitask_memory_initialization.py tests/test_multitask_memory_updates.py tests/test_three_stage_phase_runtime.py tests/test_checkpoint_roundtrip.py
```

**Acceptance:** `ThesisMultitaskModel.__mro__` contains no lifecycle mixin;
state-dict and checkpoint-extra-state key sets match Batch 0.2.

### Batch 5B.2 — Move routing, fusion, and forward orchestration

**Modify:** `src/models/thesis_multitask.py`.

Implement `forward()` as a maximum-45-line orchestration method:

```text
validate batch -> encode -> query enabled prototype branches
-> build task-specific fusion -> run enabled heads
-> calculate scores/aux -> validate outputs
```

Keep clean/synthetic batch preparation and two-view contrastive pairing as
small explicit methods. Preserve classification-path disable behavior,
continuous/discrete ablations, concat projections, CKA-gated fusion, query
temperature, and point-score outputs.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_multitask_shapes.py tests/test_thesis_multitask_cnn_shapes.py tests/test_fusion_ablation_modes.py tests/test_exp2_two_view_cka.py tests/test_thesis_multitask_classification_path_toggle.py tests/test_multitask_validation_alignment.py
```

**Acceptance:** Outputs, branch aux fields, fusion logs, and gradients match the
contract snapshots for every enabled/disabled path.

### Batch 5B.3 — Move loss and stage-step orchestration; delete mixins

**Modify:** `src/models/thesis_multitask.py`.

Build `_shared_step()` from short operations: prepare batch, forward, base
losses, optional losses, gradient diagnostics, total loss, and stage log.
`training_step`, `validation_step`, `synthetic_validation_step`, and `test_step`
remain thin delegators with current stage names.

**Remove after all focused tests pass:**

- `src/models/thesis_multitask_setup_mixin.py`
- `src/models/thesis_multitask_state_mixin.py`
- `src/models/thesis_multitask_routing_mixin.py`
- `src/models/thesis_multitask_loss_mixin.py`

Remove obsolete imports and confirm `rg -n "thesis_multitask_.*mixin" src tests`
returns no runtime dependency.

**Line target:** `thesis_multitask.py <= 490`; every method `<= 45`; each helper
module `<= 450`.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_one_multitask_train_step.py tests/test_multitask_objective_controls.py tests/test_thesis_multitask_point_score_loss.py tests/test_thesis_multitask_gradient_profiling.py tests/test_multitask_metrics_runtime.py tests/test_checkpoint_roundtrip.py tests/test_three_stage_phase_runtime.py
```

**Acceptance:** No lifecycle mixin remains; all phase losses, logs, memory
updates, and checkpoint roundtrips remain compatible.

## 12. Phase 5C — RedLamp baseline as a separate subphase

### Batch 5C.1 — Shorten construction and reuse generic diagnostics

**Modify:** `src/models/redlamp_baseline.py`.

Add an internal immutable `RedLampModelConfig` or private grouped-settings
dataclasses only if they reduce constructor ambiguity; do not change the public
flat constructor signature. Split constructor work into config validation,
attribute storage, encoder/decoder/head construction, injector construction,
and diagnostic state setup.

Use `neural_blocks.py` directly. Replace duplicated gradient-diagnostic methods
with `gradient_diagnostics.py` operations while retaining RedLamp metric/log
keys and EMA/SMA state attributes.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_redlamp_baseline.py tests/test_redlamp_baseline_runtime.py tests/test_redlamp_cnn_baseline_shapes.py tests/test_redlamp_gradient_conflict_metrics.py tests/test_redlamp_baseline_with_gradient_profiling_step.py
```

**Acceptance:** RedLamp has no import from `thesis_multitask` or thesis-specific
helpers; state-dict keys and gradient diagnostics match the snapshot.

### Batch 5C.2 — Shorten stage-step orchestration

Split batch preparation, label refurbishment, classification loss, forward,
gradient profiling, total loss, and log assembly into methods/functions below
45 lines. Keep synthetic validation separate from clean validation and preserve
all label taxonomy behavior.

**Line target:** `redlamp_baseline.py <= 480`; every method `<= 45`.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_one_redlamp_mlp_train_step.py tests/test_redlamp_realistic_validation_alignment.py tests/test_redlamp_baseline_config_surface.py tests/test_redlamp_aligned_configs.py tests/test_checkpoint_roundtrip.py
```

**Acceptance:** One backward step, realistic validation, gradient profiling,
and checkpoint restoration remain compatible.

## 13. Phase 6A — Online model and encoder adapter

### Batch 6A.1 — Move the encoder adapter to the adapter boundary

**Add:** `src/adapters/thesis_multitask.py`.

Move `ThesisMultitaskEncoderAdapter` from `online_adaptation.py`. It remains a
thin adapter over the frozen/online thesis model and exposes the existing
`forward`, `score_from_hidden`, `compute_prototype_target`, and
`encoder_parameters` behavior.

Keep `NearIdentityMLPProjector` in `online_adaptation.py` unless moving it to
`neural_blocks.py` is necessary for the 500-line target. If moved, re-export it
from `online_adaptation.py` for compatibility.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_online_reference_checkpoint.py tests/test_online_adaptation_step.py tests/test_online_tta_trainable_surface.py
```

**Acceptance:** Reference and online encoder outputs match; reference parameters
remain frozen and projector initialization remains near-identity.

### Batch 6A.2 — Shorten `OnlineAdaptationModel`

**Modify:** `src/models/online_adaptation.py`.

Split constructor work into reference checkpoint loading, adapter creation,
projector creation, trainable-surface selection, and anchor-state setup. Split
`forward()` into reference encoding, online encoding/projecting, scoring,
prototype target computation, output assembly, and validation. Split stage loss
and logging similarly.

Preserve `projector_anchor_state_dict` clone semantics and all public parameter-
group methods. The anchor dictionary owns detached tensor copies; it must not
share storage with live projector parameters.

**Line target:** `online_adaptation.py <= 480`; every method `<= 45`.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_online_adaptation_step.py tests/test_online_reference_checkpoint.py tests/test_online_tta_trainable_surface.py tests/test_online_state_roundtrip.py
```

**Acceptance:** Trainable parameter identities, anchor drift, alignment losses,
state roundtrip, and output keys remain compatible.

## 14. Phase 6B — Online-TTA orchestration and online loops

### Batch 6B.1 — Split calibration and execution support

**Keep as public facade:** `src/engine/online_tta/online_engine.py`.

**Add:**

- `src/engine/online_tta/calibration.py`
- `src/engine/online_tta/execution_context.py`
- `src/engine/online_tta/sequence_runner.py`
- `src/engine/online_tta/reporting.py`

Calibration owns clean-validation scoring and threshold-artifact creation.
Execution context owns model, optimizer, stream, buffer, and threshold assembly.
Sequence runner owns window order and calls the existing step semantics.
Reporting owns JSON-safe final payload construction and file writing.

`run_thesis_online_tta_experiment()` remains importable from `online_engine.py`.
All helper state is passed explicitly; do not introduce module-global mutable
state.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_online_entrypoint.py tests/test_online_engine_max_steps.py tests/test_online_ewma_threshold.py tests/test_threshold_artifact.py
```

**Acceptance:** Threshold artifacts, dry-run context, max-step behavior, and
report schemas match the snapshot.

### Batch 6B.2 — Make online variants explicit strategies

Replace the long variant branch with
`ONLINE_UPDATE_STRATEGIES: dict[str, Callable]` for A0, A1, and A2. Each callable
receives one explicit step context and returns the existing step-result shape.
No subclass is introduced.

Refactor `OnlineLoop.run`, online stream methods, and the remaining online/
traditional baseline methods above 50 lines only after thesis online tests pass.
Keep baseline protocol return records unchanged.

**Line targets:** every online-TTA file `<= 450`; `online_loop.py` and baseline
files remain `<= 500`; every callable `<= 45`.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_online_tta_variants.py tests/test_online_tta_triage.py tests/test_online_verification_buffer.py tests/test_online_stream.py tests/test_online_streaming_baseline_contracts.py tests/test_traditional_baseline_contracts.py
```

**Acceptance:** A0/A1/A2 update counts, optimizer use, TTL/verification buffer
order, triage decisions, and per-step records are unchanged. This locks the
adaptation-contamination mitigation.

## 15. Phase 7 — Data parsing and metric internals

### Batch 7.1 — Shorten parsers, stream, and windowing

**Modify:** `src/data/datasets/smd.py`,
`src/data/datasets/anomaly_archive.py`, `src/data/window.py`, and
`src/data/stream.py`.

Extract local helpers for path validation, raw-array loading, split assembly,
label conversion, metadata construction, and cursor/window advancement. Do not
add a new dataset class hierarchy. Existing parser and builder registry
interfaces remain the dataset strategy/factory boundary.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_smd_dataset_shapes.py tests/test_anomaly_archive_dataset_loader.py tests/test_windowizer.py tests/test_nonoverlap_tail_windowing.py tests/test_online_stream.py tests/test_split_protocol.py
```

**Acceptance:** Sequence/window values, shapes, labels, metadata, stride, and
half-open bounds remain identical; every touched callable is below 50 lines.

### Batch 7.2 — Split pointwise, VUS, range, and affiliation calculations

**Keep as public facade:** `src/metrics/pointwise.py`.

**Add:**

- `src/metrics/pointwise_ranges.py`
- `src/metrics/vus.py`
- `src/metrics/classification.py`

Move mathematical helpers by responsibility and re-export current public names
from `pointwise.py`. Keep affiliation implementation in
`src/metrics/affiliation.py`, but split its two long integrals and public
assembly into named interval/zone helpers.

Do not change score thresholds, range-buffer semantics, numeric stabilization,
NaN rules, or rounding. `compute_pointwise_metrics()` remains the single public
metric-assembly facade.

**Line targets:** `pointwise.py <= 300`; each helper `<= 450`; every callable
`<= 45`.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_pointwise_range_metrics.py tests/test_vus_pr_metric.py tests/test_affiliation_metric.py tests/test_classification_diagnostics_metrics.py tests/test_evaluation_metrics_audit.py
```

**Acceptance:** Perfect, single-class, no-positive, NaN, and threshold-aware
fixtures return the same metrics. This preserves protection against evaluation
metric inflation.

### Batch 7.3 — Finish remaining analysis/protocol long functions

**Modify:** `src/analysis/evaluation_protocol_audit.py`,
`src/analysis/anomaly_archive_kl.py`,
`src/core/config_experiment_validation.py`, and any remaining file reported by
the AST scanner.

Split report data assembly from Markdown rendering; split statistical ranking
from correction/serialization; shorten experiment validation by named section
checks. Do not change report headings, CSV/JSON fields, statistical formulas, or
protocol conclusions.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_evaluation_protocol_audit.py tests/test_anomaly_archive_kl.py tests/test_benchmark_protocol_config.py
```

**Acceptance:** The AST scanner reports no long callable outside the files
scheduled for final enforcement.

## 16. Phase 8 — Test-suite organization and consolidation

The source refactor uses the existing flat test paths through Phase 7. Reorganize
tests only after source behavior is stable, so test moves cannot be confused
with runtime failures. Current baseline is 124 `test_*.py` modules and 440
directly declared test functions. Three files already exceed 500 lines, and
several domains are fragmented across `*_additional.py`, one-test files, and
overlapping runtime/config modules.

### Target test layout

Use exactly four shallow test tiers and no deeper domain folders:

```text
tests/
├── __init__.py
├── conftest.py
├── helpers.py
├── fixtures/
├── unit/
├── contract/
├── integration/
└── smoke/
```

Definitions:

- `unit/` contains deterministic tests for one function/class using in-memory
  inputs and no subprocess, network, dataset download, or filesystem workflow.
- `contract/` locks stable boundaries: config schemas, batch/output shapes,
  registry names, state-dict/checkpoint keys, metric/report keys, and protocol
  rules.
- `integration/` joins two or more owned runtime components, including one
  train step, evaluator reconstruction, online update, artifacts, and state
  roundtrips.
- `smoke/` exercises script/wrapper orchestration, generated configs,
  subprocess plans, dry runs, and benchmark preflights.

`tests/__init__.py` makes helper imports unambiguous. `tests/conftest.py` remains
at the root so fixtures apply recursively.
`tests/helpers.py` may contain only shared small factories for toy batches,
models, configs, and checkpoint fixtures. Tests do not import from another test
module. Static JSON/YAML snapshots live under `tests/fixtures/`.

### Human-review budget

The reorganized suite must satisfy all of these limits:

- At most 60 `test_*.py` modules.
- At most 500 lines per test module, with a preferred target of 350.
- At most 12 directly declared `test_*` functions per module.
- At most 50 lines per test/helper function, with a preferred target of 35.
- One module name expresses one contract; names such as `additional`, `misc`,
  `more`, and numeric suffixes are forbidden.
- Parametrization combines cases only when setup, action, and assertion shape
  are the same. A unique failure mode remains an explicit named test.

The number of collected parametrized cases is not capped. The goal is to reduce
the code a human must read, not to hide or delete behavioral coverage.

### Batch 8.1 — Build a semantic test inventory

**Add:** `documents/logs/07-10-2026/detail/test-suite-consolidation-map.md`.

List every existing test file, its target symbols, tier, unique edge cases, and
destination module. Mark a case duplicate only when target symbol, input,
action, and expected result are identical. Record baseline output of:

```bash
.venv/bin/python -m pytest --collect-only -q
```

Add a small read-only audit function to `tests/codebase_compliance.py` that
reports test module count, direct test count, file length, and callable length.
Do not yet fail on the current layout.

**Acceptance:** All 124 modules appear exactly once in the consolidation map;
every deletion proposed later points to its surviving equivalent.

### Batch 8.2 — Create shared fixtures without a fixture framework

**Modify:** `tests/conftest.py`.

**Add:** `tests/helpers.py` only if at least three destination modules need the
same builder. Consolidate repeated toy batch/model/config builders as explicit
functions with descriptive arguments. Keep fixtures local when only one domain
uses them. Do not introduce fixture inheritance, global mutable models, or
session-scoped random generators.

Move the generic audit functions from `tests/codebase_compliance.py` into
`tests/helpers.py` during this batch, then remove `tests/codebase_compliance.py`.
This keeps the final test root limited to the approved shared files.

Define markers in `pytest.ini`:

```ini
markers =
    unit: one deterministic function or class
    contract: stable public schema or interface
    integration: multiple owned runtime components
    smoke: script or workflow orchestration
```

Default `pytest -q` continues to run all markers. Marker-specific commands are
additive convenience commands, not separate sources of truth.
Each destination module sets exactly one module-level `pytestmark` matching its
tier.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/test_config_loading.py tests/test_multitask_shapes.py tests/test_checkpoint_roundtrip.py tests/test_online_adaptation_step.py
```

**Acceptance:** Shared fixtures are deterministic and function-scoped unless a
documented immutable fixture justifies wider scope.

### Batch 8.3 — Consolidate `unit/`

Create approximately 18-22 unit modules. The detailed move map must use these
canonical contract names where applicable:

- `test_data_cleaning_and_scaling.py`
- `test_windowing.py`
- `test_synthetic_anomaly_families.py`
- `test_synthetic_anomaly_visualization.py`
- `test_neural_blocks.py`
- `test_thesis_forward_and_fusion.py`
- `test_redlamp_forward.py`
- `test_online_model_losses.py`
- `test_pointwise_metrics.py`
- `test_affiliation_metrics.py`
- `test_classification_metrics.py`
- `test_thresholding.py`
- `test_online_buffers.py`
- `test_stream_windowing.py`
- `test_analysis_statistics.py`
- `test_console_and_logging_format.py`

Required merges include:

- `test_windowizer.py`, `test_nonoverlap_tail_windowing.py`, and the pure
  window-metadata cases from `test_smd_overlap_metadata_contract.py` into
  `unit/test_windowing.py` using stride/window parametrization.
- `test_pointwise_range_metrics.py`, `test_vus_pr_metric.py`, and
  `test_evaluation_metrics_audit.py` into clear pointwise/metric modules,
  retaining named single-class and NaN cases.
- Synthetic family and visibility cases with identical structure into one
  parametrized family matrix; visualization artifact cases remain separate.
- Small TTL, verification-buffer, triage, and EWMA helpers into focused online
  unit modules.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/unit
```

**Acceptance:** Unit modules pass independently, contain no workflow
subprocesses, and satisfy the human-review budget.

### Batch 8.4 — Consolidate `contract/`

Create approximately 14-18 contract modules. Required canonical modules:

- `test_config_schema_contract.py`
- `test_config_semantic_contract.py`
- `test_config_alias_contract.py`
- `test_registry_contract.py`
- `test_batch_and_output_contract.py`
- `test_checkpoint_contract.py`
- `test_smd_data_contract.py`
- `test_anomaly_archive_contract.py`
- `test_thesis_config_contract.py`
- `test_redlamp_config_contract.py`
- `test_online_config_contract.py`
- `test_metric_output_contract.py`
- `test_benchmark_protocol_contract.py`
- `test_artifact_schema_contract.py`
- `test_src_refactor_contracts.py`
- `test_codebase_preferences_compliance.py`

Move the scanner boundary tests from
`tests/test_codebase_compliance_scanner.py` into the compliance module before
removing the original flat file.

Split the current 984-line `test_config_loading.py` and 1,028-line
`test_config_loading_additional.py` by semantic contract, not by file size.
Merge `test_config_stress_cases.py` cases into the schema/semantic owner.
Parameterize model/config families when they share the same validation rule.
There must be no `test_config_loading_additional.py` after this batch.

Merge the many RedLamp active-path/config modules into
`test_redlamp_config_contract.py` when they only assert resolved configuration.
Keep forward/training behavior out of the contract tier.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/contract
```

**Acceptance:** Every YAML, registry, tensor, state, metric, and report contract
has one obvious owner module; all prior unique config failure messages remain
covered.

### Batch 8.5 — Consolidate `integration/`

Create approximately 12-15 integration modules:

- `test_offline_training_step.py`
- `test_trainer_scheduler_and_checkpoint.py`
- `test_thesis_phase_runtime.py`
- `test_thesis_memory_lifecycle.py`
- `test_thesis_metrics_runtime.py`
- `test_redlamp_training.py`
- `test_realistic_validation.py`
- `test_evaluator_reconstruction.py`
- `test_online_adaptation.py`
- `test_online_state_and_reference.py`
- `test_online_tta_runtime.py`
- `test_artifact_sinks.py`
- `test_demo_runtime.py`

Required merges:

- Combine `test_one_train_step.py`, `test_one_multitask_train_step.py`, and
  model-specific one-step cases into clear model-parametrized integration
  modules where output/loss contracts match.
- Combine learning-rate scheduler base/additional files with
  `test_temperature_schedule.py` and trainer fallback cases by scheduler/
  checkpoint responsibility. Split into two modules if the 500-line budget
  would be exceeded; never use an `additional` suffix.
- Combine thesis memory initialization/update/bootstrap tests under one memory
  lifecycle module using phase/scenario parametrization.
- Combine online adaptation, reference checkpoint, trainable surface, variant,
  and state-roundtrip cases into the three online integration owners above.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/integration
```

**Acceptance:** Each integration module states the component boundary in its
module docstring; one backward step and checkpoint/state roundtrips remain
directly visible to a reviewer.

### Batch 8.6 — Consolidate `smoke/`

Create approximately 6-8 smoke modules:

- `test_offline_benchmark_workflows.py`
- `test_online_benchmark_workflows.py`
- `test_comparative_workflows.py`
- `test_three_stage_workflows.py`
- `test_config_generation_workflows.py`
- `test_launchers_and_preflight.py`
- `test_result_export_workflows.py`
- `test_demo_workflows.py`

Merge wrapper, preflight, launcher, orchestration, generation, verifier, and
dry-run files by workflow. Parameterize machine/seed/variant matrices rather
than keeping one module per family. Preserve subprocess failure-path tests and
skip-completed semantics as named cases.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/smoke
```

**Acceptance:** Smoke modules do not duplicate unit assertions; every workflow
has success, dry-run, and relevant failure-path coverage in one visible place.

### Batch 8.7 — Prove coverage preservation and enforce reviewability

Compare the consolidation map against the final tree. Run collection by tier
and complete suite. Exact duplicate removal is allowed only when the map names
the surviving parametrized or explicit case. Any reduction in unique behavior
requires user approval.

Add `tests/contract/test_codebase_preferences_compliance.py` to enforce the
test-suite human-review budget. Source zero-violation enforcement is added in
Phase 9 after all source batches are complete:

- `<= 60` test modules;
- `<= 500` lines per test module;
- `<= 12` directly declared tests per module;
- `<= 50` lines per test/helper function;
- forbidden ambiguous filename suffixes.

**Commands:**

```bash
.venv/bin/python -m pytest --collect-only -q
.venv/bin/python -m pytest -q tests/unit
.venv/bin/python -m pytest -q tests/contract
.venv/bin/python -m pytest -q tests/integration
.venv/bin/python -m pytest -q tests/smoke
```

**Acceptance:** At most 60 modules remain; all four tiers pass independently;
the consolidation map accounts for every original unique case; root
`conftest.py` remains the only automatic fixture-discovery file.

## 17. Phase 9 — Permanent compliance and full verification

### Batch 9.1 — Enable the zero-exception source compliance test

**Modify:** `tests/contract/test_codebase_preferences_compliance.py`.

Call `scan_source_size_violations(Path("src"))` and fail with a sorted,
grep-friendly list of `path:line symbol (N lines)`. No allowlist, baseline
suppression, generated-file exception, or tolerance is permitted.

**Command:**

```bash
.venv/bin/python -m pytest -q tests/contract/test_codebase_preferences_compliance.py
```

**Acceptance:** Zero files exceed 500 lines and zero callables exceed 50 lines.

### Batch 9.2 — Run compatibility and full-suite acceptance

Run, in order:

```bash
.venv/bin/python -m pytest -q tests/contract/test_src_refactor_contracts.py
.venv/bin/python -m pytest -q tests/unit tests/contract tests/integration
.venv/bin/python -m pytest -q
```

Then run config-driven dry-run verification:

```bash
.venv/bin/python scripts/run_thesis_offline_benchmark.py --experiment-config configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O0__machine_1_6__w20__seed6__smoke.yaml --dry-run
.venv/bin/python scripts/run_thesis_online_benchmark.py --experiment-config configs/experiment/online_benchmark/thesis/smd__thesis__online__O0_A0__machine_1_6__w20__seed6__smoke.yaml --online-variant A0 --dry-run
```

Do not use the stale default config paths in `scripts/train.py` or
`scripts/evaluate.py` as acceptance evidence; explicit active YAML paths are
required for this refactor.

**Acceptance:** Full repository-owned suite and both dry runs pass. Snapshot
surfaces remain unchanged. `git diff --check` reports no whitespace errors.

### Batch 9.3 — Synchronize documentation

**Modify:** `documents/abstract-design-notes/design_starter.md` and this detail
artifact after implementation.

Record the final source tree, removed mixins, new facades/helpers, test commands,
final AST counts, and any intentionally retained compatibility re-exports.
Documentation describes implemented behavior only.

## 18. Cross-cutting risk controls

| Risk | Required control | Evidence |
| --- | --- | --- |
| Prototype branches become redundant or accidentally merged | Preserve continuous-only, discrete-only, and fused codepaths plus branch-specific aux/log fields | Fusion-ablation and objective-control tests |
| Fusion collapses or gate semantics drift | Preserve gate parameters, CKA mode, concat projections, regularization switches, and logged gate values | Fusion, CKA, phase-runtime tests |
| Synthetic labels/masks drift during extraction | Preserve RNG ownership and call order; compare fixed-seed tensors and metadata | Synthetic injection/visibility tests |
| Adaptation consumes anomalous windows | Preserve triage thresholds, verification/TTL buffers, and variant update conditions | Online triage/variant/buffer tests |
| Projector drifts or reference encoder trains | Preserve near-identity initialization, anchor copies, frozen reference parameters, and trainable-group assertions | Online reference/trainable/state tests |
| Scheduler or checkpoint timing changes | Preserve operation order and best-metric comparison; assert NaN fallback | Scheduler and trainer fallback tests |
| Evaluation metrics become inflated or incomparable | Preserve overlap averaging, covered-point counts, threshold provenance, single-class NaNs, and comparability labels | Evaluator, VUS, range, protocol-audit tests |
| State-dict keys drift under composition | Keep trainable modules at existing top-level attributes; snapshot sorted keys | Refactor-contract and checkpoint tests |

## 19. Final definition of done

Implementation is complete only when all conditions hold simultaneously:

1. The AST compliance test reports zero violations across `src/**/*.py`.
2. Every new helper has one responsibility, explicit type hints, and a concise
   docstring/comment describing inputs, outputs, and state ownership.
3. `ThesisMultitaskModel` inherits only from `BaseModel`; the four lifecycle
   mixin files are removed and no runtime import references them.
4. RedLamp and online models do not import thesis implementation helpers except
   through the explicit encoder adapter or stable public model class required
   by the current online checkpoint path.
5. Registry names, configuration keys, model/output interfaces, state-dict
   keys, checkpoint metadata, fixed-seed augmentation, metric names, and online
   report schemas match the baseline fixture.
6. The full repository-owned test suite passes under `.venv/bin/python`.
7. The active offline and online smoke wrappers complete their dry runs with
   explicit YAML paths.
8. The design tree and detail log describe the final implemented structure.
9. The test suite uses only the four approved tiers, contains at most 60 test
   modules, and passes the enforced human-review budget.

No further implementation decision is left to the coding agent. If a batch
cannot satisfy its compatibility gate, stop at that batch and report the exact
contract mismatch before proceeding.
