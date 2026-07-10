---
date: 2026-07-10T15:55:25+0700
researcher: Codex
git_commit: 6dc99c0dd296e96bb28f563d38e00d13a0da94f8
branch: dev
repository: bachelor-thesis-2026
topic: "Structure for detailed src code-audit remediation planning"
tags: [structure, source-audit, readability, refactor, compatibility]
status: approved
last_updated: 2026-07-10
last_updated_by: Codex
---

# Structure: `src/` code-audit remediation

## Overview

This structure expands the preliminary plan into an implementation sequence
that can later be specified by `prompts/4_detail_prompt.md`. The sequence is
contract-first: it locks one complete offline vertical slice before changing
advanced prototype, synthetic-augmentation, or online-adaptation internals.

The target is zero violations of the repository limits: every owned source file
must contain at most 500 lines, and every function or method at most 50 lines.
Public YAML, registry, tensor, metric, artifact, and checkpoint contracts remain
unchanged unless a separate migration is explicitly approved.

```text
contract snapshot
      |
      v
config -> data -> model -> train/checkpoint -> evaluate
      |                                      |
      +----------- verify equivalence <------+
                         |
                         v
        advanced model and online refactors
```

## Fixed Decisions

1. `pytest.ini` continues to collect only repository-owned tests under
   `tests/`; reference-codebase tests are not part of the acceptance suite.
2. Each model keeps one registry-facing public entrypoint. Composition replaces
   lifecycle mixins, while reusable tensor operations, immutable configuration,
   and model-independent diagnostics may live in narrow helper modules.
3. Existing constructor keyword arguments, registry names, model-output keys,
   checkpoint metadata, state-dict parameter names, metric names, and seeded
   synthetic outputs are compatibility requirements.
4. Refactoring proceeds in small slices. A later phase does not begin until the
   focused tests and contract comparisons for the current phase pass.

## Implementation Phases

### Phase 0 — Freeze the audit baseline

**Purpose:** Establish reproducible evidence for the current worktree before
moving any implementation.

**Work packages:**

1. Re-run the AST inventory over `src/**/*.py` and save the file/function
   violations as the baseline attached to the detailed plan.
2. Inventory public imports, current registry names, YAML keys, output keys,
   state-dict keys, checkpoint extra-state keys, and artifact fields.
3. Treat the existing uncommitted `neural_blocks.py`, pytest discovery change,
   and related test edits as audit inputs; verify them before building further
   refactors on top.
4. Define the exact focused-test command for every later phase.

**Engineering principles:** Stable interfaces and evidence before mutation.

**Exit gate:** The detailed plan has a baseline table mapping each target symbol
to tests and compatibility evidence.

### Phase 1 — Lock the minimal offline vertical slice

**Purpose:** Protect the smallest end-to-end path before refactoring shared
infrastructure.

**Protected flow:**

```text
experiment YAML
  -> config resolution
  -> SMD sequence/scaler/window loader
  -> model forward + one backward step
  -> checkpoint save/load
  -> evaluator overlap reconstruction + metrics
```

**Work packages:**

1. Characterize batch shape `[B, L, D]`, active `L = 20`, metadata, masks, and
   labels at the loader/model boundary.
2. Characterize thesis, RedLamp, and reconstruction-baseline output keys and
   shapes without changing implementation.
3. Snapshot representative state-dict key sets and checkpoint extra state.
4. Lock threshold provenance, overlap averaging, and final record fields.

**Primary tests:** `test_config_loading.py`, `test_windowizer.py`,
`test_multitask_shapes.py`, `test_one_multitask_train_step.py`,
`test_checkpoint_roundtrip.py`, and `test_evaluator_thresholding.py`.

**Engineering principles:** Minimal vertical slice, adapter-compatible tensor
contracts, and behavior-preserving characterization tests.

**Exit gate:** The complete offline slice passes before and after a no-op
checkpoint roundtrip with identical public keys and tensor shapes.

### Phase 2 — Refactor configuration loading and validation

**Purpose:** Reduce the largest validation functions while retaining one
obvious public loading path.

**Public owner:** `src/core/config.py` continues to expose `load_yaml_config()`
and `load_experiment_config()`.

**Proposed narrow boundaries:**

1. YAML loading, reference resolution, merging, and alias normalization remain
   distinct steps coordinated by `config.py`.
2. Data, optimizer, logging, stage, model schema, and cross-section semantic
   checks move into explicitly named validator modules only where needed to
   meet the size limits.
3. `config_model_validation.py` is decomposed into field/schema validation and
   cross-field semantic validation; validation order remains explicit.
4. Repeated primitive checks use small shared validators without creating a
   general validation framework.

**Primary tests:** `test_config_loading.py`,
`test_config_loading_additional.py`, `test_config_stress_cases.py`,
`test_comparative_config_loading.py`, and `test_kaggle_config_validation.py`.

**Risks to specify in detail:** Error ordering, exact error text, duplicate-key
rejection, legacy aliases, nested config paths, and strict unknown-key checks.

**Exit gate:** All active experiment configs resolve to equivalent dictionaries;
every config source file and function satisfies the hard limits.

### Phase 3 — Refactor deterministic synthetic augmentation

**Purpose:** Separate orchestration from anomaly transformations without moving
or duplicating random-generator ownership.

**Public owner:** `SyntheticAnomalyInjector` remains importable from
`src/data/augment.py`.

**Proposed narrow boundaries:**

1. `augment.py` owns constructor validation, generator state, seed reset,
   balanced-class scheduling, batch cloning, and public `augment_batch()`.
2. Stateless tensor transformations move to one or more narrowly named helper
   modules. They receive already sampled bounds, channels, and parameters; they
   do not own random generators.
3. Metadata/mask assembly is separated from family-specific tensor mutation.
4. RedLamp class order, normal class index, anomaly family labels, and CARLA-
   informed subsequence mechanics remain unchanged.

**Primary tests:** `test_synthetic_anomaly_injection.py`,
`test_synthetic_anomaly_visibility_profile.py`,
`test_synthetic_anomaly_visualization.py`, and
`test_redlamp_realistic_validation_alignment.py`.

**Risks to specify in detail:** RNG consumption order, multiprocessing pickle
state, balanced-class remainder rotation, half-open segment bounds, visibility
metadata, and mask/label alignment.

**Exit gate:** Fixed-seed batches are tensor- and metadata-equivalent; all
augmentation source files/functions satisfy the hard limits.

### Phase 4 — Refactor offline training, evaluation, and artifact flow

**Purpose:** Make epoch mechanics readable while keeping models responsible for
the semantic meaning of their stage steps.

**Public owners:** `Trainer`, `Evaluator`, `CheckpointManager`, and
`ExperimentLogger` retain their current external interfaces.

**Work packages:**

1. Split `Trainer.train()` into epoch preparation, training-batch execution,
   validation execution, metric aggregation, scheduler decision, checkpoint
   decision, and artifact/log persistence.
2. Use explicit return objects or typed dictionaries for local epoch state;
   avoid hidden mutation shared among helpers.
3. Split `Evaluator.evaluate()` into model execution, overlap accumulation,
   entity-record reconstruction, metric calculation, and report assembly.
4. Shorten logger/checkpoint constructors and save operations without changing
   artifact paths or checkpoint payloads.

**Primary tests:** `test_one_multitask_train_step.py`,
`test_trainer_checkpoint_fallback.py`, `test_checkpoint_roundtrip.py`,
`test_multitask_metrics_runtime.py`, and `test_evaluator_thresholding.py`.

**Risks to specify in detail:** Scheduler timing, best-checkpoint comparison,
NaN fallback, clean versus synthetic validation, logging cadence, and overlap
coverage statistics.

**Exit gate:** Identical checkpoint-selection decisions and evaluation record
schemas for fixed test fixtures; all touched files/functions satisfy the hard
limits.

### Phase 5 — Replace model lifecycle mixins with composition

**Purpose:** Give each model one readable public entrypoint while preserving
prototype, fusion, loss, memory, and phase semantics.

**Work packages:**

1. Keep `ThesisMultitaskModel`, `RedLampBaseline`, and
   `OnlineAdaptationModel` as the only model-facing entrypoints for their
   respective runtimes.
2. Move immutable thesis configuration into a narrow configuration module and
   retain the current flat-keyword compatibility constructor.
3. Replace setup/state/routing/loss mixin inheritance with composition and
   small top-to-bottom orchestration methods on `ThesisMultitaskModel`.
4. Extract only reusable primitives: neural blocks, prototype/codebook tensor
   operations, fusion operations, gradient diagnostics, and generic phase
   policy/state values. Model-specific stage ordering remains visible from the
   public model entrypoint.
5. Reuse generic neural and gradient-diagnostic helpers in RedLamp without
   importing thesis-model implementation modules.
6. Keep trainable modules assigned under their existing top-level attribute
   names so checkpoint state-dict keys do not drift.

**Primary tests:** `test_thesis_multitask_config_refactor.py`,
`test_multitask_objective_controls.py`, `test_fusion_ablation_modes.py`,
`test_multitask_memory_initialization.py`, `test_multitask_memory_updates.py`,
`test_three_stage_phase_runtime.py`, `test_redlamp_baseline_runtime.py`, and
`test_redlamp_gradient_conflict_metrics.py`.

**Risks to specify in detail:** Python method-resolution changes, registered
module paths, frozen/trainable parameter sets, phase transitions, memory
bootstrap/update order, optional loss activation, and stage-log keys.

**Exit gate:** Constructor/config equivalence, forward/backward equivalence,
state-dict key equality, checkpoint roundtrip, and phase-specific trainability
all pass; obsolete lifecycle mixin files are no longer imported.

### Phase 6 — Refactor online adaptation and online-TTA orchestration

**Purpose:** Separate calibration, per-window decisions, optimizer updates, and
reporting while preserving contamination safeguards.

**Work packages:**

1. Keep the reference encoder frozen and the residual projector near-identity;
   preserve projector anchor-state and trainable-surface APIs.
2. Shorten `OnlineAdaptationModel.forward()` and stage steps by composing
   explicit score, alignment, prototype, and anchor-loss helpers.
3. Split `online_engine.py` into calibration, execution-step, sequence-runner,
   context construction, and report-finalization boundaries while retaining
   one public experiment entrypoint.
4. Keep `OnlineLoop` distinct from thesis online-TTA experiment orchestration.
5. Preserve triage thresholds, TTL/verification buffers, variant behavior,
   optimizer-step conditions, and dry-run reports.

**Primary tests:** `test_online_reference_checkpoint.py`,
`test_online_tta_trainable_surface.py`, `test_online_tta_variants.py`,
`test_online_tta_triage.py`, `test_online_engine_max_steps.py`,
`test_online_state_roundtrip.py`, and `test_online_verification_buffer.py`.

**Risks to specify in detail:** Accidental reference-encoder gradients,
projector anchor drift, anomalous-window updates, buffer expiry ordering,
threshold reuse, variant-specific optimizer semantics, and output JSON schema.

**Exit gate:** Frozen/trainable parameter identities, update counts, buffer
state, threshold artifacts, and online reports are equivalent for fixed inputs;
all touched sources satisfy the hard limits.

### Phase 7 — Refactor data parsing and metric internals

**Purpose:** Finish the remaining long functions behind stable data and metric
facades.

**Work packages:**

1. Split SMD and AnomalyArchive parsing into loading, validation, cleaning, and
   split assembly without changing sequence dictionaries.
2. Shorten window slicing while preserving half-open bounds, stride behavior,
   metadata, and non-crossing entity boundaries.
3. Keep `compute_pointwise_metrics()` and related public metric functions as
   facades; separate pointwise diagnostics, range-aware curves, VUS, binary/
   multiclass classification, and affiliation calculations by mathematical
   responsibility.
4. Refactor protocol-audit report assembly and rendering after metric outputs
   are locked.

**Primary tests:** `test_anomaly_archive_dataset_loader.py`,
`test_windowizer.py`, `test_nonoverlap_tail_windowing.py`,
`test_pointwise_range_metrics.py`, `test_vus_pr_metric.py`, and
`test_affiliation_metric.py`.

**Risks to specify in detail:** Normalize-before-windowing order, half-open
anomaly spans, NaN handling, single-class metrics, threshold-aware labels,
metric rounding, and report field names.

**Exit gate:** Data tensors/metadata and metric dictionaries are equivalent for
the same fixtures; all remaining source-size violations are zero.

### Phase 8 — Reorganize and consolidate the owned test suite

**Purpose:** Make the test suite reviewable by human developers without losing
unique behavioral coverage.

**Work packages:**

1. Organize tests into four shallow tiers: `unit`, `contract`, `integration`,
   and `smoke`; keep only shared fixtures/helpers at the test root.
2. Merge files that test one contract, replace repeated arrange/act/assert
   shapes with parametrization, and remove only exact duplicate cases.
3. Keep at most 60 test modules, at most 500 lines per test file, and at most 12
   directly declared tests per module.
4. Define pytest markers for the four tiers while keeping `pytest -q` as the
   complete default suite.
5. Compare collected cases and a semantic coverage matrix before and after each
   domain move.

**Exit gate:** The reorganized suite is shallow, grep-friendly, fully collected,
and preserves every unique edge case and contract required by earlier phases.

### Phase 9 — Enable permanent compliance and complete acceptance

**Purpose:** Convert the temporary audit inventory into a permanent repository
quality gate.

**Work packages:**

1. Add the AST compliance test over every owned `src/**/*.py` file, including
   nested and asynchronous functions, with no allowlist or exceptions.
2. Assert file length `<= 500` and function/method length `<= 50` using
   `end_lineno - lineno + 1`.
3. Run the complete repository-owned `pytest -q` suite.
4. Run representative config-load and smoke commands for the active thesis,
   RedLamp, and online experiment families.
5. Update the design tree and the final detail log with the implemented module
   boundaries and verification evidence.

**Final acceptance criteria:**

- The AST inventory reports zero violations.
- The complete repository-owned test suite passes.
- Public imports, registry names, YAML keys, output schemas, state-dict keys,
  checkpoint metadata, synthetic fixed-seed results, metric names, and online
  report schemas remain compatible.
- No lifecycle mixin remains in the thesis-model inheritance chain.
- Every new helper has one responsibility, explicit types, and a short comment
  or docstring stating its input, output, and state ownership.

## Required `4_detail_prompt.md` Expansion

The detailed plan should expand each phase into atomic edit batches. Every batch
must state:

1. Exact files and symbols modified, added, moved, or removed.
2. The public and internal interface before and after the batch.
3. State ownership and allowed mutation.
4. Compatibility assertions and focused tests run before editing.
5. Implementation order, rollback boundary, and completion condition.
6. Expected file/function line counts after the batch.
7. Exact `.venv/bin/python -m pytest ...` verification command.

## Feedback Requested Before Detail Planning

The ten implementation phases and the separate thesis/RedLamp model subphases
are approved. The detail plan must also provide a concrete test-file
consolidation map and human-review budget.
especially whether model composition should remain one phase or be separated
into thesis, RedLamp, and online-model subphases in the detailed plan.
