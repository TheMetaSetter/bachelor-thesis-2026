---
date: 2026-07-26
researcher: TheMetaSetter
repository: bachelor-thesis-2026
source_vision: documents/logs/07-26-2026/plan/vision_code-simplification-program.md
source_plan: documents/logs/07-26-2026/plan/plan-code-simplification-program.md
source_structure: documents/logs/07-26-2026/structure/structure-code-simplification-program.md
status: preliminary-detail-plan
---

# Preliminary Detailed Plan: Behavior-Preserving Simplification

## Purpose and execution rule

This document turns the 33-phase structure into a programming-ready outline.
It is still a plan, not an authorization to edit every listed file. When a
phase is selected for implementation, the code-simplification skill must first
read the live code and project conventions, inspect callers and git history,
establish a behavioral fence, and then implement one small simplification at a
time. The phase is not complete if tests must be weakened or behavior is not
understood.

The first seven phases are the initial implementation wave. Later phases are
blocked by their listed dependencies even though their detailed scope is
recorded here.

## Stable interfaces and acceptance vocabulary

The following contracts are protected in every phase:

- Data batches remain dictionaries with `x=[B,L,D]`, point labels, and entity
  metadata. Window, stride, scaler, and causal stream semantics remain stable.
- Encoders continue to expose `hidden=[B,L,H]` through the existing model
  boundary. An encoder adapter may normalize an implementation difference, but
  it may not silently change feature order or dimensionality.
- Thesis model outputs retain reconstruction, classification, point-score, and
  auxiliary fields, including uncertainty and diagnostic payloads consumed by
  evaluation and reporting.
- Training and online update ordering, phase gates, optimizer parameter groups,
  checkpoint state, thresholds, and artifact provenance remain unchanged.
- Existing config field names, aliases, generated experiment names, output
  paths, and compatibility imports remain supported unless a phase explicitly
  proves a route is dead and documents it.

The common acceptance gate for a phase is: focused tests pass without
modification; `.venv/bin/python` compilation/import checks pass; Ruff has no new
findings; the relevant smoke flow passes; and the diff contains only the phase
scope. A phase also records a before/after contract comparison and any
pre-existing failures.

## Phase 01 — CFG-01: Configuration orchestration

### Summary

Make experiment configuration loading read as one linear orchestration flow
while preserving reference resolution, merge order, alias normalization,
window-size resolution, validation, logging, and error behavior.

### File-level edits

- Inspect and, only after the fence, simplify `src/core/config.py:903-988`.
- Keep path resolution, section merge, normalization, and validation in named
  helpers with one caller each where possible.
- Do not change `data_config_path`, `model_config_path`, `task_config_path`,
  override keys, resolved section names, or logging fields.
- Update only focused config tests if a test currently asserts an accidental
  internal helper rather than the public contract; do not update expected
  values merely to make a changed behavior pass.

### Interface, pattern, and risk

The public interface remains `load_experiment_config(path) -> dict`. Use an
orchestrator plus explicit resolver/normalizer/validator helpers; do not create
a second config factory. The main risk is changing path fallback or merge order.

### Stage detail

1. Record all callers, active config families, path fallback order, log events,
   and failure cases.
2. Map each block to reference resolution, section merge, alias normalization,
   validation, or reporting.
3. Simplify one block and run config tests before the next block.
4. Load representative offline, online, baseline, smoke, and comparative YAMLs.
5. Record the final flow and unresolved `max_online_steps`/default questions.

### Acceptance criteria

`tests/test_config_loading.py`, `tests/test_config_loading_additional.py`,
`tests/benchmarks/test_benchmark_protocol_config.py`, and representative
`.venv/bin/python` config loads pass. Resolved dictionaries and error classes
match the baseline for the sampled active matrix.

## Phase 02 — CFG-02: Configuration validation ownership

### Summary

Reduce duplicate and unclear validation ownership between `src/core/config.py`,
`src/core/config_model_validation.py`, and related validators.

### File-level edits

- Inventory `_validate_model_and_task_config`, `_validate_data_runtime_config`,
  `_validate_model_and_task_semantics`, and neighboring validators.
- Assign schema, cross-section, runtime, alias, and duplicate-key checks to
  explicit boundaries.
- Preserve validation order where callers observe error precedence and preserve
  error types/messages unless an exact behavior contract says otherwise.

### Interface, pattern, and risk

The interface remains `validate_experiment_config(resolved_config) -> None`.
Use small validator functions with a single responsibility; do not introduce a
schema framework or duplicate dataclass model. The risk is accepting invalid
configs or rejecting historical active configs.

### Stage detail

1. Build a rule inventory and map every rule to its callers and test.
2. Mark exact duplicates, related-but-not-identical checks, and aliases.
3. Consolidate one rule family at a time.
4. Run positive, negative, alias, and duplicate-key tests after each slice.
5. Freeze the validator ownership map for model work.

### Acceptance criteria

All config-loading and validation tests pass unchanged. Active benchmark and
smoke configurations load with the same resolved values; invalid worker,
window, optimizer, model, and task configurations fail at the same contract
boundary.

## Phase 03 — MOD-01: Thesis model lifecycle and mixin graph

### Summary

Make the complete thesis model lifecycle followable from
`ThesisMultitaskModel` without hiding behavior in a multi-mixin MRO.

### File-level edits

- Start at `src/models/thesis_multitask.py:45-78`.
- Trace `thesis_multitask_setup_mixin.py`, `state_mixin.py`,
  `routing_mixin.py`, `loss_mixin.py`, state/memory helpers, and config parsing.
- Prefer moving lifecycle ownership into the public model or explicit composed
  helper objects; retain immutable config objects and small reusable primitives.
- Do not create another public model class or change state-dict keys.

### Interface, pattern, and risk

Keep constructor, `forward`, `training_step`, phase hooks, serialization, and
checkpoint contracts unchanged. Composition over inheritance is the target
pattern, but only when it preserves method order, `super()` behavior, registered
parameters, buffers, and attribute names. The risk is very high because mixins
currently own state and phase transitions.

### Stage detail

1. Trace constructor, forward, training step, phase transition, memory
   initialization/update, state serialization, and checkpoint load.
2. Build a method-to-owner and attribute-to-owner map, including tests and
   monkeypatches.
3. Simplify one lifecycle boundary; run model tests before another boundary.
4. Compare shapes, outputs, losses, gradients, memory state, and checkpoints.
5. Document the public model flow and any intentionally retained helper.

### Acceptance criteria

`tests/models/test_multitask_shapes.py`,
`test_one_multitask_train_step.py`, `test_multitask_memory_initialization.py`,
`test_multitask_memory_updates.py`, gradient/objective tests, and checkpoint
roundtrip tests pass. The public model remains the only model entrypoint and
the active two-stage config constructs the same parameter/state surface.

## Phase 04 — EVAL-01: Trainer/evaluator metric ownership

### Summary

Remove duplicated responsibility between trainer validation aggregation and
evaluator reporting while preserving threshold and metric semantics.

### File-level edits

- Compare `src/engine/trainer.py:482-532` with
  `src/engine/evaluator.py:415-554` and shared threshold helpers.
- Define one canonical boundary for window-to-entity reconstruction,
  point-score arrays, threshold selection, and pointwise metric input.
- Keep trainer-only checkpoint monitoring and evaluator-only trace/report
  assembly separate when their side effects differ.

### Interface, pattern, and risk

Use a shared pure helper or explicit metric service only for identical logic;
do not create a generic evaluator abstraction with hidden threshold policy. The
main risk is metric inflation or a changed threshold source.

### Stage detail

1. Capture trainer and evaluator outputs on identical batch payloads.
2. Compare reconstruction, entity stitching, threshold, VUS-PR, UQ, and key
   naming behavior.
3. Extract one identical operation and keep orchestration local.
4. Run threshold, evaluator, trace, and checkpoint-fallback tests.
5. Record the canonical metric boundary and source of each threshold.

### Acceptance criteria

`tests/evaluation/test_thresholding_helpers.py`,
`test_evaluator_thresholding.py`, `tests/evaluation/test_evaluator_trace_payload.py`,
and `tests/runtime/test_trainer_checkpoint_fallback.py` pass. Given the same
model outputs and data loader, trainer validation and evaluator report inputs
match at the documented boundary.

## Phase 05 — ONLINE-01: Canonical online runtime path

### Summary

Establish which online path is active for each experiment family and simplify
only confirmed duplicate routing.

### File-level edits

- Trace `src/engine/online_tta/online_engine.py`, `online_engine_run.py`,
  `online_engine_step.py`, `online_engine_window_core.py`, and metrics modules.
- Trace `src/engine/online_loop.py`, `src/models/online_adaptation.py` and its
  implementation, and `src/baselines/online/`.
- Build a caller/config/test matrix before modifying routing.
- Retain unverified or historical paths until an explicit compatibility result
  proves they are unnecessary.

### Interface, pattern, and risk

The public TTA entrypoint and causal stream contract remain stable. Use an
adapter for legacy entrypoints and a strategy-style dispatcher only for already
distinct A0/A1/A2 or baseline behavior. The risk is very high: online ordering,
clean-stream admission, update counts, and checkpoint resolution are scientific
protocols.

### Stage detail

1. Inventory callers, configs, test monkeypatches, and artifact outputs.
2. Classify each path active, compatibility, or unverified.
3. Simplify only a confirmed duplicate route; do not merge algorithms.
4. Run online entrypoint, variant, calibration, state, max-step, and artifact
   tests plus one representative smoke.
5. Publish the active path matrix before Phase 06.

### Acceptance criteria

`tests/online/test_online_entrypoint.py`, `test_online_tta_variants.py`,
`test_online_runtime_state.py`, `test_online_state_roundtrip.py`,
`test_online_engine_max_steps.py`, reference-checkpoint, calibration, and
artifact-integrity tests pass. A replay of the same stream emits the same
window order, update decisions, and artifact identity.

## Phase 06 — RUNTIME-01: Runtime registration lifecycle

### Summary

Make dataset/model registration and global registry lifecycle explicit across
offline, evaluation, online, benchmark, and test entrypoints.

### File-level edits

- Inspect `src/core/runtime_components.py:14-32` and `src/core/registry.py:8-55`.
- Trace registration calls in benchmark scripts and tests.
- Simplify wrappers that add no distinct registration behavior, retaining model
  and dataset names, clear semantics, and import order.

### Interface, pattern, and risk

Keep the existing registry/factory boundary; do not add a factory for registry
functions. Use explicit runtime-context functions only when offline/evaluation/
online registration genuinely differs. Risk comes from stale global state and
import-time registration.

### Stage detail

1. Record registration, clearing, and build order for each entrypoint.
2. Mark shared versus context-specific registration.
3. Simplify one wrapper and test immediately.
4. Run registry, config, import, offline, and online smoke tests.
5. Document lifecycle and test isolation rules.

### Acceptance criteria

`tests/models/test_registry.py`, config tests, public import tests, and relevant
offline/evaluation/online entrypoint tests pass. Repeated clear/register/build
cycles produce the same registered names and classes.

## Phase 07 — COMPAT-01: Facade and compatibility boundaries

### Summary

Replace implicit facade behavior with explicit, narrow compatibility boundaries
after the canonical implementation paths are known.

### File-level edits

- Inventory `src/models/online_adaptation.py`,
  `src/models/redlamp_baseline.py`, `src/models/redlamp_mlp_baseline.py`,
  `scripts/run_*.py`, and wildcard benchmark/experiment wrappers.
- Identify canonical module, legacy consumer, public names, monkeypatch targets,
  and CLI behavior for each facade.
- Convert one facade at a time to explicit forwarding or retain a documented
  adapter where import identity requires it.

### Interface, pattern, and risk

Use the adapter pattern for old names; do not re-export an entire implementation
namespace by accident. Preserve import paths, class identity where tested, CLI
flags, and monkeypatch behavior. Risk is medium/high because tests patch public
facade attributes.

### Stage detail

1. Build a facade inventory and import graph.
2. Mark required compatibility surfaces and dead candidates.
3. Replace one wildcard or `sys.modules` boundary with explicit exports.
4. Run import identity, monkeypatch, CLI, benchmark wrapper, and compile tests.
5. Record canonical import guidance and intentionally retained shims.

### Acceptance criteria

`tests/models/test_redlamp_baseline_config_surface.py`, RedLamp runtime tests,
online entrypoint tests, benchmark wrapper tests, and CLI smoke tests pass.
No active caller needs an undocumented wildcard or hidden implementation path.

## Phase 08 — CFG-03: Config field and alias ownership

### Summary and edits

Make each configuration field and legacy alias have one canonical owner across
validators, flat kwargs, and model config parsing. Inspect
`src/core/config.py`, `src/core/config_model_validation.py`, related validators,
and `src/models/thesis_multitask_impl/thesis_multitask_config_parsing.py:39-244`.
Consolidate one field family at a time; preserve resolved keys and checkpoint
metadata.

### Contract, pattern, stages, and acceptance

Use explicit normalization helpers rather than a new schema framework. Stages
are field inventory, alias map, one-family consolidation, YAML/flat-kwargs
parity, and documentation. Acceptance requires config-loading, thesis config
refactor, model construction, and checkpoint compatibility tests to pass with
identical resolved field values.

## Phase 09 — CFG-04: Public defaults and benchmark overrides

### Summary and edits

Map `src/data/api.py`, shared data YAMLs, experiment overrides, and generator
constants. Document the distinction between API defaults (`window_size=100`,
`stride=10`) and active benchmark values (`window_size=20`) without changing
either value. Verify the `max_online_steps` override separately.

### Contract, pattern, stages, and acceptance

Use explicit configuration layering, not hidden environment defaults. Stages
are default inventory, intentional/unresolved classification, documentation
slice, resolved-config parity, and closeout. Acceptance requires API tests,
benchmark generation tests, and a representative config matrix with no silent
default change.

## Phase 10 — MOD-02: Thesis forward routing

### Summary and edits

Simplify the forward path in `thesis_multitask_routing_forward_helpers.py:82-330`
and the routing mixin by naming input validation, encoding, continuous/discrete
memory lookup, fusion, uncertainty, and output assembly. Preserve
`[B,L,D]`, `[B,L,H]`, output keys, and phase gates.

### Contract, pattern, stages, and acceptance

Use composition of named forward helpers, not another inheritance layer. Stages
are forward trace, optional-branch map, one branch simplification, tensor/MC
parity, and closeout. Acceptance requires multitask shape, phase, deterministic
output, point-score, and uncertainty tests.

## Phase 11 — MOD-03: Thesis state and memory lifecycle

### Summary and edits

Separate state serialization, memory initialization, KMeans, EMA updates,
anomaly-safe filtering, and verification metadata in
`thesis_multitask_state_memory_mixin.py` and related helpers. Preserve state-dict
keys, buffers, update order, and metadata.

### Contract, pattern, stages, and acceptance

Use composed state/memory services only when mutation ownership remains visible.
Stages are mutation inventory, state-owner map, one state boundary, memory and
checkpoint parity, and closeout. Acceptance requires memory bootstrap,
initialization, update, validation alignment, reset, and checkpoint tests.

## Phase 12 — MOD-04: Thesis loss-step lifecycle

### Summary and edits

Clarify stage logging, contrastive preparation, forward invocation, loss gates,
and gradient diagnostics in `thesis_multitask_loss_step_mixin.py` and objective
helpers. Do not change weights or active Stage A/Stage B loss composition.

### Contract, pattern, stages, and acceptance

Use a named objective calculation boundary and retain explicit stage strategies
only where phase behavior differs. Stages are objective fence, responsibility
map, one-step slice, per-phase parity, and closeout. Acceptance requires one
forward/backward step, objective controls, point-score, and gradient tests.

## Phase 13 — EVAL-02: Threshold, UQ, and evaluation artifacts

### Summary and edits

Map validators/builders in `src/protocols/threshold_artifact.py`,
`src/core/uq_summary.py`, evaluator payloads, and benchmark exports. Consolidate
one artifact type at a time while preserving serialized keys, provenance,
threshold source, split names, and checkpoint roles.

### Contract, pattern, stages, and acceptance

Use explicit artifact adapters for historical formats, not a universal payload.
Stages are schema inventory, owner assignment, one read/write slice, roundtrip
and rejection tests, and closeout. Acceptance requires threshold-artifact,
online-artifact, UQ, and benchmark export tests.

## Phase 14 — ONLINE-02: Online runtime context

### Summary and edits

Simplify context assembly and finalization in
`src/engine/online_tta/online_engine_run.py:74-367`. Name data/model/optimizer,
calibration, runtime-state, checkpoint, and artifact responsibilities without
creating a parallel context object that duplicates the current dictionary.

### Contract, pattern, stages, and acceptance

Use composition of existing context helpers and preserve causal ordering.
Stages are field inventory, consumer map, one context boundary, dry-run/runtime
parity, and closeout. Acceptance requires online runtime-state, calibration,
checkpoint, artifact, and max-step tests.

## Phase 15 — ONLINE-03: Online variant dispatch

### Summary and edits

Clarify A0/A1/A2 dispatch in `online_engine_step.py:104-234`, including optimizer
parameter groups, loss selection, clean-stream gates, update counts, and reset
behavior. Remove only repeated guards proven equivalent.

### Contract, pattern, stages, and acceptance

Use a strategy-style dispatch for distinct variants; keep the causal stream and
update ordering explicit. Stages are variant matrix, branch map, one dispatch
slice, causal output comparison, and closeout. Acceptance requires all online
variant, loss, trainable-surface, and verification-cycle tests.

## Phase 16 — ONLINE-04: Stream, cursor, batcher, and persistence

### Summary and edits

Clarify ownership in `src/data/stream.py:38-267` for cursor state,
`next_window`, online views, `next_batch`, and state serialization. Preserve
window order, labels, metadata, and exact replay.

### Contract, pattern, stages, and acceptance

Use a small stream-state value boundary only if it does not duplicate cursor
logic. Stages are emitted-sequence fence, responsibility map, one utility
slice, state replay parity, and closeout. Acceptance requires online stream,
state-roundtrip, and streaming-baseline contract tests.

## Phase 17 — DATA-01: Public data API duplication

### Summary and edits

Simplify repeated SMD/anomaly-archive option assembly and wrappers in
`src/data/api.py:14-198`, reusing existing loader bundle builders in
`src/data/loaders.py`. Keep dataset-specific required fields and
`PublicDataBundle` shape.

### Contract, pattern, stages, and acceptance

Use a small explicit shared-option helper, not a generic dataset factory.
Stages are signature fence, common-option map, one wrapper slice, both-dataset
parity, and closeout. Acceptance requires loader shape, scaler, and public API
tests for both datasets.

## Phase 18 — DATA-02: Dataset parser post-processing

### Summary and edits

Compare `src/data/datasets/smd.py:62-182` and
`src/data/datasets/anomaly_archive.py:90-166`. Extract only genuinely shared
cleaning/metadata post-processing; retain parser-specific label and entity
logic.

### Contract, pattern, stages, and acceptance

Use an explicit parser adapter boundary for dataset-specific parsing. Stages are
raw-input fence, shared-stage map, one post-processing slice, golden snapshot
parity, and closeout. Acceptance requires raw-to-sequence, label, entity, and
cleaning-metadata comparisons.

## Phase 19 — AUG-01: Synthetic anomaly injector

### Summary and edits

Reduce orchestration complexity in `src/data/augment.py:38-972` without changing
the eleven-family taxonomy, RNG serialization, masks, labels, metadata, or
mixture recursion. Prefer separating dispatch/metadata mechanics from family
algorithms only after usage is proven.

### Contract, pattern, stages, and acceptance

Use a family strategy registry only because the family algorithms already have
distinct behavior; do not add another registry layer. Stages are RNG/family
fence, common operation map, one orchestration slice, seeded family parity, and
closeout. Acceptance requires family, mask, label, metadata, and RNG-state tests.

## Phase 20 — RED-01: RedLamp baseline owner

### Summary and edits

Simplify the large constructor and helper delegations in
`src/models/baseline_impl/redlamp_baseline.py`, while preserving
`src/models/redlamp_baseline.py` and `redlamp_mlp_baseline.py` compatibility
surfaces. Remove repeated setup only after constructor and checkpoint behavior
are captured.

### Contract, pattern, stages, and acceptance

Use explicit adapter exports for renamed imports; keep one baseline public owner.
Stages are baseline fence, owner map, one delegation/constructor slice, model
and benchmark parity, and closeout. Acceptance requires RedLamp config, runtime,
shape, one-step, gradient, and checkpoint tests.

## Phase 21 — RUN-01: Thesis offline benchmark runner

### Summary and edits

Compare `scripts/benchmarks/run_thesis_offline_benchmark.py` with its internal
helper. Separate read-only orchestration from model/evaluator/artifact helpers
without changing two-stage execution, manifests, reports, checkpoints, or
retention policy.

### Contract, pattern, stages, and acceptance

Use thin command orchestration and explicit artifact helpers; do not add a
runner factory. Stages are lifecycle fence, responsibility map, one helper
slice, dry-run/export and one-combination smoke, and closeout. Acceptance
requires benchmark runner, artifact export, two-stage orchestration, and one
full development-spec combination.

## Phase 22 — RUN-02: Benchmark config generators

### Summary and edits

Compare offline, online, and SMD generator scripts. Share only entity naming,
output path, and YAML-writing primitives that produce identical results.

### Contract, pattern, stages, and acceptance

Use explicit identity/path helpers, not a generic configuration factory. Stages
are matrix fence, shared primitive map, one generator slice, temporary
regeneration diff, and closeout. Acceptance requires all benchmark config
generation tests and byte/semantic comparison of generated files.

## Phase 23 — RUN-03: Stage and variant command orchestration

### Summary and edits

Trace `run_two_stage_offline_pretraining.py`, thesis offline/online runners,
and comparative SMD command builders. Make stage, variant, resume, and
checkpoint decisions explicit without changing command flags or provenance.

### Contract, pattern, stages, and acceptance

Use strategy-style command construction only for genuinely distinct stage
families. Stages are command fence, decision map, one builder slice, CLI/dry-run
and one end-to-end combination, and closeout. Acceptance requires comparative,
two-stage, preflight, and skip-completed tests.

## Phase 24 — REPORT-01: Read-only reporting extraction

### Summary and edits

Compare reporting/ops scripts that load JSON, extract identities, normalize
metrics, and build audit tables. Share read-only extraction only; retain
report-specific schemas and evidence classification.

### Contract, pattern, stages, and acceptance

Use a read-only adapter for historical layouts and preserve canonical run fields,
source paths, checkpoints, protocol, UQ, and metrics. Stages are field fence,
schema map, one loader slice, report diff, and closeout. Acceptance requires
canonical record and report-table comparisons.

## Phase 25 — REPORT-02: Re-evaluation and pruning separation

### Summary and edits

Separate audit planning from deletion-capable behavior in
`scripts/ops/re_evaluate_and_prune_thesis_runs.py`. This phase is read-only by
default and does not authorize deletion.

### Contract, pattern, stages, and acceptance

Use a dry-run manifest as the boundary between planning and writes. Stages are
safety inventory, target-manifest design, read-only planning slice, manifest
readback, and closeout. Acceptance requires exact target paths, checkpoint
roles/checksums, and no broad deletion side effect.

## Phase 26 — CFG-05: Generated configuration matrix

### Summary and edits

Inventory the large `configs/experiment/` matrix and generator ownership.
Simplify source/generated metadata or templates without renaming experiment
files, changing output directories, or changing benchmark semantics.

### Contract, pattern, stages, and acceptance

Use explicit generation metadata, not a new config abstraction. Stages are
matrix inventory, source map, one template slice, temporary regeneration diff,
and closeout. Acceptance requires all generated names, paths, keys, and values
to match the baseline.

## Phase 27 — MODEL-05: Thesis component/config ownership

### Summary and edits

Clarify construction ownership among `thesis_multitask_components.py`, encoder,
head, memory, MLP, and config objects. Keep the public thesis model as the only
model entrypoint.

### Contract, pattern, stages, and acceptance

Use composition for immutable components and adapters only at true boundaries.
Stages are construction fence, helper map, one component slice, parameter and
state-dict comparison, and closeout. Acceptance requires identical parameter
names, shapes, buffers, and checkpoint reload.

## Phase 28 — EVAL-03: Mathematical metric adapters

### Summary and edits

Inspect `src/metrics/pointwise.py` and `src/metrics/affiliation.py`. Protect
cohesive formulas; simplify only repeated input normalization, adapter, or
formatting code.

### Contract, pattern, stages, and acceptance

Use pure named helpers where they clarify metric inputs. Stages are formula
fence, adapter map, one adapter slice, golden-value comparison, and closeout.
Acceptance requires unchanged metric values, threshold inputs, and report keys.

## Phase 29 — CLI-01: Comparative runner namespace

### Summary and edits

Replace wildcard namespace dependence between
`run_comparative_smd_experiments.py` and its internal support module with
explicit interfaces for parsing, path validation, command construction, and run
records.

### Contract, pattern, stages, and acceptance

Use explicit module interfaces, not a new CLI framework. Stages are namespace
fence, support API map, one import boundary slice, CLI/plan/record tests, and
closeout. Acceptance requires comparative runner and launcher tests unchanged.

## Phase 30 — COMPAT-02: Legacy aliases and flags

### Summary and edits

Inventory stage aliases, renamed baseline names, flat kwargs, legacy flags, and
checkpoint compatibility. Centralize one alias family without deleting support.

### Contract, pattern, stages, and acceptance

Use a narrow compatibility adapter with explicit canonical names. Stages are
alias inventory, mapping, one-family slice, legacy config/checkpoint fixtures,
and closeout. Acceptance requires historical fixtures and current active configs
to resolve identically.

## Phase 31 — STATIC-01: Static import/readability cleanup

### Summary and edits

Re-run Ruff across `src`, `scripts`, and `demo`; classify F401/F403/F405/E402
signals. Remove only imports/exports proven unused by callers, tests,
notebooks, and CLI entrypoints.

### Contract, pattern, stages, and acceptance

Use no design pattern; the principle is dead-code removal after proof. Stages
are signal inventory, deadness proof, one import slice, Ruff/import/CLI parity,
and closeout. Acceptance requires no new lint signal and unchanged imports.

## Phase 32 — DEMO-01: Demo and replay entrypoints

### Summary and edits

Trace `demo/` and replay scripts, remove duplicate setup only where they call the
same canonical data/runtime APIs, and preserve documented inputs and outputs.

### Contract, pattern, stages, and acceptance

Use thin adapters to canonical runtime entrypoints. Stages are demo fence,
entrypoint map, one route slice, fixture execution, and closeout. Acceptance
requires demo stream queue and live replay tests plus documented fixture runs.

## Phase 33 — DOC-01: Documentation path and terminology drift

### Summary and edits

Correct stale `documents/design` references to the live
`documents/abstract-design-notes` SSOT where appropriate, and distinguish
active Stage A/Stage B and online terminology from historical designs.

### Contract, pattern, stages, and acceptance

Use no runtime pattern; documentation is the final interface map. Stages are
path/term inventory, current/historical classification, one documentation
slice, link/config consistency checks, and closeout. Acceptance requires no
broken SSOT links and no inactive runtime path presented as active.

## First-wave verification bundle

Before declaring Phases 01–07 complete, run the focused tests for configuration,
registry, thesis shapes and one-step training, checkpoint roundtrip, threshold
and evaluator behavior, online entrypoints/variants/state, and compatibility
imports. Then execute exactly one concrete end-to-end development-spec
combination. Do not scale to a full benchmark matrix until that flow passes.

The baseline full-suite status previously observed for this checkout was 442
passed, 1 skipped, and 10 failed. That status must be re-established at the
start of implementation so pre-existing failures are not attributed to a
simplification phase.

