---
date: 2026-07-26
researcher: TheMetaSetter
repository: bachelor-thesis-2026
source_vision: documents/logs/07-26-2026/plan/vision_code-simplification-program.md
source_plan: documents/logs/07-26-2026/plan/plan-code-simplification-program.md
status: structured-plan
---

# Structure: 33-Phase Code-Simplification Program

## Overview

This structure turns the research findings into 33 independent programming
phases. Each phase has one simplification area, several stages, explicit
contract fences, and a verification gate. Phases 01–07 are the first delivery
wave. The structure preserves the minimal vertical slice principle: the active
runtime is verified before internal ownership is changed, and advanced cleanup
does not begin until its upstream contract is stable.

The program uses composition over inheritance when simplifying the thesis model,
adapter boundaries for legacy imports and facades, strategy-style dispatch for
already distinct online variants, and the existing registry only where it is a
real construction boundary. No pattern is introduced solely to reduce line
count.

## Common stage structure

Every phase follows this five-stage shape. The phase-specific names below state
what each stage must prove.

1. **Behavioral fence.** Record callers, configuration inputs, outputs, side
   effects, errors, ordering, and relevant baseline tests.
2. **Ownership map.** Identify the canonical implementation, compatibility
   consumers, duplicated logic, and the smallest safe edit boundary.
3. **Incremental simplification.** Make one behavior-preserving change using
   explicit names, short functions, and the fewest runtime paths.
4. **Parity verification.** Run focused tests, compile/lint checks, and one
   relevant smoke flow; compare contracts and artifacts.
5. **Closeout.** Document the final owner, preserved interfaces, remaining
   uncertainty, and the next phase's entry gate.

The detailed stage content is intentionally preliminary. Before implementation,
the code-simplification skill must expand the selected phase into atomic edits,
inspect git history where needed, and stop if behavior cannot be proven.

## Implementation phases

### Phase 01 — CFG-01: Configuration orchestration

- **Stage 01.1 — Load-path fence:** Trace `load_experiment_config` callers and
  record reference resolution, merge order, normalization, validation, logs,
  and errors.
- **Stage 01.2 — Resolver ownership:** Separate path/reference resolution from
  section merging and top-level orchestration without changing order.
- **Stage 01.3 — Thin orchestration slice:** Simplify one responsibility at a
  time in `src/core/config.py`; keep explicit names and existing interfaces.
- **Stage 01.4 — Active-config parity:** Load benchmark, smoke, online,
  baseline, and comparative configs and compare resolved sections.
- **Stage 01.5 — Closeout gate:** Record the canonical loader flow and any
  unresolved default/override ambiguity.

### Phase 02 — CFG-02: Configuration validation ownership

- **Stage 02.1 — Validation inventory:** Classify every rule as schema,
  cross-section semantic, runtime, alias, or logging behavior.
- **Stage 02.2 — Rule ownership map:** Identify duplicate checks between
  `config.py` and `config_model_validation.py`.
- **Stage 02.3 — Validator slice:** Extract or consolidate one rule family
  without changing rejection messages or accepted values.
- **Stage 02.4 — Negative and active-config parity:** Run strict validation,
  duplicate-key, alias, and active configuration tests.
- **Stage 02.5 — Closeout gate:** Freeze the ownership map used by Phase 03.

### Phase 03 — MOD-01: Thesis model lifecycle and mixin graph

- **Stage 03.1 — Lifecycle fence:** Trace constructor, forward, training step,
  phase hooks, memory state, serialization, and checkpoint reload.
- **Stage 03.2 — MRO and caller map:** Identify which mixin methods are true
  lifecycle operations and which are reusable primitives.
- **Stage 03.3 — Composition boundary:** Simplify one lifecycle boundary while
  retaining the public `ThesisMultitaskModel` entrypoint.
- **Stage 03.4 — Model parity:** Run shape, one-step, objective, memory,
  gradient, and checkpoint tests.
- **Stage 03.5 — Closeout gate:** Confirm no hidden second public model path.

### Phase 04 — EVAL-01: Trainer/evaluator metric ownership

- **Stage 04.1 — Metric fence:** Capture window stitching, thresholds,
  pointwise arrays, VUS-PR inputs, uncertainty, and output keys.
- **Stage 04.2 — Duplication map:** Compare trainer validation and evaluator
  reporting line by line at the contract level.
- **Stage 04.3 — Canonical metric slice:** Route only identical logic through
  one helper boundary; preserve trainer/evaluator-specific orchestration.
- **Stage 04.4 — Metric parity:** Run threshold, evaluator, trainer fallback,
  trace, and artifact tests.
- **Stage 04.5 — Closeout gate:** Record the single threshold/metric owner.

### Phase 05 — ONLINE-01: Canonical online runtime path

- **Stage 05.1 — Active-path fence:** Build a caller/config/test matrix for
  online TTA, `online_loop`, model adaptation, and baseline online paths.
- **Stage 05.2 — Canonical-path decision:** Classify each route as active,
  compatibility, or unverified; do not remove unverified routes.
- **Stage 05.3 — Routing slice:** Simplify confirmed duplicate routing with an
  explicit adapter or strategy dispatch.
- **Stage 05.4 — Causal parity:** Run A0/A1/A2, online state, calibration,
  max-step, and artifact-integrity tests.
- **Stage 05.5 — Closeout gate:** Publish the active online path map.

### Phase 06 — RUNTIME-01: Runtime registration lifecycle

- **Stage 06.1 — Registration fence:** Trace registration, clearing, building,
  and import order across runtime entrypoints and tests.
- **Stage 06.2 — Global-state ownership:** Separate real context differences
  from wrappers that only repeat shared registration.
- **Stage 06.3 — Registry slice:** Simplify one registration boundary while
  retaining model and dataset names.
- **Stage 06.4 — Registry parity:** Run registry, config, import, offline,
  online, and evaluation smoke tests.
- **Stage 06.5 — Closeout gate:** Record the supported registration lifecycle.

### Phase 07 — COMPAT-01: Facade and compatibility boundaries

- **Stage 07.1 — Facade fence:** Inventory `sys.modules` facades, wildcard
  exports, aliases, script wrappers, and monkeypatch consumers.
- **Stage 07.2 — Canonical import map:** Give every facade one implementation
  owner and one documented compatibility reason.
- **Stage 07.3 — Explicit adapter slice:** Replace one implicit export with
  explicit forwarding while preserving import and CLI behavior.
- **Stage 07.4 — Compatibility parity:** Run public/legacy import, monkeypatch,
  benchmark wrapper, and CLI tests.
- **Stage 07.5 — Closeout gate:** Freeze the compatibility surface before P1
  implementation phases continue.

### Phase 08 — CFG-03: Config field and alias ownership

- **Stage 08.1 — Field inventory:** Map allow-lists, flat kwargs, aliases, and
  normalization sites to resolved fields.
- **Stage 08.2 — Canonical field map:** Assign one owner to each field and
  legacy name.
- **Stage 08.3 — Alias slice:** Consolidate one field family with explicit
  compatibility handling.
- **Stage 08.4 — Config/model parity:** Compare YAML construction and flat-kwarg
  model construction, including checkpoint metadata.
- **Stage 08.5 — Closeout gate:** Document field ownership.

### Phase 09 — CFG-04: Public defaults and benchmark overrides

- **Stage 09.1 — Default fence:** Capture API defaults, shared YAML defaults,
  generator constants, and experiment overrides.
- **Stage 09.2 — Drift classification:** Mark each difference intentional,
  historical, or unresolved.
- **Stage 09.3 — Documentation slice:** Simplify default selection visibility;
  do not silently change window or stride values.
- **Stage 09.4 — Resolution parity:** Load representative configs and run API
  default tests.
- **Stage 09.5 — Closeout gate:** Record the single documented interpretation.

### Phase 10 — MOD-02: Thesis forward routing

- **Stage 10.1 — Forward fence:** Capture input validation, encoder, memory,
  fusion, uncertainty, and output assembly order.
- **Stage 10.2 — Branch map:** Identify repeated optional prototype/fusion paths.
- **Stage 10.3 — Forward slice:** Name and isolate one stage without changing
  tensor shapes or output keys.
- **Stage 10.4 — Forward parity:** Run shape, deterministic output, phase, and
  Monte Carlo payload tests.
- **Stage 10.5 — Closeout gate:** Record the readable forward flow.

### Phase 11 — MOD-03: Thesis state and memory lifecycle

- **Stage 11.1 — State fence:** Trace initialization, EMA updates, resets,
  serialization, KMeans, filtering, and metadata calibration.
- **Stage 11.2 — Mutation map:** Identify every state mutation and checkpoint
  field owner.
- **Stage 11.3 — State slice:** Separate state serialization from one memory
  algorithm only when the mutation order remains explicit.
- **Stage 11.4 — State parity:** Run memory bootstrap/update/reset and checkpoint
  tests, including anomaly-safe behavior.
- **Stage 11.5 — Closeout gate:** Record state ownership and unresolved failures.

### Phase 12 — MOD-04: Thesis loss-step lifecycle

- **Stage 12.1 — Objective fence:** Capture phase gates, loss weights, forward
  calls, logging, and gradient diagnostics.
- **Stage 12.2 — Step ownership:** Separate calculation from reporting only if
  ordering and side effects are unchanged.
- **Stage 12.3 — Step slice:** Simplify one stage-step path with descriptive
  names and no weight changes.
- **Stage 12.4 — Objective parity:** Run one train step for each active phase,
  objective-control, point-score, and gradient tests.
- **Stage 12.5 — Closeout gate:** Record the active loss contract.

### Phase 13 — EVAL-02: Threshold, UQ, and evaluation artifacts

- **Stage 13.1 — Artifact fence:** Inventory threshold and UQ schemas, writers,
  readers, and benchmark export transformations.
- **Stage 13.2 — Schema ownership:** Assign one validator/builder to each
  artifact type and retain provenance fields.
- **Stage 13.3 — Contract slice:** Consolidate one read/write path without
  changing serialized keys.
- **Stage 13.4 — Artifact parity:** Run schema rejection, roundtrip, threshold,
  UQ, and export tests.
- **Stage 13.5 — Closeout gate:** Freeze artifact contracts.

### Phase 14 — ONLINE-02: Online runtime context

- **Stage 14.1 — Context fence:** Capture data/model/optimizer, calibration,
  runtime state, checkpoint, and finalization fields.
- **Stage 14.2 — Ownership map:** Identify broad dictionaries and their actual
  consumers.
- **Stage 14.3 — Context slice:** Introduce named boundaries only where they
  reduce ambiguity without adding a parallel context model.
- **Stage 14.4 — Runtime parity:** Run dry-run, calibration, checkpoint/resume,
  and artifact tests.
- **Stage 14.5 — Closeout gate:** Record context ownership.

### Phase 15 — ONLINE-03: Online variant dispatch

- **Stage 15.1 — Variant fence:** Record A0/A1/A2 update order, optimizer
  surfaces, losses, and clean-stream gates.
- **Stage 15.2 — Strategy map:** Identify genuinely distinct variant behavior
  versus repeated guards.
- **Stage 15.3 — Dispatch slice:** Make one explicit dispatch simplification.
- **Stage 15.4 — Causal parity:** Compare window sequence, update count,
  parameters, scores, and artifacts for each variant.
- **Stage 15.5 — Closeout gate:** Record variant semantics.

### Phase 16 — ONLINE-04: Stream, cursor, batcher, and persistence

- **Stage 16.1 — Stream fence:** Capture cursor movement, window creation,
  labels, views, batching, and state serialization.
- **Stage 16.2 — Responsibility map:** Separate causal cursor contract from
  view/batch utilities.
- **Stage 16.3 — Stream slice:** Simplify one utility boundary without changing
  the sequence of emitted windows.
- **Stage 16.4 — Stream parity:** Run online stream, state roundtrip, and
  streaming baseline contract tests.
- **Stage 16.5 — Closeout gate:** Record causal stream semantics.

### Phase 17 — DATA-01: Public data API duplication

- **Stage 17.1 — API fence:** Capture SMD and anomaly-archive signatures,
  defaults, required paths, and returned `PublicDataBundle` fields.
- **Stage 17.2 — Shared-option map:** Identify common assembly and dataset-only
  options.
- **Stage 17.3 — API slice:** Consolidate only common option handling.
- **Stage 17.4 — Data parity:** Run loader, shape, scaler, and public API tests
  for both datasets.
- **Stage 17.5 — Closeout gate:** Record dataset API ownership.

### Phase 18 — DATA-02: Dataset parser post-processing

- **Stage 18.1 — Parser fence:** Capture raw parsing, cleaning, labels,
  entities, and metadata for SMD and anomaly archive.
- **Stage 18.2 — Parser ownership:** Separate shared post-processing from
  dataset-specific parsing.
- **Stage 18.3 — Parser slice:** Simplify one post-processing operation with
  golden input/output fixtures.
- **Stage 18.4 — Parser parity:** Compare raw-to-sequence results and metadata.
- **Stage 18.5 — Closeout gate:** Record parser contracts.

### Phase 19 — AUG-01: Synthetic anomaly injector

- **Stage 19.1 — Augmentation fence:** Capture RNG state, family registry,
  masks, labels, metadata, mixture recursion, and batch behavior.
- **Stage 19.2 — Family ownership:** Map common segment operations to each
  family and identify family-specific semantics.
- **Stage 19.3 — Augmentation slice:** Simplify orchestration or metadata
  mechanics, never taxonomy behavior without explicit approval.
- **Stage 19.4 — Augmentation parity:** Run seeded family, mask, class-label,
  metadata, and serialization tests.
- **Stage 19.5 — Closeout gate:** Record taxonomy invariants.

### Phase 20 — RED-01: RedLamp baseline owner

- **Stage 20.1 — Baseline fence:** Capture constructor kwargs, injector setup,
  helper delegation, outputs, and checkpoint behavior.
- **Stage 20.2 — Canonical-owner map:** Identify public, implementation, alias,
  and helper consumers.
- **Stage 20.3 — Baseline slice:** Simplify one constructor/delegation boundary
  with explicit compatibility exports.
- **Stage 20.4 — Baseline parity:** Run RedLamp config, runtime, shape, one-step,
  gradient, and checkpoint tests.
- **Stage 20.5 — Closeout gate:** Record the canonical baseline path.

### Phase 21 — RUN-01: Thesis offline benchmark runner

- **Stage 21.1 — Runner fence:** Capture two-stage execution, evaluator,
  thresholds, traces, reports, checkpoints, and retention artifacts.
- **Stage 21.2 — Responsibility map:** Distinguish runner orchestration from
  reusable artifact operations.
- **Stage 21.3 — Runner slice:** Simplify one read/write-free orchestration
  boundary first.
- **Stage 21.4 — Runner parity:** Run dry-run, artifact export, and one complete
  development-spec combination.
- **Stage 21.5 — Closeout gate:** Preserve manifest and checkpoint provenance.

### Phase 22 — RUN-02: Benchmark config generators

- **Stage 22.1 — Generator fence:** Capture names, output paths, overrides,
  smoke/main differences, and matrix dimensions.
- **Stage 22.2 — Identity ownership:** Separate shared naming/path primitives
  from method-specific config.
- **Stage 22.3 — Generator slice:** Simplify one generator without changing
  generated filenames or YAML keys.
- **Stage 22.4 — Generator parity:** Generate into a temporary directory and
  compare configs and paths.
- **Stage 22.5 — Closeout gate:** Record source-versus-generated ownership.

### Phase 23 — RUN-03: Stage and variant command orchestration

- **Stage 23.1 — Command fence:** Trace comparative, two-stage, offline, and
  online command construction and resume flags.
- **Stage 23.2 — Stage-owner map:** Identify checkpoint and variant decision
  ownership.
- **Stage 23.3 — Command slice:** Simplify one command-builder boundary.
- **Stage 23.4 — Command parity:** Run CLI parsing, dry-run, skip-completed,
  manifest, and one end-to-end combination.
- **Stage 23.5 — Closeout gate:** Preserve stage provenance.

### Phase 24 — REPORT-01: Read-only reporting extraction

- **Stage 24.1 — Reporting fence:** Capture identity, metric, UQ, protocol,
  checkpoint, and source-path fields.
- **Stage 24.2 — Schema map:** Distinguish shared read-only extraction from
  report-specific aggregation.
- **Stage 24.3 — Reporting slice:** Consolidate one read-only loader only.
- **Stage 24.4 — Report parity:** Compare canonical run records and report
  tables with existing artifacts.
- **Stage 24.5 — Closeout gate:** Preserve evidence classification.

### Phase 25 — REPORT-02: Re-evaluation and pruning separation

- **Stage 25.1 — Safety fence:** Inventory all paths that can re-evaluate,
  retain, move, or delete artifacts.
- **Stage 25.2 — Dry-run boundary:** Separate manifest planning from any
  deletion-capable operation.
- **Stage 25.3 — Audit slice:** Simplify only read-only planning first.
- **Stage 25.4 — Safety parity:** Verify exact target manifests and readback;
  no deletion is part of this phase.
- **Stage 25.5 — Closeout gate:** Require explicit approval for later writes.

### Phase 26 — CFG-05: Generated configuration matrix

- **Stage 26.1 — Matrix fence:** Inventory experiment counts, naming dimensions,
  and historical/generated files.
- **Stage 26.2 — Source ownership:** Mark which files are generated and which
  are hand-maintained.
- **Stage 26.3 — Matrix slice:** Simplify generation metadata or templates only.
- **Stage 26.4 — Matrix parity:** Regenerate in a temporary location and diff
  identity, paths, and semantics.
- **Stage 26.5 — Closeout gate:** Document regeneration procedure.

### Phase 27 — MODEL-05: Thesis component/config ownership

- **Stage 27.1 — Component fence:** Trace config classes, encoder, heads,
  memory, and MLP construction.
- **Stage 27.2 — Construction map:** Identify helper primitives versus model
  lifecycle behavior.
- **Stage 27.3 — Component slice:** Simplify one construction boundary without
  creating a second model entrypoint.
- **Stage 27.4 — Component parity:** Compare parameter names, shapes, and
  state-dict contents.
- **Stage 27.5 — Closeout gate:** Record component ownership.

### Phase 28 — EVAL-03: Mathematical metric adapters

- **Stage 28.1 — Metric fence:** Separate formulas, normalization, adapters,
  and output formatting.
- **Stage 28.2 — Formula ownership:** Mark cohesive mathematical code as
  protected from cosmetic splitting.
- **Stage 28.3 — Adapter slice:** Remove or consolidate only duplicate adapters.
- **Stage 28.4 — Metric parity:** Run golden-value and report-output tests.
- **Stage 28.5 — Closeout gate:** Record unchanged metric definitions.

### Phase 29 — CLI-01: Comparative runner namespace

- **Stage 29.1 — CLI fence:** Inventory wildcard names, parser fields, run
  records, and worker command interfaces.
- **Stage 29.2 — Namespace map:** Define explicit support-module interfaces.
- **Stage 29.3 — CLI slice:** Replace one wildcard dependency with explicit
  imports or a named support boundary.
- **Stage 29.4 — CLI parity:** Run parser, plan, record, and launcher tests.
- **Stage 29.5 — Closeout gate:** Record supported CLI surface.

### Phase 30 — COMPAT-02: Legacy aliases and flags

- **Stage 30.1 — Alias fence:** Inventory stage names, baseline aliases, flat
  kwargs, and legacy flags.
- **Stage 30.2 — Compatibility map:** Link each alias to its canonical field
  and expiration/status rule.
- **Stage 30.3 — Alias slice:** Centralize one alias family without deleting
  historical support.
- **Stage 30.4 — Fixture parity:** Run legacy config and checkpoint fixtures.
- **Stage 30.5 — Closeout gate:** Record intentional compatibility only.

### Phase 31 — STATIC-01: Static import/readability cleanup

- **Stage 31.1 — Signal fence:** Re-run Ruff and classify unused imports,
  wildcard names, and E402 findings by active path.
- **Stage 31.2 — Deadness proof:** Verify each candidate has no runtime,
  monkeypatch, notebook, or CLI consumer.
- **Stage 31.3 — Static slice:** Remove one proven-dead import/export group.
- **Stage 31.4 — Static parity:** Run Ruff, import tests, and relevant CLIs.
- **Stage 31.5 — Closeout gate:** Keep only actionable remaining signals.

### Phase 32 — DEMO-01: Demo and replay entrypoints

- **Stage 32.1 — Demo fence:** Trace documented demo inputs, outputs, and
  runtime paths.
- **Stage 32.2 — Entry-point map:** Identify duplicate data loading or wrapper
  behavior.
- **Stage 32.3 — Demo slice:** Route one demo through the canonical API.
- **Stage 32.4 — Demo parity:** Run each affected demo fixture and queue test.
- **Stage 32.5 — Closeout gate:** Update demo usage if paths changed.

### Phase 33 — DOC-01: Documentation path and terminology drift

- **Stage 33.1 — Documentation fence:** Find stale `documents/design` paths and
  inactive phase names.
- **Stage 33.2 — SSOT map:** Mark current versus historical documents and terms.
- **Stage 33.3 — Documentation slice:** Correct one path/term group without
  changing scientific meaning.
- **Stage 33.4 — Link parity:** Run path/link and active-config consistency
  checks.
- **Stage 33.5 — Closeout gate:** Publish the final active terminology map.

