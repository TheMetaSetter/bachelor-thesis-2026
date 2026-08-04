---
date: 2026-07-26
researcher: TheMetaSetter
repository: bachelor-thesis-2026
source_research: documents/logs/07-26-2026/research/research-code-simplification-hotspots.md
source_vision: documents/logs/07-26-2026/plan/vision_code-simplification-program.md
status: preliminary-plan
---

# Preliminary Programming Plan: Simplifying 33 Code Areas

## Current state

The research note identifies 33 simplification areas across configuration,
data preparation, model lifecycle, training/evaluation, online adaptation,
benchmark orchestration, compatibility, reporting, and documentation. The
main problem is distributed ownership: one runtime concept often has a public
facade, an internal implementation, helper modules, and a second legacy path.

The active repository contracts are the data batch dictionary with
`x=[B,L,D]`, encoder hidden states with `hidden=[B,L,H]`, thesis outputs with
reconstruction/classification/point-score/auxiliary fields, causal online
window processing, two-stage offline execution, checkpoint role metadata, and
report artifacts with preserved provenance.

The source of truth for the simplification candidates is
`documents/logs/07-26-2026/research/research-code-simplification-hotspots.md`.
The design source path used by this repository is
`documents/abstract-design-notes/`, not the stale `documents/design/` path in
the prompt templates.

## Planning principles

Each phase has exactly one target area. Each phase is executed in the following
order, with the names refined in the structure and detail documents:

1. establish a behavioral fence from callers, tests, configs, and artifacts;
2. identify the canonical owner and all compatibility consumers;
3. make one small behavior-preserving simplification;
4. run focused parity tests, lint/compile checks, and the smallest relevant
   smoke flow;
5. record the result, residual risks, and the next phase gate.

The code-simplification skill requires incremental changes, no speculative
rewrites, no removal of error handling, and no simplification before the reason
for an existing boundary is understood. The repository additionally requires
methods/functions below 50 lines and files below 500 lines, but those limits
are not a reason to split a coherent model or algorithm mechanically.

## Dependency-aware phase order

The order below is from highest priority to lowest priority. The first seven
phases are the first delivery target. P0 areas are completed before P1 areas;
within P1, dependencies determine the order.

| Phase | Area ID | Priority | Target area | Primary files | Main dependency | Phase outcome |
|---:|---|---|---|---|---|---|
| 01 | CFG-01 | P0 | Configuration orchestration | `src/core/config.py:903-988` | None | One explicit load/merge/normalize/validate orchestration path. |
| 02 | CFG-02 | P0 | Configuration validation ownership | `src/core/config_model_validation.py:44-700`, related config validators | 01 | Schema, cross-section, and runtime validation responsibilities are explicit. |
| 03 | MOD-01 | P0 | Thesis model lifecycle and mixin graph | `src/models/thesis_multitask.py`, `src/models/thesis_multitask_impl/` | 01-02 | One readable public model flow with no hidden lifecycle ownership. |
| 04 | EVAL-01 | P0 | Trainer/evaluator metric overlap | `src/engine/trainer.py`, `src/engine/evaluator.py`, `src/engine/thresholding.py` | 01-03 | One canonical reconstruction/threshold/metric boundary. |
| 05 | ONLINE-01 | P0 | Canonical online runtime path | `src/engine/online_tta/`, `src/engine/online_loop.py`, `src/baselines/online/` | 01-04 | Active online paths are classified; redundant routing is not silently removed. |
| 06 | RUNTIME-01 | P1 | Runtime registration lifecycle | `src/core/runtime_components.py`, `src/core/registry.py` | 01, 05 | Registration ownership and global lifecycle are explicit. |
| 07 | COMPAT-01 | P1 | Facade and compatibility boundaries | `src/models/online_adaptation.py`, `src/models/redlamp_baseline.py`, `scripts/run_*.py` | 03, 05-06 | Canonical imports and narrow legacy adapters are explicit. |
| 08 | CFG-03 | P1 | Config field and alias ownership | config validators and `thesis_multitask_config_parsing.py` | 01-02, 07 | Each field and legacy alias has one canonical owner. |
| 09 | CFG-04 | P1 | Public default versus benchmark override drift | `src/data/api.py`, `configs/`, benchmark generators | 01-02 | Intentional defaults and benchmark overrides are documented and tested. |
| 10 | MOD-02 | P1 | Thesis forward routing | `thesis_multitask_routing_forward_helpers.py`, routing mixin | 03 | Forward stages and optional paths are named and traceable. |
| 11 | MOD-03 | P1 | Thesis state and memory lifecycle | state/memory mixins and serialization helpers | 03, 10 | Memory state, algorithms, and metadata ownership are separated. |
| 12 | MOD-04 | P1 | Thesis loss-step lifecycle | loss/step mixins and objective helpers | 03, 10-11 | Loss gates and stage-step order are explicit. |
| 13 | EVAL-02 | P1 | Threshold/UQ/evaluation artifact contracts | `src/protocols/threshold_artifact.py`, `src/core/uq_summary.py`, evaluator exports | 04 | Artifact schemas have one owner per output type. |
| 14 | ONLINE-02 | P1 | Online runtime context | `online_engine_run.py`, online state/checkpoint helpers | 05, 13 | Context data has named boundaries and stable metadata. |
| 15 | ONLINE-03 | P1 | Online variant dispatch | `online_engine_step.py`, online loss helpers | 05, 14 | A0/A1/A2 dispatch preserves causal and update semantics. |
| 16 | ONLINE-04 | P1 | Stream/cursor/batcher ownership | `src/data/stream.py`, online stream tests | 05, 09 | Stream, view, batching, and persistence responsibilities are clear. |
| 17 | DATA-01 | P1 | Public data API duplication | `src/data/api.py`, `src/data/loaders.py` | 09, 16 | Dataset-specific API differences are explicit; shared assembly is singular. |
| 18 | DATA-02 | P1 | Dataset parser post-processing | `src/data/datasets/smd.py`, `anomaly_archive.py` | 16-17 | Parser-specific and shared metadata/cleaning logic are separated. |
| 19 | AUG-01 | P1 | Synthetic anomaly injector ownership | `src/data/augment.py` | 03, 09, 18 | Taxonomy and labels remain identical with clearer orchestration. |
| 20 | RED-01 | P1 | RedLamp baseline implementation and delegators | `src/models/baseline_impl/redlamp_baseline.py`, helpers and shims | 07, 19 | One canonical baseline owner with explicit compatibility exports. |
| 21 | RUN-01 | P1 | Thesis offline benchmark runner | public runner and internal helpers | 04, 13, 20 | Runner orchestration and artifact export ownership are clear. |
| 22 | RUN-02 | P1 | Benchmark config generators | offline/online/SMD generator scripts | 08-09, 21 | Shared identity/path generation is tested without changing filenames. |
| 23 | RUN-03 | P1 | Stage/variant command orchestration | two-stage, thesis, comparative runners | 05, 20-22 | Stage checkpoint and command decisions have one owner. |
| 24 | REPORT-01 | P1 | Read-only reporting data extraction | `scripts/ops/`, benchmark summary scripts | 13, 21-23 | Shared identity/metric reads preserve report-specific schemas. |
| 25 | REPORT-02 | P1 | Re-evaluation/pruning separation | `scripts/ops/re_evaluate_and_prune_thesis_runs.py` | 24 | Audit planning is separated from deletion-capable execution. |
| 26 | CFG-05 | P2 | Generated configuration matrix | `configs/experiment/`, generators | 22 | Generated/source ownership is documented and reproducible. |
| 27 | MODEL-05 | P2 | Thesis component/config ownership | components, encoder, head, memory construction files | 10-12 | Component construction is discoverable without a second model entrypoint. |
| 28 | EVAL-03 | P2 | Mathematical metric adapters | `src/metrics/pointwise.py`, `affiliation.py` | 04, 13 | Only duplicated adapters are removed; formulas remain stable. |
| 29 | CLI-01 | P2 | Comparative runner namespace | comparative runner/support module | 23 | Explicit interfaces replace wildcard namespace dependence. |
| 30 | COMPAT-02 | P2 | Legacy aliases and flags | config/model compatibility paths | 08, 20, 29 | Alias inventory is centralized and fixture-tested. |
| 31 | STATIC-01 | P2 | Static import/readability cleanup | `src/`, `scripts/`, `demo/` Ruff findings | 07, 29-30 | Proven unused imports and implicit exports are cleaned incrementally. |
| 32 | DEMO-01 | P2 | Demo/replay entrypoints | `demo/`, replay scripts | 05, 07, 31 | Demos call canonical runtime surfaces directly. |
| 33 | DOC-01 | P2 | Documentation path and terminology drift | prompts and `documents/` references | all runtime phases | SSOT paths and active/legacy terminology are unambiguous. |

## First seven phases: programming scope

### Phase 01 — CFG-01: Configuration orchestration

Inspect `load_experiment_config` and all callers before extracting anything.
Preserve reference resolution, section merging, alias normalization, model
window resolution, validation order, logging, and error behavior. The target is
an orchestration function that delegates to named existing validators and path
resolvers, not a new configuration framework. Verify every active benchmark,
smoke, online, baseline, and comparative family.

### Phase 02 — CFG-02: Configuration validation ownership

Inventory the allowed-key, type, value, cross-section, and runtime checks in
`src/core/config.py`, `src/core/config_model_validation.py`, and related
modules. Establish one owner for each rule. Preserve strict rejection,
legacy-alias normalization, and active config compatibility. Run the full
configuration-focused test group before and after each small extraction.

### Phase 03 — MOD-01: Thesis model lifecycle

Trace construction, `forward`, `training_step`, phase hooks, memory state,
serialization, and checkpoint loading from the public class through the mixins
and helpers. Prefer composition or explicitly named helper ownership over a
new inheritance layer. Do not delete a mixin until callers and tests prove that
its methods are not an independent public contract.

### Phase 04 — EVAL-01: Trainer/evaluator metric ownership

Compare trainer validation aggregation and evaluator reporting at the level of
window stitching, point-score threshold source, VUS-PR/pointwise metric input,
uncertainty payloads, and output key names. Extract only behaviorally identical
logic to a shared boundary. Retain trainer-only checkpoint monitoring and
evaluator-only report assembly where their responsibilities differ.

### Phase 05 — ONLINE-01: Canonical online runtime path

Build a caller/config/test matrix for `online_tta`, `online_loop`, online
adaptation, and baseline online modules. Mark each path active, compatibility,
or unverified. Do not remove a path during this phase. Simplify only confirmed
duplicate routing after a representative A0/A1/A2 smoke and state replay test.

### Phase 06 — RUNTIME-01: Registration lifecycle

Trace where datasets and models are registered, cleared, and built for offline,
evaluation, online, benchmark, and tests. Retain the existing registry contract
and avoid adding a factory layer. Remove only wrappers that add no distinct
context behavior after import and re-registration tests pass.

### Phase 07 — COMPAT-01: Facade boundaries

Inventory `sys.modules` replacement, wildcard exports, renamed baseline aliases,
and script wrappers. For each, identify canonical implementation, legacy
consumer, and test monkeypatch surface. Replace implicit exports only in small
increments with explicit adapter exports, preserving import identity and CLI
behavior.

## Global verification policy

Every phase must use `.venv/bin/python` and focused `pytest` invocations. The
minimum verification bundle is config loading, registry behavior, the relevant
shape/forward/backward test, checkpoint roundtrip, and the smallest relevant
offline or online smoke. Full benchmark matrices are prohibited until one
concrete end-to-end combination passes after the relevant phase.

No phase may modify tests merely to accept a changed result. If a test exposes a
pre-existing failure, record it as a baseline blocker and separate diagnosis
from simplification. Artifact and pruning phases must remain read-only until a
later explicit user authorization covers an exact target.

## Open decisions that remain outside this preliminary plan

The plan does not decide whether `max_online_steps: null` is intentional, which
online implementation is canonical for every configuration, whether a legacy
facade is still externally consumed, or whether any model mixin can be removed.
Those decisions require the phase-specific evidence gates described above.

