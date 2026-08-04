---
date: 2026-07-26T22:29:24+07:00
researcher: TheMetaSetter
repository: bachelor-thesis-2026
git_commit: aae5490a5117a4a79ef4fa3ce69b4fcdc6b690ec
branch: dev
topic: Evidence-based code-simplification areas
tags: [research, code-simplification, refactoring, multivariate-tsad]
status: complete
last_updated: 2026-07-26
last_updated_by: TheMetaSetter
---

# Research: Areas That Need Simplification in the Current Codebase

## Research question

Which parts of the current multivariate time-series anomaly-detection codebase
have the highest cognitive load, duplicated responsibility, compatibility debt,
or unclear ownership, while keeping the existing runtime behavior and public
contracts unchanged?

## Scope and method

This note applies `prompts/1_research_prompt.md` to a simplification-oriented
research question. The prompt says to document the codebase as it exists and to
verify the data, model, online adaptation, evaluation, and reporting contracts
before proposing changes. The requested simplification assessment is therefore
limited to observed structural complexity. It does not approve an
implementation, remove a path, change a default, or select a new architecture.

The prompt refers to `documents/design/idea.md` and
`documents/design/design_starter.md`; that directory is not present in the
current checkout. The corresponding design sources actually present under the
repository SSOT are `documents/abstract-design-notes/idea.md`,
`documents/abstract-design-notes/design_starter.md`,
`documents/abstract-design-notes/stream_design.md`, and
`documents/abstract-design-notes/codebase-modernization-simple-refactor-plan.md`.
Those files were used instead of guessing a missing path.

The inspection covered the active `src/` runtime, benchmark and experiment
entrypoints under `scripts/`, experiment/configuration families under
`configs/`, relevant tests, and the current design notes. File size was used as
a signal, not as proof: a long mathematical metric implementation can be
cohesive, while a short facade can still create a second public path.

## Summary

The largest simplification opportunity is to make ownership explicit. The
current code has a valid-looking high-level flow, but the same responsibility
often appears in several layers:

```text
experiment YAML
  -> src/core/config.py
  -> config_model_validation.py and related validators
  -> data API/loaders/window or stream
  -> model public class + mixins + helpers
  -> trainer or online TTA engine
  -> evaluator/threshold/metric assembly
  -> benchmark runner and report/artifact scripts
```

The highest-priority areas are:

1. configuration loading and validation ownership;
2. the thesis multitask model's mixin and helper graph;
3. trainer/evaluator overlap around reconstruction, thresholding, and
   pointwise metrics;
4. public facades, wildcard imports, and legacy entrypoints;
5. the coexistence of multiple online-adaptation paths;
6. benchmark runners and generated configuration builders that repeat the
   same experiment identity and artifact logic.

The current active benchmark evidence also shows a contract/default drift that
must be resolved before changing shared helpers. Public data API defaults are
`window_size=100` and `stride=10` (`src/data/api.py:14-61`), while the active
benchmark configs and generator constants use `window_size=20` and usually
override the stride (`configs/data/smd_benchmark_machine_1_6_window20.yaml`,
`scripts/benchmarks/generate_online_benchmark_configs.py:23-28`). A loaded online
smoke config explicitly resolves `max_online_steps` to `None`, even though its
base task config declares `16`; that is an observed configuration value, not a
recommendation to change it.

## Current pipeline and contracts

### Data preparation

The public data API builds separate SMD and anomaly-archive dictionaries with
nearly the same window, stride, batch, worker, validation, and maximum-window
arguments (`src/data/api.py:14-118`). It then exposes two near-identical loader
wrappers (`src/data/api.py:121-198`). The shared lower-level loader already
contains the more canonical sequence-to-window pipeline:

```text
dataset parser
  -> clean/validate sequences
  -> fit scaler on train data
  -> scale train/validation/test
  -> create WindowDataset instances
  -> create DataLoader instances
```

`WindowDataset` materializes index triples and creates windows on demand in
`src/data/loaders.py:150-383`. `src/data/window.py:17-83` contains the core
window slicing policy. `src/data/stream.py:38-187` adds an online cursor and
state persistence; `OnlineWindowBatcher` in `src/data/stream.py:194-267` adds
batch views, state, and serialization in the same module.

The batch contract observed in the code is a dictionary containing at least
`x`, `point_labels`, and `meta`. The documented tensor shape is
`x=[B,L,D]`; point labels are window-aligned and are converted to one label per
window by `src/data/api.py:201-206`. Baseline adapters flatten the same tensor
to `[B,L*D]` in `src/data/api.py:209-215`. Any data simplification must retain
these shapes, metadata, scaler provenance, entity boundaries, and the causal
stream cursor.

### Synthetic anomaly augmentation

`SyntheticAnomalyInjector` owns an eleven-family taxonomy, random-state
serialization, segment selection, affected-channel selection, mutation,
metadata, labels, and batch orchestration (`src/data/augment.py:21-118`). The
family methods occupy most of the file; many use the shared
`_apply_segment_update` path (`src/data/augment.py:301-317`), and the mixture
family dispatches back through the family registry.

This is a high-cognitive-load area, but the taxonomy, `synthetic_anomaly_mask`,
classification labels, family metadata, and deterministic RNG state are model
and experiment contracts. The evidence supports simplifying ownership and
repeated mechanics only after family-by-family parity tests; it does not support
deleting families or changing injection semantics from this reconnaissance.

### Modeling and training

The public thesis model is a small class, but its lifecycle is distributed
through four mixins and many helper modules
(`src/models/thesis_multitask.py:45-78`). The constructor delegates encoder,
prototype memory, fusion parameters, task heads, synthetic injectors, optional
loss configuration, and phase-specific trainability to different methods.
The forward implementation is in
`src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py:82-330`
and combines input validation, encoding, anomaly masks, continuous/discrete
memory updates, prototype lookup, fusion, uncertainty, and output assembly.

The effective model output contract is a dictionary with reconstruction,
classification, point-score, and auxiliary outputs. The auxiliary payload is
also used for uncertainty, Monte Carlo samples, traces, and diagnostics by the
evaluation path. The hidden representation is produced by the encoder as
`hidden=[B,L,H]`. This means that reducing the number of files is safe only if
phase hooks, memory state, checkpoint state, output keys, and gradient behavior
remain traceable and unchanged.

Training orchestration is concentrated in a 394-line `Trainer.train`
implementation (`src/engine/trainer.py:534-927`). It coordinates model phase
hooks, synthetic preparation, memory initialization, optimization, validation,
checkpoint selection, diagnostics, and artifact state. The long method is not
automatically redundant, but its responsibilities are mixed.

### Online adaptation

The public online TTA module is a facade that re-exports the implementation and
also uses wildcard imports from shared, step, window-core, and window-metric
modules (`src/engine/online_tta/online_engine.py:9-25`). The runtime sequence
builder is in `src/engine/online_tta/online_engine_run.py:74-252`; it creates
data/model/optimizer context, calibrates per entity, processes windows, and
synchronizes runtime state. Variant-specific updates are in
`src/engine/online_tta/online_engine_step.py:104-234`.

There is also an older `src/engine/online_loop.py:1-219` path, a separate
`src/models/online_impl/online_adaptation.py` implementation behind
`src/models/online_adaptation.py`, and several baseline online modules under
`src/baselines/online/`. The current evidence shows multiple online concepts,
not yet a proof that all are active for the same benchmark. This is therefore a
high-value ownership investigation before any removal.

### Evaluation and reporting

`Evaluator.evaluate` accumulates window outputs back onto entity timelines and
stores pointwise payloads and traces (`src/engine/evaluator.py:415-554`). The
trainer independently reconstructs pointwise records, selects a validation
threshold at quantile `0.99`, computes pointwise metrics, and assembles stage
keys (`src/engine/trainer.py:482-532`). This is the clearest observed overlap.

Benchmark execution is also split between a large public runner and an internal
helper with many corresponding responsibilities:
`scripts/benchmarks/run_thesis_offline_benchmark.py:55-758` and
`scripts/benchmarks/_internal/run_thesis_offline_benchmark_helpers.py:55-619`.
The code writes reports, score payloads, traces, thresholds, UQ summaries, and
retention artifacts. Shared artifact identity and provenance must remain intact;
report scripts should not be merged merely because they all read JSON.

## Candidate simplification areas

Priority means expected reduction in reader effort and ownership ambiguity. It
does not mean implementation order or permission to edit. “Parity required” is
the minimum evidence needed before changing that surface.

| ID | Priority | Area and evidence | Expected simplification | Risk | Parity required |
|---|---|---|---|---|---|
| CFG-01 | P0 | `src/core/config.py:903-988` mixes reference resolution, section merging, alias normalization, window resolution, validation, and logging. | Keep one thin orchestration path with clear stage ownership. | High: active YAML behavior. | Load every active benchmark family and compare resolved dictionaries. |
| CFG-02 | P0 | `src/core/config_model_validation.py:44-700` contains very large model/task validation plus data-runtime checks that overlap `src/core/config.py`. | Separate schema, cross-section semantics, and runtime checks without duplicate rules. | High: invalid configs may be accepted/rejected differently. | Existing config-loading, alias, duplicate-key, and negative-value tests. |
| CFG-03 | P1 | Key allow-lists and semantic branches are spread across config modules; flat thesis kwargs are parsed separately in `src/models/thesis_multitask_impl/thesis_multitask_config_parsing.py:39-244`. | Make each field's canonical owner and alias visible. | High: checkpoint/config compatibility. | Snapshot resolved configs and model construction from YAML and flat kwargs. |
| CFG-04 | P1 | Public data API defaults are 100/10, while active benchmark generators use `WINDOW_SIZE=20` and generated overrides. | Make intentional defaults versus benchmark overrides explicit and singular. | High: changes experiment identity. | API tests plus resolved active-config matrix; no default change without approval. |
| CFG-05 | P2 | Approximately 110 thesis online, 72 STUMPY, 72 KMeans-AD, 72 IForest, and many other generated experiment files repeat structural fields. | Reduce generator/template duplication and document generated-versus-source ownership. | High: file names and output paths are provenance. | Regenerate into a temporary directory and compare YAML and paths byte-for-byte. |
| RUNTIME-01 | P1 | `src/core/runtime_components.py:14-32` exposes shared, offline, evaluation, and online registration wrappers; registry state is global in `src/core/registry.py:8-55`. | Clarify one registration lifecycle and keep only real context differences. | Medium/high: import and registry tests. | Clear/re-register tests for every entrypoint and model family. |
| MOD-01 | P0 | `src/models/thesis_multitask.py:45-78` uses four mixins; state, routing, loss, setup, and helper files together define one model lifecycle. | Reduce MRO and cross-file navigation; make lifecycle phases locally traceable. | Very high: phase hooks, memory, checkpoints, outputs. | Forward/backward, phase transition, memory initialization, checkpoint roundtrip, and output-key snapshots. |
| MOD-02 | P1 | `thesis_multitask_routing_forward_helpers.py:82-330` combines validation, encoding, prototype memories, fusion, uncertainty, and assembly. | Give forward stages explicit boundaries and reduce repeated optional-path branches. | Very high: tensor and score semantics. | `[B,L,D]`/`[B,L,H]` shape tests, deterministic outputs, A/B/C stage behavior, MC payload tests. |
| MOD-03 | P1 | `thesis_multitask_state_memory_mixin.py:1-676` combines state serialization, memory initialization, KMeans, EMA updates, anomaly-safe filtering, and metadata calibration. | Separate state ownership from memory algorithms and reporting metadata. | Very high: training dynamics and checkpoint state. | Initialization, EMA, reset, serialization, and anomaly-safe adaptation tests. |
| MOD-04 | P1 | `thesis_multitask_loss_step_mixin.py:16-374` combines stage logging, contrastive preparation, forward calls, loss gates, and gradient diagnostics. | Make one step's order and phase gates readable without duplicating stage wrappers. | High: loss weighting and gradient behavior. | One train step per phase, loss dictionaries, gradient-profile tests. |
| MODEL-05 | P2 | `thesis_multitask_components.py` and helper modules contain config classes, encoder, heads, memory and MLP construction alongside model-facing names. | Clarify component/config ownership; retain one canonical construction path. | Medium/high. | Config roundtrip and model parameter/state-dict comparison. |
| EVAL-01 | P0 | Trainer pointwise aggregation and evaluator both reconstruct timeline records and assemble threshold/metric outputs (`src/engine/trainer.py:482-532`, `src/engine/evaluator.py:415-554`). | One canonical reconstruction/threshold/metric boundary. | Very high: reported metrics. | Threshold, VUS-PR, pointwise metric, entity stitching, and trainer/evaluator parity tests. |
| EVAL-02 | P1 | Threshold artifacts, UQ summaries, evaluator payloads, and benchmark exports each validate or reshape related metadata. | Make one explicit artifact contract per output type. | High: audit/provenance. | Read/write roundtrip and schema rejection tests with existing artifacts. |
| EVAL-03 | P2 | `src/metrics/pointwise.py` and `src/metrics/affiliation.py` are long but mostly mathematical; wrappers and normalization helpers may still repeat. | Remove only duplicate adapters, not cohesive formulas. | Medium: scientific metric semantics. | Golden-value tests against current outputs. |
| ONLINE-01 | P0 | `src/engine/online_tta/online_engine.py` is a facade over multiple modules, while `src/engine/online_loop.py` is another loop and baseline online classes form another family. | Establish canonical active path and narrow compatibility boundaries. | Very high: protocol and causal order. | Caller/config inventory, import tests, one full online smoke per variant, state replay. |
| ONLINE-02 | P1 | Online context construction, calibration, runtime state, and artifact finalization are concentrated in `online_engine_run.py:74-367`. | Replace broad context dictionaries with named ownership boundaries. | High: checkpoint and threshold metadata. | Dry-run context snapshot, per-entity calibration, checkpoint/resume, artifact comparison. |
| ONLINE-03 | P1 | `online_engine_step.py:104-234` branches over A0/A1/A2-like update behavior and loss/optimizer conditions. | Make variant dispatch explicit and remove repeated guards only after semantics are mapped. | Very high: adaptation protocol. | A0/A1/A2 causal-window and update-count tests. |
| ONLINE-04 | P1 | `src/data/stream.py:38-267` combines cursor, next-window behavior, online view construction, batching, and persistence. | Separate stream cursor contract from batch/view and serialization concerns. | High: causal alignment. | State replay and exact window/label/meta sequence comparison. |
| DATA-01 | P1 | `src/data/api.py:14-118` duplicates SMD and anomaly-archive config construction; `:121-198` duplicates public wrappers. | Share common option assembly while retaining dataset-specific required fields. | Medium/high: defaults and path handling. | Public API signature/config snapshots for both datasets. |
| DATA-02 | P1 | `src/data/datasets/smd.py:62-182` and `src/data/datasets/anomaly_archive.py:90-166` have large parser methods with overlapping cleaning/metadata stages. | Isolate shared parser post-processing from dataset-specific parsing. | High: labels, entity IDs, cleaning metadata. | Raw-to-parsed sequence and metadata golden snapshots. |
| AUG-01 | P1 | `src/data/augment.py:38-972` owns registry, RNG, segment mechanics, eleven family implementations, mixture recursion, labels, masks, and batch orchestration. | Separate orchestration/metadata from family algorithms and share common mechanics. | Very high: synthetic taxonomy and training labels. | Seeded family parity, masks, class labels, metadata, and serialized RNG tests. |
| RED-01 | P1 | `src/models/baseline_impl/redlamp_baseline.py:32-470` has a 192-line constructor, repeated injector construction, and many delegations to helpers; public shims include wildcard import and alias modules. | One canonical baseline owner plus explicit, narrow compatibility exports. | High: baseline benchmark identity. | Legacy/public import tests, state-dict, one-step, checkpoint, and benchmark smoke. |
| RUN-01 | P1 | Offline runner and internal helper each contain config loading, model construction, two-stage execution, evaluator creation, thresholding, and artifact export. | Keep runner orchestration thin and centralize only proven shared artifact operations. | Very high: reproducibility and retention. | Manifest, report, score, trace, threshold, UQ, and checkpoint path comparison. |
| RUN-02 | P1 | Offline, online, and SMD config generators repeat entity naming, output paths, overrides, YAML writing, and matrix loops. | Shared identity/path primitives with method-specific overrides. | High: output layout and skip-completed behavior. | Temporary regeneration diff and path-contract tests. |
| RUN-03 | P1 | Two-stage pretraining, thesis offline/online benchmark, and comparative SMD runners each encode stage/variant decisions in separate command paths. | Make stage/variant decision ownership explicit and remove repeated command assembly. | Very high: stage checkpoint provenance. | One concrete end-to-end combination before any matrix expansion. |
| REPORT-01 | P1 | Report/ops scripts independently load JSON, resolve experiment roots, extract identities, normalize metrics, and build audit tables. | Share read-only identity/metric extraction while preserving report-specific schemas. | High: evidence classification and provenance. | Compare compact reports and manifests against current retained artifacts. |
| REPORT-02 | P1 | `re_evaluate_and_prune_thesis_runs.py` combines re-evaluation, retention decisions, manifest creation, and deletion-capable operations. | Separate read-only audit planning from approved pruning execution. | Very high: destructive artifact scope. | Dry-run manifest, explicit target list, and readback after any approved action. |
| CLI-01 | P2 | `scripts/experiments/run_comparative_smd_experiments.py:4` wildcard-imports internal support and both files are large. | Replace implicit namespace dependence with explicit command/record interfaces. | Medium/high: CLI and experiment records. | CLI parsing, plan serialization, and run-record snapshots. |
| COMPAT-01 | P1 | Facade files use `sys.modules` replacement or wildcard exports, including `src/models/online_adaptation.py:1-4`, `src/models/redlamp_baseline.py:1`, and benchmark script wrappers. | Keep compatibility files visibly thin and make canonical imports discoverable. | Medium/high: tests monkeypatch public facades. | Import identity and monkeypatch behavior tests. |
| COMPAT-02 | P2 | Stage aliases, renamed baseline names, flat kwargs, and legacy flags are normalized in several locations. | Inventory each alias and route it through one compatibility boundary. | High: historical configs/checkpoints. | Legacy config and checkpoint fixture matrix. |
| STATIC-01 | P2 | Ruff reports unused imports, wildcard F405/F403, and module-order E402 signals across analysis, experiment, demo, and wrapper scripts. | Remove proven unused imports and implicit namespaces after caller inspection. | Low individually, medium for wrappers. | Ruff plus import/CLI smoke tests. |
| DEMO-01 | P2 | Demo/replay scripts have separate online/offline paths and static import signals. | Keep demos as thin calls into canonical runtime APIs. | Medium: user-facing examples. | Run each documented demo with its fixture/config. |
| DOC-01 | P2 | Prompt and design references use `documents/design`, while the live SSOT uses `documents/abstract-design-notes`. | Normalize documentation paths and mark historical references. | Low runtime, high reader confusion. | Link/path audit; no code behavior change. |

## Priority interpretation

P0 areas should be researched first because they affect several downstream
consumers and already have direct evidence of overlapping responsibility. P1
areas are substantial but depend on the P0 ownership decisions. P2 areas are
cleanup or documentation opportunities that should follow active-path and
contract verification.

The following files are large but should not be split merely to satisfy a line
count: mathematical metric implementations, parser logic that expresses a
single dataset contract, and artifact validators whose long schema is the
actual source of truth. The simplification target is duplicated decision-making
and unclear ownership, not file size by itself.

## Historical context

`documents/abstract-design-notes/codebase-modernization-simple-refactor-plan.md:7-47`
already identifies behavior-preserving simplification, narrow compatibility
boundaries, `src/core/config.py`, trainer/evaluator overlap, and the RedLamp
compatibility shim as intended modernization surfaces. Its documented passes
also call for config loading checks, public-entrypoint tests, metric parity, and
model-contract preservation (`:79-150`, `:190-193`, `:275-319`). The current
reconnaissance confirms those themes and expands them with the online duplicate
path, data/augmentation ownership, generated config matrix, and report-script
provenance concerns.

The design notes also describe a public thesis model with internal mixin
modules. That structure has reduced individual file size, but it has not
removed the model lifecycle's cross-file navigation cost. The current report
therefore treats the mixin graph as a simplification candidate while preserving
the public model class and all active contracts.

## Open questions before implementation

1. Which online path is canonical for each active configuration family:
   `online_tta`, `online_loop`, or a baseline-specific online implementation?
2. Is `max_online_steps: null` intentionally unlimited in smoke and main
   configs, or is the base task value `16` intended to survive the generated
   override? This must be resolved from the benchmark specification and tests,
   not from the field name.
3. Which model mixin owns each public hook, state key, and phase transition, and
   which helper is allowed to mutate prototype memory?
4. Are trainer-side validation metrics and evaluator-side reported metrics
   required to use exactly the same threshold source and aggregation policy?
5. Which compatibility imports are still used by notebooks, tests, old
   checkpoints, or external benchmark scripts?
6. Which report and pruning artifacts are primary evidence, and which are
   derived views that can safely share loaders or normalizers?
7. Is the distinction between public API defaults (`window_size=100`) and active
   benchmark defaults (`window_size=20`) intentional and documented in one SSOT
   location?
8. Does every synthetic anomaly family need the current metadata and mask
   fields, or is some metadata only used by diagnostics? This requires a caller
   inventory before extracting or deleting fields.
9. Which current test failures are pre-existing configuration/memory contract
   failures versus failures caused by a candidate simplification? The previous
   focused verification observed 442 passed, 1 skipped, and 10 failed in the
   full suite, so green status must not be assumed.

## Verification performed

The repository metadata at research time was commit
`aae5490a5117a4a79ef4fa3ce69b4fcdc6b690ec`, branch `dev`, researcher
`TheMetaSetter`. A representative offline smoke config and online smoke config
were loaded through `.venv/bin/python` and passed the current config loader and
validator. The resolved offline config used SMD, `window_size=20`, `stride=1`,
and `thesis_multitask`; the resolved online config used SMD, `window_size=20`,
`stride=1`, `online_adaptation`, and `max_online_steps=None` after overrides.

No source code was changed for this research note. Existing working-tree
changes, including unrelated documents and scripts and the prior targeted
cleanup in `src/engine/online_tta/online_optimizer.py` and
`src/models/online_impl/online_adaptation.py`, were preserved.

## Code references

- Prompt: `prompts/1_research_prompt.md`
- SSOT design sources: `documents/abstract-design-notes/idea.md`,
  `documents/abstract-design-notes/design_starter.md`,
  `documents/abstract-design-notes/stream_design.md`
- Modernization plan: `documents/abstract-design-notes/codebase-modernization-simple-refactor-plan.md`
- Config loading: `src/core/config.py:903-988`
- Config validation: `src/core/config_model_validation.py:44-700`
- Public data API: `src/data/api.py:14-215`
- Shared loaders and windows: `src/data/loaders.py:150-383`,
  `src/data/window.py:17-83`
- Streaming: `src/data/stream.py:38-267`
- Synthetic anomalies: `src/data/augment.py:21-972`
- Thesis model surface: `src/models/thesis_multitask.py:45-78`
- Thesis forward: `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py:82-330`
- Trainer: `src/engine/trainer.py:482-927`
- Evaluator: `src/engine/evaluator.py:408-554`
- Online facade/runtime: `src/engine/online_tta/online_engine.py:9-25`,
  `src/engine/online_tta/online_engine_run.py:74-367`
- Online adaptation model: `src/models/online_adaptation.py:1-4`,
  `src/models/online_impl/online_adaptation.py`
- Benchmark runners: `scripts/benchmarks/run_thesis_offline_benchmark.py:55-758`,
  `scripts/benchmarks/_internal/run_thesis_offline_benchmark_helpers.py:55-619`
- Config generators: `scripts/benchmarks/generate_offline_benchmark_configs.py:22-137`,
  `scripts/benchmarks/generate_online_benchmark_configs.py:23-247`

