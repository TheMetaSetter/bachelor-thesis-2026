---
date: 2026-09-15T12:00:42+07:00
researcher: OpenAI Codex
topic: "Grand final codebase minimalization"
status: complete
revision: 25b2b545
branch: dev
---

# Research: Grand Final Codebase Minimalization

## Summary

The repository has four active runtime lanes: THESIS offline training, THESIS online test-time adaptation (TTA), offline baselines, and online baselines. The smallest safe redesign keeps those lanes separate and rewrites only their orchestration seams. The shared batch and model-output contracts already provide the thin boundary needed for this redesign.

The current worktree contains unrelated user changes, including deleted configurations and logs. This research does not modify or restore those files.

## Research question

Identify the minimum source surface that must change for the grand final minimalization, compare three minimal redesigns, and select one design that supports simple unit tests, mocks, and integration tests.

## System context

The project is a research codebase for time-series anomaly detection. The active script-facing flow is configuration-driven and uses shared data, model, trainer, evaluator, checkpoint, and artifact infrastructure.

The four active runtime lanes are:

1. THESIS offline training and evaluation.
2. THESIS online TTA.
3. Offline traditional and neural baselines.
4. Online frozen and adaptive baselines.

The project must preserve different scoring, calibration, checkpoint, and adaptation semantics across these lanes.

## Confirmed execution paths

### THESIS offline

`scripts/run_thesis_offline_benchmark.py` delegates to `scripts/benchmarks/run_thesis_offline_benchmark.py`. That runner prepares the configuration and delegates the two-stage lifecycle to `scripts/experiments/run_two_stage_offline_pretraining.py`.

Stage A trains the encoder and task components. Stage B initializes memory once, then trains the remaining stage-specific components. The two-stage ordering is normative and must not be replaced by a generic single-stage loop.

### THESIS online TTA

`scripts/run_thesis_online_benchmark.py` calls the public online TTA engine in `src/engine/online_tta/online_engine.py`. The implementation enters `run_thesis_online_tta_experiment` in `src/engine/online_tta/online_engine_run.py:609-682`.

Each causal window is prepared in `online_engine_window_core.py:43-262`. The path computes scores, updates EWMA state, makes a triage decision, optionally verifies a gray-zone window, optionally updates the projector, and emits an online record. Score extraction and buffer updates live in `online_engine_window_metrics.py:87-262`.

### Offline baselines

`scripts/benchmarks/run_offline_benchmark.py:25-50` selects traditional baseline builders. RedLamp remains a separate neural model path registered through `src/core/runtime_components.py:10-18` and implemented in `src/models/baseline_impl/redlamp_baseline.py`.

### Online baselines

`scripts/benchmarks/run_online_streaming_benchmark.py:27-57` selects CANDI, M2N2, Stumpy, KMeansAD, and Isolation Forest. The online benchmark contract keeps traditional baselines frozen and allows the named adaptive methods to update according to their own protocols.

## Shared contracts

`src/core/contracts.py:90-119` defines offline and online batch shapes. Online batches require strictly increasing absolute indices. `src/core/contracts.py:121-155` defines the model-output vocabulary.

`src/core/runtime_components.py:10-28` centralizes registration. This is an existing seam, not a reason to create another factory or plugin layer.

## Minimal modification surface

The following symbols are the smallest confirmed change surface:

| Priority | File and symbols | Required change | Must remain unchanged |
| --- | --- | --- | --- |
| 1 | `scripts/benchmarks/run_thesis_offline_benchmark.py:76-1257` | Reduce orchestration duplication and make stage execution, calibration, and export linear | Two-stage ordering, checkpoint roles, threshold provenance |
| 2 | `src/engine/online_tta/online_engine_run.py:253-439,609-682` | Make the existing per-window event lifecycle the explicit local seam | Causal order, callback behavior, A0/A1/A2 semantics |
| 3 | `src/engine/online_tta/online_engine_window_core.py:43-262` and `online_engine_window_metrics.py:87-262` | Keep score, triage, verification, and adaptation responsibilities distinct | Raw-input score identity, EWMA rules, verification buffer, frozen source model |
| 4 | `scripts/benchmarks/run_offline_benchmark.py:25-539` and `run_online_streaming_benchmark.py:27-585` | Simplify only outer selection and reporting | Native baseline fit, calibration, scoring, and update protocols |
| 5 | `src/core/runtime_components.py:10-28` | Remove redundant registration paths only after import evidence | Existing registered model names and active configs |

The following files are not initial rewrite targets: `src/core/contracts.py`, `src/engine/trainer.py`, `src/engine/evaluator.py`, `src/models/thesis_multitask.py`, `src/models/baseline_impl/redlamp_baseline.py`, and baseline implementations. They already provide usable boundaries or contain model semantics that are safer to preserve during the first minimalization slice.

## Current score and artifact contract

`documents/spec/full-spec-v4.md:21-25` makes raw-input MSE with `point_score_transform: identity` the current default. `documents/spec/full-spec-v4.md:116-140` requires schema version 5, clean-validation calibration, checkpoint and configuration checksums, and rejection of mismatched artifacts.

Historical sigmoid artifacts remain readable but must not be selected by the current raw evaluator or online runtime. This is a compatibility boundary, not a reason to maintain duplicate current execution paths.

## Existing test evidence

The repository already has focused tests for shared contracts, offline baseline behavior, THESIS online wrapper wiring, online baseline families, generic online entrypoints, and RedLamp configuration resolution:

| Test file | Current evidence |
| --- | --- |
| `tests/compliance/test_src_refactor_contracts.py:27-72` | Registry, output keys, state-dict surface, and config-tree contracts |
| `tests/benchmarks/test_run_offline_benchmark.py:102-148` | Baseline fit, calibration, score export, thresholds, and reports |
| `tests/benchmarks/test_thesis_online_benchmark_wrapper.py:62-184` | THESIS checkpoint wiring and retention behavior |
| `tests/online/test_online_streaming_baseline_contracts.py:69-150` | Five online baseline families |
| `tests/online/test_online_entrypoint.py:7-130` | Generic online adaptation entrypoint |
| `tests/models/test_redlamp_baseline_active_benchmark_config.py:11-25` | RedLamp active benchmark resolution |

The previous minimalization follow-up reports 611 passing tests and 2 skips. It intentionally did not remove CANDI, M2N2, traditional baselines, generic online baselines, notebooks, or historical documents because ownership was unresolved.

## Evidence

- `src/core/contracts.py:90-155` — shared batch and model-output contracts.
- `src/core/runtime_components.py:10-28` — centralized runtime registration.
- `scripts/benchmarks/run_thesis_offline_benchmark.py:76-1257` — THESIS offline orchestration surface.
- `src/engine/online_tta/online_engine_run.py:253-439,609-682` — THESIS online sequence and public experiment lifecycle.
- `src/engine/online_tta/online_engine_window_core.py:43-262` — causal window event processing.
- `src/engine/online_tta/online_engine_window_metrics.py:87-262` — online score and buffer processing.
- `scripts/benchmarks/run_offline_benchmark.py:25-539` — offline baseline selection and execution.
- `scripts/benchmarks/run_online_streaming_benchmark.py:27-585` — online baseline selection and execution.
- `documents/spec/full-spec-v4.md:21-25,116-140` — current raw-input score and artifact identity rules.
- `documents/logs/2026-09-15/implementation/implementation-codebase-minimalization-followup.md:20-34` — previous verification and unresolved retained scope.

## Three minimal redesign alternatives

### Alternative A: Universal method runner

Create one runner interface for THESIS, RedLamp, traditional baselines, CANDI, and M2N2.

This reduces visible script duplication, but it forces incompatible stage, calibration, score, and adaptation lifecycles through one abstraction. It also makes mocks less truthful because a mock must implement fields that many methods do not use.

### Alternative B: Separate runtime owners with explicit existing seams

Keep the four runners separate. Simplify each runner only around an existing boundary:

```text
shared construction
    -> runtime-specific execution
    -> runtime-specific calibration and scoring
    -> common report/provenance export
```

For THESIS online TTA, the existing per-window event dictionary remains the seam. Tests can construct a small event, mock the model and buffer, and verify one decision without starting a stream. Integration tests can run one offline artifact into one online stream.

This design changes the fewest semantic contracts and keeps each method's native behavior visible.

### Alternative C: Artifact-centered runtime package

Make every offline run produce a versioned package containing the checkpoint, scaler, threshold artifact, protocol identity, resolved configuration hash, and provenance. Make every online runner consume that package.

This strengthens reproducibility, but it introduces a new migration and compatibility contract. It is larger than the current minimalization need because v4 already defines the required artifact identity fields.

## MECE decision tree

Use these mutually exclusive questions:

1. Does the design preserve different method lifecycles? If no, reject it.
2. Does the design reuse existing contracts rather than create a new universal interface? If no, reject it for this minimalization.
3. Can a unit test exercise one local rule with a small mock? If no, reject it.
4. Can one integration test trace an offline artifact into its matching online path? If yes, prefer it.

Alternative A fails question 1. Alternative C passes question 1 but fails question 2 because it adds a new artifact migration boundary. Alternative B passes all four questions.

## Selected design

Select **Alternative B: separate runtime owners with explicit existing seams**.

The redesign will simplify orchestration and remove only proven dead paths. It will not merge the THESIS engine with the generic `src/engine/online_loop.py` path, because the generic path is separately reachable and tested. It will not delete active baseline families. It will preserve the current score and artifact contracts.

## Conflicts and uncertainties

- `documents/spec/online-tta-thesis-spec.md:906-964` leaves exact online quantiles and the online EWMA calibration procedure undecided. This redesign will preserve the current implementation and add tests around its observed behavior; it will not settle new statistical policy.
- `codebase_preferences.md:69-75` requires one public model entrypoint without lifecycle mixins, while `src/models/thesis_multitask.py:33-44` still distributes lifecycle behavior through mixins. A later model rewrite may address this, but it is not required for the first orchestration minimalization slice.
- `documents/abstract-design-notes/codebase-modernization-simple-refactor-plan.md` refers to an absent `src/models/redlamp_mlp_baseline.py`. That reference is stale or historical and is not an active deletion target.

## Open questions resolved for this task

The user requested that the design be chosen on their behalf. Therefore this plan keeps all currently reachable runtime families, treats the four public benchmark wrappers as supported, and deletes only code proven unreachable by imports, configurations, tests, and documentation searches.
