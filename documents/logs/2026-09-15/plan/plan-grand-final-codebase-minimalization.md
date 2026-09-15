---
date: 2026-09-15T12:00:42+07:00
planner: OpenAI Codex
topic: "Grand final codebase minimalization"
status: ready
revision: 25b2b545
branch: dev
related_research: documents/logs/2026-09-15/research/research-grand-final-codebase-minimalization.md
---

# Implementation Plan: Grand Final Codebase Minimalization

## Summary

Rewrite only the four runtime orchestration boundaries while preserving the existing batch, model-output, score, checkpoint, calibration, and baseline contracts. Use Alternative B: separate runtime owners with explicit existing seams.

The implementation is test-first. Each behavior change starts with a focused unit or integration test, then the smallest source change makes that test pass.

## Request

Perform the final minimalization of the codebase. Research the current code first, compare three minimal designs, choose one on the user's behalf, and produce high-level phases, sequential stages, and indivisible atomic steps. Keep the test suite minimal while stress-testing contracts, one forward/backward path, checkpoint round trips, data shapes, synthetic anomaly injection, offline-to-online artifact identity, and the four runtime lanes.

## Current state

The repository has four active runtime lanes. THESIS offline uses a normative two-stage lifecycle. THESIS online TTA uses a causal per-window event path. Offline and online baselines retain method-specific protocols. Shared contracts and registration already exist.

## Desired end state

- Each runtime lane has one readable orchestration path.
- The THESIS offline path remains two-stage.
- The THESIS online path remains causal and method-specific.
- Baselines retain native calibration and adaptation behavior.
- Shared infrastructure is reused without imposing a universal method interface.
- Tests prove local rules with mocks and prove one complete artifact flow with integration tests.
- Unreachable code is deleted only after evidence-based import and configuration searches.

## Scope

### In scope

- `scripts/benchmarks/run_thesis_offline_benchmark.py`
- `src/engine/online_tta/online_engine_run.py`
- `src/engine/online_tta/online_engine_window_core.py`
- `src/engine/online_tta/online_engine_window_metrics.py`
- `scripts/benchmarks/run_offline_benchmark.py`
- `scripts/benchmarks/run_online_streaming_benchmark.py`
- `src/core/runtime_components.py`
- Focused tests and runtime documentation.

### Out of scope

- Changing raw-input MSE semantics.
- Changing Stage A or Stage B semantics.
- Merging generic online adaptation with THESIS online TTA.
- Removing active baseline families.
- Replacing all model mixins in the first slice.
- Broad dataset generalization.

## Evidence

- `src/core/contracts.py:90-155` — shared batch and model-output contracts.
- `src/core/runtime_components.py:10-28` — shared registration boundary.
- `scripts/benchmarks/run_thesis_offline_benchmark.py:76-1257` — THESIS offline orchestration.
- `src/engine/online_tta/online_engine_run.py:253-439,609-682` — THESIS online orchestration.
- `documents/spec/full-spec-v4.md:21-25,116-140` — current score and artifact contract.
- `documents/logs/2026-09-15/implementation/implementation-codebase-minimalization-followup.md:20-34` — previous test result and retained scope.

## Implementation approach

Use a MECE decision tree. First preserve runtime-specific semantics. Then prefer existing seams over new universal abstractions. Then require direct unit-testability with small mocks. Finally require one integration path from an offline artifact to online execution.

This selects separate runtime owners with explicit event and artifact boundaries. It avoids a new registry, factory, adapter hierarchy, or package schema because the repository already has sufficient contracts.

## Phase 1: Freeze the testable boundary

### Goal

Turn the current contracts and observed runtime behavior into minimal executable tests before source rewrites begin.

### Changes

#### 1. Add focused contract tests

- **File:** `tests/compliance/test_src_refactor_contracts.py`
- **Symbol:** existing contract tests
- **Change:** Add only missing assertions for batch shapes, required output keys, and registration ownership.
- **Reason:** Protect the shared thin waist before changing callers.
- **Dependencies:** `src/core/contracts.py`, `src/core/runtime_components.py`

#### 2. Add local online event tests

- **File:** `tests/online/test_online_entrypoint.py`
- **Symbol:** existing online entrypoint tests
- **Change:** Add mocked cases for normal, gray-zone, and hard-old-normality event decisions.
- **Reason:** Make the selected online seam independently testable.
- **Dependencies:** `online_engine_window_core.py`, `online_engine_window_metrics.py`

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest tests/compliance/test_src_refactor_contracts.py tests/online/test_online_entrypoint.py` — existing and new focused tests pass.

#### Manual

- [ ] Review the event fields and confirm each field has one producer and one consumer.

### Risks

- **Risk:** A test may encode an undocumented behavior.
- **Mitigation:** Cite the active specification or executable implementation beside each assertion.

## Phase 2: Simplify THESIS offline orchestration

### Goal

Make the two-stage offline execution, calibration, evaluation, and export readable as one linear owner without changing stage semantics.

### Changes

#### 1. Simplify stage orchestration

- **File:** `scripts/benchmarks/run_thesis_offline_benchmark.py`
- **Symbol:** stage orchestration functions around the current Stage A/Stage B calls
- **Change:** Remove duplicated preparation and reporting branches; retain explicit Stage A, Stage B initialization, Stage B training, evaluation, and export order.
- **Reason:** The current runner is the active owner and is larger than the repository's 500-line preference.
- **Dependencies:** `scripts/experiments/run_two_stage_offline_pretraining.py`, `src/engine/trainer.py`, checkpoint writers.

#### 2. Preserve artifact identity

- **File:** `scripts/benchmarks/run_thesis_offline_benchmark.py`
- **Symbol:** threshold and artifact export helpers
- **Change:** Keep schema version 5, raw-input score identity, clean-validation source, checkpoint checksum, and resolved-config checksum in one export path.
- **Reason:** Online TTA consumes these artifacts.
- **Dependencies:** `full-spec-v4.md`, online wrapper tests.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest tests/benchmarks/test_thesis_offline_benchmark.py tests/benchmarks/test_full_spec_runtime_readiness.py` — offline orchestration and artifact contract tests pass.

#### Manual

- [ ] Inspect one smoke artifact tree and confirm the canonical output hierarchy and retained checkpoints.

### Risks

- **Risk:** Stage order or checkpoint role changes during extraction.
- **Mitigation:** Add assertions for Stage A, Stage B initialization, Stage B best checkpoint, and threshold provenance before rewriting.

## Phase 3: Simplify THESIS online TTA

### Goal

Make the existing causal window event the only local decision seam while preserving scoring, EWMA, triage, verification, and projector adaptation semantics.

### Changes

#### 1. Keep event preparation explicit

- **File:** `src/engine/online_tta/online_engine_window_core.py`
- **Symbol:** `_prepare_online_window_event`, `_run_current_window_action`, `_admit_and_verify_gray_zone`, `_process_online_window`
- **Change:** Keep each responsibility linear and remove only duplicated field assembly or branching.
- **Reason:** These functions already form the smallest testable online seam.
- **Dependencies:** online metrics, verification buffer, projector optimizer.

#### 2. Keep score production separate

- **File:** `src/engine/online_tta/online_engine_window_metrics.py`
- **Symbol:** `_score_online_window`, `_extract_online_window_scores`, `_update_online_window_buffers`
- **Change:** Preserve raw-input operational scores, normalized diagnostics, EWMA rules, and buffer state while reducing duplicate conversions.
- **Reason:** Score identity must remain independent from action selection.
- **Dependencies:** v4 score contract.

#### 3. Simplify sequence orchestration

- **File:** `src/engine/online_tta/online_engine_run.py`
- **Symbol:** `_run_online_sequence`, `_build_online_execution_context`, `_run_online_execution_sequences`, `_finalize_online_execution`
- **Change:** Keep one sequence lifecycle and one finalization path; preserve callback output and runtime-state export.
- **Reason:** The current file owns cross-window state and is the correct integration boundary.
- **Dependencies:** THESIS online benchmark wrapper.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest tests/benchmarks/test_thesis_online_benchmark_wrapper.py tests/online/test_online_entrypoint.py` — A0, A1, A2 wiring, checkpoint identity, and mocked event tests pass.

#### Manual

- [ ] Review one online record and confirm labels, predictions, score identity, triage decision, and artifact provenance remain distinct.

### Risks

- **Risk:** A refactor accidentally adapts frozen source state.
- **Mitigation:** Unit-test parameter identity and projector-only updates for A1/A2.

## Phase 4: Simplify baseline orchestration

### Goal

Reduce duplicated outer runner logic without forcing traditional, neural, frozen, and adaptive baselines into one lifecycle.

### Changes

#### 1. Preserve offline baseline protocols

- **File:** `scripts/benchmarks/run_offline_benchmark.py`
- **Symbol:** `BASELINE_BUILDERS`, `_instantiate_baseline`, `run_offline_benchmark`
- **Change:** Keep native fit and score behavior; simplify only configuration selection, common report assembly, and error boundaries.
- **Reason:** The builder registry is already the active selection seam.
- **Dependencies:** traditional baseline protocols and RedLamp tests.

#### 2. Preserve online baseline protocols

- **File:** `scripts/benchmarks/run_online_streaming_benchmark.py`
- **Symbol:** `BASELINE_BUILDERS`, `_instantiate_baseline`, `_compute_final_metrics`, `run_online_streaming_benchmark`
- **Change:** Keep frozen traditional baselines and adaptive CANDI/M2N2 behavior separate; share only final report fields.
- **Reason:** The online benchmark contract explicitly distinguishes these methods.
- **Dependencies:** `tests/online/test_online_streaming_baseline_contracts.py`.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest tests/benchmarks/test_run_offline_benchmark.py tests/online/test_online_streaming_baseline_contracts.py tests/models/test_redlamp_baseline_active_benchmark_config.py` — baseline-specific contracts pass.

#### Manual

- [ ] Confirm one traditional baseline remains frozen and one adaptive baseline still updates only according to its native protocol.

### Risks

- **Risk:** Common report normalization erases a method-specific score.
- **Mitigation:** Preserve native fields and normalize only final report metadata.

## Phase 5: Remove proven dead paths and document ownership

### Goal

Delete only unreachable code and make each active runtime owner explicit.

### Changes

#### 1. Consolidate registration calls

- **File:** `src/core/runtime_components.py`
- **Symbol:** registration functions
- **Change:** Remove redundant calls only after import, configuration, and test searches prove they are not supported surfaces.
- **Reason:** Keep one obvious registration boundary.
- **Dependencies:** all active scripts and config trees.

#### 2. Audit deletion candidates

- **File:** no source deletion is predetermined
- **Symbol:** wrappers, aliases, obsolete configs, and stale imports identified during the audit
- **Change:** Delete only candidates with no active import, config, test, notebook, or documented support reference.
- **Reason:** The previous cleanup correctly retained ambiguous families.
- **Dependencies:** all preceding phases.

#### 3. Update ownership documentation

- **File:** `documents/abstract-design-notes/` and the relevant current runtime notes
- **Symbol:** runtime ownership sections
- **Change:** Document the four lanes, their public entrypoints, artifact boundaries, and non-merging rule for generic online adaptation.
- **Reason:** The documentation is the repository's single source of truth.
- **Dependencies:** completed runtime changes.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest` — full suite passes with no new failures.
- [ ] `.venv/bin/python -m compileall src scripts tests` — changed Python files compile.

#### Manual

- [ ] Run one complete THESIS offline-to-online smoke combination and inspect the retained artifacts.

### Risks

- **Risk:** A historical path is deleted while a notebook or external workflow still uses it.
- **Mitigation:** Search imports, configs, tests, notebooks, and documentation before every deletion; retain a narrow compatibility wrapper when evidence is incomplete.

## Testing strategy

The minimal test suite has four layers:

1. Unit tests for score transforms, EWMA selection, triage decisions, artifact identity, and baseline report normalization.
2. Mocked unit tests for the online event lifecycle and projector update rules.
3. Focused integration tests for offline artifact export and online artifact consumption.
4. One end-to-end smoke combination after all focused tests pass.

Tests must cover one forward pass, one backward pass, checkpoint save/load, loader shapes, synthetic anomaly injection, and configuration initialization through existing focused tests or minimal additions to those test files.

## Migration and rollback

No stored-data migration is planned. Keep schema version 5 artifacts and historical sigmoid artifacts readable. Implement each phase in small commits or reversible worktree steps. If a phase changes benchmark outputs, revert only that phase and compare its artifact manifest against the pre-change smoke result.

## Documentation

Update only the current runtime ownership and minimalization notes. Do not rewrite historical specifications. Add terminology mappings whenever a runtime object is renamed.

## Final verification

- [ ] Focused contract, THESIS, baseline, checkpoint, shape, and injection tests pass.
- [ ] Full `pytest` passes.
- [ ] One SMD THESIS offline smoke run completes.
- [ ] Its Stage-B checkpoint and threshold artifact are consumed by one matching online run.
- [ ] No active path imports a deleted symbol.
- [ ] Output artifacts retain provenance and minimal disk usage.

## Assumptions and non-blocking uncertainties

- The four current public benchmark wrappers remain supported because the user requested final minimalization without narrowing runtime scope.
- Exact online quantile policy remains unchanged and is tested as current behavior; deciding new statistical policy is outside this redesign.
- The THESIS model mixin rewrite remains a later slice unless tests show that orchestration changes cannot be completed without it.
