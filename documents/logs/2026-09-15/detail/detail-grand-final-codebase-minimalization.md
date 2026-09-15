---
date: 2026-09-15T12:00:42+07:00
topic: "Grand final codebase minimalization"
status: ready
revision: 25b2b545
source_structure: documents/logs/2026-09-15/structure/structure-grand-final-codebase-minimalization.md
related_documents:
  - documents/logs/2026-09-15/research/research-grand-final-codebase-minimalization.md
  - documents/logs/2026-09-15/plan/plan-grand-final-codebase-minimalization.md
---

# Detailed Implementation: Grand Final Codebase Minimalization

## Summary

Implement Alternative B with five phases. Every behavior change follows red-green-refactor: write the smallest failing test, make the smallest source change, run the focused test, then run the relevant integration test.

## Source structure

The approved structure has five phases: executable contracts, THESIS offline simplification, THESIS online simplification, baseline simplification, and evidence-based cleanup.

## Current state

The active system has four runtime lanes. Shared contracts exist. THESIS offline and online paths have method-specific semantics. Baselines have separate native protocols. The current worktree also contains unrelated user changes; implementation must not revert them.

## Desired end state

Each lane has one readable orchestration path, existing artifact and score contracts remain valid, and minimal tests can isolate local rules or run one complete offline-to-online smoke path.

## Scope

### In scope

- Existing benchmark orchestration files named below.
- Focused tests in existing test modules.
- Only small proposed tests when no existing test location is suitable.
- Current ownership documentation.

### Out of scope

- Universal runner abstraction.
- New artifact schema.
- New calibration policy.
- Generic online loop merger.
- Removal of active baseline families.

## Evidence

- `src/core/contracts.py:90-155` — shared input and output shapes.
- `scripts/benchmarks/run_thesis_offline_benchmark.py:76-1257` — offline owner.
- `src/engine/online_tta/online_engine_window_core.py:43-262` — online event lifecycle.
- `src/engine/online_tta/online_engine_window_metrics.py:87-262` — online score and buffer updates.
- `documents/spec/full-spec-v4.md:21-25,116-140` — raw score and artifact identity.
- `tests/benchmarks/test_thesis_online_benchmark_wrapper.py:62-184` — online integration evidence.

## Phase 1: Executable contracts exist before rewrites

### Goal

Create minimal tests that fail if the shared contract or selected event behavior changes.

### Dependencies

Current source and specification evidence only.

### Detailed changes

#### 1. Protect shared contracts

- **File:** `tests/compliance/test_src_refactor_contracts.py`
- **Symbol:** existing registry, model-output, and config-tree tests
- **Current responsibility:** Verify selected source-level contracts.
- **Change:** Add only missing assertions for required batch keys, online index ordering, model output keys, and registered active model names.
- **Reason:** These assertions protect the smallest shared boundary.
- **Inputs:** Minimal dictionaries and tensors already used by the test module.
- **Outputs:** Assertion failures identify contract drift.
- **Errors:** Invalid shape or missing key must fail directly.
- **Dependencies:** `src/core/contracts.py`, `src/core/runtime_components.py`.
- **Compatibility:** Do not rename existing contract keys.

#### 2. Test score and artifact identity

- **File:** `tests/benchmarks/test_full_spec_runtime_readiness.py`
- **Symbol:** existing runtime-readiness fixture and assertions
- **Current responsibility:** Provide artifacts and metadata for the current export contract.
- **Change:** Assert schema version 5, raw-input score identity, clean-validation calibration, checkpoint checksum, and configuration checksum.
- **Reason:** The online lane must consume an offline artifact without guessing its identity.
- **Inputs:** Existing fixture artifact payload.
- **Outputs:** A validated artifact payload.
- **Errors:** Mismatched score identity or missing provenance fails.
- **Dependencies:** v4 specification and offline export code.
- **Compatibility:** Historical artifacts remain readable but cannot satisfy the current raw protocol.

#### 3. Test local online decisions with mocks

- **File:** `tests/online/test_online_entrypoint.py`
- **Symbol:** existing entrypoint tests; add focused event cases
- **Current responsibility:** Test the generic online entrypoint and selected online behavior.
- **Change:** Add cases for normal, gray-zone, hard-old-normality, A0, A1, and A2 using mocked model outputs, thresholds, and buffers.
- **Reason:** The event lifecycle must be testable without loading a dataset or starting a full stream.
- **Inputs:** Small event dictionaries matching the existing event fields.
- **Outputs:** Triage decision, optional update action, and event record.
- **Errors:** Missing required event fields fail at the boundary rather than producing a partial record.
- **Dependencies:** `online_engine_window_core.py`, `online_engine_window_metrics.py`.
- **Compatibility:** Preserve existing event callback record fields.

### Tests

#### Contract and event behavior

- **Location:** Existing test files above.
- **Level:** Unit.
- **Setup:** Small tensors, event dictionaries, mocked model and buffer.
- **Action:** Call one local contract or event function.
- **Expected result:** Required fields, score identity, triage, and update rules match the current specification.
- **Edge cases:** Empty optional labels, overlapping windows, hard-old-normality guard, frozen source parameters.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest tests/compliance/test_src_refactor_contracts.py tests/benchmarks/test_full_spec_runtime_readiness.py tests/online/test_online_entrypoint.py` — focused tests pass.

#### Manual

- [ ] Read each new assertion and verify its source evidence is cited in the test comment or test name.

### Risks and recovery

- **Risk:** A test encodes an old score field.
- **Mitigation:** Assert v4 field names and distinguish historical fields.
- **Verification:** Runtime-readiness test rejects historical identity for current artifacts.
- **Recovery:** Remove only the incorrect assertion and re-check the v4 evidence.

### Complete when

- Tests exist before source rewrites.
- Focused tests pass against the current implementation.

## Phase 2: THESIS offline has one readable two-stage owner

### Goal

Simplify offline orchestration without changing Stage A, Stage B, calibration, evaluation, or export semantics.

### Dependencies

Phase 1 tests pass.

### Detailed changes

#### 1. Make stage order explicit

- **File:** `scripts/benchmarks/run_thesis_offline_benchmark.py`
- **Symbol:** current stage orchestration around the calls at lines 76-107
- **Current responsibility:** Prepare and run the THESIS offline benchmark.
- **Change:** Keep one linear sequence: resolve configuration, run Stage A, initialize Stage B memory once, run Stage B, calibrate on clean validation, evaluate, export.
- **Reason:** The normative two-stage lifecycle should be visible without following duplicate branches.
- **Inputs:** Existing resolved configuration and dataset loaders.
- **Outputs:** Stage checkpoints, threshold artifact, metrics, provenance.
- **Errors:** Propagate stage and artifact failures with their stage name.
- **Dependencies:** `run_two_stage_offline_pretraining.py`, trainer, evaluator.
- **Compatibility:** Preserve checkpoint roles and existing output paths.

#### 2. Keep one artifact assembly point

- **File:** `scripts/benchmarks/run_thesis_offline_benchmark.py`
- **Symbol:** current threshold and export helpers around the existing export functions
- **Current responsibility:** Build and write offline artifacts.
- **Change:** Remove duplicate payload assembly and retain one path for score identity, threshold source, checkpoint checksum, config checksum, and selected diagnostics.
- **Reason:** The v4 artifact is the online input contract.
- **Inputs:** Raw scores, thresholds, selected checkpoint, resolved configuration.
- **Outputs:** Minimal report-ready stage artifacts.
- **Errors:** Reject incomplete identity instead of writing a partial current artifact.
- **Dependencies:** v4 contract and online wrapper.
- **Compatibility:** Do not overwrite historical sigmoid artifacts.

### Tests

#### Stage-order unit test

- **Location:** `tests/benchmarks/test_thesis_offline_benchmark.py` if present; otherwise add a focused test to the nearest existing THESIS benchmark test module.
- **Level:** Unit with mocks.
- **Setup:** Mock Stage A, Stage B, calibration, evaluation, and export calls.
- **Action:** Invoke the offline runner.
- **Expected result:** Calls occur exactly once in the required order.
- **Edge cases:** Stage A failure stops Stage B; missing Stage-B checkpoint stops export.

#### Offline artifact integration test

- **Location:** Existing THESIS benchmark test module.
- **Level:** Integration.
- **Setup:** One small SMD-like fixture and temporary output directory.
- **Action:** Run the offline export path.
- **Expected result:** Checkpoint roles, v4 identity fields, and provenance exist.
- **Edge cases:** Mismatched entity or variant is rejected.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest tests/benchmarks/test_thesis_offline_benchmark.py tests/benchmarks/test_full_spec_runtime_readiness.py` — stage and artifact tests pass.

#### Manual

- [ ] Inspect one smoke output directory and confirm only initialization, best, provenance, summary, and selected diagnostics remain.

### Risks and recovery

- **Risk:** Refactoring changes the selected best checkpoint.
- **Mitigation:** Assert checkpoint role and selection metadata before rewriting.
- **Verification:** Offline integration test compares manifest fields.
- **Recovery:** Restore the previous orchestration branch while retaining the new tests.

### Complete when

- One offline runner path produces the same required artifact contract.
- Stage order tests pass.

## Phase 3: THESIS online has one explicit causal event lifecycle

### Goal

Simplify online control flow while preserving causal scoring, EWMA, triage, verification, and projector-only adaptation.

### Dependencies

Phase 1 event tests and Phase 2 offline artifact output.

### Detailed changes

#### 1. Preserve score production as a separate step

- **File:** `src/engine/online_tta/online_engine_window_metrics.py`
- **Symbol:** `_score_online_window`, `_extract_online_window_scores`, `_update_online_window_buffers`
- **Current responsibility:** Compute scores and update online score state.
- **Change:** Remove duplicated conversions while keeping raw-input operational scores, normalized diagnostics, current-window scores, and EWMA overlap behavior.
- **Reason:** The scorer must not decide whether a model adapts.
- **Inputs:** Window batch, model output, scaler, threshold state.
- **Outputs:** Score fields and updated buffer state.
- **Errors:** Missing scaler for raw-input reconstruction remains an explicit error.
- **Dependencies:** v4 score contract and online event preparation.
- **Compatibility:** Keep current event field names until all callers are migrated in the same change.

#### 2. Preserve the decision lifecycle

- **File:** `src/engine/online_tta/online_engine_window_core.py`
- **Symbol:** `_prepare_online_window_event`, `_run_current_window_action`, `_admit_and_verify_gray_zone`, `_process_online_window`
- **Current responsibility:** Prepare one event, choose an action, verify gray-zone data, and build outputs.
- **Change:** Keep one explicit order: score, classify triage, admit or reject, verify if required, update only permitted parameters, emit record.
- **Reason:** This is the smallest direct unit-test boundary.
- **Inputs:** Event dictionary, threshold values, verification buffer, adaptation model.
- **Outputs:** Event record and optional adaptation metric.
- **Errors:** Invalid event fields fail before an update or record is emitted.
- **Dependencies:** online metrics, optimizer, verification buffer.
- **Compatibility:** Preserve A0 no-update, A1 verified PNN update, and A2 hard-old-normality guard semantics.

#### 3. Simplify sequence lifecycle

- **File:** `src/engine/online_tta/online_engine_run.py`
- **Symbol:** `_run_online_sequence`, `_build_online_execution_context`, `_run_online_execution_sequences`, `_finalize_online_execution`
- **Current responsibility:** Build context, process sequences, invoke callbacks, and export final state.
- **Change:** Remove duplicate setup/finalization paths and retain one context, one sequence loop, and one final export.
- **Reason:** Cross-window state belongs here, not inside local scoring functions.
- **Inputs:** Offline artifact, online config, stream sequences, optional callback.
- **Outputs:** Online records, metrics, retention bundle, runtime state.
- **Errors:** Artifact mismatch stops execution before consuming the stream.
- **Dependencies:** online model registration and event core.
- **Compatibility:** Preserve callback records and retention policy.

### Tests

#### Event unit tests

- **Location:** `tests/online/test_online_entrypoint.py`.
- **Level:** Unit.
- **Setup:** Mock model outputs, thresholds, buffer, and optimizer.
- **Action:** Process one event for A0, A1, and A2.
- **Expected result:** Correct triage, update target, and record fields.
- **Edge cases:** Overlap rejection, missing scaler, frozen source model, and empty verification buffer.

#### Offline-to-online integration test

- **Location:** `tests/benchmarks/test_thesis_online_benchmark_wrapper.py`.
- **Level:** Integration.
- **Setup:** Temporary v4 offline artifact and one short online stream fixture.
- **Action:** Invoke the public THESIS online wrapper.
- **Expected result:** The wrapper selects the matching checkpoint and threshold artifact, processes causal windows, and writes the expected retention bundle.
- **Edge cases:** Entity, variant, checkpoint, or score-identity mismatch is rejected before stream processing.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest tests/online/test_online_entrypoint.py tests/benchmarks/test_thesis_online_benchmark_wrapper.py` — event and integration tests pass.

#### Manual

- [ ] Inspect one online record and confirm operational score, diagnostic score, ground-truth label, prediction, and triage decision are separate fields.

### Risks and recovery

- **Risk:** A refactor updates frozen source parameters.
- **Mitigation:** Assert source parameter identity before and after A1/A2.
- **Verification:** Mocked update tests inspect optimizer parameter ownership.
- **Recovery:** Disable the changed action branch and retain the old local event path until the test is corrected.

### Complete when

- A0, A1, A2 unit cases pass.
- One offline artifact runs through one causal online integration test.

## Phase 4: Baseline runners share only outer concerns

### Goal

Simplify configuration and final report assembly while keeping each baseline's native lifecycle.

### Dependencies

Phase 1 contract tests.

### Detailed changes

#### 1. Simplify offline baseline shell

- **File:** `scripts/benchmarks/run_offline_benchmark.py`
- **Symbol:** `BASELINE_BUILDERS`, `_instantiate_baseline`, `run_offline_benchmark`
- **Current responsibility:** Select a baseline, run it, calibrate it, and export results.
- **Change:** Keep builder selection, but make common setup and final report construction linear. Do not move native fit or score logic into a common interface.
- **Reason:** Traditional and neural baselines have different protocols.
- **Inputs:** Existing configuration and dataset batches.
- **Outputs:** Existing metrics and report fields plus preserved native diagnostics.
- **Errors:** Unknown method names fail at selection; native method errors retain their method context.
- **Dependencies:** traditional baseline protocol, RedLamp model path.
- **Compatibility:** Keep active method names and config resolution.

#### 2. Simplify online baseline shell

- **File:** `scripts/benchmarks/run_online_streaming_benchmark.py`
- **Symbol:** `BASELINE_BUILDERS`, `_instantiate_baseline`, `_compute_final_metrics`, `run_online_streaming_benchmark`
- **Current responsibility:** Select and execute online baseline methods and compute final metrics.
- **Change:** Share only stream setup, provenance, and final report assembly. Keep frozen traditional and adaptive CANDI/M2N2 loops separate.
- **Reason:** The benchmark contract explicitly distinguishes update protocols.
- **Inputs:** Stream configuration and method-specific configuration.
- **Outputs:** Method-native records and common final metrics.
- **Errors:** Invalid method configuration fails before stream consumption where possible.
- **Dependencies:** `src/baselines/online/`, online baseline contract.
- **Compatibility:** Preserve one configuration per entity/seed for adaptive methods.

### Tests

#### Baseline unit tests

- **Location:** `tests/benchmarks/test_run_offline_benchmark.py` and `tests/online/test_online_streaming_baseline_contracts.py`.
- **Level:** Unit.
- **Setup:** Mock builders and minimal synthetic batches/streams.
- **Action:** Select one traditional, one neural, one frozen, and one adaptive baseline.
- **Expected result:** Each method keeps its native fit, score, and update behavior while final reports use common metadata.
- **Edge cases:** Unknown method, missing calibration artifact, and adaptive method with invalid entity configuration.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest tests/benchmarks/test_run_offline_benchmark.py tests/online/test_online_streaming_baseline_contracts.py tests/models/test_redlamp_baseline_active_benchmark_config.py` — baseline tests pass.

#### Manual

- [ ] Confirm a traditional online baseline never updates model state and an adaptive baseline still performs its documented update.

### Risks and recovery

- **Risk:** Common reporting drops method-specific scores.
- **Mitigation:** Preserve native record fields and normalize only report metadata.
- **Verification:** Existing baseline contract tests compare native fields.
- **Recovery:** Revert only report assembly changes and retain method-specific runners.

### Complete when

- All active baseline families pass focused tests.
- No baseline-specific lifecycle is routed through THESIS code.

## Phase 5: Proven dead paths are removed and ownership is documented

### Goal

Delete only verified dead paths and make the final runtime ownership map explicit.

### Dependencies

Phases 2–4 pass their focused and integration tests.

### Detailed changes

#### 1. Audit registration and aliases

- **File:** `src/core/runtime_components.py` and all callers found by project-wide search
- **Symbol:** registration functions and re-export wrappers
- **Current responsibility:** Register shared, offline, evaluation, and online components.
- **Change:** Remove duplicate calls or aliases only when no active import, config, test, notebook, or current document references them.
- **Reason:** Keep one obvious construction path without breaking supported users.
- **Inputs:** Import and reference search results.
- **Outputs:** Smaller registration surface.
- **Errors:** Keep a compatibility wrapper when evidence is incomplete.
- **Dependencies:** all runtime scripts and configs.
- **Compatibility:** Preserve the four public benchmark wrappers.

#### 2. Audit obsolete configuration and source paths

- **File:** Exact candidates identified during implementation; no path is predetermined.
- **Symbol:** Unreachable wrapper, alias, obsolete config, or stale import.
- **Current responsibility:** Historical or duplicate behavior if still referenced.
- **Change:** Delete only after reference searches and focused tests prove the candidate is unreachable.
- **Reason:** Avoid repeating the previous cleanup's unresolved-scope problem.
- **Inputs:** Project-wide references and test results.
- **Outputs:** Smaller repository with no broken imports.
- **Errors:** A reference blocks deletion and becomes an explicit compatibility item.
- **Dependencies:** all earlier phases.
- **Compatibility:** Preserve historical documents and readable artifacts unless they are proven outside support scope.

#### 3. Update current design notes

- **File:** Relevant current note under `documents/abstract-design-notes/`
- **Symbol:** runtime ownership and minimalization sections
- **Current responsibility:** Describe intended architecture and runtime boundaries.
- **Change:** State the four active lanes, the selected separate-owner design, the v4 artifact boundary, and the non-merging rule for generic online adaptation.
- **Reason:** `documents/` is the project source of truth.
- **Inputs:** Completed implementation evidence.
- **Outputs:** Accurate current documentation.
- **Errors:** Do not describe planned behavior as implemented.
- **Dependencies:** completed source changes and tests.
- **Compatibility:** Preserve historical notes as historical.

### Tests

#### Full regression and smoke test

- **Location:** Existing repository test suite plus one configured SMD smoke combination.
- **Level:** Integration and end-to-end smoke.
- **Setup:** Existing `.venv`, one concrete development combination, temporary output location.
- **Action:** Run the full suite, compile changed Python files, then run offline and matching online smoke paths.
- **Expected result:** All tests pass, imports resolve, artifacts remain minimal, and online consumes the matching offline artifacts.
- **Edge cases:** Existing warning-only fixtures, historical artifact reads, and missing optional diagnostics.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest` — full suite passes.
- [ ] `.venv/bin/python -m compileall src scripts tests` — changed Python files compile.

#### Manual

- [ ] Inspect `git diff` and confirm only intended source, test, and current documentation files changed.
- [ ] Inspect the smoke artifact tree and confirm no per-forward-pass tensors were added.

### Risks and recovery

- **Risk:** A notebook or historical workflow depends on a deletion candidate.
- **Mitigation:** Search notebooks and documents before deletion; preserve a wrapper when uncertain.
- **Verification:** Import collection and full pytest pass.
- **Recovery:** Restore the exact deleted path without reverting unrelated user changes.

### Complete when

- Full tests and one end-to-end smoke combination pass.
- Every deletion has evidence.
- Current documentation matches executable behavior.

## Interface and data changes

No new universal interface or stored-data schema is planned. Preserve the batch contract, model-output contract, v4 score identity, schema version 5 artifact fields, checkpoint roles, and current baseline method names.

## Deployment and rollout

Implement one phase at a time. Run focused tests after every test-first change. Run the full suite only after each runtime lane has passed its focused integration test. Do not launch a broad benchmark batch before the single development smoke combination passes.

## Documentation changes

- Update current runtime ownership notes.
- Record terminology mappings for any renamed runtime field.
- Leave historical specifications unchanged.
- Record final verification and artifact paths in the implementation log.

## Final verification

- [ ] Tests were written before each corresponding source rewrite.
- [ ] Shared contracts, forward/backward behavior, checkpoint round trip, loader shapes, and anomaly injection remain tested.
- [ ] THESIS offline remains two-stage.
- [ ] THESIS online remains causal and artifact-compatible.
- [ ] Baselines retain native protocols.
- [ ] Full pytest passes.
- [ ] One offline-to-online smoke combination passes.
- [ ] Dead-path deletions have evidence.

## Assumptions and non-blocking uncertainties

- The current four benchmark wrappers remain supported.
- Current online calibration behavior remains unchanged until a separate statistical decision is approved.
- A future model-focused slice may replace lifecycle mixins, but this orchestration slice must not expand to include it without new evidence.
