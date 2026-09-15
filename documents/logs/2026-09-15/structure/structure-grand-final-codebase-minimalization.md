---
date: 2026-09-15T12:00:42+07:00
topic: "Grand final codebase minimalization"
status: approved
revision: 25b2b545
related_documents:
  - documents/logs/2026-09-15/research/research-grand-final-codebase-minimalization.md
  - documents/logs/2026-09-15/plan/plan-grand-final-codebase-minimalization.md
---

# Implementation Structure: Grand Final Codebase Minimalization

## Summary

The implementation will preserve four runtime lanes and simplify their existing orchestration boundaries. The phase order starts with executable contracts, then changes the THESIS offline path, the THESIS online path, baseline orchestration, and finally cleanup.

## Request

Perform a final minimalization with test-first development. Keep tests minimal, mockable, and strong enough to stress contracts, checkpoint identity, artifact flow, and one end-to-end smoke combination.

## Confirmed context

- `src/core/contracts.py` already defines shared batch and model-output contracts.
- THESIS offline has a normative two-stage lifecycle.
- THESIS online TTA has a causal per-window event path.
- Baselines have separate native protocols.
- v4 raw-input MSE and schema version 5 are current.

## Scope

### In scope

- Shared contract tests.
- THESIS offline orchestration.
- THESIS online event orchestration.
- Offline and online baseline outer orchestration.
- Evidence-based dead-path cleanup.
- Current runtime documentation.

### Out of scope

- Universal method runner.
- New artifact migration schema.
- New statistical calibration policy.
- Generic online loop merger.
- Broad model-mixin rewrite in the first slice.

## Proposed phases

### Phase 1: Executable contracts exist before rewrites

**Result:** Focused tests capture the current shared contracts and the local online event decisions.

**Scope:** Contract tests, mocked event tests, checkpoint and artifact identity assertions.

**Depends on:** Current source and specification evidence.

**Verification:** Focused pytest commands pass before source refactoring begins.

**Risks:** Tests could encode stale documentation.

**Complete when:** Each selected behavior has a test and an evidence citation.

### Phase 2: THESIS offline has one readable two-stage owner

**Result:** Stage A, Stage B, calibration, evaluation, and export remain ordered and have less duplicated orchestration.

**Scope:** THESIS offline benchmark runner and its tests.

**Depends on:** Phase 1 contracts.

**Verification:** One offline smoke run produces valid checkpoints, thresholds, metrics, and provenance.

**Risks:** Stage semantics or artifact roles could change.

**Complete when:** Existing offline tests and artifact identity checks pass.

### Phase 3: THESIS online has one explicit causal event lifecycle

**Result:** Score, triage, verification, adaptation, and export remain separate and directly testable.

**Scope:** Online event, metrics, and sequence-run modules.

**Depends on:** Phase 1 event tests and Phase 2 artifact output.

**Verification:** A0, A1, A2, checkpoint identity, and offline-to-online integration tests pass.

**Risks:** Causality or frozen-state rules could regress.

**Complete when:** One matching offline artifact runs through one online stream.

### Phase 4: Baseline runners share only outer concerns

**Result:** Baselines remain native while report assembly and configuration handling become simpler.

**Scope:** Offline and online baseline benchmark scripts.

**Depends on:** Phase 1 contracts.

**Verification:** Existing baseline contract tests pass for traditional, neural, frozen, and adaptive families.

**Risks:** Report normalization could erase native semantics.

**Complete when:** Baseline-specific score and update behavior remains unchanged.

### Phase 5: Proven dead paths are removed and ownership is documented

**Result:** The repository has no unverified duplicate owner or dead import, and current documentation names each active lane.

**Scope:** Registration calls, aliases, obsolete configs, and current design notes.

**Depends on:** Phases 2–4.

**Verification:** Full pytest, compilation, import audit, and one end-to-end smoke combination pass.

**Risks:** Historical or notebook consumers may depend on an apparently dead path.

**Complete when:** Every deletion has import/config/test/documentation evidence.

## Dependency summary

| Phase | Requires | Enables |
| --- | --- | --- |
| 1 | Current contracts and specifications | Safe source rewrites |
| 2 | Phase 1 tests | Stable offline artifact input |
| 3 | Phases 1–2 | Verified offline-to-online path |
| 4 | Phase 1 tests | Complete lane coverage |
| 5 | Phases 2–4 | Final cleanup and documentation |

## Decisions confirmed

- Use Alternative B: separate runtime owners with explicit existing seams.
- Keep all currently reachable runtime families.
- Preserve v4 raw-input score and schema version 5 artifacts.
- Keep generic online adaptation separate from THESIS online TTA.
- Write tests before each source change.

## Non-blocking uncertainties

- Exact online quantile values remain a future statistical decision. The implementation will preserve and test current behavior.
- The THESIS mixin structure conflicts with the repository preference, but the first minimalization slice does not depend on rewriting it.

## Feedback requested

The user explicitly requested the design choice and all four documents in one procedure. The structure is therefore treated as approved for detailed atomic steps.
