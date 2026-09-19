---
date: 2026-09-15
planner: OpenAI Codex
topic: "Reduce the tsad-lib pytest suite to 28 collected contract tests"
status: proposed
related_research: documents/logs/2026-09-15/research/research-tsad-test-suite-reduction.md
---

# Plan: A 28-Case TSAD Test Suite

## Summary

The implementation will replace 27 scattered test files with five contract-test files. The final `pytest --collect-only` result must be at most 30. The target is 28.

The implementation will use one test per risk boundary. It will loop through repeated members only when they use the same contract. The suite will keep all 15 ready reference models, all dataset adapter families, the full O2 offline story, A0/A1/A2 online rules, the public notebook and CLI paths, and the SMD release gate.

## Request

Keep the `tsad-lib` test suite at at most 30 cases. Keep the design and implementation minimal.

## Current state

The current suite collects 119 nodes. Its repetition comes mostly from parameterization and one-example-per-member tests. The code already has a single pytest command through `tsad-lib/.venv/bin/python`.

## Desired end state

`PYTHONPATH=src .venv/bin/python -m pytest --collect-only -q -p no:cacheprovider tests` collects exactly 28 nodes. The full suite passes. Every retained test has one named risk boundary.

## Scope

### In scope

- Fold repeated pytest examples into 28 named contract tests.
- Keep a loop over every ready reference model inside one contract test.
- Keep a loop over every dataset adapter family inside contract tests.
- Remove old tests after their replacement contract passes.
- Update the development specification checklist with fresh count evidence.

### Out of scope

- Change `src/tsad` runtime behavior.
- Add a second test runner, marker tier, or CI service.
- Test the 25 reference models that are not TSAD-ready.
- Reduce the runtime work done by the all-ready-reference-model contract.

## Chosen approach

Use boundary-contract folding. This is Strategy 1 in the research note. It is the smallest approach that preserves current important evidence.

## Phase 1: establish the 28-case map

### Goal

The coding agent knows exactly which old test behavior belongs in each new contract test.

### Stage 1.1: freeze the count rule

1. Run pytest collection before changing tests.
2. Record the current 119-node count.
3. Define the acceptance rule as at most 30 collected nodes.
4. Set the planned count to 28 collected nodes.

### Stage 1.2: define the five boundaries

1. Define five proposed test files: data, baseline, THESIS, public, and release.
2. Assign five tests to the data file.
3. Assign four tests to the baseline file.
4. Assign eight tests to the THESIS file.
5. Assign six tests to the public file.
6. Assign five tests to the release file.

**Complete when:** the plan names 28 tests and every current risk boundary has one owner.

## Phase 2: replace repeated examples with contract tests

### Goal

The suite keeps coverage while reducing pytest nodes.

### Stage 2.1: fold data and baseline families

1. Write the five data contracts before deleting data tests.
2. Run the five data contracts.
3. Write the four baseline contracts before deleting baseline tests.
4. Run the four baseline contracts.
5. Delete only old tests whose behavior is now asserted by a passing contract.

### Stage 2.2: fold THESIS and public paths

1. Write the eight THESIS contracts before deleting THESIS tests.
2. Run the eight THESIS contracts.
3. Write the six public contracts before deleting public tests.
4. Run the six public contracts.
5. Delete only old tests whose behavior is now asserted by a passing contract.

### Stage 2.3: fold release checks

1. Write the five release contracts before deleting release tests.
2. Run the five release contracts.
3. Delete only old tests whose behavior is now asserted by a passing contract.

**Complete when:** the new suite has five files and no old test file duplicates a new contract.

## Phase 3: prove and record the small suite

### Goal

The repository proves both a small collection count and preserved runtime boundaries.

### Stage 3.1: verify collection and behavior

1. Run pytest collection.
2. Require 28 collected nodes.
3. Run the complete sibling suite with `tsad-lib/.venv`.
4. Require every collected test to pass.

### Stage 3.2: record evidence

1. Update the development specification checklist only after the suite passes.
2. Record the final command and collected count.
3. State that the cap counts pytest nodes, while folded tests still check repeated members.

**Complete when:** the spec records fresh 28-node evidence and the full suite passes.

## Test map

| File | Tests |
| --- | --- |
| `tests/test_data_contracts.py` | types and discovery; adapter inspection; adapter loading; pipeline; scores and metrics |
| `tests/test_baseline_contracts.py` | catalog and smoke map; Isolation Forest; reference adapter; all ready reference models |
| `tests/test_thesis_contracts.py` | O2 config; contrastive loss; offline stages; fusion override; A0; A1; A2; state restore |
| `tests/test_public_contracts.py` | notebook discovery; baseline run; reference run; THESIS smoke; device rule; CLI route |
| `tests/test_release_contracts.py` | artifacts and reports; resume roots; matrix preflight; document contract; tutorial notebooks |

## Verification

```bash
cd '/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/tsad-lib'
PYTHONPATH=src .venv/bin/python -m pytest --collect-only -q -p no:cacheprovider tests
PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests
```

## Risks and recovery

The main risk is a folded loop hiding the member that failed. Each loop must include the model or adapter name in its assertion message. If a folded test loses a unique failure explanation, split only that boundary into two tests. Do not exceed 30 nodes.

The migration is safe because test-only files change. If the 28-case suite fails to catch a regression that the former suite caught, restore the smallest missing contract assertion and keep the cap.
