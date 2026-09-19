---
date: 2026-09-15
topic: "Reduce the tsad-lib pytest suite to 28 collected contract tests"
status: proposed
related_documents:
  - documents/logs/2026-09-15/research/research-tsad-test-suite-reduction.md
  - documents/logs/2026-09-15/plan/plan-tsad-test-suite-reduction.md
---

# Structure: A 28-Case TSAD Test Suite

## Summary

The suite moves from many small examples to 28 contract tests in five files. The order is simple: map the risks, replace one boundary at a time, then prove the count.

## Phase 1: make the test map

**Result:** Each important runtime boundary has one future test owner.

### Stage 1.1: count the current suite

The agent collects the current tests and records 119 nodes.

### Stage 1.2: name the target suite

The agent creates the five-file, 28-test map. No old test is deleted yet.

**Complete when:** every retained behavior appears in one target contract.

## Phase 2: build the five contract files

**Result:** The new tests express the library story without duplicated pytest nodes.

### Stage 2.1: build data contracts

The agent folds types, discovery, adapters, pipeline, scores, and metrics into five tests.

### Stage 2.2: build baseline contracts

The agent folds catalog, smoke-map, Isolation Forest, adapter, and ready reference model checks into four tests.

### Stage 2.3: build THESIS contracts

The agent folds O2, contrastive loss, offline stages, routing override, A0, A1, A2, and state restore into eight tests.

### Stage 2.4: build public contracts

The agent folds notebook API, baseline, reference, THESIS smoke, device, and CLI checks into six tests.

### Stage 2.5: build release contracts

The agent folds artifacts, output roots, SMD preflight, documents, and tutorials into five tests.

**Complete when:** each new file passes before old duplicated tests are removed.

## Phase 3: remove duplicates and verify

**Result:** `pytest` collects 28 nodes and the TSAD runtime story remains protected.

### Stage 3.1: remove replaced files

The agent deletes an old test only after the matching new contract passes.

### Stage 3.2: run final proof

The agent collects the suite, runs the suite, and records the final result in the development specification.

**Complete when:** collection is 28, all tests pass, and the checklist has evidence.

## Dependency summary

| Phase | Requires | Enables |
| --- | --- | --- |
| 1 | Current suite | Exact replacement map |
| 2 | Exact replacement map | Passing contract tests |
| 3 | Passing contract tests | Safe duplicate removal and final count |

## Decision

The cap counts collected pytest nodes. A contract test may loop through ready models or adapter families because those members share one contract.
