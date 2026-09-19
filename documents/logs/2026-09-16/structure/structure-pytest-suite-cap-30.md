---
date: 09-15-2026
topic: "Reduce the repository pytest suite to 28 collected contract tests"
status: proposed
revision: 61cb84bb450624031f6a27fde86262fd180565e9
branch: dev
related_documents:
  - documents/logs/09-15-2026/research/research-pytest-suite-cap-30.md
  - documents/logs/09-15-2026/plan/plan-pytest-suite-cap-30.md
---

# Implementation Structure: Pytest Suite Cap

## Summary

The implementation will reduce 615 collected nodes to a target of 28 by folding repeated examples into six groups of boundary contracts. The phases run from mapping, to replacement, to deletion, to proof.

## Proposed phases and sequential stages

### Phase 1: Establish the replacement map

**Result:** Every retained behavior has one target contract owner.

#### Stage 1.1: Record the baseline

Collect the current suite and record the 615-node baseline and the at-most-30 acceptance rule.

#### Stage 1.2: Assign boundaries

Assign current tests to the six boundary groups and set the target budget to 28 nodes.

**Complete when:** the map covers required evidence and identifies duplicate tests without deleting them.

### Phase 2: Build replacement contracts

**Result:** New contract tests pass while the old tests remain available for comparison.

#### Stage 2.1: Build configuration and data contracts

Cover configuration loading, loader shapes and batch size, data contracts, synthetic anomaly injection, and visualization output.

#### Stage 2.2: Build model and baseline contracts

Cover output shapes, one forward/backward pass, baseline selection, reference-model behavior, and shared model-family contracts.

#### Stage 2.3: Build evaluation and engine contracts

Cover score and metric rules, thresholds, training behavior, checkpoint round trips, scheduler/fallback behavior, and provenance.

#### Stage 2.4: Build online runtime contracts

Cover causal streaming, adaptation isolation, verification buffering, online state restoration, and threshold identity.

#### Stage 2.5: Build public and release contracts

Cover script or CLI routing, smoke execution, output structure, documentation/compliance checks, and release preflight.

**Complete when:** each target contract passes and every folded loop reports its member name on failure.

### Phase 3: Remove proven duplicates

**Result:** The suite contains only the new boundary contracts and no unverified deletion.

#### Stage 3.1: Remove group duplicates

Remove old tests one boundary group at a time after the corresponding replacement group passes.

#### Stage 3.2: Recheck coverage

Compare the replacement map with the old test inventory and restore any missing unique assertion without exceeding 30 nodes.

**Complete when:** no deleted test owns an uncovered risk boundary.

### Phase 4: Prove and record the cap

**Result:** The normal pytest command collects no more than 30 passing nodes and the repository records the evidence.

#### Stage 4.1: Collect and run

Collect the final suite, require 28 target nodes and no more than 30 nodes, then run all collected tests.

#### Stage 4.2: Record evidence

Record the exact command, count, pass result, and the distinction between pytest nodes and inputs checked inside folded tests.

**Complete when:** collection and execution pass and the documentation contains fresh evidence.

## Dependency summary

| Phase | Requires | Enables |
| --- | --- | --- |
| 1 | Current test inventory | Exact replacement map |
| 2 | Replacement map | Passing replacement contracts |
| 3 | Passing replacement contracts | Safe deletion |
| 4 | Final test files | Verified capped suite |
