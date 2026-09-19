---
date: 09-15-2026
planner: OpenAI Codex
topic: "Reduce the repository pytest suite to 28 collected contract tests"
status: ready
revision: 61cb84bb450624031f6a27fde86262fd180565e9
branch: dev
related_research: documents/logs/09-15-2026/research/research-pytest-suite-cap-30.md
---

# Implementation Plan: Pytest Suite Cap

## Summary

Replace repeated pytest nodes with 28 boundary-focused contract tests. Keep runtime code unchanged. Delete an old test only after its behavior has a passing replacement.

## Request and constraint

Keep the complete repository test suite at no more than 30 collected pytest cases. Preserve the required tensor, training, checkpoint, loader, synthetic anomaly, visualization, and configuration evidence.

## Current and desired state

`pytest --collect-only -q` currently collects 615 nodes. The desired state is one ordinary pytest suite that collects 28 nodes and passes without parameterization creating extra nodes.

## High-level phases

### Phase 1: Establish the replacement map

Define the 28-node budget, classify current tests by risk boundary, and map every retained behavior to one target contract.

### Phase 2: Build replacement contracts

Create compact contract tests for configuration/data, models/baselines, evaluation, engine/checkpoints, online runtime, and public/release behavior. Run each group before removing its duplicates.

### Phase 3: Remove proven duplicates

Delete only old tests whose behavior is explicitly covered by a passing replacement. Keep distinct contracts separate when their failure meaning differs.

### Phase 4: Prove and record the cap

Collect the final suite, run it, verify the count is at most 30, and record fresh evidence in the project documentation.

## Scope

### In scope

- Test-file organization and pytest collection count.
- Folding repeated cases into single contract tests.
- Updating the relevant development documentation with final evidence.

### Out of scope

- Changes to `src/` runtime behavior.
- A second test runner or hidden test tier.
- Removing required behavior solely because it is expensive to test.

## Verification

Use the repository’s existing command and interpreter. Before execution, confirm the exact environment path from the project configuration. The acceptance checks are collection at no more than 30 nodes and a passing full suite.
