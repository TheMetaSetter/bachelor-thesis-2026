---
date: 2026-09-15 13:50:19 +07
topic: "Apply the future SMD matrix smoke budget"
status: proposed
revision: 25b2b5455bbf9ca17076291d86262575c123b44e
related_documents:
  - documents/logs/2026-09-15/research/research-smd-smoke-budget-contract.md
  - documents/logs/2026-09-15/plan/plan-smd-smoke-budget-contract.md
---

# Implementation Structure: SMD Smoke Budget Contract

## Summary

The work has three phases. First the configuration states the SMD smoke budget. Next the runner turns that budget into an exact causal range. Last, tests and documents prove the new story.

## Scope

This structure follows Design A. It changes only `ServerMachineDataset` matrix smoke runs. Wet SMD and tiny ICCAD teaching runs remain outside this structure.

## Phase 1: name the requested work

**Result:** A resolved SMD smoke configuration contains three Stage A epochs, two Stage B epochs, and 4,096 online windows.

### Stage 1.1: add the count contract

`RunConfig` accepts one optional online window count and rejects invalid counts.

### Stage 1.2: resolve the SMD smoke values

`ExperimentRunner` adds the three fixed values only when a request is SMD matrix smoke.

## Phase 2: create one exact online range

**Result:** The online runner sees 4,096 causal windows, and its artifact proves the count.

### Stage 2.1: derive and select the range

The runner derives 4,115 points from 4,096 windows, window size 20, and stride 1. It reuses the existing anomaly-centered selector.

### Stage 2.2: fail closed and write evidence

The runner rejects insufficient SMD smoke input. It writes both point and window counts. The smoke gate checks both values.

## Phase 3: prove and record the story

**Result:** Tests and documents distinguish SMD smoke from the wet matrix.

### Stage 3.1: prove the exact counts

Focused tests prove configuration, range arithmetic, artifact gate, and one real runtime path.

### Stage 3.2: update the two SSOT stories

The matrix note and development specification explain that the wet range stays 2,048 points while the new smoke range is 4,115 points.

## Dependency summary

| Phase | Requires | Enables |
| --- | --- | --- |
| 1 | Design A scope | An explicit smoke request |
| 2 | Phase 1 config | Exact causal execution and artifact evidence |
| 3 | Phase 2 artifact schema | Tests and aligned documents |

## Feedback needed

The structure is ready only if “future smoke run” means SMD matrix smoke. If it also means ICCAD tutorial smoke, this structure must change to Design B or C.

