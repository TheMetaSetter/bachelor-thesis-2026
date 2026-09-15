---
date: 2026-09-15 13:50:19 +07
planner: OpenAI Codex
topic: "Apply the future SMD matrix smoke budget"
status: proposed
revision: 25b2b5455bbf9ca17076291d86262575c123b44e
branch: dev
related_research: documents/logs/2026-09-15/research/research-smd-smoke-budget-contract.md
---

# Implementation Plan: SMD Smoke Budget Contract

## Summary

This plan follows Design A from the research report. Future `ServerMachineDataset` matrix smoke runs will use three Stage A epochs, two Stage B epochs, and exactly 4,096 online windows. Wet matrix runs and tiny ICCAD teaching runs will not change.

## Request

The future smoke contract is five offline epochs in total and 4,096 online TTA windows. The change must stay minimal and preserve real training, threshold fitting, evaluation, and artifact checks.

## Current state

`ExperimentRunner` creates one configuration for every request. The configuration defaults to one Stage A epoch and one Stage B epoch. The runner always asks for a 2,048-point online range when labels are present. The matrix smoke gate accepts only an artifact with `point_count: 2048`.

## Desired end state

A future SMD matrix smoke run writes a resolved configuration with `stage_a_epochs: 3`, `stage_b_epochs: 2`, and `online_window_count: 4096`. Its online range artifact records `point_count: 4115` and `window_count: 4096`.

## Scope

### In scope

- `ServerMachineDataset` with `benchmark_smoke`.
- THESIS epoch and online-window settings.
- Range artifact, smoke gate, focused tests, and SMD documents.

### Out of scope

- The 1,764-unit wet matrix.
- Model architecture, losses, threshold formula, and metric formula.
- ICCAD tutorial data size and tutorial runtime.
- Public notebook and CLI parameters.

## Implementation approach

The runner will add three values to the existing resolved configuration only for SMD matrix smoke. It will reuse the existing range selector. It will calculate the point span from the requested number of windows. It will fail closed when a SMD matrix smoke cannot produce exactly 4,096 windows.

## Phase 1: make one SMD smoke contract

### Goal

The resolved configuration names every requested budget and validates the new window count.

### Changes

- **File:** `tsad-lib/src/tsad/types.py`
- **Symbol:** `RunConfig`
- **Change:** Add optional `online_window_count`. Reject zero or negative values.
- **Reason:** The artifact needs one clear field for the requested count.

- **File:** `tsad-lib/src/tsad/experiment.py`
- **Symbol:** `ExperimentRunner.run_request()`
- **Change:** Add `stage_a_epochs=3`, `stage_b_epochs=2`, and `online_window_count=4096` only for SMD matrix smoke.
- **Reason:** Existing notebook and CLI calls keep their interface.

### Verification

- [ ] `tests/test_config.py` proves positive validation.
- [ ] A resolved SMD smoke config records `3`, `2`, and `4096`.

## Phase 2: run exactly the requested online work

### Goal

The online runner receives exactly 4,096 causal windows from one anomaly-containing SMD range.

### Changes

- **File:** `tsad-lib/src/tsad/experiment.py`
- **Symbol:** `ExperimentRunner.run_request()`
- **Change:** Derive `4115` points from the existing window size and stride. Select and filter one range. Reject a short, unlabeled, or non-anomalous SMD smoke input. Write both point and window counts.

- **File:** `tsad-lib/src/tsad/matrix.py`
- **Symbol:** `_smoke_gate()`
- **Change:** Require `point_count: 4115` and `window_count: 4096` for the new smoke contract.

### Verification

- [ ] `tests/test_matrix.py` proves the range gives 4,096 stride-1 windows.
- [ ] `tests/test_public_runtime.py` proves one real one-channel SMD smoke creates 4,096 online records.

## Phase 3: keep the story and the gate aligned

### Goal

The documents distinguish the new SMD smoke contract from the unchanged wet matrix.

### Changes

- **File:** `documents/notes/smd-all-machines-experiment-matrix.md`
- **Change:** State that wet runs keep their 2,048-point range. State that future SMD smoke uses 4,115 points for 4,096 windows.

- **File:** `documents/spec/tsad-lib-development-spec.md`
- **Change:** Update the smoke evidence story only after a new artifact passes the new gate.

### Verification

- [ ] The updated documents contain both contracts without calling them the same thing.
- [ ] One new O2-A0 SMD smoke artifact passes `_smoke_gate()`.

## Testing strategy

Use focused pytest tests first. Then run:

```bash
cd /Users/conquerormikrokosmos/Downloads/LAPTOP\ MAC/MYUNIVERSITY/ĐẠI\ HỌC\ QUỐC\ GIA\ TPHCM/ĐH\ KHOA\ HỌC\ TỰ\ NHIÊN/Khoá\ luận\ tốt\ nghiệp/tsad-lib
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_config.py tests/test_matrix.py tests/test_public_runtime.py
```

Run one real O2-A0 SMD smoke after these tests. Do not start wet runs.

## Risks and recovery

The old smoke artifact has 2,048 points and will not pass the new gate. Keep it unchanged as historical evidence. Write the new run to its own non-overwriting artifact path or use the existing resume guard only after identity validation.

## Blocking decision

This plan assumes the new values apply to SMD matrix smoke only. If they must apply to tiny ICCAD teaching smoke runs too, do not implement this plan. Choose Design B or C from the research report first.

