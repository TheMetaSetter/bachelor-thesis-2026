---
date: 2026-09-15 13:50:19 +07
topic: "Apply the future SMD matrix smoke budget"
status: proposed
revision: 25b2b5455bbf9ca17076291d86262575c123b44e
source_structure: documents/logs/2026-09-15/structure/structure-smd-smoke-budget-contract.md
related_documents:
  - documents/logs/2026-09-15/research/research-smd-smoke-budget-contract.md
  - documents/logs/2026-09-15/plan/plan-smd-smoke-budget-contract.md
---

# Detailed Implementation: SMD Smoke Budget Contract

## Summary

This detail follows Design A. It gives future SMD matrix smoke runs five offline epochs and 4,096 online windows. It does not change wet runs or tiny ICCAD teaching runs.

## Current state

`RunConfig` holds Stage A and Stage B epoch counts, both defaulting to one. `ExperimentRunner` directly requests a 2,048-point online range. `_smoke_gate()` accepts only `point_count: 2048`.

## Desired end state

One SMD matrix smoke request resolves to:

```text
stage_a_epochs: 3
stage_b_epochs: 2
online_window_count: 4096
online point_count: 4115
```

## Phase 1: name the requested work

### Stage 1.1: add the count contract

#### Atomic steps

1. Add `online_window_count: int | None = None` to `RunConfig` in `tsad-lib/src/tsad/types.py`.
2. Reject a non-null online window count that is not positive in `RunConfig.validate()`.
3. Add one `tests/test_config.py` case for a rejected zero count.

**Complete when:** `RunConfig` can record a positive requested count and rejects zero.

### Stage 1.2: resolve SMD smoke values

#### Atomic steps

1. Identify `ServerMachineDataset` and `benchmark_smoke` in `ExperimentRunner.run_request()`.
2. Add `stage_a_epochs: 3` to that request's configuration values.
3. Add `stage_b_epochs: 2` to that request's configuration values.
4. Add `online_window_count: 4096` to that request's configuration values.
5. Add one focused test that reads the resolved SMD smoke configuration.

**Complete when:** the resolved configuration records all three requested values.

## Phase 2: create one exact online range

### Stage 2.1: derive and select the range

#### Atomic steps

1. Read `config.online_window_count` before selecting the SMD smoke range.
2. Calculate the required point count as `window_size + (online_window_count - 1) * online_stride`.
3. Pass the calculated point count to `select_online_range()`.
4. Keep only online windows fully inside the selected range.
5. Assert that the retained window count equals `config.online_window_count`.

**Complete when:** a 4,115-point SMD range produces exactly 4,096 windows.

### Stage 2.2: fail closed and write evidence

#### Atomic steps

1. Reject an SMD smoke test series shorter than the calculated point count.
2. Reject an SMD smoke test series without labels.
3. Reject an SMD smoke test series without an anomaly label.
4. Add `window_count` to `online_range.json`.
5. Change `_smoke_gate()` in `tsad-lib/src/tsad/matrix.py` to require `point_count: 4115`.
6. Change `_smoke_gate()` to require `window_count: 4096`.
7. Update the smoke-gate fixture in `tests/test_matrix.py`.

**Complete when:** an old 2,048-point artifact fails the new gate and a new 4,115-point, 4,096-window artifact passes.

## Phase 3: prove and record the story

### Stage 3.1: prove exact behavior

#### Atomic steps

1. Add a `tests/test_matrix.py` case that derives 4,115 points for 4,096 stride-1 windows.
2. Add a `tests/test_public_runtime.py` fixture with a 4,115-point labeled `ServerMachineDataset` test series.
3. Run one O2-A0 SMD smoke with that fixture.
4. Assert that `online_records.json` contains 4,096 records.
5. Assert that `online_range.json` records 4,115 points and 4,096 windows.
6. Run the focused pytest command from the plan.

**Complete when:** source-level and end-to-end tests both prove the requested counts.

### Stage 3.2: update the SSOT stories

#### Atomic steps

1. Add the SMD smoke-only range rule to `documents/notes/smd-all-machines-experiment-matrix.md`.
2. Keep the wet 2,048-point matrix rule in the same document.
3. Replace the old smoke evidence sentence in `documents/spec/tsad-lib-development-spec.md` after the new artifact exists.
4. Add the new artifact path and its counts to the development specification.

**Complete when:** both documents distinguish 2,048 wet points from 4,096 smoke windows.

## Verification

```bash
cd /Users/conquerormikrokosmos/Downloads/LAPTOP\ MAC/MYUNIVERSITY/ĐẠI\ HỌC\ QUỐC\ GIA\ TPHCM/ĐH\ KHOA\ HỌC\ TỰ\ NHIÊN/Khoá\ luận\ tốt\ nghiệp/tsad-lib
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_config.py tests/test_matrix.py tests/test_public_runtime.py
```

Then run one real `ServerMachineDataset / machine-1-6 / O2-A0 / seed 6` smoke. Confirm its manifest is completed, its Stage B hash matches its threshold, and its online artifact records both counts.

## Risk and recovery

The old smoke artifact remains valid historical evidence for the old contract but cannot pass the new gate. Do not overwrite it. If the new smoke fails, inspect its failed report row and write a new output path after fixing the exact cause.

## Blocking scope decision

This detailed plan is valid only for SMD matrix smoke. A literal global policy would break the current 80-point ICCAD teaching smoke. Choose Design B or C before applying the same rule to tutorials.

