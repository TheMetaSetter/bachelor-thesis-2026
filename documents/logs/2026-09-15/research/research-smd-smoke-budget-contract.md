---
date: 2026-09-15 13:50:19 +07
researcher: OpenAI Codex
topic: "Find the smallest change for future SMD smoke runs: Stage A 3 epochs, Stage B 2 epochs, and 4,096 online windows"
status: complete
revision: 25b2b5455bbf9ca17076291d86262575c123b44e
branch: dev
---

# Research: SMD Smoke Budget Contract

## Summary

The current `tsad-lib` smoke path runs Stage A once, Stage B once, and selects a 2,048-point online range. With a 20-point window and stride 1, that range creates 2,029 online windows. A future SMD matrix smoke needs three Stage A epochs, two Stage B epochs, and exactly 4,096 online windows.

The smallest safe change is to add one explicit `online_window_count` configuration field. `ExperimentRunner` should set it to `4096` only for `ServerMachineDataset` matrix smoke runs. It should derive the required point span as `window_size + (online_window_count - 1) * online_stride`. The required span is `4,115` points.

## Research question

Which existing `tsad-lib` lines must change for the future SMD smoke budget, and which minimal designs are possible?

## Story of the current run

A notebook or CLI call enters `tsad.api.run()`. It creates `ExperimentRunner`. The runner creates `RunConfig` with the current defaults. Stage A reads `stage_a_epochs`. Stage B reads `stage_b_epochs`.

The runner then builds online windows with stride 1. It asks `select_online_range()` for a 2,048-point range. It keeps only windows fully inside that range. `make_windows()` starts one window at each point. Therefore, a point range of length `P` produces `P - 20 + 1` windows when the stride is 1.

For the requested future smoke:

```text
4,096 windows + 20 window points - 1 = 4,115 points
```

## Confirmed change surface

| File | Current responsibility | Minimal change needed |
| --- | --- | --- |
| `tsad-lib/src/tsad/types.py:84-85,110-111` | Stores and validates Stage A and Stage B epoch counts. | Add and validate an optional `online_window_count`. Keep existing epoch fields. |
| `tsad-lib/src/tsad/experiment.py:336-351` | Creates resolved configuration for a request. | Set Stage A to `3`, Stage B to `2`, and online window count to `4096` for the selected SMD matrix smoke scope. |
| `tsad-lib/src/tsad/experiment.py:417-444` | Selects the online range and writes its artifact. | Derive a 4,115-point range, reject an insufficient labeled test series, assert 4,096 windows, and write `window_count`. |
| `tsad-lib/src/tsad/matrix.py:59-69` | Selects one anomaly-containing point range. | Reuse it with the derived 4,115-point length. No new selector is required. |
| `tsad-lib/src/tsad/matrix.py:72-99` | Accepts a smoke artifact only when it records 2,048 points. | Require the new smoke evidence: 4,115 points and 4,096 windows. |
| `tsad-lib/tests/test_matrix.py:33-73` | Checks the old 2,048-point contract and smoke gate. | Check the new SMD smoke contract. |
| `tsad-lib/tests/test_config.py` | Checks resolved configuration behavior. | Check positive and explicit online window count behavior. |
| `tsad-lib/tests/test_public_runtime.py` | Exercises public `run()`. | Add one small `ServerMachineDataset` smoke case that proves 4,096 records. |
| `documents/notes/smd-all-machines-experiment-matrix.md:21-23,392` | Defines the wet SMD matrix as 2,048 points. | State that the wet matrix remains 2,048 points while the future SMD smoke contract uses 4,096 windows. |
| `documents/spec/tsad-lib-development-spec.md:1731-1787` | Records the current SMD smoke evidence. | Replace the old smoke-range story after a new artifact proves the new contract. |

## Three alternative designs

### Design A: SMD matrix smoke only — recommended

The runner applies `3`, `2`, and `4096` only when the request uses `ServerMachineDataset` and `experiment_type="benchmark_smoke"`. The public notebook and CLI interfaces stay unchanged. Other smoke uses, including the tiny ICCAD teaching notebooks, keep their current behavior.

This design changes the fewest code paths. It fits the request that followed an SMD all-machines smoke run.

### Design B: one strict rule for every smoke run

`RunConfig` defaults become `3`, `2`, and `4096` for every `benchmark_smoke` request. A smoke run fails if its labeled test series cannot supply 4,115 points.

This design follows the literal phrase “all future smoke runs.” It also makes the current 80-point ICCAD tutorial smoke runs fail. The tutorials would need larger data or a different experiment type.

### Design C: explicit smoke purpose

Add a public `smoke_mode` value such as `matrix` or `tutorial`. `matrix` means `3`, `2`, and `4096`. `tutorial` keeps a small input. The notebook API and CLI both expose the value.

This design keeps both meanings clear. It adds a public option, CLI parsing, validation, tests, and documentation. It is less minimal than Design A.

## Conflict and uncertainty

The existing matrix and development specification define a 2,048-point online range for the wet all-machine benchmark. The ICCAD notebooks define an 80-point test series and call it a smoke run. The available code does not decide whether the new requirement replaces those two contracts.

This report recommends Design A. It treats the new values as an SMD matrix smoke contract. If the new rule must cover ICCAD tutorials too, choose Design B or C before implementation.

## Evidence

- `tsad-lib/src/tsad/types.py:84-111` — current epoch defaults are `1` and validation already owns training-size checks.
- `tsad-lib/src/tsad/models/thesis.py:208-229` — Stage A repeats its real optimizer loop by `stage_a_epochs`.
- `tsad-lib/src/tsad/models/thesis.py:286-302` — Stage B repeats its real optimizer loop by `stage_b_epochs`.
- `tsad-lib/src/tsad/experiment.py:417-423` — the runner hard-codes the current 2,048-point online range.
- `tsad-lib/src/tsad/data/windows.py:6-19` — stride-1 windows begin at every valid absolute index.
- `tsad-lib/src/tsad/matrix.py:59-69` — range selection already centers an anomaly-containing range.
- `tsad-lib/src/tsad/matrix.py:72-99` — the smoke gate currently expects 2,048 points.
- `tsad-lib/notebooks/tutorials/iccad-smoke-apple-silicon-full.ipynb` — the teaching smoke creates an 80-point test series.
- `documents/notes/smd-all-machines-experiment-matrix.md:21-23` — the wet matrix records its 2,048-point range.

