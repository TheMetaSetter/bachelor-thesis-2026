---
date: 2026-09-15 15:30:00 +07
researcher: OpenAI Codex
topic: "Find the smallest change for one THESIS smoke contract across datasets"
status: complete
related_documents:
  - documents/logs/2026-09-15/research/research-smd-smoke-budget-contract.md
  - documents/spec/tsad-lib-development-spec.md
  - documents/notes/smd-all-machines-experiment-matrix.md
---

# Research: One THESIS Smoke Contract

## The story today

`run()` sends every selection to `ExperimentRunner.run_request()`. The runner creates one `RunConfig`. That configuration starts with one Stage A epoch and one Stage B epoch.

Only the `thesis` branch runs Stage A, memory initialization, Stage B, threshold fitting, and online TTA. Reference models also read `stage_a_epochs` in their own runner. Therefore, changing the shared `RunConfig` defaults would accidentally change reference-model training.

The THESIS branch now selects an anomaly-containing 2,048-point online range when labels are present. With a 20-point window and stride 1, it contains 2,029 windows. The requested smoke run needs 4,096 windows. It therefore needs 4,115 points:

```text
20 + (4096 - 1) * 1 = 4115
```

The ICCAD tutorial notebooks currently create 80 test points. They call `run(..., models=["thesis"])`. Under one THESIS smoke contract, they must select at least 4,115 labelled test points or stop calling themselves smoke notebooks. The real ICCAD parquet has 39,365 rows, so its source has enough total rows. This research does not confirm that one chosen field has a valid 4,115-point labelled interval; the implementation must select and validate it.

## Minimal code change surface

| File | Current code | Minimal change |
| --- | --- | --- |
| `tsad-lib/src/tsad/experiment.py:336-351` | Builds one configuration for each request. | Add fixed values only for `model == "thesis"` and `experiment_type == "benchmark_smoke"`. |
| `tsad-lib/src/tsad/experiment.py:415-445` | Chooses and records the THESIS online range. | Require 4,115 labelled points, retain exactly 4,096 windows, and write both counts. |
| `tsad-lib/src/tsad/matrix.py:59-99` | Selects labelled ranges and checks the SMD O2-A0 smoke artifact. | Reuse the selector with 4,115 points and check both evidence counts. |
| `tsad-lib/tests/test_public_runtime.py` | Exercises one public THESIS request with 40 test points. | Add one mocked integration test with a 4,115-point labelled fixture. |
| `tsad-lib/tests/test_matrix.py` | Checks the old 2,048-point smoke artifact. | Check 4,115 points and 4,096 windows. |
| `tsad-lib/tests/test_thesis_runtime.py` | Protects the small generic `RunConfig` defaults. | Keep the defaults at 1/1 and add a non-THESIS guard test. |
| `tsad-lib/notebooks/tutorials/iccad-smoke-*.ipynb` | Creates an 80-point THESIS smoke input. | Select a validated 4,115-point ICCAD test segment and explain the stricter smoke contract. |
| `documents/spec/tsad-lib-development-spec.md` and `documents/notes/smd-all-machines-experiment-matrix.md` | Describe old 2,048-point smoke evidence. | State the new THESIS smoke contract after fresh artifacts prove it. |

`tsad-lib/src/tsad/types.py`, `config.py`, `api.py`, and `cli.py` do not need a change. The values are a fixed internal smoke rule, not a user setting.

## Three designs

### Design A: private THESIS smoke rule — selected

`experiment.py` owns three private constants and one small pure helper. The helper returns fixed values only for `thesis` plus `benchmark_smoke`. The THESIS runtime derives its required point range from the constants. No public API, CLI flag, or `RunConfig` field is added.

This has one small rule, preserves the generic defaults, and can be tested without a neural model.

### Design B: public smoke profile

Add `smoke_profile` to `RunConfig`, `run()`, and the CLI. A profile names `tutorial` or `full` values.

This makes the choice visible. It adds a public interface, parser behavior, validation, and more documentation. The request has one fixed smoke option, so this extra choice has no current job.

### Design C: change all `benchmark_smoke` defaults

Make every smoke request use 3/2 epochs and 4,096 windows.

This is short in code but wrong at the method boundary. Reference models use the shared Stage A field. Their training would silently change even though they do not have Stage B or THESIS online TTA.

## Decision: MECE decision tree

The decision tree is mutually exclusive and complete:

```text
Is the selected model THESIS?
├── no  → keep current configuration and method flow
└── yes → Is experiment_type benchmark_smoke?
          ├── no  → keep wet configuration and wet range policy
          └── yes → Stage A=3, Stage B=2, online windows=4096
```

Design A is selected. It makes the requested boundary explicit and gives tests three negative cases: non-THESIS smoke, THESIS wet, and non-THESIS wet.

## Minimal test story

A small unit test calls the pure helper for all four tree leaves. It proves that only one leaf returns `3`, `2`, and `4096`. A second unit test proves the point calculation returns `4,115` for a 20-point window and stride 1.

One integration test calls public `run()` with a 4,115-point labelled fixture. It uses a recording fake for `ThesisOnlineRunner`. The fake keeps the test fast, but it receives the real filtered online windows. The test checks the resolved configuration, the 4,096 received windows, and `online_range.json`.

One separate real smoke command uses the actual THESIS runner on `ServerMachineDataset`, O2-A0, seed 6. It runs all three Stage A epochs, both Stage B epochs, and all 4,096 causal windows. It is evidence for the runtime, not a permanent pytest cost.

## Evidence

- `tsad-lib/src/tsad/types.py:84-111` — shared defaults are 1/1 and validation owns them.
- `tsad-lib/src/tsad/models/time_series_library/runner.py:42-53` — reference models read `stage_a_epochs`.
- `tsad-lib/src/tsad/experiment.py:383-445` — only THESIS runs both offline stages and online TTA.
- `tsad-lib/src/tsad/data/windows.py:6-19` — stride-1 window construction gives the point-to-window formula.
- `tsad-lib/src/tsad/matrix.py:59-99` — the selector and SMD smoke gate already exist.
- `tsad-lib/notebooks/tutorials/iccad-smoke-apple-silicon-full.ipynb` — current tutorial creates 80 test points.

## Supersession note

`research-smd-smoke-budget-contract.md` limited the rule to `ServerMachineDataset`. This report supersedes that scope. The new fixed contract applies to every THESIS smoke run, regardless of dataset. It does not apply to any non-THESIS method.
