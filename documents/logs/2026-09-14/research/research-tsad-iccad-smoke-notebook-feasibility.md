---
date: 2026-09-14 00:00:00 +07:00
researcher: OpenAI Codex
topic: "Minimal ICCAD smoke notebooks for tsad-lib"
status: complete
---

# Research: Minimal ICCAD smoke notebooks

## Summary

The local ICCAD source is too wide for a first notebook. `pivoted_data_all.parquet` has 39,365 rows and 117,449 columns. The existing `run()` path also cannot consume it directly: it needs train and test files, while `IccadAdapter` requires explicit `value_columns` that `load_series()` does not provide.

The smallest teaching path is notebook-local preparation. A notebook could read one fixed telemetry column, remove missing values, map source anomaly windows into known-incident labels, create a tiny `.npy` train/test dataset under its own tutorial workspace, and then call the ordinary `run()` API with `datasets=["ICCAD"]`.

## Current path

`IccadAdapter` can read a parquet file only when its caller supplies `value_columns`. `load_series()` supplies only entity-like options. `ExperimentRunner.run_request()` also requires both train and test splits.

The `.npy` fallback already supports the required tutorial layout:

```text
tutorial-workspace/
  ICCAD/
    train/iccad-mini.npy
    test/iccad-mini.npy
    test_label/iccad-mini.npy
```

For this layout, the existing runner sees `ICCAD`, `iccad-mini`, train values, test values, and test labels without a new library API.

## Teaching data contract

The full notebook fixes one source column: `datacenter1_CLIENT_component1_GET_200_endpoint330_avg`. It has a continuous local run through the selected period.

`anomaly_windows.csv` supplies operational incident windows. The labels apply to time, not to one individual channel. They do not identify which channel caused an anomaly. A zero only says that a point is outside a known incident. It does not prove normal telemetry.

The notebook can start immediately after source event `a7` ends. Its first 80 five-minute observations, from 04:25 to 11:00 UTC on 2024-03-01, are outside every known incident. They do not prove that training data contains only normal points. Its next 80 observations, from 11:05 to 17:40 UTC, form the test split. Source event `a8` occupies four test time-steps.

`location_downtime.csv` records location downtime intervals. It does not provide a clean-normal label. The user accepts known-incident-free points as the training rule and accepts possible unlabeled anomaly contamination. The notebooks must name this boundary. Test labels must never enter THESIS training, validation, threshold fitting, or scoring.

## Required notebook set

```text
notebooks/tutorials/iccad-smoke-apple-silicon-full.ipynb
notebooks/tutorials/iccad-smoke-apple-silicon-practice.ipynb
notebooks/tutorials/iccad-smoke-docker-cuda-full.ipynb
notebooks/tutorials/iccad-smoke-docker-cuda-practice.ipynb
```

Each pair tells the same story. The full version contains runnable code. The practice version keeps the same Markdown, cells, outputs, and order, but replaces only short student actions with `# TODO`.

## Boundary

The notebook plan depends on the minimal Way 1 device plan. The Apple notebook needs native THESIS support for `mps`. The Docker notebook needs native THESIS support for container-visible `cuda:0`.

The plan does not add an ICCAD runtime adapter, a generic split API, a benchmark protocol, or a full-dataset run.
