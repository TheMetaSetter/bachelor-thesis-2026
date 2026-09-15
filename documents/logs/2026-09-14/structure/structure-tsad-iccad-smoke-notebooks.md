---
date: 2026-09-14 00:00:00 +07:00
topic: "Minimal ICCAD smoke notebooks for tsad-lib"
status: approved
related_documents:
  - documents/logs/2026-09-14/research/research-tsad-iccad-smoke-notebook-feasibility.md
  - documents/logs/2026-09-14/plan/plan-tsad-iccad-smoke-notebooks.md
---

# Implementation Structure: Minimal ICCAD smoke notebooks

## Summary

The user asked for the whole workflow in one request. This structure is approved for detail.

The notebooks first make a tiny local dataset. Then they run one native model. Finally, they give students a safe place to fill a few missing lines.

## Phase 1: prepare one small source

### Stage 1.1: read one named telemetry field

The full notebook reads `datacenter1_CLIENT_component1_GET_200_endpoint330_avg` and `interval_start`. It removes missing telemetry values. It never reads all 117,449 columns.

### Stage 1.2: make source labels visible

The full notebook maps `anomaly_windows.csv` to timestamps. These are known operational incident labels. They are shared across channels. A zero does not prove normal telemetry.

### Stage 1.3: write the existing run layout

The notebook writes `train/iccad-mini.npy`, `test/iccad-mini.npy`, and `test_label/iccad-mini.npy` in a tutorial-only workspace. The runtime needs no ICCAD adapter change. Candidate train contamination is accepted for this smoke tutorial.

**Complete when:** Existing discovery finds train and test splits for `ICCAD/iccad-mini`.

## Phase 2: teach Apple Silicon

### Stage 2.1: check MPS

The notebook checks MPS before the model runs. It stops with a short message when MPS is unavailable.

### Stage 2.2: run the full story

The notebook calls `run()` with native `thesis`, O2, A0, seed 6, and `device="mps"`. It shows `report.table()`.

### Stage 2.3: create the practice story

The practice copy preserves every explanation. It replaces four short code snippets with TODO markers.

**Complete when:** A student can read or practice the same MPS story without reading library internals.

## Phase 3: teach Docker CUDA

### Stage 3.1: explain container GPU mapping

The notebook says Docker selects the host GPU. Inside the container, one exposed GPU is `cuda:0`.

### Stage 3.2: check CUDA

The notebook checks CUDA availability before the model runs. It prints the container-visible device name.

### Stage 3.3: run and practise the same story

The full notebook calls `run(..., device="cuda:0")`. The practice copy uses the same four TODO locations as the Apple practice notebook.

**Complete when:** Students do not need to know whether the rented host GPU is an RTX or Tesla model, or which host index it had.

## Phase 4: preserve honest limits

### Stage 4.1: label the run correctly

Each notebook calls itself a smoke run. It says that candidate training points are outside known incidents and may contain unlabeled anomalies. Test points contain four time-steps of source event `a8`. Labels do not name the anomalous channel. The result is not a benchmark claim.

### Stage 4.2: check each final table

Each full notebook produces a `Report.table()` result and a `resolved_config.yaml` with the intended device.

### Stage 4.3: check each practice notebook

Each practice notebook has exactly four TODO snippets. Each TODO asks for a small action already explained by nearby Markdown.

**Complete when:** The teaching content is simple, runnable, and does not hide scientific limits.
