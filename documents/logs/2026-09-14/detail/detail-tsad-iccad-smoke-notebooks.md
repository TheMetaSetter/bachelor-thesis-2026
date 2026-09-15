# Minimal ICCAD Smoke Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write four short student notebooks that smoke-run native THESIS on a tiny ICCAD-derived dataset.

**Architecture:** Each notebook creates a three-file `.npy` dataset in a tutorial workspace and calls the existing `run()` API. The Apple notebook uses `mps`. The Docker notebook uses container-visible `cuda:0`. A practice copy replaces only four short snippets with TODOs.

**Tech Stack:** Jupyter Notebook, Python, pandas, PyArrow, NumPy, PyTorch, `tsad-lib`.

**Spec:** `documents/logs/2026-09-14/structure/structure-tsad-iccad-smoke-notebooks.md`

## Global Constraints

- Finish the minimal Way 1 device plan before executing these notebooks.
- Use native `thesis`, O2, A0, seed 6, and `benchmark_smoke`.
- Read only `interval_start` and `datacenter1_CLIENT_component1_GET_200_endpoint330_avg` from ICCAD.
- Use 80 non-missing points for the source train split and 80 points for the source test split.
- Call labels known operational incident labels.
- Say that candidate training points may contain unlabeled anomalies.
- Do not call the smoke result a benchmark result.
- Keep each code cell at eight lines or fewer.
- Keep each TODO snippet at three lines or fewer.

---

## Phase 1: make the small ICCAD source

### Stage 1.1: write the shared data story

### Task 1: Add the data preparation cells to the Apple full notebook

**Files:**

- Create: `tsad-lib/notebooks/tutorials/iccad-smoke-apple-silicon-full.ipynb`

- [x] **Step 1: Create the title Markdown cell**

Write that the notebook is a small ICCAD smoke run, not a benchmark.

- [x] **Step 2: Create the import code cell**

Import `Path`, `numpy`, `pandas`, `pyarrow.parquet`, and `run`.

- [x] **Step 3: Create the path code cell**

Set `DATA` to the local thesis `data` directory and set `WORKSPACE` to `outputs/tutorial-iccad-apple`.

- [x] **Step 4: Create the source-read code cell**

Read only `interval_start` and the fixed telemetry field from `pivoted_data_all.parquet`.

- [x] **Step 5: Create the missing-value code cell**

Drop rows where the fixed telemetry field is missing.

- [x] **Step 6: Create the timestamp code cell**

Convert `interval_start` to UTC timestamps.

- [x] **Step 7: Create the event-window code cell**

Read `anomaly_windows.csv` and convert its start and end columns to UTC timestamps.

- [x] **Step 8: Create the source-label code cell**

Mark a point positive when its timestamp lies inside one known source incident window.

- [x] **Step 9: Create the segment-selection code cell**

Start at the first point after event `a7` ends and take 160 consecutive points.

- [x] **Step 10: Create the train-array code cell**

Write the first 80 values of the selected segment as a float32 array with one channel.

- [x] **Step 11: Create the test-array code cell**

Write the final 80 values of the selected segment as a float32 array with one channel.

- [x] **Step 12: Create the label-array code cell**

Write the final 80 source temporal labels of the selected segment as an integer array.

- [x] **Step 13: Create the train-file code cell**

Save the train array as `WORKSPACE/ICCAD/train/iccad-mini.npy`.

- [x] **Step 14: Create the test-file code cell**

Save the test array as `WORKSPACE/ICCAD/test/iccad-mini.npy`.

- [x] **Step 15: Create the test-label-file code cell**

Save the label array as `WORKSPACE/ICCAD/test_label/iccad-mini.npy`.

### Stage 1.2: prove the source is usable

### Task 2: Check the prepared source

**Files:**

- Modify: `tsad-lib/notebooks/tutorials/iccad-smoke-apple-silicon-full.ipynb`

- [x] **Step 1: Create the shape-check code cell**

Assert that the train array has shape `(80, 1)`.

- [x] **Step 2: Create the train-label-check code cell**

Assert that every candidate train point lies outside a known incident. Do not call this proof of normality.

- [x] **Step 3: Create the test-label-check code cell**

Assert that the test labels contain at least two positive points.

- [x] **Step 4: Run the full notebook through the source checks**

Execute cells from the title through the test-label-check cell in Jupyter.

Expected: Three `.npy` files exist, candidate train labels are all zero, and event `a8` spans four test points.

## Phase 2: teach Apple Silicon

### Stage 2.1: write the full run

### Task 3: Add the Apple runtime cells

**Files:**

- Modify: `tsad-lib/notebooks/tutorials/iccad-smoke-apple-silicon-full.ipynb`

- [x] **Step 1: Create the MPS explanation cell**

Explain that MPS is the PyTorch device name for supported Apple Silicon acceleration.

- [x] **Step 2: Create the MPS-check code cell**

Assert that `torch.backends.mps.is_available()` is true.

- [x] **Step 3: Create the run code cell**

Call `run()` with `data_root=WORKSPACE`, `datasets=["ICCAD"]`, `models=["thesis"]`, `entities=["iccad-mini"]`, `variant="O2"`, `online_variant="A0"`, `seed=6`, and `device="mps"`.

- [x] **Step 4: Create the report code cell**

Call `report.table()`.

- [x] **Step 5: Create the artifact explanation cell**

Explain that `resolved_config.yaml` records `device: mps`.

- [ ] **Step 6: Execute the Apple full notebook**

Run every cell in Jupyter on an MPS-capable Apple Silicon host.

Expected: The report table appears and the resolved configuration records `mps`.

### Stage 2.2: write the Apple practice version

### Task 4: Create the Apple practice notebook

**Files:**

- Create: `tsad-lib/notebooks/tutorials/iccad-smoke-apple-silicon-practice.ipynb`

- [x] **Step 1: Copy the Apple full notebook cells**

Copy every cell in the same order.

- [x] **Step 2: Replace the telemetry-field assignment with a TODO**

Replace only the field-name assignment with `# TODO: choose the fixed telemetry field`.

- [x] **Step 3: Replace the label expression with a TODO**

Replace only the source-label expression with `# TODO: mark timestamps inside anomaly windows`.

- [x] **Step 4: Replace the segment start with a TODO**

Replace only the event-boundary selection with `# TODO: start after event a7 ends`.

- [x] **Step 5: Replace the device value with a TODO**

Replace only `device="mps"` with `# TODO: choose the Apple device`.

- [x] **Step 6: Check the Apple practice notebook**

Confirm it has exactly four TODO snippets and no missing explanation cells.

## Phase 3: teach Docker CUDA

### Stage 3.1: write the full run

### Task 5: Create the Docker full notebook

**Files:**

- Create: `tsad-lib/notebooks/tutorials/iccad-smoke-docker-cuda-full.ipynb`

- [x] **Step 1: Copy the data cells from the Apple full notebook**

Copy the ICCAD preparation cells without changing their order.

- [x] **Step 2: Create the Docker explanation cell**

Explain that Docker exposes one selected host GPU and the notebook calls it `cuda:0`.

- [x] **Step 3: Create the Docker launch Markdown cell**

Show one command that exposes `device=<host-gpu-id>` and mounts the thesis data directory. State that the selected GPU becomes `cuda:0` inside the container.

- [x] **Step 4: Create the CUDA-check code cell**

Assert that `torch.cuda.is_available()` is true.

- [x] **Step 5: Create the CUDA-name code cell**

Print `torch.cuda.get_device_name(0)`.

- [x] **Step 6: Create the Docker path code cell**

Set `DATA` to the data directory mounted inside the container.

- [x] **Step 7: Create the Docker run code cell**

Call `run()` with the same request as Apple and `device="cuda:0"`.

- [x] **Step 8: Create the Docker report code cell**

Call `report.table()`.

- [ ] **Step 9: Execute the Docker full notebook**

Run every cell in a container with one host GPU exposed.

Expected: The report table appears and the resolved configuration records `cuda:0`.

### Stage 3.2: write the Docker practice version

### Task 6: Create the Docker practice notebook

**Files:**

- Create: `tsad-lib/notebooks/tutorials/iccad-smoke-docker-cuda-practice.ipynb`

- [x] **Step 1: Copy the Docker full notebook cells**

Copy every cell in the same order.

- [x] **Step 2: Replace the telemetry-field assignment with a TODO**

Replace only the field-name assignment with the Apple practice TODO.

- [x] **Step 3: Replace the label expression with a TODO**

Replace only the source-label expression with the Apple practice TODO.

- [x] **Step 4: Replace the segment start with a TODO**

Replace only the event-boundary selection with the Apple practice TODO.

- [x] **Step 5: Replace the device value with a TODO**

Replace only `device="cuda:0"` with `# TODO: choose the container device`.

- [x] **Step 6: Check the Docker practice notebook**

Confirm it has exactly four TODO snippets and no missing explanation cells.

## Phase 4: close the tutorial evidence

### Task 7: Verify the four notebooks

**Files:**

- Verify: all four tutorial notebooks

- [x] **Step 1: Read the Apple full title cell**

Confirm it calls the result a smoke run.

- [x] **Step 2: Read the Docker full title cell**

Confirm it calls the result a smoke run.

- [x] **Step 3: Read the Apple practice TODOs**

Confirm every TODO is three lines or fewer.

- [x] **Step 4: Read the Docker practice TODOs**

Confirm every TODO is three lines or fewer.

- [ ] **Step 5: Inspect the Apple artifact**

Confirm the completed `resolved_config.yaml` contains `device: mps`.

- [ ] **Step 6: Inspect the Docker artifact**

Confirm the completed `resolved_config.yaml` contains `device: cuda:0`.
