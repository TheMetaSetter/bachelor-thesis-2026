# Remaining SMD 4-V100 tmux Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the complete fair benchmark matrix for the 8 selected SMD machines on 4 Tesla V100 GPUs and 44 CPU cores through resumable `tmux` sessions.

**Architecture:** Separate GPU-bound runs from CPU-only traditional baselines.
Offline GPU and CPU queues run concurrently, then a completion barrier verifies all required checkpoints before online runs start.
Online GPU and CPU queues then run concurrently and one collector writes the compact metric reports.

**Tech Stack:** Bash, Python, PyTorch, CUDA, `CUDA_VISIBLE_DEVICES`, `taskset`, `tmux`, YAML configs, JSON manifests, Pytest, and the existing benchmark runners.

**Spec:** `documents/logs/09-09-2026/research/design-remaining-smd-benchmark-scripts.md` and `documents/notes/smd-28-machine-drift-zoo.md`.

## Global Constraints

- Use exactly these 8 new machines: `machine-3-1`, `machine-3-5`, `machine-3-2`, `machine-3-11`, `machine-3-10`, `machine-1-3`, `machine-1-1`, and `machine-2-8`.
- Keep `machine-1-6`, `machine-3-4`, and `machine-3-9` excluded.
- Use seeds `6`, `8`, and `36`.
- Run THESIS offline variants `O0` and `O1`.
- Run THESIS online variants `A0`, `A1`, and `A2` for each offline variant.
- Run RedLamp, STUMPY, KMeansAD, Isolation Forest, CANDI, and M2N2 according to the existing benchmark design.
- Keep the exact main metric name `VUS-PR@FPR-budget`.
- Report `VUS-PR@FPR-budget`, `VUS-PR`, `Affiliation F1-score`, `VUS-ROC`, and `raw-FPR` only.
- Use the same deterministic 2048-point anomaly-containing online subsequence for every method, seed, and variant within one entity.
- Keep smoke and wet modes separate.
- Retain only summary artifacts, required checkpoints, protocol/config provenance, and compact logs.
- Treat 44 cores as CPU/vCPU cores; if the server reports a different resource type, stop and revise the CPU allocation.
- Never run two GPU-bound processes on the same V100.
- Do not start the wet matrix until one complete smoke combination has passed on the cloud server.

## Resource Allocation

The matrix contains 144 offline runs, 264 online runs, and 408 runs in total.

| Queue | `tmux` sessions | GPU assignment | CPU cores | Runs |
| --- | ---: | --- | --- | ---: |
| Offline GPU | 4 | One V100 per session | 8 cores per session | 72 |
| Offline CPU | 2 | None | 6 cores per session | 72 |
| Online GPU | 4 | One V100 per session | 8 cores per session | 192 |
| Online CPU | 2 | None | 6 cores per session | 72 |

The four GPU sessions use CPU core ranges `0-7`, `8-15`, `16-23`, and `24-31`.

The two CPU sessions use CPU core ranges `32-37` and `38-43`.

Each GPU session reserves eight CPU cores for the main trainer, DataLoader workers, CUDA host-side work, and library overhead.

GPU workers set `CUDA_VISIBLE_DEVICES` to one physical GPU and expose four data-loader workers offline and two data-loader workers online.

CPU workers set `CUDA_VISIBLE_DEVICES` to an empty value and cap BLAS, OpenMP, and NumPy thread pools to the assigned CPU range.

The four GPU sessions use 32 CPU cores in total, and the two CPU sessions use the remaining 12 CPU cores.

## Phase 1: Cloud Preflight

### Stage 1.1: Verify the remote checkout

- [ ] Read the remote repository path and virtual-environment path from `cloud-gpu.txt`.
- [ ] Verify the remote Git revision without changing the checkout.
- [ ] Verify that `.venv/bin/python` exists and imports PyTorch.
- [ ] Verify that the dataset root contains matching `train`, `test`, and `test_label` files for all 8 machines.

### Stage 1.2: Verify hardware and process limits

- [ ] Run `nvidia-smi -L` and confirm four Tesla V100 devices.
- [ ] Run `nvidia-smi` and record available memory on every GPU.
- [ ] Run `nproc` and confirm at least 44 CPU cores.
- [ ] Run `tmux -V`, `taskset --version`, and `df -h`.
- [ ] Confirm no unrelated GPU jobs occupy devices `0`, `1`, `2`, or `3`.

### Stage 1.3: Run the cloud smoke preflight

- [ ] Add the exact 8 entity IDs to an explicit entity-selection input.
- [ ] Generate a smoke manifest without starting a worker.
- [ ] Verify that the manifest contains 408 planned run records before execution.
- [ ] Verify that every online record has a 2048-point range with at least one anomaly point.
- [ ] Verify that GPU-bound records use CUDA and traditional records use CPU.

## Phase 2: Manifest and Configuration Support

### Stage 2.1: Add explicit entity selection

**Files:**

- Modify: `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py`
- Test: `tests/benchmarks/test_remaining_smd_benchmark_matrix.py`

- [ ] Add an `--entity-id` repeated CLI option or an `--entity-file` option.
- [ ] Reject an empty selection and reject IDs that are not discovered in the dataset.
- [ ] Preserve the existing three-machine exclusion rule even when explicit IDs are supplied.
- [ ] Preserve sorted entity order so assignment remains reproducible.
- [ ] Test that the selected input produces exactly the 8 requested machines.

### Stage 2.2: Add resource metadata to the manifest

**Files:**

- Modify: `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py`
- Test: `tests/benchmarks/test_remaining_smd_benchmark_matrix.py`

- [ ] Add `resource_class: "gpu"` to THESIS offline, RedLamp, THESIS online, CANDI, and M2N2 records.
- [ ] Add `resource_class: "cpu"` to STUMPY, KMeansAD, and Isolation Forest records.
- [ ] Add `phase_group: "offline"` or `phase_group: "online"` to every record.
- [ ] Add dependency metadata for THESIS online records on the matching Stage B checkpoint and threshold artifact.
- [ ] Add dependency metadata for CANDI and M2N2 records on the matching RedLamp checkpoint.
- [ ] Test that each run has one resource class, one phase group, and the expected dependency paths.

### Stage 2.3: Bound data-loader and thread usage

**Files:**

- Modify: `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py`
- Test: `tests/benchmarks/test_remaining_smd_benchmark_matrix.py`

- [ ] Set GPU offline configs to `num_workers: 4`.
- [ ] Set GPU online configs to `num_workers: 2`.
- [ ] Set CPU baseline configs to `num_workers: 0` where the runner does not need data-loader subprocesses.
- [ ] Keep batch sizes and training epochs unchanged from the approved smoke and wet policy.
- [ ] Test that resource-specific worker limits are present in generated YAML files.

## Phase 3: Queue Runner and tmux Sessions

### Stage 3.1: Extend the existing matrix launcher

**Files:**

- Modify: `scripts/benchmarks/run_remaining_smd_matrix.sh`
- Modify: `scripts/benchmarks/run_remaining_smd_smoke.sh`
- Modify: `scripts/benchmarks/run_remaining_smd_wet.sh`
- Test: `tests/benchmarks/test_remaining_smd_benchmark_matrix.py`

- [ ] Replace the hard-coded `GPU_COUNT=2` restriction with an explicit `--gpu-count` value validated against the discovered GPU count.
- [ ] Add `--entity-file`, `--phase-group`, `--resource-class`, `--cpu-worker-count`, and `--cpu-core-list` options.
- [ ] Keep the existing two-GPU default for backward compatibility unless the cloud launcher explicitly passes four GPUs.
- [ ] Make worker assignment deterministic by sorting runs by estimated cost and then assigning them round-robin.
- [ ] Ensure each worker executes one run at a time.
- [ ] Ensure `--skip-completed` checks the exact report path from the manifest.
- [ ] Ensure a failed worker does not trigger online execution.

### Stage 3.2: Add cloud orchestration wrapper

**Files:**

- Create: `scripts/benchmarks/run_remaining_smd_cloud_tmux.sh`
- Test: `tests/benchmarks/test_remaining_smd_cloud_launcher.sh`

- [ ] Generate one manifest for the selected 8 machines.
- [ ] Start four offline GPU sessions named `smd-offline-gpu-0` through `smd-offline-gpu-3`.
- [ ] Start two offline CPU sessions named `smd-offline-cpu-0` and `smd-offline-cpu-1`.
- [ ] Pin each GPU session with `CUDA_VISIBLE_DEVICES` and its assigned `taskset` CPU range.
- [ ] Pin each CPU session with `CUDA_VISIBLE_DEVICES=` and its assigned `taskset` CPU range.
- [ ] Attach a separate `tmux pipe-pane` log to every session.
- [ ] Store logs under `outputs/tmux_logs/remaining_smd/<mode>/`.
- [ ] Wait for all offline sessions before launching any online session.
- [ ] Verify every required offline checkpoint and report before opening the online sessions.
- [ ] Start four online GPU sessions named `smd-online-gpu-0` through `smd-online-gpu-3`.
- [ ] Start two online CPU sessions named `smd-online-cpu-0` and `smd-online-cpu-1`.
- [ ] Start the collector only after all online sessions exit successfully.
- [ ] Return a non-zero exit code if any worker fails.

### Stage 3.3: Add resource-safe environment settings

**Files:**

- Create: `scripts/benchmarks/_remaining_smd_resource_env.sh`
- Test: `tests/benchmarks/test_remaining_smd_cloud_launcher.sh`

- [ ] Set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` for GPU sessions so DataLoader workers do not multiply thread counts.
- [ ] Set the same variables to `6` for CPU sessions and pin each CPU session to its six-core mask.
- [ ] Set GPU session CPU masks to `0-7`, `8-15`, `16-23`, and `24-31`.
- [ ] Set CPU session masks to `32-37` and `38-43`.
- [ ] Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` only if the V100 smoke run shows allocator fragmentation.
- [ ] Do not enable multiple processes per GPU.
- [ ] Print the effective GPU index, CPU mask, process ID, and run ID before each run.

## Phase 4: Smoke Execution

### Stage 4.1: Run one end-to-end smoke combination

- [ ] Select one machine, one seed, one THESIS offline variant, and one online variant.
- [ ] Run Stage A for 3 epochs and Stage B for 2 epochs on one V100.
- [ ] Verify the best Stage B checkpoint and threshold artifact.
- [ ] Run one THESIS online adaptation stream on the selected 2048-point range.
- [ ] Verify `VUS-PR@FPR-budget`, `VUS-PR`, `Affiliation F1-score`, `VUS-ROC`, and `raw-FPR` are written.
- [ ] Run one CANDI and one M2N2 smoke case on CUDA.
- [ ] Verify that their model, optimizer, adapter, and batch tensors are on the assigned V100.
- [ ] Run one traditional baseline smoke case on a CPU session.

### Stage 4.2: Run the complete smoke matrix

- [ ] Start the 4-GPU and 2-CPU smoke queues.
- [ ] Monitor all sessions with `tmux list-sessions`, `nvidia-smi`, and `tail` on exact log files.
- [ ] Confirm that no worker uses a GPU outside its assigned `CUDA_VISIBLE_DEVICES` value.
- [ ] Confirm that the offline barrier blocks online sessions until all required checkpoints exist.
- [ ] Run the compact collector and inspect the Markdown table layout.
- [ ] Stop before wet execution if any checkpoint, dependency, device, or metric contract fails.

## Phase 5: Wet Execution

### Stage 5.1: Prepare the wet matrix

- [ ] Generate the wet manifest for the same 8 machines and seeds.
- [ ] Verify Stage A uses 25 epochs and Stage B uses 5 epochs.
- [ ] Verify RedLamp uses 30 epochs.
- [ ] Verify online runs use the same entity-specific 2048-point anomaly-containing ranges as smoke mode.
- [ ] Verify W&B names remain within the human-centered artifact naming policy.

### Stage 5.2: Execute offline queues

- [ ] Start the four GPU offline sessions.
- [ ] Start the two CPU offline sessions.
- [ ] Resume only missing runs with `--skip-completed` after a failure.
- [ ] Verify all 144 offline reports and all required best checkpoints.

### Stage 5.3: Execute online queues

- [ ] Start the four GPU online sessions only after the offline barrier passes.
- [ ] Start the two CPU online sessions at the same barrier.
- [ ] Resume only missing online reports with `--skip-completed`.
- [ ] Verify all 264 online reports before collection.

### Stage 5.4: Collect and audit results

- [ ] Run `scripts/benchmarks/collect_remaining_smd_metrics.py` against the final manifest.
- [ ] Check that every entity, seed, method, offline variant, and online variant has one report row.
- [ ] Check that the tables report only `VUS-PR@FPR-budget`, `VUS-PR`, `Affiliation F1-score`, `VUS-ROC`, and `raw-FPR`.
- [ ] Check that failed or missing runs are listed separately from valid metric rows.
- [ ] Save the compact JSON and Markdown reports under the wet output root.

## Phase 6: Verification

### Stage 6.1: Static and unit verification

- [ ] Run `bash -n` on every modified shell script.
- [ ] Run focused Pytest tests for entity selection, manifest counts, resource classes, dependency paths, and worker assignment.
- [ ] Run `git diff --check`.
- [ ] Run `.venv/bin/python -m py_compile` on every modified Python file.

### Stage 6.2: Runtime verification

- [ ] Confirm the cloud smoke run exits with status zero.
- [ ] Confirm `nvidia-smi` shows at most one active benchmark process per V100.
- [ ] Confirm CPU utilization stays within the assigned 44-core partition.
- [ ] Confirm no online run starts with a missing checkpoint.
- [ ] Confirm collector output is reproducible when run twice without new benchmark data.

## Planned Cloud Commands

The following commands are examples for the implementation after the launcher changes are complete.

```bash
cd /root/bachelor-thesis-2026
```

```bash
nvidia-smi -L
```

```bash
nproc
```

```bash
bash scripts/benchmarks/run_remaining_smd_cloud_tmux.sh --mode smoke --gpu-count 4 --cpu-worker-count 2 --entity-id machine-3-1 --entity-id machine-3-5 --entity-id machine-3-2 --entity-id machine-3-11 --entity-id machine-3-10 --entity-id machine-1-3 --entity-id machine-1-1 --entity-id machine-2-8
```

```bash
tmux list-sessions
```

```bash
nvidia-smi
```

```bash
bash scripts/benchmarks/run_remaining_smd_cloud_tmux.sh --mode wet --gpu-count 4 --cpu-worker-count 2 --entity-id machine-3-1 --entity-id machine-3-5 --entity-id machine-3-2 --entity-id machine-3-11 --entity-id machine-3-10 --entity-id machine-1-3 --entity-id machine-1-1 --entity-id machine-2-8 --skip-completed
```

## Expected Runtime

The 4-GPU wet run is expected to take roughly 12–27 hours after smoke validation.

The 4-GPU smoke run is expected to take roughly 1–2 hours.

These are extrapolations, not measurements on this server.

The first one-machine smoke combination must establish the actual V100 throughput before the full wet estimate is trusted.

## Self-Review

The plan covers explicit machine selection, all 408 runs, GPU and CPU separation, 4-GPU placement, 44-core limits, checkpoint dependencies, smoke and wet modes, resumability, logging, collection, and verification.

No code change is authorized by this plan alone.
