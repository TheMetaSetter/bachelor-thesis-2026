---
date: 2026-09-09 15:06:11 +07:00
researcher: OpenAI Codex
topic: "Detect code changes for 4-V100 tmux orchestration on 8 SMD machines"
status: complete
revision: 672f9af37a5570f6fb52fe23d704a78beb4eae1e
branch: dev
---

# Research: Detect code changes for 4-V100 tmux orchestration on 8 SMD machines

## Summary

The active launcher supports only two GPU workers, assigns complete entities to those workers, and starts one coordinator `tmux` session.

The active generator discovers all 25 non-excluded SMD entities and has no CLI option to select only the requested 8 entities.

The active generated GPU configs request 12 offline DataLoader workers and 8 online DataLoader workers, but the launcher does not reserve CPU cores for those workers.

The active code already propagates `device: cuda` into THESIS, CANDI, and M2N2 runtime paths, so those model classes do not need a new GPU interface based on current evidence.

The active online report stores the last online history record as `final_metrics`, while the collector expects a nested metric dictionary containing `VUS-PR@FPR-budget`, `VUS-PR`, `Affiliation F1-score`, `VUS-ROC`, and `raw-FPR`.

The code lines that require modification are concentrated in the remaining-SMD generator, matrix launcher, cloud `tmux` wrapper, resource settings, and tests.

## Research question

Which active code lines must change to run all offline and online methods for the 8 selected SMD machines on 4 Tesla V100 GPUs and 44 CPU cores through resource-aware `tmux` sessions?

The investigation covers entity selection, run-manifest construction, device assignment, DataLoader worker limits, CPU pinning, phase dependencies, session lifecycle, resumability, metric collection, and tests.

## System context

The requested machines are the first 11 rows of `documents/notes/smd-28-machine-drift-zoo.md` after removing `machine-1-6`, `machine-3-9`, and `machine-3-4`.

The resulting 8 machines are `machine-3-1`, `machine-3-5`, `machine-3-2`, `machine-3-11`, `machine-3-10`, `machine-1-3`, `machine-1-1`, and `machine-2-8`.

The matrix has 144 offline runs, 264 online runs, and 408 runs in total for seeds `6`, `8`, and `36`.

The resource assumption is that the stated 44 cores are CPU or vCPU cores.

## Execution path

`run_remaining_smd_smoke.sh` and `run_remaining_smd_wet.sh` pass their arguments directly to `run_remaining_smd_matrix.sh`.

`run_remaining_smd_matrix.sh` builds generator arguments, creates a manifest, selects a runner for each record, and starts the coordinator.

The current coordinator launches only GPU workers `0` and `1`, passes `--gpu-count 2`, waits for both workers, and then collects metrics.

Each current worker sets `CUDA_VISIBLE_DEVICES` to its GPU index and selects all runs belonging to its assigned entity subset.

The current worker does not distinguish GPU-bound records from CPU-only traditional baselines.

The generator creates THESIS and RedLamp offline configs with `device: cuda`, CANDI and M2N2 online configs with `device: cuda`, and traditional baselines with `device: cpu`.

The training CLI builds the dataset using the configured data settings and passes the configured device to `Trainer`.

The online baseline runner applies the configured device to CANDI and M2N2, calibrates the baseline, selects the configured test range, runs the stream, and writes the report.

## Detailed findings

### Entity selection and run counts

**Implemented:** `discover_remaining_entities` validates the three split directories, checks matching entity files, removes the fixed exclusion list, sorts the result, and returns all remaining entities at `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:36-58`.

**Implemented:** `build_matrix_plan` creates 2 THESIS offline runs, 1 RedLamp offline run, 3 traditional offline runs, 6 THESIS online runs, and 5 online baseline runs per entity-seed at `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:160-226`.

**Configured:** `write_matrix_configs` always calls `discover_remaining_entities(dataset_root)` and therefore writes the full remaining-25 matrix at `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:502-520`.

**Configured:** The generator CLI accepts only `--dataset-root`, `--output-root`, `--smoke`, and `--dry-run` at `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:556-573`.

**Modification candidate:** `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:502-520` and `:556-573` need an explicit entity-selection input for the requested 8 machines.

### Device assignment and GPU count

**Configured:** The matrix launcher defaults to `GPU_COUNT=2` at `scripts/benchmarks/run_remaining_smd_matrix.sh:4-14`.

**Configured:** The launcher rejects every GPU count other than two at `scripts/benchmarks/run_remaining_smd_matrix.sh:33-40`.

**Implemented:** A worker exports `CUDA_VISIBLE_DEVICES` and then executes the assigned run records at `scripts/benchmarks/run_remaining_smd_matrix.sh:121-146`.

**Implemented:** The current worker assignment uses entity index modulo GPU count at `scripts/benchmarks/run_remaining_smd_matrix.sh:133-144`.

**Configured:** The coordinator loops over only GPU indices `0` and `1` at `scripts/benchmarks/run_remaining_smd_matrix.sh:148-165`.

**Modification candidate:** `scripts/benchmarks/run_remaining_smd_matrix.sh:7`, `:33-40`, and `:148-165` need configurable four-GPU worker creation and validation against the cloud device count.

### CPU resources used by GPU sessions

**Configured:** The generated shared data config requests `num_workers: 12` at `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:229-250`.

**Configured:** Smoke RedLamp overrides still request `num_workers: 12` at `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:342-350`.

**Configured:** THESIS online configs request `num_workers: 8` at `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:390-399`.

**Implemented:** The training CLI passes the generated data configuration to `build_dataset` at `scripts/cli/train.py:215-228`.

**Implemented:** The training CLI passes the configured device to `Trainer` at `scripts/cli/train.py:283-317`.

**Inference:** Four GPU sessions with the current worker counts can create more DataLoader subprocesses than the 44-core host can serve predictably.

**Modification candidate:** `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:229-250`, `:342-350`, and `:390-399` need resource-aware worker counts, with the proposed cloud policy using 4 offline workers and 2 online workers per GPU session.

**Modification candidate:** The cloud launcher needs CPU masks and thread environment limits because `CUDA_VISIBLE_DEVICES` does not reserve host CPU cores.

### GPU-capable model paths

**Implemented:** THESIS offline and RedLamp configs use `device: cuda` at `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:274-350`.

**Implemented:** THESIS online configs use `device: cuda` and pass the device through the online task configuration at `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:371-445`.

**Implemented:** CANDI and M2N2 configs use `device: cuda`, while STUMPY, KMeansAD, and Isolation Forest use `device: cpu` at `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:448-488`.

**Implemented:** The online runner overwrites CANDI and M2N2 baseline kwargs with the benchmark device at `scripts/benchmarks/run_online_streaming_benchmark.py:191-200` and calls calibration and stream execution with that device at `:351-371` and `:455-461`.

**Implemented:** Adaptive baselines construct and validate their backbone device at `src/baselines/online/adaptive.py:186-209` and `:335-348`.

**Conclusion:** Current evidence does not require new model-level GPU plumbing for CANDI or M2N2.

### Phase dependencies and resumability

**Implemented:** The matrix launcher executes each entity worker's records in manifest order at `scripts/benchmarks/run_remaining_smd_matrix.sh:127-145`.

**Configured:** THESIS online configs reference the matching Stage B checkpoint and threshold artifact at `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:371-429`.

**Configured:** CANDI and M2N2 configs reference the matching RedLamp checkpoint at `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:448-457`.

**Implemented:** The launcher skips a run when its expected report exists if `--skip-completed` is set at `scripts/benchmarks/run_remaining_smd_matrix.sh:73-82`.

**Gap:** The current launcher has no explicit offline-to-online barrier that verifies all required checkpoint paths before online workers start.

**Modification candidate:** `scripts/benchmarks/run_remaining_smd_matrix.sh:148-165` and the new cloud wrapper need phase barriers and dependency validation.

### tmux lifecycle and logging

**Implemented:** The current launcher creates one detached session named `<session-prefix>-<mode>` at `scripts/benchmarks/run_remaining_smd_matrix.sh:180-193`.

**Gap:** The current launcher does not create separate GPU and CPU sessions, does not pin CPU masks, and does not attach per-worker `tmux pipe-pane` logs.

**Modification candidate:** `scripts/benchmarks/run_remaining_smd_matrix.sh:180-193` or a proposed cloud wrapper must create four GPU sessions and CPU baseline sessions with explicit names, masks, and log paths.

### Metric reporting

**Implemented:** The collector maps the exact requested metric names and three FPR budgets at `scripts/benchmarks/collect_remaining_smd_metrics.py:10-17` and `:37-53`.

**Implemented:** The collector reads each manifest record's report path and produces compact JSON and Markdown output at `scripts/benchmarks/collect_remaining_smd_metrics.py:63-95` and `:98-130`.

**Implemented:** The online baseline runner writes `final_metrics` from the last metric-history record at `scripts/benchmarks/run_online_streaming_benchmark.py:470-504`.

**Implemented:** The THESIS online runner also writes `final_metrics` from the last metric-history record at `scripts/benchmarks/run_thesis_online_benchmark.py:131-151`.

**Unverified:** Existing tests prove that the collector can read a synthetic nested `final_metrics` payload at `tests/benchmarks/test_remaining_smd_benchmark_matrix.py:110-128`, but they do not prove that a live online history record contains all requested final metrics.

**Modification candidate:** `scripts/benchmarks/run_online_streaming_benchmark.py:392-496` needs a focused live-like contract test and may need a final-metric aggregation change if the runtime record lacks the requested VUS fields.

### Existing tests

**Tested:** The current matrix tests cover exclusions, epoch settings, short-range selection, run counts for one entity-seed, metric extraction, online checkpoint paths, and device policy at `tests/benchmarks/test_remaining_smd_benchmark_matrix.py:28-204`.

**Tested:** The current smoke launcher test expects only GPU 0 and GPU 1 at `tests/benchmarks/test_remaining_smd_benchmark_matrix.py:131-150`.

**Modification candidate:** The test suite needs four-GPU dry-run coverage, exact 8-entity selection, resource-class partitioning, CPU masks, worker-count limits, barriers, failure propagation, and live-like online metric completeness.

## Evidence

- `scripts/benchmarks/run_remaining_smd_matrix.sh:7` — the launcher defaults to two GPUs.
- `scripts/benchmarks/run_remaining_smd_matrix.sh:33-40` — the launcher rejects GPU counts other than two.
- `scripts/benchmarks/run_remaining_smd_matrix.sh:121-146` — workers set `CUDA_VISIBLE_DEVICES` and execute manifest records.
- `scripts/benchmarks/run_remaining_smd_matrix.sh:148-165` — the coordinator creates only GPU workers 0 and 1.
- `scripts/benchmarks/run_remaining_smd_matrix.sh:180-193` — the launcher creates one detached `tmux` session.
- `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:36-58` — the generator discovers and excludes entities.
- `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:160-226` — the generator defines the offline and online run matrix.
- `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:229-250` — the shared data config sets the offline DataLoader worker count.
- `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:371-488` — online configs define checkpoint paths, ranges, devices, and baseline methods.
- `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:502-573` — the generator always writes the full discovered matrix and exposes no entity-selection CLI.
- `scripts/cli/train.py:215-228` — training builds the dataset from configured data settings.
- `scripts/cli/train.py:283-317` — training passes the configured device to `Trainer`.
- `scripts/benchmarks/run_online_streaming_benchmark.py:351-371` — online baselines receive the configured device during construction and calibration.
- `scripts/benchmarks/run_online_streaming_benchmark.py:455-496` — online execution and final report fields are assembled.
- `scripts/benchmarks/run_thesis_online_benchmark.py:131-151` — THESIS online compact execution copies the last history record into `final_metrics`.
- `src/baselines/online/adaptive.py:186-209` — adaptive baselines place the backbone on the requested device.
- `scripts/benchmarks/collect_remaining_smd_metrics.py:10-17` — the collector defines the requested metric contract.
- `tests/benchmarks/test_remaining_smd_benchmark_matrix.py:131-150` — the current launcher test assumes two GPUs.
- `documents/logs/09-09-2026/research/design-remaining-smd-benchmark-scripts.md:37-43` — the approved prior design documents two GPU workers and entity-level ordering.
- `documents/notes/smd-28-machine-drift-zoo.md:1-46` — the first 11 ranked machines and drift scores are listed.

## Configuration observed

| Setting | Active value | Evidence | Scope |
| --- | --- | --- | --- |
| GPU worker count | `2` | `scripts/benchmarks/run_remaining_smd_matrix.sh:7,37-40` | Current matrix launcher |
| Coordinator GPU loop | `0 1` | `scripts/benchmarks/run_remaining_smd_matrix.sh:151-155` | Current coordinator |
| Offline DataLoader workers | `12` | `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:239,345` | Generated training configs |
| Online DataLoader workers | `8` | `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:398` | Generated THESIS online configs |
| THESIS and RedLamp device | `cuda` | `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:310,384` | Generated neural configs |
| CANDI and M2N2 device | `cuda` | `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:468` | Generated adaptive baseline configs |
| Traditional baseline device | `cpu` | `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:468` | Generated traditional baseline configs |
| Online range length | `2048` | `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:32,536-540` | Generated online matrix |
| Requested budgets | `0.1%`, `0.5%`, `1%` | `scripts/benchmarks/collect_remaining_smd_metrics.py:10` | Compact report |

## Conflicts and uncertainties

The approved prior design says two GPU workers, while the current request requires four Tesla V100 workers.

The current code assigns runs by entity, so a resource-aware scheduler must preserve output ownership while separating GPU and CPU records.

The exact cloud driver, CUDA version, V100 memory size, CPU topology, and current job occupancy are not established by the local checkout.

The phrase “44 core GPUs” is treated as 44 CPU or vCPU cores because Tesla V100 devices expose thousands of GPU cores; the remote preflight must confirm this interpretation.

The current online report path has no live-run evidence that `final_metrics` contains all requested metric fields.

## Open questions

The repository does not establish whether `taskset` is installed on the cloud host.

The repository does not establish whether the remote host permits all four GPUs and all 44 CPU cores to be pinned by one user.

The repository does not establish whether the cloud host uses W&B network access during wet runs.
