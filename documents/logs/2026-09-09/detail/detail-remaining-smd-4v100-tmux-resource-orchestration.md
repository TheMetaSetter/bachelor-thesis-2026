---
date: 2026-09-09 15:06:11 +07:00
topic: "Resource-aware 4-V100 tmux orchestration for 8 SMD machines"
status: implementation_complete_cloud_pending
revision: 672f9af37a5570f6fb52fe23d704a78beb4eae1e
source_structure: documents/logs/09-09-2026/structure/structure-remaining-smd-4v100-tmux-resource-orchestration.md
related_documents:
  - documents/logs/09-09-2026/research/research-remaining-smd-4v100-tmux-resource-orchestration.md
  - documents/logs/09-09-2026/plan/plan-remaining-smd-4v100-tmux-resource-orchestration.md
  - documents/logs/09-09-2026/research/design-remaining-smd-benchmark-scripts.md
  - documents/notes/smd-28-machine-drift-zoo.md
---

# Detailed Implementation: Resource-aware 4-V100 tmux orchestration for 8 SMD machines

## Summary

The implementation will extend the current remaining-SMD generator and launcher without changing model behavior.

The final workflow will create one manifest for 8 machines and 408 runs, then execute six resource queues through `tmux`.

Four GPU sessions will reserve CPU masks `0-7`, `8-15`, `16-23`, and `24-31`.

Two CPU sessions will reserve CPU masks `32-37` and `38-43`.

## Source structure

The approved structure contains six phases: resource contract, resource-aware manifest, queue execution, dependency recovery, metric reporting, and smoke/wet rollout.

The phase order is preserved because online THESIS, CANDI, and M2N2 runs require offline artifacts.

## Current state

`run_remaining_smd_matrix.sh` rejects GPU counts other than two and launches only GPU workers 0 and 1 at `:7, :33-40, :148-165`.

The same launcher creates one detached `tmux` session at `:180-193` and does not create CPU-affinity masks.

`generate_remaining_smd_benchmark_configs.py` discovers all non-excluded entities at `:36-58` and exposes no explicit entity-selection option at `:556-573`.

The generator requests 12 offline DataLoader workers and 8 THESIS online DataLoader workers at `:229-250, :342-350, :390-399`.

The online baseline code already places CANDI and M2N2 on the configured device at `run_online_streaming_benchmark.py:191-200,351-371` and `src/baselines/online/adaptive.py:186-209`.

The online baseline and THESIS online runners currently copy their last metric-history record into `final_metrics` at `run_online_streaming_benchmark.py:496` and `run_thesis_online_benchmark.py:131-151`.

## Desired end state

The generator accepts repeated `--entity-id` values and rejects IDs that are absent or explicitly excluded.

The manifest contains `resource_class`, `phase_group`, and dependency paths for every run.

GPU-bound records include THESIS offline, RedLamp, THESIS online, CANDI, and M2N2.

CPU-only records include STUMPY, KMeansAD, and Isolation Forest.

The launcher can run one filtered queue at a time and the cloud wrapper composes those queues into phase barriers.

Every GPU process sees one logical CUDA device through `CUDA_VISIBLE_DEVICES`.

Every worker is pinned to a non-overlapping CPU mask.

The online report contains a complete final metric dictionary or is explicitly marked incomplete.

## Scope

### In scope

- `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py`.
- `scripts/benchmarks/run_remaining_smd_matrix.sh`.
- Proposed `scripts/benchmarks/run_remaining_smd_cloud_tmux.sh`.
- Proposed `scripts/benchmarks/_remaining_smd_resource_env.sh`.
- `scripts/benchmarks/run_remaining_smd_smoke.sh` and `run_remaining_smd_wet.sh`.
- `scripts/benchmarks/run_online_streaming_benchmark.py` and `run_thesis_online_benchmark.py` if the metric contract check fails.
- Focused benchmark and launcher tests.
- The benchmark CLI documentation.

### Out of scope

- Changes to THESIS, RedLamp, CANDI, or M2N2 model architecture.
- New GPU implementations for traditional baselines.
- Changes to seeds, window size, online range length, FPR budgets, or metric definitions.
- Execution on `machine-1-6`, `machine-3-4`, or `machine-3-9`.
- Remote cleanup outside exact processes and output paths created by this workflow.

## Evidence

- `scripts/benchmarks/run_remaining_smd_matrix.sh:7,33-40,148-165` — two-GPU restriction and coordinator loop.
- `scripts/benchmarks/run_remaining_smd_matrix.sh:121-146` — current worker environment and entity-based assignment.
- `scripts/benchmarks/run_remaining_smd_matrix.sh:180-193` — current single-session `tmux` lifecycle.
- `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:36-58` — current entity discovery.
- `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:160-226` — current matrix construction.
- `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:229-250,342-350,390-399` — current worker-count settings.
- `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:371-488` — current dependency and device settings.
- `scripts/benchmarks/run_online_streaming_benchmark.py:351-496` — current online execution and report assembly.
- `scripts/benchmarks/run_thesis_online_benchmark.py:131-151` — current THESIS online compact report assembly.
- `scripts/benchmarks/collect_remaining_smd_metrics.py:10-17,37-53` — exact metric names and FPR budgets.
- `tests/benchmarks/test_remaining_smd_benchmark_matrix.py:131-204` — current launcher and generated-config tests.

## Phase 1: Establish the cloud resource contract

### Stage 1.1: Verify the remote tools and hardware

**File:** Proposed new operational section in `scripts/benchmarks/run_remaining_smd_cloud_tmux.sh`.

**Current responsibility:** No current remaining-SMD script verifies four GPUs, 44 CPU cores, CPU affinity tools, or existing job occupancy.

**Change:** Add a preflight mode that performs read-only checks before writing configs or starting sessions.

**Inputs:** `--mode smoke|wet`, `--gpu-count 4`, two GPU/CPU mask lists, dataset root, output root, and selected entity IDs.

**Outputs:** A zero exit status only when all hardware, tools, dataset, and selection checks pass.

**Errors:** Return non-zero for missing Python, missing dataset splits, fewer than four GPUs, fewer than 44 CPU cores, missing `tmux`, missing `taskset`, or occupied target GPUs.

**Atomic steps:**

- [ ] Add `--preflight` parsing to the proposed cloud wrapper.
- [ ] Run `nvidia-smi -L` and parse the number of visible GPUs.
- [ ] Run `.venv/bin/python -c` to check `torch.cuda.device_count()` and `torch.cuda.is_available()`.
- [ ] Run `nproc` and compare the result with `44`.
- [ ] Check that `tmux` and `taskset` resolve through `command -v`.
- [ ] Check the exact `train`, `test`, and `test_label` directories.
- [ ] Check the eight requested entity files in each split.
- [ ] Print the selected masks and resource counts without exposing credentials.

**Verification:** Run the preflight on the cloud host and expect four V100 devices, at least 44 CPU cores, and zero unrelated jobs on devices 0–3.

**Complete when:** The wrapper refuses to launch if any required resource check fails.

### Stage 1.2: Freeze the selected matrix

**File:** `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py`, functions `build_matrix_plan` and `mode_settings`.

**Current responsibility:** The generator defines the method, seed, epoch, subsequence, and metric policy for the remaining-SMD matrix.

**Change:** Preserve those policies and calculate the selected matrix as 144 offline runs, 264 online runs, and 408 total runs.

**Inputs:** Eight entity IDs, seeds `6`, `8`, and `36`, and smoke or wet mode.

**Outputs:** A deterministic manifest with one record per method, variant, entity, seed, and phase.

**Atomic steps:**

- [ ] Pass the eight entity IDs to `build_matrix_plan` in a local dry-run check.
- [ ] Count records grouped by `phase`.
- [ ] Count records grouped by `runner`.
- [ ] Assert `144` offline records.
- [ ] Assert `264` online records.
- [ ] Assert `408` total records.
- [ ] Assert every online record retains a 2048-point range.

**Verification:** Run the existing `.venv/bin/python` matrix-count snippet or its equivalent Pytest fixture and expect the exact counts.

**Complete when:** The selected matrix is fixed before resource scheduling is implemented.

## Phase 2: Make the manifest resource-aware

### Stage 2.1: Add explicit entity selection

**File:** `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py`, functions `discover_remaining_entities`, `write_matrix_configs`, and `main`.

**Current responsibility:** `discover_remaining_entities` returns every valid non-excluded entity, and `main` exposes only dataset, output, smoke, and dry-run options.

**Change:** Accept repeated `--entity-id` options and use them as a validated subset of discovered entities.

**Inputs:** Zero or more entity IDs from the CLI.

**Outputs:** A sorted tuple of selected valid entities.

**Errors:** Raise a clear `ValueError` for an empty selection, an excluded ID, or an ID absent from the train/test/test-label intersection.

**Compatibility:** When no `--entity-id` is supplied, preserve the current all-remaining-entities behavior.

**Atomic steps:**

- [ ] Add an optional `selected_entity_ids` argument to `discover_remaining_entities`.
- [ ] Validate the discovered split intersection before applying the selection.
- [ ] Reject every ID in `EXCLUDED_ENTITY_IDS`.
- [ ] Reject every requested ID not present in the discovered intersection.
- [ ] Return selected IDs in sorted order.
- [ ] Add `--entity-id` with `action="append"` to the generator parser.
- [ ] Pass parsed IDs through `main` into dry-run and config-generation paths.

**Test:** Add a Pytest case that selects exactly the eight requested IDs and rejects `machine-1-6`.

**Verification:** Run the focused benchmark test and expect the exact eight-ID tuple.

**Complete when:** The generator can produce the selected matrix without creating configs for other entities.

### Stage 2.2: Add resource and dependency metadata

**File:** `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py`, function `_add_run` and `build_matrix_plan`.

**Current responsibility:** `_add_run` writes run identity, output path, and report path.

**Change:** Add `resource_class`, `phase_group`, and explicit dependency paths to each manifest record.

**Inputs:** Existing `runner`, `method`, `variant`, `entity_id`, `seed`, and phase values.

**Outputs:** Manifest fields with values from the closed sets `gpu|cpu` and `offline|online`.

**Rules:** THESIS offline, RedLamp, THESIS online, CANDI, and M2N2 are GPU records; STUMPY, KMeansAD, and Isolation Forest are CPU records.

**Atomic steps:**

- [ ] Add a local classification helper that maps each runner/method pair to `resource_class`.
- [ ] Set `phase_group` from the existing phase value.
- [ ] Add a Stage B checkpoint dependency to every THESIS online record.
- [ ] Add a threshold-artifact dependency to every THESIS online record.
- [ ] Add a RedLamp best-checkpoint dependency to every CANDI and M2N2 record.
- [ ] Preserve the existing output and report paths.
- [ ] Add tests for all five runner families and all eight online baseline methods.

**Errors:** Reject any unsupported runner/method pair rather than silently assigning a resource class.

**Verification:** Inspect one manifest record from each runner family and expect the correct resource class and dependency paths.

**Complete when:** A worker can filter the manifest without inspecting YAML content or guessing dependencies.

### Stage 2.3: Bound DataLoader workers per resource class

**Files:** `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py`, functions `_data_config`, `_redlamp_config`, `_thesis_online_config`, and `_online_baseline_config`.

**Current responsibility:** The generator writes 12 offline workers, 12 smoke RedLamp workers, and 8 THESIS online workers.

**Change:** Use 4 offline workers for GPU training, 2 workers for GPU online adaptation, and zero DataLoader workers for CPU traditional baseline configs.

**Inputs:** Runner type and smoke or wet mode.

**Outputs:** Generated YAML with resource-specific `num_workers` values.

**Rules:** Keep existing batch sizes, epochs, device values, subsequence ranges, and metric settings unchanged.

**Atomic steps:**

- [ ] Change the shared generated offline data config from `num_workers: 12` to `num_workers: 4`.
- [ ] Change the smoke RedLamp `data_overrides.num_workers` value from `12` to `4`.
- [ ] Change THESIS online `data_overrides.num_workers` from `8` to `2`.
- [ ] Add `data_overrides.num_workers: 0` to traditional offline baseline configs.
- [ ] Add `data_overrides.num_workers: 0` to traditional online baseline configs.
- [ ] Keep CANDI and M2N2 online configs at `num_workers: 2`.
- [ ] Add tests that inspect generated YAML values for every resource class.

**Verification:** Generate one smoke and one wet config for each runner family and expect the resource-specific worker values.

**Complete when:** Four GPU sessions can reserve eight CPU cores each without inheriting the old 12/8 worker counts.

## Phase 3: Build resource-specific queues

### Stage 3.1: Extend the matrix launcher interface

**File:** `scripts/benchmarks/run_remaining_smd_matrix.sh`, argument parser and validation block at `:4-40`.

**Current responsibility:** The launcher parses mode, GPU index, GPU count, paths, tmux behavior, and resume behavior, then rejects non-two-GPU requests.

**Change:** Add phase/resource filters and accept four GPUs when explicitly requested.

**Inputs:** `--mode`, `--role`, `--gpu-index`, `--gpu-count`, `--resource-class`, `--phase-group`, `--cpu-mask`, `--entity-id`, and existing path flags.

**Outputs:** One worker process that runs only the selected queue.

**Compatibility:** Keep the current two-GPU default and existing wrapper behavior when the new cloud wrapper is not used.

**Atomic steps:**

- [ ] Keep `GPU_COUNT=2` as the backward-compatible default.
- [ ] Replace the exact-two validation with integer validation for `1` through the discovered or requested GPU count.
- [ ] Add parser state for `RESOURCE_CLASS`, `PHASE_GROUP`, `CPU_MASK`, and repeated entity IDs.
- [ ] Pass entity-selection values to manifest generation.
- [ ] Pass phase and resource filters to worker execution.
- [ ] Reject a GPU worker without a valid GPU index and CPU mask.
- [ ] Reject a CPU worker with a non-empty CUDA device assignment.

**Test:** Extend the launcher dry-run test to assert GPU indices 0, 1, 2, and 3 and both CPU resource classes.

**Verification:** Run the launcher in dry-run mode with `--gpu-count 4` and expect no two-GPU rejection.

**Complete when:** The launcher accepts the cloud resource contract while preserving the old default.

### Stage 3.2: Filter and assign queue records

**File:** `scripts/benchmarks/run_remaining_smd_matrix.sh`, worker function at `:121-146` and embedded manifest reader at `:133-144`.

**Current responsibility:** The worker selects records by entity modulo GPU count and always exports `CUDA_VISIBLE_DEVICES`.

**Change:** Select records by `phase_group` and `resource_class`, then assign them deterministically to one worker.

**Inputs:** Manifest path, phase group, resource class, worker index, worker count, and optional entity selection.

**Outputs:** A tab-separated stream containing only records owned by the worker.

**Rules:** GPU workers set one CUDA device; CPU workers set an empty CUDA device; every run executes once.

**Atomic steps:**

- [ ] Read `phase_group` and `resource_class` from each manifest record.
- [ ] Discard records that do not match the worker filters.
- [ ] Preserve manifest order for the filtered records.
- [ ] Assign filtered records by `record_index % worker_count`.
- [ ] Export `CUDA_VISIBLE_DEVICES` only for GPU workers.
- [ ] Clear `CUDA_VISIBLE_DEVICES` for CPU workers.
- [ ] Print the worker class, worker index, CPU mask, GPU index, and run ID before execution.

**Errors:** Return non-zero when a run record lacks required identity, output, or report fields.

**Test:** Use a temporary manifest with known resource and phase fields and assert disjoint queue membership and complete record coverage.

**Complete when:** The six queues partition all 408 records without overlap.

### Stage 3.3: Create cloud tmux sessions

**Proposed new file:** `scripts/benchmarks/run_remaining_smd_cloud_tmux.sh`.

**Current responsibility:** No current script composes separate GPU and CPU queues with barriers.

**Change:** Add a cloud-only wrapper that starts six named sessions for each phase and collects results after online completion.

**Interface:**

```text
run_remaining_smd_cloud_tmux.sh
  --mode smoke|wet
  --gpu-count 4
  --entity-id ENTITY_ID ...
  [--dataset-root PATH]
  [--output-root PATH]
  [--skip-completed]
  [--preflight]
```

**Atomic steps:**

- [ ] Parse the mode, GPU count, repeated entity IDs, dataset root, output root, resume flag, and preflight flag.
- [ ] Reject a GPU count other than `4` in the cloud wrapper.
- [ ] Generate the selected manifest before creating sessions.
- [ ] Create exact log directories under `outputs/tmux_logs/remaining_smd/<mode>/`.
- [ ] Create four offline GPU sessions named `smd-offline-gpu-0` through `smd-offline-gpu-3`.
- [ ] Create two offline CPU sessions named `smd-offline-cpu-0` and `smd-offline-cpu-1`.
- [ ] Pass one GPU index and one CPU mask to each GPU session.
- [ ] Pass one CPU mask and no GPU index to each CPU session.
- [ ] Attach one `tmux pipe-pane` log to each session.
- [ ] Write one worker completion marker containing the worker exit status.

**Errors:** Refuse to overwrite an existing session name unless the session is confirmed to be a session created by this run.

**Verification:** Run the wrapper with a dry-run option or a temporary manifest and inspect all six session commands.

**Complete when:** The wrapper can create six non-overlapping resource sessions without starting online work.

### Stage 3.4: Pin CPU cores and thread pools

**Proposed new file:** `scripts/benchmarks/_remaining_smd_resource_env.sh`.

**Current responsibility:** The current launcher sets only `CUDA_VISIBLE_DEVICES` and does not set CPU affinity or thread caps.

**Change:** Provide one small shell function that runs a command under the selected CPU mask and resource environment.

**Interface:**

```text
run_with_remaining_smd_resources RESOURCE_CLASS CPU_MASK GPU_INDEX COMMAND [ARG]...
```

**Rules:** GPU masks are `0-7`, `8-15`, `16-23`, and `24-31`; CPU masks are `32-37` and `38-43`.

**Atomic steps:**

- [ ] Validate that the CPU mask is non-empty and contains only the assigned range.
- [ ] Set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` for GPU workers.
- [ ] Set the same variables to `6` for CPU workers.
- [ ] Set `CUDA_VISIBLE_DEVICES` to the selected GPU for GPU workers.
- [ ] Clear `CUDA_VISIBLE_DEVICES` for CPU workers.
- [ ] Invoke the command through `taskset -c CPU_MASK`.
- [ ] Print the effective environment before the first run in each session.

**Risk:** CPU workers may inherit library thread pools that ignore one environment variable, so the cloud smoke run must inspect process counts and CPU usage.

**Verification:** Run a harmless shell command through each mask and confirm its CPU affinity with `taskset -pc`.

**Complete when:** No queue can run outside its assigned CPU mask or GPU device.

## Phase 4: Enforce dependencies and recovery

### Stage 4.1: Run offline queues concurrently

**Files:** Proposed cloud wrapper and existing `run_remaining_smd_matrix.sh`.

**Current responsibility:** The current coordinator waits for two workers but does not distinguish phases or validate dependencies.

**Change:** Start four offline GPU workers and two offline CPU workers concurrently and wait for six completion markers.

**Atomic steps:**

- [ ] Create the offline GPU sessions before the offline CPU sessions.
- [ ] Record each session's PID or completion-marker path.
- [ ] Poll only the exact six session markers created by the wrapper.
- [ ] Read one exit status from each marker.
- [ ] Set the offline phase status to failed if any marker is non-zero.
- [ ] Do not create online sessions when the offline phase status is failed.

**Verification:** Use a temporary worker command that exits non-zero and verify that no online session is created.

**Complete when:** The offline phase has one unambiguous success or failure status.

### Stage 4.2: Validate checkpoint dependencies

**File:** Proposed dependency-check section in `run_remaining_smd_cloud_tmux.sh` or a small helper owned by the benchmark scripts.

**Current responsibility:** Configs contain dependency paths, but the current coordinator does not validate them before online launch.

**Change:** Validate exact files from the manifest after offline success.

**Atomic steps:**

- [ ] Load all online records from the manifest.
- [ ] Read each THESIS Stage B checkpoint dependency.
- [ ] Read each THESIS threshold-artifact dependency.
- [ ] Read each CANDI RedLamp checkpoint dependency.
- [ ] Confirm each dependency is a regular file.
- [ ] Confirm the dependency path contains the same entity and seed as the online run.
- [ ] Record missing paths in a phase-specific error log.

**Errors:** Return non-zero when any required dependency is missing or mismatched.

**Verification:** Delete or rename one temporary dependency file and expect online launch to be blocked.

**Complete when:** Every online run has validated inputs before the first online session starts.

### Stage 4.3: Run online queues and resume safely

**Files:** Proposed cloud wrapper and existing launcher `--skip-completed` path at `run_remaining_smd_matrix.sh:73-82`.

**Current responsibility:** The launcher skips a run when its report path exists, but the phase coordinator does not combine this with dependency barriers.

**Change:** Start four online GPU sessions and two online CPU sessions after dependency validation and preserve exact report-based resume behavior.

**Atomic steps:**

- [ ] Create online GPU sessions only after the offline barrier passes.
- [ ] Create online CPU sessions at the same online-phase barrier.
- [ ] Pass `--skip-completed` to every worker when the wrapper receives it.
- [ ] Keep one run process per session at a time.
- [ ] Write a completion marker after each worker exits.
- [ ] Return non-zero if any online worker fails.
- [ ] Invoke the collector only when every online worker exits successfully.

**Verification:** Pre-create one exact report path and confirm that only that run is skipped.

**Complete when:** A failed or interrupted online phase can resume without rerunning valid reports.

## Phase 5: Validate final metrics and compact reports

### Stage 5.1: Test online final-metric assembly

**Files:** `scripts/benchmarks/run_online_streaming_benchmark.py:392-496` and `scripts/benchmarks/run_thesis_online_benchmark.py:131-151`.

**Current responsibility:** Both runners store the last history record as `final_metrics`.

**Change:** First test whether live-like history contains the requested complete metric dictionary, then change the assembly only if the test proves it does not.

**Inputs:** A short online stream with records and metric-history entries.

**Outputs:** A final metric dictionary containing `vus_pr_at_fpr_budget`, `vus_pr`, `affiliation_f1`, `vus_roc`, and `fpr`, or an explicit incomplete status.

**Atomic steps:**

- [ ] Inspect `_process_online_window` output shape for one online window.
- [ ] Confirm whether per-step metrics include all requested aggregate metrics.
- [ ] Add a live-like test fixture with the actual runtime key shape.
- [ ] Run the online runner against the fixture without a cloud job.
- [ ] If the requested keys are absent, add one final aggregation step over retained in-memory scores and labels.
- [ ] Keep raw score arrays out of `summary_only` artifacts after aggregation.
- [ ] Make the report status explicit when aggregation cannot produce complete metrics.

**Risk:** Computing an aggregate from incomplete stream records can silently create false metrics, so missing inputs must produce a missing status rather than a numeric default.

**Verification:** The test must fail for an incomplete final history record and pass for a complete final metric dictionary.

**Complete when:** One real cloud smoke report contains all requested final metrics or clearly reports why it cannot.

### Stage 5.2: Preserve the collector contract

**File:** `scripts/benchmarks/collect_remaining_smd_metrics.py`, functions `extract_requested_metrics`, `collect_manifest_metrics`, and `_markdown`.

**Current responsibility:** The collector maps three FPR budgets and five requested metric labels into compact JSON and Markdown.

**Change:** Keep the metric names and table shape stable while ensuring incomplete payloads remain marked missing.

**Atomic steps:**

- [ ] Keep `FPR_BUDGET_LABELS` as `0.001 -> 0.1%`, `0.005 -> 0.5%`, and `0.01 -> 1%`.
- [ ] Keep `REQUESTED_METRICS` unchanged.
- [ ] Pass the complete final metric dictionary from online reports to `extract_requested_metrics`.
- [ ] Preserve `status: missing` when no report or complete candidate exists.
- [ ] Keep summary-only output free of score arrays, predictions, labels, and traces.
- [ ] Run the existing collector tests without changing their expected labels.

**Verification:** Collect one complete payload and one incomplete payload and inspect both status and values.

**Complete when:** The collector emits only the approved metric fields and does not convert missing metrics into valid numeric rows.

### Stage 5.3: Update tests for the report contract

**Files:** `tests/benchmarks/test_remaining_smd_benchmark_matrix.py` and a focused online runner test file.

**Atomic steps:**

- [ ] Keep the existing synthetic collector test for key normalization.
- [ ] Add a complete live-like online payload test.
- [ ] Add an incomplete online payload test.
- [ ] Assert that incomplete payloads are not reported as completed metric rows.
- [ ] Assert all three FPR budget labels are present in a complete row.
- [ ] Assert `raw-FPR` maps from the unadjusted `fpr` field.

**Verification:** Run the focused Pytest selection and expect all report-contract tests to pass.

**Complete when:** A report reader can distinguish valid, incomplete, and missing online results.

## Phase 6: Roll out smoke and wet execution

### Stage 6.1: Run static and focused automated checks

**Files:** All modified Python and shell files plus their focused tests.

**Atomic steps:**

- [ ] Run `bash -n scripts/benchmarks/run_remaining_smd_matrix.sh`.
- [ ] Run `bash -n scripts/benchmarks/run_remaining_smd_cloud_tmux.sh`.
- [ ] Run `.venv/bin/python -m py_compile` on every modified Python file.
- [ ] Run focused generator and collector tests with `pytest`.
- [ ] Run `git diff --check`.
- [ ] Generate a temporary eight-entity manifest and verify `408` records.

**Expected result:** Static checks pass and the temporary manifest has the exact entity and run counts.

**Complete when:** No code or generated-config check fails before cloud execution.

### Stage 6.2: Run one complete smoke combination

**Files:** Cloud-generated smoke configs and exact output paths from the manifest.

**Atomic steps:**

- [ ] Select one entity and seed.
- [ ] Run one THESIS offline O0 configuration on one V100.
- [ ] Verify Stage A uses 3 epochs.
- [ ] Verify Stage B uses 2 epochs.
- [ ] Verify the best Stage B checkpoint exists.
- [ ] Run one THESIS online A0 configuration on the selected 2048-point range.
- [ ] Run one CANDI configuration on CUDA.
- [ ] Run one M2N2 configuration on CUDA.
- [ ] Run one traditional baseline configuration on a CPU mask.
- [ ] Inspect the final reports for all requested metrics.

**Manual verification:** Use `nvidia-smi`, `tmux list-sessions`, exact worker logs, and exact report paths.

**Complete when:** The one-combination flow proves GPU placement, CPU affinity, checkpoint dependencies, online adaptation, and metric completeness.

### Stage 6.3: Run the complete smoke matrix

**Files:** Proposed cloud wrapper, smoke manifest, smoke outputs, and smoke logs.

**Atomic steps:**

- [ ] Start the four offline GPU sessions.
- [ ] Start the two offline CPU sessions.
- [ ] Wait for the offline barrier.
- [ ] Validate all offline dependencies.
- [ ] Start the four online GPU sessions.
- [ ] Start the two online CPU sessions.
- [ ] Wait for all online completion markers.
- [ ] Run the collector.
- [ ] Inspect the Markdown table for all eight entities and three seeds.

**Expected result:** The smoke matrix exits zero and writes compact JSON and Markdown reports.

**Complete when:** No session shares a GPU, no session leaves its CPU mask, and all expected smoke rows are accounted for.

### Stage 6.4: Run the wet matrix and final audit

**Files:** Proposed cloud wrapper, wet manifest, wet outputs, wet logs, and CLI documentation.

**Atomic steps:**

- [ ] Confirm the complete smoke matrix passed.
- [ ] Generate the wet manifest for the same eight entities.
- [ ] Verify Stage A uses 25 epochs.
- [ ] Verify Stage B uses 5 epochs.
- [ ] Verify RedLamp uses 30 epochs.
- [ ] Start the offline queues.
- [ ] Validate all offline dependencies.
- [ ] Start the online queues.
- [ ] Resume missing records only with `--skip-completed` after an interruption.
- [ ] Run the collector after all online workers succeed.
- [ ] Verify all rows use the exact metric names and three FPR budgets.
- [ ] Preserve missing-run and worker-failure logs for audit.

**Risk and recovery:** Stop the dependent phase on any failure, preserve exact logs, repair only the failing path, and resume from exact report paths.

**Complete when:** The wet matrix has 408 accounted-for records and no unverified numeric metric row.

## Interface and data changes

The manifest adds `resource_class`, `phase_group`, and dependency fields without removing existing run identity or output fields.

The generator adds repeated `--entity-id` input while preserving all-remaining behavior when the option is absent.

The cloud wrapper adds `--gpu-count`, entity-selection, preflight, and resume controls without changing individual runner CLIs.

The output report keeps the existing compact metric names and does not retain raw online arrays under `summary_only`.

## Deployment and rollout

Run read-only hardware and environment checks before creating benchmark outputs.

Run one smoke combination before the full smoke matrix.

Run the full smoke matrix before wet training.

Keep the existing two-GPU launcher default available for historical workflows.

Do not terminate unrelated remote jobs or remove broad output directories.

## Documentation changes

- Update `documents/logs/09-09-2026/command/remaining-smd-benchmark-cli.md` with the new entity-selection, four-GPU, CPU-mask, `tmux`, barrier, resume, and collection commands.
- Update `documents/logs/09-09-2026/research/design-remaining-smd-benchmark-scripts.md` if the approved design record must reflect four GPU workers instead of two.
- Keep `VUS-PR@FPR-budget` unchanged in all command and report documentation.

## Final verification

- [ ] The manifest contains exactly 8 entities and 408 runs.
- [ ] The resource plan uses four V100 devices and 44 CPU cores.
- [ ] GPU sessions reserve 32 CPU cores for trainer and DataLoader work.
- [ ] CPU baseline sessions reserve 12 CPU cores.
- [ ] Offline completion is required before online launch.
- [ ] All required dependencies point to matching entity, seed, and variant outputs.
- [ ] Reports contain only the requested metrics and identify incomplete results.
- [ ] The complete smoke matrix passes before wet execution.

## Execution record

The local implementation is complete for entity selection, resource metadata, worker limits, four-GPU and two-CPU queue orchestration, CPU affinity, phase barriers, dependency validation, resume markers, final metric assembly, compact collection, and CLI documentation.

The final focused verification passed `27` benchmark, online-wrapper, and online-engine tests.

Shell syntax checks and Python compilation passed for all modified files.

The eight-machine matrix count is `408` runs with `144` offline runs and `264` online runs.

The full `tests/benchmarks` and `tests/online` selection had `167` passed, `1` skipped, and `2` failures in pre-existing tests outside this change: one found `19` instead of `18` unrelated THESIS configs, and one lacked its fake offline checkpoint.

The cloud smoke and wet runs are pending because the strict SSH probe detected a changed host key for `[159.48.242.10]:20602` and refused the connection.

Do not bypass strict host-key checking or replace the known key without anh confirming the server fingerprint.
