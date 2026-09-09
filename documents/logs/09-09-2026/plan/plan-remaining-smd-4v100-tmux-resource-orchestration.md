---
date: 2026-09-09 15:06:11 +07:00
planner: OpenAI Codex
topic: "Resource-aware 4-V100 tmux orchestration for 8 SMD machines"
status: ready
revision: 672f9af37a5570f6fb52fe23d704a78beb4eae1e
branch: dev
related_research: documents/logs/09-09-2026/research/research-remaining-smd-4v100-tmux-resource-orchestration.md
---

# Implementation Plan: Resource-aware 4-V100 tmux orchestration for 8 SMD machines

## Summary

The implementation will extend the remaining-SMD benchmark launcher so it can select exactly 8 machines, split GPU-bound and CPU-only runs, reserve CPU cores for GPU DataLoader workers, enforce offline-to-online dependencies, and run through resumable `tmux` sessions.

The plan preserves the existing runner commands, output layout, short anomaly-containing online ranges, exact metric names, and two-GPU backward compatibility.

## Request

Run THESIS offline Stage A and Stage B, RedLamp, all traditional baselines, THESIS online variants, CANDI, and M2N2 for the 8 selected machines on four Tesla V100 GPUs and 44 CPU cores.

Use four GPU sessions with one V100 and eight CPU cores per session.

Use two CPU sessions with six CPU cores per session for traditional baselines.

Run smoke mode before wet mode and stop wet execution when smoke or dependency validation fails.

## Current state

The current launcher hard-codes two GPU workers at `scripts/benchmarks/run_remaining_smd_matrix.sh:7,37-40,151-155`.

The current launcher creates one detached `tmux` session at `scripts/benchmarks/run_remaining_smd_matrix.sh:180-193`.

The current generator discovers all 25 non-excluded entities and exposes no explicit entity-selection CLI at `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:502-573`.

The current generator requests 12 offline DataLoader workers and 8 THESIS online DataLoader workers at `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:229-250,342-350,390-399`.

The current model and adaptive-baseline paths already receive CUDA devices at `scripts/cli/train.py:283-317`, `scripts/benchmarks/run_online_streaming_benchmark.py:351-371`, and `src/baselines/online/adaptive.py:186-209`.

## Desired end state

The generator writes a manifest for exactly 8 selected machines and 408 run records.

The manifest labels every run as GPU-bound or CPU-only and as offline or online.

Four GPU workers use `CUDA_VISIBLE_DEVICES=0`, `1`, `2`, and `3`, with CPU masks `0-7`, `8-15`, `16-23`, and `24-31`.

Two CPU workers use CPU masks `32-37` and `38-43`, with CUDA disabled.

Offline queues complete before the online queues start.

Online THESIS runs use matching Stage B checkpoints and thresholds, while CANDI and M2N2 use matching RedLamp checkpoints.

The collector writes only `VUS-PR@FPR-budget`, `VUS-PR`, `Affiliation F1-score`, `VUS-ROC`, and `raw-FPR`.

## Scope

### In scope

- Add explicit selection of the 8 machines.
- Add resource and dependency metadata to the manifest.
- Reduce generated DataLoader worker counts to fit the 44-core host.
- Add four-GPU and CPU queue orchestration.
- Add CPU affinity, thread limits, per-session logs, barriers, failure propagation, and resume behavior.
- Add tests for manifest counts, resource partitioning, session commands, dependencies, and online metric completeness.
- Update CLI and operational documentation.

### Out of scope

- Changing model architecture or training hyperparameters.
- Moving STUMPY, KMeansAD, or Isolation Forest to GPU.
- Changing the online subsequence selection policy.
- Running the benchmark on the three excluded machines.
- Replacing `tmux` with a cluster scheduler.
- Removing existing two-GPU behavior.

## Evidence

- `scripts/benchmarks/run_remaining_smd_matrix.sh:7,37-40,151-155` — current two-GPU restriction.
- `scripts/benchmarks/run_remaining_smd_matrix.sh:180-193` — current single-session `tmux` lifecycle.
- `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:36-58` — current entity discovery and exclusion.
- `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:160-226` — current run matrix.
- `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:229-250,342-350,390-399` — current DataLoader worker counts.
- `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:371-488` — current checkpoint, range, and device configuration.
- `scripts/benchmarks/run_online_streaming_benchmark.py:351-496` — current device and online report flow.
- `scripts/benchmarks/collect_remaining_smd_metrics.py:10-17,37-53,78-130` — current metric collection contract.
- `tests/benchmarks/test_remaining_smd_benchmark_matrix.py:131-204` — current launcher, path, range, and device tests.

## Implementation approach

Use one explicit manifest as the source of truth for entity, seed, method, phase, resource class, output path, report path, and dependency paths.

Use separate queue filters instead of changing individual runner semantics.

Use a cloud wrapper to create four GPU sessions and two CPU sessions, because the current matrix launcher owns a single two-worker coordinator and does not model CPU affinity.

Reserve 32 CPU cores for GPU sessions and 12 CPU cores for traditional baseline sessions.

Add a final-metric contract check before wet execution because the current online report copies the last history record into `final_metrics` at `scripts/benchmarks/run_online_streaming_benchmark.py:496`.

## Phase 1: Freeze resource and benchmark contracts

### Goal

Define the exact 8-machine matrix, resource classes, CPU masks, DataLoader limits, dependencies, and acceptance metrics.

### Changes

Record the 408-run matrix and the `4 × 8 + 2 × 6 = 44` CPU allocation in the manifest and operational documentation.

### Verification

Run the generator in dry-run mode and verify the exact entity list, run count, device classes, and online ranges.

### Risks

An incorrect entity selection or CPU interpretation can invalidate the entire batch, so the cloud preflight must verify both before any training starts.

## Phase 2: Add explicit manifest and configuration support

### Goal

Make the generator produce only the selected 8 machines and attach resource-aware configuration to every run.

### Changes

Extend `generate_remaining_smd_benchmark_configs.py` with explicit entity selection, resource classes, phase groups, dependency paths, and bounded worker counts.

Keep generated devices and existing method settings unchanged except for the required worker-count limits.

### Verification

Use focused Pytest tests to verify entity validation, 408 records, 144 offline records, 264 online records, resource classes, dependency paths, and worker counts.

### Risks

Selecting an invalid or excluded entity could create incomplete or unfair results, so the generator must reject invalid input before writing configs.

## Phase 3: Implement resource-aware queue execution

### Goal

Run each manifest record on the correct GPU or CPU queue with bounded host resources.

### Changes

Extend `run_remaining_smd_matrix.sh` for configurable GPU counts, resource filters, phase filters, deterministic assignment, and CPU affinity.

Add a cloud `tmux` wrapper that starts four GPU sessions, two CPU sessions, per-session logs, and one worker per resource slot.

### Verification

Run dry-run shell tests and inspect the generated commands for `CUDA_VISIBLE_DEVICES`, `taskset`, thread limits, session names, and log paths.

### Risks

Two processes on one GPU or excessive DataLoader threads could cause out-of-memory errors or CPU oversubscription, so the wrapper must validate one-GPU-per-worker and the exact 44-core partition.

## Phase 4: Enforce dependencies and recovery

### Goal

Prevent online runs from starting before their required offline artifacts exist and allow safe resumption after partial failure.

### Changes

Add offline completion checks for THESIS Stage B and RedLamp checkpoints, threshold artifacts, report paths, and worker exit status.

Preserve `--skip-completed` and make failed phase barriers return non-zero without starting the next phase.

### Verification

Test missing-checkpoint, failed-worker, completed-report, and resume scenarios with temporary manifests and exact paths.

### Risks

A false completion signal could evaluate the wrong checkpoint, so dependency checks must validate file existence and the manifest's expected path for the matching entity, seed, and variant.

## Phase 5: Validate metric completeness and reports

### Goal

Produce reliable compact reports for all valid runs with the exact requested metrics.

### Changes

Verify or repair the live-like online final-metric assembly before collection.

Keep the collector's root metric contract and Markdown layout unchanged unless a live-like test proves that the current online report cannot satisfy it.

### Verification

Run unit tests with complete and incomplete online history records and verify that missing metric rows are marked missing rather than silently reported as valid.

### Risks

The current synthetic collector test may pass while a live online report lacks VUS metrics, so the smoke run must inspect one real online report before wet execution.

## Phase 6: Cloud smoke, wet rollout, and audit

### Goal

Run one complete smoke path, then the full smoke matrix, then the wet matrix with reproducible operational evidence.

### Changes

Add operational commands for read-only hardware preflight, smoke launch, session monitoring, wet launch, resume, collection, and final audit.

Update benchmark CLI documentation with the 4-V100 and 44-core allocation.

### Verification

Require zero exit status, correct GPU occupancy, bounded CPU use, complete checkpoints, complete reports, and compact Markdown tables before declaring success.

### Risks

Cloud driver, W&B connectivity, disk space, or GPU occupancy may differ from the local environment, so the rollout must stop on failed preflight or smoke evidence.

## Testing strategy

Use unit tests for entity selection, run counts, resource classification, worker limits, and dependency paths.

Use shell dry-run tests for generated `tmux`, `taskset`, environment, and queue-filter commands.

Use one cloud end-to-end smoke combination to test actual CUDA placement, checkpoint creation, online adaptation, and metric completeness.

Use the complete smoke matrix to test concurrency, barriers, resume behavior, and report collection.

## Migration and rollback

Keep the existing two-GPU default when the cloud wrapper is not used.

Write new outputs under separate `smd_remaining` smoke and wet roots, so the orchestration change does not overwrite historical benchmark outputs.

If a worker fails, stop the dependent phase, preserve exact logs and outputs, repair only the failing code or configuration, and resume with `--skip-completed`.

Do not delete output trees or terminate unrelated remote processes.

## Documentation

Update `documents/logs/09-09-2026/command/remaining-smd-benchmark-cli.md` with selected-machine, four-GPU, CPU-mask, `tmux`, smoke, wet, resume, and collection commands.

Update `documents/logs/09-09-2026/research/design-remaining-smd-benchmark-scripts.md` only if the approved design record must reflect the four-GPU resource model.

## Final verification

- [ ] The selected manifest contains exactly 8 entities and 408 runs.
- [ ] The resource partition consumes exactly 4 GPUs and 44 CPU cores.
- [ ] Offline runs finish before online runs start.
- [ ] Every online dependency points to the matching entity, seed, and variant checkpoint.
- [ ] Reports contain only the requested metrics and distinguish missing rows from valid rows.
- [ ] The full smoke matrix passes before the wet matrix starts.

## Assumptions and non-blocking uncertainties

- The cloud host exposes 44 CPU or vCPU cores rather than 44 GPU cores.
- `taskset` is available on the cloud host; the preflight must verify it.
- W&B access is available for wet THESIS and RedLamp runs.
- One V100 can hold one GPU-bound process with the current model and batch settings; smoke execution must verify memory usage.
