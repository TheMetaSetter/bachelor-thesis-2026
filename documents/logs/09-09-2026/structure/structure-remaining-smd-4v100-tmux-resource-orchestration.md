---
date: 2026-09-09 15:06:11 +07:00
topic: "Resource-aware 4-V100 tmux orchestration for 8 SMD machines"
status: approved
revision: 672f9af37a5570f6fb52fe23d704a78beb4eae1e
related_documents:
  - documents/logs/09-09-2026/research/research-remaining-smd-4v100-tmux-resource-orchestration.md
  - documents/logs/09-09-2026/plan/plan-remaining-smd-4v100-tmux-resource-orchestration.md
  - documents/notes/smd-28-machine-drift-zoo.md
---

# Implementation Structure: Resource-aware 4-V100 tmux orchestration for 8 SMD machines

## Summary

The implementation will convert the current two-GPU entity launcher into a resource-aware six-queue workflow.

Four GPU queues will use one Tesla V100 and eight CPU cores per session.

Two CPU queues will use six CPU cores per session for traditional baselines.

The user explicitly requested all four workflow documents in one turn, so this structure is treated as approved for detailed expansion.

## Request

Run the complete offline and online benchmark matrix for `machine-3-1`, `machine-3-5`, `machine-3-2`, `machine-3-11`, `machine-3-10`, `machine-1-3`, `machine-1-1`, and `machine-2-8`.

Use seeds `6`, `8`, and `36`, the existing method variants, the 2048-point anomaly-containing online ranges, and the exact requested metrics.

Use smoke mode before wet mode.

## Confirmed context

- `scripts/benchmarks/run_remaining_smd_matrix.sh:7,37-40,151-155` currently supports only two GPU workers.
- `scripts/benchmarks/run_remaining_smd_matrix.sh:180-193` currently creates one detached `tmux` session.
- `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:502-573` currently generates every non-excluded entity and has no entity-selection CLI.
- `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:229-250,342-350,390-399` currently requests more DataLoader workers than the planned four-GPU CPU allocation can safely assume.
- `scripts/benchmarks/run_online_streaming_benchmark.py:496` currently copies the last online history record into `final_metrics`.
- `scripts/benchmarks/collect_remaining_smd_metrics.py:10-17` defines the exact report metric contract.

## Scope

### In scope

- Explicit selection of the 8 machines.
- Resource-aware manifest records.
- Four GPU worker sessions with CPU affinity.
- Two CPU baseline sessions with CPU affinity.
- Offline-to-online dependency barriers.
- Failure propagation and resumable execution.
- Online final-metric contract validation.
- Smoke, wet, logging, collection, and operational documentation.

### Out of scope

- Model architecture changes.
- New GPU implementations for traditional baselines.
- Changes to seeds, metrics, online ranges, or training budgets.
- Execution on the three excluded machines.
- Replacement of `tmux` with another scheduler.

## Proposed phases

### Phase 1: Establish the cloud resource contract

**Result:** The workflow has one explicit resource contract for 4 GPUs, 44 CPU cores, six worker queues, and the 8-machine benchmark matrix.

**Stages:**

1. Verify the remote checkout and required tools.
2. Verify four V100 devices, 44 CPU cores, memory, disk, and job occupancy.
3. Freeze the selected entity list, 408-run count, CPU masks, and worker counts.

**Depends on:** User-provided cloud resource description.

**Verification:** Read-only remote preflight and generator dry-run.

**Risks:** Incorrect hardware interpretation or active unrelated jobs can invalidate resource placement.

**Complete when:** The host passes preflight and the dry-run manifest has exactly 8 entities and 408 records.

### Phase 2: Make the manifest resource-aware

**Result:** Every run record contains enough information to select its phase, resource class, output path, report path, and checkpoint dependencies.

**Stages:**

1. Add explicit entity selection to the generator.
2. Add GPU/CPU and offline/online classifications.
3. Apply worker-count limits to generated configs.

**Depends on:** Phase 1 entity and resource contract.

**Verification:** Pytest checks for entity validation, counts, classes, dependencies, ranges, and generated YAML values.

**Risks:** Invalid selection or incorrect classification can create incomplete or unfair comparisons.

**Complete when:** The generated manifest and configs represent the exact matrix without modifying method semantics.

### Phase 3: Build resource-specific queues

**Result:** The launcher can execute only the records assigned to one phase and one resource class.

**Stages:**

1. Add phase and resource filters to the matrix launcher.
2. Add deterministic worker assignment and CPU/GPU environment construction.
3. Add four-GPU and two-CPU `tmux` session creation with per-session logs.

**Depends on:** Phase 2 manifest fields.

**Verification:** Shell dry-runs show correct queue membership, `CUDA_VISIBLE_DEVICES`, `taskset`, thread limits, and session names.

**Risks:** CPU oversubscription and accidental GPU sharing can slow or fail runs.

**Complete when:** Dry-run commands consume 32 CPU cores for GPU sessions and 12 CPU cores for CPU sessions.

### Phase 4: Add phase barriers and recovery

**Result:** Online queues start only when all required offline outputs exist, and failed batches can resume safely.

**Stages:**

1. Run offline GPU and CPU queues concurrently.
2. Verify Stage B, RedLamp, threshold, and report dependencies.
3. Run online GPU and CPU queues concurrently with failure propagation and resume support.

**Depends on:** Phase 3 queue execution.

**Verification:** Temporary-manifest tests cover missing files, failed workers, completed reports, and `--skip-completed`.

**Risks:** A false barrier pass can evaluate an absent or wrong checkpoint.

**Complete when:** A missing dependency prevents online launch and a repaired batch resumes without rerunning valid outputs.

### Phase 5: Validate final metrics and compact reports

**Result:** Every valid run produces a report that satisfies the requested metric contract and the collector distinguishes missing runs.

**Stages:**

1. Test live-like online final-metric assembly.
2. Collect compact JSON and Markdown reports.
3. Verify table completeness and exact metric names.

**Depends on:** Phase 4 completed run reports.

**Verification:** Unit tests plus one real cloud smoke report for each GPU baseline family.

**Risks:** A synthetic collector fixture can pass while live online reports lack final VUS metrics.

**Complete when:** Reports contain `VUS-PR@FPR-budget`, `VUS-PR`, `Affiliation F1-score`, `VUS-ROC`, and `raw-FPR` for each completed run.

### Phase 6: Roll out smoke and wet execution

**Result:** The complete 8-machine benchmark runs reproducibly on the cloud host with auditable logs and resumable outputs.

**Stages:**

1. Run one complete smoke combination.
2. Run the complete smoke matrix.
3. Run the wet matrix only after smoke approval.
4. Perform final resource, dependency, and report audit.

**Depends on:** Phases 1–5 passing.

**Verification:** `bash -n`, focused Pytest, cloud smoke, `nvidia-smi`, `tmux`, exact log inspection, and final collector output.

**Risks:** Cloud CUDA, W&B, disk, or job-occupancy differences can break otherwise valid local checks.

**Complete when:** All 408 runs are either validly reported or explicitly marked missing with preserved failure evidence.

## Dependency summary

| Phase | Requires | Enables |
| --- | --- | --- |
| Phase 1 | Cloud resource description | Stable entity and resource contract |
| Phase 2 | Phase 1 contract | Resource-aware manifest and configs |
| Phase 3 | Phase 2 manifest | Executable GPU and CPU queues |
| Phase 4 | Phase 3 queues | Safe online launch and resume |
| Phase 5 | Phase 4 reports | Compact metric tables |
| Phase 6 | Phases 1–5 | Full smoke and wet benchmark |

## Decisions confirmed

- Four GPU sessions reserve 8 CPU cores each for trainer, DataLoader workers, and host-side CUDA work.
- Two CPU sessions reserve 6 CPU cores each for traditional baselines.
- GPU session masks are `0-7`, `8-15`, `16-23`, and `24-31`.
- CPU session masks are `32-37` and `38-43`.
- CANDI, M2N2, THESIS, and RedLamp use GPU queues.
- STUMPY, KMeansAD, and Isolation Forest use CPU queues.
- Offline queues complete before online queues begin.

## Non-blocking uncertainties

- The cloud host's exact CUDA, driver, V100 memory, CPU topology, `taskset`, and W&B availability remain runtime checks.
- The online final-metric assembly remains a smoke-run acceptance check because current tests use a synthetic nested payload.

## Feedback requested

The user requested the full sequence in one turn, so no additional structure approval is required before the detailed document.
