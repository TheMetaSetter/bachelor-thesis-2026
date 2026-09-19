---
date: 2026-09-16
status: execution in progress
source: detail-smd-smoke-wet-launchers.md
---

# Finish the remaining launcher work

The old detail story has 77 unchecked items. This file is the small work list
for them. Each item keeps its original meaning. We execute local items first.
Remote items need a reachable server and real evidence.

## Phase 1: one offline result, three online variants

The runner first trains one offline result. Then it starts fresh A0, A1, and A2
runs from that result. This keeps the three online variants comparable.

### Stage 1.1: phase routing

- [x] Add a call-count scenario requiring one offline fit for three A variants.
- [x] Run the focused public THESIS contract.
- [x] Route `run_cell` to the selected phase.
- [x] Verify the upstream checkpoint hash before online loading.
- [x] Verify the threshold's checkpoint hash before online loading.
- [x] Load a fresh model state for each A variant.
- [x] Run the focused test.

### Stage 1.2: online results

- [x] Add an online metric scenario with deliberately different offline and online scores.
- [x] Run the focused test.
- [x] Build online point scores using the audited causal score definition.
- [x] Apply coverage filtering to the online timeline.
- [x] Evaluate online labels only after causal scoring finishes.
- [x] Write separate offline and online metric rows.
- [x] Run the focused test.
- [x] Run the full sibling suite.

## Phase 2: safe stage reuse

An output is reusable only when its files and identities are complete. A broken
attempt stays visible and never becomes a completed stage.

### Stage 2.1: resume validation

- [x] Add changed-code, changed-data, changed-threshold, truncated-file, and partial-stage rows to the resume contract.
- [x] Run that test.
- [x] Validate all completed-stage hashes before reuse.
- [x] Retain an incomplete attempt for inspection.
- [x] Start a fresh attempt from the last valid input/config boundary when resume is explicit.
- [x] Reject a mismatched completed stage.
- [x] Run that test.
- [x] Run the full sibling suite.

## Phase 3: bounded child workers

The coordinator gives each child a fixed GPU, CPU list, and thread budget. A
worker starts one model child at a time. A failed child fails its cell.

### Stage 3.1: worker setup

- [x] Apply the worker CPU affinity before model initialization.
- [x] Apply native thread-pool limits before library initialization.
- [x] Set PyTorch thread count inside each model child.
- [x] Start one model child at a time in each worker.
- [x] Run the focused test.

### Stage 3.2: worker failure and shutdown

- [x] Add an offline-child-failure scenario with fake processes.
- [x] Run that test.
- [x] Collect every offline child exit code.
- [x] Add a coordinator-termination scenario with owned fake process groups.
- [x] Run that test.
- [x] Stop dispatching new jobs on SIGTERM or SIGINT.
- [x] Signal only the coordinator's owned child process groups.
- [x] Wait for those child processes.
- [x] Record nonzero exits as failed cells.
- [x] Run that test.

## Phase 4: interruption tests

The publication test kills a writer before and after publication. It also
simulates a full disk. Only a fully published stage can be reused.

### Stage 4.1: publication failures

- [x] Add kill-before-publication to the existing resume contract.
- [x] Add kill-after-publication to the same contract.
- [x] Add a disk-full write error to the same contract.
- [x] Run that focused test.
- [x] Verify unpublished attempts are never counted as completed.
- [x] Verify published stages remain reusable.

## Phase 5: remote smoke evidence

The remote story begins with read-only checks. Only a reachable server with
matching code, resources, and dependencies may run the smoke.

### Stage 5.1: remote preflight

- [x] Verify the SSH host identity.
- [x] Inspect the existing remote checkout path.
- [x] Verify its code identity against the approved manifest.
- [x] Verify its installed dependencies.
- [x] Inspect the current GPU inventory.
- [x] Inspect the effective CPU allocation.
- [x] Inspect occupied GPU processes.
- [x] Verify the output filesystem supports the lock/publication contract.
- [x] Verify the campaign's persistent-storage location. `/root/tsad-lib` and `/workspace` use overlay storage without a mounted persistent volume.

### Stage 5.2: remote canary

- [x] Start the smoke script for machine-1-3 and seed 6. The corrected launcher started the real coordinator inside `tsad-smd-smoke`.
- [x] Run a real O2-A0 machine-1-3 canary through the public runner on the remote GPU host.
- [x] Verify O2 Stage A records three epochs (local CPU smoke).
- [x] Verify O2 Stage B records two epochs (local CPU smoke).
- [x] Verify O2-A0 records 4,096 causal windows (local CPU smoke).
- [x] Verify the O2-A0 offline and online metric artifacts separately (local CPU smoke).
- [ ] Permit the coordinator to dispatch the remaining selected smoke cells only after the canary passes.
- [ ] Detach the tmux client during an active cell.
- [ ] Reconnect to that session.
- [ ] Verify the active cell continued.
- [x] Verify every selected O–A pair used its matching offline artifact across the nine thesis smoke pairs.
- [x] Verify every selected baseline has a real result across the preserved smoke roots.
- [x] Record peak RAM, VRAM, and artifact size from the smoke. The resource log recorded peak host RAM used of 6,557,523,968 bytes, GPU memory used of 407, 401, 297, and 449 MiB for physical GPUs 0–3, and a 279,990-byte output tree.
- [x] Record the final smoke gate status. The preserved smoke campaign is `failed`: 13 of 28 cells completed and 15 failed.

## Phase 6: wet launch evidence

The wet run starts only after smoke hashes, settings, resources, and method
gates agree. The coordinator writes one final report from all cell records.

### Stage 6.1: wet preflight

- [ ] Verify smoke source hashes match the wet source.
- [ ] Verify the approved scientific settings match the wet manifest.
- [ ] Verify free resources cover the measured working-set budget.
- [ ] Verify every required method gate passes.
- [ ] Start the wet script only when wet execution is authorized.

### Stage 6.2: wet completion

- [ ] Verify the offline barrier result.
- [ ] Verify online completion records.
- [ ] Write the aggregate report once all workers exit.
- [ ] Compare completed logical IDs with the immutable manifest.
- [ ] Report missing or failed IDs explicitly.
- [ ] Verify W&B stage identities without duplicating completed local runs.
- [ ] Record whether the experiment is complete.

## Execution rule

For each local behavior, write or extend one focused test first, run it, make
the smallest code change, rerun it, and then run the full sibling suite. Mark
the matching item here only after the evidence exists. After all possible items
are complete, copy the checked state back to the source detail document.

## Current block

Explicit resume now preserves an incomplete attempt under a hidden
`.incomplete-*` directory, then starts a fresh run from the valid input and
configuration boundary. It does not reuse incomplete online optimizer state.

The configured SSH host was first accepted with ED25519 fingerprint
`SHA256:DwAQKmgn7yQVJlzNJgUkRsRcuhtKq2UQU600G1RT55I`. The host later
restarted and now presents `SHA256:fxtQf/kQEgGat4GZ9gYIRTWM/yaGm/u04ghi5CAoKZU`.
Strict SSH rejects that changed key until the user verifies it. No new key was
accepted automatically.

The local CPU smoke used real `ServerMachineDataset/machine-1-3` data and
labels. It completed O2-A0 with three Stage A epochs, two Stage B epochs, and
4,096 causal windows. Its artifacts and preflight report are under
`tsad-lib/outputs/local_smoke`. This proves local behavior only; it does not
close the remote canary gate.

The local Mac has no `nvidia-smi`. Therefore the four-GPU coordinator cannot
dispatch locally. Its resource checks remain remote items, not failed local
items.

The remote preflight found four RTX 3070 GPUs at indices `0`, `1`, `2`, and
`3`, an effective CPU set of `11-21,33-65`, a 44-core quota, and no active GPU
processes. An exact temporary-file probe confirmed `fsync` and atomic rename
on `/workspace`. The probe files were removed. `/workspace` is writable but
has no mounted persistent storage, so that gate remains open.

The checkout lives at `/workspace/tsad-lib`. Its virtual environment has
PyTorch `2.14.0+cu130`, CUDA support, the reference dependencies, and the
development test tools. The sibling suite now passes 30 local tests. The new
coordinator uses one child per assigned worker and keeps the offline barrier.
Before the host key changed, a corrected smoke coordinator published 12 cell
completions in tmux after the canary. It did not reach a final status, so the
coordinator, reconnect, peak-resource, and final-smoke items remain open. A
measured wrapper could not run because the old image had no `/usr/bin/time`;
no peak RAM or VRAM claim is recorded.

The local preflight report is under `tsad-lib/outputs/local_preflight`. It
records 28 entities, 1,596 offline units, 756 online units, 2,352 total units,
and ready local scientific gates. It does not replace remote GPU evidence.

## Fresh local and remote story

The local runtime now initializes its offline cache before a worker calls
`run_request()` directly. ETSformer now moves frequency values to the input
device before GPU indexing. The local suite has 30 passing tests. A fresh
local O2-A0 smoke used real `machine-1-3` data, three Stage-A epochs, two
Stage-B epochs, and 4,096 causal windows. All required local artifacts exist.

The server still has `/root/tsad-lib` at commit `9aa4c35`. Its source hashes
do not match the local fixes. Its root filesystem is an overlay without a
mounted persistent volume, and it has no `cc` or `gcc`. It still exposes four
RTX 3070 GPUs at physical indices 0, 1, 2, and 3, with 44 CPUs. No tmux job
is running. No remote source was edited. The remote smoke and wet gates stay
unchecked until the local source is published and the server has persistent
storage and a C compiler.
