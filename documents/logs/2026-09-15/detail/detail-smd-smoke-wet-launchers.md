---
date: 2026-09-15
status: implementation started; runtime prerequisites block launch
source_structure: ../structure/structure-smd-smoke-wet-launchers.md
---

# Build the two launchers in small steps

## Approved scope and current progress

The user selected the 16 TSAD-compatible baselines already listed by `tsad-lib`: Isolation Forest and the 15 reference anomaly models. THESIS remains a separate method with O0/O1/O2 and A0/A1/A2. This replaces the older RedLamp/CANDI/M2N2/Stumpy/KMeansAD scope below; those missing adapters no longer block this campaign.

Every model uses window size 20, training stride 1, validation stride 20, and offline test stride 20. THESIS online execution keeps stride 1. The new manifest contains 1,596 offline cells and 756 online cells, or 2,352 total. The one-entity, one-seed smoke manifest contains 19 offline cells and nine online cells. These are planned cells, not completed experiments.

Fairness starts with the same source splits, seed list, preprocessing rule, evaluation coverage, and metric definitions. Baselines and THESIS offline results use the same test timeline. Online results belong in a separate table with the same causal range across A variants. Equal epochs do not imply equal compute; record training budgets and elapsed time rather than claim compute parity. Never select checkpoints or thresholds with test labels.

The second implementation batch also passed all 28 pytest contracts on 2026-09-15. The two Bash contract cases passed too. New assertions failed before their corresponding changes. O0/O1 no longer call the two-view loss; O2 still does. The data pipeline now separates training stride from offline evaluation stride. Isolation Forest no longer silently uses a two-point window. CUDA index parsing accepts valid logical indices. Signed finite thresholds remain valid for Isolation Forest. `build_manifest` fixes model names, counts, dependencies, and unique output paths without starting training.

The sibling now has a small `setup.sh`. It installs the local package with reference dependencies, keeps an existing `.venv`, and checks imports. It does not download data, install a CUDA driver, change the system timezone, or start experiments. Its shell boundary is tested with a fake Python executable; package installation was not executed during these tests.

The sibling now contains `matrix_execution.py`, `run_smd_smoke.sh`, and `run_smd_wet.sh`. The coordinator parses non-sequential GPU indices, reserves four GPU workers with six CPU IDs each, two CPU workers with eight IDs each, and four coordinator IDs. It writes its plan atomically and takes one campaign lock. The wet launcher requires completed smoke evidence from the same data root. The reference runner now obeys `batch_size`, seeds PyTorch, and moves its model and batches to its selected device.

This batch still does not finish the experiment launcher. The coordinator currently stops after a safe preflight plan. It does not dispatch model cells, so it cannot claim a smoke or wet experiment has run. Three-A phase reuse, separate online metric artifacts, identity-aware resume checks, bounded logical worker queues, and publication failure tests now have local contract coverage. Actual child-process control and real remote smoke remain unchecked. A1/A2 verification, masked A1 updates, hard-old guards, projector-to-inference wiring, and train-only minibatches now have local contract coverage. The current reference scoring path still needs a common score-space audit before comparisons are called fair. No wet run is authorized by a passing unit suite.

Terminology: `train_stride` is new. The existing `offline_stride` field now means validation and offline test stride; its stored name is kept for compatibility. `online_stride` is unchanged. The manifest names validation and test stride separately and assigns both 20.

The agent follows Design D. Bash starts the experiment inside tmux. A small Python module gives each worker a fixed list. Existing model runners perform the computation. The agent writes each test assertion before changing the behavior it checks.

The read-only remote preflight was attempted on 2026-09-16. The configured
host now presents a different ED25519 key, so strict SSH rejected it before
login. Its current fingerprint is
`SHA256:DwAQKmgn7yQVJlzNJgUkRsRcuhtKq2UQU600G1RT55I`. Remote identity,
dependencies, resources, tmux continuity, smoke artifacts, and wet execution
remain unverified. No remote process or file was changed.

The local Mac has no `nvidia-smi`, so the four-GPU coordinator cannot dispatch
locally. The real local CPU smoke uses the public runner and remains valid.

These steps implement orchestration and its immediate runtime boundaries. They also identify scientific prerequisites that cannot be fixed by shell code. Missing baseline implementations and full online verification need their own source audit before coding; this document does not invent their mathematics. A blocked prerequisite leaves the launch blocked.

## Files and interfaces

Existing sibling files to extend are `src/tsad/matrix.py`, `src/tsad/experiment.py`, `src/tsad/artifacts.py`, `src/tsad/config.py`, `src/tsad/variants.py`, and the existing contract tests. Runtime prerequisites also touch `models/thesis.py` and `models/thesis_online.py`. Read their local instructions before edits.

The sibling now has `scripts/run_smd_smoke.sh`, `scripts/run_smd_wet.sh`, `src/tsad/matrix_execution.py`, and `tests/launchers.test.sh`. The last file has two Bach contract cases, so the test suite remains at 28 pytest cases plus two Bach cases. The Bach test is self-contained. It does not start an experiment.

Proposed interfaces use simple records and functions:

```python
build_manifest(mode, data_root, protocol) -> dict
discover_resources() -> dict
assign_jobs(jobs, resources) -> dict
run_cell(cell, campaign_root, device) -> dict
run_campaign(manifest, campaign_root, resources, resume=False) -> int
```

`matrix.py` owns manifest construction. `matrix_execution.py` owns resource discovery and process scheduling. `experiment.py` owns `run_cell`; it reuses existing stage methods. `artifacts.py` owns completed-stage validation and publication. These are proposed names, not currently callable APIs.

The coordinator module accepts `--mode smoke|wet`, `--data-root PATH`, `--campaign-root PATH`, and `--resume`. The two scripts fix the mode and forward the remaining arguments as an argument array. Reject a forwarded second mode. Default resume is false. No `eval` or shell-generated command fragments are needed for model execution.

One cell contains `id`, `dataset`, `entity`, `seed`, `method`, `phase`, `resource`, `config`, `online_range`, `dependencies`, and `output_path`. THESIS cells also carry `offline_variant` and, for online runs, `online_variant`. Baseline cells carry their method-specific profile instead. A THESIS offline identity excludes A0/A1/A2; a THESIS online identity includes its matching O and A variants. Baseline identities do not carry THESIS O/A ablations.

One resource record contains `gpus: [{index, uuid, name, memory_total_mb, memory_free_mb}]`, `cpu_ids`, `cpu_quota`, `ram_available_bytes`, and `disk_available_bytes`. Worker records contain `name`, `gpu_uuid`, and `cpu_ids`. Keep the public notebook `devices()` output simple; detailed rental inventory belongs here.

## Test budget

Keep the existing 28 pytest tests. Extend the relevant contract test with named scenario rows and a useful assertion message. Add exactly two Bach test functions: one for smoke and one for wet. Their assertion partners do not count as separate cases. Total top-level cases: 30. Also report scenario counts; loops do not make coverage free.

Use `test_smd_matrix_preflight_contract` for counts, resources, and dependency order; `test_output_root_resume_contract` for concurrent starts and interrupted writes; `test_artifact_and_report_contract` for the sole report writer; `test_public_device_contract` for CUDA; `test_o2_config_contract` for all offline loss sets; and the existing A0/A1/A2 contracts for scientific gates. Each scenario is introduced before its corresponding implementation.

For every behavior below: add the named assertion, run the focused test to see the expected failure, make the stated change, then rerun that test. These are separate operations. A stage ends with the complete sibling suite. Do not tick an implementation item from a mocked scheduling result alone.

## Phase 1: every job has one meaning

### Stage 1.1: settle the experiment contract

Tools: Markdown, source inspection, existing configuration test. Outcome: a single explicit protocol.

- [x] Record the user's baseline-scope answer in the plan.
- [x] Record O0 losses as reconstruction and classification.
- [x] Record O1 losses as O0 plus BPSL.
- [x] Record O2 losses as O0 plus the existing two-view contrastive computation.
- [x] Record direct routing for all O variants.
- [x] Record wet Stage A 25 epochs as the proposed explicit generator-derived setting.
- [x] Record wet Stage B 5 epochs as the proposed explicit generator-derived setting.
- [x] Record RedLamp smoke 5 and wet 30 epochs as out of scope for the approved 16-model matrix.
- [x] Resolve training stride against the matrix and current sibling protocol before freezing the manifest; training is 1, validation and offline test are 20, online is 1, and window size is 20.
- [x] Update the stale six-GPU section to the requested four-GPU target.
- [x] Replace stale 504/924 totals with the resolved matrix totals: 1,596 offline, 756 online, and 2,352 total.
- [x] Add an O0 loss-set assertion to `test_o2_config_contract`.
- [x] Run that focused test.
- [x] Change O0's catalog loss tuple.
- [x] Add an O1 loss-set assertion to the same contract.
- [x] Run that focused test.
- [x] Change O1's catalog loss tuple and preserve its BPSL switch.
- [x] Run that focused test.
- [x] Add a contrastive-call-count scenario for O0, O1, and O2 to the existing offline stage contract.
- [x] Run that focused test.
- [x] Gate the Stage A contrastive call on the selected loss tuple.
- [x] Gate the contrastive addition to the total loss on the same condition.
- [x] Run that focused test.
- [x] Run the full sibling suite.

O2's existing contrastive function remains the only implementation. Do not add a second loss because of its alternate name. Stage A currently always adds contrastive loss, so both the catalog and runtime selection must change.

### Stage 1.2: build the manifest

Tools: `matrix.py`, Python, release contract. Outcome: one immutable list before any training.

- [x] Add a named base-matrix count scenario to `test_smd_matrix_preflight_contract`.
- [x] Run that focused test.
- [x] Define explicit baseline names from the resolved scope.
- [x] Expand 28 entities from the train/test/test-label intersection.
- [x] Expand seeds 6, 8, and 36.
- [x] Create the three THESIS offline cells per entity and seed.
- [x] Create the nine THESIS online cells per entity and seed.
- [x] Create the matrix baseline cells using their native phase contracts.
- [x] Assign a unique ID to each cell.
- [x] Attach each online cell's offline dependency ID.
- [x] Assign each cell its resource class.
- [x] Assert unique output paths.
- [x] Assert the resolved phase counts.
- [x] Run that focused test.
- [x] Add a shared-range scenario to the same contract.
- [x] Run that focused test.
- [x] Generate one anomaly-containing wet range of 2,048 points per entity.
- [x] Generate the THESIS smoke range of 4,115 points for the chosen entity.
- [x] Record each range's source-label hash.
- [x] Remove labels from the worker's scoring/adaptation input.
- [x] Run that focused test.
- [x] Run the full sibling suite.

For the original matrix, assert 588 offline, 1,176 online, and 1,764 total. Its smoke subset is 7 offline and 14 online cells for machine-1-3, seed 6. If TSLib is added, preserve this subtotal and show separate extension counts; do not keep the old grand total. Baseline smoke ranges may share the selected THESIS range for coverage, but their training flow does not inherit THESIS epochs or TTA.

### Stage 1.3: check runnable methods

Tools: public, baseline, and THESIS contract tests; source audit. Outcome: honest readiness.

- [x] Add a mocked available `cuda:0` assertion to `test_public_device_contract`.
- [x] Run that test to reproduce CUDA rejection.
- [x] Correct the extra backslash in `validate_device`.
- [x] Run that test.
- [x] Add a missing-matrix-model scenario to the preflight contract.
- [x] Run that test.
- [x] Make preflight reject each missing required runner by name.
- [x] Run that test.
- [x] Record the missing RedLamp adapter as out of scope for the approved 16-model matrix.
- [x] Record the missing CANDI adapter as out of scope for the approved 16-model matrix.
- [x] Record the missing M2N2 adapter as out of scope for the approved 16-model matrix.
- [x] Record the missing Stumpy adapter as out of scope for the approved 16-model matrix.
- [x] Record the missing KMeansAD adapter as out of scope for the approved 16-model matrix.
- [x] Record the missing frozen online Isolation Forest path as out of scope; Isolation Forest has no online cell.
- [x] Add an A1 verification scenario where nonempty tensors are insufficient evidence.
- [x] Run the A1 contract.
- [x] Record its source-audited verification implementation requirement.
- [x] Add an A2 overlapping-interval rejection scenario.
- [x] Run the A2 contract.
- [x] Record its source-audited guard implementation requirement.
- [x] Add a controlled projector-change scenario requiring a change in the next online inference output.
- [x] Run the A2 contract.
- [x] Record the projector-to-inference connection.
- [x] Add a peak-batch-shape assertion to the existing offline stage contract.
- [x] Run that test.
- [x] Record train-only minibatch execution as a resource prerequisite; Stage A and Stage B now use `batch_size`.
- [x] Replace constant-ready gates with checks of the corresponding evidence.
- [x] Run the focused preflight test.

Do not leave an intentionally failing suite as a completed stage. Complete the scientific prerequisite implementation through its own reviewed source-level steps before closing this stage. Its tests must establish prototype verification, verified masks, capacity triggers, hard-old guards, online contrastive semantics, and fresh A-variant state. Do not substitute generic MSE or nonempty checks for those contracts. This is the current largest launch blocker.

## Phase 2: completed stages survive interruption

### Stage 2.1: separate phase ownership

Tools: existing runner methods, Python, THESIS/public contracts. Outcome: one offline training result feeds three fresh online runs.

- [x] Add a call-count scenario requiring one offline fit for three A variants.
- [x] Run the focused public THESIS contract.
- [x] Extract the offline phase from `run_request` into a callable phase boundary.
- [x] Extract the online phase into a callable boundary that consumes an offline result.
- [x] Route `run_cell` to the selected phase.
- [x] Keep the notebook combined call delegating to both boundaries.
- [x] Run the focused test.
- [x] Add a checkpoint-identity mismatch assertion.
- [x] Run the focused test.
- [x] Verify the upstream checkpoint hash before online loading.
- [x] Verify the threshold's checkpoint hash before online loading.
- [x] Load a fresh model state for each A variant.
- [x] Run the focused test.
- [x] Add an online metric scenario with deliberately different offline and online scores.
- [x] Run the focused test.
- [x] Build online point scores using the audited causal score definition.
- [x] Apply coverage filtering to the online timeline.
- [x] Evaluate online labels only after causal scoring finishes.
- [x] Write separate offline and online metric rows.
- [x] Run the focused test.
- [x] Run the full sibling suite.

Online score definition must come from the scientific prerequisite audit. Do not silently label a window mean as a point score. Preserve absolute indices. Emit all five metric families and the three exact FPR budgets. The wet range is 2,048 points, which yields 2,029 complete stride-one windows at length 20; do not confuse that with the 4,096-window smoke contract.

### Stage 2.2: publish complete artifacts

Tools: `artifacts.py`, tempfile, pathlib, hashes, existing release tests. Outcome: partial output never counts as complete.

- [x] Add an interrupted-write scenario to `test_output_root_resume_contract`.
- [x] Run that focused test.
- [x] Create each attempt directory on the same filesystem as its final stage directory.
- [x] Direct the stage writer to that attempt directory.
- [x] Read back every required artifact.
- [x] Compute required artifact hashes.
- [x] Write a completion manifest containing those hashes.
- [x] Flush artifact files before publication.
- [x] Publish the stage with a same-filesystem rename while holding its owner lock.
- [x] Flush the parent directory where supported.
- [x] Run that focused test.
- [x] Add a report-collision scenario to `test_artifact_and_report_contract`.
- [x] Run that test.
- [x] Disable shared aggregate report writes inside matrix workers.
- [x] Make the coordinator write the aggregate report from per-cell results.
- [x] Run that test.
- [x] Run the full sibling suite.

Use `campaign_root/outputs/<experiment_type>/<dataset>/<entity>/<seed>/<method>/<phase>/<variant_stage>/`. A stage manifest contains stage ID, scientific configuration hash, code hash, data hash, upstream hashes, artifact hashes, and completion status. Log GPU UUID separately. Both the initial and best checkpoint remain required where the stage contract calls for them.

### Stage 2.3: resume verified work

Tools: local advisory lock, artifact validator, release test. Outcome: one owner and deterministic restart.

- [x] Add a simultaneous-start scenario to the existing resume contract.
- [x] Run that test.
- [x] Hold a nonblocking campaign lock for the coordinator lifetime.
- [x] Reject a second coordinator for that campaign.
- [x] Run that test.
- [x] Add changed-code, changed-data, changed-threshold, truncated-file, and partial-stage rows to the resume contract.
- [x] Run that test.
- [x] Compare scientific manifest identity before resume.
- [x] Validate all completed-stage hashes before reuse.
- [x] Reuse a matching complete stage without rewriting it.
- [x] Retain an incomplete attempt for inspection.
- [x] Start a fresh attempt from the last valid input/config boundary when resume is explicit.
- [x] Reject a mismatched completed stage.
- [x] Run that test.
- [x] Run the full sibling suite.

Resume preserves an incomplete attempt and starts a fresh attempt from the last valid input/config boundary. It does not reuse today's incomplete `ThesisOnlineState`, which lacks projector and optimizer state. Exact mid-window resume is outside this minimal design. Automatic repeated retries are disabled.

## Phase 3: workers stay inside their resources

### Stage 3.1: discover the rental's resources

Tools: subprocess, `nvidia-smi`, `sched_getaffinity`, cgroup files, release test. Outcome: verified resource IDs.

- [x] Add hardware scenarios with GPU indices 0–3, 3–6, and 2/5/8/11.
- [x] Add a sparse 44-CPU-ID scenario.
- [x] Add fewer-than-four-GPUs and CPU-quota-below-44 scenarios.
- [x] Run the focused preflight contract.
- [x] Parse the live GPU inventory into resource records.
- [x] Restrict selection to the container's allowed GPUs.
- [x] Reject a duplicate GPU UUID.
- [x] Probe each selected UUID in a fresh CUDA child process.
- [x] Verify each child sees one device and its selected UUID.
- [x] Read the CPU affinity list.
- [x] Read the effective CPU quota.
- [x] Reject resources below the requested four-GPU/44-slot plan.
- [x] Read available RAM.
- [x] Read free output disk space.
- [x] Record the resource snapshot in `campaign_plan.json`.
- [x] Run the focused test.
- [x] Run the full sibling suite.

Do not hard-code the supplied UUIDs as an eternal server identity. Keep them as the supplied snapshot. Record actual UUIDs each launch. Unknown memory limits or occupied GPUs block the launch until checked. Determine disk/RAM estimates from smoke measurements before wet execution; do not invent a safety threshold from GPU names.

### Stage 3.2: run fixed queues

Tools: Python subprocesses, affinity, thread environment variables, fake workers. Outcome: bounded concurrency and a real offline barrier.

- [x] Add a disjoint-worker-allocation scenario to the existing preflight contract.
- [x] Run that test.
- [x] Slice four six-ID groups from the allowed CPU list.
- [x] Slice two eight-ID groups from the remaining CPU list.
- [x] Reserve the remaining four IDs for the coordinator.
- [x] Assign sorted GPU jobs round-robin to four lists.
- [x] Assign sorted CPU jobs round-robin to two lists.
- [x] Declare one UUID in each GPU child's `CUDA_VISIBLE_DEVICES` environment record.
- [x] Declare hidden CUDA in each CPU child's environment record.
- [x] Apply the worker CPU affinity before model initialization.
- [x] Apply native thread-pool limits before library initialization.
- [x] Set PyTorch thread count inside each model child.
- [x] Start one model child at a time in each worker.
- [x] Run the focused test.
- [x] Add an offline-child-failure scenario with fake processes.
- [x] Run that test.
- [x] Collect every offline child exit code.
- [x] Block online scheduling if any offline cell fails validation.
- [x] Run that test.
- [x] Add a coordinator-termination scenario with owned fake process groups.
- [x] Run that test.
- [x] Stop dispatching new jobs on SIGTERM or SIGINT.
- [x] Signal only the coordinator's owned child process groups.
- [x] Wait for those child processes.
- [x] Record nonzero exits as failed cells.
- [x] Run that test.
- [x] Run the full sibling suite.

Pass argument lists with `shell=False`. Worker failures do not become success merely because tmux started. A device disappearing or a disk-write failure produces a failed cell. SIGKILL cannot run cleanup code; locked stage ownership and attempt directories must still prevent false completion on restart. Advisory locks only coordinate this launcher, not unrelated tenants; inspect existing GPU processes before starting.

### Stage 3.3: expose the two scripts

Tools: Bash 4.3+, pinned Bach, tmux. Outcome: two minimal user entry points.

- [x] Write the smoke Bach case with the Python launch boundary mocked.
- [x] Add paths-with-spaces to that case.
- [x] Add an existing-session scenario to that case.
- [x] Run that Bach case; the collision was rejected.
- [x] Create `run_smd_smoke.sh` with fixed smoke mode.
- [x] Resolve the script's sibling library root.
- [x] Forward arguments without word splitting.
- [x] Start a named detached server-side tmux session when outside tmux.
- [x] Execute the coordinator directly when already inside tmux.
- [x] Reject a colliding session name.
- [x] Run the smoke Bach case.
- [x] Write the wet Bach case with a failed-smoke-gate scenario.
- [x] Add coordinator-nonzero-exit propagation to that case.
- [x] Run that Bach case; the coordinator boundary was mocked and failures propagated.
- [x] Create `run_smd_wet.sh` with fixed wet mode.
- [x] Use the same coordinator module and argument contract.
- [x] Require matching smoke evidence before dispatching wet jobs.
- [x] Run the wet Bach case.
- [x] Run `bash -n` on both production scripts.
- [x] Run both Bach cases.
- [x] Run the full sibling suite.

The launch boundary is a short named Bash function so Bach can mock it before an absolute `.venv/bin/python` call. Source guards expose that function without starting tmux. Temporary test roots contain any redirects. Use no real SSH, CUDA, tmux, or experiment process in Bach. tmux acceptance is tested separately on the server.

## Phase 4: prove the experiment

### Stage 4.1: stress the boundaries

Tools: existing integration tests and Bach. Outcome: reproducible failure evidence.

- [x] Add kill-before-publication to the existing resume contract.
- [x] Add kill-after-publication to the same contract.
- [x] Add a disk-full write error to the same contract.
- [x] Run that focused test.
- [x] Verify unpublished attempts are never counted as completed.
- [x] Verify published stages remain reusable.
- [x] Run the complete sibling suite.
- [x] Run both Bach cases on supported Bash.
- [x] Collect the pytest test count.
- [x] Count the Bach test functions excluding assertion partners.
- [x] Assert the combined top-level case count is at most 30: pytest collects 29 tests.

Existing verification command, from the sibling root:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests
```

Proposed Bach verification command after the test file exists:

```bash
bash tests/launchers.test.sh
```

### Stage 4.2: run the real smoke

Tools: SSH, tmux, installed sibling Python, artifact verifier. Outcome: real training and evaluation for every selected family.

- [x] Reread `gpu-server.txt` locally without printing its private fields.
- [x] Verify the SSH host identity.
- [x] Inspect the existing remote checkout path.
- [x] Verify its code identity against the approved manifest.
- [x] Verify its installed dependencies.
- [x] Inspect the current GPU inventory.
- [x] Inspect the effective CPU allocation.
- [x] Inspect occupied GPU processes.
- [x] Verify the output filesystem supports the lock/publication contract.
- [ ] Verify the campaign's persistent-storage location.
- [ ] Start the smoke script for machine-1-3 and seed 6.
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
- [ ] Record peak RAM, VRAM, and artifact size from the smoke.
- [ ] Record the final smoke gate status.

An unready baseline stops this procedure at preflight. It must not disappear from the report. The script's launch-success exit only means tmux started; `campaign_status.json` supplies the final experiment status. Preserve that status after the tmux session ends.

### Stage 4.3: launch and audit wet results

Tools: wet script, coordinator, report writer, W&B. Outcome: accountable results for the full resolved manifest.

- [ ] Verify smoke source hashes match the wet source.
- [ ] Verify the approved scientific settings match the wet manifest.
- [ ] Verify free resources cover the measured working-set budget.
- [ ] Verify every required method gate passes.
- [ ] Start the wet script only when wet execution is authorized.
- [ ] Verify the offline barrier result.
- [ ] Verify online completion records.
- [ ] Write the aggregate report once all workers exit.
- [ ] Compare completed logical IDs with the immutable manifest.
- [ ] Report missing or failed IDs explicitly.
- [ ] Verify W&B stage identities without duplicating completed local runs.
- [ ] Record whether the experiment is complete.

Run commands will be documented using the verified remote path after implementation. Do not publish a guessed `/root/tsad-lib` path as an existing checkout. This planning task does not start the 1,764-unit wet experiment.

## Current execution evidence

The SSH host is reachable with the verified ED25519 fingerprint
`SHA256:DwAQKmgn7yQVJlzNJgUkRsRcuhtKq2UQU600G1RT55I`. The remote checkout is
`/workspace/tsad-lib`. It has PyTorch `2.14.0+cu130`, CUDA support, reference
dependencies, and test tools. The sibling suite passes 28 tests locally and
remotely.

The remote host has four RTX 3070 GPUs at indices `0`, `1`, `2`, and `3`, a
44-core effective CPU quota, and no active GPU process during preflight. The
real remote O2-A0 canary used machine-1-3 and seed 6. The preserved smoke roots
contain completed results for all nine thesis O–A pairs and all 16 selected
baselines. Their checkpoint and threshold identities were checked.

The shell coordinator still stops after preflight. It does not dispatch cells.
Persistent storage and peak RAM/VRAM measurements are also unverified. The wet
matrix was not started.

## Exit condition

The launcher implementation is done when its contracts pass. The experiment is ready when every scientific prerequisite, resource check, and real smoke result passes. These are different claims. Keep both visible in the final report.
