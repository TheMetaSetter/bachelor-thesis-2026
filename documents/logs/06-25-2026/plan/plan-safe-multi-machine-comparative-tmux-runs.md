# Safe Multi-Machine Comparative TMUX Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the comparative and three-stage experiment launch paths so that sequential `train + val + test` runs can be executed safely across one rented 3-GPU machine plus one separate 1-GPU machine, with deterministic GPU pinning, shard-safe artifacts, operational smoke validation, and coarse-grained recovery support before any rental begins.

**Architecture:** Keep the existing model files, loss semantics, metric semantics, and dataset contracts unchanged. Concentrate safety work in orchestration scripts, preflight validation, small generated-config utilities, and coarse-grained execution recovery. Reuse the already safer three-stage launch patterns where possible, and avoid introducing broad abstractions or hidden control paths.

**Tech Stack:** Bash, Python 3, tmux, PyTorch, YAML, Weights & Biases, JSON execution reports.

---

## Current State

- The comparative SMD launch path is split between `scripts/launch_tmux_comparative_smd_experiment.sh` and `scripts/run_comparative_smd_experiments.py`.
- The three-stage launch path already supports `--gpu-index`, GPU-name validation, and `CUDA_VISIBLE_DEVICES` masking through `scripts/launch_tmux_three_stage_experiment.sh` and `scripts/preflight_three_stage_server.py`.
- The comparative main configs already isolate `output_dir` and `checkpoint_dir` by method, entity, and seed, which is a good base for sharding.
- The comparative main SMD configs currently use `num_workers: 16`, which is acceptable for a single job on a single host but risky for three concurrent jobs on a 32-core rented machine.
- The current comparative smoke configs are CPU-only, use `num_workers: 0`, and disable Weights & Biases. They validate functional correctness but do not validate rented-host operational risk.
- The current runners write execution reports, but neither comparative nor three-stage orchestration currently resumes from those reports.
- The core training path saves rich checkpoints, but `scripts/train.py` currently supports only initialization-style warm starts, not true interrupted-run resume.

## Established Contracts to Preserve

- **Batch contract:** The data path continues to produce the current dictionary-style window batch contract centered on `x: Tensor[B, L, D]`, optional point labels, and metadata.
- **Encoder and model contracts:** The safety patch set must not alter `src/models/redlamp_mlp_baseline.py`, `src/models/thesis_multitask.py`, or their output semantics.
- **Experiment-config contract:** The current config-driven registry and `load_experiment_config(...)` flow remain the single source of runtime assembly.
- **Artifact contract:** Each experiment still writes into its own `output_dir` and `checkpoint_dir`; no run should share those paths with another run.
- **Fairness contract:** The safety patch set must not change optimization objectives, metric definitions, synthetic label semantics, or batch size unless the change is explicitly scoped as smoke-only or host-level orchestration-only.

## Design Options

### Option A: Launcher-Only Hardening

- Add GPU pinning and shard-safe path handling to the comparative tmux launcher.
- Keep the comparative Python runner mostly unchanged.
- Use manual host-specific config lists and manual restart logic.

**Strengths:** Lowest implementation risk, fastest to land.  
**Weaknesses:** Still fragile if a rented machine dies mid-sweep; still easy for the operator to make sharding mistakes.

### Option B: Orchestration Hardening Plus Coarse Resume

- Add comparative GPU pinning.
- Add a comparative preflight path modeled after the three-stage preflight path.
- Add host-level worker overrides through generated temporary configs.
- Add execution-report-based skip logic for already completed runs and stages.
- Add a GPU-based operational stress-smoke path.

**Strengths:** Best balance between safety, implementation scope, and time-to-readiness before renting.  
**Weaknesses:** Does not recover an interrupted training run from its last optimizer step; only resumes at run or stage boundaries.

### Option C: Full Resume-Oriented Reliability Patch

- Implement all of Option B.
- Add true offline training resume from checkpoints, including optimizer, scheduler, and epoch continuation.

**Strengths:** Strongest protection against rented-machine interruption.  
**Weaknesses:** Touches core training semantics, requires more testing, and introduces higher pre-rental implementation risk.

## Selected Approach

**Selected approach:** **Option B** before renting, with **Option C explicitly deferred** unless the rented platform proves unusually unstable or unless there is remaining time after the orchestration hardening is verified.

This approach aligns best with the repository’s readability-first and least-codepath principles:

- it keeps safety work in scripts and orchestration layers;
- it reuses patterns already present in the safer three-stage path;
- it does not perturb model semantics or fairness-critical experiment definitions;
- it materially reduces the risk of wasting rented GPU hours.

## File Structure and Responsibility Map

### Existing files to modify

- `scripts/launch_tmux_comparative_smd_experiment.sh`
  - Add GPU pinning arguments and shard-safe launch behavior.
- `scripts/run_comparative_smd_experiments.py`
  - Add generated override-config support, optional skip-completed behavior, and stronger execution-report semantics.
- `scripts/run_three_stage_offline_pretraining.py`
  - Add stage-level skip-completed behavior from the existing execution report.
- `src/core/console.py`
  - Only if needed for concise orchestration summaries during smoke and full launch. This is not part of the first safety-critical patch batch.

### New files to create

- `scripts/preflight_comparative_smd_server.py`
  - Comparative analog of the existing three-stage server preflight.
- `configs/experiment/comparative_stress_smoke/...`
  - A small GPU-based stress-smoke suite that is representative of the rented-host execution path.
- `tests/test_comparative_preflight.py`
  - Validate the new comparative preflight logic.
- `tests/test_comparative_run_plan.py`
  - Validate generated run records, unique paths, and host-level overrides.
- `tests/test_comparative_resume_skip.py`
  - Validate skip-completed semantics from execution reports.
- `tests/test_three_stage_resume_skip.py`
  - Validate stage-level skip behavior for the three-stage runner.

### Existing files to leave unchanged in the first patch batch

- `src/models/redlamp_mlp_baseline.py`
- `src/models/thesis_multitask.py`
- `src/engine/trainer.py`
- `scripts/train.py`

These files should remain unchanged in the pre-rental safety patch batch unless a later phase explicitly activates true interrupted-run resume.

## Risk and Mitigation

- **Risk:** Three comparative jobs on the rented 3-GPU host bind to the same physical GPU.  
  **Mitigation:** Add comparative `--gpu-index` support and launch-time masking through `CUDA_VISIBLE_DEVICES`, using the same pattern as the three-stage launcher.

- **Risk:** Three concurrent jobs oversubscribe the rented host CPU through dataloader workers.  
  **Mitigation:** Add host-level `data.num_workers` override support through generated temporary experiment configs, and validate intended worker settings in preflight.

- **Risk:** Functional smoke passes, but rented-host operational failures still occur.  
  **Mitigation:** Add a GPU-based stress-smoke suite with nonzero workers and the same launch path shape as the main experiment path.

- **Risk:** The operator accidentally launches overlapping config subsets on different machines.  
  **Mitigation:** Add explicit shard-oriented config-list inputs and fail-fast duplicate-path validation for the supplied subset.

- **Risk:** A rented machine dies after several runs or stages complete.  
  **Mitigation:** Add execution-report-based skip-completed logic for comparative runs and three-stage phases so reruns continue from the next unfinished unit.

- **Risk:** Implementation scope expands into core training refactors before rental.  
  **Mitigation:** Defer true interrupted-run checkpoint resume until after orchestration hardening is landed and validated.

- **Risk:** Console cleanup or progress-bar work delays safety-critical launch hardening.  
  **Mitigation:** Treat console minimalization as a separate patch batch unless it is strictly needed for smoke usability.

## Open Questions

- The exact physical GPU index ordering on the rented hosts is still unknown. This blocks final tmux command synthesis but does not block the orchestration plan.
- The preferred Weights & Biases mode for the GPU stress-smoke phase is not yet locked. `online` best matches real runs, while `offline` reduces network instability as a confounder.
- The exact safe worker count for the rented 3-GPU, 32-core host is still to be confirmed empirically, although the research evidence already indicates that `16` is too aggressive for three concurrent jobs.

## Implementation Plan

### Task 1: Harden the Comparative TMUX Launcher for Multi-GPU Pinning

**Files:**
- Modify: `scripts/launch_tmux_comparative_smd_experiment.sh`
- Reference: `scripts/launch_tmux_three_stage_experiment.sh`

**Objective:** Make the comparative tmux launcher safe for one-process-per-GPU execution on rented hosts.

- [ ] Add `--gpu-index` to the comparative shell launcher so one session can be pinned to one physical GPU.
- [ ] Add `--required-gpu-name-substring` to the comparative shell launcher so the launcher can fail fast on the wrong rented host type.
- [ ] Prefix the launched training process with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu_index>`.
- [ ] Ensure `--dry-run` prints the fully masked command so the operator can audit it before renting.
- [ ] Ensure session name, log path, and report directory are all easy to override per shard and per machine.

**Acceptance criteria:**
- The comparative dry-run output clearly shows GPU masking, report directory, log path, and the exact supplied config subset.
- Launch behavior matches the same GPU-selection semantics already used by the three-stage launcher.

### Task 2: Add a Comparative Server Preflight Path

**Files:**
- Create: `scripts/preflight_comparative_smd_server.py`
- Modify: `scripts/launch_tmux_comparative_smd_experiment.sh`
- Reference: `scripts/preflight_three_stage_server.py`

**Objective:** Provide a fail-fast preflight path for rented comparative runs before any tmux session is launched.

- [ ] Validate all supplied config paths and resolve them exactly once.
- [ ] Validate dataset roots for the provided shard.
- [ ] Validate duplicate `output_dir` and `checkpoint_dir` collisions within the shard.
- [ ] Validate that all supplied comparative configs use `device: cuda` for GPU launch mode.
- [ ] Validate that the selected physical GPU index exists and that its name contains the required substring.
- [ ] Write a durable preflight summary JSON artifact under the comparative report directory.
- [ ] Add `--require-launch-ready` semantics so the shell launcher exits nonzero if the comparative shard is not ready.

**Acceptance criteria:**
- A wrong GPU index, wrong GPU name, missing dataset root, or duplicate artifact path causes preflight failure before any training process starts.
- The comparative preflight summary is durable and easy to inspect after dry-run or failure.

### Task 3: Add Shard-Safe Config Input and Host-Level Worker Overrides

**Files:**
- Modify: `scripts/run_comparative_smd_experiments.py`
- Modify: `scripts/launch_tmux_comparative_smd_experiment.sh`
- Reference: `src/core/config.py`

**Objective:** Reduce copy-paste risk across two machines and allow safe worker settings without duplicating many experiment YAML files.

- [ ] Add a shard-friendly input mechanism so the comparative launcher can run a caller-supplied subset of config paths rather than only the hard-coded full list.
- [ ] Add a `--data-num-workers-override` path in the comparative runner.
- [ ] Implement the override through generated temporary experiment configs written under the comparative report directory, preserving the original experiment YAMLs untouched.
- [ ] Reuse the existing `data_overrides` composition pattern rather than inventing a second config system.
- [ ] Record the generated config paths in the manifest so the exact launched artifacts remain reproducible.

**Acceptance criteria:**
- The same base comparative configs can be launched with different worker counts on different hosts without editing the canonical YAML files.
- The manifest clearly captures the exact generated config paths used at runtime.

### Task 4: Add an Operational Stress-Smoke Suite

**Files:**
- Create: `configs/experiment/comparative_stress_smoke/...`
- Modify: `scripts/launch_tmux_comparative_smd_experiment.sh`
- Modify: `scripts/run_comparative_smd_experiments.py`

**Objective:** Add a small smoke path that pressures the real rented-host failure modes rather than only CPU-side logic.

- [ ] Create a dedicated comparative stress-smoke config family that uses `device: cuda`, a small but nontrivial number of windows, and nonzero workers.
- [ ] Keep these configs tiny enough to finish quickly, but close enough to real execution to exercise GPU pinning, dataloader workers, checkpoint writing, evaluation writing, and tmux orchestration.
- [ ] Separate this suite from the existing CPU functional smoke configs instead of replacing them.
- [ ] Ensure the comparative launcher can run either the functional smoke suite or the stress-smoke suite explicitly.

**Acceptance criteria:**
- The stress-smoke suite can be launched concurrently across multiple GPUs and produces the same style of artifacts as the full main runs.
- It detects launch-time and dataloader-time failures that the CPU smoke suite does not detect.

### Task 5: Add Coarse-Grained Resume Through Skip-Completed Behavior

**Files:**
- Modify: `scripts/run_comparative_smd_experiments.py`
- Modify: `scripts/run_three_stage_offline_pretraining.py`

**Objective:** Prevent already completed runs or stages from being repeated after a rented-machine interruption.

- [ ] Teach the comparative runner to read its existing execution report when an explicit skip-completed mode is enabled.
- [ ] Skip a comparative run if its `run_id` is already listed as completed and its expected completion artifacts still exist.
- [ ] Teach the three-stage runner to skip already completed phase names from its execution report when an explicit skip-completed mode is enabled.
- [ ] Recompute and rewrite execution reports after the resumed run finishes.
- [ ] Keep this feature explicit and opt-in rather than automatic, so debugging remains transparent.

**Acceptance criteria:**
- If a rented machine dies after several full comparative runs finish, rerunning the same shard with skip-completed enabled continues from the next unfinished run.
- If a three-stage run dies after some stages finish, rerunning with skip-completed enabled continues from the next unfinished stage.

### Task 6: Defer True Interrupted-Run Training Resume

**Files:**
- Defer for later evaluation: `scripts/train.py`, `src/engine/trainer.py`

**Objective:** Acknowledge the larger reliability patch without letting it delay the pre-rental safety batch.

- [ ] Do not implement optimizer-state resume in the first orchestration hardening batch.
- [ ] Preserve this as a second-phase task only if the rented platform proves unstable or if the coarse-grained skip-completed behavior is insufficient.

**Acceptance criteria:**
- The first safety patch batch lands without altering the semantics of the main training loop.
- A separate follow-up plan can be written later if full resume becomes necessary.

### Task 7: Validation and Pre-Rental Readiness Gate

**Files:**
- Create: `tests/test_comparative_preflight.py`
- Create: `tests/test_comparative_run_plan.py`
- Create: `tests/test_comparative_resume_skip.py`
- Create: `tests/test_three_stage_resume_skip.py`
- Optionally modify: `tests/test_config_loading.py`

**Objective:** Prove that the new safety behavior is correct before any rental begins.

- [ ] Add tests for comparative preflight success and failure cases.
- [ ] Add tests for generated override-config behavior, especially `data.num_workers` overrides.
- [ ] Add tests for duplicate-path rejection on a provided shard.
- [ ] Add tests for comparative skip-completed logic from an existing execution report.
- [ ] Add tests for three-stage skip-completed logic from an existing stage execution report.
- [ ] Run the existing config-loading tests plus the new comparative safety tests before any rented-host command is prepared.

**Acceptance criteria:**
- The orchestration hardening patch set has targeted automated coverage.
- The patch set can be validated locally before any rented runtime is involved.

## Validation Procedures

### Local validation before renting

1. Run config-loading and new orchestration safety tests locally.
2. Run comparative launcher dry-runs for:
   - one 1-GPU shard;
   - one 3-GPU shard;
   - one intentionally invalid GPU or path scenario.
3. Confirm that manifest, report, and preflight artifact paths are disjoint and deterministic.

### Rented-host validation immediately after provisioning

1. Collect `nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total --format=csv,noheader`.
2. Run comparative preflight on each shard with the exact intended `--gpu-index`.
3. Run the new stress-smoke suite concurrently on the intended host topology.
4. Confirm:
   - each session binds to a distinct GPU;
   - no artifact collisions occur;
   - the worker override behaves as expected;
   - checkpoint and evaluation artifacts are written successfully.

### Full-launch readiness gate

The rented-host full experiment should not begin until all of the following are true:

- comparative preflight reports `launch_ready`;
- GPU pinning has been verified from the real host outputs;
- stress-smoke passes on both machines;
- skip-completed behavior has been tested at least once through a simulated interrupted run or a controlled rerun;
- host-specific worker counts have been locked.

## What This Plan Deliberately Does Not Change

- It does not change model architecture, losses, metrics, or fairness-critical training semantics.
- It does not alter the 300-epoch budget or stage definitions.
- It does not merge baseline and thesis code paths more aggressively than they already are.
- It does not make broad refactors to the data loader internals before the orchestration safety layer is landed.

## Final Recommendation Before Renting

Do **not** rent yet. First implement and validate:

1. comparative GPU pinning;
2. comparative GPU-aware preflight;
3. host-level worker override support;
4. GPU-based stress-smoke;
5. skip-completed resume at run and stage level.

Only after these five pieces are validated should the repository be considered operationally ready for the planned two-machine comparative sweep.
