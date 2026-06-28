---
date: 2026-06-25 20:03:44 +0700
researcher: TheMetaSetter
git_commit: 89a598f643cf0c20b0ab540b926e6b71f27e975f
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed implementation plan for safe multi-machine comparative tmux runs before renting GPU servers"
tags: [detail, orchestration, tmux, gpu, comparative, safety]
status: complete
last_updated: 2026-06-25
last_updated_by: TheMetaSetter
---

# Detailed Plan: Safe Multi-Machine Comparative TMUX Runs Before Renting

**Date**: 2026-06-25 20:03:44 +0700  
**Researcher**: TheMetaSetter  
**Git Commit**: `89a598f643cf0c20b0ab540b926e6b71f27e975f`  
**Branch**: `dev`

## Objective

Implement the pre-rental safety hardening required for comparative SMD experiments that will be launched across one rented 3-GPU machine and one separate 1-GPU machine. The detailed scope is intentionally limited to orchestration, validation, smoke execution, and coarse-grained recovery. It must not change model semantics, metric semantics, or fairness-critical training settings.

## Governing Constraints

1. Preserve the current batch contract, encoder contract, model output contract, evaluation metric semantics, and 300-epoch thesis-stage semantics.
2. Keep all model logic inside their current model files, in accordance with `codebase_preferences.md`.
3. Prefer composition over inheritance and reuse existing launcher and preflight patterns instead of introducing new framework layers.
4. Keep the number of codepaths low. New behavior must be explicit through CLI flags or small generated config overlays.
5. Do not implement true interrupted-run optimizer-state resume in this batch. That belongs to a later reliability phase if still necessary.

## Contract Lock Before Any Edit

The following interfaces are part of the patch boundary and must remain unchanged:

- **Dataset batch contract**
  - The comparative and three-stage paths continue to consume batches shaped around `x: Tensor[B, L, D]` with optional `point_labels`, `mask`, `timestamps`, and `meta`.
- **Encoder contract**
  - No change to encoder input or output semantics in `src/models/redlamp_mlp_baseline.py` or `src/models/thesis_multitask.py`.
- **Model output contract**
  - No change to training-step, validation-step, realistic-validation-step, or test-step return payloads.
- **Task contract**
  - No change to anomaly-class taxonomy, label refurbishment semantics, `lambda_recon`, `lambda_cls`, or stage semantics.
- **Training engine contract**
  - `scripts/train.py`, `src/engine/trainer.py`, and `scripts/evaluate.py` continue to accept resolved experiment configs and produce the current checkpoint and evaluation artifacts.

This detailed patch is therefore an **orchestration safety layer** around the current training and evaluation graph, not a change to the graph itself.

## Design Pattern Application

- **Composition over inheritance**
  - New safety behavior is assembled by composing existing runners, preflight logic, and generated config overlays. No new class hierarchy is introduced.
- **Adapter pattern**
  - The new comparative preflight script acts as an adapter that translates comparative-shard intent into the same style of GPU and readiness validation already used in the three-stage path.
- **Strategy pattern**
  - Smoke mode selection, skip-completed behavior, and host-specific worker settings are expressed as explicit strategy flags or explicit config subsets rather than hidden branching.
- **Registry and factory preservation**
  - The current registry-based dataset/model build path remains untouched. All new orchestration logic continues to route through `load_experiment_config(...)`, `scripts/train.py`, and `scripts/evaluate.py`.

## Phased Detailed Plan

### Phase 0: Scope Guard and Non-Regression Boundary

**Phase summary tied to the thesis objectives:**  
The thesis objective is fair, reproducible comparison between baseline and main method. Before hardening multi-machine execution, the codebase must explicitly lock the fairness-sensitive surfaces so the safety patch cannot accidentally perturb training or evaluation semantics.

**Files:**
- Modify: `documents/logs/06-25-2026/detail/detail-safe-multi-machine-comparative-tmux-runs.md`
- Reference only: `src/models/redlamp_mlp_baseline.py`
- Reference only: `src/models/thesis_multitask.py`
- Reference only: `scripts/train.py`
- Reference only: `scripts/evaluate.py`

**Edits within this phase:**
- Record the non-edit boundary directly in the detail note and later in commit messages.
- Treat all model files and evaluation metrics as frozen surfaces for this patch batch.
- Treat `device: cuda` in experiment configs as canonical; GPU selection will occur through environment masking only.

**Explicit edit content:**
- No code changes are made in this phase.
- This phase exists to ensure that subsequent phases do not drift into model refactoring or metric redesign.

**Test plan and validation steps:**
- Reuse existing config-loading tests after the full patch batch to ensure no accidental config semantic drift.

**Acceptance criteria:**
- The implementation branch does not modify:
  - `src/models/redlamp_mlp_baseline.py`
  - `src/models/thesis_multitask.py`
  - metric computations under `src/metrics/`
  - evaluation semantics in `scripts/evaluate.py`

### Phase 1: Comparative GPU Pinning and Shard-Safe TMUX Launch

**Phase summary tied to the thesis objectives:**  
The comparative experiments are only fair if each run executes deterministically on the intended hardware without accidental cross-GPU collision. This phase brings the comparative launcher up to the same operational standard already used by the three-stage launcher.

**Files:**
- Modify: `scripts/launch_tmux_comparative_smd_experiment.sh`
- Reference: `scripts/launch_tmux_three_stage_experiment.sh`

**Edits within this phase:**
- Add `--gpu-index` support.
- Add `--required-gpu-name-substring` support.
- Add explicit `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- Add explicit `CUDA_VISIBLE_DEVICES=<gpu_index>`.
- Ensure dry-run output prints the exact masked command.
- Ensure `session_name`, `report_dir`, and `log_path` remain caller-overridable and visible in dry-run output.

**Explicit edit content:**
- Extend shell argument parsing with:
  - `--gpu-index`
  - `--required-gpu-name-substring`
- Build the tmux inner command so that the Python runner is invoked under masked GPU visibility.
- Ensure the displayed command string in `--dry-run` mode includes the same environment prefix that will be used in live launch.

**Interface and contract definitions:**
- Input interface:
  - comparative launcher arguments now include GPU-selection metadata.
- Output interface:
  - no change to experiment outputs;
  - added clarity in launcher stdout about which physical GPU is being targeted.

**Risk mitigation steps:**
- Prevent accidental multi-process binding onto GPU 0.
- Prevent silent use of the wrong rented host type.

**Test plan and validation steps:**
- Dry-run with:
  - one valid GPU index;
  - one intentionally invalid GPU index;
  - one custom session name and report directory.
- Verify that printed launch commands are unambiguous.

**Acceptance criteria:**
- Comparative dry-run shows:
  - selected GPU index,
  - masked command,
  - session name,
  - log path,
  - report directory.
- Live launch path uses the same GPU masking style as the three-stage launcher.

### Phase 2: Comparative GPU-Aware Preflight

**Phase summary tied to the thesis objectives:**  
Before any rented machine is paid for, the comparative path needs a preflight layer that can fail fast on misconfigured shards, missing data roots, artifact collisions, or mismatched GPUs.

**Files:**
- Create: `scripts/preflight_comparative_smd_server.py`
- Modify: `scripts/launch_tmux_comparative_smd_experiment.sh`
- Modify: `scripts/run_comparative_smd_experiments.py`
- Reference: `scripts/preflight_three_stage_server.py`

**Edits within this phase:**
- Create a dedicated comparative preflight script.
- Reuse current comparative path validation:
  - config resolution,
  - unique artifact-path checks,
  - dataset-root checks.
- Add GPU checks:
  - target GPU index existence,
  - target GPU name substring,
  - `device: cuda` expectation across supplied configs.
- Write a durable comparative preflight summary JSON into the comparative report directory.
- Allow the tmux launcher to require launch readiness before opening a detached session.

**Explicit edit content:**
- The new preflight script should accept:
  - `--config-paths`
  - `--report-dir`
  - `--gpu-index`
  - `--required-gpu-name-substring`
  - `--print-json`
  - `--require-launch-ready`
- The summary JSON should record:
  - supplied config paths,
  - experiment names,
  - dataset roots,
  - output directories,
  - checkpoint directories,
  - chosen GPU index,
  - GPU validation result,
  - readiness status.

**Interface and contract definitions:**
- Input contract:
  - preflight receives a comparative shard, not the full universe of experiments.
- Output contract:
  - a single JSON summary artifact plus exit code semantics suitable for launch gating.

**Design pattern application:**
- Adapter pattern: this script adapts comparative-run semantics into the same readiness-check style already used by the three-stage path.

**Risk mitigation steps:**
- Artifact collision risk is blocked before any run starts.
- Wrong-machine or wrong-GPU launches fail fast.
- Manual copy-paste errors across shards become more visible.

**Test plan and validation steps:**
- Unit tests for:
  - valid shard,
  - duplicate output path in shard,
  - missing dataset root,
  - non-`cuda` device in a supposedly GPU shard,
  - invalid GPU index.

**Acceptance criteria:**
- Comparative launch can be made conditional on explicit `launch_ready`.
- Failure conditions are durable and inspectable through a JSON summary, not only shell stderr.

### Phase 3: Host-Level Worker Overrides Through Generated Overlay Configs

**Phase summary tied to the thesis objectives:**  
Fair comparative training must stay configuration-driven, but rented hosts require different worker settings from local or single-GPU hosts. This phase introduces a host-level override without duplicating the canonical experiment YAML set.

**Files:**
- Modify: `scripts/run_comparative_smd_experiments.py`
- Modify: `scripts/launch_tmux_comparative_smd_experiment.sh`
- Reference: `src/core/config.py`
- Reference: `src/data/loaders.py`

**Edits within this phase:**
- Add a launcher flag for worker override, for example `--data-num-workers-override`.
- Apply the override by generating temporary resolved experiment configs under the comparative report directory rather than mutating canonical YAMLs.
- Use the existing experiment-config layering style so that the override is explicit and reproducible.
- Record generated config paths in the comparative manifest.

**Explicit edit content:**
- Generated overlay configs must preserve:
  - original experiment name,
  - original seed,
  - original output and checkpoint directories,
  - original model and task definitions.
- The only intended data change in this phase is `data.num_workers`, and optionally `data.min_num_workers` if needed later.

**Interface and contract definitions:**
- Dataset contract remains identical.
- Training engine contract remains identical.
- The only changed interface is orchestration-time config materialization.

**Design pattern application:**
- Composition over inheritance: worker policy is layered through generated configs, not through subclassed loaders or special-case trainer branches.

**Risk mitigation steps:**
- This phase reduces CPU oversubscription risk on the rented 3-GPU host.
- It also prevents ad hoc manual edits to canonical experiment YAMLs, which would harm reproducibility.

**Test plan and validation steps:**
- Validate that generated configs:
  - carry the overridden worker value,
  - preserve all other key fields,
  - are written to a deterministic report-local location.

**Acceptance criteria:**
- One shard can be launched with worker count `4` on the 3-GPU rented host while another shard can use worker count `8` on the 1-GPU host, without cloning or editing canonical experiment configs.

### Phase 4: Comparative GPU Stress-Smoke Suite

**Phase summary tied to the thesis objectives:**  
The current CPU functional smoke suite is insufficient for rented-host safety. A separate GPU stress-smoke suite is required to validate the exact orchestration path that will later run the real comparative sweep.

**Files:**
- Create: `configs/experiment/comparative_stress_smoke/...`
- Modify: `scripts/launch_tmux_comparative_smd_experiment.sh`
- Modify: `scripts/run_comparative_smd_experiments.py`
- Reference: existing comparative smoke configs

**Edits within this phase:**
- Keep the current CPU smoke configs untouched as functional smoke.
- Create a second smoke family for operational validation.
- Use:
  - `device: cuda`
  - nonzero workers
  - small capped window counts
  - short epoch budget
  - standard checkpoint/evaluation path
- Provide explicit launcher support to choose:
  - functional smoke,
  - stress smoke,
  - or no smoke.

**Explicit edit content:**
- Stress-smoke configs should be small enough to finish quickly on a rented host.
- They must still exercise:
  - GPU pinning,
  - dataloader workers,
  - checkpoint writes,
  - evaluation artifact writes,
  - W&B mode chosen for launch realism.

**Interface and contract definitions:**
- The smoke configs continue to use the same experiment file structure as the main configs.
- No alternate trainer or evaluator codepath is introduced.

**Risk mitigation steps:**
- Catch host-level failures that CPU smoke cannot catch, including:
  - masked-GPU mistakes,
  - dataloader contention,
  - checkpoint path issues,
  - evaluation artifact issues.

**Prototype, fusion, contamination, projector, and metric-risk alignment:**
- This patch batch does not alter prototype modules, fusion, adaptation, or metric formulas.
- The mitigation here is non-interference:
  - the stress-smoke suite must use the current baseline and thesis models unchanged,
  - ensuring that safety validation does not contaminate the scientific comparison.

**Test plan and validation steps:**
- Local tests validate config loading and manifest generation for the new smoke family.
- On rented hardware, run all intended concurrent smoke sessions together before starting any full sweep.

**Acceptance criteria:**
- Stress-smoke launches through the exact comparative orchestration path.
- It produces:
  - `best.pt`,
  - `final.pt`,
  - `evaluation_metrics.json`,
  - `evaluation_records.json`,
  - `evaluation_curves.json`,
  - and the comparative execution report.

### Phase 5: Coarse-Grained Skip-Completed Recovery

**Phase summary tied to the thesis objectives:**  
The thesis experiments should be reproducible and cost-efficient. When a rented host dies after several completed runs or stages, the operator should not need to recompute already completed work.

**Files:**
- Modify: `scripts/run_comparative_smd_experiments.py`
- Modify: `scripts/run_three_stage_offline_pretraining.py`
- Reference: current comparative and three-stage execution reports

**Edits within this phase:**
- Add explicit `--skip-completed` behavior to the comparative runner.
- Read the existing execution report if present.
- Skip a run only if:
  - its `run_id` appears as completed in the report, and
  - its essential output artifacts still exist.
- Add parallel behavior for the three-stage path:
  - skip completed stages when `--skip-completed` is enabled,
  - continue from the next unfinished stage,
  - preserve existing stage command construction.

**Explicit edit content:**
- Comparative essential artifacts for skip eligibility:
  - `best.pt` or completed evaluation outputs depending on run family,
  - `evaluation_metrics.json`,
  - and execution report membership.
- Three-stage essential artifacts for skip eligibility:
  - stage checkpoint output for completed stages,
  - evaluation outputs for completed final evaluation.

**Interface and contract definitions:**
- Recovery here is at run boundary or stage boundary only.
- Trainer state semantics remain untouched.

**Design pattern application:**
- Strategy pattern: recovery behavior is explicit and opt-in, not hidden.

**Risk mitigation steps:**
- Reduce wasted rented compute after machine interruption.
- Avoid silent skipping by requiring both execution-report evidence and artifact existence.

**Test plan and validation steps:**
- Unit tests for:
  - empty report,
  - partial report,
  - stale report with missing artifacts,
  - resumed comparative execution,
  - resumed three-stage execution.

**Acceptance criteria:**
- A rerun after partial completion resumes from the first unfinished comparative run or three-stage phase when `--skip-completed` is explicitly enabled.
- No completed unit is skipped solely because it appears in a report while its output artifacts are missing.

### Phase 6: Validation Matrix and Pre-Rental Readiness Gate

**Phase summary tied to the thesis objectives:**  
This phase converts the code changes into an operational decision rule: only rent machines after the hardened launch path has passed local and host-level validation.

**Files:**
- Create: `tests/test_comparative_preflight.py`
- Create: `tests/test_comparative_run_plan.py`
- Create: `tests/test_comparative_resume_skip.py`
- Create: `tests/test_three_stage_resume_skip.py`
- Optionally modify: `tests/test_config_loading.py`
- Optionally create: `documents/logs/06-25-2026/detail/detail-safe-multi-machine-comparative-tmux-runs.md` follow-up section when results are known

**Edits within this phase:**
- Add local automated tests for all new orchestration surfaces.
- Add dry-run validation steps to be executed before renting.
- Add rented-host validation steps to be executed immediately after provisioning and before the full sweep.

**Explicit validation matrix:**

1. **Local static validation**
   - comparative preflight on a valid shard
   - comparative preflight on an invalid shard
   - generated worker override configs
   - skip-completed logic from synthetic reports

2. **Local dry-run validation**
   - one 1-GPU shard dry-run
   - one 3-GPU shard dry-run
   - one intentionally invalid GPU dry-run

3. **Rented-host smoke validation**
   - collect `nvidia-smi` inventory
   - run preflight per shard with the intended physical GPU index
   - run all intended stress-smoke sessions concurrently

4. **Full-launch gate**
   - only after all stress-smoke sessions pass should the real comparative sweep begin

**Acceptance criteria:**
- The repository can be declared “ready to rent” only if:
  - comparative GPU-aware preflight passes,
  - stress-smoke passes concurrently on the intended topology,
  - skip-completed behavior has been tested,
  - host-specific worker counts have been locked,
  - no fairness-critical training code was modified.

## Detailed Non-Goals

The following edits are explicitly out of scope for this detailed patch batch:

- Any change to loss functions, memory initialization semantics, or stage definitions.
- Any change to evaluation metric formulas such as VUS-PR, VUS-ROC, or affiliation-F1.
- Any attempt to change prototype fusion behavior, prototype redundancy control, adaptation contamination control, projector drift control, or online adaptation learning rules.

These risks remain scientifically important, but they are handled in this patch batch by preserving the current code paths exactly and by adding regression validation around orchestration only.

## Final Execution Order

1. Phase 0: scope guard
2. Phase 1: comparative GPU pinning
3. Phase 2: comparative GPU-aware preflight
4. Phase 3: host-level worker override support
5. Phase 4: GPU stress-smoke suite
6. Phase 5: coarse-grained skip-completed recovery
7. Phase 6: validation matrix and readiness gate

## Final Recommendation

Do not rent yet. The repository should first land and validate Phases 1 through 6 above. After that, the next practical step is not the full sweep immediately, but the rented-host concurrent stress-smoke on the exact two-machine topology that will be used for the real experiment.
