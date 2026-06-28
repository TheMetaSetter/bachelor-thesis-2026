---
date: 2026-06-25 19:53:07 +0700
researcher: TheMetaSetter
git_commit: 89a598f643cf0c20b0ab540b926e6b71f27e975f
branch: dev
repository: bachelor-thesis-2026
topic: "Safe patch points for multi-machine comparative tmux runs on rented GPU servers"
tags: [research, time-series, anomaly-detection, orchestration, tmux, gpu]
status: complete
last_updated: 2026-06-25
last_updated_by: TheMetaSetter
---

# Research: Safe patch points for multi-machine comparative tmux runs on rented GPU servers

**Date**: 2026-06-25 19:53:07 +0700  
**Researcher**: TheMetaSetter  
**Git Commit**: `89a598f643cf0c20b0ab540b926e6b71f27e975f`  
**Branch**: `dev`

## Research Question

Use the research prompt workflow to identify which parts of the current codebase are directly involved in multi-machine comparative experiment launches and which of those parts should be patched to make rented-server execution safer, especially for GPU pinning, smoke validation, worker contention, artifact collisions, and failure recovery.

## Summary

The current repository already has a strong config-driven experiment core, unique per-run artifact directories, and a three-stage launcher with explicit GPU index handling. However, the comparative path is materially less safe than the three-stage path for rented multi-machine operation.

The most important findings are the following:

1. The comparative tmux launcher does not currently pin jobs to a specific GPU via `CUDA_VISIBLE_DEVICES`, whereas the three-stage launcher already does.
2. The comparative runner has only a lightweight config-and-path preflight. It does not perform GPU readiness validation analogous to the three-stage preflight path.
3. The current comparative smoke configs are CPU-only, use `num_workers: 0`, and disable Weights & Biases. They are suitable for functional checks but not for operational stress validation on rented GPU hosts.
4. The current comparative and three-stage runners record execution reports, but neither path currently resumes work from those reports. A machine failure therefore requires manual restart planning.
5. The training path saves `best.pt` and `final.pt`, but there is no general-purpose mid-run resume path that restores optimizer, scheduler, and epoch progression for ordinary offline runs.
6. The data loader configuration used by the main comparative SMD entity configs keeps `num_workers: 16`, which is reasonable for one run on one host but risky for three concurrent jobs on a 32-core rented machine because each process builds persistent worker pools.

These findings indicate that the highest-value safety patches are orchestration patches rather than model patches.

## Detailed Findings

### Data Preparation

The SMD comparative experiments use entity-specific data configs such as:

- `configs/data/smd_rtx3090_machine_1_6_20_stride1.yaml`
- `configs/data/smd_rtx3090_machine_3_1_20_stride1.yaml`
- `configs/data/smd_rtx3090_machine_3_9_20_stride1.yaml`

All three active main comparative configs currently set:

- `window_size: 20`
- `stride: 1`
- `batch_size: 256`
- `num_workers: 16`
- `validation_split_ratio: 0.2`

Relevant code and config:

- `configs/data/smd_rtx3090_machine_1_6_20_stride1.yaml:1-10`
- `configs/data/smd_rtx3090_machine_3_1_20_stride1.yaml:1-10`
- `configs/data/smd_rtx3090_machine_3_9_20_stride1.yaml:1-10`

The actual loader construction occurs in `src/data/loaders.py`. The code resolves `num_workers`, builds three `DataLoader` instances, and enables `persistent_workers=True` whenever `resolved_num_workers > 0`.

Relevant code:

- `src/data/loaders.py:25-50` - worker resolution
- `src/data/loaders.py:96-132` - train, validation, and test loader construction

This matters operationally because three simultaneous jobs on one 32-core machine do not merely create one pool of 16 workers each. Each job constructs separate train, validation, and test loaders with persistent worker support enabled. The code does not cap the total worker budget across jobs or hosts.

The public data API exposes `num_workers` and `min_num_workers`, but it does not provide a host-level safety override, shard-level override, or concurrency-aware worker cap.

Relevant code:

- `src/data/api.py:14-52`
- `src/data/api.py:107-140`
- `src/core/config.py:931-944`

#### Safety implication

For single-run execution this design is acceptable. For three simultaneous comparative jobs on one 32-core rented machine, it is a realistic source of CPU oversubscription, dataloader instability, and reduced GPU utilization. The codebase currently leaves that scheduling responsibility entirely to the operator.

### Modeling and Training

The training entrypoint remains simple and stable. It loads the resolved config, seeds the runtime, builds the dataset bundle, constructs the model, and delegates training to `Trainer`.

Relevant code:

- `scripts/train.py:224-260`

#### GPU device semantics

The training and evaluation paths move tensors and models to `experiment_config["device"]`, which in the comparative configs is currently the string `cuda`.

Relevant code:

- `scripts/train.py:242-250`
- `src/engine/trainer.py:134-147`
- `src/engine/trainer.py:583-665`
- `scripts/evaluate.py:86-92`
- `src/engine/evaluator.py:142-148`
- `src/engine/evaluator.py:199-223`

This means the code is compatible with standard Linux GPU masking. If a process is launched with `CUDA_VISIBLE_DEVICES=<physical_index>`, then `device: cuda` correctly resolves to the first visible logical GPU inside that process. The comparative path therefore does not need model-level GPU changes. It needs launcher-level masking.

#### Comparative orchestration path

The comparative orchestration logic is defined in two layers:

- `scripts/launch_tmux_comparative_smd_experiment.sh`
- `scripts/run_comparative_smd_experiments.py`

The shell launcher defines fixed smoke and main config lists and writes one log file and one report directory per tmux session.

Relevant code:

- `scripts/launch_tmux_comparative_smd_experiment.sh:5-18`
- `scripts/launch_tmux_comparative_smd_experiment.sh:19-43`
- `scripts/launch_tmux_comparative_smd_experiment.sh:130-156`
- `scripts/launch_tmux_comparative_smd_experiment.sh:192-198`

The Python comparative runner validates config resolution, dataset roots, and duplicate output/checkpoint paths within the provided run set. It then executes all smoke runs sequentially, followed by all main runs sequentially.

Relevant code:

- `scripts/run_comparative_smd_experiments.py:65-95` - dataset root and artifact-path validation
- `scripts/run_comparative_smd_experiments.py:211-247` - manifest construction
- `scripts/run_comparative_smd_experiments.py:267-308` - sequential execution and failure handling
- `scripts/run_comparative_smd_experiments.py:327-339` - main program flow

#### Three-stage orchestration path

The three-stage launcher is materially safer. It accepts `--gpu-index`, validates a target GPU via a dedicated preflight script, and launches the training command under `CUDA_VISIBLE_DEVICES=<gpu_index>`.

Relevant code:

- `scripts/launch_tmux_three_stage_experiment.sh:8-15`
- `scripts/launch_tmux_three_stage_experiment.sh:140-149`
- `scripts/launch_tmux_three_stage_experiment.sh:182-201`

The preflight script checks exact 300-epoch budget readiness, test-window readiness, tmux availability, and GPU readiness for a specific physical GPU index.

Relevant code:

- `scripts/preflight_three_stage_server.py:121-155`
- `scripts/preflight_three_stage_server.py:226-280`
- `scripts/preflight_three_stage_server.py:294-313`

#### Safety implication

The comparative path currently lacks the three concrete protections already present in the three-stage path:

1. explicit GPU index selection;
2. GPU-name validation before launch;
3. launch-readiness reporting tied to the intended GPU target.

### Evaluation

For baseline comparative runs, the comparative runner trains and then evaluates by invoking `scripts/evaluate.py` with the experiment config and the expected best-checkpoint path.

Relevant code:

- `scripts/run_comparative_smd_experiments.py:132-154`

The evaluation script writes:

- `evaluation_records.json`
- `evaluation_metrics.json`
- `evaluation_curves.json`
- `resolved_experiment_config.json`

Relevant code:

- `scripts/evaluate.py:160-182`

These artifacts provide a clean run-completion signal at the run level. However, the current comparative runner does not consult those artifacts when restarting after failure. It always iterates the full command list from the top of the supplied plan.

Relevant code:

- `scripts/run_comparative_smd_experiments.py:288-307`

For the three-stage path, a similar issue exists one level deeper. The script writes a stage execution report and records completed stage names, but it does not skip completed stages or resume from the report on restart.

Relevant code:

- `scripts/run_three_stage_offline_pretraining.py:619-710`

#### Safety implication

The repository already produces enough metadata to support coarse-grained resume behavior, but the current launch logic does not use that metadata to continue safely after rented-server interruption.

## Patch-Relevant Findings

### 1. Comparative GPU pinning is missing

The comparative tmux launcher does not currently prepend the actual training process with `CUDA_VISIBLE_DEVICES=...`, nor does it expose a `--gpu-index` argument.

Relevant code:

- `scripts/launch_tmux_comparative_smd_experiment.sh:136-156`

By contrast, the three-stage launcher already follows the correct masking pattern:

- `scripts/launch_tmux_three_stage_experiment.sh:182-201`

#### Patch relevance

This is the highest-priority patch candidate because it directly determines whether three simultaneous jobs on one host will bind to distinct physical GPUs or accidentally converge onto one device.

### 2. Comparative preflight is too weak for rented multi-GPU hosts

The comparative runner’s `--preflight-only` mode currently validates:

- config resolution;
- duplicate paths within the provided config set;
- dataset-root existence.

Relevant code:

- `scripts/run_comparative_smd_experiments.py:217-226`
- `scripts/run_comparative_smd_experiments.py:337-338`

It does not validate:

- intended GPU index;
- GPU model identity;
- whether the target host is suitable for concurrent comparative launch;
- whether the chosen smoke or main plan reflects the actual concurrency topology.

The three-stage path already has a richer pattern to copy from.

Relevant code:

- `scripts/preflight_three_stage_server.py:244-278`

#### Patch relevance

This is a high-priority patch candidate because it is the natural place to fail fast before renting hours are wasted.

### 3. Comparative smoke configs are functional, not operational

The current smoke configs explicitly use:

- `device: cpu`
- `num_workers: 0`
- capped window counts
- `use_wandb: false`
- `wandb_mode: disabled`

Relevant code:

- `configs/experiment/comparative/baseline/smd__redlamp_mlp_baseline__comparative-single-stage-machine_1_6__w20__seed6__smoke.yaml:8-19`
- `configs/experiment/comparative/baseline/smd__redlamp_mlp_baseline__comparative-single-stage-machine_1_6__w20__seed6__smoke.yaml:36-45`
- `configs/experiment/comparative/thesis/smd__thesis_multitask__comparative-three-stage-machine_1_6__w20__seed6__smoke.yaml:8-19`
- `configs/experiment/comparative/thesis/smd__thesis_multitask__comparative-three-stage-machine_1_6__w20__seed6__smoke.yaml:36-55`

#### Patch relevance

These smoke configs are useful for logic validation, but they do not pressure the real failure modes of a rented GPU run. A distinct stress-smoke path would be needed if the goal is operational safety rather than only config correctness.

### 4. No comparative sharding support is built into the launcher

The comparative launcher hard-codes one full smoke set and one full main config list. It does not expose a way to:

- run only a subset of configs;
- select a shard by entity or seed;
- name that shard explicitly in the manifest;
- protect the user from accidentally launching overlapping config sets on different machines.

Relevant code:

- `scripts/launch_tmux_comparative_smd_experiment.sh:19-43`
- `scripts/launch_tmux_comparative_smd_experiment.sh:136-156`

#### Patch relevance

This is a medium-to-high priority patch candidate because multi-machine safety currently depends on careful manual copy-paste.

### 5. No general-purpose resume path exists for ordinary offline training

The trainer saves:

- `best.pt` when the monitored metric improves;
- `final.pt` at the end of training.

Relevant code:

- `src/engine/trainer.py:852-956`

The checkpoint payload is rich enough to support resume because it includes:

- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict` when present
- `epoch`
- `metric_history`

Relevant code:

- `src/engine/checkpoint.py:28-79`

However, the ordinary training entrypoint only supports `initialization_checkpoint_path`, which loads model weights before training but does not resume an interrupted run’s optimizer state, scheduler state, or epoch counter.

Relevant code:

- `scripts/train.py:102-118`
- `src/engine/checkpoint.py:81-106`

#### Patch relevance

This is a high-priority safety patch if rented-machine interruption is treated as a realistic scenario. It is larger than a launcher patch, but it directly protects expensive long runs.

### 6. Comparative and three-stage execution reports are write-only

Both orchestration paths emit execution reports with completed units:

- comparative: completed run ids;
- three-stage: completed stage names.

Relevant code:

- `scripts/run_comparative_smd_experiments.py:272-307`
- `scripts/run_three_stage_offline_pretraining.py:633-710`

But neither script reads those reports on restart to skip already completed work.

#### Patch relevance

This is a medium-to-high priority patch candidate because it is easier to implement than full optimizer-state resume and still reduces repeated work after host failure.

### 7. Worker policy is not concurrency-aware

The config validator accepts either an integer or `"auto"` for `data.num_workers`.

Relevant code:

- `src/core/config.py:931-944`

The loader resolver maps `"auto"` to all visible CPUs subject to a floor, not to a safe per-job cap.

Relevant code:

- `src/data/loaders.py:25-43`

#### Patch relevance

This is a medium-priority safety patch candidate. It is not a correctness issue for single runs, but it becomes an operational risk on the three-GPU rented machine. A patch here should prefer explicit host- or shard-level overrides rather than broad hidden heuristics.

## Code References

- `scripts/launch_tmux_comparative_smd_experiment.sh:5-18` - default session and report-path conventions
- `scripts/launch_tmux_comparative_smd_experiment.sh:19-43` - hard-coded comparative smoke/main config lists
- `scripts/launch_tmux_comparative_smd_experiment.sh:136-156` - comparative tmux inner command without GPU masking
- `scripts/run_comparative_smd_experiments.py:65-95` - duplicate artifact-path and dataset-root validation
- `scripts/run_comparative_smd_experiments.py:211-247` - comparative manifest generation
- `scripts/run_comparative_smd_experiments.py:267-308` - sequential comparative execution and failure report writing
- `scripts/launch_tmux_three_stage_experiment.sh:140-149` - target-GPU preflight arguments
- `scripts/launch_tmux_three_stage_experiment.sh:182-201` - three-stage tmux launch with `CUDA_VISIBLE_DEVICES`
- `scripts/preflight_three_stage_server.py:226-280` - three-stage launch-readiness summary
- `scripts/train.py:102-118` - initialization-only checkpoint loading
- `src/engine/trainer.py:852-956` - best/final checkpoint save logic
- `src/engine/checkpoint.py:28-79` - checkpoint payload and save path
- `src/engine/checkpoint.py:81-106` - checkpoint loading path
- `src/data/loaders.py:25-50` - worker resolution
- `src/data/loaders.py:96-132` - persistent worker-enabled loader construction
- `configs/experiment/comparative/baseline/smd__redlamp_mlp_baseline__comparative-single-stage-machine_1_6__w20__seed6__smoke.yaml:8-19` - CPU smoke data overrides
- `configs/experiment/comparative/thesis/smd__thesis_multitask__comparative-three-stage-machine_1_6__w20__seed6__smoke.yaml:8-19` - CPU smoke data overrides

## Pipeline Documentation

The current comparative launch pipeline is:

1. shell launcher resolves a fixed smoke config list and fixed main config list;
2. Python runner resolves configs and validates local path uniqueness within that run set;
3. smoke configs run first in CPU mode with reduced windows and disabled Weights & Biases;
4. main configs run sequentially, each using either:
   - baseline single-stage train followed by evaluate; or
   - thesis three-stage offline pre-training orchestration followed by evaluate;
5. artifacts are written into per-experiment `output_dir` and `checkpoint_dir` trees.

The current three-stage orchestration pipeline is:

1. validate exact epoch budget and materialize a stage manifest;
2. generate per-stage configs and stage output directories;
3. optionally prepare a Stage 2 recovery initialization checkpoint;
4. run stage subprocesses sequentially;
5. run final evaluation;
6. write a stage execution report.

In both cases, execution reports are durable, but they are not yet consumed for automatic resume or skip behavior.

## Historical Context (from documents/)

The design documents emphasize a thin, config-driven runtime waist:

- `documents/design/idea.md` frames the thesis as a modular multivariate anomaly-detection system with window length `L = 20`, config-driven experimentation, and stage-aware offline training.
- `documents/design/design_starter.md` emphasizes the same small-contract philosophy: datasets should emit a stable batch contract, models should stay self-contained, and the engine layer should remain thin.

This historical context supports safety patches that:

- remain orchestration-centric rather than model-centric;
- preserve readability;
- avoid introducing many hidden code paths;
- reuse existing preflight and manifest patterns when possible.

## Open Questions

1. The exact physical GPU index mapping on the rented machines has not yet been verified. This requires the host outputs of `nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total --format=csv,noheader`.
2. The exact safe worker budget for the three-GPU rented machine is still host-dependent. The code evidence indicates that `num_workers: 16` is likely too aggressive for three concurrent jobs, but the optimal replacement should be confirmed with a parallel stress smoke on the real machine.
3. The repository currently has no stated policy for whether Weights & Biases should remain `online` during rented-machine stress smoke. If network instability is expected, a launcher-level override may be safer than editing experiment YAMLs by hand.

## Conclusion

The codebase does not require architectural model changes to become meaningfully safer for rented multi-machine comparative runs. The most important patch points all lie in orchestration, preflight, concurrency control, and recovery behavior. The strongest immediate safety gains would come from:

1. comparative GPU pinning support;
2. comparative GPU-aware preflight;
3. a stress-smoke path that reflects real GPU-host conditions;
4. run-level or stage-level resume and skip logic;
5. explicit per-host worker overrides for concurrent runs.
