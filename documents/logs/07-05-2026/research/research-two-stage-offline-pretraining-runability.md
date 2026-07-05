---
date: 2026-07-05 13:26:08 +07:00
researcher: TheMetaSetter
git_commit: 525b7175650ed0e4048d9cf77fd0226bf811f815
branch: dev
repository: bachelor-thesis-2026
topic: "Can the current codebase run the offline_pretraining_two_stage_kmeans_memory design?"
tags: [research, time-series, anomaly-detection, multi-stage, offline-pretraining, kmeans, thesis-multitask]
status: complete
last_updated: 2026-07-05
last_updated_by: TheMetaSetter
---

# Research: Can the current codebase run the two-stage offline pretraining design?

**Date**: 2026-07-05 13:26:08 +07:00  
**Researcher**: TheMetaSetter  
**Git Commit**: 525b7175650ed0e4048d9cf77fd0226bf811f815  
**Branch**: dev

## Research Question
Can the current repository run the experiment described in `documents/design/offline_pretraining_two_stage_kmeans_memory_design.md`?

## Summary
Yes, the codebase currently supports the approved two-stage offline pretraining flow at the level of configuration validation, manifest generation, stage planning, Stage B memory initialization, and dry-run orchestration.

The experiment is wired through a dedicated runner script, a two-stage config schema, and a model implementation that can switch between Stage A and Stage B by `training_phase`. The current verification covered:

1. config loading and validation for the two-stage experiment YAML files,
2. the two-stage training plan builder,
3. the model phase switch between Stage A and Stage B,
4. a dry-run of the full runner orchestration.

I did not execute the full 100-epoch training job in this research pass. The orchestration path is present and the dry-run succeeds.

## Detailed Findings

### Data Preparation
- The active experiment config points to the SMD machine-specific data file and uses the windowed SMD setup for the new rerun.
- The main experiment config is `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-two-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml`.
- That config references `configs/data/smd_rtx3090_machine_3_4_20_stride1.yaml`, which keeps the SMD machine-3-4 split and the `window20` / `stride1` contract.
- The task config is `configs/task/multitask_tsad_redlamp_multiclass_window20.yaml`, so the anomaly taxonomy is the RedLamp multiclass set with synthetic anomaly augmentation enabled.

The runner materializes stage-specific configs from the experiment config. During Stage B initialization it rebuilds a dataset bundle from the resolved data config, sets `shuffle_train = False` and `num_workers = 0`, and passes the resulting `train_loader` into the model memory initialization hook.

### Modeling and Training
- `src/models/thesis_multitask.py` still owns the public model entrypoint.
- The constructor delegates into mixins for setup, state, routing, and loss logic, but the public API remains `ThesisMultitaskModel`.
- The setup mixin stores `training_phase`, `freeze_memories_after_initialization`, `continuous_num_prototypes`, and `discrete_codebook_size`.
- For the two-stage rerun, the model starts Stage A with `memory_initialized = False` and `memory_training_enabled = False`.
- The phase helpers separate Stage A and Stage B behavior:
  - Stage A uses the contrastive objective.
  - Stage B uses the prototype path and freezes the encoder.
- The setup code still builds the continuous prototype bank, discrete codebook, fusion scalars, reconstruction head, and classification head from the config.
- The state mixin exposes `maybe_initialize_memories_from_loader(...)`, which is the hook used at the end of Stage A to initialize memory banks from the latent token pool.
- `src/engine/trainer.py` calls that hook during training, so the model can initialize memory without a separate engine-specific codepath.

The approved two-stage config values are present in the active model config:

- `continuous_num_prototypes: 32`
- `discrete_codebook_size: 60`
- `discrete_query_mode: cosine_topk`
- `training_phase: stage_a_multitask_pretraining`
- `freeze_memories_after_initialization: true`
- `freeze_recovered_zipped_encoder_during_warmup: true`

That matches the design document’s two-stage direction.

### Evaluation
- The runner uses `checkpoint_monitor_metric: val_synth_vus_pr`.
- The final evaluation command is built from the Stage B config and the Stage B best checkpoint.
- The runner writes a manifest and then an execution report under the experiment output directory.

The config and runner tests confirm the expected epoch split:

- Stage A = 80 epochs
- Stage B = 20 epochs
- Total = 100 epochs

## Code References
- `documents/design/offline_pretraining_two_stage_kmeans_memory_design.md:1-204` - approved two-stage design contract
- `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-two-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml:7-47` - active two-stage experiment config
- `configs/model/thesis_multitask_two_stage_window20.yaml:1-59` - model config for the two-stage rerun
- `configs/task/multitask_tsad_redlamp_multiclass_window20.yaml:1-24` - task config and anomaly taxonomy
- `src/core/config.py:228-276` - two-stage config validation
- `src/core/config_experiment_validation.py:13-91` - experiment-level validation and mutual exclusion of three-stage vs two-stage
- `src/models/thesis_multitask.py:37-67` - public model entrypoint and constructor
- `src/models/thesis_multitask_setup_mixin.py:135-170` - phase storage and two-stage initialization reset
- `src/models/thesis_multitask_setup_mixin.py:189-346` - phase helpers and trainable-surface freezing
- `src/models/thesis_multitask_setup_mixin.py:359-455` - memory bank and task-head construction
- `src/models/thesis_multitask_state_mixin.py:273-323` - memory initialization from the training loader
- `src/engine/trainer.py:565-586` - trainer hook that calls memory initialization
- `scripts/run_two_stage_offline_pretraining.py:45-91` - epoch budget and two-stage plan builder
- `scripts/run_two_stage_offline_pretraining.py:105-145` - stage config generation
- `scripts/run_two_stage_offline_pretraining.py:151-199` - manifest materialization
- `scripts/run_two_stage_offline_pretraining.py:206-245` - Stage B initialization checkpoint construction
- `scripts/run_two_stage_offline_pretraining.py:270-309` - orchestration and execution report
- `tests/test_offline_pretraining_two_stage_config_loading.py:6-43` - config budget and contract checks
- `tests/test_offline_pretraining_two_stage_runner.py:6-69` - plan builder and phase-surface checks

## Pipeline Documentation

### High-Level Flow
```text
experiment yaml
  -> load_experiment_config()
  -> validate_experiment_config()
  -> build two-stage manifest
  -> Stage A config
  -> Stage A training
  -> Stage B initialization checkpoint
  -> Stage B config
  -> Stage B training
  -> evaluate.py on best Stage B checkpoint
```

### Stage Flow
```text
Stage A
  encoder trainable
  reconstruction + classification + contrastive losses
  end-of-stage memory initialization from train loader

Stage B
  encoder frozen
  continuous memory frozen
  discrete memory frozen
  fusion and prediction heads trainable
```

### Runner Flow
```text
configured experiment
    |
    v
materialize_two_stage_run_manifest()
    |
    +--> stage A config
    +--> stage B config
    |
    v
execute_two_stage_plan()
    |
    +--> optional Stage B initialization checkpoint
    +--> train.py for Stage A
    +--> train.py for Stage B
    +--> evaluate.py for final checkpoint
```

## Historical Context (from documents/)
- This rerun is explicitly two-stage and supersedes the older three-stage intent for this specific experiment.
- The design document keeps `thesis_multitask.py` as the only model in scope for this rerun.
- The repository still contains older three-stage configs, but the current exp4 two-stage config is the approved path for this experiment.

## Open Questions
- I did not execute the full 100-epoch training job, so end-to-end runtime on the target GPU is still unverified.
- The active exp4 config uses `wandb_mode: online`, so a real run still depends on W&B authentication and network availability.
- The dataset files referenced by the SMD config must exist locally for a full run.

## Verification Performed
- `./.venv/bin/python -m pytest -q tests/test_offline_pretraining_two_stage_config_loading.py tests/test_offline_pretraining_two_stage_runner.py`
- `./.venv/bin/python scripts/run_two_stage_offline_pretraining.py --experiment-config configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-two-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml --dry-run`

## Result
The repository currently has the codepath needed to run the approved two-stage offline pretraining experiment, and the configuration / orchestration surface is consistent with the design document. The remaining unknown is only the full long-running training execution on the target device and dataset, which was not part of this research pass.
