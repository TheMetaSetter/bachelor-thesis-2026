---
date: 2026-05-17 16:50:31 +07
researcher: TheMetaSetter
git_commit: 9aaa94e6a88e95c8dfcdbdb71bc08fc5f3ad921c
branch: dev
repository: bachelor-thesis-2026
topic: "Current codebase support for AdamW, cosine scheduling, gradient clipping configuration, and cosine scheduler stepping"
tags: [research, time-series, anomaly-detection, training-runtime]
status: complete
last_updated: 2026-05-17
last_updated_by: TheMetaSetter
---

# Research: Current codebase support for AdamW, cosine scheduling, gradient clipping configuration, and cosine scheduler stepping

**Date**: 2026-05-17 16:50:31 +07  
**Researcher**: TheMetaSetter  
**Git Commit**: 9aaa94e6a88e95c8dfcdbdb71bc08fc5f3ad921c  
**Branch**: dev

## Research Question

What is the current implementation state of the repository for the four planned training-runtime changes: `optimizer_name: adamw`, `scheduler_name: cosine`, gradient clipping norm from configuration, and cosine-specific scheduler stepping logic?

## Summary

The active offline training path currently supports a narrow training runtime surface. The experiment configuration stores only `learning_rate`, `weight_decay`, and an optional scheduler block. The offline entrypoint always constructs `torch.optim.Adam`, the scheduler validation and builder accept only `reduce_on_plateau`, and the trainer always steps the scheduler with a validation metric at the end of each epoch. No gradient clipping logic is present in the active trainer loop, and no gradient clipping configuration field is validated or loaded. The current implementation therefore does not yet support the desired `AdamW + cosine + configurable gradient clipping` experiment family without code changes.

## Detailed Findings

### Data Preparation

- The directly referenced experiment configuration is `configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml`. It targets SMD machine `2-1` through `configs/data/smd_rtx3090_machine_2_1_20.yaml`, uses the `redlamp_mlp_baseline` model, and runs for `300` epochs.
- The experiment file already names a run as `..._adamw_cosine`, but the live configuration still specifies only `learning_rate`, `weight_decay`, and a `reduce_on_plateau` scheduler. The optimizer family and cosine scheduler are not represented in the active configuration schema.
- The design documents continue to require explicit YAML-controlled experiment behavior and ablation-friendly switches. The current optimization runtime is more limited than that intended control surface.

### Modeling and Training

#### Optimizer construction

- `scripts/train.py` constructs the offline optimizer directly as `torch.optim.Adam(...)`.
- The optimizer type is not read from configuration. There is no `optimizer_name` field in the current validation logic or in the active experiment files.
- `scripts/run_online_adaptation.py` also constructs `torch.optim.Adam(...)` directly for the online adaptation path, which is a separate optimizer construction site from offline training.

#### Scheduler construction

- `src/core/config.py` validates an optional scheduler block but accepts only `scheduler_name: reduce_on_plateau`.
- The allowed scheduler monitor metrics are `val_loss`, `val_synth_loss`, `val_synth_roc_auc`, and `val_synth_pr_auc`.
- `scripts/train.py` mirrors that restriction: it rejects any scheduler name except `reduce_on_plateau` and builds only `torch.optim.lr_scheduler.ReduceLROnPlateau`.
- Existing experiment configurations with schedulers all use `reduce_on_plateau`; there are no cosine scheduler examples in `configs/experiment/`.

#### Scheduler stepping

- `src/engine/trainer.py` uses a single scheduler stepping path:
  - it requires a configured `scheduler_monitor_metric`,
  - reads the corresponding metric from the epoch metrics,
  - calls `self.scheduler.step(monitor_value)`,
  - and logs whether the learning rate was reduced.
- This contract is appropriate for `ReduceLROnPlateau`, but it is not the same contract used by standard cosine schedulers such as `CosineAnnealingLR`, which normally call `step()` without a validation metric.
- The best-checkpoint monitor is also coupled to the scheduler monitor metric when a scheduler is present. This is useful for plateau scheduling, but a non-metric scheduler such as cosine would need an explicit checkpoint-monitor rule if the current behavior is preserved conceptually.

#### Gradient clipping

- The active offline trainer loop executes `zero_grad()`, `loss.backward()`, then `optimizer.step()`.
- No call to `torch.nn.utils.clip_grad_norm_` or an equivalent clipping operation exists under `src/`, `scripts/`, `tests/`, or `configs/`.
- `src/core/config.py` does not validate any field such as `gradient_clip_norm`, `max_grad_norm`, or `grad_clip_norm`.
- The current codebase therefore has no implemented runtime path for reading a clipping norm from YAML and applying it during optimization.

#### Checkpoint behavior

- `src/engine/checkpoint.py` already saves and restores scheduler state generically whenever a scheduler object is provided.
- The existing checkpoint test covers scheduler state roundtripping only for `ReduceLROnPlateau`.
- The generic checkpoint implementation is not inherently limited to plateau scheduling, but cosine-specific coverage does not yet exist.

### Evaluation

- The requested changes do not alter the evaluation data path directly.
- However, training metrics currently include optimizer learning-rate values and scheduler-monitor values generated by the plateau-specific stepping function. A cosine scheduler would still need learning-rate logging, but it would not naturally emit a validation monitor metric.
- Existing tests assert plateau-specific metric keys such as `scheduler_monitor_val_loss` and `scheduler_lr_reduced`, showing that the training metrics surface is currently shaped around plateau scheduling semantics.

## Code References

- `configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml:1` - referenced experiment configuration using SMD machine `2-1`, `300` epochs, and a plateau scheduler despite an `adamw_cosine` run name.
- `src/core/config.py:164` - optimizer validation currently includes only numeric `learning_rate` and `weight_decay`.
- `src/core/config.py:242` - scheduler validation accepts only `reduce_on_plateau`.
- `scripts/train.py:79` - scheduler builder supports only `ReduceLROnPlateau`.
- `scripts/train.py:176` - offline optimizer construction is hard-coded to `torch.optim.Adam`.
- `src/engine/trainer.py:110` - scheduler stepping is metric-driven and calls `scheduler.step(monitor_value)`.
- `src/engine/trainer.py:306` - active training loop has no gradient clipping between backward pass and optimizer step.
- `src/engine/checkpoint.py:28` - scheduler state is already checkpointed generically when present.
- `scripts/run_online_adaptation.py:111` - online adaptation optimizer construction is separately hard-coded to `torch.optim.Adam`.
- `tests/test_learning_rate_scheduler.py:90` - scheduler test fixtures currently describe only `reduce_on_plateau`.
- `tests/test_checkpoint_roundtrip.py:57` - scheduler checkpoint roundtrip test currently covers only `ReduceLROnPlateau`.

## Pipeline Documentation

For the active offline path, the present optimization pipeline is:

1. Load and validate the resolved experiment configuration.
2. Build the model from the merged model and task configuration.
3. Construct `torch.optim.Adam` from `optimizer.learning_rate` and `optimizer.weight_decay`.
4. Optionally construct `ReduceLROnPlateau` from `optimizer.scheduler`.
5. For each training batch:
   - move the batch to device,
   - call the model's `training_step`,
   - run backward propagation,
   - call the optimizer step.
6. After validation at the end of each epoch:
   - aggregate validation metrics,
   - step the scheduler from one configured monitor metric,
   - append optimizer learning-rate metrics,
   - save checkpoints including scheduler state when present.

The runtime currently has one optimizer code path, one scheduler code path, and no gradient-clipping code path.

## Historical Context (from documents/)

- `documents/design/idea.md` describes a research codebase that should keep optimization and objective behavior explicit and ablation-friendly through configuration.
- `documents/design/design_starter.md` presents the intended configuration surface with optimizer selection and `grad_clip_norm` as explicit training controls in the illustrative YAML design. The present implementation has not yet reached that broader configuration surface.
- `codebase_preferences.md` emphasizes readability, explicit configuration, and minimizing hidden code paths. Any future extension for `AdamW`, cosine scheduling, and gradient clipping should remain aligned with that preference by keeping the control flow visible rather than implicit.

## Open Questions

1. Should cosine scheduling be stepped once per epoch, matching the current trainer boundary, or per iteration, matching the style used in the CANDI reference implementation?
2. If cosine scheduling does not require a monitor metric, what metric should determine the best checkpoint: retain `val_loss`, allow a separate `checkpoint_monitor_metric`, or preserve the current plateau monitor behavior only for plateau schedulers?
3. Should optimizer selection be extended only for the offline training path first, or should the online adaptation path receive the same optimizer-selection abstraction in the same programming pass?
4. What exact YAML schema should represent gradient clipping: `gradient_clip_norm`, `max_grad_norm`, or another field name already preferred by the design documentation?
