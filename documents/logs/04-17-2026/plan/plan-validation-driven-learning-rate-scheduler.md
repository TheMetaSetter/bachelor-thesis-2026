# Validation-Driven Learning Rate Scheduler

## Summary

The offline multitask training path should gain a conservative validation-driven learning rate scheduler. The correct first implementation is `ReduceLROnPlateau` monitored on clean `val_loss`, not on synthetic validation classification metrics. This keeps optimizer control aligned with the current checkpoint-selection policy and the real offline anomaly-detection validation surface.

The scheduler must be explicit in YAML, external to the model file, stepped by the trainer once per epoch after clean validation metrics are aggregated, and persisted in checkpoints for reproducibility.

## Current State

- [scripts/train.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/scripts/train.py) constructs a plain `torch.optim.Adam` from `optimizer.learning_rate` and `optimizer.weight_decay`.
- [src/engine/trainer.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/engine/trainer.py) uses clean `val_loss` for best-checkpoint selection and logs both clean `val_*` and synthetic `val_synth_*` metrics.
- [src/engine/checkpoint.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/engine/checkpoint.py) persists model and optimizer state, but not scheduler state.
- The current runtime has no learning rate scheduling and no learning rate logging to W&B.

## Design Choice

Use `torch.optim.lr_scheduler.ReduceLROnPlateau` with these principles:

- monitor only clean `val_loss`
- use conservative settings for long runs
- log the learning rate every epoch
- preserve current validation semantics and checkpoint policy

Recommended scheduler block for the 300-epoch server run:

```yaml
optimizer:
  learning_rate: 0.001
  weight_decay: 0.0
  scheduler:
    scheduler_name: reduce_on_plateau
    monitor_metric: val_loss
    factor: 0.5
    patience: 20
    threshold: 0.0001
    threshold_mode: rel
    cooldown: 3
    min_lr: 1.0e-5
```

## Planned Implementation

### 1. Extend config validation

Modify [src/core/config.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/core/config.py) to validate an optional optimizer scheduler block.

Required validation:

- `scheduler_name == reduce_on_plateau`
- `monitor_metric == val_loss`
- `0 < factor < 1`
- `patience >= 0`
- `cooldown >= 0`
- `threshold >= 0`
- `threshold_mode in {rel, abs}`
- `0 < min_lr <= learning_rate`

### 2. Build the scheduler in the runtime assembly layer

Modify [scripts/train.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/scripts/train.py):

- add a helper to build `ReduceLROnPlateau` from experiment config
- return `None` when no scheduler block is present
- pass the scheduler into the trainer

This preserves separation of concerns:

- config owns declaration
- script owns runtime assembly
- trainer owns stepping
- model remains untouched

### 3. Step the scheduler from clean val_loss in the trainer

Modify [src/engine/trainer.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/engine/trainer.py):

- accept `scheduler` in the constructor
- step the scheduler only after clean `val_loss` is aggregated
- do not use `val_synth_*` metrics
- keep best checkpoint selection unchanged

Log these metrics each epoch:

- `optimizer_lr`
- `optimizer_lr_group_0`
- `scheduler_monitor_val_loss`
- `scheduler_lr_reduced`

These should flow into `metrics.jsonl` and W&B automatically through the existing logger.

### 4. Persist scheduler state in checkpoints

Modify [src/engine/checkpoint.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/engine/checkpoint.py):

- save `scheduler_state_dict` when a scheduler exists
- restore it on load when provided
- keep backward compatibility with checkpoints that do not contain scheduler state

## Robust Test Cases

### Config validation tests

Extend [tests/test_config_loading.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/tests/test_config_loading.py):

- valid scheduler block loads
- invalid scheduler name is rejected
- invalid monitor metric is rejected
- invalid factor is rejected
- invalid patience is rejected
- invalid cooldown is rejected
- invalid threshold mode is rejected
- `min_lr > learning_rate` is rejected

### Scheduler unit tests

Add [tests/test_learning_rate_scheduler.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/tests/test_learning_rate_scheduler.py):

- scheduler builder returns `None` when config omits scheduler
- scheduler builder returns `ReduceLROnPlateau` when config is valid
- LR stays unchanged when `val_loss` improves
- LR drops after enough stagnant epochs
- cooldown prevents immediate repeated reductions
- LR does not go below `min_lr`

### Trainer integration tests

Extend [tests/test_multitask_validation_alignment.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/tests/test_multitask_validation_alignment.py) or add a dedicated trainer scheduler test:

- epoch metrics include `optimizer_lr`
- epoch metrics include `scheduler_monitor_val_loss`
- epoch metrics include `scheduler_lr_reduced`
- scheduler responds only to clean `val_loss`
- worsening `val_synth_loss` alone does not reduce LR
- best checkpoint is still chosen from clean `val_loss`

### Checkpoint round-trip tests

Extend [tests/test_checkpoint_roundtrip.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/tests/test_checkpoint_roundtrip.py):

- save scheduler state
- reload scheduler state into a fresh optimizer+scheduler pair
- load succeeds when scheduler state is absent

## Acceptance Criteria

- clean `val_loss` is the only scheduler monitor
- trainer logs LR state to W&B and `metrics.jsonl`
- scheduler is fully YAML-driven
- scheduler state is checkpointed
- smoke configs remain scheduler-free
- the multitask model file stays unchanged
- the scheduler behavior is covered by robust config, unit, trainer, and checkpoint tests
