---
date: 2026-05-19 21:35:23 +07+0700
researcher: TheMetaSetter
git_commit: a9487c8ec34cbe8253fb21e9ba06dba9ef12949b
branch: dev
repository: bachelor-thesis-2026
topic: "Root cause analysis of scheduler monitor metric validation error for val_synth_vus_pr"
tags: [research, time-series, anomaly-detection, multi-class]
status: complete
last_updated: 2026-05-19
last_updated_by: TheMetaSetter
---

# Research: Root cause analysis of scheduler monitor metric validation error for val_synth_vus_pr

**Date**: 2026-05-19 21:35:23 +07+0700  
**Researcher**: TheMetaSetter  
**Git Commit**: a9487c8ec34cbe8253fb21e9ba06dba9ef12949b  
**Branch**: dev

## Research Question
Why does `python3 scripts/train.py --experiment-config configs/experiment/smd_thesis_multitask_redlamp_multiclass_window20_nobootstrap.yaml` fail with:
`ValueError: optimizer.scheduler.monitor_metric must be one of: val_loss, val_synth_loss, val_synth_roc_auc, val_synth_pr_auc`?

## Summary
The failure is caused by configuration-validation contract drift between environments, not by a malformed experiment YAML in the current local repository snapshot. The experiment YAML sets `optimizer.scheduler.monitor_metric: val_synth_vus_pr`, and the current local `src/core/config.py` explicitly allows this value. The error text provided by the user corresponds to an older validator state that did not include `val_synth_vus_pr` in the scheduler monitor whitelist. Therefore, the run on the remote server is using an older code version than the local repository state used in this analysis.

## Detailed Findings

### Data Preparation
- The referenced experiment uses SMD machine-level data through `configs/data/smd_rtx3090_machine_2_1_20.yaml` as declared in `configs/experiment/smd_thesis_multitask_redlamp_multiclass_window20_nobootstrap.yaml`.
- The current task concerns configuration validation before the data loader is constructed; the process fails in config validation stage.

### Modeling and Training
- The experiment config uses `reduce_on_plateau` scheduler and sets:
  - `monitor_metric: val_synth_vus_pr`
  - `patience: 15`
  - `epochs: 300`
- In training orchestration, `scripts/train.py` accepts `reduce_on_plateau` and forwards `monitor_metric` directly to `torch.optim.lr_scheduler.ReduceLROnPlateau` setup.
- Scheduler mode is selected as `max` for non-loss metrics, so `val_synth_vus_pr` is operationally compatible.

### Evaluation
- `checkpoint_monitor_metric` is also `val_synth_vus_pr` in the same config.
- Current local config validator already whitelists `val_synth_vus_pr` for checkpoint monitoring and scheduler monitoring.
- The remote traceback message indicates a whitelist that ends at `val_synth_pr_auc`, which is an earlier contract state.

## Code References
- `configs/experiment/smd_thesis_multitask_redlamp_multiclass_window20_nobootstrap.yaml:16` - scheduler type is `reduce_on_plateau`
- `configs/experiment/smd_thesis_multitask_redlamp_multiclass_window20_nobootstrap.yaml:17` - monitor metric is `val_synth_vus_pr`
- `configs/experiment/smd_thesis_multitask_redlamp_multiclass_window20_nobootstrap.yaml:24` - checkpoint monitor metric is `val_synth_vus_pr`
- `src/core/config.py:290` - scheduler monitor metric whitelist check begins
- `src/core/config.py:295` - current local whitelist includes `val_synth_vus_pr`
- `scripts/train.py:162` - monitor metric extracted from scheduler config
- `scripts/train.py:167` - `ReduceLROnPlateau` instantiated

## Pipeline Documentation
The training pipeline resolves experiment YAML references, merges overrides, and validates optimizer/scheduler contracts before any model or dataloader construction. This failure occurs in the contract-validation gate inside `validate_experiment_config`, which blocks execution prior to runtime training setup.

## Historical Context (from documents/)
`documents/design/idea.md` and `documents/design/design_starter.md` define the active thesis setting with window length `L = 20`, RedLamp-style multiclass anomaly handling, and explicit configuration-driven training behavior. The analyzed experiment file aligns with that current design context.

## Open Questions
- Which exact commit hash is deployed on the GPU server where the traceback was produced?
- Does the server worktree contain local uncommitted edits in `src/core/config.py` that reintroduced the older whitelist?
- Is the invoked path definitely `/root/bachelor-thesis-2026/src/core/config.py` from the expected branch?
