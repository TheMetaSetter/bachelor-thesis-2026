---
date: 2026-06-08 16:00:46 +0700
researcher: Codex
git_commit: 9a77fe5014bc3b1cbd8ddb2e3b5c80cadad24060
branch: dev
repository: bachelor-thesis-2026
topic: "Current codebase status for rerunning the RedLamp baseline with balanced class sampling"
tags: [research, time-series, anomaly-detection, multi-class]
status: complete
last_updated: 2026-06-08
last_updated_by: Codex
---

# Research: Current codebase status for rerunning the RedLamp baseline with balanced class sampling

**Date**: 2026-06-08 16:00:46 +0700  
**Researcher**: Codex  
**Git Commit**: 9a77fe5014bc3b1cbd8ddb2e3b5c80cadad24060  
**Branch**: dev

## Research Question
Use `prompts/1_research_prompt.md` to determine the current codebase state before planning a baseline rerun with `train_balance_classes=true` for RedLamp on SMD machine `2-1`.

## Summary
The repository already contains the synthetic augmentation mechanism needed for balanced class sampling, including multiclass class-quota logic and tests that validate balanced binary and balanced multiclass behavior inside `SyntheticAnomalyInjector`. The RedLamp baseline code also exposes a `cnn_simple` encoder path, so the rerun target can be read as a RedLamp baseline with a simple CNN encoder plus balanced class sampling. The configuration layer also accepts `train_balance_classes` for multitask experiments.

The runtime wiring now accepts the task-level `train_balance_classes` flag in `RedLampMLPBaseline` as an alias that forwards into the injector, and the rerun experiment configs point to the dedicated CNN model config plus the balanced task config. The existing MLP baseline configs remain unchanged.

## Detailed Findings

### Data Preparation
- The active synthetic augmentation surface is `src/data/augment.py`.
- Balanced sampling is implemented by `_balanced_class_quota()` and `_sample_class_labels()`, which construct a per-batch class quota when `self.train_balance_classes` is enabled.
- In multiclass mode, the injector uses the RedLamp taxonomy `normal + 11 anomaly classes` and rotates remainder allocation round-robin across classes.
- The batch augmentation path writes `classification_labels`, `classification_class_names`, `synthetic_anomaly_mask`, and `augmentation_metadata` into the output batch.

### Modeling and Training
- `scripts/train.py` merges `experiment_config["task"]` into `model_kwargs` before building the model.
- `src/core/config.py` validates `train_balance_classes` as a boolean for multitask-style experiment configs.
- `src/models/redlamp_mlp_baseline.py` does not expose `train_balance_classes` as a constructor argument.
- The RedLamp baseline now accepts `train_balance_classes` and folds it into the injector balance flag alongside `balance_binary_classes_within_batch`.
- The baseline constructor hard-codes `classification_label_mode="redlamp_multiclass"` for both injectors, so the active label taxonomy is already multiclass.
- The same baseline file also exposes `encoder_family: "cnn_simple"`, and the simple CNN encoder path keeps the hidden-state contract unchanged.
- A dedicated rerun model config and task config were added so the balanced CNN baseline can be launched without mutating the MLP baseline defaults.

### Evaluation
- The baseline experiment config monitors `val_realistic_vus_pr` and runs for `300` epochs.
- The current baseline experiment file still points at the shared multiclass task config with `train_balance_classes: false`.
- No existing baseline experiment YAML in `configs/experiment/baseline/` or `configs/experiment/scale/` was found with `train_balance_classes: true`.

## Code References
- `src/data/augment.py:722-774` - balanced class quota and label sampling logic.
- `src/data/augment.py:776-860` - batch augmentation path, label materialization, and logging.
- `src/models/redlamp_mlp_baseline.py:87-250` - baseline constructor, encoder setup, and injector wiring.
- `src/core/config.py:700-725` - validation of `train_balance_classes` in the task section.
- `scripts/train.py:51-78` - merging task config into model kwargs before model construction.
- `configs/task/multitask_tsad_redlamp_multiclass_window20.yaml:1-20` - current task config, including `train_balance_classes: false`.
- `configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml:1-30` - active baseline experiment YAML.
- `tests/test_synthetic_anomaly_injection.py:91-194` - balanced binary and balanced multiclass injector tests.
- `tests/test_config_loading.py:220-230` - config loading test that accepts `train_balance_classes: true`.

## Pipeline Documentation
The current pipeline already supports balanced synthetic augmentation at the injector level. The injector can produce either stochastic anomaly sampling or class-balanced sampling. In multiclass mode, balanced sampling is implemented as class quotas over the RedLamp taxonomy, followed by per-batch shuffling. The output batch then carries explicit class labels and anomaly masks for downstream training.

For the RedLamp baseline rerun, the practical status is:
1. The mechanism exists.
2. The config validator accepts the field.
3. The rerun experiment configs now enable it through a dedicated balanced task config.
4. The baseline model already supports a `simple CNN encoder` path, so the rerun uses that encoder family rather than the MLP default.

## Historical Context (from documents/)
- `documents/logs/05-30-2026/detail/detail-sampling-rules-train-balanced-val-realistic.md` and `documents/logs/05-31-2026/detail/detail-sampling-rules-train-balance-val-realistic-implementation.md` indicate that balanced training and realistic validation were already treated as separate config concerns in the repository.
- `documents/logs/06-07-2026/plan/plan-simple-cnn-backbone-for-redlamp-baseline-and-thesis-multitask.md` shows the RedLamp baseline is an active experimental surface with CNN-family changes already being considered, so the baseline path remains central to ongoing work.

## Open Questions
- None for the current rerun wiring. The remaining question is execution-level, not config-level: which of the new rerun configs should be launched first.
