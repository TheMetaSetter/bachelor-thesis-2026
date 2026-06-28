# Research: Runtime Estimation and Console/Progress Instrumentation for Comparative SMD Experiments

## Metadata
- Date: 2026-06-25 19:20:39 +0700
- Author: TheMetaSetter
- Git commit: `89a598f643cf0c20b0ab540b926e6b71f27e975f`
- Branch: `dev`

## Research Question
Estimate the total runtime of the planned comparative experiment and trace the exact files and code paths that currently produce excessive console output or need to change to support a minimal console UI with realtime progress and ETA.

## Scope
- Comparative runner:
  - `src/models/thesis_multitask.py`
  - `src/models/redlamp_mlp_baseline.py`
  - `scripts/run_three_stage_offline_pretraining.py`
  - `scripts/run_comparative_smd_experiments.py`
- Training/evaluation pipeline:
  - `src/engine/trainer.py`
  - `src/engine/evaluator.py`
  - `src/core/console.py`
  - `src/data/loaders.py`
  - `src/data/collate.py`
  - `src/data/augment.py`

## Executive Summary
The current codebase is structurally capable of providing a clean console plus realtime progress/ETA, but the main bottleneck is not missing infrastructure. The main problem is that many batch-level `console_print(...)` calls are still active in trainer, evaluator, collate, augmentation, and the thesis multitask model. The simplest and safest cleanup path is:

1. add a global verbosity gate in `src/core/console.py`;
2. suppress batch-level prints by default;
3. insert progress bars at trainer/evaluator/orchestrator level instead of model/data internals;
4. keep only stage-level and epoch-level summaries in the console;
5. surface realtime ETA from measured loop timings rather than guessed static estimates.

For the planned 18-run comparative experiment (`2 methods x 3 entities x 3 seeds`), a structure-based estimate gives roughly `18-34 hours`, with a more plausible middle band around `22.5-27 hours` on an RTX 3090 if average effective batch-like pass time stays around `0.10-0.12s`. This is a code-structure estimate, not yet a measured RTX 3090 benchmark.

## Findings

### 1. Current experiment size

The comparative launcher is `scripts/launch_tmux_comparative_smd_experiment.sh`, which delegates to `scripts/run_comparative_smd_experiments.py`. That orchestrator currently enumerates the top-3 SMD entities and three seeds for both methods, producing 18 full runs total.

Relevant code:
- `scripts/launch_tmux_comparative_smd_experiment.sh`
- `scripts/run_comparative_smd_experiments.py`

### 2. Runtime-critical dataset sizes

Using the current SMD configs (`window_size=20`, `stride=1`, `batch_size=256`, `validation_split_ratio=0.2`), the active per-entity window counts are:

#### `machine-1-6`
- train sequence length: `23688`
- test sequence length: `23689`
- train windows: `18932`
- val windows: `4718`
- test windows: `23670`
- train batches: `74`
- val batches: `19`
- test batches: `93`

#### `machine-3-1`
- train sequence length: `28700`
- test sequence length: `28700`
- train windows: `22941`
- val windows: `5721`
- test windows: `28681`
- train batches: `90`
- val batches: `23`
- test batches: `113`

#### `machine-3-9`
- train sequence length: `28713`
- test sequence length: `28713`
- train windows: `22952`
- val windows: `5723`
- test windows: `28694`
- train batches: `90`
- val batches: `23`
- test batches: `113`

These counts matter because the current trainer does more than one validation-style pass per epoch.

Relevant code:
- `src/data/loaders.py:49-107`

### 3. Per-epoch pass structure is heavier than it looks

The current `Trainer.train()` loop does:

1. one full training pass over `train_loader`;
2. one clean validation pass through `_run_validation_epoch(...)`;
3. one `val_realistic` pass if the model exposes `realistic_validation_step(...)`;
4. one more pass through `Evaluator.evaluate(model, val_loader)` if `validation_evaluator_config` is present.

So each epoch currently performs:

#### `machine-1-6`
- `74 + 19 + 19 + 19 = 131` batch-like passes per epoch

#### `machine-3-1`
- `90 + 23 + 23 + 23 = 159` batch-like passes per epoch

#### `machine-3-9`
- `90 + 23 + 23 + 23 = 159` batch-like passes per epoch

Relevant code:
- `src/engine/trainer.py:584-845`
- `src/engine/trainer.py:698-723`
- `src/engine/trainer.py:802-815`

### 4. Approximate total batch-like work for the comparative run

Per single baseline run:

#### `machine-1-6`
- `131 x 300 + 93 = 39,393`

#### `machine-3-1`
- `159 x 300 + 113 = 47,813`

#### `machine-3-9`
- `159 x 300 + 113 = 47,813`

Across all 9 baseline runs:
- `405,057` batch-like passes

The thesis three-stage method has the same 300-epoch trainer body plus a small fixed overhead:
- Stage 2 activation-signature capture: about `16` encoder passes total (`8` batches per source model);
- Stage 3 memory initialization: about `32` more encoder-oriented batch passes in rough practice when `memory_initialization_batches=16`.

So the thesis total is only slightly larger:
- all 9 thesis runs: about `405,489` batch-like passes

Combined rough total:
- about `810,546` batch-like passes

Relevant code:
- `scripts/run_three_stage_offline_pretraining.py:50`
- `scripts/run_three_stage_offline_pretraining.py:344-408`
- `src/models/thesis_multitask.py:1390-1605`

### 5. Runtime estimate

If the server sustains the following average effective time per batch-like pass:

- `0.08s` -> about `18.0h`
- `0.10s` -> about `22.5h`
- `0.12s` -> about `27.0h`
- `0.15s` -> about `33.8h`

So a reasonable structural estimate is:

- optimistic: about `18h`
- practical middle range: about `22.5-27h`
- pessimistic but still plausible: about `34h`

This estimate excludes unusual slowdowns such as dataloader stalls, W&B backpressure, checkpoint I/O spikes, or thermal throttling.

### 6. Main console noise sources

The current console spam is dominated by batch-level prints.

#### Central print entrypoint
- `src/core/console.py:47-61`

`console_print(...)` currently prints unconditionally and has no verbosity threshold.

#### Trainer batch prints
- `src/engine/trainer.py:417-451`
- `src/engine/trainer.py:620-665`

Current noisy messages include:
- `Processing validation batch`
- `Completed validation batch`
- `Processing training batch`
- `Completed optimizer step`

#### Evaluator batch prints
- `src/engine/evaluator.py:156-184`
- `src/engine/evaluator.py:204-279`

Current noisy messages include:
- `Evaluating batch`
- `Produced evaluation batch outputs`

#### Data-layer batch prints
- `src/data/collate.py:12-44`
- `src/data/augment.py:939-947`

These currently emit per-batch collation/augmentation summaries, which are not console-friendly during long runs.

#### Model-level batch prints
- `src/models/thesis_multitask.py:2467-2481`
- `src/models/thesis_multitask.py:3194-3210`

These are especially costly from a readability perspective because they fire inside the hot path of train/val/test.

### 7. Best insertion points for progress bars and ETA

The safest place to add progress bars is not inside the model. It is at the outer loops:

#### Epoch/train progress
- `src/engine/trainer.py:620-665`

This is the right place for a per-epoch training progress bar because:
- the loop knows `epoch`, `max_epochs`, `batch_index`, and total batches;
- the optimizer step and scheduler step are already centralized there.

#### Validation progress
- `src/engine/trainer.py:417-451`
- `src/engine/evaluator.py:204-279`

This is the right place for progress bars for:
- clean validation;
- realistic validation;
- metric evaluation passes.

#### Three-stage orchestration progress
- `scripts/run_three_stage_offline_pretraining.py`

This is the right place for a high-level stage progress UI:
- Stage 1 classification pretraining
- Stage 1 reconstruction pretraining
- Stage 2 zipping
- optional Stage 2 recovery
- Stage 3 memory initialization + warm-up
- main multitask pretraining
- final test

#### Comparative run progress
- `scripts/run_comparative_smd_experiments.py`

This is the right place for an outermost run-level progress bar:
- current method
- current entity
- current seed
- completed runs out of 18
- global ETA

### 8. Existing timing hooks that can be reused

The codebase already records some timing-oriented signals:

- `src/engine/evaluator.py:138-151`
  - batch forward-pass timing accumulation
- `src/engine/evaluator.py:271`
  - `forward_pass_seconds_mean`
- `src/engine/trainer.py:274-299`
  - scheduler progress already computes fractional training progress

This means ETA support can be implemented with small, local additions rather than a large redesign.

## Code References

### Console and logging
- `src/core/console.py:47-61`
- `src/data/loaders.py:24-31`
- `src/data/loaders.py:75-107`
- `src/data/collate.py:12-44`
- `src/data/augment.py:939-947`
- `src/models/thesis_multitask.py:2467-2481`
- `src/models/thesis_multitask.py:3194-3210`

### Trainer/evaluator loops
- `src/engine/trainer.py:417-451`
- `src/engine/trainer.py:584-845`
- `src/engine/evaluator.py:156-279`

### Three-stage overhead
- `scripts/run_three_stage_offline_pretraining.py:50`
- `scripts/run_three_stage_offline_pretraining.py:344-408`
- `scripts/run_three_stage_offline_pretraining.py:503-508`
- `src/models/thesis_multitask.py:1390-1605`

## Implications for a Minimal, Safe Refactor

The least risky plan is:

1. add a verbosity level or `quiet_console` flag in `src/core/console.py`;
2. default all batch-level debug prints to hidden;
3. keep stage/epoch summaries visible;
4. introduce progress bars only in trainer/evaluator/orchestrator loops;
5. keep file logging/W&B logging intact even if console output is minimized.

This preserves:
- experiment reproducibility;
- W&B tracking;
- existing control flow;
- baseline/thesis fairness.

It also avoids a messy anti-pattern where progress state is fragmented across model, dataloader, and evaluator internals.

## Recommended Next Implementation Targets

### First-priority files
- `src/core/console.py`
- `src/engine/trainer.py`
- `src/engine/evaluator.py`

### Second-priority cleanup files
- `src/data/collate.py`
- `src/data/augment.py`
- `src/models/thesis_multitask.py`

### Optional outer-run UX files
- `scripts/run_three_stage_offline_pretraining.py`
- `scripts/run_comparative_smd_experiments.py`

## Conclusion

The codebase already has the right structural boundaries for a clean progress-aware console. The current problem is primarily uncontrolled batch-level printing. The simplest robust fix is to centralize verbosity control in `src/core/console.py`, then add progress/ETA at trainer/evaluator/orchestrator boundaries. Based on current loop structure and window counts, the full 18-run comparative experiment is likely to take about `18-34 hours`, with `22.5-27 hours` as the most practical current estimate before a real RTX 3090 timing smoke run.
