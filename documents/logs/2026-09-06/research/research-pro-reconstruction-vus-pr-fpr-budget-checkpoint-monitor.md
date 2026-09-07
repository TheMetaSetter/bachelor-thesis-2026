---
date: 2026-09-06 13:51:05 +07:00
researcher: OpenAI Codex
topic: "Detect code parts that must change to monitor VUS-PR@FPR-budget for pro-reconstruction checkpoint selection"
status: complete
revision: cc42b4e3faa433d9d81e33719dfab0cbbe1cee66
branch: dev
---

# Research: VUS-PR@FPR-budget checkpoint monitor

## Summary

The current code already computes budgeted VUS-PR, but it returns three nested values. The trainer can save a best checkpoint only from one scalar epoch metric. The required modification surface is therefore the metric-to-epoch-metric boundary, checkpoint-monitor validation, and experiment-config generation. The reconstruction MSE path already uses normalized input-output values and does not need this metric-monitor change.

The active tree has no file or configuration named `pro-reconstruction`. The closest active experiment is the `O1_recon075_cls025` two-stage offline configuration. This mapping is an inference, not a confirmed canonical name.

## Research question

Read `prompts/1_research_prompt.md` and detect the code parts that need modification to run the `pro-reconstruction` experiment while monitoring `VUS-PR@FPR-budget` for best-checkpoint saving.

## System context

The offline benchmark entry point loads an experiment config and delegates two-stage training to `scripts/experiments/run_two_stage_offline_pretraining.py`. That runner creates Stage-A and Stage-B configs by copying the parent config. The training CLI passes `checkpoint_monitor_metric` into `Trainer`. The evaluator and trainer both call `compute_pointwise_metrics` for validation metrics.

The closest active reconstruction-focused model config sets `lambda_recon: 0.75`, `lambda_cls: 0.25`, and enables the point-score loss. The model's reconstruction-loss boundary defaults to `normalized_input` and computes squared error between `outputs["recon"]` and `batch["x"]` in that space.

## Execution path

1. The offline benchmark loads the resolved experiment config.
2. The two-stage runner copies the parent config into Stage A and Stage B and preserves `checkpoint_monitor_metric`.
3. The trainer performs validation and calls `compute_pointwise_metrics` on timeline-reconstructed point scores.
4. `compute_pointwise_metrics` adds `vus_pr_at_fpr_budget` as a mapping keyed by the three fixed budgets.
5. The trainer prefixes that mapping into epoch metrics, while also exposing the ordinary scalar `val_synth_vus_pr`.
6. The trainer resolves an allowed monitor name, casts the selected epoch value to `float`, compares it in `max` mode, and saves `best.pt` when it improves.

The current path cannot use the nested budget mapping directly at step 6.

## Detailed findings

### 1. Budgeted VUS implementation already exists

`src/metrics/pointwise.py` defines the fixed budgets `(0.001, 0.005, 0.01)`. `compute_budgeted_vus_metrics` calculates constrained VUS-PR and normalized partial VUS-ROC for every budget and returns nested dictionaries. `compute_pointwise_metrics` places the VUS-PR mapping under `vus_pr_at_fpr_budget`.

This code is currently an uncommitted working-tree change. The existing test checks the three keys and a separable-score result, but it does not test selecting one scalar budget value for checkpoint monitoring.

### 2. Trainer is the primary runtime modification surface

`Trainer._aggregate_reconstructed_pointwise_metrics` receives the nested mapping and currently emits it as a nested epoch metric named `val_synth_vus_pr_at_fpr_budget_pointwise`. It emits the ordinary scalar `val_synth_vus_pr` separately.

`Trainer._resolve_best_checkpoint_monitor` whitelists only scalar monitor names and does not include a budget-specific name. The checkpoint loop then calls `float(epoch_metrics[best_checkpoint_monitor_metric])`. A nested mapping would fail here, so a selected budget must be exposed as a scalar before checkpoint resolution.

### 3. Config validation must recognize the scalar monitor name

`src/core/config.py` validates `checkpoint_monitor_metric` against a fixed set that does not include a budget-specific VUS-PR name. The same file validates `optimizer.scheduler.monitor_metric` for `reduce_on_plateau`. If that scheduler is ever paired with the new monitor, its whitelist must also recognize the same scalar name.

The current reconstruction matrix uses a cosine scheduler, so the scheduler branch is not part of the current runtime path. It remains a compatibility surface if the new metric is reused with `reduce_on_plateau`.

### 4. Matrix/config generation must select the monitor

`scripts/benchmarks/generate_full_direct_recon075_cls025_matrix.py` generates all 18 direct-routing configs but currently inherits the old `val_synth_vus_pr` monitor from each source config. The generated configs therefore cannot request a budget-specific scalar monitor until the generator writes the new canonical metric name.

The active `O1_recon075_cls025` main config also currently sets `checkpoint_monitor_metric: val_synth_vus_pr`. If this is the intended `pro-reconstruction` run, its config must select the same scalar budget monitor as the generated matrix.

`scripts/experiments/run_two_stage_offline_pretraining.py` copies the parent monitor into both generated stage configs. It is not the owner of the monitor policy, but its generated Stage-A and Stage-B configs must be verified after the parent/generator change.

### 5. Checkpoint metadata needs a scalar value

`src/engine/thresholding.py` stores the monitor name and monitor value in checkpoint evaluation metadata and casts the value to `float`. This path is compatible once the trainer provides a scalar budget-specific epoch metric. It must be regression-tested because a nested value would violate the cast contract.

### 6. Tests lack the required scalar-monitor contract

`tests/evaluation/test_budgeted_vus_metrics.py` covers the budgeted metric family but not the epoch-metric name or scalar extraction. `tests/benchmarks/test_benchmark_config_generation.py` currently expects `val_synth_vus_pr`, so its expectation must change for the new experiment configs.

Trainer checkpoint tests under `tests/runtime/` cover scalar monitor selection and saving, but no test currently proves that a budget-specific VUS-PR value determines the selected epoch and saved `best.pt`.

### 7. Reconstruction MSE path already matches the experiment requirement

The active reconstruction model config sets the loss weights and point-score settings. `ThesisMultitaskModel.reconstruction_squared_error` computes `(reconstruction - target).square()` when `reconstruction_loss_space` is `normalized_input`. The loss core averages this squared error, with the configured clean-position mask when `reconstruction_normal_only` is enabled.

No change to this MSE path is required solely to add VUS-PR@FPR-budget checkpoint monitoring.

### 8. Offline benchmark evaluation is not the checkpoint-monitor owner

`scripts/benchmarks/run_thesis_offline_benchmark.py` uses the evaluator for final clean-validation, synthetic-validation, and test metrics and exports the nested budgeted metrics. It does not resolve or save the training best checkpoint. No direct modification in this file is established by the current question.

## Evidence

- `src/metrics/pointwise.py:23` — fixed FPR budgets are `0.001`, `0.005`, and `0.01`.
- `src/metrics/pointwise.py:404-485` — budgeted VUS-PR/VUS-ROC are computed for all budgets and returned as nested mappings.
- `src/metrics/pointwise.py:708-778` — `compute_pointwise_metrics` exposes `vus_pr_at_fpr_budget` as a nested mapping.
- `src/engine/trainer.py:513-569` — validation pointwise metrics are prefixed into epoch metrics; only ordinary `vus_pr` is flattened to `stage_vus_pr`.
- `src/engine/trainer.py:384-405` — checkpoint monitor names are whitelisted and mapped to `max`/`min` modes.
- `src/engine/trainer.py:896-903` — checkpoint selection requires the monitor key and casts its value to `float`.
- `src/core/config.py:403-415` — experiment checkpoint monitor validation excludes budget-specific names.
- `src/core/config.py:452-470` — ReduceLROnPlateau monitor validation uses another fixed whitelist.
- `src/engine/thresholding.py:98-127` — checkpoint metadata records a scalar monitor value.
- `scripts/benchmarks/generate_full_direct_recon075_cls025_matrix.py:48-109` — matrix loss/fusion overrides are set, but checkpoint monitor is inherited.
- `scripts/experiments/run_two_stage_offline_pretraining.py:115-170` — Stage-A and Stage-B configs copy the parent monitor.
- `configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O1_recon075_cls025__machine_3_4__w20__seed36__main.yaml:21` — active reconstruction-focused main config still monitors `val_synth_vus_pr`.
- `configs/model/thesis_multitask_two_stage_point_score_window20_recon075_cls025.yaml:48-60` — reconstruction/classification weights and point-score settings.
- `src/models/thesis_multitask.py:93-125` — normalized-input reconstruction MSE branch.
- `src/models/thesis_multitask_impl/thesis_multitask_loss_core_mixin.py:30-52` — reconstruction loss reduction and clean-position masking.
- `tests/evaluation/test_budgeted_vus_metrics.py:13-34` — current budgeted VUS tests.
- `tests/benchmarks/test_benchmark_config_generation.py:50-52` — current generator test expects the old monitor.
- `documents/notes/vus-metrics-at-fpr-budgets.md:2-36` — documented algorithm accepts one `Budget`, while the current implementation evaluates the three locked budgets.
- `documents/spec/two-stage-offline-pretraining-spec.md:1604-1710` — validation may select checkpoints, and VUS metrics do not require a threshold.
- `documents/spec/two-stage-offline-pretraining-spec.md:1820-1834` — validation logging includes VUS-PR but does not define one FPR budget as the checkpoint monitor.

## Configuration observed

| Setting | Active value | Evidence | Scope |
| --- | --- | --- | --- |
| `FPR_BUDGETS` | `0.001`, `0.005`, `0.01` | `src/metrics/pointwise.py:23` | Metric implementation |
| `checkpoint_monitor_metric` | `val_synth_vus_pr` | `configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O1_recon075_cls025__machine_3_4__w20__seed36__main.yaml:21` | Active main config |
| `lambda_recon` | `0.75` | `configs/model/thesis_multitask_two_stage_point_score_window20_recon075_cls025.yaml:48` | Reconstruction-focused model |
| `lambda_cls` | `0.25` | `configs/model/thesis_multitask_two_stage_point_score_window20_recon075_cls025.yaml:49` | Reconstruction-focused model |
| `reconstruction_loss_space` default | `normalized_input` | `src/models/thesis_multitask.py:77,109-125` | Model loss boundary |
| Matrix generator monitor | inherited from source config | `scripts/benchmarks/generate_full_direct_recon075_cls025_matrix.py:70-95` | 18 generated configs |
| Stage monitor propagation | copied from parent config | `scripts/experiments/run_two_stage_offline_pretraining.py:119-158` | Stage A and Stage B |

## Conflicts and uncertainties

- `VUS-PR@FPR-budget` is a family of three scalar values in the active implementation, not one metric. The available files do not choose `0.001`, `0.005`, or `0.01` for checkpoint selection.
- The active tree contains no literal `pro-reconstruction` identifier. The likely mapping to `O1_recon075_cls025` is inferred from filenames and the historical commit message `add smoke config for pro-reconstruction run`; it is not declared in an active config.
- The budgeted implementation, note, and test are uncommitted working-tree files. Their behavior is current in this checkout but not represented by the recorded revision alone.
- The development specification names ordinary VUS-PR for validation logging and checkpointing but does not define the new budget-specific monitor contract.

## Open questions

1. Which single FPR budget should control `best.pt`: `0.001`, `0.005`, or `0.01`?
2. Is `pro-reconstruction` exactly the active `O1_recon075_cls025` experiment, or should it receive a separate canonical experiment/config name?
3. Should the selected budget-specific scalar be used for both Stage A and Stage B, or only for one stage?
