---
date: 2026-06-30 23:42:09 +07
researcher: TheMetaSetter
git_commit: 0fd6abfb39ed704bf41a83845ec826b577ebdd94
branch: dev
repository: bachelor-thesis-2026
topic: "Synthetic anomaly ratios in train and validation, and the computation path for val_synth_vus_pr"
tags: [research, synthetic-anomaly, val-synth, vus-pr, smd, benchmark]
status: complete
last_updated: 2026-06-30
last_updated_by: TheMetaSetter
---

# Research: Synthetic anomaly ratios in train and validation, and the computation path for `val_synth_vus_pr`

**Date**: 2026-06-30 23:42:09 +07  
**Researcher**: TheMetaSetter  
**Git Commit**: `0fd6abfb39ed704bf41a83845ec826b577ebdd94`  
**Branch**: `dev`

## Research Question

What synthetic anomaly ratio is currently used in the active benchmark train and validation paths, and how is `val_synth_vus_pr` computed in the active codebase?

## Summary

In the active SMD benchmark configs, both `redlamp_baseline` and `thesis_multitask` use the same task config, so they inherit the same synthetic augmentation policy for `train` and `val_synth`. That policy enables balanced multiclass synthetic augmentation with 12 window classes: 1 normal class and 11 anomaly classes. Because balancing is done at the window level, not at the point level, about `11/12` of synthetic windows are anomalous. Each anomalous window then receives one contiguous anomaly segment whose sampled length is between `20%` and `30%` of the window length. With `window_size = 20`, this means segment lengths of 4, 5, or 6 timesteps.

As a result, the effective anomaly point ratio is not `0.5`, even though the config still contains `anomaly_probability: 0.5`. In the active balanced branch, that Bernoulli probability is ignored. The actual synthetic point ratio observed in the benchmark audit is about `22.8%` for both `train` and `val_synth`.

The metric `val_synth_vus_pr` is computed from synthetic validation point scores and synthetic validation point labels after reconstructing overlap-averaged window scores back onto the timeline. The trainer then selects a threshold at the `0.95` quantile of covered point scores, computes pointwise metrics, and computes VUS-PR by averaging range-aware PR-AUC values across buffer sizes from `0` to `vus_max_buffer_size`.

## Detailed Findings

### Data Preparation

The active benchmark configs for both methods point to the same task config file: `configs/task/multitask_tsad_redlamp_multiclass_window20_benchmark_fixed_synth.yaml`. That task config sets:

- `use_synthetic_augmentation: true`
- `use_synthetic_validation: true`
- `synthetic_train_seed: 7`
- `synthetic_validation_seed: 7`
- `classification_label_mode: redlamp_multiclass`
- `anomaly_probability: 0.5`
- `train_balance_classes: true`
- `min_segment_fraction: 0.2`
- `max_segment_fraction: 0.3`

The benchmark baseline config and the benchmark thesis config both reference that task config and therefore share the same synthetic policy.

In `SyntheticAnomalyInjector`, when `train_balance_classes` is `true`, the branch that uses Bernoulli anomaly sampling is skipped. Instead, the injector builds a near-uniform class quota over the active class set. In multiclass mode, the active class indices are:

- class `0`: `normal`
- classes `1..11`: the 11 RedLamp anomaly families

So the active training and validation synthetic policy is:

1. Balance windows across 12 classes.
2. Mark approximately `11/12` of windows as anomalous.
3. For each anomalous window, sample one contiguous segment.
4. Segment length is sampled from the discrete set `{4, 5, 6}` because:
   - `min_segment_length = int(20 * 0.2) = 4`
   - `max_segment_length = int(20 * 0.3) = 6`

That gives an expected segment-length fraction:

$$
\mathbb{E}\left[\frac{\text{segment length}}{20}\right]
=
\frac{4 + 5 + 6}{3 \cdot 20}
=
\frac{5}{20}
=
0.25
$$

At the window level, the expected anomalous-window fraction is:

$$
\frac{11}{12} \approx 0.9167
$$

So the rough expected point-level anomaly ratio is:

$$
\frac{11}{12} \times 0.25 \approx 0.2292
$$

This matches the measured benchmark audit numbers closely.

Measured ratios already documented in the benchmark audit:

- `machine-1-6`, `train`: `1736 / 1894` anomalous windows, anomaly point ratio `8652 / 37880 = 0.22841`
- `machine-1-6`, `val`: `4324 / 4718` anomalous windows, anomaly point ratio `21531 / 94360 = 0.22818`
- `machine-3-4`, `train`: same as `machine-1-6` because the active benchmark split has the same train-window count
- `machine-3-4`, `val`: same as `machine-1-6` because the active benchmark split has the same validation-window count
- `machine-3-9`, `train`: anomaly point ratio `10451 / 45920 = 0.22759`
- `machine-3-9`, `val`: anomaly point ratio `26129 / 114460 = 0.22828`

These values show that the practical synthetic anomaly ratio in both `train` and `val_synth` is about `22.8%` at the timestep level.

### Modeling and Training

The baseline model builds two injectors:

- `self.synthetic_anomaly_injector` for `train`
- `self.synthetic_validation_injector` for `val_synth`

Both are configured from the same synthetic settings, with deterministic seeds `synthetic_train_seed` and `synthetic_validation_seed`.

The thesis model does the same thing. It also builds:

- `self.synthetic_anomaly_injector`
- `self.synthetic_validation_injector`

from the same task-level synthetic config.

The trainer calls `prepare_synthetic_validation_epoch()` before each `val_synth` epoch. Because the active task config fixes `synthetic_validation_seed: 7`, the synthetic validation corruption is deterministic across epochs within one run.

### Evaluation

The active `val_synth_vus_pr` path works as follows.

First, during the synthetic validation epoch, the trainer stores payloads containing:

- `meta`
- `point_scores`
- synthetic point labels from `synthetic_anomaly_mask`

Then the trainer reconstructs entity timelines from overlapping windows by summing scores into the covered interval and dividing by the number of covering windows at each timestep:

$$
s_t
=
\frac{\sum_{w \ni t} s_{w,t}}{\sum_{w \ni t} 1}
$$

where:

- \(s_{w,t}\) is the point score assigned by window \(w\) to timestep \(t\),
- the sum runs over all windows that cover timestep \(t\).

For labels, the reconstructed point label at a timestep is the maximum over overlapping window labels:

$$
y_t = \max_{w \ni t} y_{w,t}
$$

After reconstruction, only covered timesteps are kept. The trainer concatenates covered scores and covered labels across entities into one evaluation vector.

The threshold used for thresholded pointwise metrics is selected as the `0.95` quantile of the covered point scores:

$$
\tau = \mathrm{Quantile}_{0.95}(S_{\text{covered}})
$$

with one extra rule: if there are positive scores, the quantile is computed on positive scores only.

Binary predictions are then:

$$
\hat y_t = \mathbf{1}[s_t > \tau]
$$

The exported `val_synth_vus_pr` is the `vus_pr` field returned by `compute_pointwise_metrics(...)`.

Inside `compute_vus_pr_exact_naive(...)`, the repository computes a range-aware PR area for every buffer size:

$$
b \in \{0, 1, 2, \dots, B_{\max}\}
$$

where `B_max = vus_max_buffer_size`.

For each threshold \(\theta\) in a grid built from the score range, predictions are:

$$
\hat y_t(\theta) = \mathbf{1}[s_t > \theta]
$$

The repository then builds range-aware labels \(\tilde y_t^{(b)}\) by softening labels around anomaly boundaries when nearby predictions exist. Using those softened labels, it computes:

$$
\mathrm{TP} = \sum_t \tilde y_t^{(b)} \hat y_t(\theta)
$$

$$
\mathrm{FP} = \sum_t \left(1 - \tilde y_t^{(b)}\right)\hat y_t(\theta)
$$

$$
\mathrm{Precision}^{(b)}(\theta)
=
\begin{cases}
1, & \text{if } \mathrm{TP} + \mathrm{FP} = 0 \\
\frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}, & \text{otherwise}
\end{cases}
$$

The recall is multiplied by an existence reward:

$$
\mathrm{Recall}^{(b)}(\theta)
=
\frac{\mathrm{TP}}{\mathrm{PositiveMass}^{(b)}} \cdot \mathrm{ExistenceReward}
$$

where:

- \(\mathrm{PositiveMass}^{(b)}\) is the positive mass of the softened range labels,
- \(\mathrm{ExistenceReward}\) is the fraction of true anomaly ranges for which at least one positive prediction exists inside the range.

From all threshold points for a fixed buffer size \(b\), the code computes one range-aware PR area:

$$
\mathrm{AP}^{(b)}
=
\int \mathrm{Precision}^{(b)} \, d(\mathrm{Recall}^{(b)})
$$

implemented numerically as a trapezoidal area after monotonic envelope correction.

Finally, VUS-PR is the arithmetic mean across all buffer sizes:

$$
\mathrm{VUS\text{-}PR}
=
\frac{1}{B_{\max}+1}
\sum_{b=0}^{B_{\max}} \mathrm{AP}^{(b)}
$$

In the active benchmark configs:

- `vus_max_buffer_size = 20`
- `vus_num_thresholds = 200`

So `val_synth_vus_pr` is the mean of these range-aware PR-AUC values across buffer sizes `0..20`, using a threshold grid built from the reconstructed synthetic validation point scores.

## Code References

- `configs/task/multitask_tsad_redlamp_multiclass_window20_benchmark_fixed_synth.yaml:1` - active synthetic task settings shared by benchmark configs
- `configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_9__w20__seed6__main.yaml:9` - baseline benchmark config points to shared task config
- `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-three-stage-machine_3_9__w20__seed6__main.yaml:9` - thesis benchmark config points to shared task config
- `src/data/augment.py:41` - synthetic injector constructor arguments
- `src/data/augment.py:203` - segment-length sampling from `min_segment_fraction` and `max_segment_fraction`
- `src/data/augment.py:813` - class-balanced sampling branch
- `src/data/augment.py:866` - batch augmentation entrypoint
- `src/data/augment.py:728` - single-window anomaly injection and point-mask construction
- `src/models/redlamp_baseline.py:252` - baseline train synthetic injector
- `src/models/redlamp_baseline.py:265` - baseline validation synthetic injector
- `src/models/redlamp_baseline.py:306` - baseline uses synthetic injector in `train`
- `src/models/redlamp_baseline.py:308` - baseline uses synthetic validation injector in `val_synth`
- `src/models/thesis_multitask.py:1044` - thesis train synthetic injector
- `src/models/thesis_multitask.py:1055` - thesis validation synthetic injector
- `src/engine/trainer.py:705` - trainer resets deterministic synthetic validation RNG each epoch
- `src/engine/trainer.py:717` - `val_synth` pointwise labels come from `synthetic_anomaly_mask`
- `src/engine/trainer.py:526` - reconstruct pointwise timelines from window payloads
- `src/engine/trainer.py:533` - threshold selected at score quantile `0.95`
- `src/engine/trainer.py:559` - exported metric name `val_synth_vus_pr`
- `src/engine/evaluator.py:24` - threshold selection helper
- `src/engine/evaluator.py:121` - overlap reconstruction logic
- `src/engine/evaluator.py:170` - covered-point extraction logic
- `src/metrics/pointwise.py:222` - range-aware precision and recall helper
- `src/metrics/pointwise.py:381` - VUS-PR exact naive computation
- `src/metrics/pointwise.py:542` - pointwise metric bundle exposing `vus_pr`
- `documents/logs/06-30-2026/research/research-final-pre-benchmark-audit.md:58` - measured synthetic anomaly ratios for active benchmark entities

## Pipeline Documentation

For the active SMD benchmark:

1. Load clean train and validation windows from the dataset split.
2. In `train`, inject synthetic anomalies window-by-window using the train injector.
3. In `val_synth`, inject synthetic anomalies over the validation windows using the validation injector with deterministic seed reset.
4. The model outputs `point_scores` per window.
5. The trainer reconstructs overlap-averaged pointwise scores back onto the entity timeline.
6. Covered point labels for `val_synth` come from `synthetic_anomaly_mask`.
7. A threshold is selected from reconstructed covered point scores.
8. `compute_pointwise_metrics(...)` returns scalar pointwise metrics including `vus_pr`.
9. The trainer logs that value under `val_synth_vus_pr`.

## Historical Context (from documents/)

The benchmark audit note from 2026-06-30 already documented that the active synthetic benchmark setup is balanced over 12 classes at the window level, which implies an anomaly point ratio around `22.8%` rather than a rare-event ratio such as `1%`, `3%`, or `5%`. The current research note confirms that this behavior matches the active code path and explains the formulas used by the runtime.

## Open Questions

1. The current synthetic ratio is clear in the code and in the audit note, but whether this ratio is scientifically appropriate for the final thesis benchmark is a separate design question, not an implementation question.
2. The current VUS-PR implementation is an in-repository range-aware implementation. If external benchmark comparability is later required, the exact equivalence to another public VUS implementation would need a separate audit.
