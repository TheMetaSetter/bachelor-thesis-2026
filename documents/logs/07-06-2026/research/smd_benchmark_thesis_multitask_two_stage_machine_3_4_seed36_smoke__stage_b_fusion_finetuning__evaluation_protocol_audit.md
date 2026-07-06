# Evaluation Protocol Audit: smd_benchmark_thesis_multitask_two_stage_machine_3_4_seed36_smoke__stage_b_fusion_finetuning

- Dataset: `smd`
- Scaler fit scope: `train_only_before_windowing`
- Benchmark comparability: `non_comparable`
- Protocol status: `truncated_smoke_evaluation`

## Split Summary

### train

- Windows: 16
- Label regime: `all_zero`
- Positive ratio: 0.000000
- Evaluated points: 35/18950
- Truncated coverage: `True`
- Truncation reason: `max_window_cap`

### val

- Windows: 8
- Label regime: `all_zero`
- Positive ratio: 0.000000
- Evaluated points: 160/4737
- Truncated coverage: `True`
- Truncation reason: `max_window_cap`

### test

- Windows: 16
- Label regime: `mixed`
- Positive ratio: 0.041246
- Evaluated points: 320/23687
- Truncated coverage: `True`
- Truncation reason: `max_window_cap`

## Benchmark Protocol Status

A benchmark-comparable run must evaluate a future test timeline that contains both normal and anomalous timesteps after reconstruction.

- Benchmark comparability: `non_comparable`
- Protocol status: `truncated_smoke_evaluation`

## Warnings

- Test split window coverage is truncated relative to the raw test timeline. Evaluation artifacts do not cover the full labeled timeline.
- Configured max_test_windows=16 truncates the evaluated test timeline. Treat this as a truncated smoke evaluation, not a full-timeline test.
- Reconstructed evaluation labels still contain only one class after window aggregation.

## Evaluation Coverage

- Evaluated points: 320/23687
- Truncated evaluation artifact: `True`

## Metric Regime Interpretation

The evaluated test vector contains both normal and anomalous labels, so the pointwise metrics are at least defined in the usual binary sense.

- The remaining question is then whether evaluated coverage truly matches the intended raw test timeline.

- The observed threshold was 3.270198.
