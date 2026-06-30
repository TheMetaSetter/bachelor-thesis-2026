# Evaluation Protocol Audit: smd_comparative_redlamp_baseline_single_stage_machine_1_6_smoke_seed6

- Dataset: `smd`
- Scaler fit scope: `train_only_before_windowing`
- Benchmark comparability: `non_comparable`
- Protocol status: `truncated_smoke_evaluation`

## Split Summary

### train

- Windows: 64
- Label regime: `all_zero`
- Positive ratio: 0.000000
- Evaluated points: 83/18951
- Truncated coverage: `True`
- Truncation reason: `max_window_cap`

### val

- Windows: 32
- Label regime: `all_zero`
- Positive ratio: 0.000000
- Evaluated points: 51/4737
- Truncated coverage: `True`
- Truncation reason: `max_window_cap`

### test

- Windows: 64
- Label regime: `mixed`
- Positive ratio: 0.156528
- Evaluated points: 83/23689
- Truncated coverage: `True`
- Truncation reason: `max_window_cap`

## Benchmark Protocol Status

A benchmark-comparable run must evaluate a future test timeline that contains both normal and anomalous timesteps after reconstruction.

- Benchmark comparability: `non_comparable`
- Protocol status: `truncated_smoke_evaluation`

## Warnings

- Test split window coverage is truncated relative to the raw test timeline. Evaluation artifacts do not cover the full labeled timeline.
- Configured max_test_windows=64 truncates the evaluated test timeline. Treat this as a truncated smoke evaluation, not a full-timeline test.
- Reconstructed evaluation labels still contain only one class after window aggregation.

## Evaluation Coverage

- Evaluated points: 83/23689
- Truncated evaluation artifact: `True`

## Metric Regime Interpretation

The evaluated test vector contains both normal and anomalous labels, so the pointwise metrics are at least defined in the usual binary sense.

- The remaining question is then whether evaluated coverage truly matches the intended raw test timeline.

- The observed threshold was 1.526325.
