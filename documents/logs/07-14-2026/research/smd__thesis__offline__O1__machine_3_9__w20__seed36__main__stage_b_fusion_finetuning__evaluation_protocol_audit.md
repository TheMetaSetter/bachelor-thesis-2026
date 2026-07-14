# Evaluation Protocol Audit: smd__thesis__offline__O1__machine_3_9__w20__seed36__main__stage_b_fusion_finetuning

- Dataset: `smd`
- Scaler fit scope: `train_only_before_windowing`
- Benchmark comparability: `non_comparable`
- Protocol status: `truncated_smoke_evaluation`

## Split Summary

### train

- Windows: 22952
- Label regime: `all_zero`
- Positive ratio: 0.000000
- Evaluated points: 22971/22971
- Truncated coverage: `False`
- Truncation reason: `None`

### val

- Windows: 287
- Label regime: `all_zero`
- Positive ratio: 0.000000
- Evaluated points: 5740/5742
- Truncated coverage: `True`
- Truncation reason: `window_stride_remainder`

### test

- Windows: 1435
- Label regime: `mixed`
- Positive ratio: 0.010553
- Evaluated points: 28700/28713
- Truncated coverage: `True`
- Truncation reason: `window_stride_remainder`

## Benchmark Protocol Status

A benchmark-comparable run must evaluate a future test timeline that contains both normal and anomalous timesteps after reconstruction.

- Benchmark comparability: `non_comparable`
- Protocol status: `truncated_smoke_evaluation`

## Warnings

- Test split window coverage is truncated relative to the raw test timeline. Evaluation artifacts do not cover the full labeled timeline.
- Current window size and stride leave an uncovered suffix on the raw test timeline. Use benchmark-comparable coverage such as test_stride=1 or another setting that still covers the full labeled timeline.

## Evaluation Coverage

- Evaluated points: 28700/28713
- Truncated evaluation artifact: `True`

## Metric Regime Interpretation

The evaluated test vector contains both normal and anomalous labels, so the pointwise metrics are at least defined in the usual binary sense.

- The remaining question is then whether evaluated coverage truly matches the intended raw test timeline.

- The observed threshold was 3.231342.
