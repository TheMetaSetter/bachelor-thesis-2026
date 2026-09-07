---
date: 2026-09-06 14:30:19 +07
researcher: OpenAI Codex
topic: "Rank all SMD entities by value-distribution shift from train to test"
status: complete
revision: cc42b4e3faa433d9d81e33719dfab0cbbe1cee66
branch: dev
---

# Research: SMD train-to-test value-distribution shift

## Summary

`machine-1-6` has the largest average marginal value-distribution shift under
the stated KL measure, followed by `machine-3-9` and `machine-3-1`.
The calculation uses all test points, including labeled anomalies.

## Research question

Calculate and rank the value-distribution shift between the train and test
series for every SMD entity using KL divergence. Suggest one other shift metric.

## Method

For each of the 28 entities and each of its 38 channels, the calculation:

1. loads the raw train series and the full raw test series;
2. makes 64 equal-width histogram bins over the combined train/test value range
   for that channel;
3. adds `1e-12` to every bin before normalization; and
4. computes `KL(P_test || P_train)`.

The entity score is the arithmetic mean over its 38 channel scores. This is a
marginal value-distribution measurement. It does not measure temporal order,
autocorrelation, or cross-channel dependence.

The auxiliary score is mean Jensen-Shannon divergence (JSD), computed from the
same channel histograms. JSD is symmetric and lies in `[0, ln(2)]` when using
natural logarithms. Higher values mean greater shift for both scores.

## KL ranking

| KL rank | Entity | Mean KL(test || train) | Mean JSD |
| ---: | --- | ---: | ---: |
| 1 | machine-1-6 | 4.5823 | 0.1565 |
| 2 | machine-3-9 | 4.3421 | 0.1222 |
| 3 | machine-3-1 | 4.2062 | 0.1143 |
| 4 | machine-3-5 | 2.0883 | 0.0550 |
| 5 | machine-3-2 | 1.4060 | 0.1000 |
| 6 | machine-3-11 | 1.3789 | 0.0289 |
| 7 | machine-3-10 | 1.3666 | 0.0960 |
| 8 | machine-1-3 | 1.1821 | 0.0617 |
| 9 | machine-1-1 | 1.0840 | 0.0626 |
| 10 | machine-2-8 | 1.0024 | 0.0347 |
| 11 | machine-3-4 | 0.9628 | 0.0268 |
| 12 | machine-1-4 | 0.9213 | 0.0681 |
| 13 | machine-3-8 | 0.8644 | 0.0464 |
| 14 | machine-3-7 | 0.8128 | 0.0237 |
| 15 | machine-2-9 | 0.7618 | 0.0462 |
| 16 | machine-2-5 | 0.7593 | 0.0220 |
| 17 | machine-1-8 | 0.6980 | 0.0579 |
| 18 | machine-2-3 | 0.5224 | 0.0435 |
| 19 | machine-2-7 | 0.4719 | 0.0226 |
| 20 | machine-2-4 | 0.3379 | 0.0193 |
| 21 | machine-2-6 | 0.3151 | 0.0174 |
| 22 | machine-1-2 | 0.2056 | 0.0201 |
| 23 | machine-3-6 | 0.1856 | 0.0192 |
| 24 | machine-1-5 | 0.1696 | 0.0179 |
| 25 | machine-1-7 | 0.1315 | 0.0152 |
| 26 | machine-2-1 | 0.1160 | 0.0195 |
| 27 | machine-2-2 | 0.0980 | 0.0114 |
| 28 | machine-3-3 | 0.0529 | 0.0116 |

## Evidence

- `data/ServerMachineDataset/train/*.txt` and
  `data/ServerMachineDataset/test/*.txt` — 28 matching raw train/test entity
  files; every loaded matrix has 38 channels.
- `src/analysis/anomaly_archive_kl.py:83-112` — the repository's histogram KL
  helper confirms the 64-bin combined-range convention and additive smoothing
  used for this calculation.
- `scripts/analysis/rank_smd_train_test_normal_drift.py:88-103` — the existing
  SMD-specific analysis instead masks labeled anomalous test points. This
  report deliberately does not apply that mask because the research question
  asks for the full test series.

## Interpretation and limitations

The rank is useful for choosing entities with different marginal train/test
conditions. It is not a ranking of anomaly difficulty or model performance.
For example, a large mean KL can be caused by a small number of channels with
test-only histogram bins. The JSD column is a bounded, symmetric cross-check;
it substantially lowers the relative position of `machine-3-11` and
`machine-3-4`, suggesting their high mean KL is concentrated rather than broad.

JSD is the recommended additional metric if the goal is a stable symmetric
summary on this same histogram representation. The result still depends on the
64-bin choice, and neither score captures temporal or multivariate shift.
