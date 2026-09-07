# SMD 28-machine drift zoo

This note ranks raw SMD train-to-test value-distribution shift from high to
low. It uses the entire test series, including labeled anomalous points.

For each entity, the score is the mean of `KL(P_test || P_train)` over its 38
channels. Each channel uses 64 equal-width bins over the combined train/test
value range and `1e-12` additive smoothing. `Mean JSD` is the symmetric,
bounded cross-check.

| KL rank | Entity | Mean KL(test \|\| train) | Mean JSD |
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

Read the detailed methodology and limitations in
`documents/logs/2026-09-06/research/research-smd-train-test-value-distribution-shift.md`.

This ranking measures only marginal value-distribution shift. It does not
measure temporal order, autocorrelation, or cross-channel dependence.
