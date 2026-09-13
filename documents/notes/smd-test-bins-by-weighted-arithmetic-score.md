# SMD Test Bins by `weighted_arithmetic_score`

This note records a new three-bin partition of SMD test machines.

## Simple procedure

1. Reuse the 27 machines in the current `realistic_metric_1` bins.
2. Sort machines by `weighted_arithmetic_score` in ascending order.
3. Put nine consecutive machines into each new bin.
4. Keep `machine-2-8` outside the bins to preserve the same machine set.

## Metric

Let (j=JSD/\ln(2)) and (k=KL/(KL+\ln(2))).

The weighted arithmetic score is `weighted_arithmetic_score = (j + k) / 2`.

## Bin 1 — Low

Range: `0.043812–0.177809`.

| Rank | Machine | `realistic_metric_1` | `weighted_arithmetic_score` |
|---:|---|---:|---:|
| 1 | machine-3-3 | 20.833333 | 0.043812 |
| 2 | machine-2-2 | 2.894737 | 0.070129 |
| 3 | machine-2-1 | 10.857143 | 0.085777 |
| 4 | machine-1-7 | 0.968085 | 0.090706 |
| 5 | machine-1-5 | 29.866667 | 0.111227 |
| 6 | machine-3-6 | 2.116705 | 0.119451 |
| 7 | machine-1-2 | 1.794118 | 0.128865 |
| 8 | machine-2-6 | 0.093023 | 0.168793 |
| 9 | machine-2-4 | 0.417544 | 0.177809 |

## Bin 2 — Medium

Range: `0.218843–0.334435`.

| Rank | Machine | `realistic_metric_1` | `weighted_arithmetic_score` |
|---:|---|---:|---:|
| 10 | machine-2-7 | 17.857143 | 0.218843 |
| 11 | machine-2-3 | 2.632653 | 0.246258 |
| 12 | machine-2-5 | 1.696970 | 0.277239 |
| 13 | machine-3-7 | 8.642857 | 0.286953 |
| 14 | machine-1-8 | 5.875000 | 0.292657 |
| 15 | machine-2-9 | 0.221491 | 0.295114 |
| 16 | machine-3-4 | 1.565217 | 0.310041 |
| 17 | machine-3-8 | 0.698473 | 0.310973 |
| 18 | machine-1-4 | 4.122449 | 0.334435 |

## Bin 3 — High

Range: `0.350175–0.547160`.

| Rank | Machine | `realistic_metric_1` | `weighted_arithmetic_score` |
|---:|---|---:|---:|
| 19 | machine-1-1 | 0.547972 | 0.350175 |
| 20 | machine-3-11 | 0.333333 | 0.353600 |
| 21 | machine-1-3 | 1.491103 | 0.359679 |
| 22 | machine-3-10 | 0.347826 | 0.401013 |
| 23 | machine-3-2 | 5.500000 | 0.407035 |
| 24 | machine-3-5 | 0.425703 | 0.415044 |
| 25 | machine-3-1 | 25.672840 | 0.511684 |
| 26 | machine-3-9 | 26.312102 | 0.519289 |
| 27 | machine-1-6 | 0.302730 | 0.547160 |

## Outside the bins

| Machine | `realistic_metric_1` | `weighted_arithmetic_score` |
|---|---:|---:|
| machine-2-8 | N/A | 0.320647 |
