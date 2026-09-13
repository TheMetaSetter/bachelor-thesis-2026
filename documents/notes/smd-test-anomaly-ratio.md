# SMD Test-Series Anomaly Ratio

This note records the point-level anomaly ratio for all 28 SMD machines.

## Formula

`anomaly_ratio = anomaly points / total test points`

An anomaly point is a test point whose label is `1` in `test_label/<machine>.txt`.

## Results

| Machine | Anomaly points | Test points | `anomaly_ratio` | Percentage |
|---|---:|---:|---:|---:|
| machine-1-1 | 2694 | 28479 | 0.094596018 | 9.459602% |
| machine-1-2 | 542 | 23694 | 0.022874989 | 2.287499% |
| machine-1-3 | 817 | 23703 | 0.034468211 | 3.446821% |
| machine-1-4 | 720 | 23707 | 0.030370777 | 3.037078% |
| machine-1-5 | 100 | 23706 | 0.004218341 | 0.421834% |
| machine-1-6 | 3708 | 23689 | 0.156528346 | 15.652835% |
| machine-1-7 | 2398 | 23697 | 0.101194244 | 10.119424% |
| machine-1-8 | 763 | 23699 | 0.032195451 | 3.219545% |
| machine-2-1 | 1170 | 23694 | 0.049379590 | 4.937959% |
| machine-2-2 | 2833 | 23700 | 0.119535865 | 11.953586% |
| machine-2-3 | 269 | 23689 | 0.011355481 | 1.135548% |
| machine-2-4 | 1694 | 23689 | 0.071509984 | 7.150998% |
| machine-2-5 | 980 | 23689 | 0.041369412 | 4.136941% |
| machine-2-6 | 424 | 28743 | 0.014751418 | 1.475142% |
| machine-2-7 | 417 | 23696 | 0.017597907 | 1.759791% |
| machine-2-8 | 161 | 23703 | 0.006792389 | 0.679239% |
| machine-2-9 | 1755 | 28722 | 0.061102987 | 6.110299% |
| machine-3-1 | 308 | 28700 | 0.010731707 | 1.073171% |
| machine-3-10 | 1047 | 23693 | 0.044190267 | 4.419027% |
| machine-3-11 | 198 | 28696 | 0.006899916 | 0.689992% |
| machine-3-2 | 1109 | 23703 | 0.046787326 | 4.678733% |
| machine-3-3 | 632 | 23703 | 0.026663292 | 2.666329% |
| machine-3-4 | 977 | 23687 | 0.041246253 | 4.124625% |
| machine-3-5 | 426 | 23691 | 0.017981512 | 1.798151% |
| machine-3-6 | 1194 | 28726 | 0.041565133 | 4.156513% |
| machine-3-7 | 434 | 28705 | 0.015119317 | 1.511932% |
| machine-3-8 | 1371 | 28704 | 0.047763378 | 4.776338% |
| machine-3-9 | 303 | 28713 | 0.010552711 | 1.055271% |

## Data source

The test series come from `data/ServerMachineDataset/test/`.

The point labels come from `data/ServerMachineDataset/test_label/`.

## Three equal-sized bins

Because 28 machines cannot be divided into three exactly equal groups, the sorted list uses sizes 10, 9, and 9.

Machines are sorted by `anomaly_ratio` in ascending order.

### Bin 1 — Low

Range: `0.004218341–0.017981512`.

| Machine | `anomaly_ratio` |
|---|---:|
| machine-1-5 | 0.004218341 |
| machine-2-8 | 0.006792389 |
| machine-3-11 | 0.006899916 |
| machine-3-9 | 0.010552711 |
| machine-3-1 | 0.010731707 |
| machine-2-3 | 0.011355481 |
| machine-2-6 | 0.014751418 |
| machine-3-7 | 0.015119317 |
| machine-2-7 | 0.017597907 |
| machine-3-5 | 0.017981512 |

### Bin 2 — Medium

Range: `0.022874989–0.044190267`.

| Machine | `anomaly_ratio` |
|---|---:|
| machine-1-2 | 0.022874989 |
| machine-3-3 | 0.026663292 |
| machine-1-4 | 0.030370777 |
| machine-1-8 | 0.032195451 |
| machine-1-3 | 0.034468211 |
| machine-3-4 | 0.041246253 |
| machine-2-5 | 0.041369412 |
| machine-3-6 | 0.041565133 |
| machine-3-10 | 0.044190267 |

### Bin 3 — High

Range: `0.046787326–0.156528346`.

| Machine | `anomaly_ratio` |
|---|---:|
| machine-3-2 | 0.046787326 |
| machine-3-8 | 0.047763378 |
| machine-2-1 | 0.049379590 |
| machine-2-9 | 0.061102987 |
| machine-2-4 | 0.071509984 |
| machine-1-1 | 0.094596018 |
| machine-1-7 | 0.101194244 |
| machine-2-2 | 0.119535865 |
| machine-1-6 | 0.156528346 |
