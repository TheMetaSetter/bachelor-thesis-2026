# Online report: `VUS-PR@FPR-budget`

This report contains only `VUS-PR@FPR-budget` at FPR-budget values of 0.1%, 0.5%, and 1%.

The evaluation uses SMD entities `machine_1_6`, `machine_3_4`, and `machine_3_9` with seeds 6, 8, and 36.

The matrix contains 54 THESIS runs and 45 baseline runs.

THESIS covers offline variants O0 and O1 with online variants A0, A1, and A2.

The online baselines are M2N2, CANDI, IForest, KMeans-AD, and StumPy.

The evaluation reads the existing causal online records and does not retrain any method.

The fairness policy uses window size 20, online stride 1, endpoint-sliced labels, no point adjustment, and test labels only for metric measurement.

The metric uses `max_buffer_size=20` and `num_thresholds=200`.

The result JSON files retain only the root identity fields `method`, `offline_variant`, `online_variant`, `entity`, and `seed`, plus `VUS-PR@FPR-budget`.

## Mean across 9 entity-seed runs

| Method | Offline | Online | FPR-budget 0.1% | FPR-budget 0.5% | FPR-budget 1% |
|---|---|---|---:|---:|---:|
| CANDI | - | main | 0.165103 | 0.277713 | 0.348186 |
| IForest | - | main | 0.001435 | 0.001532 | 0.002026 |
| KMeans-AD | - | main | 0.111146 | 0.232235 | 0.330298 |
| M2N2 | - | main | 0.165103 | 0.277713 | 0.348186 |
| StumPy | - | main | 0.006161 | 0.006142 | 0.006123 |
| THESIS | O0 | A0 | 0.632025 | 0.689340 | 0.687847 |
| THESIS | O0 | A1 | 0.632025 | 0.689340 | 0.687847 |
| THESIS | O0 | A2 | 0.631897 | 0.688284 | 0.686792 |
| THESIS | O1 | A0 | 0.628358 | 0.692103 | 0.693699 |
| THESIS | O1 | A1 | 0.628358 | 0.692103 | 0.693699 |
| THESIS | O1 | A2 | 0.628240 | 0.690740 | 0.693411 |

## Detailed results

| Method | Offline | Online | Entity | Seed | FPR-budget 0.1% | FPR-budget 0.5% | FPR-budget 1% |
|---|---|---|---|---:|---:|---:|---:|
| CANDI | - | main | machine_1_6 | 6 | 0.084983 | 0.249483 | 0.420557 |
| CANDI | - | main | machine_1_6 | 8 | 0.000000 | 0.207159 | 0.389984 |
| CANDI | - | main | machine_1_6 | 36 | 0.071959 | 0.229465 | 0.412574 |
| CANDI | - | main | machine_3_4 | 6 | 0.292100 | 0.292100 | 0.292100 |
| CANDI | - | main | machine_3_4 | 8 | 0.289788 | 0.289788 | 0.289788 |
| CANDI | - | main | machine_3_4 | 36 | 0.264740 | 0.525118 | 0.622366 |
| CANDI | - | main | machine_3_9 | 6 | 0.191102 | 0.292661 | 0.292661 |
| CANDI | - | main | machine_3_9 | 8 | 0.194612 | 0.296053 | 0.296053 |
| CANDI | - | main | machine_3_9 | 36 | 0.096642 | 0.117588 | 0.117588 |
| IForest | - | main | machine_1_6 | 6 | 0.000000 | 0.000000 | 0.000000 |
| IForest | - | main | machine_1_6 | 8 | 0.000000 | 0.000000 | 0.000000 |
| IForest | - | main | machine_1_6 | 36 | 0.000000 | 0.000000 | 0.000000 |
| IForest | - | main | machine_3_4 | 6 | 0.000000 | 0.000000 | 0.000000 |
| IForest | - | main | machine_3_4 | 8 | 0.000000 | 0.000000 | 0.000000 |
| IForest | - | main | machine_3_4 | 36 | 0.000000 | 0.000000 | 0.000000 |
| IForest | - | main | machine_3_9 | 6 | 0.000000 | 0.000000 | 0.000113 |
| IForest | - | main | machine_3_9 | 8 | 0.006833 | 0.007689 | 0.011786 |
| IForest | - | main | machine_3_9 | 36 | 0.006080 | 0.006102 | 0.006333 |
| KMeans-AD | - | main | machine_1_6 | 6 | 0.028848 | 0.066285 | 0.108385 |
| KMeans-AD | - | main | machine_1_6 | 8 | 0.024445 | 0.064151 | 0.070442 |
| KMeans-AD | - | main | machine_1_6 | 36 | 0.000000 | 0.000000 | 0.000000 |
| KMeans-AD | - | main | machine_3_4 | 6 | 0.252021 | 0.265740 | 0.292564 |
| KMeans-AD | - | main | machine_3_4 | 8 | 0.267042 | 0.288759 | 0.583998 |
| KMeans-AD | - | main | machine_3_4 | 36 | 0.247012 | 0.273655 | 0.576126 |
| KMeans-AD | - | main | machine_3_9 | 6 | 0.031906 | 0.428095 | 0.456918 |
| KMeans-AD | - | main | machine_3_9 | 8 | 0.082890 | 0.484315 | 0.494680 |
| KMeans-AD | - | main | machine_3_9 | 36 | 0.066154 | 0.219112 | 0.389569 |
| M2N2 | - | main | machine_1_6 | 6 | 0.084983 | 0.249483 | 0.420557 |
| M2N2 | - | main | machine_1_6 | 8 | 0.000000 | 0.207159 | 0.389984 |
| M2N2 | - | main | machine_1_6 | 36 | 0.071959 | 0.229465 | 0.412574 |
| M2N2 | - | main | machine_3_4 | 6 | 0.292100 | 0.292100 | 0.292100 |
| M2N2 | - | main | machine_3_4 | 8 | 0.289788 | 0.289788 | 0.289788 |
| M2N2 | - | main | machine_3_4 | 36 | 0.264740 | 0.525118 | 0.622366 |
| M2N2 | - | main | machine_3_9 | 6 | 0.191102 | 0.292661 | 0.292661 |
| M2N2 | - | main | machine_3_9 | 8 | 0.194612 | 0.296053 | 0.296053 |
| M2N2 | - | main | machine_3_9 | 36 | 0.096642 | 0.117588 | 0.117588 |
| StumPy | - | main | machine_1_6 | 6 | 0.000000 | 0.000000 | 0.000000 |
| StumPy | - | main | machine_1_6 | 8 | 0.000000 | 0.000000 | 0.000000 |
| StumPy | - | main | machine_1_6 | 36 | 0.000000 | 0.000000 | 0.000000 |
| StumPy | - | main | machine_3_4 | 6 | 0.018482 | 0.018425 | 0.018369 |
| StumPy | - | main | machine_3_4 | 8 | 0.018482 | 0.018425 | 0.018369 |
| StumPy | - | main | machine_3_4 | 36 | 0.018482 | 0.018425 | 0.018369 |
| StumPy | - | main | machine_3_9 | 6 | 0.000000 | 0.000000 | 0.000000 |
| StumPy | - | main | machine_3_9 | 8 | 0.000000 | 0.000000 | 0.000000 |
| StumPy | - | main | machine_3_9 | 36 | 0.000000 | 0.000000 | 0.000000 |
| THESIS | O0 | A0 | machine_1_6 | 6 | 0.722455 | 0.716590 | 0.707948 |
| THESIS | O0 | A0 | machine_1_6 | 8 | 0.726504 | 0.719043 | 0.709250 |
| THESIS | O0 | A0 | machine_1_6 | 36 | 0.726504 | 0.717038 | 0.708495 |
| THESIS | O0 | A0 | machine_3_4 | 6 | 0.935243 | 0.935243 | 0.935243 |
| THESIS | O0 | A0 | machine_3_4 | 8 | 0.937546 | 0.937546 | 0.937546 |
| THESIS | O0 | A0 | machine_3_4 | 36 | 0.926012 | 0.926012 | 0.926012 |
| THESIS | O0 | A0 | machine_3_9 | 6 | 0.245942 | 0.383818 | 0.383818 |
| THESIS | O0 | A0 | machine_3_9 | 8 | 0.234447 | 0.430387 | 0.430387 |
| THESIS | O0 | A0 | machine_3_9 | 36 | 0.233576 | 0.438379 | 0.451926 |
| THESIS | O0 | A1 | machine_1_6 | 6 | 0.722455 | 0.716590 | 0.707948 |
| THESIS | O0 | A1 | machine_1_6 | 8 | 0.726504 | 0.719043 | 0.709250 |
| THESIS | O0 | A1 | machine_1_6 | 36 | 0.726504 | 0.717038 | 0.708495 |
| THESIS | O0 | A1 | machine_3_4 | 6 | 0.935243 | 0.935243 | 0.935243 |
| THESIS | O0 | A1 | machine_3_4 | 8 | 0.937546 | 0.937546 | 0.937546 |
| THESIS | O0 | A1 | machine_3_4 | 36 | 0.926012 | 0.926012 | 0.926012 |
| THESIS | O0 | A1 | machine_3_9 | 6 | 0.245942 | 0.383818 | 0.383818 |
| THESIS | O0 | A1 | machine_3_9 | 8 | 0.234447 | 0.430387 | 0.430387 |
| THESIS | O0 | A1 | machine_3_9 | 36 | 0.233576 | 0.438379 | 0.451926 |
| THESIS | O0 | A2 | machine_1_6 | 6 | 0.722455 | 0.716590 | 0.707948 |
| THESIS | O0 | A2 | machine_1_6 | 8 | 0.726504 | 0.719043 | 0.709250 |
| THESIS | O0 | A2 | machine_1_6 | 36 | 0.726504 | 0.717038 | 0.708495 |
| THESIS | O0 | A2 | machine_3_4 | 6 | 0.935243 | 0.935243 | 0.935243 |
| THESIS | O0 | A2 | machine_3_4 | 8 | 0.937546 | 0.937546 | 0.937546 |
| THESIS | O0 | A2 | machine_3_4 | 36 | 0.924855 | 0.924855 | 0.924855 |
| THESIS | O0 | A2 | machine_3_9 | 6 | 0.245942 | 0.383818 | 0.383818 |
| THESIS | O0 | A2 | machine_3_9 | 8 | 0.234447 | 0.422044 | 0.422044 |
| THESIS | O0 | A2 | machine_3_9 | 36 | 0.233576 | 0.438379 | 0.451926 |
| THESIS | O1 | A0 | machine_1_6 | 6 | 0.726504 | 0.719043 | 0.714469 |
| THESIS | O1 | A0 | machine_1_6 | 8 | 0.726504 | 0.718105 | 0.714982 |
| THESIS | O1 | A0 | machine_1_6 | 36 | 0.726504 | 0.719043 | 0.708052 |
| THESIS | O1 | A0 | machine_3_4 | 6 | 0.931373 | 0.940703 | 0.940703 |
| THESIS | O1 | A0 | machine_3_4 | 8 | 0.818497 | 0.945558 | 0.945558 |
| THESIS | O1 | A0 | machine_3_4 | 36 | 0.932948 | 0.932948 | 0.932948 |
| THESIS | O1 | A0 | machine_3_9 | 6 | 0.324709 | 0.373468 | 0.373468 |
| THESIS | O1 | A0 | machine_3_9 | 8 | 0.234141 | 0.456985 | 0.456985 |
| THESIS | O1 | A0 | machine_3_9 | 36 | 0.234040 | 0.423071 | 0.456127 |
| THESIS | O1 | A1 | machine_1_6 | 6 | 0.726504 | 0.719043 | 0.714469 |
| THESIS | O1 | A1 | machine_1_6 | 8 | 0.726504 | 0.718105 | 0.714982 |
| THESIS | O1 | A1 | machine_1_6 | 36 | 0.726504 | 0.719043 | 0.708052 |
| THESIS | O1 | A1 | machine_3_4 | 6 | 0.931373 | 0.940703 | 0.940703 |
| THESIS | O1 | A1 | machine_3_4 | 8 | 0.818497 | 0.945558 | 0.945558 |
| THESIS | O1 | A1 | machine_3_4 | 36 | 0.932948 | 0.932948 | 0.932948 |
| THESIS | O1 | A1 | machine_3_9 | 6 | 0.324709 | 0.373468 | 0.373468 |
| THESIS | O1 | A1 | machine_3_9 | 8 | 0.234141 | 0.456985 | 0.456985 |
| THESIS | O1 | A1 | machine_3_9 | 36 | 0.234040 | 0.423071 | 0.456127 |
| THESIS | O1 | A2 | machine_1_6 | 6 | 0.726504 | 0.719043 | 0.714469 |
| THESIS | O1 | A2 | machine_1_6 | 8 | 0.726504 | 0.718105 | 0.714982 |
| THESIS | O1 | A2 | machine_1_6 | 36 | 0.726504 | 0.719043 | 0.708052 |
| THESIS | O1 | A2 | machine_3_4 | 6 | 0.930310 | 0.939553 | 0.939553 |
| THESIS | O1 | A2 | machine_3_4 | 8 | 0.818497 | 0.945558 | 0.945558 |
| THESIS | O1 | A2 | machine_3_4 | 36 | 0.932948 | 0.932948 | 0.932948 |
| THESIS | O1 | A2 | machine_3_9 | 6 | 0.324709 | 0.373468 | 0.373468 |
| THESIS | O1 | A2 | machine_3_9 | 8 | 0.234141 | 0.456985 | 0.456985 |
| THESIS | O1 | A2 | machine_3_9 | 36 | 0.234040 | 0.411961 | 0.454685 |
