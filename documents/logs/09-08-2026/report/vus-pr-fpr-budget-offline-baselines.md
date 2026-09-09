# Offline baseline report: `VUS-PR@FPR-budget`

This report contains only `VUS-PR@FPR-budget` at FPR-budget values of 0.1%, 0.5%, and 1%.

The evaluation uses SMD entities `machine_1_6`, `machine_3_4`, and `machine_3_9` with seeds 6, 8, and 36.

The four offline baselines are IForest, KMeans-AD, RedLamp, and StumPy with channel AB.

The score artifacts were already materialized by each baseline evaluation, so this run only computes the requested metric.

The metric uses `max_buffer_size=20` and `num_thresholds=200`.

The result JSON files retain only the root identity fields `method`, `entity`, and `seed`, plus `VUS-PR@FPR-budget`.

## Mean across 9 entity-seed runs

| Method | FPR-budget 0.1% | FPR-budget 0.5% | FPR-budget 1% |
|---|---:|---:|---:|
| IForest | 0.003026 | 0.008651 | 0.011401 |
| KMeans-AD | 0.114587 | 0.207223 | 0.231289 |
| RedLamp | 0.113069 | 0.120313 | 0.134879 |
| StumPy channel AB | 0.000000 | 0.001046 | 0.000609 |

## Detailed results

| Method | Entity | Seed | FPR-budget 0.1% | FPR-budget 0.5% | FPR-budget 1% |
|---|---|---:|---:|---:|---:|
| IForest | machine_1_6 | 6 | 0.003122 | 0.003724 | 0.004433 |
| IForest | machine_1_6 | 8 | 0.010288 | 0.041343 | 0.048829 |
| IForest | machine_1_6 | 36 | 0.005574 | 0.030732 | 0.048249 |
| IForest | machine_3_4 | 6 | 0.000000 | 0.000000 | 0.000000 |
| IForest | machine_3_4 | 8 | 0.000000 | 0.000000 | 0.000000 |
| IForest | machine_3_4 | 36 | 0.000000 | 0.000000 | 0.000000 |
| IForest | machine_3_9 | 6 | 0.000000 | 0.000000 | 0.000000 |
| IForest | machine_3_9 | 8 | 0.008251 | 0.002063 | 0.001100 |
| IForest | machine_3_9 | 36 | 0.000000 | 0.000000 | 0.000000 |
| KMeans-AD | machine_1_6 | 6 | 0.000180 | 0.001289 | 0.002247 |
| KMeans-AD | machine_1_6 | 8 | 0.000529 | 0.005814 | 0.007291 |
| KMeans-AD | machine_1_6 | 36 | 0.000000 | 0.000021 | 0.001957 |
| KMeans-AD | machine_3_4 | 6 | 0.000000 | 0.014752 | 0.052551 |
| KMeans-AD | machine_3_4 | 8 | 0.028147 | 0.041079 | 0.068485 |
| KMeans-AD | machine_3_4 | 36 | 0.028485 | 0.065137 | 0.066888 |
| KMeans-AD | machine_3_9 | 6 | 0.193162 | 0.548745 | 0.608496 |
| KMeans-AD | machine_3_9 | 8 | 0.571431 | 0.720073 | 0.711828 |
| KMeans-AD | machine_3_9 | 36 | 0.209350 | 0.468094 | 0.561858 |
| RedLamp | machine_1_6 | 6 | 0.000036 | 0.000036 | 0.000036 |
| RedLamp | machine_1_6 | 8 | 0.000018 | 0.000018 | 0.000018 |
| RedLamp | machine_1_6 | 36 | 0.000027 | 0.000027 | 0.000027 |
| RedLamp | machine_3_4 | 6 | 0.242705 | 0.307549 | 0.307549 |
| RedLamp | machine_3_4 | 8 | 0.386344 | 0.386344 | 0.517441 |
| RedLamp | machine_3_4 | 36 | 0.318355 | 0.318710 | 0.318710 |
| RedLamp | machine_3_9 | 6 | 0.022277 | 0.022277 | 0.022277 |
| RedLamp | machine_3_9 | 8 | 0.025578 | 0.025578 | 0.025578 |
| RedLamp | machine_3_9 | 36 | 0.022277 | 0.022277 | 0.022277 |
| StumPy channel AB | machine_1_6 | 6 | 0.000000 | 0.000018 | 0.000009 |
| StumPy channel AB | machine_1_6 | 8 | 0.000000 | 0.000018 | 0.000009 |
| StumPy channel AB | machine_1_6 | 36 | 0.000000 | 0.000018 | 0.000009 |
| StumPy channel AB | machine_3_4 | 6 | 0.000000 | 0.000000 | 0.000213 |
| StumPy channel AB | machine_3_4 | 8 | 0.000000 | 0.000000 | 0.000213 |
| StumPy channel AB | machine_3_4 | 36 | 0.000000 | 0.000000 | 0.000213 |
| StumPy channel AB | machine_3_9 | 6 | 0.000000 | 0.003119 | 0.001604 |
| StumPy channel AB | machine_3_9 | 8 | 0.000000 | 0.003119 | 0.001604 |
| StumPy channel AB | machine_3_9 | 36 | 0.000000 | 0.003119 | 0.001604 |
