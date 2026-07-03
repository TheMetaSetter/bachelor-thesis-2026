# Research: Anomaly span length summary

This note summarizes anomaly span lengths extracted directly from local datasets under `data/`.

## Assumptions

- SWaT uses merged.csv.
- NASA uses half-open intervals [start, end).
- AnomalyArchive uses half-open filename intervals [anomaly_start_index, anomaly_end_index).
- SMD uses ServerMachineDataset test_label files only.

## Dataset summary

| dataset | num_series | num_spans | num_anomalous_points | mean | median | min | max | zero_anomaly_series |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anomaly_archive | 203 | 250 | 49113 | 196.452 | 100.0 | 0 | 1700 | 0 |
| iops | 29 | 1470 | 54560 | 37.1156462585034 | 8.0 | 2 | 1121 | 0 |
| nasa | 81 | 105 | 64704 | 616.2285714285714 | 120.0 | 10 | 4217 | 0 |
| smd | 28 | 327 | 29444 | 90.0428134556575 | 11.0 | 2 | 3161 | 0 |
| swat | 1 | 1 | 54621 | 54621.0 | 54621.0 | 54621 | 54621 | 0 |

## Notes

- Selected datasets: anomaly_archive, nasa, smd, iops, swat
- Series without anomaly spans are omitted from the span-level CSV but still counted in dataset-level summary.
