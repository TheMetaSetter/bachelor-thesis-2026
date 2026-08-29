# Benchmark metric reporting bundle

Source revision: 820c88867114e8ef92299935eff756b3eb0e0aaf

## Available metric files

- Offline: 54 files in offline_phase_tables/offline_metrics_54/.
  - THESIS Stage B: 18.
  - Traditional ML: 27.
  - RedLamp encoder evaluation: 9.
- Online: 99 files in online_phase_tables/online_metrics_99/.
  - THESIS: 54.
  - M2N2: 9.
  - CANDI: 9.
  - Isolation Forest: 9.
  - KMeansAD: 9.
  - StumPy: 9.

## Important scope note

M2N2 and CANDI do not have separate offline detection metrics in the remote output tree. They use the RedLamp encoder checkpoint for their online runs. The manifest records this as not_available; it does not copy RedLamp metrics under the M2N2 or CANDI names.

Use benchmark_metrics_manifest.json for the complete file inventory and provenance. Use offline_phase_tables/offline_report_data.json for the compact offline reporting payload.

The online Table 3 metric payload is stored at
`online_phase_tables/online_table3_metrics.json`. It contains the 99 per-run
records, the method/variant means, and the method/variant/entity means for
FPR, VUS-PR, affiliation F1, and VUS-ROC.

The THESIS online FPR audit is stored at
`online_phase_tables/online_fpr_metrics.json`. Its compact tables are
`online_phase_tables/online_fpr_table.md` and
`online_phase_tables/online_fpr_table.tex`. FPR uses the causal endpoint
labels and the strict runtime rule `online/ewma_point_score > online/threshold`.

The rendered comparison table is stored at
`online_phase_tables/online_phase_metric_table_report3.md`.

The offline uncertainty comparison table is stored at
`offline_phase_tables/offline_phase_metric_table_report2.md`. It compares
`mean_of_variances` between clean validation and test for THESIS variants O0
and O1 across the three SMD entities.
