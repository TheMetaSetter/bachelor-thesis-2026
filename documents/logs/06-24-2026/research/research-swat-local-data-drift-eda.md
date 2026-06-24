---
date: 2026-06-24 21:17:00 +07
researcher: Codex
git_commit: 3952bad44d222bb1a0db25d41af84245d65e088e
branch: dev
repository: bachelor-thesis-2026
topic: "Local SWaT presence check and train-test distribution drift EDA"
tags: [research, swat, eda, drift, kl-divergence]
status: complete
last_updated: 2026-06-24
last_updated_by: Codex
---

# Research: Local SWaT presence check and train-test distribution drift EDA

**Date**: 2026-06-24 21:17:00 +07  
**Researcher**: Codex  
**Git Commit**: `3952bad44d222bb1a0db25d41af84245d65e088e`  
**Branch**: `dev`

## Research Question

Check whether SWaT already exists under `data/` in the current workspace. If it does, run a short EDA focused on KL divergence and other distribution-drift metrics between the train sequence and the test sequence.

## Summary

SWaT is already present locally under `data/SWaT/` with three CSV files: `normal.csv`, `attack.csv`, and `merged.csv`. The local file layout is therefore sufficient for direct EDA without downloading anything else.

For the local data that is actually present, the effective split is:

- train sequence: `data/SWaT/normal.csv`
- test sequence: `data/SWaT/attack.csv`

The local `attack.csv` is fully labeled `Attack`, not a mixed normal-plus-attack test stream. This matters a lot for interpretation. The measured drift between `normal.csv` and `attack.csv` is therefore intentionally very large, because the comparison is effectively **normal-only train versus attack-only test**.

To avoid one-metric bias, the EDA computed:

- histogram-based `KL(train || test)`
- histogram-based `KL(test || train)`
- Jensen-Shannon divergence
- Wasserstein distance
- Kolmogorov-Smirnov statistic
- standardized absolute mean shift

The strongest drift features are concentrated around `AIT201`, `AIT402`, `PIT501`, `PIT502`, `PIT503`, `AIT501`, and `AIT504`.

## Detailed Findings

### Local Data Presence

SWaT is already present locally:

- `data/SWaT/normal.csv`
- `data/SWaT/attack.csv`
- `data/SWaT/merged.csv`

The raw local directory size is about `815M`.

### Data Preparation Assumptions Used for EDA

The EDA used the local CSVs directly.

- Column names were stripped because some SWaT columns contain leading spaces.
- `Timestamp` and `Normal/Attack` were excluded from feature drift calculations.
- The remaining `51` columns were coerced to numeric.
- Small parse gaps were forward-filled and backward-filled.

This mirrors the broad spirit of the reference SWaT loader in `M2N2`, which also uses `data/SWaT` and numeric coercion plus forward fill for SWaT loading.

### Dataset Shapes and Labels

From `outputs/eda_swat_drift_2026-06-24/swat_drift_summary.json`:

- train rows: `1,387,098`
- test rows: `54,621`
- merged rows: `1,441,719`
- feature count: `51`
- test label counts: `Attack = 54,621`
- merged label counts:
  - `Normal = 1,387,098`
  - `Attack = 54,621`
- attack fraction in `merged.csv`: `0.037886023559375995`

### Train vs Test Drift

For `train_normal_vs_test_attack`:

- median `KL(train || test)`: `0.9119754221240752`
- mean `KL(train || test)`: `1.7095001072426963`
- median `KL(test || train)`: `1.1404711982143674`
- mean `KL(test || train)`: `1.5535797693816136`
- median Jensen-Shannon divergence: `0.20457869744797724`
- mean Jensen-Shannon divergence: `0.18006588977272778`
- median Wasserstein distance: `0.6047679787452024`
- mean Wasserstein distance: `20.81701787233951`
- median KS statistic: `0.5451787978701341`
- mean KS statistic: `0.40667220913158286`

The raw mean standardized mean shift is not reliable as a global summary because several actuator channels are constant in train, which makes division by near-zero standard deviation explode numerically.

The near-zero-train-variance features are:

- `P202`
- `P204`
- `P206`
- `P401`
- `P404`
- `P502`
- `P601`
- `P603`

If those constant-train features are excluded, the standardized mean shift becomes much more interpretable:

- filtered mean standardized mean shift: `3.5015085575606877`
- filtered median standardized mean shift: `1.3971969600481424`

### Strongest Drift Features

Top features by Jensen-Shannon divergence for `train_normal_vs_test_attack`:

1. `AIT201` with JS `0.5340048539166762`
2. `AIT402` with JS `0.418860242244838`
3. `PIT502` with JS `0.39574728196307835`
4. `AIT501` with JS `0.3699596409999041`
5. `PIT503` with JS `0.36266071161583435`

Top features by `KL(train || test)`:

1. `PIT503` with KL `14.779019440107728`
2. `AIT201` with KL `10.091121542193388`
3. `PIT501` with KL `8.831991885624532`
4. `AIT504` with KL `4.159403078936038`
5. `AIT401` with KL `4.142287687612748`

Top features by KS statistic:

1. `P201` with KS `0.7402253669534671`
2. `AIT201` with KS `0.7331484312493246`
3. `AIT504` with KS `0.6823534948988658`
4. `MV201` with KS `0.6460041580553444`
5. `AIT501` with KS `0.644424013663203`

### Train vs Merged Drift

For `train_normal_vs_merged_all_rows`, the drift is much smaller:

- median `KL(train || merged)`: `0.0027561185712303214`
- mean `KL(train || merged)`: `0.006210088341160504`
- median Jensen-Shannon divergence: `0.0007448783128877291`
- mean Jensen-Shannon divergence: `0.0019229209169445434`
- median KS statistic: `0.02065465678018019`
- mean KS statistic: `0.015407192896102635`

This is consistent with the fact that `merged.csv` is dominated by the normal portion, and only about `3.79%` of rows are labeled `Attack`.

## Code References

- [data/SWaT/normal.csv](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/data/SWaT/normal.csv)
- [data/SWaT/attack.csv](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/data/SWaT/attack.csv)
- [data/SWaT/merged.csv](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/data/SWaT/merged.csv)
- [M2N2 SWaT loader](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/bsc-thesis-ref-codebases/M2N2-master/data/load_data.py:195)
- [Summary artifact JSON](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/outputs/eda_swat_drift_2026-06-24/swat_drift_summary.json:1)
- [Per-feature metrics: train vs test attack](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/outputs/eda_swat_drift_2026-06-24/train_vs_test_attack_feature_metrics.csv)
- [Per-feature metrics: train vs merged](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/outputs/eda_swat_drift_2026-06-24/train_vs_merged_feature_metrics.csv)

## Pipeline Documentation

This EDA is external to the active thesis runtime pipeline. The current thesis codebase still validates only `dataset_name in {"smd", "anomaly_archive"}` and therefore does not yet expose SWaT as a first-class dataset in the active experiment configs.

The local SWaT files can still be inspected directly for drift analysis, which is what this note documents.

## Historical Context (from documents/)

The design documents already mention SWaT as a target benchmark family, but the current runtime implementation has not yet promoted SWaT into the supported dataset set for the main pipeline.

## Open Questions

1. Should the thesis codebase eventually standardize `data/SWaT/normal.csv` and `data/SWaT/attack.csv` into an official parser under `src/data/datasets/`?
2. If SWaT enters the main pipeline later, should the official test protocol use only `attack.csv`, or should a mixed timeline similar to `merged.csv` also be supported explicitly?
