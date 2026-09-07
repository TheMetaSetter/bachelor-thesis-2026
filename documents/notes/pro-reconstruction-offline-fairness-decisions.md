# Pro-reconstruction offline fairness decisions

Date: 2026-09-06

Status: agreed experiment contract; local implementation and smoke preflight are complete, remote execution is not started.

## Scope

- Train the main method for 54 offline combinations: `O0/O1 × machine-1-6/machine-3-4/machine-3-9 × seed 6/8/36 × FPR budget {0.1%, 0.5%, 1%}`.
- Each main-method combination runs Stage A, Stage-B initialization, Stage B, and offline evaluation.
- Evaluate RedLamp, iForest, KMeans-AD, and StumPy offline with their existing model and hyperparameter configurations.
- Do not change baseline hyperparameters or tune them again; keep their old method configurations and change only evaluation conditions required below.
- Traditional baselines may fit on train data because their existing evaluator requires that fit before scoring; this is not hyperparameter tuning or a new baseline-training experiment.
- M2N2 and CANDI have no separate offline detection matrix, so they are outside this scope.

## Main-method training contract

- Use normalized-input reconstruction MSE for the reconstruction loss.
- Set `lambda_recon: 0.75` and `lambda_cls: 0.25`.
- O0 disables point-score loss.
- O1 enables balanced point-score loss in Stage A.
- Both Stage A and Stage B use `fusion_mode: direct_branch_routing`.
- Use 10 data-loading workers.
- The 54 combinations create 108 W&B training runs because each combination has Stage A and Stage B.

## Score and calibration contract

- Use normalized-input reconstruction MSE as the anomaly score for the main method and RedLamp.
- Keep the native score for traditional baselines: iForest negative decision score, KMeans-AD nearest-centroid distance, and StumPy AB-join distance.
- Score every synthetic-validation window before filtering calibration values.
- Set the point threshold to q99 of finite point scores whose synthetic point label is `0`, including normal points inside an anomalous window.
- Set the window threshold to q99 of finite window scores whose synthetic window label is normal.
- Use the two thresholds for point-level and window-level offline predictions respectively.
- Do not use clean validation for these offline thresholds.
- Use test labels only after prediction to calculate final metrics.

## Budgeted VUS metrics

- Calculate VUS-PR and VUS-ROC at FPR budgets `0.001`, `0.005`, and `0.01`.
- One evaluation computes all three FPR budgets; it does not create three separate evaluation runs.
- Persist `vus_pr_at_fpr_budget` and `vus_roc_at_fpr_budget`, each keyed by the three FPR budgets.

## Fair-comparison rules

- Compare the same entity, seed, window length 20, train/validation/test split, q99 calibration quantile, and FPR budgets whenever method assumptions permit.
- Fit all learned baseline state only on train data.
- Keep each baseline's model, native score, and inference mechanism.
- Do not add THESIS fusion, point-score loss, Monte Carlo logic, or THESIS-specific threshold transforms to a baseline.
- Record unavailable or failed cells instead of replacing them with another method's result.

## Current implementation status

- The budgeted VUS implementation returns both metric mappings for all three budgets.
- The new normalized-input protocol selects point and window thresholds from synthetic-normal validation values.
- The evaluator and trainer expose normalized-input MSE and stable scalar VUS-PR budget monitor names.
- The new matrix generator writes 54 unique configs and two-stage materialization writes 108 unique Stage A and Stage B W&B names.
- The traditional runner has a native-score path that preserves iForest, KMeans-AD, and StumPy score semantics.
- The RedLamp evaluator accepts an explicit protocol and isolated output root for evaluation-only runs.
- Local smoke preflight and focused tests pass, while full GPU training remains unexecuted.

## Deferred operational items

- The exact canonical runtime name for `pro-reconstruction`; the active tree has reconstruction-focused and `recon075_cls025_direct` names but no literal `pro-reconstruction` configuration.
- Each budget-specific run monitors its matching scalar metric: `val_synth_vus_pr_at_fpr_budget_0_001`, `val_synth_vus_pr_at_fpr_budget_0_005`, or `val_synth_vus_pr_at_fpr_budget_0_01`, with mode `max`.
- The isolated output root is `outputs/benchmark_pro_reconstruction_vus_budget`.
- The remote execution command remains deferred until the one-combination smoke flow is run on the target GPU host.
