---
date: 2026-06-26T14:42:00+00:00
researcher: Codex
topic: "Forensic audit for anomaly_archive_redlamp_mlp_baseline_staffiii_window20_adamw_cosine_warmup10_vus_pr_confmat"
status: complete
---

# Forensic Audit: anomaly_archive_redlamp_mlp_baseline_staffiii_window20_adamw_cosine_warmup10_vus_pr_confmat

## Evaluated Run

- Experiment config: `configs/experiment/scale/anomaly_archive__redlamp_mlp_baseline__staffiii-window20-adamw-cosine-warmup10-vus-pr-confmat__w20__seed11__default.yaml`
- Dataset: `anomaly_archive`
- Data config path: `configs/data/anomaly_archive_staffiii_full.yaml`

## Verified Protocol Facts

- `comparison_mode`: `pre_vs_anomaly`
- Test label regime: `all_one`
- Test positive ratio: `1.000000`
- Test windows: `21`
- Test truncated coverage: `False`

## Observed Metric Bundle

- `precision`: 1.0
- `recall`: 0.05
- `pr_auc`: 1.0
- `roc_auc`: nan
- `threshold`: 0.072061
- `vus_pr`: nan

## Causal Interpretation

The evaluated test vector is all anomalous. This is a protocol-special single-class regime, not a standard full-timeline anomaly-detection test.

- Every evaluated timestep is labeled anomalous, so there are no true negatives in the test vector.
- Precision can stay at 1.0 as long as the model predicts at least one positive, because false positives are impossible in an all-positive label regime.
- Recall becomes the fraction of anomalous timesteps whose scores exceed the threshold, so a low recall simply means the model predicted only a small positive subset.
- PR-AUC can appear perfect or otherwise degenerate in this regime and should not be interpreted as normal model-quality evidence.
- ROC-AUC becomes undefined because the evaluator sees only one label class.
- VUS-PR and VUS-ROC can also become undefined because the range-based metric code expects both positive and negative labels.
- The observed recall of 0.050000 is therefore consistent with a model that flagged only about 5.00% of an already-all-anomalous test timeline.
- The observed threshold was 0.072061.

## Raw Warnings From Audit Layer

- AnomalyArchive comparison modes are protocol-specific slices and are not directly comparable to full-timeline SMD-style pointwise anomaly detection.
- Test labels contain only one class. Pointwise metrics such as ROC-AUC, PR-AUC, and VUS can become degenerate or misleading.

## Conclusion

The strongest repository-grounded conclusion is that this run is not using a standard mixed-label full-timeline test regime. The current config constructs an all-positive `anomaly_archive` test slice under `pre_vs_anomaly`, so several pointwise metrics become degenerate or misleading by construction. If the observed bundle was `precision = 1`, `recall` very low, `pr_auc = 1`, `roc_auc = NaN`, and `vus_pr = NaN`, that bundle is consistent with the implemented protocol and does not by itself prove either a strong or weak anomaly detector.
