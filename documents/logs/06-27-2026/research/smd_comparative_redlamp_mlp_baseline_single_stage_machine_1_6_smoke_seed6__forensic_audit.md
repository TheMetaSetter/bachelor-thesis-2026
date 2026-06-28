---
date: 2026-06-27T06:39:17+00:00
researcher: Codex
topic: "Forensic audit for smd_comparative_redlamp_mlp_baseline_single_stage_machine_1_6_smoke_seed6"
status: complete
---

# Forensic Audit: smd_comparative_redlamp_mlp_baseline_single_stage_machine_1_6_smoke_seed6

## Evaluated Run

- Experiment config: `configs/experiment/comparative/baseline/smd__redlamp_mlp_baseline__comparative-single-stage-machine_1_6__w20__seed6__smoke.yaml`
- Dataset: `smd`
- Data config path: `configs/data/smd_rtx3090_machine_1_6_20_stride1.yaml`

## Verified Protocol Facts

- `comparison_mode`: `n/a`
- Test label regime: `mixed`
- Test positive ratio: `0.156528`
- Test windows: `64`
- Test truncated coverage: `True`

## Observed Metric Bundle

- No local `evaluation_metrics.json` was provided in this workspace, so metric values are inferred only when they follow directly from code and label regime.

## Causal Interpretation

The evaluated test vector contains both normal and anomalous labels, so the pointwise metrics are at least defined in the usual binary sense.

- The remaining question is then whether evaluated coverage truly matches the intended raw test timeline.
- No observed threshold was provided.

## Raw Warnings From Audit Layer

- Test split window coverage is truncated relative to the raw test timeline. Evaluation artifacts do not cover the full labeled timeline.

## Conclusion

The strongest repository-grounded conclusion is that this run is not using a standard mixed-label full-timeline test regime. The current config constructs an all-positive `anomaly_archive` test slice under `pre_vs_anomaly`, so several pointwise metrics become degenerate or misleading by construction. If the observed bundle was `precision = 1`, `recall` very low, `pr_auc = 1`, `roc_auc = NaN`, and `vus_pr = NaN`, that bundle is consistent with the implemented protocol and does not by itself prove either a strong or weak anomaly detector.
