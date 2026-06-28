---
date: 2026-06-27T08:27:21+00:00
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

- `precision`: 0.0
- `recall`: 0.0
- `pr_auc`: 0.0
- `roc_auc`: nan
- `threshold`: 2.415213108062744
- `vus_pr`: nan
- `vus_roc`: nan

## Causal Interpretation

The evaluated test vector contains both normal and anomalous labels, so the pointwise metrics are at least defined in the usual binary sense.

- The remaining question is then whether evaluated coverage truly matches the intended raw test timeline.
- The observed threshold was 2.415213.

## Raw Warnings From Audit Layer

- Test split window coverage is truncated relative to the raw test timeline. Evaluation artifacts do not cover the full labeled timeline.
- Configured max_test_windows=64 truncates the evaluated test timeline. Treat this as a truncated smoke evaluation, not a full-timeline test.

## Conclusion

The strongest repository-grounded conclusion is that this run is dominated by truncated early-prefix evaluation rather than by the full raw test timeline. In plain words, the evaluator only looked at an early slice of the test series, while the later suffix stayed outside evaluated coverage. That means weird metric bundles here must be interpreted first as protocol artifacts of partial coverage, not immediately as model quality evidence.
