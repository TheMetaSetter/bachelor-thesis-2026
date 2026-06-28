---
date: 2026-06-26 13:42:22 +07 +0700
researcher: Artificial Intelligence Agent
git_commit: 89a598f643cf0c20b0ab540b926e6b71f27e975f
branch: dev
repository: bachelor-thesis-2026
topic: "Deep audit of metric degeneracy and evaluation visualization semantics for SMD smoke runs and future forensic visualization work"
tags: [research, time-series, anomaly-detection, metrics, evaluation, smd, visualization]
status: complete
last_updated: 2026-06-26
last_updated_by: Artificial Intelligence Agent
---

# Research: Deep audit of metric degeneracy and evaluation visualization semantics for SMD smoke runs and future forensic visualization work

**Date**: 2026-06-26 13:42:22 +07 +0700  
**Researcher**: Artificial Intelligence Agent  
**Git Commit**: `89a598f643cf0c20b0ab540b926e6b71f27e975f`  
**Branch**: `dev`

## Research Question

Use the repository's research workflow to inspect the metric pipeline as deeply as possible, with special attention to suspicious combinations such as degenerate recall and AUC values, the correctness of evaluation labels and overlap reconstruction, and what a trustworthy visualization workflow would need to show for SMD and later SWaT inspection.

## Summary

The strongest repository-grounded finding is not that the current local smoke outputs are using all-positive labels. The stronger finding is that the inspected SMD smoke evaluations are being run on a **truncated prefix of the test window stream** because the smoke experiment configurations explicitly set `max_test_windows: 64`, and `scripts/evaluate.py` rebuilds the dataset bundle directly from that resolved config before evaluation. Since the active SMD loader materializes overlapping windows in increasing start-index order and stops once `max_windows` is reached, these smoke evaluations only cover the earliest segment of the test timeline rather than the full sequence. For the inspected entities, the first real anomaly occurs **after** that prefix, so the saved `evaluation_records.json` labels become all-zero over the reconstructed coverage, which in turn produces `roc_auc = NaN`, `pr_auc = 0.0`, `recall = 0.0`, and `vus_pr = NaN`.

This means the local smoke artifacts currently found in `outputs/` are not evidence that the models fail on the full test timeline. They are evidence that the **evaluation scope is truncated by config** and that downstream metric files and visualization files inherit that truncation without surfacing it clearly. A second important finding is that the existing visualization script overlays "ground truth" from `evaluation_record["point_labels"]`, not from the raw SMD test labels reloaded from the parser. Therefore, the current visualization path can make a truncated or degenerate evaluation look visually legitimate instead of exposing that only a prefix was evaluated.

## Detailed Findings

### Data Preparation

SMD parsing is implemented in [`src/data/datasets/smd.py:14-180`](../../../../src/data/datasets/smd.py#L14). For each selected entity, the parser loads:

- `train/<entity>.txt` into a feature matrix,
- `test/<entity>.txt` into a test feature matrix,
- `test_label/<entity>.txt` into a pointwise binary anomaly vector.

The parser explicitly assigns:

- `train.point_labels = 0` for the train split,
- `val.point_labels = 0` for the validation split,
- `test.point_labels = test_labels.clone()` for the test split.

So the raw SMD test labels are present and are not inherently lost at parse time.

The cleaning pipeline in [`src/data/cleaning.py:1-38`](../../../../src/data/cleaning.py#L1) is conservative and does not rewrite labels. The scaler in [`src/data/scalers.py:1-53`](../../../../src/data/scalers.py#L1) only standardizes `x` and preserves the rest of the sequence dictionary unchanged. This means the raw-sequence path does **not** explain the loss of anomalous labels in the inspected evaluation artifacts.

Window generation is implemented in [`src/data/loaders.py:177-220`](../../../../src/data/loaders.py#L177). The important behavior is:

1. windows are emitted in increasing `start_index` order;
2. stride is controlled by config;
3. if `max_windows` is set, construction stops immediately once that limit is reached.

This early stop is decisive for smoke semantics. With `window_size = 20`, `stride = 1`, and `max_test_windows = 64`, only windows with:

- `start_index = 0, 1, ..., 63`
- `end_index = 20, 21, ..., 83`

are materialized. Therefore the evaluated timeline coverage is only the prefix up to timestep index `82` inclusive, even though the test sequence itself is much longer.

### Modeling and Training

The pointwise anomaly score for both active offline models is reconstruction-based:

- [`src/models/redlamp_mlp_baseline.py:376-384`](../../../../src/models/redlamp_mlp_baseline.py#L376)
- [`src/models/thesis_multitask.py:2438-2448`](../../../../src/models/thesis_multitask.py#L2438)

Both compute:

```python
point_scores = torch.mean((recon - batch["x"]) ** 2, dim=-1)
window_scores = point_scores.mean(dim=1)
```

So the score construction is consistent across the two compared model families for the audit target considered here.

The training-time realistic validation path in the trainer is not the same as the offline test evaluation path. It reconstructs pointwise payloads from validation batches and may use synthetic supervision fields depending on stage semantics, as shown in [`src/engine/trainer.py:401-535`](../../../../src/engine/trainer.py#L401). That path is relevant for checkpoint monitoring, but it is not the same code that wrote the local `evaluation_records.json` files inspected in this audit. Therefore the local offline artifact investigation must prioritize `scripts/evaluate.py` plus `src/engine/evaluator.py`.

### Evaluation

The offline evaluation entrypoint is [`scripts/evaluate.py:79-178`](../../../../scripts/evaluate.py#L79). The critical behavior is:

1. it rebuilds the dataset bundle directly from `experiment_config["data"]`;
2. that data config includes smoke overrides such as `max_test_windows`;
3. it then passes `data_bundle["loaders"]["test"]` into `Evaluator.evaluate(...)`.

So evaluation does not silently "switch to full test" at runtime. It uses whatever test-window cap the resolved config already contains.

The evaluator in [`src/engine/evaluator.py:199-295`](../../../../src/engine/evaluator.py#L199) appends, for every batch:

```python
{
    "meta": batch["meta"],
    "point_scores": point_scores,
    "point_labels": batch["point_labels"].detach().cpu(),
}
```

It then reconstructs per-entity pointwise records via [`accumulate_pointwise_window_payload(...)`](../../../../src/engine/evaluator.py#L44), which:

- sums overlapping scores,
- counts overlap multiplicity,
- combines labels with `torch.maximum(...)`.

After reconstruction, it concatenates the reconstructed per-entity arrays, selects a threshold with [`select_point_score_threshold(...)`](../../../../src/engine/evaluator.py#L23), and computes metrics with [`compute_pointwise_metrics(...)`](../../../../src/metrics/pointwise.py#L505).

The metric helper itself is simple and does not introduce the inspected smoke failure mode:

- thresholded predictions are `score > threshold`,
- `roc_auc` and `pr_auc` are computed from the continuous scores,
- `precision`, `recall`, and `f1` are computed from the thresholded binary predictions,
- `vus_pr` and `vus_roc` return `NaN` when the label array contains fewer than two classes.

The degenerate local smoke artifacts therefore arise from the **evaluation arrays presented to the metric helper**, not from a hidden formula inside the helper.

### Concrete Artifact Cross-Checks

Three local output directories were inspected:

1. `outputs/comparative/smd_smoke/thesis_multitask/machine_1_6/seed6`
2. `outputs/comparative/smd_smoke/redlamp_mlp_baseline/machine_1_6/seed6`
3. `outputs/smd_offline_pretraining_three_stage_machine_3_4_window20_smoke_m1_20260624T200210`

For these runs, `evaluation_records.json` contained reconstructed `point_labels` that were all zero for the inspected entity, while the raw SMD parser returned non-zero anomaly counts:

- `machine-1-6`: raw test anomaly count `3708`, first anomaly at timestep `246`
- `machine-3-4`: raw test anomaly count `977`, first anomaly at timestep `2734`

The resolved experiment configs for those runs explicitly contain:

- `max_test_windows: 64`

in:

- [`configs/experiment/comparative/thesis/smd__thesis_multitask__comparative-three-stage-machine_1_6__w20__seed6__smoke.yaml:14-19`](../../../../configs/experiment/comparative/thesis/smd__thesis_multitask__comparative-three-stage-machine_1_6__w20__seed6__smoke.yaml#L14)
- [`configs/experiment/comparative/baseline/smd__redlamp_mlp_baseline__comparative-single-stage-machine_1_6__w20__seed6__smoke.yaml:14-19`](../../../../configs/experiment/comparative/baseline/smd__redlamp_mlp_baseline__comparative-single-stage-machine_1_6__w20__seed6__smoke.yaml#L14)
- [`configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml:15-19`](../../../../configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml#L15)

Given `window_size = 20` and `stride = 1`, these smoke evaluations do not reach timestep `246`, let alone `2734`. Therefore the observed artifact pattern:

- `point_labels` all zero in `evaluation_records.json`,
- `roc_auc = NaN`,
- `pr_auc = 0.0`,
- `recall = 0.0`,
- `vus_pr = NaN`

is consistent with the current smoke configuration and does **not** require a model bug to explain it.

### Visualization

The current visualization script is [`scripts/visualize_evaluation_results.py:1-224`](../../../../scripts/visualize_evaluation_results.py#L1). It reloads the raw SMD test sequence from the parser, but it does **not** use the raw parser labels as the blue ground-truth overlay. Instead, it takes:

```python
point_labels = torch.tensor(evaluation_record["point_labels"], dtype=torch.long)
```

from the saved evaluation record.

That means the plot can visually endorse a degenerate or truncated evaluation record as if it were the true ground truth. In the inspected smoke scenario, this is especially misleading because the saved `evaluation_record["point_labels"]` are all zero only because the run evaluated an early prefix without anomalies, not because the raw SMD test sequence is anomaly-free.

Therefore, the existing visualization path is useful as a score-viewer, but it is **not yet a forensic ground-truth checker**.

## Code References

- [`src/data/datasets/smd.py:14-180`](../../../../src/data/datasets/smd.py#L14) - SMD parser and raw test-label attachment
- [`src/data/loaders.py:65-87`](../../../../src/data/loaders.py#L65) - window dataset construction with per-split `max_windows`
- [`src/data/loaders.py:177-220`](../../../../src/data/loaders.py#L177) - ordered window materialization with early stop on `max_windows`
- [`scripts/evaluate.py:79-138`](../../../../scripts/evaluate.py#L79) - evaluation rebuilds dataset bundle from resolved config
- [`src/engine/evaluator.py:44-128`](../../../../src/engine/evaluator.py#L44) - overlap-aware reconstruction of scores and labels
- [`src/engine/evaluator.py:199-295`](../../../../src/engine/evaluator.py#L199) - offline evaluation loop and metric call
- [`src/metrics/pointwise.py:505-563`](../../../../src/metrics/pointwise.py#L505) - pointwise metric computation
- [`scripts/visualize_evaluation_results.py:49-126`](../../../../scripts/visualize_evaluation_results.py#L49) - current per-entity visualization overlays saved evaluation-record labels
- [`configs/experiment/comparative/thesis/smd__thesis_multitask__comparative-three-stage-machine_1_6__w20__seed6__smoke.yaml:14-19`](../../../../configs/experiment/comparative/thesis/smd__thesis_multitask__comparative-three-stage-machine_1_6__w20__seed6__smoke.yaml#L14) - smoke test-window cap
- [`configs/experiment/comparative/baseline/smd__redlamp_mlp_baseline__comparative-single-stage-machine_1_6__w20__seed6__smoke.yaml:14-19`](../../../../configs/experiment/comparative/baseline/smd__redlamp_mlp_baseline__comparative-single-stage-machine_1_6__w20__seed6__smoke.yaml#L14) - smoke test-window cap
- [`configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml:15-19`](../../../../configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml#L15) - smoke test-window cap

## Pipeline Documentation

For the inspected SMD smoke runs, the active runtime path is:

`resolved smoke experiment config -> build_dataset(data, including max_test_windows) -> WindowDataset ordered test prefix -> collate_windows -> model.test_step -> Evaluator overlap reconstruction -> threshold by score quantile -> pointwise metrics -> evaluation_records.json/evaluation_metrics.json -> visualization script overlays saved reconstructed labels`

The most important semantic consequence is:

- `evaluation_records.json` does not necessarily represent the full raw test timeline;
- it represents only whatever subset of windows the resolved config materialized;
- because reconstruction allocates a full-length output vector, uncovered timesteps remain zero-labeled and zero-scored by construction.

This full-length output shape can hide coverage truncation unless the visualization explicitly shows the covered prefix and overlap counts.

## Historical Context (from documents/)

The design documents and prior research notes consistently treat SMD evaluation as overlap-aware and timeline-based rather than naive window-flattened evaluation. That intent is reflected in the evaluator structure. However, the smoke experiment configs deliberately cap the number of windows for speed, and this practical smoke shortcut changes the semantic meaning of the produced evaluation artifacts. The current visualization script appears to have been written as a reader-facing plotter for saved evaluation artifacts, not as a deep correctness checker for evaluation coverage or label provenance.

This explains why a human reader could over-trust a smoke artifact that is technically consistent with its config but semantically incomplete for full test interpretation.

## Open Questions

1. The earlier user-reported metric combination with `recall = 0.05`, `precision = 1`, `pr_auc = 1`, and `threshold = 0.072061` was not found among the current local SMD smoke artifacts inspected in this pass. It may come from a different run, a different dataset, or a different artifact not currently traced.
2. The local smoke artifacts confirm a strong **all-zero prefix-evaluation** failure mode. They do not yet confirm whether any full-length non-smoke evaluation run has an independent label-lineage bug.
3. The visualization path should be audited further to decide whether future forensic plots should overlay:
   - raw parser labels,
   - evaluation-record labels,
   - or both simultaneously with an explicit mismatch panel.
4. SWaT local CSV files exist and expose a `Normal/Attack` column, but SWaT is still not part of the active thesis runtime registry. Any SWaT visualization in the next phase should be treated as direct CSV exploration rather than active runtime evaluation evidence.

## Follow-up 2026-06-26 13:42:22 +07 +0700

### Command-specific audit: `anomaly_archive__redlamp_mlp_baseline__staffiii-window20-adamw-cosine-warmup10-vus-pr-confmat__w20__seed11__default.yaml`

The user later provided the exact evaluation command:

```bash
/root/bachelor-thesis-2026/scripts/evaluate.py \
  --experiment-config configs/experiment/scale/anomaly_archive__redlamp_mlp_baseline__staffiii-window20-adamw-cosine-warmup10-vus-pr-confmat__w20__seed11__default.yaml \
  --checkpoint-path outputs/anomaly_archive_redlamp_mlp_baseline_staffiii_window20_adamw_cosine_warmup10_vus_pr_confmat/checkpoints/best.pt
```

This command changes the diagnosis materially relative to the SMD smoke findings above.

The resolved experiment config uses:

- [`configs/experiment/scale/anomaly_archive__redlamp_mlp_baseline__staffiii-window20-adamw-cosine-warmup10-vus-pr-confmat__w20__seed11__default.yaml:12`](../../../../configs/experiment/scale/anomaly_archive__redlamp_mlp_baseline__staffiii-window20-adamw-cosine-warmup10-vus-pr-confmat__w20__seed11__default.yaml#L12)
- [`configs/data/anomaly_archive_staffiii_full.yaml:1-10`](../../../../configs/data/anomaly_archive_staffiii_full.yaml#L1)

The crucial data semantics are:

- `dataset_name: anomaly_archive`
- `comparison_mode: pre_vs_anomaly`
- `inclusive_anomaly_end: false`
- `window_size: 20`
- `stride: 10`

Under these settings, the parser in [`src/data/datasets/anomaly_archive.py:85-162`](../../../../src/data/datasets/anomaly_archive.py#L85):

1. uses the pre-anomaly region as train/val,
2. uses only the annotated anomaly interval as the test sequence,
3. assigns:

```python
test_point_labels = torch.ones(test_values.size, dtype=torch.long)
```

when `comparison_mode == "pre_vs_anomaly"`.

This means the evaluated test sequence is **all-positive by construction** for this command.

The resolved local reproduction of the config confirms:

- test length = `220`
- unique test labels = `[1]`
- label sum = `220`
- number of test windows = `21`

Therefore the user-reported metric pattern:

- `recall = 0.05`
- `precision = 1`
- `pr_auc = 1`
- `roc_auc = NaN`
- `vus_pr = NaN`

is not merely plausible; it is the expected behavior of the current evaluation semantics.

Reason:

1. the label array seen by `compute_pointwise_metrics(...)` is all ones;
2. `roc_auc_score(...)` is undefined for single-class labels, so `_safe_metric(...)` returns `NaN`;
3. `compute_vus_pr_exact_naive(...)` returns `NaN` when `len(np.unique(label_array)) < 2`;
4. thresholding is based on the score quantile in [`src/engine/evaluator.py:23-41`](../../../../src/engine/evaluator.py#L23);
5. with an all-positive label array and a 95th-percentile threshold, roughly the top 5 percent of scored timesteps become predicted positives, so:

   - `precision = 1` because there are no negative labels to create false positives,
   - `recall ≈ 0.05` because only about 5 percent of positive timesteps are predicted positive,
   - `pr_auc = 1` becomes an artifact of the all-positive regime rather than evidence of a discriminative detector.

For this exact command, the primary issue is therefore **evaluation protocol semantics**, not first-line model failure.

Stated bluntly as a codebase investigation finding:

- if the repository wants meaningful ROC-AUC, PR-AUC, VUS-PR, or thresholded recall/precision for anomaly detection,
- then evaluating `anomaly_archive` with `comparison_mode: pre_vs_anomaly`
- and treating the isolated anomaly segment as the entire test sequence
- creates a single-class test label regime that invalidates several of the reported metrics before any model quality judgment can be made.
