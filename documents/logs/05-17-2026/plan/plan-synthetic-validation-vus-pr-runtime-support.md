---
date: 2026-05-17 21:27:16 +07+0700
researcher: TheMetaSetter
git_commit: 5012240e27ce4ad2b1ee613e3c328a4b26b685a2
branch: dev
repository: bachelor-thesis-2026
topic: "Implementation plan for VUS-PR computation on synthetic anomalous validation windows"
tags: [plan, time-series, anomaly-detection, multi-class, validation]
status: draft
last_updated: 2026-05-17
last_updated_by: TheMetaSetter
research_source: documents/logs/05-17-2026/research/research-codebase-state-before-planning-training-optimization-runtime-support.md
---

# Plan: Implementation plan for VUS-PR computation on synthetic anomalous validation windows

**Date**: 2026-05-17 21:27:16 +07+0700
**Researcher**: TheMetaSetter
**Git Commit**: 5012240e27ce4ad2b1ee613e3c328a4b26b685a2
**Branch**: dev

## Plan Objective

The objective of this planning document is to define an implementation path for computing a validation metric named `val_synth_vus_pr`, where the metric is evaluated on validation windows after synthetic anomaly injection rather than on the clean validation split. The plan must preserve the repository's current contracts, remain configuration-driven, and support future use of `val_synth_vus_pr` as a checkpoint-monitor metric without changing the cosine scheduler semantics.

## Current State

- The repository already exposes a standardized batch contract through the SMD window dataset. Batches contain `x`, `point_labels`, `mask`, `timestamps`, and `meta`, and the `meta` dictionary includes window boundaries that support overlap-aware reconstruction [src/data/loaders.py](/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/data/loaders.py:83).
- The SMD validation split is currently created by slicing the original training series and assigning zero-valued point labels to the entire validation segment. Therefore, `val_vus_pr` computed on `val_loader` is a clean-validation metric and not a synthetic-anomaly metric [src/data/datasets/smd.py](/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/data/datasets/smd.py:149).
- The offline trainer already distinguishes between `validation_step()` and `synthetic_validation_step()`. Clean validation is executed first, and synthetic validation is executed second when supported by the model [src/engine/trainer.py](/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/trainer.py:408).
- The current synthetic validation path only aggregates classification metrics from `logits` and `classification_labels`, which yields metrics such as `val_synth_roc_auc` and `val_synth_pr_auc`. It does not reconstruct synthetic point-score timelines and does not compute `val_synth_vus_pr` [src/engine/trainer.py](/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/trainer.py:443).
- The overlap-aware pointwise metric implementation already exists in the evaluator. It reconstructs per-entity point scores from overlapping windows and computes thresholded pointwise metrics together with VUS-PR when configured [src/engine/evaluator.py](/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/evaluator.py:158).
- Runtime configuration validation already accepts explicit optimizer names, gradient clipping, cosine scheduling, and checkpoint-monitor metrics including `val_vus_pr`. The configuration layer does not yet recognize `val_synth_vus_pr` [src/core/config.py](/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/core/config.py:256).

## Established Contracts to Preserve

- The batch contract must remain a dictionary centered on `batch["x"]` with shape `[B, L, D]`, optional pointwise labels, and metadata describing entity and window boundaries.
- The encoder contract must remain unchanged. Models continue to expose the thesis-facing hidden representation through their existing model output schema rather than by introducing a second runtime-specific representation path.
- The model output contract must remain unchanged for offline models. Existing fields such as `hidden`, `pooled`, `recon`, `logits`, `point_scores`, `window_scores`, and `aux` remain authoritative.
- The trainer must continue to treat cosine learning-rate scheduling as an arithmetic per-batch policy that does not depend on any monitored validation metric.

## Design Options

- Option A: reuse the existing evaluator almost directly by adding a synthetic-validation evaluation helper that consumes outputs from `synthetic_validation_step()` and rebuilds synthetic point-score timelines from the synthetic batch metadata and masks.
- Option B: extract the evaluator's overlap-aware reconstruction logic into a smaller reusable utility, then let both the clean validation evaluator and a new synthetic validation evaluator call the same reconstruction and pointwise metric functions.
- Option C: extend `synthetic_validation_step()` outputs and trainer aggregation so that synthetic point-score records are collected during the validation loop and post-processed at epoch end without invoking `model.test_step()`.

## Recommended Direction

Option B aligns best with the current repository structure.

The current codebase already separates model-owned step logic from engine-owned metric aggregation. Extracting a small reusable point-score reconstruction utility from `src/engine/evaluator.py` preserves the stable model contracts while avoiding duplicated overlap-aware logic in `Trainer`. This approach also respects readability and keeps the number of codepaths limited. It allows clean validation and synthetic validation to share the same pointwise metric core, while differing only in the source of batch labels and batch outputs.

## Proposed Implementation Scope

### Phase 1: Make synthetic validation pointwise labels explicit

- Review the output of `synthetic_validation_step()` in [src/models/redlamp_mlp_baseline.py](/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/redlamp_mlp_baseline.py:306) and [src/models/thesis_multitask.py](/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/thesis_multitask.py:1962).
- Verify that the synthetic batch returned from these methods includes a pointwise target for anomaly localization, preferably `synthetic_anomaly_mask` or an equivalent tensor with shape `[B, L]`.
- If the pointwise synthetic target is not always exposed in a stable field, add one stable field under the existing batch dictionary rather than introducing a parallel synthetic-batch structure.

### Phase 2: Reuse overlap-aware point-score reconstruction

- Refactor [src/engine/evaluator.py](/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/evaluator.py:158) to expose a small helper that accepts:
  - batch metadata with `entity_id`, `start_index`, and `end_index`;
  - per-window `point_scores`;
  - per-window pointwise labels for the same windows.
- Keep the public evaluator interface intact for test-set evaluation.
- Use composition instead of inheritance. The trainer should call a helper or a small evaluation adapter rather than subclassing the evaluator.

### Phase 3: Add synthetic validation VUS computation to the trainer

- Extend [src/engine/trainer.py](/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/trainer.py:420) so that, during `val_synth` processing, the trainer retains:
  - synthetic point scores;
  - synthetic pointwise labels;
  - metadata needed for overlap-aware reconstruction.
- After the `val_synth` loop, compute a synthetic validation pointwise metric payload using the reused evaluator logic.
- Prefix these metrics with `val_synth_`, including at minimum:
  - `val_synth_vus_pr`
  - `val_synth_threshold`
  - any supporting pointwise metrics that help interpret the synthetic validation path.

### Phase 4: Extend configuration and checkpoint semantics

- Update [src/core/config.py](/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/core/config.py:256) so that `checkpoint_monitor_metric` accepts `val_synth_vus_pr`.
- If the repository keeps metric-driven schedulers, ensure `reduce_on_plateau` monitor validation remains unchanged. `val_synth_vus_pr` should affect checkpoint selection only unless explicitly chosen elsewhere in future work.
- Preserve the current design rule that cosine scheduling does not require a scheduler monitor metric.

### Phase 5: Add experiment-surface support

- Update relevant experiment YAML files under `configs/experiment/` to allow synthetic-VUS checkpoint experiments once the metric exists.
- Do not replace existing clean-validation VUS experiments automatically. Instead, create explicit experiment names or monitor settings so that future comparisons remain reproducible.

## Specific Files and Interfaces

- Modify [src/engine/evaluator.py](/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/evaluator.py) to expose reusable overlap-aware reconstruction helpers.
- Modify [src/engine/trainer.py](/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/trainer.py) to collect synthetic validation point-score state and emit `val_synth_vus_pr`.
- Potentially modify [src/models/redlamp_mlp_baseline.py](/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/redlamp_mlp_baseline.py) and [src/models/thesis_multitask.py](/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/thesis_multitask.py) if a stable pointwise synthetic label field must be guaranteed.
- Modify [src/core/config.py](/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/core/config.py) to validate `val_synth_vus_pr` as a legal checkpoint monitor.
- Add or extend tests in:
  - `tests/test_learning_rate_scheduler.py`
  - `tests/test_evaluator_thresholding.py`
  - `tests/test_multitask_validation_alignment.py`
  - a new focused file if synthetic VUS behavior becomes too large for existing test modules.

## Design Pattern and Engineering Guidance

- Use separation of concerns by keeping model files responsible for generating synthetic validation outputs, while keeping pointwise metric computation inside the engine layer.
- Use composition over inheritance. A small reusable helper is preferable to a second evaluator subclass with duplicated code.
- Preserve the registry and factory structure already used by the scripts. This work belongs at the engine and configuration layers, not at the registry surface.
- Preserve single responsibility. The trainer should coordinate loops and metric aggregation, not own duplicate implementations of overlap-aware reconstruction mathematics.
- Maintain stable interfaces. New synthetic pointwise labels should be added as explicit batch keys rather than through special-case branching on model type.

## Test Plan

- Add a unit test proving that synthetic validation pointwise labels can be reconstructed through the same overlap-aware mechanism used by the clean evaluator.
- Add a trainer-level test showing that one epoch with `synthetic_validation_step()` now records `val_synth_vus_pr`.
- Add a checkpoint-monitor test showing that `checkpoint_monitor_metric: val_synth_vus_pr` selects the checkpoint with the highest synthetic VUS-PR score.
- Preserve existing tests for:
  - `val_vus_pr` on clean validation;
  - `val_synth_pr_auc` on synthetic validation;
  - cosine scheduling behavior;
  - gradient clipping behavior.

## Validation Procedure

- Run focused tests for evaluator reconstruction, trainer metric emission, and checkpoint selection.
- Run the existing regression tests that cover offline training, checkpoint roundtrip, and pointwise metric computation.
- Execute a short smoke experiment on an SMD machine-specific configuration with reduced epochs and reduced window counts.
- Verify that the epoch log includes both `val_synth_pr_auc` and `val_synth_vus_pr`.
- Verify that cosine learning-rate updates remain batch-driven and do not depend on any validation metric.

## Risk and Mitigation

- Risk: synthetic pointwise labels are not consistently available across models. Mitigation: define one stable batch key for synthetic anomaly masks and require both active offline models to populate it during synthetic validation.
- Risk: overlap-aware reconstruction logic is duplicated between clean and synthetic validation. Mitigation: extract shared helper functions from the evaluator instead of reimplementing them in the trainer.
- Risk: synthetic VUS-PR may be numerically unstable on small smoke runs or windows without positive synthetic timesteps. Mitigation: document the expected NaN behavior for degenerate cases and test it explicitly.
- Risk: checkpoint semantics become confused with scheduler semantics. Mitigation: keep `checkpoint_monitor_metric` independent from cosine scheduling and preserve current plateau scheduler behavior unchanged.
- Risk: the synthetic validation path becomes model-specific. Mitigation: standardize the required synthetic pointwise label field at the batch contract level.

## Open Questions

- Should `val_synth_vus_pr` be computed from the model's ordinary `point_scores` field during `synthetic_validation_step()`, or should the repository explicitly distinguish clean-scoring and synthetic-scoring outputs if future models diverge?
- Should synthetic validation pointwise metrics be computed only when `use_synthetic_validation` is enabled, or should the trainer also guard on the presence of a stable synthetic pointwise label field?
- Once `val_synth_vus_pr` exists, should the RedLamp MLP baseline experiment configurations switch their `checkpoint_monitor_metric` immediately, or should both clean-VUS and synthetic-VUS experiment families be retained in parallel for comparison?

## Minimal Vertical Slice

Before any broader experiment-surface changes, the first implementation slice should be:

1. Standardize access to synthetic pointwise labels during `val_synth`.
2. Reuse overlap-aware reconstruction for those synthetic windows.
3. Emit `val_synth_vus_pr` in one trainer epoch.
4. Prove checkpoint selection can use `val_synth_vus_pr`.

This slice is sufficient to establish that the repository can evaluate synthetic anomalous validation windows with a thesis-facing pointwise metric without changing the encoder contract, batch contract, or cosine scheduler semantics.
