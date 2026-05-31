---
date: 2026-04-01 13:21:39 +0700
researcher: TheMetaSetter
git_commit: 597dc3a4a4a01f933e133424b78c67fdd51a75f0
branch: dev
repository: bachelor-thesis-2026
topic: "Pre-Phase-4 research on ablation readiness, detail-document alignment, and branch-collapse controls"
tags: [research, time-series, anomaly-detection, multi-class, ablation, fusion]
status: complete
last_updated: 2026-04-01
last_updated_by: TheMetaSetter
---

# Research: Pre-Phase-4 research on ablation readiness, detail-document alignment, and branch-collapse controls

**Date**: 2026-04-01 13:21:39 +0700
**Researcher**: TheMetaSetter
**Git Commit**: 597dc3a4a4a01f933e133424b78c67fdd51a75f0
**Branch**: dev

## Research Question

Before entering Phase 4, what relevant implementation surfaces already exist for extensive ablation study, how does the current repository align with the older detail document at `documents/logs/03-31-2026/detail/detail-smd-loader-windowing.md`, and what branch-collapse control mechanisms around the continuous and discrete branches are already implemented in the current codebase?

## Summary

Terminology normalized on 2026-04-02. Current design target: gate entropy regularization. Current implementation status: `src/models/thesis_multitask.py` now uses gate-entropy regularization directly while retaining the legacy margin field only for backward checkpoint compatibility.

The current repository already exposes the main offline multitask objective in a form that is structurally compatible with future ablations. The active multitask model keeps the continuous branch, discrete branch, fusion equations, reconstruction head, classification head, and stage-specific objective helpers in one file. The configuration surface already exposes prototype toggles, prototype counts, Gumbel-Softmax temperature, fusion-logit initialization, and per-term loss weights. The model also records branch-specific diagnostics, scalar fusion values, and per-term losses in the stage logs.

The main limitation is not the existence of ablation-relevant components, but the narrowness of the surrounding experiment surface. The repository does not currently contain a dedicated ablation runner, stage scheduler, or experiment configuration family for continuous-only, discrete-only, fused, warm-up, and temperature-annealing variants. The current implementation therefore documents and computes the correct ingredients for future ablations, but does not yet present them as an extensive experiment matrix.

The older detail document is now only partially aligned with the repository. Its architectural direction toward a registry-driven, one-model-one-file offline path is reflected in the current code. However, several file-level expectations from that document no longer match the active repository, such as the presence of `src/data/stream.py`, `configs/experiment/baseline/smd__thesis_multitask__multitask__w100__seed7__default.yaml`, and `tests/test_model_contracts.py`. The current code instead follows the later closure-oriented interpretation more closely.

## Detailed Findings

### Data Preparation

- The active data path remains centered on SMD parsing, scaling, and windowed loading through `src/data/datasets/smd.py` and `src/data/loaders.py`.
- The synthetic anomaly path is active through `src/data/augment.py`, where batch augmentation preserves the original batch shape while appending `classification_labels`, `synthetic_anomaly_mask`, and `augmentation_metadata`.
- The augmentation metadata is explicit and serialization-friendly. Each synthetic anomaly sample records anomaly family, start and end indices, affected channels, and family-specific parameters.
- The implemented anomaly families are `seasonal`, `trend`, `global`, `contextual`, and `shapelet`. These are subsequence-style families rather than the earlier local pointwise perturbation path.

### Modeling and Training

- The active multitask model is `ThesisMultitaskModel` in `src/models/thesis_multitask.py`.
- The encoder exposes the thesis-facing hidden representation with shape `[B, L, H]`.
- The continuous prototype branch computes soft prototype retrieval through a learnable prototype bank and token-level attention weights.
- The discrete branch computes relaxed assignments with `torch.nn.functional.gumbel_softmax` and reconstructs a quantized hidden state from a learnable codebook.
- The fusion path computes two separate fused states:
  - reconstruction path: `H_rec = beta * H_disc + (1 - beta) * H_cont`
  - classification path: `H_cls = alpha * H_disc + (1 - alpha) * H_cont`
- The model already implements the following objective helpers directly in the model file:
  - reconstruction loss
  - cross-entropy classification loss
  - cross-branch diversity loss
  - variance floor loss
  - covariance reduction loss
  - prototype usage loss
  - gate entropy regularization in the design surface, with a barrier-style gate helper still used in the current code
- The total offline loss is already a weighted sum of those components in the active training step.

### Evaluation and Reporting

- The offline scripts `scripts/train.py` and `scripts/evaluate.py` rebuild datasets and models through the registry path.
- The visualization script `scripts/visualize_synthetic_anomalies.py` produces a saved inspection artifact for augmented windows and their masks.
- The current logger writes serialized scalar metrics, but there is no dedicated ablation runner or experiment report aggregator at the script layer.

## Evidence Relevant to Ablation Readiness

### Existing ablation-friendly surfaces

- `configs/model/thesis_multitask.yaml` already exposes:
  - `continuous_enabled`
  - `continuous_num_prototypes`
  - `discrete_enabled`
  - `discrete_codebook_size`
  - `gumbel_temperature`
  - `alpha_logit_init`
  - `beta_logit_init`
  - `lambda_cls`
  - `lambda_div`
  - `lambda_var`
  - `lambda_cov`
  - `lambda_use`
  - `lambda_gate`
  - `variance_floor_gamma`
  - `gate_barrier_margin`
- `configs/task/multitask_tsad.yaml` already exposes augmentation toggles and parameters:
  - `use_synthetic_augmentation`
  - `anomaly_probability`
  - `min_segment_fraction`
  - `max_segment_fraction`
  - `spike_scale`
- The model logs `alpha`, `beta`, `continuous_norm`, `discrete_norm`, and each active loss term during stage execution.
- The active tests already verify that the multitask path returns fusion diagnostics and that `alpha_logit` and `beta_logit` receive gradients.

### Missing ablation-oriented surfaces

- No `run_ablation.py` or equivalent experiment launcher exists in the active repository.
- No current experiment configuration family exists for:
  - continuous-only runs
  - discrete-only runs
  - fused runs
  - gate-off warm-up runs
  - temperature-annealing runs
  - individual loss-term dropouts
- No current scheduler surface exists for:
  - freezing `alpha` and `beta` during a warm-up stage
  - turning `lambda_gate` on only after warm-up when gate entropy regularization is needed
  - annealing `gumbel_temperature` across training
- No test currently checks that the limiting cases `alpha = beta = 0` and `alpha = beta = 1` reproduce exact continuous-only and discrete-only fusion behavior.

## Evidence Relevant to Branch-Collapse Controls

### Implemented controls

- The continuous and discrete branches are both exposed in `outputs["aux"]`.
- Cross-branch decorrelation is implemented as `_compute_cross_branch_diversity_loss`.
- Branch-wise variance stabilization is implemented as `_compute_variance_floor_loss`.
- Branch-wise covariance reduction is implemented as `_compute_covariance_reduction_loss`.
- Discrete usage balancing is implemented as `_compute_prototype_usage_loss`.
- Fusion saturation is discouraged by `_compute_gate_regularization_loss` in the current code, while the design target is gate entropy regularization.
- Stage logs include `alpha`, `beta`, and branch norms, which makes the active code path observable during training.

### Controls present in design documents but not yet operationalized as staged behavior

- The design documents describe a short warm-up with `alpha = beta = 0.5` frozen and `lambda_gate = 0`.
- The design documents also describe annealing the Gumbel temperature from a softer initial value toward a sharper final value.
- The current code exposes the scalar and temperature parameters, but the trainer does not currently implement staged freezing, staged unfreezing, or annealing logic.
- The model contains a direct `TODO` comment inside `_compute_fusion_outputs` indicating that branch-collapse strategy is not fully finalized at the code level.

## Alignment with the Older Detail Document

### Areas that remain aligned

- The repository follows the one-model-one-file interpretation for both the reconstruction and multitask models.
- The scripts use the registry-driven dataset path rather than bypassing it with a direct active constructor call.
- The multitask model owns its active branch logic, fusion logic, and objective helpers.
- A maintained synthetic anomaly visualization script now exists.

### Areas that are no longer literally aligned

- The older detail document still expects `src/data/stream.py`, but that file is no longer part of the active repository.
- The older detail document expects `configs/experiment/baseline/smd__thesis_multitask__multitask__w100__seed7__default.yaml`, but the active repository does not currently contain that experiment file.
- The older detail document expects `tests/test_model_contracts.py`, while the active repository instead uses `tests/test_multitask_shapes.py` and related targeted tests.
- The current repository has moved beyond placeholder multitask sections. The multitask file now owns real prototype, fusion, and loss logic rather than only reserving placeholders.

## Code References

- `src/models/thesis_multitask.py:41` - multitask model definition
- `src/models/thesis_multitask.py:143` - continuous prototype branch
- `src/models/thesis_multitask.py:173` - discrete prototype branch
- `src/models/thesis_multitask.py:208` - fusion scalar computation for `alpha` and `beta`
- `src/models/thesis_multitask.py:332` - reconstruction and cross-entropy classification losses
- `src/models/thesis_multitask.py:346` - cross-branch diversity loss
- `src/models/thesis_multitask.py:355` - variance floor loss
- `src/models/thesis_multitask.py:366` - covariance reduction loss
- `src/models/thesis_multitask.py:385` - discrete usage loss
- `src/models/thesis_multitask.py:393` - current barrier-style gate helper; design target is gate entropy regularization
- `src/models/thesis_multitask.py:400` - stage logging of fusion and branch diagnostics
- `src/models/thesis_multitask.py:431` - total multitask objective assembly
- `src/data/augment.py:8` - synthetic anomaly injector
- `src/data/augment.py:81` - family-specific subsequence injection
- `src/data/augment.py:189` - batch augmentation contract
- `configs/model/thesis_multitask.yaml:1` - multitask ablation-facing configuration surface
- `configs/task/multitask_tsad.yaml:1` - augmentation-facing task configuration
- `scripts/visualize_synthetic_anomalies.py:19` - anomaly visualization helper
- `tests/test_multitask_shapes.py:8` - multitask contract test
- `tests/test_one_multitask_train_step.py:8` - multitask train-step test with gate gradients

## Pipeline Documentation

The active offline multitask path takes a batch with `x` shaped `[B, L, D]`, prepares synthetic anomaly labels and masks when training without pre-supplied multitask labels, computes hidden states with the encoder, forms continuous and discrete branch states, fuses them into reconstruction and classification task states, computes decoder reconstruction and pooled classification logits, and then evaluates the weighted objective over reconstruction, classification, branch decorrelation, branch variance, branch covariance, discrete usage, and gate entropy regularization directly in code.

The active implementation therefore already preserves the same conceptual separation needed for future ablations: branch construction, branch fusion, prediction heads, and loss terms are distinct surfaces inside one model file. The missing layer is an experiment-management surface that systematically sweeps those surfaces.

## Historical Context (from documents/)

- `documents/design/idea.md` and `documents/design/design_starter.md` both describe the intended fused-task formulation in which only `H_rec` and `H_cls` drive the prediction heads, while pre-fusion branch outputs remain observable for regularization and ablations.
- The current multitask model already follows that high-level contract.
- The design documents describe exact future ablations as limiting cases of the same model:
  - continuous-only with `alpha = beta = 0`
  - discrete-only with `alpha = beta = 1`
  - fused with learned `alpha` and `beta`
- The design documents also describe a warm-up stage and Gumbel-temperature annealing, but those training-stage controls are not yet surfaced in the trainer or experiment scripts.

## Open Questions

- The design documents describe branch-collapse mitigation through staged training and temperature scheduling, but the current code only exposes static parameters for those controls.
- The current repository exposes the ingredients for extensive ablations, but not yet the script and configuration layer that would run them systematically.
- The older detail document remains directionally useful, but it no longer matches the active file inventory in several places and would need rewriting if it is meant to document the repository as it exists today.
