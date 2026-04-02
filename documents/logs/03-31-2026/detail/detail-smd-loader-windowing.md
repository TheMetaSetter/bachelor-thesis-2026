---
date: 2026-03-31 16:48:00 +0700
planner: Artificial Intelligence Agent
git_commit: 597dc3a4a4a01f933e133424b78c67fdd51a75f0
branch: dev
repository: bachelor-thesis-2026
topic: "Updated detail document for the current offline SMD vertical slice and pre-Phase-4 gate"
tags: [detail, smd, windowing, baseline, multitask, pre-phase-4]
status: complete
last_updated: 2026-04-01
last_updated_by: Artificial Intelligence Agent
source_plan: documents/logs/03-31-2026/plan/detail-smd-loader-windowing.md
source_research: documents/logs/04-01-2026/research/research-pre-phase-4-ablation-readiness-and-branch-collapse-controls.md
---

# Detail: Updated detail document for the current offline SMD vertical slice and pre-Phase-4 gate

## Overview

Terminology normalized on 2026-04-02. Current design target: gate entropy regularization. Current implementation status: `src/models/thesis_multitask.py` now uses gate-entropy regularization directly while retaining the legacy margin field only for backward checkpoint compatibility.

This document rewrites the earlier detail note so that it matches the repository as it exists after the revised Phase 1 to Phase 3 closure work. The current offline path is no longer a placeholder-era vertical slice with split task files and external model-specific helper modules. It is now a registry-driven, one-model-one-file offline codebase with a reconstruction baseline, a multitask prototype-fusion model, CARLA-style synthetic anomaly injection, a maintained anomaly-visualization surface, and a design-level commitment to objective modularity.

The purpose of this document is therefore no longer to describe a hypothetical first slice. Its purpose is to document the current active offline structure, the design rules that now matter before Phase 4, and the remaining pre-Phase-4 work needed for ablation readiness.

## Current Architectural Contracts

The current repository still follows four stable runtime layers:

1. configuration
2. data
3. model
4. engine

The active batch contract remains:

```python
batch = {
    "x": Tensor[B, L, D],
    "point_labels": Optional[Tensor[B, L]],
    "mask": Optional[Tensor[B, L, D]],
    "timestamps": Optional[Tensor[B, L]],
    "meta": list[dict],
}
```

The active model-output contract remains:

```python
outputs = {
    "hidden": Tensor[B, L, H],
    "pooled": Optional[Tensor[B, H]],
    "recon": Optional[Tensor[B, L, D]],
    "logits": Optional[Tensor],
    "point_scores": Optional[Tensor[B, L]],
    "window_scores": Optional[Tensor[B]],
    "aux": dict,
}
```

The active step-output contract remains:

```python
step_output = {
    "loss": Tensor,
    "log": dict[str, float],
    "outputs": dict,
    "loss_terms": dict[str, Tensor],
    "batch": dict,
}
```

## Design Rules that Now Govern the Repository

- One model must live in one file.
- The reconstruction baseline keeps inference, score computation, and stage methods inside `src/models/reconstruction_mlp_ae.py`.
- The multitask prototype-fusion model keeps encoder logic, branch logic, fusion logic, objective helpers, and stage methods inside `src/models/thesis_multitask.py`.
- The active code path must not depend on model-specific files under `src/tasks/`, `src/losses/`, or `src/models/modules/`.
- `scripts/train.py` and `scripts/evaluate.py` must use the registry-driven dataset path.
- Synthetic anomaly generation and anomaly inspection belong to the offline multitask path and therefore remain pre-Phase-4 responsibilities.
- The multitask loss should remain objective-modular: a small default objective first, with extra regularizers added only when diagnostics justify them.
- Readability remains the primary constraint, in direct agreement with `codebase_preferences.md`.

## Current Active File Inventory

The current offline Phase 1 to Phase 3 path is centered on the following files:

```text
configs/data/smd.yaml
configs/data/smd_smoke.yaml
configs/experiment/smd_vertical_slice.yaml
configs/experiment/smd_smoke_test.yaml
configs/model/reconstruction_mlp_ae.yaml
configs/model/thesis_multitask.yaml
configs/task/reconstruction.yaml
configs/task/multitask_tsad.yaml
src/core/config.py
src/core/contracts.py
src/core/registry.py
src/core/seed.py
src/data/base.py
src/data/scalers.py
src/data/window.py
src/data/collate.py
src/data/loaders.py
src/data/datasets/smd.py
src/data/augment.py
src/models/base_model.py
src/models/reconstruction_mlp_ae.py
src/models/thesis_multitask.py
src/metrics/pointwise.py
src/engine/trainer.py
src/engine/evaluator.py
src/engine/checkpoint.py
src/engine/logger.py
scripts/train.py
scripts/evaluate.py
scripts/visualize_synthetic_anomalies.py
tests/test_config_loading.py
tests/test_smd_dataset_shapes.py
tests/test_windowizer.py
tests/test_model_shapes.py
tests/test_one_train_step.py
tests/test_checkpoint_roundtrip.py
tests/test_registry.py
tests/test_multitask_shapes.py
tests/test_one_multitask_train_step.py
tests/test_synthetic_anomaly_injection.py
tests/test_synthetic_anomaly_visualization.py
configs/experiment/smd_multitask.yaml
configs/experiment/smd_multitask_smoke.yaml
configs/experiment/smd_multitask_continuous_only.yaml
configs/experiment/smd_multitask_discrete_only.yaml
configs/experiment/smd_multitask_fused.yaml
configs/experiment/smd_multitask_no_diversity.yaml
configs/experiment/smd_multitask_no_variance.yaml
configs/experiment/smd_multitask_no_covariance.yaml
configs/experiment/smd_multitask_no_usage.yaml
configs/experiment/smd_multitask_no_gate.yaml
configs/experiment/smd_multitask_no_augmentation.yaml
scripts/run_ablation.py
src/data/stream.py
src/models/online_adaptation.py
src/engine/online_loop.py
scripts/run_online_adaptation.py
tests/test_fusion_ablation_modes.py
tests/test_temperature_schedule.py
tests/test_ablation_runner.py
tests/test_online_stream.py
tests/test_online_adaptation_step.py
tests/test_online_state_roundtrip.py
tests/test_online_reference_checkpoint.py
tests/test_online_entrypoint.py
```

The previous expectation that `src/data/stream.py` and `configs/experiment/smd_multitask.yaml` were absent from the active path is no longer accurate for the current repository. Those files now exist together with the first conservative Phase 4 runtime scaffold, while the offline ablation-ready multitask family remains the gate that justifies that scaffold.

## Phase 1 - Current Closure State

The current reconstruction path is already aligned with the stricter reading of `codebase_preferences.md`.

- `src/models/reconstruction_mlp_ae.py` owns the active reconstruction logic.
- `src/engine/trainer.py` calls model-owned `training_step` and `validation_step`.
- `src/engine/evaluator.py` calls model-owned `test_step`.
- `scripts/train.py` and `scripts/evaluate.py` both construct the data path through `build_dataset(...)`.
- The checkpoint, config-loading, shape, and registry tests exist and run on the active path.

This means the earlier Phase 1 design objective is no longer to create the offline slice. That slice already exists. The relevant detail concern before Phase 4 is to preserve this narrow path without reintroducing split logic or extra hidden construction paths.

## Phase 2 - Current Closure State

The current multitask model is no longer a placeholder reservation. It is the active owner of:

- continuous prototype retrieval
- discrete Gumbel-Softmax codebook assignment
- task-specific fusion through `alpha` and `beta`
- reconstruction and classification heads
- a modular weighted-sum offline objective surface
- stage logging and stage methods

This means the earlier Phase 2 direction toward one-model-one-file is already realized in active code. The detail work before Phase 4 is therefore not to move more logic into the model boundary, but to preserve that boundary while making its objective surface more sweepable for ablations and more incremental in how regularizers are introduced.

## Phase 3 - Current Closure State

The current repository already contains:

- CARLA-style subsequence anomaly families in `src/data/augment.py`
- active model-owned consumption of augmented batches in `src/models/thesis_multitask.py`
- user-visible anomaly inspection in `scripts/visualize_synthetic_anomalies.py`
- tests for injection, visualization, shapes, and one multitask train step

The main detail concern before Phase 4 is not whether augmentation exists. It is whether the augmentation and fusion controls are exposed clearly enough for large ablation studies and later thesis reporting.

That remaining concern is now addressed in code through explicit multitask ablation YAMLs, trainer-driven fusion warm-up, temperature annealing, a compact ablation runner, and JSONL metrics that include discrete-usage concentration and schedule state.

## Updated Pre-Phase-4 Priorities

Before online adaptation begins, the repository should close the following offline engineering priorities:

1. Preserve the active one-model-one-file structure.
2. Preserve the registry-only dataset and model construction path.
3. Keep synthetic anomaly generation explicit and inspectable.
4. Add an extensive ablation surface around the existing multitask model rather than adding new model variants.
5. Make branch-collapse controls observable and schedule-aware.
6. Keep the default loss minimal and add regularizers only against observed failure modes.
7. Bring the detail documents into direct agreement with `codebase_preferences.md` and the actual repository tree.

## Branch-Collapse Controls that Already Exist

The current code already implements the following branch-collapse controls inside `src/models/thesis_multitask.py`:

- cross-branch decorrelation loss
- branch-wise variance floor loss
- branch-wise covariance reduction loss
- discrete usage balancing loss
- gate entropy regularization on `alpha` and `beta`
- logging of `alpha`, `beta`, continuous norm, and discrete norm

This means the current repository already contains the main mathematical ingredients described in `documents/design/idea.md`. The earlier pre-Phase-4 gaps around schedule control, ablation YAMLs, and diagnostics have now been implemented in the multitask path. The remaining deferred work is no longer basic gate closure. It is later-slice scope such as drift injection, broader online adaptation strategies, and NGD-style optimization.

## Pre-Phase-4 Ablation Readiness

The repository should next support extensive ablations without introducing new codepaths. The concrete detail checklist for that work now lives in:

`documents/logs/04-01-2026/detail/detail-pre-phase-4-ablation-readiness-checklist.md`

The most important implementation targets from that checklist are now present in code:

- explicit experiment YAMLs for continuous-only, discrete-only, fused, and loss-drop variants
- trainer-level support for fusion warm-up and temperature annealing
- exact limiting-case tests for `alpha = beta = 0` and `alpha = beta = 1`
- script-level support for repeatable ablation runs and compact result summaries
- reproducibility improvements around resolved-config persistence and optional Weights & Biases logging

## What This Document Intentionally Removes from the Older Version

This rewrite intentionally removes or demotes several assumptions from the earlier version of this file:

- `src/data/stream.py` is now treated as part of the active conservative Phase 4 scaffold rather than as a hypothetical future placeholder.
- The repository is no longer described as if `ThesisMultiTaskModel` were still partly placeholder-based.
- The repository is no longer described as if `src/tasks/`, model-specific `src/losses/`, or model-specific `src/models/modules/` were acceptable active dependencies.
- The repository is no longer described as if the main remaining need before Phase 4 were basic vertical-slice construction.

## Completion Standard

This updated detail document is complete when it accurately describes the repository as it exists now:

- an active offline SMD codebase with one-model-one-file multitask logic
- explicit synthetic anomaly generation and visualization
- a Phase 1 to Phase 3 path that is already runnable and ablation-ready from YAML alone
- a conservative Phase 4 scaffold built around `src/data/stream.py`, `src/models/online_adaptation.py`, `src/engine/online_loop.py`, and `scripts/run_online_adaptation.py`
- explicit acknowledgment that drift injection, encoder unfreezing, and NGD-style optimization remain deferred beyond the first online slice

Phase 4 is no longer absent from the repository. The current state is instead a gate-respecting first Phase 4 slice: projector-first, clean-stream-only, checkpointable, and dependent on the canonical offline multitask checkpoint from `configs/experiment/smd_multitask.yaml`.
