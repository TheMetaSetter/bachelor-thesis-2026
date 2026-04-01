---
date: 2026-03-31 16:48:00 +0700
planner: Artificial Intelligence Agent
git_commit: 7779c876c7da79c961ec7ac18f710620d5172533
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed implementation plan for the minimum runnable SMD vertical slice"
tags: [detail, smd, loader, stream, windowing, baseline, evaluation]
status: complete
last_updated: 2026-03-31
last_updated_by: Artificial Intelligence Agent
source_plan: documents/logs/03-31-2026/plans/plan-smd-loader-windowing.md
source_structure: documents/logs/03-31-2026/plans/structure-smd-loader-windowing.md
---

# Detail: Detailed implementation plan for the minimum runnable SMD vertical slice

## Overview

This detailed plan updates the earlier phased implementation order so that the debt captured in `documents/logs/03-31-2026/research/research-problems-1-3-5-6-codebase-preferences-alignment.md` is closed before any Phase 4 online-adaptation work begins. In particular, problem 1, problem 3, problem 5, and problem 6 must be resolved inside phases 1 to 3 or in an explicit pre-Phase-4 gate.

The implementation must still prioritize a narrow but complete offline vertical slice first. However, the vertical slice must now obey the stricter reading of `codebase_preferences.md`: one model per file, with forward logic, score computation, and stage-specific losses colocated for readability. The online-adaptation stage remains later work and is blocked until the earlier model-file, augmentation, visualization, and registry debt is closed.

## Global Architectural Contracts

The codebase should be organized around four stable runtime layers: configuration, data, model, and engine. Reporting and ablation scripts can sit on top of those layers, but the core runtime path should not introduce an extra task layer that splits one model across multiple files.

### Dataset and window contracts

The raw dataset contract should be:

```python
raw_sequence = {
    "x": Tensor[T, D],
    "point_labels": Optional[Tensor[T]],
    "mask": Optional[Tensor[T, D]],
    "timestamps": Optional[Tensor[T]],
    "meta": {
        "dataset_name": str,
        "entity_id": str,
        "split": str,
        "num_channels": int,
        "sequence_length": int,
    },
}
```

The window contract should be:

```python
window = {
    "x": Tensor[L, D],
    "point_labels": Optional[Tensor[L]],
    "mask": Optional[Tensor[L, D]],
    "timestamps": Optional[Tensor[L]],
    "meta": {
        "dataset_name": str,
        "entity_id": str,
        "split": str,
        "start_index": int,
        "end_index": int,
        "window_size": int,
        "stride": int,
    },
}
```

### Batch, model-output, and step-output contracts

The batched input contract should be:

```python
batch = {
    "x": Tensor[B, L, D],
    "point_labels": Optional[Tensor[B, L]],
    "mask": Optional[Tensor[B, L, D]],
    "timestamps": Optional[Tensor[B, L]],
    "meta": list[dict],
}
```

Every model should return:

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

Every stage method should return a serialization-friendly dictionary:

```python
step_output = {
    "loss": Tensor,
    "log": dict[str, float],
    "outputs": dict,
}
```

### Design rules that now become mandatory

- One model must live in one file. If a file defines `ReconstructionMLPAutoencoder`, that file should also contain the reconstruction losses, score computation, and `training_step`, `validation_step`, and `test_step` methods for that model.
- The same rule applies to `ThesisMultiTaskModel` and the later online adaptation model. Model-specific prototype logic, fusion logic, projector logic, and stage-specific losses should not be split into separate `tasks/`, `losses/`, `modules/`, or `heads/` files.
- Datasets and models should be created through a minimal registry or factory. `scripts/train.py` and `scripts/evaluate.py` must not register a component and then bypass the registry with a direct constructor call.
- CARLA-aligned synthetic anomaly injection and user-visible anomaly inspection belong before Phase 4, because they are part of the offline multitask path rather than the online-adaptation path.
- Readability remains the primary goal. Repetition is acceptable if it keeps one model readable from top to bottom in a single file.

## Phase 1 - Build the minimum runnable SMD vertical slice

### Phase summary

This phase establishes the first end-to-end thesis slice for SMD: parser, split logic, scaler, stream, windowizer, dataloaders, one reconstruction baseline, one trainer, one evaluator, and checkpoint round-tripping. The main correction in this revised plan is that the reconstruction model is now fully self-contained in one model file and the scripts must use the dataset registry end-to-end.

### File-level edits

The following files should be created in this phase:

```text
configs/data/smd.yaml
configs/model/reconstruction_mlp_ae.yaml
configs/experiment/smd_reconstruction.yaml
src/core/config.py
src/core/contracts.py
src/core/registry.py
src/core/seed.py
src/data/base.py
src/data/scalers.py
src/data/window.py
src/data/collate.py
src/data/loaders.py
src/data/stream.py
src/data/datasets/smd.py
src/models/base_model.py
src/models/reconstruction_mlp_ae.py
src/metrics/pointwise.py
src/engine/trainer.py
src/engine/evaluator.py
src/engine/checkpoint.py
src/engine/logger.py
scripts/train.py
scripts/evaluate.py
tests/test_smd_dataset_shapes.py
tests/test_windowizer.py
tests/test_model_shapes.py
tests/test_one_train_step.py
tests/test_checkpoint_roundtrip.py
tests/test_config_loading.py
tests/test_registry.py
```

### Explicit edit content

`src/core/config.py` should implement `load_yaml_config`, `load_experiment_config`, and `validate_experiment_config`. Validation should reject missing sections, mismatched file references, and invalid scalar types for windowing and optimization settings.

`src/core/contracts.py` should define `validate_raw_sequence`, `validate_window`, `validate_batch`, `validate_model_outputs`, and `validate_evaluation_record`. Each function should verify required keys, tensor rank, tensor axis order, and metadata fields.

`src/core/registry.py` should define small registries for datasets and models. The implementation should remain dictionary-based and readable, with explicit `register_*` and `build_*` functions.

`src/data/datasets/smd.py` should define `SMDDatasetParser`. It should read `data/ServerMachineDataset/train`, `test`, and `test_label`, preserve one machine per entity, create a per-machine validation tail split, and emit native `raw_sequence` dictionaries without embedding scaling or windowing logic.

`src/data/scalers.py` should define `SequenceStandardScaler`. The scaler should fit only on the post-split training sequences, store featurewise statistics, support state serialization, and preserve labels and metadata unchanged during transformation.

`src/data/stream.py` should define `DatasetStream`. It should expose `next_point`, `next_window`, `reset`, `state_dict`, and `load_state_dict` so the offline data layer already respects the future online-consumption contract.

`src/data/window.py` should define `Windowizer` and `slice_sequence_into_windows`. The default configuration should be `window_size=100` and `stride=10`. The implementation must never produce windows that cross machine boundaries.

`src/data/collate.py` should define `collate_windows`, which stacks windows into `[B, L, D]` and preserves optional fields as `None` when absent.

`src/data/loaders.py` should define `WindowDataset` and `build_smd_dataloaders`. This function should coordinate parsing, splitting, scaling, windowing, dataset creation, and PyTorch dataloader construction.

`src/models/reconstruction_mlp_ae.py` should define `ReconstructionMLPAutoencoder`. This file should contain model configuration parsing, module construction, the forward pass, score computation, reconstruction loss computation, and `training_step`, `validation_step`, and `test_step`. The implementation must expose `outputs["hidden"]` as `[B, L, H]` even if the internal encoder path flattens windows before projecting them back to time-major hidden states.

`src/engine/trainer.py` should define `Trainer`. It should run epoch-based offline training, call `model.training_step`, invoke validation through `model.validation_step`, and coordinate checkpoint saving.

`src/engine/evaluator.py` should define `Evaluator`. It should convert per-window outputs back into point-level scores on each original machine timeline by averaging overlapping contributions.

`src/engine/checkpoint.py` should define `CheckpointManager`. It should save and load model state, optimizer state, scaler state, config snapshots, epoch, and metric history.

`src/engine/logger.py` should define a thin logging helper that writes metrics and artifact paths in a serialization-friendly form suitable for later Weights and Biases integration.

`scripts/train.py` should build the experiment graph entirely from config and registry components, then train and save the best checkpoint.

`scripts/evaluate.py` should load a saved checkpoint, rebuild the test pipeline through the same registry path, run evaluation, and write metrics and evaluation records to disk.

### Design pattern application

Composition over inheritance should be applied by keeping parser, scaler, stream, windowizer, dataset, collate function, model, trainer, evaluator, and checkpoint manager as separate components with explicit data flow.

The registry pattern is applied minimally through `src/core/registry.py` so scripts instantiate named datasets and models without branching across many conditionals.

The one-model-one-file rule is enforced here already: the reconstruction baseline does not get a separate task file.

### Test plan and validation steps

`tests/test_smd_dataset_shapes.py` should verify that all machine files are parsed, that `entity_id` is preserved, and that train and test labels align with sequence lengths.

`tests/test_windowizer.py` should verify correct window count, tensor shape `[L, D]`, metadata `start_index` and `end_index`, and machine-boundary preservation.

`tests/test_model_shapes.py` should verify that the reconstruction model returns `hidden`, `recon`, `point_scores`, and `window_scores` with the documented shapes.

`tests/test_one_train_step.py` should run one forward pass and one backward pass on one reduced SMD batch or a synthetic stand-in batch with the same shape contract.

`tests/test_checkpoint_roundtrip.py` should verify that save and load restore model weights, optimizer state, scaler state, and config content.

`tests/test_config_loading.py` should verify YAML load success for valid experiment files and expected failure for missing required keys.

`tests/test_registry.py` should verify that the registered dataset and model builders resolve to the same components used by the scripts.

### Acceptance criteria

- The parser reads all 28 SMD machine files from the raw directory structure.
- The validation split is reproducible and derived per machine from training data only.
- The scaler is fit on training-only timesteps and is restored exactly through checkpoint load.
- The windowizer produces windows of shape `[100, 38]` with `stride=10`.
- `ReconstructionMLPAutoencoder` trains for at least one epoch without shape or contract failures.
- A saved checkpoint can be loaded and yields identical model and scaler parameters.
- `scripts/train.py` and `scripts/evaluate.py` both use the registry path rather than bypassing it with a direct dataset-builder call.

## Phase 2 - Reserve the thesis architecture inside one self-contained multitask model file

### Phase summary

This phase prepares the codebase for the later thesis architecture without reintroducing the split-file debt from problem 1. The goal is to preserve the minimal vertical slice while fixing the model boundary early: the future multitask model gets one file now, even if some internals are still placeholders.

### File-level edits

The following files should be added or extended in this phase:

```text
configs/model/thesis_multitask.yaml
configs/experiment/smd_multitask.yaml
src/models/thesis_multitask.py
tests/test_model_contracts.py
tests/test_registry.py
```

### Explicit edit content

`src/models/thesis_multitask.py` should define `ThesisMultiTaskModel` as one readable file. It may contain internal helper classes or clearly delimited sections for the encoder block, continuous prototype block, discrete prototype block, fusion block, reconstruction head, classification head, score computation, and stage methods. Even if some of those sections are still pass-through stubs in this phase, their public home should already be this one file.

`configs/model/thesis_multitask.yaml` should define the fields that later prototype work will require, including prototype counts, fusion parameters, classification head dimensions, and anomaly-injection settings.

`tests/test_model_contracts.py` should verify that both `ReconstructionMLPAutoencoder` and `ThesisMultiTaskModel` preserve `[B, L, H]` semantics and return the required output keys.

### Design pattern application

The adapter pattern still applies conceptually, but it should be implemented inside the model file when the adapter is model-specific.

Composition is still allowed inside a model file through internal helper classes. What is no longer allowed is scattering the same model across `modules/`, `heads/`, `losses/`, and `tasks/` directories.

### Acceptance criteria

- A stable `ThesisMultiTaskModel` file exists and matches the thesis-facing hidden contract.
- The model registry can construct both the reconstruction baseline and the multitask model through explicit names.
- No separate `src/tasks/` or model-specific `src/losses/` files are introduced for the thesis models.

## Phase 3 - Add CARLA-aligned synthetic anomaly augmentation and user-visible inspection

### Phase summary

This phase extends the codebase from a reconstruction-only baseline to a multitask anomaly-detection setting that includes synthetic anomaly injection for classification. In the revised plan, this phase is also where problem 3 and problem 5 are closed. CARLA alignment and anomaly-visualization support are explicitly Phase-3 work, not deferred to Phase 4 or Phase 5.

### File-level edits

The following files should be added or extended in this phase:

```text
configs/model/thesis_multitask.yaml
configs/experiment/smd_multitask.yaml
src/models/thesis_multitask.py
scripts/visualize_synthetic_anomalies.py
tests/test_synthetic_anomaly_injection.py
tests/test_multitask_shapes.py
tests/test_one_multitask_train_step.py
tests/test_synthetic_anomaly_visualization.py
```

### Explicit edit content

`src/models/thesis_multitask.py` should implement the actual multitask path: reconstruction loss, classification loss, prototype regularization if enabled, and CARLA-aligned synthetic anomaly injection. Because of the one-model-one-file rule, the anomaly injector should live in this file as an internal helper class or well-delimited section unless it is genuinely dataset-generic and independent of the thesis model.

The synthetic anomaly mechanism should follow the referenced CARLA augmentation at the mechanism level rather than as simple local point perturbations. The implementation should support subsequence-style anomaly families such as seasonal, trend, global, contextual, and shapelet-like anomalies, or a clearly documented equivalent mapping if naming differs.

The augmentation path should accept clean windows and return augmented windows, anomaly labels, anomaly masks, and augmentation metadata without mutating the original parser output in place.

`scripts/visualize_synthetic_anomalies.py` should generate user-visible inspection artifacts for injected anomalies. A minimal acceptable output is a saved static plot or an exported image per selected example that shows the clean window, the injected window, and the anomaly interval or mask clearly enough for manual review.

`tests/test_synthetic_anomaly_injection.py` should verify shape preservation, metadata retention, nontrivial anomaly-label creation, and the presence of anomaly masks or equivalent localization metadata.

`tests/test_multitask_shapes.py` should verify that multitask outputs include `hidden`, `recon`, `logits`, `point_scores`, and `aux` with valid shapes.

`tests/test_one_multitask_train_step.py` should verify one forward and backward pass under the multitask model.

`tests/test_synthetic_anomaly_visualization.py` should verify that the visualization script or helper writes an inspection artifact successfully on a reduced synthetic example.

### Risk mitigation

Prototype redundancy should be mitigated through branch-specific diagnostics and later ablation hooks.

Fusion collapse should be mitigated by logging branch-usage statistics and by ensuring task-specific fused outputs remain separately inspectable inside the multitask model outputs.

Evaluation inflation should be mitigated by reporting classification and anomaly-detection metrics separately rather than conflating them.

### Acceptance criteria

- Synthetic anomaly injection follows the intended CARLA-style subsequence mechanism rather than only pointwise local perturbations.
- The multitask model preserves the fixed output contract while adding classification outputs.
- One multitask training step completes successfully with combined reconstruction and classification losses.
- A user can export or inspect injected anomaly examples without writing ad hoc notebook code.

## Pre-Phase-4 Gate - Close problems 1, 3, 5, and 6 before online adaptation

Phase 4 must not begin until all of the following statements are true:

- Problem 1 is closed: `src/models/reconstruction_mlp_ae.py` and `src/models/thesis_multitask.py` each keep their forward logic, scoring logic, and stage-specific loss logic in the same file.
- Problem 3 is closed: the multitask model uses a CARLA-aligned subsequence anomaly mechanism instead of the earlier simplified local perturbation logic.
- Problem 5 is closed: there is a maintained script or helper that exports user-visible synthetic anomaly inspection artifacts.
- Problem 6 is closed: `scripts/train.py` and `scripts/evaluate.py` instantiate the dataset path through `build_dataset(...)` or the equivalent registry builder, not through direct calls to a concrete loader helper after registration.
- Phase 1, Phase 2, and Phase 3 acceptance criteria all pass together.

This gate is intentionally strict. The goal is to prevent the online-adaptation stage from being built on top of known offline design debt.

## Phase 4 - Introduce the online adaptation stage with a residual projector

### Phase summary

This phase adds the online adaptation path only after the offline baseline and multitask architecture are already stable and the pre-Phase-4 gate has passed. The objective is to align an online encoder pathway with a frozen reference representation through a lightweight projector while protecting the system from contamination, drift, and unstable update behavior.

### File-level edits

The following files should be added or extended in this phase:

```text
configs/model/online_adaptation.yaml
configs/experiment/smd_online_adaptation.yaml
src/models/online_adaptation.py
src/engine/online_loop.py
tests/test_projector_shapes.py
tests/test_online_state_roundtrip.py
tests/test_online_adaptation_step.py
```

### Explicit edit content

`src/models/online_adaptation.py` should define the online-adaptation model as one readable file. It should contain the frozen reference encoder path, the online encoder path, the residual projector, alignment losses, contamination safeguards, and online-stage step methods. Projector logic should not be split into a separate module file if it is specific to this model.

`src/engine/online_loop.py` should define a stream-oriented execution loop that consumes `DatasetStream`, applies `model.online_step` or `model.training_step` in online mode, and records state snapshots for recovery and analysis.

`configs/model/online_adaptation.yaml` should define update cadence, learning rates, projector constraints, frozen-module settings, and contamination safeguards.

### Acceptance criteria

- The projector and online model can execute one constrained online update step.
- The online state can be checkpointed and restored reproducibly.
- The online path preserves the same standardized contracts as the offline path.
- No Phase-4 file reintroduces split task logic for the online model.

## Phase 5 - Consolidate evaluation, ablations, and reproducible reporting

### Phase summary

This phase consolidates the codebase into a reproducible thesis experimentation platform. By the time this phase begins, the earlier debt from problems 1, 3, 5, and 6 should already be closed. Phase 5 is therefore reserved for evaluation breadth, ablations, reporting, and data-versioning support rather than for fixing the earlier offline design issues.

### File-level edits

The following files should be added or extended in this phase:

```text
src/metrics/eventwise.py
src/metrics/uncertainty.py
scripts/run_ablation.py
scripts/export_results.py
tests/test_evaluation_record_schema.py
tests/test_metric_consistency.py
dvc.yaml
```

### Explicit edit content

`src/metrics/eventwise.py` should define any secondary eventwise metrics needed for thesis reporting, but the codebase should continue to treat pointwise metrics as the default and most conservative baseline.

`src/metrics/uncertainty.py` should define any uncertainty statistics associated with prototype or adaptation behavior.

`scripts/run_ablation.py` should run controlled experiments across baseline, prototype-enabled, augmentation-enabled, and online-adaptation-enabled settings using consistent config references.

`scripts/export_results.py` should convert evaluation outputs into clean, serialization-friendly artifacts for reporting and later Weights and Biases logging.

`dvc.yaml` should track derived augmented datasets or reproducible data preparation steps where synthetic anomaly injection creates versioned artifacts that are intentionally materialized on disk.

### Acceptance criteria

- Evaluation records are saved with a stable schema and enough metadata for reproducibility.
- Ablation scripts can compare baseline and extended variants without changing core contracts.
- Derived data and augmentation artifacts are tracked through reproducible configuration or DVC definitions when those artifacts are materialized.

## Final Implementation Order

The recommended build order is now:

1. Implement Phase 1 completely and verify its acceptance criteria.
2. Add the self-contained multitask model boundary from Phase 2.
3. Complete Phase 3, including CARLA alignment and anomaly visualization.
4. Pass the explicit pre-Phase-4 gate for problems 1, 3, 5, and 6.
5. Add online adaptation and projector safeguards in Phase 4.
6. Consolidate ablations, reporting, and data versioning in Phase 5.

This ordering is intentionally stricter than the previous version. It keeps the online-adaptation stage downstream of the offline codebase corrections that the research note identified as necessary.
