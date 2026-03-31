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

This detailed plan specifies how to implement the first executable thesis milestone for the SMD dataset while preserving the repository constraints in `codebase_preferences.md` and the architectural direction in `documents/design/`. The implementation must prioritize a narrow but complete offline vertical slice first, freeze the thesis-facing contracts early, and defer higher-risk thesis modules until the baseline path is correct, readable, checkpointable, and testable.

## Global Architectural Contracts

The codebase should be organized around five stable layers: configuration, data, model, task, and engine. Each layer should have one clear responsibility and should communicate through explicit contracts rather than implicit assumptions.

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
    },
}
```

### Batch and model-output contracts

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

The model-output contract should be:

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

### Task and engine contracts

Tasks should implement a strategy-style interface that receives a standardized `model` and `batch`, computes losses and metrics, and returns a serialization-friendly dictionary. The engine should never inspect model internals directly. It should operate only on the standardized task outputs and registered components.

### Design pattern rules

- Composition should be preferred over inheritance across the codebase.
- Encoder-specific integration should use an adapter pattern so future backbones can satisfy the fixed `hidden: [B, L, H]` contract.
- Task-specific behavior should use a strategy pattern so reconstruction, classification, and online adaptation can share a stable trainer interface.
- Datasets, models, and tasks should be created through a minimal registry or factory rather than through hard-coded branching in scripts.

## Phase 1 - Build the minimum runnable SMD vertical slice

### Phase summary

This phase establishes the first end-to-end thesis slice for SMD: parser, split logic, scaler, stream, windowizer, dataloaders, one reconstruction baseline, one reconstruction task, one trainer, one evaluator, and checkpoint round-tripping. The purpose of this phase is to prove that the repository can execute the core path `SMD loader -> encoder-facing model -> train/eval loop` while preserving readability-first implementation and stable contracts.

### File-level edits

The following files should be created in this phase:

```text
configs/data/smd.yaml
configs/model/reconstruction_mlp_ae.yaml
configs/task/reconstruction.yaml
configs/experiment/smd_vertical_slice.yaml
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
src/tasks/base_task.py
src/tasks/reconstruction_task.py
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
```

### Explicit edit content

`src/core/config.py` should implement `load_yaml_config`, `load_experiment_config`, and `validate_experiment_config`. Validation should reject missing sections, mismatched file references, and invalid scalar types for windowing and optimization settings.

`src/core/contracts.py` should define `validate_raw_sequence`, `validate_window`, `validate_batch`, `validate_model_outputs`, and `validate_evaluation_record`. Each function should verify required keys, tensor rank, tensor axis order, and metadata fields.

`src/core/registry.py` should define small registries for datasets, models, and tasks. The implementation should remain dictionary-based and readable, with explicit `register_*` and `build_*` functions.

`src/data/datasets/smd.py` should define `SMDDatasetParser`. It should read `data/ServerMachineDataset/train`, `test`, and `test_label`, preserve one machine per entity, create a per-machine validation tail split, and emit native `raw_sequence` dictionaries without embedding scaling or windowing logic.

`src/data/scalers.py` should define `SequenceStandardScaler`. The scaler should fit only on the post-split training sequences, store featurewise statistics, support state serialization, and preserve labels and metadata unchanged during transformation.

`src/data/stream.py` should define `DatasetStream`. It should expose `next_point`, `next_window`, `reset`, `state_dict`, and `load_state_dict` so the offline data layer already respects the future online-consumption contract.

`src/data/window.py` should define `Windowizer` and `slice_sequence_into_windows`. The default configuration should be `window_size=100` and `stride=10`. The implementation must never produce windows that cross machine boundaries.

`src/data/collate.py` should define `collate_windows`, which stacks windows into `[B, L, D]` and preserves optional fields as `None` when absent.

`src/data/loaders.py` should define `WindowDataset` and `build_smd_dataloaders`. This function should coordinate parsing, splitting, scaling, windowing, dataset creation, and PyTorch dataloader construction.

`src/models/base_model.py` should define a small abstract `BaseModel` interface with `forward(batch: dict) -> dict`.

`src/models/reconstruction_mlp_ae.py` should define `ReconstructionMLPAutoencoder`. This file should contain model configuration parsing, module construction, the forward pass, and score computation. The implementation must expose `outputs["hidden"]` as `[B, L, H]` even if the internal encoder path flattens windows before projecting them back to time-major hidden states.

`src/tasks/base_task.py` should define a minimal task interface with `training_step`, `validation_step`, and `test_step`.

`src/tasks/reconstruction_task.py` should define `ReconstructionTask`. It should compute reconstruction loss from `batch["x"]` and `outputs["recon"]`, derive scalar logs, and return standardized dictionaries that the engine can consume without dataset-specific logic.

`src/metrics/pointwise.py` should define the pointwise metric functions used by evaluation, including ROC-AUC, PR-AUC, precision, recall, and F1.

`src/engine/trainer.py` should define `Trainer`. It should run epoch-based offline training, call the task methods, invoke validation, and coordinate checkpoint saving.

`src/engine/evaluator.py` should define `Evaluator`. It should convert per-window outputs back into point-level scores on each original machine timeline by averaging overlapping contributions.

`src/engine/checkpoint.py` should define `CheckpointManager`. It should save and load model state, optimizer state, scaler state, config snapshots, epoch, and metric history.

`src/engine/logger.py` should define a thin logging helper that writes metrics and artifact paths in a serialization-friendly form suitable for later Weights and Biases integration.

`scripts/train.py` should build the experiment graph entirely from config and registry components, then train and save the best checkpoint.

`scripts/evaluate.py` should load a saved checkpoint, rebuild the test pipeline, run evaluation, and write metrics and evaluation records to disk.

### Interface and contract definitions

The data layer must preserve raw sequences as `[T, D]`, windows as `[L, D]`, and batches as `[B, L, D]`. The model layer must never assume a specific dataset beyond this contract. The first baseline is allowed to be SMD-only operationally, but its external interfaces must already be future-safe.

The evaluator should consume only `point_scores`, `window_scores`, `meta`, and labels rather than inspecting reconstruction internals. This preserves the ability to replace the reconstruction baseline with adapter-wrapped encoders later.

### Design pattern application

Composition over inheritance should be applied by keeping parser, scaler, stream, windowizer, dataset, collate function, model, task, trainer, evaluator, and checkpoint manager as separate components with explicit data flow.

The adapter pattern is reserved in this phase by freezing the encoder-facing output contract now, even though the first baseline is a simple reconstruction model rather than a pretrained encoder wrapper.

The strategy pattern is applied by making `ReconstructionTask` responsible for loss computation while the trainer remains task-agnostic.

The registry pattern is applied minimally through `src/core/registry.py` so scripts instantiate named dataset, model, and task components without branching across many conditionals.

### Risk mitigation

Contract drift should be mitigated by validating `raw_sequence`, `window`, `batch`, and `outputs` at component boundaries during development and test execution.

Machine-boundary leakage should be mitigated by parsing the raw SMD files one machine at a time and building windows per entity only.

Validation leakage should be mitigated by creating validation splits before fitting the scaler.

Evaluation metric inflation should be mitigated by reporting honest pointwise metrics and selecting thresholds without test-label leakage.

### Test plan and validation steps

`tests/test_smd_dataset_shapes.py` should verify that all machine files are parsed, that `entity_id` is preserved, and that train/test labels align with sequence lengths.

`tests/test_windowizer.py` should verify correct window count, tensor shape `[L, D]`, metadata `start_index` and `end_index`, and machine-boundary preservation.

`tests/test_model_shapes.py` should verify that the reconstruction model returns `hidden`, `recon`, `point_scores`, and `window_scores` with the documented shapes.

`tests/test_one_train_step.py` should run one forward pass and one backward pass on one reduced SMD batch or a synthetic stand-in batch with the same shape contract.

`tests/test_checkpoint_roundtrip.py` should verify that save and load restore model weights, optimizer state, scaler state, and config content.

`tests/test_config_loading.py` should verify YAML load success for valid experiment files and expected failure for missing required keys.

### Acceptance criteria

- The parser reads all 28 SMD machine files from the raw directory structure.
- The validation split is reproducible and derived per machine from training data only.
- The scaler is fit on training-only timesteps and is restored exactly through checkpoint load.
- The windowizer produces windows of shape `[100, 38]` with `stride=10`.
- The reconstruction baseline trains for at least one epoch without shape or contract failures.
- A saved checkpoint can be loaded and yields identical model and scaler parameters.
- The evaluator reassembles overlapping point scores back to original test timelines and reports pointwise metrics.

## Phase 2 - Reserve extension points for continuous and discrete prototype branches

### Phase summary

This phase prepares the codebase for the later thesis architecture without implementing full prototype learning yet. The goal is to preserve the minimal vertical slice while ensuring that future continuous and discrete prototype branches can be introduced through localized edits rather than broad structural changes.

### File-level edits

The following files should be added or extended in this phase:

```text
src/models/base_encoder.py
src/models/modules/continuous_prototypes.py
src/models/modules/discrete_prototypes.py
src/models/modules/fusion.py
configs/model/prototype_placeholders.yaml
tests/test_registry.py
tests/test_encoder_contracts.py
```

### Explicit edit content

`src/models/base_encoder.py` should define a small abstract encoder interface that returns `hidden`, optional `pooled`, and `aux`.

`src/models/modules/continuous_prototypes.py` should define a placeholder module interface for attention-like lookup over continuous prototypes. The implementation can be a simple identity stub first, but the file path and public API should be fixed.

`src/models/modules/discrete_prototypes.py` should define a placeholder module interface for codebook-based prototype lookup. It should also be allowed to return pass-through outputs initially.

`src/models/modules/fusion.py` should define a placeholder task-fusion module that accepts continuous and discrete branch outputs and returns named task-specific representations.

`configs/model/prototype_placeholders.yaml` should define the fields that later prototype work will require, even if default values disable the modules in this phase.

### Interface and contract definitions

The encoder interface should remain:

```python
{
    "hidden": Tensor[B, L, H],
    "pooled": Optional[Tensor[B, H]],
    "aux": dict,
}
```

Prototype modules should accept `hidden: [B, L, H]` and return branch-specific representations with the same time-major orientation so that downstream fusion remains shape-stable.

The fusion module should produce named outputs such as `hidden_reconstruction` and `hidden_classification` rather than anonymous tuples. This will reduce ambiguity when the multitask branch is added.

### Design pattern application

The adapter pattern becomes explicit in this phase through `BaseEncoder`, which will later wrap external encoders while preserving the thesis contract.

Composition over inheritance is preserved by modeling prototype branches and fusion as pluggable modules rather than as subclasses of one monolithic architecture.

The registry should be extended to cover encoders or architecture variants only if this reduces conditional logic. It should remain minimal and readable.

### Risk mitigation

Prototype redundancy risk should be mitigated structurally by keeping the continuous and discrete branches separate in code, metrics, and later ablation hooks.

Fusion-collapse risk should be mitigated by defining named intermediate outputs now so later diagnostics can inspect branch usage directly rather than inferring it indirectly.

### Test plan and validation steps

`tests/test_registry.py` should verify that dataset, model, task, and encoder registrations resolve to the expected callables.

`tests/test_encoder_contracts.py` should verify that encoder and placeholder prototype modules preserve `[B, L, H]` semantics and return named dictionaries.

### Acceptance criteria

- A stable `BaseEncoder` interface exists and matches the thesis-facing hidden contract.
- Placeholder continuous, discrete, and fusion modules can be instantiated without changing the Phase 1 execution path.
- The registry can construct future-facing encoder or model variants through explicit names.

## Phase 3 - Add task-specific fusion and synthetic anomaly augmentation for classification

### Phase summary

This phase extends the codebase from a reconstruction-only baseline to a multitask anomaly-detection setting that includes synthetic anomaly injection for classification. The objective is to support thesis-aligned supervised anomaly-type learning while preserving the validated SMD data contracts and not contaminating the offline baseline path.

### File-level edits

The following files should be added or extended in this phase:

```text
src/data/augment.py
src/tasks/multitask_tsad_task.py
src/models/thesis_multitask.py
src/losses/classification.py
src/losses/prototype.py
configs/task/multitask_tsad.yaml
configs/model/thesis_multitask.yaml
tests/test_synthetic_anomaly_injection.py
tests/test_multitask_shapes.py
tests/test_one_multitask_train_step.py
```

### Explicit edit content

`src/data/augment.py` should implement CARLA-inspired synthetic anomaly injection as a separate utility or adapter component. It should accept clean windows and return augmented windows, anomaly labels, and augmentation metadata without mutating the original parser output in place.

`src/tasks/multitask_tsad_task.py` should define `MultitaskTSADTask`. It should compute reconstruction loss, classification loss, and any later prototype-related regularizers through clearly named loss terms.

`src/models/thesis_multitask.py` should define the multitask architecture that composes an encoder, prototype modules, a fusion module, a reconstruction head, and a classification head.

`src/losses/classification.py` and `src/losses/prototype.py` should keep classification and prototype-specific loss definitions outside the model file to prevent the architecture from becoming unreadable.

### Interface and contract definitions

The augmentation utility should return windows that still satisfy the standard `window` or `batch` contract, with additional supervision carried through labels or `aux` fields rather than through schema drift.

The multitask model should still return the standardized output dictionary, with `logits` used for classification and `aux` used for prototype diagnostics.

The task should remain the owner of loss weighting and optimization-facing behavior.

### Design pattern application

The strategy pattern is critical here because `MultitaskTSADTask` must coexist with `ReconstructionTask` under the same trainer.

The adapter pattern should be used for synthetic anomaly injection by keeping augmentation as a separate pre-model transformation rather than embedding anomaly generation inside the model.

Composition should continue to govern the architecture, with encoder, continuous prototypes, discrete prototypes, fusion, and heads combined explicitly.

### Risk mitigation

Prototype redundancy should be mitigated through branch-specific diagnostics and later ablation hooks.

Fusion collapse should be mitigated by logging branch-usage statistics and by ensuring task-specific fused outputs remain separately inspectable.

Evaluation inflation should be mitigated by reporting classification and anomaly-detection metrics separately rather than conflating them.

### Test plan and validation steps

`tests/test_synthetic_anomaly_injection.py` should verify shape preservation, metadata retention, and nontrivial anomaly-label creation.

`tests/test_multitask_shapes.py` should verify that multitask outputs include `hidden`, `recon`, `logits`, `point_scores`, and `aux` with valid shapes.

`tests/test_one_multitask_train_step.py` should verify one forward and backward pass under the multitask task.

### Acceptance criteria

- Synthetic anomaly injection operates as an explicit, testable preprocessing or task component.
- The multitask model preserves the fixed output contract while adding classification outputs.
- One multitask training step completes successfully with combined reconstruction and classification losses.

## Phase 4 - Introduce the online adaptation stage with a residual projector

### Phase summary

This phase adds the online adaptation path after the offline baseline and multitask architecture are already stable. The objective is to align an online encoder pathway with a frozen reference representation through a lightweight projector while protecting the system from contamination, drift, and unstable update behavior.

### File-level edits

The following files should be added or extended in this phase:

```text
src/tasks/online_adaptation_task.py
src/models/modules/projector.py
src/engine/online_loop.py
configs/task/online_adaptation.yaml
tests/test_projector_shapes.py
tests/test_online_state_roundtrip.py
tests/test_online_adaptation_step.py
```

### Explicit edit content

`src/models/modules/projector.py` should define a lightweight residual projector `g(.)` that maps online representations into the frozen reference space while preserving dimensional compatibility.

`src/tasks/online_adaptation_task.py` should define `OnlineAdaptationTask`. It should compute alignment losses, prototype-alignment losses if enabled, and safeguards that limit parameter updates to intended modules only.

`src/engine/online_loop.py` should define a stream-oriented execution loop that consumes `DatasetStream`, applies the online task, and records state snapshots for recovery and analysis.

`configs/task/online_adaptation.yaml` should define update cadence, learning rates, projector constraints, frozen-module settings, and contamination safeguards.

### Interface and contract definitions

The online task should consume the same standardized batch structure as offline tasks, plus any view-specific augmentation packaging required for the reference and online branches.

The projector should accept and return `[B, L, H]` tensors so the online path remains compatible with the same prototype and scoring interfaces.

The online engine should persist its state through checkpointable task, model, optimizer, and stream components.

### Design pattern application

The strategy pattern remains necessary so online adaptation can be introduced without changing the offline trainer.

The adapter pattern continues to protect the reference encoder and any online encoder variant behind a fixed representation interface.

Composition remains the governing principle because projector, augmentation views, alignment losses, and streaming logic should stay in distinct modules.

### Risk mitigation

Adaptation contamination should be mitigated by restricting updates to the projector or explicitly allowed online modules, by monitoring score drift, and by disabling adaptation on windows that exceed contamination heuristics.

Projector drift should be mitigated by residual initialization, bounded learning rates, norm monitoring, and checkpoint rollback rules.

Evaluation inflation should be mitigated by reporting pre-adaptation and post-adaptation metrics separately and by preserving the same core anomaly-detection evaluation pipeline.

### Test plan and validation steps

`tests/test_projector_shapes.py` should verify that the projector preserves `[B, L, H]` semantics.

`tests/test_online_state_roundtrip.py` should verify that projector, optimizer, and stream states survive save and load operations.

`tests/test_online_adaptation_step.py` should verify one online update step with frozen reference parameters and trainable projector parameters.

### Acceptance criteria

- The projector and online task can execute one constrained online update step.
- The online state can be checkpointed and restored reproducibly.
- The online path preserves the same standardized contracts as the offline path.

## Phase 5 - Consolidate evaluation, ablations, and reproducible reporting

### Phase summary

This phase consolidates the codebase into a reproducible thesis experimentation platform. The objective is to ensure that evaluation, ablations, artifact logging, and reporting remain compatible with the minimal contracts defined at the start of implementation.

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

`dvc.yaml` should track derived augmented datasets or reproducible data preparation steps where synthetic anomaly injection creates versioned artifacts.

### Interface and contract definitions

The evaluator should emit a stable `evaluation_record` schema that includes dataset identity, entity identity, split, metric values, threshold source, checkpoint reference, and any configuration hashes needed for reproducibility.

All scripts should consume the same experiment configuration structure instead of inventing one-off CLI conventions.

### Design pattern application

Factory and registry usage should remain minimal but sufficient so ablation scripts can swap configurations rather than hard-coded modules.

Composition should remain visible at the script level, with exporters, evaluators, metrics, and logging utilities kept independent.

### Risk mitigation

Evaluation metric inflation should be mitigated by always reporting conservative pointwise metrics first, documenting threshold selection rules, and separating official metrics from debugging-only summaries.

Prototype redundancy, fusion collapse, and projector drift should all be analyzed through explicit ablations rather than inferred from end metrics alone.

### Test plan and validation steps

`tests/test_evaluation_record_schema.py` should verify that saved evaluation outputs match the documented schema.

`tests/test_metric_consistency.py` should verify deterministic metric behavior on controlled synthetic inputs.

Manual validation should confirm that ablation scripts preserve common configs and do not silently change preprocessing or threshold rules.

### Acceptance criteria

- Evaluation records are saved with a stable schema and enough metadata for reproducibility.
- Ablation scripts can compare baseline and extended variants without changing core contracts.
- Derived data and augmentation artifacts are tracked through reproducible configuration or DVC definitions.

## Final Implementation Order

The recommended build order remains intentionally conservative:

1. Implement Phase 1 completely and verify its acceptance criteria.
2. Add only the structural extension points from Phase 2.
3. Introduce multitask fusion and synthetic anomaly augmentation in Phase 3.
4. Add online adaptation and projector safeguards in Phase 4.
5. Consolidate ablations, reporting, and data versioning in Phase 5.

This ordering best satisfies the thesis objective because it proves the executable SMD foundation first and only then introduces the higher-risk thesis-specific modules that depend on those contracts.
