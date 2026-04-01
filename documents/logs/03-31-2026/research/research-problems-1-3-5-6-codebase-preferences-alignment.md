---
date: 2026-03-31 21:00:13 +07+0700
researcher: TheMetaSetter
git_commit: 87c0e9b2a092b3e3b5a5b6f6ea5b54b4b948555d
branch: dev
repository: bachelor-thesis-2026
topic: "Problems 1, 3, 5, and 6 against codebase_preferences.md"
tags: [research, time-series, anomaly-detection, multi-class]
status: complete
last_updated: 2026-03-31
last_updated_by: TheMetaSetter
---

# Research: Problems 1, 3, 5, and 6 against codebase_preferences.md

**Date**: 2026-03-31 21:00:13 +07+0700
**Researcher**: TheMetaSetter
**Git Commit**: 87c0e9b2a092b3e3b5a5b6f6ea5b54b4b948555d
**Branch**: dev

## Research Question

Document the current repository implementation relevant to problems 1, 3, 5, and 6, which concern model-file organization, synthetic anomaly injection alignment with CARLA, anomaly-injection visualization support, and registry usage in training and evaluation entrypoints.

## Summary

The repository currently implements a working vertical slice for SMD using standardized batch and model-output contracts, a reconstruction baseline, and a multitask model with prototype and fusion modules. For the requested problems, the codebase diverges from `codebase_preferences.md` in four specific ways. First, model inference and training logic are separated across `src/models/` and `src/tasks/`, rather than being colocated in a single self-contained model file. Second, synthetic anomaly injection is implemented as simple local perturbations and does not follow the subsequence anomaly mechanism defined in the referenced CARLA augmenter. Third, the repository contains a basic anomaly-injection test but no code path that visualizes or exports injected samples for user inspection. Fourth, the script entrypoints register a dataset builder in the registry but instantiate the data bundle by calling the concrete SMD loader function directly.

## Detailed Findings

### Data Preparation

- The active data pipeline remains centered on standardized batches that expose `x` with shape `[B, L, D]`, matching the contract described in `documents/design/design_starter.md`.
- Problem 3 concerns the augmentation stage attached to the multitask task. In `src/tasks/multitask_tsad_task.py`, the training path conditionally calls `SyntheticAnomalyInjector.augment_batch` before the model forward pass.
- The current injector in `src/data/augment.py` samples a contiguous segment inside one channel, then applies one of three local transformations: `spike`, `dropout`, or `level_shift`. It returns an augmented batch with `classification_labels`, `synthetic_anomaly_mask`, and `augmentation_metadata`.
- The referenced CARLA mechanism in `bsc-thesis-ref-codebases/CARLA-main/data/augment.py` follows a different pattern. It builds seasonal, trend, global, contextual, and shapelet subsequence anomalies and returns one sampled anomalous window from that family. The present repository does not implement that mechanism.
- Problem 5 concerns inspection of injected anomalies. The repository includes `tests/test_synthetic_anomaly_injection.py`, which verifies tensor shapes, labels, and metadata, but the repository search did not find plotting utilities, Matplotlib usage, or export functions in `src`, `tests`, or `scripts`.

### Modeling and Training

Terminology normalized on 2026-04-02. Current design target: gate entropy regularization. Current implementation status: the code still uses a barrier-style gate term and should be updated separately.

- Problem 1 concerns model-file organization. `codebase_preferences.md` requires all logic related to one model, including inference and training logic, to live in one single file for that model.
- The reconstruction baseline is currently split between `src/models/reconstruction_mlp_ae.py`, which defines the architecture and `forward`, and `src/tasks/reconstruction_task.py`, which computes reconstruction loss and training, validation, and test metrics.
- The multitask pipeline is also split. `src/models/thesis_multitask.py` defines the encoder, prototype branches, fusion module invocation, and heads. `src/tasks/multitask_tsad_task.py` performs batch preparation, synthetic anomaly injection, reconstruction loss, cross-entropy classification loss, prototype regularization, and accuracy computation.
- The multitask model further depends on separate module files under `src/models/modules/` for continuous prototypes, discrete prototypes, and fusion. This means the implementation is modular and readable as a framework, but it is not self-contained in the single-file sense stated in the preferences document.

### Evaluation

- Problem 6 concerns the entrypoint scripts rather than metric correctness. Both `scripts/train.py` and `scripts/evaluate.py` register the dataset builder under the registry with `register_dataset("smd", build_smd_dataloaders)`.
- After registration, both scripts instantiate the data bundle by calling `build_smd_dataloaders(...)` directly rather than calling `build_dataset("smd", ...)` from `src/core/registry.py`.
- The rest of the train and evaluation flow continues through the common engine components. `scripts/train.py` uses `build_model` and `build_task`, while `scripts/evaluate.py` uses `build_model`, loads a checkpoint, rebuilds a task, and runs the evaluator on the test loader.
- This means the direct dataset code path and the registry dataset code path currently coexist in the entry scripts.

## Code References

- `codebase_preferences.md:36` - one-model one-file requirement begins
- `codebase_preferences.md:40` - inference and training logic should be in one file
- `codebase_preferences.md:7` - CARLA augmentation alignment requirement
- `codebase_preferences.md:81` - user-facing anomaly visualization requirement
- `codebase_preferences.md:62` - least amount of codepaths principle
- `src/models/reconstruction_mlp_ae.py:12` - reconstruction model definition and forward path
- `src/tasks/reconstruction_task.py:11` - reconstruction loss and stage logic
- `src/models/thesis_multitask.py:43` - multitask model definition
- `src/tasks/multitask_tsad_task.py:14` - multitask task logic including augmentation and losses
- `src/models/modules/continuous_prototypes.py:9` - continuous prototype branch
- `src/models/modules/discrete_prototypes.py:9` - discrete prototype branch
- `src/models/modules/fusion.py:9` - fusion module
- `src/data/augment.py:37` - current synthetic anomaly injector implementation
- `bsc-thesis-ref-codebases/CARLA-main/data/augment.py:23` - referenced CARLA subsequence anomaly mechanism
- `tests/test_synthetic_anomaly_injection.py:8` - current anomaly-injection test coverage
- `scripts/train.py:25` - dataset builder registration in train script
- `scripts/train.py:45` - direct `build_smd_dataloaders` call in train script
- `scripts/evaluate.py:23` - dataset builder registration in evaluation script
- `scripts/evaluate.py:44` - direct `build_smd_dataloaders` call in evaluation script
- `src/core/registry.py:25` - registry-based dataset construction path

## Pipeline Documentation

The present repository exposes a contract-based anomaly-detection pipeline. Datasets and loaders prepare batches as dictionaries centered on `x: Tensor[B, L, D]`. Models consume those dictionaries and return standardized outputs such as `hidden`, `pooled`, `recon`, `logits`, `point_scores`, `window_scores`, and `aux`. For the multitask path, synthetic anomalies are injected during training in the task layer rather than the model layer. The training script constructs the data bundle, model, task, optimizer, checkpoint manager, experiment logger, and trainer. The evaluation script reconstructs the model and task, restores a checkpoint, and writes JSON evaluation outputs. The registry is used for models and tasks in the scripts, but dataset construction is still called directly through the SMD loader helper.

## Historical Context (from documents/)

`documents/design/design_starter.md` describes a framework organized around stable data, model, task, and engine contracts. That design document emphasizes standardized batch dictionaries and model outputs, with composition across layers. `documents/design/idea.md` describes a thesis-facing architecture that uses windows of length one hundred, continuous and discrete prototype branches, task-specific fusion, synthetic anomaly injection for anomaly-type classification, and a later online-adaptation phase. The current codebase matches the contract-oriented framing from `design_starter.md` more closely than it matches the self-contained one-model one-file requirement in `codebase_preferences.md`.

## Open Questions

- The repository contains `src/data/stream.py`, but this research pass did not identify an active online-adaptation implementation connected to the requested problems.
- The repository-level preferences require user-visible anomaly visualization, but the exact expected output form, such as static image export, notebook inspection, or script-based preview, is not specified in the preference file.
- The preference file refers to following CARLA mechanisms, but it does not define whether exact code parity or mechanism-level equivalence is the intended threshold for compliance.

## Follow-up 2026-03-31 21:05:51 +07+0700

### Follow-up Question

If the repository is meant to continue following the staged implementation order in `documents/logs/03-31-2026/plans/detail-smd-loader-windowing.md`, how does Phase 4 relate to problems 1, 3, 5, and 6?

### Follow-up Findings

- The detailed plan distributes these concerns across multiple phases rather than placing them inside Phase 4. The final implementation order explicitly sequences Phase 3 before Phase 4 and Phase 5 after Phase 4.
- Phase 4 itself is defined only as the online-adaptation stage with a residual projector, an online task, an online loop, and online-state tests. In the current repository snapshot, those Phase 4 files are not present. A search across `src`, `tests`, and `configs` did not find `online_adaptation`, `projector`, or `online_loop` files.
- Problem 1 does not belong to Phase 4 in the current plan. The detailed plan itself formalizes a five-layer architecture with separate model, task, and engine responsibilities, and Phase 1 explicitly includes both `src/models/reconstruction_mlp_ae.py` and `src/tasks/reconstruction_task.py`. Phase 3 continues the same pattern with `src/models/thesis_multitask.py` and `src/tasks/multitask_tsad_task.py`. Therefore, the phased plan as written preserves the model-task split that produced problem 1.
- Problem 3 belongs to Phase 3, not Phase 4. The detailed plan states that `src/data/augment.py` should implement CARLA-inspired synthetic anomaly injection in Phase 3. The current repository does contain a Phase 3-style augmentation file and multitask task, but the implemented augmentation logic remains simpler than the CARLA reference mechanism.
- Problem 5 is also tied more closely to Phase 3 and Phase 5 than to Phase 4. Phase 3 requires `tests/test_synthetic_anomaly_injection.py` and specifies shape preservation, metadata retention, and anomaly-label creation. Phase 5 adds reporting and export infrastructure. The current detailed plan does not explicitly assign user-facing visualization of injected samples to Phase 4.
- Problem 6 traces back to Phase 1. The detailed plan states that datasets, models, and tasks should be instantiated through a minimal registry and that `scripts/train.py` should build the experiment graph entirely from config and registry components. The current scripts register the SMD dataset builder, but then instantiate the data bundle through `build_smd_dataloaders(...)` directly, so the plan’s registry-oriented script path is only partially reflected in code.

### Follow-up Interpretation of the Current State

Within the repository as it exists today, advancing to Phase 4 and addressing problems 1, 3, 5, and 6 are not the same unit of work. Problems 3 and 6 are tied to earlier phases, problem 5 sits between the current Phase 3 test surface and the later reporting surface of Phase 5, and problem 1 is structurally in tension with the layered plan itself. As a result, the present phased document does not describe a single Phase 4 step that would absorb all four problems.

### Additional Code References

- `documents/logs/03-31-2026/plans/detail-smd-loader-windowing.md:93` - task and engine responsibilities are separated
- `documents/logs/03-31-2026/plans/detail-smd-loader-windowing.md:101` - strategy-pattern task separation is explicit
- `documents/logs/03-31-2026/plans/detail-smd-loader-windowing.md:102` - registry-oriented creation rule
- `documents/logs/03-31-2026/plans/detail-smd-loader-windowing.md:131` - Phase 1 includes the reconstruction model file
- `documents/logs/03-31-2026/plans/detail-smd-loader-windowing.md:133` - Phase 1 includes the reconstruction task file
- `documents/logs/03-31-2026/plans/detail-smd-loader-windowing.md:175` - reconstruction loss remains in the task layer
- `documents/logs/03-31-2026/plans/detail-smd-loader-windowing.md:187` - train script should build through config and registry components
- `documents/logs/03-31-2026/plans/detail-smd-loader-windowing.md:315` - Phase 3 starts
- `documents/logs/03-31-2026/plans/detail-smd-loader-windowing.md:340` - CARLA-inspired augmentation is assigned to Phase 3
- `documents/logs/03-31-2026/plans/detail-smd-loader-windowing.md:374` - Phase 3 anomaly test expectations
- `documents/logs/03-31-2026/plans/detail-smd-loader-windowing.md:386` - Phase 4 starts
- `documents/logs/03-31-2026/plans/detail-smd-loader-windowing.md:397` - expected Phase 4 files
- `documents/logs/03-31-2026/plans/detail-smd-loader-windowing.md:412` - online loop depends on `DatasetStream`
- `documents/logs/03-31-2026/plans/detail-smd-loader-windowing.md:454` - Phase 5 starts
- `documents/logs/03-31-2026/plans/detail-smd-loader-windowing.md:468` - export/reporting script is assigned to Phase 5
- `documents/logs/03-31-2026/plans/detail-smd-loader-windowing.md:471` - `dvc.yaml` is assigned to Phase 5
- `documents/logs/03-31-2026/plans/detail-smd-loader-windowing.md:522` - final implementation order begins with Phase 1
- `documents/logs/03-31-2026/plans/detail-smd-loader-windowing.md:525` - Phase 4 follows Phase 3
- `documents/logs/03-31-2026/plans/detail-smd-loader-windowing.md:526` - Phase 5 follows Phase 4
- `src/data/stream.py:8` - `DatasetStream` already exists as a pre-Phase-4 support component
