---
date: 2026-05-20 20:36:36 +07
researcher: Artificial Intelligence Agent
git_commit: 1e44b7b33a3b9813e62242d2bdbed004c7a0ef65
branch: dev
repository: bachelor-thesis-2026
topic: "Current codebase status for implementing Experiment 2 (offline pre-training phase two-view contrastive + CKA-gated fusion)"
tags: [research, time-series, anomaly-detection, multi-class]
status: complete
last_updated: 2026-05-20
last_updated_by: Artificial Intelligence Agent
---

# Research: Current codebase status for implementing Experiment 2 (offline pre-training phase two-view contrastive + CKA-gated fusion)

**Date**: 2026-05-20 20:36:36 +07  
**Researcher**: Artificial Intelligence Agent  
**Git Commit**: 1e44b7b33a3b9813e62242d2bdbed004c7a0ef65  
**Branch**: dev

## Research Question
Use `prompts/1_research_prompt.md` to inspect and document the current repository state in order to prepare implementation work for Experiment 2.

## Summary
The repository currently provides a complete offline pipeline for SMD with fixed-length windows, synthetic anomaly injection, multitask prototype model training, synthetic validation, checkpointing, and evaluation. The active design documents in `documents/design/` already define the offline pre-training phase two-view contrastive + CKA-gated fusion target and Experiment Protocol v2. In code, the baseline multitask path does not yet include the new CKA-gated per-sample fusion or the revised branch-routing and token-partitioned memory update semantics specified for Experiment 2. The runtime and configuration surfaces required for implementation are present and coherent: batch contracts, synthetic mask propagation, scheduler/checkpoint monitor controls, and synthetic validation stage hooks already exist.

## Detailed Findings

### Data Preparation
- Dataset source and parser path:
  - SMD parsing is implemented through `SMDDatasetParser` and built via `SMDDatasetBuilder` in `src/data/loaders.py`.
- Windowing and batching:
  - The active training path constructs windows in `WindowDataset` inside `src/data/loaders.py`, not via the generic `Windowizer` class.
  - `window_size`, `stride`, and `batch_size` are read from data config and injected into `WindowDataset` construction.
- Scaling and preprocessing:
  - `SequenceCleaningPipeline` and `SequenceStandardScaler` are applied before window materialization in the dataset builder.
- Synthetic augmentation:
  - Synthetic anomaly families and label taxonomy are implemented in `src/data/augment.py`.
  - The multitask model invokes synthetic augmentation at stage-specific points (`train` and `val_synth`) and carries `synthetic_anomaly_mask` and `classification_labels` in the prepared batch.

### Modeling and Training
- Model architecture surface:
  - `ThesisMultitaskModel` is the active offline multitask owner in one file (`src/models/thesis_multitask.py`) with encoder, prototype branches, fusion, losses, and stage steps.
  - Current model config for RedLamp multiclass uses `continuous_num_prototypes: 16` and `discrete_codebook_size: 16`.
- Loss and objective:
  - Existing objective is reconstruction + classification with optional terms (`diversity`, `variance`, `covariance`, `usage`, `gate`) governed by config flags and lambdas.
  - Existing stage logging already includes synthetic validation metrics and memory state telemetry.
- Memory lifecycle:
  - Memory initialization and update lifecycle already exists (`maybe_initialize_memories_from_loader`, train-only update gates, read-only eval behavior).
  - Current implementation collects normal tokens for initialization from clean and synthetic-normal positions, with train-time continuous/discrete updates already separated by branch function but not yet constrained by the Experiment 2 token-partition semantics.
- Scheduler/checkpoint control:
  - Trainer supports explicit `scheduler_monitor_metric` and `checkpoint_monitor_metric` with accepted monitor keys including `val_synth_vus_pr` and `val_vus_pr`.
- Experiment configuration state:
  - `configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20-nobootstrap__w20__seed11__default.yaml` sets `bootstrap_encoder_epochs: 0`, scheduler monitor `val_synth_vus_pr`, and checkpoint monitor `val_synth_vus_pr`.

### Evaluation
- Evaluation entrypoint:
  - `scripts/evaluate.py` rebuilds model and dataset from config, loads checkpoint, and evaluates on test loader.
- Pointwise score reconstruction:
  - `src/engine/evaluator.py` merges overlapping window scores back to entity timelines using metadata indices.
- Thresholding and metrics:
  - Point-score threshold is selected by quantile from concatenated test scores (`select_point_score_threshold`).
  - Pointwise metrics, including VUS-related metrics when enabled, are produced by `compute_pointwise_metrics`.
- Reporting outputs:
  - Evaluation writes `evaluation_records.json`, `evaluation_metrics.json`, `evaluation_curves.json`, and logs prefixed metrics via `ExperimentLogger`.

## Code References
- `src/data/loaders.py:112` - SMD dataset builder orchestration and preprocessing.
- `src/data/loaders.py:165` - `WindowDataset` construction with config-driven `window_size` and `stride`.
- `src/data/augment.py:17` - synthetic anomaly family taxonomy and injector surface.
- `src/core/contracts.py:91` - offline batch contract validation.
- `src/models/thesis_multitask.py:320` - multitask model entrypoint and component construction.
- `src/models/thesis_multitask.py:1337` - stage-specific batch preparation and synthetic augmentation flow.
- `src/models/thesis_multitask.py:1398` - `val_synth` augmentation branch.
- `src/models/thesis_multitask.py:1549` - stage-step objective assembly and logging.
- `src/models/thesis_multitask.py:846` - memory initialization trigger path.
- `src/engine/trainer.py:210` - best-checkpoint monitor resolution and supported monitor metrics.
- `src/engine/trainer.py:511` - synthetic validation stage execution in each epoch.
- `src/engine/evaluator.py:197` - test evaluation loop.
- `configs/model/thesis_multitask_redlamp_multiclass.yaml:9` - continuous prototype count.
- `configs/model/thesis_multitask_redlamp_multiclass.yaml:11` - discrete codebook size.
- `configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20-nobootstrap__w20__seed11__default.yaml:18` - scheduler monitor metric.
- `configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20-nobootstrap__w20__seed11__default.yaml:25` - checkpoint monitor metric.

## Pipeline Documentation
- Offline training pipeline:
  - Resolved experiment config -> runtime registration -> SMD dataset bundle build -> model build (model + task merge) -> trainer loop.
- Validation structure:
  - Standard validation stage plus optional synthetic validation stage (`val_synth`) with deterministic synthetic injector reset each epoch.
- Evaluation pipeline:
  - Checkpoint restore -> test loader scoring -> overlap reconstruction to timeline -> threshold selection -> pointwise metrics and curves.

## Historical Context (from documents/)
- `documents/design/idea.md` and `documents/design/design_starter.md` currently designate `documents/design/offline_pretraining_phase_two_view_contrastive_design.md` as the authoritative implementation contract for the offline two-view contrastive work.
- The SSOT design now states active window length as 20 and includes CKA-gated per-sample fusion and Experiment Protocol v2.

## Open Questions
- No blocking repository-structure ambiguity was found for initiating Experiment 2 implementation. The remaining gaps are implementation gaps relative to the documented SSOT, not unresolved file ownership or runtime wiring ambiguity.
