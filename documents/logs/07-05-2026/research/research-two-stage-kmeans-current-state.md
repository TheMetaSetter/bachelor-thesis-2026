---
date: 2026-07-05 00:00:00 +07:00
researcher: Artificial Intelligence Agent
git_commit: c0ef2451ab524914cbb2343a031c8455f1737a5c
branch: dev
repository: bachelor-thesis-2026
topic: "Current state of the codebase before planning the two-stage k-means memory redesign"
tags: [research, time-series, anomaly-detection, multi-stage, kmeans]
status: complete
last_updated: 2026-07-05
last_updated_by: Artificial Intelligence Agent
---

# Research: Current state of the codebase before planning the two-stage k-means memory redesign

**Date**: 2026-07-05 00:00:00 +07:00  
**Researcher**: Artificial Intelligence Agent  
**Git Commit**: `c0ef2451ab524914cbb2343a031c8455f1737a5c`  
**Branch**: `dev`

## Research Question
Use the existing research prompt to document the current state of the codebase before planning implementation work for `documents/design/offline_pretraining_two_stage_kmeans_memory_design.md`.

## Summary
The repository already contains a working two-stage orchestration surface for `thesis_multitask`, including two-stage config validation, a two-stage runner, and stage-aware model behavior. However, the current memory initialization path is still heuristic-based: it collects latent pools from the training split and seeds the prototype banks with a covering-vector selection routine, not with k-means. The code therefore has most of the wiring needed for the approved design, but the memory bootstrap semantics still need to be aligned with the new k-means contract.

## Detailed Findings

### Data Preparation
- The fixed-length window contract is implemented in [src/data/window.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/ĐA%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/window.py). It slices raw sequences into windows of `window_size` with a configurable stride and preserves metadata such as `entity_id`, `split`, `start_index`, and `end_index`.
- Runtime batch validation is centralized in [src/core/contracts.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/ĐA%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%20NHIE%CC%82N/Kho%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/core/contracts.py). The offline batch contract remains `batch["x"]` with shape `[B, L, D]`, plus optional labels and metadata.
- Synthetic anomaly augmentation lives in [src/data/augment.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/ĐA%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/augment.py). The active taxonomy there is the RedLamp family set, and the injector emits `classification_labels` plus `synthetic_anomaly_mask`.

### Modeling and Training
- The main model remains self-contained in [src/models/thesis_multitask.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/ĐA%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%20NHIE%CC%82N/Kho%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/models/thesis_multitask.py) and is decomposed into setup, state, routing, and loss mixins. The file already recognizes the two active phases `stage_a_multitask_pretraining` and `stage_b_fusion_finetuning`.
- Two-stage phase support is present in [src/models/thesis_multitask_setup_mixin.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/ĐA%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%20NHIE%CC%82N/Kho%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/models/thesis_multitask_setup_mixin.py) and [src/models/thesis_multitask_state_mixin.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/ĐA%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%20NHIE%CC%82N/Kho%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/models/thesis_multitask_state_mixin.py). Stage A keeps the encoder trainable; Stage B freezes the encoder and switches the active surface to the fusion path.
- The model already has a prototype path, discrete query modes, optional losses, and stage-aware freeze logic. The active discrete path still contains Gumbel-style machinery for legacy compatibility, even though the new design wants the cosine-top-k path to be the intended runtime direction.
- The training loop in [src/engine/trainer.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/ĐA%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%20NHIE%CC%82N/Kho%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/engine/trainer.py) already calls `maybe_initialize_memories_from_loader(...)` at the start of each epoch and persists `memory_initialized` in checkpoint metadata.
- The two-stage orchestration script [scripts/run_two_stage_offline_pretraining.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/ĐA%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%20NHIE%CC%82N/Kho%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/scripts/run_two_stage_offline_pretraining.py) materializes per-stage YAMLs, writes a manifest, trains Stage A, prepares a Stage B initialization checkpoint, then runs Stage B and evaluation.

### Evaluation
- The engine continues to compute validation metrics, classification diagnostics, and checkpoint selection in [src/engine/trainer.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/ĐA%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%20NHIE%CC%82N/Kho%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%82p/bachelor-thesis-2026/src/engine/trainer.py).
- The two-stage runner writes a manifest and an execution report under the experiment output directory, so the offline pre-training flow is already observable as a separate orchestration artifact.

## Code References
- [documents/design/offline_pretraining_two_stage_kmeans_memory_design.md](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/ĐA%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%20NHIE%CC%82N/Kho%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%82p/bachelor-thesis-2026/documents/design/offline_pretraining_two_stage_kmeans_memory_design.md) - active SSOT for the approved two-stage rerun
- [configs/model/thesis_multitask_two_stage_window20.yaml](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/ĐA%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%20NHIE%CC%82N/Kho%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%82p/bachelor-thesis-2026/configs/model/thesis_multitask_two_stage_window20.yaml) - current two-stage model defaults
- [src/core/config.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/ĐA%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%20NHIE%CC%82N/Kho%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%82p/bachelor-thesis-2026/src/core/config.py) - two-stage config validation
- [scripts/run_two_stage_offline_pretraining.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/ĐA%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%20NHIE%CC%82N/Kho%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%82p/bachelor-thesis-2026/scripts/run_two_stage_offline_pretraining.py) - two-stage orchestration
- [src/models/thesis_multitask_state_mixin.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/ĐA%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%20NHIE%CC%82N/Kho%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%82p/bachelor-thesis-2026/src/models/thesis_multitask_state_mixin.py) - current memory bootstrap and token-pool selection logic

## Pipeline Documentation
The current offline path is already stage-aware:

```text
Stage A
  -> train encoder + task heads
  -> collect latent pools from train split
  -> initialize prototype memories

Stage B
  -> freeze encoder + memory banks
  -> train fusion heads and prediction heads
```

The important mismatch with the approved SSOT is that the memory initialization step still uses a selection heuristic rather than k-means. Continuous memory is seeded from clean latent tokens, and discrete memory is seeded from class-stratified synthetic/anomaly tokens, but the centroids themselves are not currently produced by a k-means routine.

## Historical Context (from documents/)
- The older three-stage material is still supported in code for compatibility, but the new SSOT explicitly treats it as historical context.
- The current two-stage design in `documents/design/offline_pretraining_two_stage_kmeans_memory_design.md` already fixes the target contract: `Stage A` = `80` epochs, `Stage B` = `20` epochs, continuous prototypes = `32`, discrete codebook size = `60`, discrete query mode = `cosine_topk`, and memory initialization from training split only.

## Open Questions
- The repository still needs a k-means implementation for memory bootstrap, so the exact placement of the clustering routine inside the current model or runner remains to be planned.
- The current model still carries legacy discrete-assignment and Gumbel-softmax code paths for backward compatibility; the exact cleanup boundary for the new design should be planned before implementation.
