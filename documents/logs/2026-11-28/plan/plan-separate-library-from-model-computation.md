---
date: 2026-11-28T10:25:00+07:00
planner: Antigravity
topic: "Separate library logic from model computation flow in src/methods"
status: ready
related_research: documents/logs/2026-11-28/research/research-methods-overcomplication.md
---

# Implementation Plan: Separate Library Logic from Model Computation Flow

## Summary

Separate library/runtime orchestration (`iter_batches`, tensor conversion, `save_checkpoint`/`load_checkpoint`, `resume`, `on_checkpoint`, `limits`, score-envelope building, and YAML loading) from the pure mathematical computation flow of each model in `src/methods/` and `src/methods/mustard/`.

## Request

Separate logics of the library out of the computation flow of each model so the computation flow of each model is clear, concise, and easy to edit in the future (referencing the clarity of `DCAGPT` in `lucidrains/deep-cross-attention`).

## Phases, Stages, and Atomic Steps

### Phase 1: Library Runtime Extraction & Config Resolution
- **Stage 1.1: Resolve online MUSTARD `source_method` in `src/config.py`**
  - [ ] Step 1 (`read_file`): Read `resolve_run` in `src/config.py`.
  - [ ] Step 2 (`edit_file`): Attach `source_method` in `resolve_run` for online MUSTARD runs.
- **Stage 1.2: Create `src/method_runtime.py`**
  - [ ] Step 1 (`write_file`): Implement `fit_neural_stages` and `_run_stage` (<= 50 lines each).
  - [ ] Step 2 (`write_file`): Implement `fit_endpoint_model` (<= 50 lines).
  - [ ] Step 3 (`write_file`): Implement `require_no_labels`, `to_device_tensor`, `build_score_record`, and `corrected_cross_entropy`.

### Phase 2: Simplify `src/methods/mustard/` Computation Flow
- **Stage 2.1: Vectorize signature verification and contrastive loss**
  - [ ] Step 1 (`read_file`): Check `tests/test_mustard.py` expectations.
  - [ ] Step 2 (`edit_file`): Vectorize `src/methods/mustard/signature_verification.py` with `torch.topk` integer tensors.
  - [ ] Step 3 (`edit_file`): Vectorize `src/methods/mustard/contrastive.py` with masked matrix multiplication.
  - [ ] Step 4 (`edit_file`): Add explicit `to_dict()` / `from_dict()` on `VerificationBuffer` and use them in `src/methods/mustard/state.py`.
- **Stage 2.2: Clean `Mustard` computation flow**
  - [ ] Step 1 (`edit_file`): Move `NearIdentityMLPProjector` from `src/encoders.py` to `src/methods/mustard/model.py`.
  - [ ] Step 2 (`edit_file`): Simplify `Mustard.__init__` (remove disk YAML loading).
  - [ ] Step 3 (`edit_file`): Delegate `Mustard.fit` to `fit_neural_stages` and keep `forward()`, `_initialize_memory()`, `score()`, `update()` concise.
  - [ ] Step 4 (`edit_file`): Reuse `corrected_cross_entropy` in `src/methods/mustard/losses.py` and remove redundant mode save/restore in `Mustard.score`.

### Phase 3: Simplify Baseline Methods & Shared Augmentation
- **Stage 3.1: Separate library logic from neural baselines**
  - [ ] Step 1 (`edit_file`): Refactor `src/methods/candi.py` to delegate `fit()` to `fit_neural_stages`.
  - [ ] Step 2 (`edit_file`): Vectorize `SANA` in `src/methods/candi_adapter.py` with grouped `nn.Conv1d`.
  - [ ] Step 3 (`edit_file`): Refactor `src/methods/m2n2.py` to define its decoder locally and delegate `fit()` to `fit_neural_stages`.
  - [ ] Step 4 (`edit_file`): Refactor `src/methods/redlamp.py` to use `corrected_cross_entropy`, `fit_neural_stages`, and direct `[B, L, C]` masks.
  - [ ] Step 5 (`edit_file`): Refactor `src/methods/moment.py` to delegate `fit()` to `fit_neural_stages`.
- **Stage 3.2: Separate library logic from traditional baselines**
  - [ ] Step 1 (`edit_file`): Move `_window_matrix` out of `kmeans_ad.py` into `src/method_runtime.py`.
  - [ ] Step 2 (`edit_file`): Delegate `KMeansAD.fit`, `IsolationForestAD.fit`, and `StumpyChannelAB.fit` to `fit_endpoint_model`.
- **Stage 3.3: Simplify augmentation**
  - [ ] Step 1 (`read_file`): Check `tests/test_augmentation.py`.
  - [ ] Step 2 (`edit_file`): Return `[B, L, C]` `cell_mask` directly from `augment()` in `src/methods/augmentation.py`.

### Phase 4: Verification & Compliance Check
- **Stage 4.1: Automated & manual verification**
  - [ ] Step 1 (`run_command`): Run `.venv/bin/python -m pytest -q`.
  - [ ] Step 2 (`run_command`): Check line counts with `wc -l src/methods/*.py src/methods/mustard/*.py src/method_runtime.py`.
  - [ ] Step 3 (`run_command`, `edit_file`): Review `git diff` and update `documents/logs/2026-11-16/structure/proposed-next-version-tree.md` if `src/method_runtime.py` is added.
