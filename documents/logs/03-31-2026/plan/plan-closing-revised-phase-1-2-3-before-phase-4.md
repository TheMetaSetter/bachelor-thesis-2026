---
date: 2026-03-31 22:36:32 +0700
planner: TheMetaSetter
git_commit: 87c0e9b2a092b3e3b5a5b6f6ea5b54b4b948555d
branch: dev
repository: bachelor-thesis-2026
topic: "Implementation plan for closing revised phase 1, 2, and 3 work before phase 4"
tags: [plan, time-series, anomaly-detection, phase-closure, smd]
status: complete
last_updated: 2026-03-31
last_updated_by: TheMetaSetter
source_research: documents/logs/03-31-2026/research/research-applying-revised-phase-1-2-3-before-phase-4.md
---

# Plan: Implementation plan for closing revised phase 1, 2, and 3 work before phase 4

## Current State

- The repository already contains an executable offline SMD path with parser, scaler, window dataset, collate function, registry, training script, evaluation script, reconstruction baseline, multitask model, synthetic anomaly injector, and focused tests.
- The existing code already follows the core batch and output contracts in most places. `src/data/loaders.py` emits a batch centered on `x: [B, L, D]`, and both `src/models/reconstruction_mlp_ae.py` and `src/models/thesis_multitask.py` return `hidden`, `recon`, `point_scores`, `window_scores`, and `aux`.
- The present implementation still reflects the older layered split rather than the revised one-model-one-file plan. Model-specific behavior is distributed across `src/models/`, `src/tasks/`, `src/losses/`, and `src/models/modules/`.
- `scripts/train.py` and `scripts/evaluate.py` register the SMD dataset builder in `src/core/registry.py` but then instantiate the data path through `build_smd_dataloaders(...)` directly. The registry and direct builder path therefore coexist.
- The synthetic anomaly path exists, but it remains simpler than the revised Phase 3 requirement. `src/data/augment.py` injects `spike`, `dropout`, and `level_shift` anomalies into one contiguous segment of one channel, rather than a CARLA-style subsequence family.
- Repository update on 2026-04-02: the visualization script now exists at `scripts/visualize_synthetic_anomalies.py`, and the repository now contains a conservative Phase 4 implementation at `src/data/stream.py`, `src/models/online_adaptation.py`, `src/engine/online_loop.py`, and `scripts/run_online_adaptation.py`. This older plan should therefore be read as the gate-closing work that has since been realized, not as the current absence state.

## Design Options

### Option A: Direct closure migration to the revised one-model-one-file architecture

This option treats the revised design documents as the target state immediately. The reconstruction baseline, the multitask thesis model, and the later online model each become self-contained model files with internal stage methods, while `src/tasks/`, model-specific `src/losses/`, and model-specific `src/models/modules/` are retired as model logic is folded back into the owning model file.

This option aligns most closely with the revised `documents/design/design_starter.md` and `documents/design/idea.md`. It is the most coherent closure path before Phase 4.

### Option B: Compatibility-first migration with temporary task wrappers

This option preserves the existing `task` strategy surface temporarily while migrating logic incrementally into the model files. The trainer and evaluator would first support model-owned step methods, but compatibility wrappers in `src/tasks/` would remain during the transition to reduce the size of each change set.

This option is safer operationally for a dirty worktree and a live codebase, but it must remain temporary. If it is not explicitly time-boxed, it will preserve the exact split-file debt the revised documents are trying to close.

### Option C: Feature-first closure that keeps the current split architecture

This option would keep the present `model + task + loss + module` arrangement and only add the missing CARLA-style anomalies, registry cleanup, and visualization surface.

This option does not align with the revised documents and should not be selected. It would close only part of the pre-Phase-4 gate and leave problem 1 unresolved.

## Selected Approach

The recommended approach is **Option A executed with the change sequencing discipline of Option B**.

In practical terms, this means:

- the target architecture is the revised one-model-one-file layout;
- migrations should happen in bounded slices that preserve testability after each slice;
- compatibility wrappers may exist only briefly while trainer and evaluator are being switched from task-owned stage methods to model-owned stage methods;
- the end of Phase 3 should remove model-specific reliance on `src/tasks/`, model-specific `src/losses/`, and model-specific `src/models/modules/`.

This approach aligns best with the revised design documents, the latest research note, and `codebase_preferences.md`.

## Risk And Mitigation

- Risk: collapsing model logic into one file may create very large, unreadable files.
  Mitigation: keep each model file internally structured into clearly delimited sections such as encoder block, prototype block, fusion block, scoring block, and stage-method block. Use small private helper methods and lightweight inner helper classes when necessary, but keep them in the same file.
- Risk: trainer and evaluator migration may break the current runnable path.
  Mitigation: first extend `BaseModel` and the engine to support `training_step`, `validation_step`, and `test_step`, then move one model at a time. Keep focused regression tests running after each slice.
- Risk: the dataset registry cleanup may change script behavior unexpectedly.
  Mitigation: update `tests/test_registry.py` and add script-level smoke assertions so the registry path and script path are the same code path before removing the direct loader call.
- Risk: CARLA-aligned augmentation may change batch semantics or silently invalidate existing tests.
  Mitigation: preserve the existing batch keys and add explicit tests for anomaly masks, classification labels, metadata retention, and deterministic reduced fixtures.
- Risk: user-visible anomaly visualization may become notebook-only and fall out of maintenance.
  Mitigation: implement it as a script under `scripts/` with a testable helper function, and require a saved artifact path as part of the acceptance criteria.
- Risk: some generic utilities in `src/tasks/`, `src/losses/`, or `src/models/modules/` may still be useful beyond one model.
  Mitigation: distinguish carefully between genuinely generic infrastructure and model-specific logic. Preserve only the generic pieces. Fold model-specific computation back into the owning model file.
- Risk: Phase 4 work may start early because the repository already contains `src/data/stream.py`.
  Mitigation: treat the revised pre-Phase-4 gate as a formal release gate. No online adaptation files should be added until the Phase 1 to Phase 3 closure tests are green and the gate conditions are documented as passed.

## Open Questions

- The revised documents require CARLA-aligned subsequence anomaly families. The exact correspondence between the current anomaly taxonomy and the intended CARLA family names still needs to be fixed in implementation.
- The repository uses `documents/logs/MM-DD-YYYY/plan/` rather than `documents/logs/MM-DD-YYYY/plans/`. The current plan follows the existing repository convention.
- The revised documents favor one-model-one-file over the earlier task strategy. The migration plan below assumes that the final stable interface is model-owned stage methods and that task wrappers are transitional only.

## Detailed Implementation Plan

### 1. Close revised Phase 1 by unifying the runnable offline vertical slice around registry-driven scripts and a self-contained reconstruction baseline

Modify these files first:

```text
src/models/base_model.py
src/models/reconstruction_mlp_ae.py
src/tasks/reconstruction_task.py
src/engine/trainer.py
src/engine/evaluator.py
src/core/registry.py
scripts/train.py
scripts/evaluate.py
tests/test_one_train_step.py
tests/test_checkpoint_roundtrip.py
tests/test_registry.py
tests/test_model_shapes.py
configs/experiment/smd_vertical_slice.yaml
```

Implementation instructions:

- Extend or confirm `src/models/base_model.py` so every model exposes:
  - `forward(batch: dict) -> dict`
  - `training_step(batch: dict) -> dict`
  - `validation_step(batch: dict) -> dict`
  - `test_step(batch: dict) -> dict`
- Move the reconstruction loss and metric logic from `src/tasks/reconstruction_task.py` into `src/models/reconstruction_mlp_ae.py`.
  - `ReconstructionMLPAutoencoder.training_step` should compute reconstruction loss from `outputs["recon"]` and `batch["x"]`.
  - `validation_step` and `test_step` should return the same standardized step structure with stage-specific log keys.
  - Preserve `validate_batch(batch)` and `validate_model_outputs(outputs)` calls in the model file.
- Update `src/engine/trainer.py` so the trainer calls `self.model.training_step(batch_on_device)` and `self.model.validation_step(batch_on_device)` instead of routing through `self.task`.
- Update `src/engine/evaluator.py` so the evaluator calls `model.test_step(batch_on_device)` directly.
- Keep the engine responsible only for looping, optimization, checkpointing, and logging. Model-specific losses must no longer live in the engine.
- Update `scripts/train.py` and `scripts/evaluate.py` so both scripts use `build_dataset("smd", experiment_config["data"])` or an equivalent registry call rather than calling `build_smd_dataloaders(...)` directly.
- Preserve `src/core/registry.py` as the minimal factory surface, but stop relying on `build_task(...)` for model-specific offline training.
- `src/tasks/reconstruction_task.py` may remain temporarily as a compatibility shim during the migration, but it should no longer be the active code path once the Phase 1 closure is complete.

Contract enforcement requirements:

- Batch contract remains `x: [B, L, D]`, optional `point_labels: [B, L]`, optional `mask: [B, L, D]`, optional `timestamps: [B, L]`, and `meta: list[dict]`.
- Model output contract remains `hidden`, `pooled`, `recon`, `logits`, `point_scores`, `window_scores`, and `aux`.
- Step-output contract remains:

```python
{
    "loss": Tensor,
    "log": dict[str, float],
    "outputs": dict,
}
```

Validation procedure:

- `tests/test_model_shapes.py` must still pass with the model-only reconstruction path.
- `tests/test_one_train_step.py` must exercise `ReconstructionMLPAutoencoder.training_step`.
- `tests/test_checkpoint_roundtrip.py` must confirm that the migration does not break checkpoint save and load.
- `tests/test_registry.py` must prove that the scripts use the registry path for datasets and models.

Acceptance conditions for revised Phase 1:

- the reconstruction baseline is readable and self-contained in `src/models/reconstruction_mlp_ae.py`;
- the train and evaluate scripts construct data through the registry path only;
- the trainer and evaluator no longer depend on a reconstruction task object.

### 2. Close revised Phase 2 by establishing the self-contained multitask model boundary

Modify these files next:

```text
src/models/thesis_multitask.py
src/models/modules/continuous_prototypes.py
src/models/modules/discrete_prototypes.py
src/models/modules/fusion.py
src/tasks/multitask_tsad_task.py
src/losses/classification.py
src/losses/prototype.py
tests/test_multitask_shapes.py
tests/test_one_multitask_train_step.py
tests/test_registry.py
configs/model/thesis_multitask.yaml
configs/task/multitask_tsad.yaml
```

Implementation instructions:

- Move model-specific prototype logic from:
  - `src/models/modules/continuous_prototypes.py`
  - `src/models/modules/discrete_prototypes.py`
  - `src/models/modules/fusion.py`
  into `src/models/thesis_multitask.py` as private helper classes or private helper methods.
- Move model-specific loss logic from:
  - `src/losses/classification.py`
  - `src/losses/prototype.py`
  into `src/models/thesis_multitask.py`.
- Move multitask stage logic from `src/tasks/multitask_tsad_task.py` into:
  - `ThesisMultitaskModel.training_step`
  - `ThesisMultitaskModel.validation_step`
  - `ThesisMultitaskModel.test_step`
- Preserve the current public output contract:
  - `hidden`
  - `pooled`
  - `recon`
  - `logits`
  - `point_scores`
  - `window_scores`
  - `aux`
- Preserve the current semantic sections in the multitask model file:
  - encoder block;
  - continuous prototype branch;
  - discrete prototype branch;
  - fusion logic;
  - reconstruction head;
  - classification head;
  - scoring logic;
  - stage-specific loss logic.
- Keep `src/tasks/multitask_tsad_task.py` only as a temporary compatibility shim if needed while tests are being migrated. It should not remain an active dependency at Phase 2 closure.

Configuration instructions:

- `configs/model/thesis_multitask.yaml` should become the primary configuration surface for:
  - encoder dimensions;
  - continuous prototype controls;
  - discrete codebook controls;
  - fusion mode;
  - classification head shape;
  - loss weights;
  - synthetic anomaly controls.
- `configs/task/multitask_tsad.yaml` should be deprecated or reduced to a temporary transition file if backward compatibility is still required during migration.

Validation procedure:

- `tests/test_multitask_shapes.py` must pass using the self-contained `ThesisMultitaskModel`.
- `tests/test_one_multitask_train_step.py` must call `model.training_step(batch)` rather than task-owned logic.
- `tests/test_registry.py` must verify that `build_model("thesis_multitask")` produces the active training object without requiring a separate multitask task object.

Acceptance conditions for revised Phase 2:

- `src/models/thesis_multitask.py` is the authoritative home for the multitask architecture and its stage behavior;
- model-specific prototype, fusion, and loss logic are no longer required from separate module and loss files;
- the multitask training path remains contract-stable and test-covered.

### 3. Close revised Phase 3 by replacing the simplified augmentation path with a CARLA-aligned subsequence mechanism and adding user-visible anomaly inspection

Modify or add these files:

```text
src/models/thesis_multitask.py
src/data/augment.py
scripts/visualize_synthetic_anomalies.py
tests/test_synthetic_anomaly_injection.py
tests/test_synthetic_anomaly_visualization.py
tests/test_multitask_shapes.py
tests/test_one_multitask_train_step.py
configs/model/thesis_multitask.yaml
```

Implementation instructions:

- Replace the current local `spike` / `dropout` / `level_shift` augmentation surface with a CARLA-aligned subsequence anomaly mechanism.
- The new augmentation path must still preserve the existing batch schema:
  - `x`
  - `point_labels`
  - `classification_labels`
  - `synthetic_anomaly_mask`
  - `augmentation_metadata`
- Keep augmentation metadata explicit and serialization-friendly. Each example should record whether an anomaly was injected, the anomaly family, the affected segment bounds, and any affected channels.
- The augmentation implementation may remain in `src/data/augment.py` only if it is kept genuinely reusable and detached from one particular model. If it becomes tightly thesis-model-specific, fold it into `src/models/thesis_multitask.py` to respect the one-model-one-file rule.
- Add `scripts/visualize_synthetic_anomalies.py`.
  - The script should build or load a reduced batch, apply the augmentation path, and save inspection artifacts to a user-visible output directory.
  - The saved artifact should show at minimum the clean signal, the augmented signal, and the anomaly interval or mask.
- Ensure the multitask model can consume the revised augmented batch without changing the batch contract used elsewhere in the repository.

Validation procedure:

- Expand `tests/test_synthetic_anomaly_injection.py` so it checks:
  - batch shape preservation;
  - anomaly mask shape;
  - classification label shape;
  - metadata retention;
  - nontrivial segment localization;
  - anomaly family naming.
- Add `tests/test_synthetic_anomaly_visualization.py` to confirm that the visualization script or helper emits an artifact successfully.
- Re-run `tests/test_multitask_shapes.py` and `tests/test_one_multitask_train_step.py` after the augmentation change to verify that the multitask path remains stable.

Acceptance conditions for revised Phase 3:

- the augmentation path reflects the intended CARLA-style subsequence mechanism rather than only simple local perturbations;
- the repository has a maintained script-level inspection surface for synthetic anomalies;
- the multitask path remains test-covered under the revised augmentation scheme.

### 4. Apply the explicit pre-Phase-4 gate as a formal repository checkpoint

Before any Phase 4 file is added, verify all of the following:

- `scripts/train.py` uses registry-driven dataset construction only.
- `scripts/evaluate.py` uses registry-driven dataset construction only.
- `src/engine/trainer.py` and `src/engine/evaluator.py` call model-owned stage methods.
- `src/models/reconstruction_mlp_ae.py` owns reconstruction-stage logic directly.
- `src/models/thesis_multitask.py` owns multitask-stage logic directly.
- The active multitask anomaly injector is CARLA-aligned at the mechanism level.
- `scripts/visualize_synthetic_anomalies.py` exists and is test-covered.
- No active Phase 1 to Phase 3 path still depends on model-specific logic being stored in `src/tasks/`, `src/losses/`, or `src/models/modules/`.

Only after those conditions are met should the repository begin adding:

```text
src/models/online_adaptation.py
src/engine/online_loop.py
tests/test_online_adaptation_step.py
tests/test_online_state_roundtrip.py
configs/model/online_adaptation.yaml
```

## Test Plan

The closure work should be validated in this order:

1. Phase 1 regression set
   - `pytest -q tests/test_config_loading.py tests/test_smd_dataset_shapes.py tests/test_windowizer.py tests/test_model_shapes.py tests/test_one_train_step.py tests/test_checkpoint_roundtrip.py tests/test_registry.py`
2. Phase 2 regression set
   - `pytest -q tests/test_multitask_shapes.py tests/test_one_multitask_train_step.py tests/test_registry.py`
3. Phase 3 regression set
   - `pytest -q tests/test_synthetic_anomaly_injection.py tests/test_synthetic_anomaly_visualization.py tests/test_multitask_shapes.py tests/test_one_multitask_train_step.py`
4. Pre-Phase-4 gate regression set
   - run the full Phase 1 to Phase 3 suite together and confirm no active script path reintroduces the removed split-file logic.

## Validation Procedures

- Perform a file-level validation pass after each migration slice to ensure that no script still imports the retired active path.
- Inspect `git diff` after each slice to confirm that model-specific logic is moving toward the owning model file rather than merely being copied.
- Treat any remaining imports from `src/tasks/`, `src/losses/`, or `src/models/modules/` into the active reconstruction or multitask training path as evidence that the relevant phase is not yet closed.
- Do not begin any implementation of projector logic, online adaptation state, or an online loop until the gate above is explicitly satisfied.

## Recommended Sequence

1. Migrate the reconstruction path to model-owned stage methods and registry-only data construction.
2. Migrate the multitask path to model-owned stage methods and fold model-specific module and loss logic into `src/models/thesis_multitask.py`.
3. Replace the simplified augmentation mechanism with the CARLA-aligned subsequence mechanism and add anomaly visualization.
4. Run the full pre-Phase-4 regression set.
5. Mark revised phases 1, 2, and 3 as closed only after the gate conditions are met.

This sequence is the most consistent with the revised design documents and the current repository state. It closes the known offline design debt before any online adaptation code is introduced.
