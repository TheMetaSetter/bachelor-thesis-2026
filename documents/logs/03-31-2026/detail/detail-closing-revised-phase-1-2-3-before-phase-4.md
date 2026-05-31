---
date: 2026-03-31 22:55:02 +0700
planner: TheMetaSetter
git_commit: 87c0e9b2a092b3e3b5a5b6f6ea5b54b4b948555d
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed implementation plan for closing revised phase 1, 2, and 3 work before phase 4"
tags: [detail, phase-closure, smd, multitask, fusion, anomaly-injection]
status: complete
last_updated: 2026-03-31
last_updated_by: TheMetaSetter
source_plan: documents/logs/03-31-2026/plan/plan-closing-revised-phase-1-2-3-before-phase-4.md
---

# Detail: Detailed implementation plan for closing revised phase 1, 2, and 3 work before phase 4

## Overview

This detailed plan translates `documents/logs/03-31-2026/plan/plan-closing-revised-phase-1-2-3-before-phase-4.md` into a programming-level execution sequence for the current repository. The user explicitly requested that this detail pass proceed directly from the plan document while partly skipping `prompts/3_structure_prompt.md`. Accordingly, this document assumes the revised directory intent from `documents/design/design_starter.md` and focuses on file edits, interfaces, transition sequencing, and measurable closure criteria.

The governing constraints are:

- strictly follow `codebase_preferences.md`;
- keep one model in one file;
- keep the runnable path readable and linear;
- close revised Phase 1, Phase 2, and Phase 3 before any Phase 4 implementation begins;
- implement fusion and other thesis calculations in exact agreement with the notations in `documents/design/idea.md`.

In particular, the multitask fusion code must not use the current `identity` or `average` shortcut as the final active path. The active implementation must follow:

$$
H_{\text{rec}} = \beta \hat H^{(d)} + (1-\beta)\hat H^{(c)},
\qquad
H_{\text{cls}} = \alpha \hat H^{(d)} + (1-\alpha)\hat H^{(c)},
$$

with

$$
\alpha = \sigma(a) \in (0,1),
\qquad
\beta = \sigma(b) \in (0,1),
\qquad
a,b \in \mathbb{R}.
$$

This same strictness applies to the other thesis-facing calculations documented in `documents/design/idea.md`, including the continuous prototype lookup, the discrete codebook assignment, the fused prediction paths, and the offline objective.

## Global Contracts And Implementation Rules

### Runtime layers

The active runtime should be organized around four layers:

1. Configuration
2. Data
3. Model
4. Engine

The previous task layer may remain temporarily only as a migration shim. It must not remain part of the final active path for the offline baseline or multitask model, because `codebase_preferences.md` requires one-model-one-file and colocated training logic.

### Dataset and batch contracts

The current repository already uses a batch structure compatible with the revised documents. The active contracts to preserve are:

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

```python
batch = {
    "x": Tensor[B, L, D],
    "point_labels": Optional[Tensor[B, L]],
    "mask": Optional[Tensor[B, L, D]],
    "timestamps": Optional[Tensor[B, L]],
    "meta": list[dict],
}
```

The active default window parameters remain:

- `L = 100`
- `stride = 10`

### Model output and step-output contracts

Every active model must return:

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

Every active stage method must return:

```python
step_output = {
    "loss": Tensor,
    "log": dict[str, float],
    "outputs": dict,
}
```

### Design pattern rules

- Composition over inheritance remains the main rule inside data and engine code.
- The adapter pattern is still allowed for encoder integration, but model-specific adapter logic should remain in the owning model file.
- The earlier strategy pattern for tasks is now downgraded to a temporary migration device only. The final active offline path should use model-owned `training_step`, `validation_step`, and `test_step`.
- Registry or factory construction remains required for datasets and models, but the registry must not coexist with a second direct-constructor path in active scripts.

## Phase 1 - Close the offline vertical slice around a self-contained reconstruction model

### Phase summary

This phase closes the revised Phase 1 debt by keeping the existing SMD runnable path intact while removing the split between `src/models/reconstruction_mlp_ae.py` and `src/tasks/reconstruction_task.py`, and by making `scripts/train.py` and `scripts/evaluate.py` use a single registry-driven dataset construction path. The thesis objective of this phase is to preserve the minimal `SMD -> baseline model -> train/eval` slice while making it consistent with the one-model-one-file rule.

### File-level edits

The following files should be modified in this phase:

```text
src/models/base_model.py
src/models/reconstruction_mlp_ae.py
src/tasks/reconstruction_task.py
src/engine/trainer.py
src/engine/evaluator.py
src/core/registry.py
scripts/train.py
scripts/evaluate.py
tests/test_model_shapes.py
tests/test_one_train_step.py
tests/test_checkpoint_roundtrip.py
tests/test_registry.py
configs/experiment/baseline/smd__thesis_multitask__vertical-slice__w100__seed7__default.yaml
```

### Explicit edit content

`src/models/base_model.py`

- Keep `forward(batch)` as the canonical model inference interface.
- Ensure `training_step(batch)`, `validation_step(batch)`, and `test_step(batch)` are mandatory abstract methods.
- Do not add task-owned arguments to these methods.

`src/models/reconstruction_mlp_ae.py`

- Keep the existing architecture and output contract.
- Add:
  - `_compute_reconstruction_loss(outputs, batch) -> Tensor`
  - `_build_stage_log(stage_name, outputs, loss) -> dict[str, float]`
  - `training_step(batch) -> dict`
  - `validation_step(batch) -> dict`
  - `test_step(batch) -> dict`
- Keep `validate_batch(batch)` before forward computation.
- Keep `validate_model_outputs(outputs)` before returning outputs.
- Compute the reconstruction loss as:

$$
\mathcal{L}_{\text{recon}}
=
\frac{1}{B L D} \left\| X - \hat X \right\|_F^2.
$$

- Keep `outputs["point_scores"]` as mean squared error across channels per timestep.
- Keep `outputs["window_scores"]` as the mean of `point_scores` across the window dimension.

`src/tasks/reconstruction_task.py`

- Reduce this file to an optional temporary compatibility wrapper only if needed during migration.
- After the Phase 1 closure tests pass, the active training and evaluation path must not depend on this file.

`src/engine/trainer.py`

- Remove the engine’s reliance on `self.task.training_step(self.model, batch_on_device)`.
- Make the trainer call:
  - `self.model.training_step(batch_on_device)`
  - `self.model.validation_step(batch_on_device)`
- Keep the engine responsible only for:
  - device transfer;
  - optimizer step;
  - checkpoint calls;
  - experiment logging;
  - epoch aggregation.

`src/engine/evaluator.py`

- Remove the evaluator’s reliance on `self.task.test_step(model, batch_on_device)`.
- Make the evaluator call `model.test_step(batch_on_device)` directly.
- Keep overlap aggregation and pointwise metric computation unchanged in this phase.

`src/core/registry.py`

- Preserve `register_dataset`, `register_model`, and `build_*` functions.
- The registry may keep task builders temporarily for compatibility, but the active Phase 1 path should not require a task builder for the reconstruction baseline.

`scripts/train.py`

- Keep the component-registration helper, but make the active data path use `build_dataset("smd", experiment_config["data"])` or the same logical registry call.
- Remove the direct call to `build_smd_dataloaders(...)` from the active code path.
- Remove the active dependency on a reconstruction task object once the model-owned stage methods are in place.

`scripts/evaluate.py`

- Mirror the training script’s registry-only data construction.
- Remove the active dependency on a reconstruction task object.

### Interface and contract definitions

The Phase 1 closure must preserve:

- `batch["x"]` as `[B, L, D]`;
- `outputs["hidden"]` as `[B, L, H]`;
- `outputs["recon"]` as `[B, L, D]`;
- `step_output["loss"]` as a scalar tensor;
- `step_output["log"]` as a serialization-friendly dictionary.

### Design pattern application

- Composition over inheritance remains visible in the data layer and engine layer.
- Registry usage is enforced as the only active construction path for the scripts.
- The strategy pattern is retired from the active reconstruction path. The owning model file now contains both inference and training logic.

### Risk mitigation

- Prevent regression in the runnable path by changing the engine interface first, then migrating the reconstruction model stage methods, then removing the task dependency from the scripts.
- Prevent hidden dual codepaths by treating any remaining direct call to `build_smd_dataloaders(...)` inside active scripts as a Phase 1 failure.
- Prevent schema drift by leaving `src/core/contracts.py` checks in place.

### Test plan and validation steps

- `tests/test_model_shapes.py` must still verify the reconstruction output shapes.
- `tests/test_one_train_step.py` must be updated to call the model-owned training step.
- `tests/test_checkpoint_roundtrip.py` must still pass unchanged from the user perspective.
- `tests/test_registry.py` must verify registry-based dataset and model construction for the active script path.

Recommended Phase 1 regression command:

```bash
pytest -q tests/test_config_loading.py tests/test_smd_dataset_shapes.py tests/test_windowizer.py tests/test_model_shapes.py tests/test_one_train_step.py tests/test_checkpoint_roundtrip.py tests/test_registry.py
```

### Acceptance criteria

- `src/models/reconstruction_mlp_ae.py` is self-contained for inference and training logic.
- `src/engine/trainer.py` and `src/engine/evaluator.py` call model-owned stage methods.
- `scripts/train.py` and `scripts/evaluate.py` use the registry-driven dataset path only.
- The Phase 1 regression command passes.

## Phase 2 - Close the multitask model boundary and implement strict thesis notation inside one file

### Phase summary

This phase closes the revised Phase 2 debt by making `src/models/thesis_multitask.py` the authoritative home for the multitask model, including the prototype branches, fusion equations, score computation, and stage-specific losses. The thesis objective of this phase is not merely to keep a multitask model present, but to make its active implementation follow the notation in `documents/design/idea.md` exactly enough that the code and the thesis document describe the same model.

### File-level edits

The following files should be modified in this phase:

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

### Explicit edit content

`src/models/thesis_multitask.py`

- Reorganize the file into explicit top-to-bottom sections:
  - encoder block;
  - continuous prototype block;
  - discrete prototype block;
  - fusion scalars and fusion equations;
  - reconstruction head;
  - classification head;
  - offline objective helpers;
  - stage methods.
- Move the active logic from:
  - `src/models/modules/continuous_prototypes.py`
  - `src/models/modules/discrete_prototypes.py`
  - `src/models/modules/fusion.py`
  - `src/losses/classification.py`
  - `src/losses/prototype.py`
  - `src/tasks/multitask_tsad_task.py`
  into this file.

Strict mathematical implementation requirements:

- Encoder output:

$$
H = f_\theta(X) \in \mathbb{R}^{B \times L \times d_h}.
$$

- Continuous branch:

$$
s^{(c)}_{b\ell k}
=
\frac{\langle h_{b,\ell}, p_k^{(c)} \rangle}{\sqrt{d_h}},
\qquad
a^{(c)}_{b\ell k}
=
\frac{\exp(s^{(c)}_{b\ell k})}{\sum_{j=1}^{K_c} \exp(s^{(c)}_{b\ell j})},
$$

$$
\hat h^{(c)}_{b,\ell}
=
\sum_{k=1}^{K_c} a^{(c)}_{b\ell k} p_k^{(c)}.
$$

- Discrete branch:

$$
q_{b,\ell} = W_d h_{b,\ell} + b_d,
$$

$$
\pi_{b\ell k}
=
\frac{\exp\left((q_{b\ell k} + g_{b\ell k}) / \tau\right)}
{\sum_{j=1}^{K_d} \exp\left((q_{b\ell j} + g_{b\ell j}) / \tau\right)},
$$

$$
\hat h^{(d)}_{b,\ell}
=
\sum_{k=1}^{K_d} \pi_{b\ell k} e_k^{(d)}.
$$

- Fusion must follow exactly:

$$
H_{\text{rec}} = \beta \hat H^{(d)} + (1-\beta)\hat H^{(c)},
\qquad
H_{\text{cls}} = \alpha \hat H^{(d)} + (1-\alpha)\hat H^{(c)}.
$$

- Fusion scalar parameterization must follow:

$$
\alpha = \sigma(a),
\qquad
\beta = \sigma(b).
$$

- The default prediction paths must follow:

$$
\hat X = D_\phi(H_{\text{rec}}),
\qquad
\hat y = C_\psi(\mathrm{MeanPool}(H_{\text{cls}})).
$$

- The active code must therefore not use:
  - one shared fused tensor for both heads;
  - the current `average` fusion as the final logic;
  - direct prediction from branch-local hidden states as the default path.

- The offline objective must include the same components named in `documents/design/idea.md`:

$$
\mathcal{L}_{\text{total}} =
\mathcal{L}_{\text{recon}} +
\lambda_{\text{cls}} \mathcal{L}_{\text{cls}} +
\lambda_{\text{div}} \mathcal{L}_{\text{div}} +
\lambda_{\text{var}} \mathcal{L}_{\text{var}} +
\lambda_{\text{cov}} \mathcal{L}_{\text{cov}} +
\lambda_{\text{use}} \mathcal{L}_{\text{use}} +
\lambda_{\text{gate}} \mathcal{L}_{\text{gate}}.
$$

Terminology normalized on 2026-04-02. Current design target: gate entropy regularization. Current implementation status: `src/models/thesis_multitask.py` now uses gate-entropy regularization directly while retaining the legacy margin field only for backward checkpoint compatibility.

- The file should implement each active loss term with the same symbol names in code comments or helper names where reasonable, for example:
  - `_compute_reconstruction_loss`
  - `_compute_classification_loss`
  - `_compute_cross_branch_diversity_loss`
  - `_compute_variance_floor_loss`
  - `_compute_covariance_reduction_loss`
  - `_compute_prototype_usage_loss`
  - `_compute_gate_regularization_loss` as the current barrier-style helper, or a future helper aligned with gate entropy regularization once the implementation is updated

- If a loss term is intentionally staged in later activation, the helper should still exist in the file and return zero when disabled by config. This preserves the exact objective surface without inventing hidden codepaths.

`configs/model/thesis_multitask.yaml`

- Replace `fusion_mode: identity` or `fusion_mode: average` as the primary active path.
- Add explicit fields for:
  - `continuous_num_prototypes`
  - `discrete_codebook_size`
  - `gumbel_temperature`
  - `alpha_logit_init`
  - `beta_logit_init`
  - `lambda_cls`
  - `lambda_div`
  - `lambda_var`
  - `lambda_cov`
  - `lambda_use`
  - `lambda_gate` for gate entropy regularization in the design surface
- Use clear names that map directly onto the thesis notation.

`src/tasks/multitask_tsad_task.py`

- Reduce this file to a temporary compatibility shim only if absolutely required during migration.
- The active multitask path must not depend on this file once Phase 2 is closed.

`src/models/modules/*.py` and `src/losses/*.py`

- Keep only genuinely reusable infrastructure, if any remains.
- If a file is only model-specific logic for the thesis model, it should no longer be required by the active code path after this phase.

### Interface and contract definitions

The multitask model must preserve the same public contract:

- `hidden: [B, L, d_h]`
- `recon: [B, L, D]`
- `logits: [B, C]` or equivalent classification shape
- `point_scores: [B, L]`
- `window_scores: [B]`

Additional batch keys required by the multitask path:

```python
batch["classification_labels"]: Tensor[B]
batch["synthetic_anomaly_mask"]: Tensor[B, L]
batch["augmentation_metadata"]: list[dict]
```

These keys may be absent in clean inference batches, but the model-owned training step must prepare or require them explicitly through one consistent path.

### Design pattern application

- Composition remains valid inside one model file through private helper methods and private helper classes.
- The adapter pattern still applies to the encoder boundary, but model-specific adapter logic stays in `src/models/thesis_multitask.py`.
- The earlier strategy pattern for tasks is not the final active design here. It is transitional at most.
- Registry construction still applies to the model itself.

### Risk mitigation

- Prototype redundancy:
  keep branch-specific diagnostics in `outputs["aux"]`, including continuous weights, discrete assignments, and fused scalar values.
- Fusion collapse:
  log `alpha`, `beta`, branch norms, and fused-branch contribution summaries during training.
- Readability degradation:
  separate each mathematical block with concise comments matching the notation names from `documents/design/idea.md`.

### Test plan and validation steps

- `tests/test_multitask_shapes.py` must verify:
  - `hidden`, `recon`, `logits`, `point_scores`, and `aux`;
  - presence of fused-task tensors;
  - presence of `alpha` and `beta` in `aux` or an equivalent inspection field.
- `tests/test_one_multitask_train_step.py` must exercise `ThesisMultitaskModel.training_step(batch)`.
- Add assertions that the active fusion path is not the deprecated identity or average shortcut when the thesis mode is enabled.

Recommended Phase 2 regression command:

```bash
pytest -q tests/test_multitask_shapes.py tests/test_one_multitask_train_step.py tests/test_registry.py
```

### Acceptance criteria

- `src/models/thesis_multitask.py` fully owns the active multitask path.
- The fusion implementation follows the exact `alpha` and `beta` notation from `documents/design/idea.md`.
- The active code path no longer relies on model-specific task, loss, or fusion-module files.
- The Phase 2 regression command passes.

## Phase 3 - Close augmentation debt with CARLA-aligned subsequence anomalies and visualization

### Phase summary

This phase closes the revised Phase 3 debt by replacing the simplified local perturbation injector with a CARLA-aligned subsequence mechanism and by adding a maintained inspection surface for injected anomalies. The thesis objective is to ensure that anomaly-type supervision is generated in a form consistent with the stated design rather than as a placeholder local perturbation pipeline.

### File-level edits

The following files should be modified or added in this phase:

```text
src/data/augment.py
src/models/thesis_multitask.py
scripts/visualize_synthetic_anomalies.py
tests/test_synthetic_anomaly_injection.py
tests/test_synthetic_anomaly_visualization.py
tests/test_multitask_shapes.py
tests/test_one_multitask_train_step.py
configs/model/thesis_multitask.yaml
```

### Explicit edit content

`src/data/augment.py`

- Replace the active three-type local injector with a CARLA-aligned subsequence anomaly mechanism.
- Preserve the public batch-returning interface if this file remains genuinely reusable.
- The mechanism should generate anomaly windows and masks through anomaly families aligned with the CARLA reference rather than only per-channel local edits.
- Keep augmentation metadata explicit and serialization-friendly:
  - `is_synthetic_anomaly`
  - `anomaly_family`
  - `start_index`
  - `end_index`
  - `affected_channels`
  - any family-specific parameters needed for reproducibility.

`src/models/thesis_multitask.py`

- Integrate the revised augmentation path into the model-owned training step.
- The model must consume the augmented batch without creating a second hidden codepath for clean training.
- Keep:
  - `classification_labels`
  - `synthetic_anomaly_mask`
  - `augmentation_metadata`
  visible in the active training path.

`scripts/visualize_synthetic_anomalies.py`

- Create a script-level inspection tool that:
  - constructs or loads a reduced batch;
  - applies the active augmentation path;
  - saves artifacts to disk;
  - shows clean and augmented windows clearly.
- A minimal artifact may be a static image per chosen sample.
- The output should be suitable for manual inspection by the user without notebook-only work.

### Interface and contract definitions

The augmentation path must preserve the base batch contract and extend it only through explicit extra keys:

```python
augmented_batch = {
    "x": Tensor[B, L, D],
    "point_labels": Tensor[B, L],
    "classification_labels": Tensor[B],
    "synthetic_anomaly_mask": Tensor[B, L],
    "augmentation_metadata": list[dict],
    "mask": Optional[Tensor[B, L, D]],
    "timestamps": Optional[Tensor[B, L]],
    "meta": list[dict],
}
```

### Design pattern application

- If augmentation remains a reusable data-level concern, it may stay in `src/data/augment.py`.
- If augmentation becomes tightly tied to the thesis model objective, move the active logic into `src/models/thesis_multitask.py` and keep only generic helpers in `src/data/augment.py`.
- The guiding rule is still one-model-one-file and the least amount of codepaths.

### Risk mitigation

- Adaptation contamination is still a Phase 4 concern and must not be addressed by prematurely adding online code here.
- Evaluation metric inflation is mitigated by keeping the augmentation surface explicit and separately testable rather than hiding synthetic labels in evaluation code.
- Reproducibility risk is mitigated by recording anomaly-family metadata and preparing for later DVC-tracked augmented artifacts.

### Test plan and validation steps

- Expand `tests/test_synthetic_anomaly_injection.py` to check:
  - shape preservation;
  - anomaly mask content;
  - classification label content;
  - anomaly family metadata;
  - metadata retention from the source batch.
- Add `tests/test_synthetic_anomaly_visualization.py` to confirm that the visualization path emits an artifact.
- Re-run `tests/test_multitask_shapes.py` and `tests/test_one_multitask_train_step.py` with the revised augmentation path.

Recommended Phase 3 regression command:

```bash
pytest -q tests/test_synthetic_anomaly_injection.py tests/test_synthetic_anomaly_visualization.py tests/test_multitask_shapes.py tests/test_one_multitask_train_step.py
```

### Acceptance criteria

- The active augmentation path is CARLA-aligned at the mechanism level.
- The repository contains a maintained visualization script for injected anomalies.
- The multitask training path remains green under the revised augmentation path.
- The Phase 3 regression command passes.

## Pre-Phase-4 Gate

Phase 4 must not begin until all of the following are true:

- Phase 1 acceptance criteria are satisfied.
- Phase 2 acceptance criteria are satisfied.
- Phase 3 acceptance criteria are satisfied.
- `scripts/train.py` and `scripts/evaluate.py` use only the registry-driven dataset path.
- The active reconstruction path is model-owned.
- The active multitask path is model-owned.
- The active fusion path implements `H_rec` and `H_cls` using the exact `alpha` and `beta` notation from `documents/design/idea.md`.
- No active Phase 1 to Phase 3 codepath depends on model-specific logic in `src/tasks/`, `src/losses/`, or `src/models/modules/`.

Only after this gate is passed should the repository add:

```text
src/models/online_adaptation.py
src/engine/online_loop.py
tests/test_online_adaptation_step.py
tests/test_online_state_roundtrip.py
configs/model/online_adaptation.yaml
```

## Final Validation Sequence

Run the following sequence in order:

1. Phase 1 regression set

```bash
pytest -q tests/test_config_loading.py tests/test_smd_dataset_shapes.py tests/test_windowizer.py tests/test_model_shapes.py tests/test_one_train_step.py tests/test_checkpoint_roundtrip.py tests/test_registry.py
```

2. Phase 2 regression set

```bash
pytest -q tests/test_multitask_shapes.py tests/test_one_multitask_train_step.py tests/test_registry.py
```

3. Phase 3 regression set

```bash
pytest -q tests/test_synthetic_anomaly_injection.py tests/test_synthetic_anomaly_visualization.py tests/test_multitask_shapes.py tests/test_one_multitask_train_step.py
```

4. Pre-Phase-4 gate regression set

```bash
pytest -q tests/test_config_loading.py tests/test_smd_dataset_shapes.py tests/test_windowizer.py tests/test_model_shapes.py tests/test_one_train_step.py tests/test_checkpoint_roundtrip.py tests/test_registry.py tests/test_multitask_shapes.py tests/test_one_multitask_train_step.py tests/test_synthetic_anomaly_injection.py tests/test_synthetic_anomaly_visualization.py
```

## Completion Standard

This detailed plan is complete when the repository can close revised Phase 1, Phase 2, and Phase 3 without ambiguity and without appealing to a separate structure step. The decisive signals are not only that the code runs, but that the active implementation path matches `documents/design/idea.md`, `documents/design/design_starter.md`, and `codebase_preferences.md` at the same time.
