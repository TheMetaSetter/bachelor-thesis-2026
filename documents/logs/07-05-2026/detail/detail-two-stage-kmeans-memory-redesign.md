# Two-Stage KMeans Memory Redesign Detailed Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the repository from the current heuristic memory-bootstrap state to the approved two-stage k-means memory design, while preserving the existing data contract, model contract, registry contract, and runner contract.

**Architecture:** Keep the current `thesis_multitask` model file as the owner of offline semantics, keep the dedicated two-stage runner as the owner of orchestration, and keep the trainer generic. The implementation should proceed as a minimal vertical slice: first stabilize the offline interfaces, then replace memory initialization with deterministic k-means, then confirm fusion and synthetic classification behavior, then add online adaptation only after the offline contracts are fixed.

**Tech Stack:** Python, PyTorch, PyYAML, `pytest`.

---

## Overview

The current repository already contains the structural pieces needed for the redesign: a stable batch contract, a self-contained multitask model, stage-aware freezing logic, a two-stage runner, and a test suite that exercises configuration loading and memory initialization. The main gap is semantic rather than structural: memory bootstrap still uses a covering-vector heuristic, while the approved design requires k-means-based initialization from training-split latent tokens only.

This detailed plan converts the outline into a buildable sequence. The first phases lock the existing offline vertical slice and replace the bootstrap logic with k-means. Later phases keep the continuous and discrete prototype branches, confirm task-specific fusion and synthetic anomaly supervision, and finally prepare the online adaptation stage with a residual projector and explicit safeguards.

---

## Phase 1: Stabilize the core contracts and the minimal vertical slice

### Phase Summary

This phase preserves the current repository-wide contracts before any algorithmic changes. The batch contract remains `batch["x"]` with shape `[B, L, D]`; the encoder contract remains `outputs["hidden"]` with shape `[B, L, H]`; and the model-output contract remains the fixed top-level dictionary validated by `src/core/contracts.py`. The purpose is to keep the current offline path readable and testable while ensuring later prototype work lands on a stable interface.

### Files and Edits

- Modify: `src/core/contracts.py`
- Modify: `src/data/window.py`
- Modify: `src/core/registry.py`
- Modify: `src/models/thesis_multitask.py`
- Modify: `src/models/thesis_multitask_setup_mixin.py`
- Modify: `src/models/thesis_multitask_state_mixin.py`
- Modify: `tests/test_windowizer.py`
- Modify: `tests/test_model_shapes.py`
- Modify: `tests/test_multitask_shapes.py`
- Modify: `tests/test_one_multitask_train_step.py`

### Detailed Edit Content

1. In `src/core/contracts.py`, preserve the current validators and make sure they remain the single source of truth for:
   - raw sequence validation,
   - window validation,
   - batch validation,
   - online batch validation,
   - model-output validation.

   The main objective is not to broaden the contract. The objective is to keep the exact shape and key checks stable so later phases can rely on them without adding alternate schemas.

2. In `src/data/window.py`, keep the fixed-length window slicing logic and ensure it continues to propagate:
   - `entity_id`,
   - `split`,
   - `start_index`,
   - `end_index`,
   - `window_size`,
   - absolute indexing metadata.

   The implementation must continue to produce windows that are compatible with the batch contract used by the trainer and the model.

3. In `src/core/registry.py`, keep the dataset/model registry as the mechanism for instantiating components from config names. No direct imports should be introduced into the trainer or runner as a replacement for registry-based construction.

4. In `src/models/thesis_multitask.py`, preserve the self-contained model-file structure. The public model stays in one file; the mixins remain internal organization only. Do not split the model across task-specific files.

5. In `src/models/thesis_multitask_setup_mixin.py` and `src/models/thesis_multitask_state_mixin.py`, keep the stage-aware flags and lifecycle state reporting intact. The phase labels `stage_a_multitask_pretraining` and `stage_b_fusion_finetuning` must remain the active two-stage labels.

### Interface and Contract Definitions

- Dataset / batch contract:
  - `batch["x"]`: `Tensor[B, L, D]`
  - `batch["point_labels"]`: optional `Tensor[B, L]`
  - `batch["mask"]`: optional `Tensor[B, L, D]`
  - `batch["timestamps"]`: optional `Tensor[B, L]`
  - `batch["meta"]`: list of metadata dictionaries

- Encoder contract:
  - `outputs["hidden"]`: `Tensor[B, L, H]`
  - `outputs["pooled"]`: optional `Tensor[B, H]`
  - `outputs["aux"]`: dict with encoder metadata

- Model-output contract:
  - `hidden`, `pooled`, `recon`, `logits`, `point_scores`, `window_scores`, `aux`
  - The `aux` dictionary may carry branch and memory state, but the top-level keys must stay stable.

### Design Pattern Application

- Composition over inheritance: keep the model assembled through mixins, but do not create a deeper inheritance tree.
- Adapter pattern for encoders: preserve the encoder contract so MLP and CNN encoders can remain interchangeable under one interface.
- Strategy pattern for tasks: keep reconstruction and classification behavior selected by phase and runtime configuration rather than by separate model classes.
- Registry/factory pattern: preserve `src/core/registry.py` as the single build path for datasets and models.

### Risk Mitigation

- Prototype redundancy is not addressed yet; only the contract surface is stabilized here.
- Fusion collapse is not changed yet; preserve the current monitoring hooks so later phases can compare branches.
- Adaptation contamination is not yet relevant in this phase, but the online batch contract should remain unchanged so the future projector can reuse it.
- Projector drift is deferred until the online phase.
- Metric inflation is avoided by not changing evaluation semantics at this stage.

### Test Plan and Validation

- Run `pytest tests/test_windowizer.py -v`.
- Run `pytest tests/test_model_shapes.py tests/test_multitask_shapes.py tests/test_one_multitask_train_step.py -v`.
- Confirm that the current model still accepts a single batch schema and emits the expected top-level outputs.

### Acceptance Criteria

- Windowing still produces fixed-length windows with correct metadata.
- The batch and output validators still pass for the current model surfaces.
- The multitask model remains instantiable from the registry and still completes one training step.

---

## Phase 2: Replace memory initialization with k-means-based prototype bootstrapping

### Phase Summary

This phase changes the one behavior that is explicitly mismatched with the SSOT: memory initialization. The continuous prototype bank must be seeded from clean latent tokens only, and the discrete codebook must be seeded from class-stratified synthetic anomaly tokens only. The model should remain the owner of this logic so that the runner and trainer stay simple.

### Files and Edits

- Modify: `src/models/thesis_multitask_state_mixin.py`
- Modify: `src/models/thesis_multitask_setup_mixin.py` only if any initialization flags need simplification
- Modify: `src/models/thesis_multitask_components.py` only if a small helper or config constant is required
- Modify: `tests/test_multitask_memory_initialization.py`
- Modify: `tests/test_offline_pretraining_two_stage_runner.py`
- Modify: `tests/test_offline_pretraining_two_stage_config_loading.py`

### Detailed Edit Content

1. In `src/models/thesis_multitask_state_mixin.py`, replace the current `_select_covering_vectors(...)`-based memory bootstrap with a deterministic k-means routine. The state mixin should still:
   - collect latent pools from the training split only,
   - distinguish continuous and discrete pools,
   - store the initialization metadata,
   - update `memory_initialized`, `memory_training_enabled`, and `memory_initialization_epoch`.

2. Add a private helper such as `_run_kmeans(tokens, k, num_iterations, seed_mode)` that:
   - normalizes tokens before clustering,
   - seeds centers deterministically,
   - alternates assignment and centroid recomputation,
   - returns normalized centroids.

3. Update the continuous memory initialization path so it consumes only clean latent tokens and writes exactly `continuous_num_prototypes` centroids into `continuous_prototype_bank`.

4. Update the discrete memory initialization path so it groups anomaly tokens by class and writes exactly `discrete_codebook_size` centroids into `discrete_codebook`.

5. Keep the checkpoint extra-state contract unchanged so that Stage B initialization still materializes `stage_b_init.pt` and preserves `memory_initialized=True`.

### Interface and Contract Definitions

- Continuous memory initialization:
  - Input: clean train latent tokens
  - Method: k-means
  - Output: `continuous_num_prototypes` normalized centroids

- Discrete memory initialization:
  - Input: synthetic train latent tokens grouped by class
  - Method: per-class k-means
  - Output: `discrete_codebook_size` normalized centroids

- Bootstrap state contract:
  - `memory_initialized`: boolean
  - `memory_training_enabled`: boolean
  - `memory_initialization_epoch`: optional integer
  - `continuous_memory_source_label`: explicit string
  - `discrete_memory_source_label`: explicit string

### Design Pattern Application

- Composition over inheritance: keep the bootstrap logic inside the existing model state mixin rather than extracting a separate training system.
- Adapter pattern for encoders: use the same encoder output contract to derive latent pools for clustering, regardless of MLP or CNN encoder family.
- Strategy pattern for tasks: keep the discrete and continuous bootstrap strategies separate, because they have different source pools and different cluster semantics.
- Registry/factory pattern: keep dataset construction unchanged so the bootstrap consumes the same train loader that the trainer already builds.

### Risk Mitigation

- Prototype redundancy is controlled by testing both banks independently and asserting exact prototype counts.
- Fusion collapse is not directly modified in this phase, but the memory source labels should remain visible in diagnostics so later phases can inspect branch usage.
- Adaptation contamination is irrelevant here because the bootstrap pool must come from the training split only.
- Projector drift is not introduced yet.
- Evaluation metric inflation is avoided by keeping the stage handoff and checkpoint metadata explicit.

### Test Plan and Validation

- Add or tighten tests in `tests/test_multitask_memory_initialization.py` so they verify:
  - continuous initialization uses k-means centroids,
  - discrete initialization uses class-stratified centroids,
  - memory buffers are normalized,
  - memory metadata reflects the approved source labels.
- Run `pytest tests/test_multitask_memory_initialization.py -v`.
- Run `pytest tests/test_offline_pretraining_two_stage_runner.py tests/test_offline_pretraining_two_stage_config_loading.py -v`.

### Acceptance Criteria

- Continuous memory is no longer initialized from a heuristic covering-vector routine.
- Discrete memory is no longer initialized from raw token collection alone.
- Stage B still receives an initialization checkpoint with initialized memory state.
- The tests confirm exact prototype counts and normalized centroid buffers.

---

## Phase 3: Align the task-specific fusion and synthetic anomaly classification path

### Phase Summary

This phase confirms that the continuous and discrete prototype branches remain separate until fusion, then verifies that reconstruction and classification use the fused task-specific representations. Synthetic anomaly injection remains the classification supervision mechanism, and the objective surface should remain modular so that optional regularizers stay configuration-driven.

### Files and Edits

- Modify: `src/models/thesis_multitask_routing_mixin.py`
- Modify: `src/models/thesis_multitask_loss_mixin.py`
- Modify: `src/models/thesis_multitask_setup_mixin.py`
- Modify: `src/data/augment.py` only if taxonomy or metadata fields need to be aligned with the current SSOT
- Modify: `tests/test_thesis_multitask_classification_path_toggle.py`
- Modify: `tests/test_multitask_objective_controls.py`
- Modify: `tests/test_synthetic_anomaly_injection.py`
- Modify: `tests/test_one_multitask_train_step.py`

### Detailed Edit Content

1. In `src/models/thesis_multitask_routing_mixin.py`, keep the continuous and discrete lookup paths explicit and separate. The continuous branch should continue to expose prototype context, and the discrete branch should continue to expose quantized hidden state, but the Stage A / Stage B semantics must remain clear in the output metadata.

2. In `src/models/thesis_multitask_loss_mixin.py`, keep the objective modular and stage-aware. The implementation should continue to compute:
   - reconstruction loss,
   - classification loss,
   - optional contrastive loss,
   - optional regularizers.

   The default path should stay compact, and optional losses should remain gated by configuration.

3. In `src/models/thesis_multitask_setup_mixin.py`, keep the fusion parameters and task-head construction consistent with the approved contract:
   - reconstruction and classification should each consume a fused task-specific representation,
   - the branch outputs should remain observable but not become additional default prediction heads.

4. In `src/data/augment.py`, preserve the RedLamp-based anomaly taxonomy and synthetic anomaly metadata. The injector should continue to emit classification labels and anomaly masks that can be consumed by the multitask model and the memory bootstrap logic.

### Interface and Contract Definitions

- Branch contract:
  - continuous branch returns `prototype_context`
  - discrete branch returns `quantized_hidden`

- Fusion contract:
  - reconstruction and classification each receive a fused representation
  - the fusion path must remain task-specific rather than branch-local

- Synthetic augmentation contract:
  - `classification_labels`
  - `synthetic_anomaly_mask`
  - `augmentation_metadata`

### Design Pattern Application

- Composition over inheritance: keep fusion and losses inside the model file through mixins instead of splitting them into separate task classes.
- Adapter pattern for encoders: the same hidden-state interface should feed both prototype branches without dataset-specific branching.
- Strategy pattern for tasks: reconstruction, classification, and optional regularizers should be selected by configuration and phase, not by separate model implementations.
- Registry/factory pattern: the synthetic injector remains one registered behavior attached to the multitask model configuration.

### Risk Mitigation

- Prototype redundancy is monitored by branch-level output logging and tests that inspect continuous-only, discrete-only, and fused behavior.
- Fusion collapse is mitigated by preserving metrics that compare branch contribution and gate behavior.
- Adaptation contamination is still out of scope, but synthetic augmentation must remain restricted to offline training and validation paths.
- Projector drift remains out of scope here.
- Evaluation metric inflation is mitigated by keeping synthetic and real validation paths distinct.

### Test Plan and Validation

- Run `pytest tests/test_thesis_multitask_classification_path_toggle.py -v`.
- Run `pytest tests/test_multitask_objective_controls.py -v`.
- Run `pytest tests/test_synthetic_anomaly_injection.py -v`.
- Run `pytest tests/test_one_multitask_train_step.py -v`.

### Acceptance Criteria

- The model still completes one offline training step with the current multitask configuration.
- The synthetic augmentation path still produces classification labels and anomaly masks with the expected shape.
- The fusion path remains task-specific and does not collapse into a single shared decoder/classifier path.

---

## Phase 4: Formalize the online adaptation stage with a residual projector

### Phase Summary

This phase is intentionally downstream of the offline slice. It should not be built until the offline model, memory banks, and fusion path are stable. The online stage will align a trainable online encoder to a frozen reference encoder through a residual projector, with warm-starting and gated updates to reduce drift and contamination.

### Files and Edits

- Modify: `src/models/online_adaptation.py`
- Modify: `src/engine/online_loop.py`
- Modify: `scripts/run_online_adaptation.py`
- Modify: `configs/model/online_adaptation.yaml`
- Modify: `configs/task/online_adaptation.yaml`
- Modify: `tests/test_online_adaptation_step.py`
- Modify: `tests/test_online_reference_checkpoint.py`
- Modify: `tests/test_online_state_roundtrip.py`

### Detailed Edit Content

1. In `src/models/online_adaptation.py`, keep the online adaptation model aligned to the same hidden-state contract used offline. The online model should:
   - load a frozen reference encoder,
   - expose a trainable online encoder,
   - insert a lightweight projector between the online encoder and the reference space,
   - keep the projector residual and near-identity by default.

2. In `src/engine/online_loop.py`, keep the adaptation loop conservative. The loop should only update the trainable subset selected for online adaptation and should preserve explicit gating so anomalous batches do not become uncontrolled update targets.

3. In `scripts/run_online_adaptation.py`, keep the runner separate from offline training and evaluation. The script should accept a reference checkpoint, an online config, and a clear update policy.

4. In `configs/model/online_adaptation.yaml` and `configs/task/online_adaptation.yaml`, define the projector, reference checkpoint, update schedule, and gating controls explicitly rather than inheriting hidden defaults.

### Interface and Contract Definitions

- Online batch contract:
  - offline batch fields plus `view_a` and `view_b`

- Online encoder contract:
  - the online encoder must produce the same hidden-state interface as the offline encoder

- Projector contract:
  - residual adapter
  - warm-start capable
  - low-magnitude initialization near identity

- Reference alignment contract:
  - frozen reference encoder
  - mapped online representation aligned to reference space

### Design Pattern Application

- Composition over inheritance: keep the online projector as a module attached to the online model, not as a separate training hierarchy.
- Adapter pattern for encoders: treat the projector as the adapter that reconciles online and reference spaces.
- Strategy pattern for tasks: allow online adaptation strategies to be selected by config, such as alignment-only or alignment-plus-anchor-regularization.
- Registry/factory pattern: keep the online model and task available through the existing config-driven builders.

### Risk Mitigation

- Prototype redundancy is not the main concern here, but the online model should still reuse the same output contract so branch semantics remain comparable.
- Fusion collapse is not a primary online issue, but frozen reference comparisons should remain available for diagnostics.
- Adaptation contamination is the main risk; mitigate it by gating updates away from clearly anomalous batches and by keeping the reference encoder frozen.
- Projector drift is the main architectural risk; mitigate it with residual initialization, warm-starting, and optional anchor regularization.
- Evaluation metric inflation is mitigated by evaluating adaptation on explicit validation splits and by keeping pre-adaptation and post-adaptation metrics separate.

### Test Plan and Validation

- Run `pytest tests/test_online_adaptation_step.py -v`.
- Run `pytest tests/test_online_reference_checkpoint.py -v`.
- Run `pytest tests/test_online_state_roundtrip.py -v`.

### Acceptance Criteria

- The online model can load a frozen reference checkpoint and step through one adaptation batch.
- The projector remains residual and near identity at initialization.
- The online state can be saved and restored without losing the reference/alignment contract.

---

## Phase 5: Validate the full pipeline through ablations, evaluation, and reporting

### Phase Summary

This phase makes the redesign measurable. The repository should be able to compare the minimal vertical slice against the prototype-enabled and adaptation-enabled variants, then report the results with explicit metric definitions, checkpoint paths, and stage labels. The goal is to keep evaluation interpretable rather than conflated across offline, prototype, and adaptation behavior.

### Files and Edits

- Modify: `src/engine/evaluator.py`
- Modify: `src/engine/thresholding.py`
- Modify: `src/metrics/pointwise.py`
- Modify: `src/metrics/classification_diagnostics.py`
- Modify: `scripts/evaluate.py`
- Modify: `scripts/run_ablation.py`
- Modify: `scripts/visualize_evaluation_results.py`
- Modify: `scripts/visualize_training_metrics.py`
- Modify: `tests/test_evaluation_protocol_audit.py`
- Modify: `tests/test_evaluator_thresholding.py`
- Modify: `tests/test_multitask_metrics_runtime.py`

### Detailed Edit Content

1. In `src/engine/evaluator.py` and `src/engine/thresholding.py`, keep the evaluation path explicit about:
   - pointwise scores,
   - window-level scores,
   - threshold selection,
   - synthetic versus non-synthetic validation outputs.

2. In `src/metrics/pointwise.py` and `src/metrics/classification_diagnostics.py`, preserve metric definitions so that gains from prototype branches or adaptation do not get mixed with changes in thresholding or evaluation protocol.

3. In `scripts/run_ablation.py`, keep ablation choices explicit and aligned with the redesign:
   - continuous-only,
   - discrete-only,
   - fused,
   - offline only,
   - offline plus online adaptation.

4. In `scripts/evaluate.py` and the visualization scripts, keep the reporting outputs organized by stage and experiment name so that the new design can be traced through artifacts, not only through scalar metrics.

### Interface and Contract Definitions

- Evaluation record contract:
  - `entity_id`
  - `point_scores`
  - `point_labels`
  - `num_points`

- Thresholding contract:
  - threshold selection must be explicit and reproducible

- Reporting contract:
  - stage labels must remain visible in metric keys and artifact paths

### Design Pattern Application

- Composition over inheritance: metrics and evaluation helpers should stay modular, not folded into the model classes.
- Adapter pattern for encoders: evaluation should consume the same output interface regardless of encoder family.
- Strategy pattern for tasks: ablation variants should be selected by configuration, not by separate ad hoc scripts.
- Registry/factory pattern: experiment construction should remain config-driven.

### Risk Mitigation

- Prototype redundancy is handled by ablation comparisons and explicit logging of branch usage.
- Fusion collapse is handled by comparing fused, continuous-only, and discrete-only results.
- Adaptation contamination is handled by separating offline validation from online adaptation validation.
- Projector drift is handled by reporting pre-update and post-update metrics separately.
- Evaluation metric inflation is handled by audit tests and explicit thresholding semantics.

### Test Plan and Validation

- Run `pytest tests/test_evaluation_protocol_audit.py -v`.
- Run `pytest tests/test_evaluator_thresholding.py -v`.
- Run `pytest tests/test_multitask_metrics_runtime.py -v`.
- Run the smallest available smoke experiment for the two-stage path and confirm the artifact set is written correctly.

### Acceptance Criteria

- Evaluation metrics remain reproducible and stage-aware.
- Ablation outputs distinguish continuous-only, discrete-only, fused, offline-only, and online-adapted behavior.
- Reporting artifacts retain the stage and experiment identity needed to trace the redesign.

---

## Execution Order and Dependency Notes

The order is strict. Phase 1 must finish before any bootstrap change is made, because the later phases assume a stable batch and output contract. Phase 2 can then replace memory initialization without forcing a redesign of the trainer or runner. Phase 3 confirms the fusion and synthetic supervision path that depends on the new memory banks. Phase 4 only starts after the offline path is stable, because the online projector relies on the same hidden-state contract. Phase 5 is the final verification layer and should not introduce new behavior.

This ordering preserves the minimal vertical slice principle and keeps each phase independently testable.

## Acceptance Summary

The plan is acceptable when all of the following are true:

- The batch, encoder, and model-output contracts remain stable.
- The two-stage runner still materializes Stage A, Stage B, and the Stage B initialization checkpoint.
- Memory bootstrap uses k-means and respects the approved source-pool split.
- The discrete runtime path is simplified enough that `cosine_topk` is the intended approved path.
- The online adaptation stage has a residual projector with explicit safeguards.
- The evaluation and ablation surfaces can distinguish baseline, prototype, and adaptation behavior without metric ambiguity.

