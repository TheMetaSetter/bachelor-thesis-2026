---
date: 2026-05-20 21:07:10 +07
author: Artificial Intelligence Agent
git_commit: 1e44b7b33a3b9813e62242d2bdbed004c7a0ef65
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed implementation plan for Experiment 2: offline pre-training phase two-view contrastive + CKA-gated per-sample fusion"
tags: [detail-plan, implementation, time-series, anomaly-detection, multitask]
status: complete
last_updated: 2026-05-20
last_updated_by: Artificial Intelligence Agent
source_plan: documents/logs/05-20-2026/plan/plan-exp2-offline-pretraining-two-view-contrastive-cka-gated-fusion.md
source_research: documents/logs/05-20-2026/research/research-exp2-current-codebase-status.md
---

# Detailed Plan: Experiment 2 Implementation

## Objective
The objective is to implement Experiment 2 in the offline pre-training phase using a two-view contrastive objective and CKA-gated per-sample fusion while preserving the current repository contracts, design constraints, and runtime surfaces. The implementation shall remain consistent with `documents/design/offline_pretraining_phase_two_view_contrastive_design.md` and maintain backwards compatibility for existing experiments.

## Phase 1 — Configuration and Contract Surface Preparation

### Phase summary
This phase creates explicit, configuration-driven switches for Experiment 2 without altering default behavior for existing experiments. It secures stable interfaces before model behavior is extended.

### File-level edits
1. `src/models/thesis_multitask.py`
- Extend `ObjectiveConfig` and parsing keys to include:
  - `enable_two_view_contrastive: bool = False`
  - `contrastive_temperature: float = 0.1`
  - `lambda_contrastive: float = 1.0`
  - `enable_cka_gated_fusion: bool = False`
  - `cka_eps: float = 1.0e-6`
- Store these fields in `_store_config_values(...)`.

2. `configs/model/thesis_multitask_redlamp_multiclass.yaml`
- Add explicit fields for Experiment 2 flags and values with defaults disabled.

3. `configs/experiment/`
- Add one dedicated Exp2 config file (for example, `smd_thesis_multitask_redlamp_multiclass_window20_exp2.yaml`) that enables Experiment 2 options and keeps:
  - `bootstrap_encoder_epochs: 0`
  - `optimizer.scheduler.monitor_metric: val_synth_vus_pr`
  - `checkpoint_monitor_metric: val_synth_vus_pr`

### Interface and contract definitions
- **Dataset contract** remains unchanged and is supplied by `SMDDatasetBuilder` and `WindowDataset`.
- **Batch contract** remains `x`, `point_labels`, `mask`, `timestamps`, `meta` validated by `validate_batch`.
- **Encoder contract** remains hidden output `[B, L, H]`.
- **Model output contract** remains unchanged at the top-level keys; new diagnostics are placed only in `aux` and `log` fields.

### Design pattern application
- **Composition over inheritance**: new behaviors are integrated as optional internal helpers inside `ThesisMultitaskModel`.
- **Registry/factory pattern**: no change to registration API in train/evaluate scripts.

### Acceptance criteria
1. Existing configs (including Exp1) parse and run without requiring new keys.
2. Exp2 config parses with no unknown-key errors.
3. No contract validator (`validate_batch`, `validate_model_outputs`) breaks due to this phase.

## Phase 2 — Two-View Path and InfoNCE Integration

### Phase summary
This phase introduces two-view computation and InfoNCE into offline pre-training stages while preserving existing stage method signatures.

### File-level edits
1. `src/models/thesis_multitask.py`
- Add helper to build two-view tensors from existing synthetic injector output:
  - `x` (normal view), `x'` (injected view), `synthetic_anomaly_mask M`.
- Add helper to compute InfoNCE from:
  - anchors at `M=0`
  - in-batch negatives over all tokens except positive pair.
- Integrate contrastive loss in:
  - `training_step(...)`
  - `validation_synthetic_step(...)`
- Add stage logs:
  - `train_contrastive_loss`
  - `val_synth_contrastive_loss`

### Explicit edit content
- Add conditional branch:
  - if `enable_two_view_contrastive` is `False`, current behavior is preserved.
  - if `True`, compute two-view tensors and contrastive term.
- Add total loss assembly:
  - `loss_total = existing_loss + lambda_contrastive * contrastive_loss`.

### Interface and contract definitions
- No public engine interface changes are allowed.
- Step outputs must keep existing schema and append contrastive values in `log` only.

### Design pattern application
- **Strategy pattern (task behavior)**: stage-specific logic remains in model step methods and is activated by config flags.

### Risk mitigation
- **Adaptation contamination risk equivalent in offline synthetic path**: keep anchor selection strictly `M=0` and use deterministic synthetic validation reset.
- **Evaluation inflation risk**: do not replace existing detection metrics with contrastive metrics; additively log only.

### Acceptance criteria
1. Train step with Exp2 enabled returns finite `loss` and includes `train_contrastive_loss`.
2. Val_synth step with Exp2 enabled includes `val_synth_contrastive_loss`.
3. Train/val_synth without Exp2 flags remain behaviorally unchanged.

## Phase 3 — CKA-Gated Per-Sample Fusion

### Phase summary
This phase replaces fixed scalar fusion behavior (for Exp2-enabled path) by per-sample gates derived from linear CKA features, using two separate gating MLPs.

### File-level edits
1. `src/models/thesis_multitask.py`
- Add linear CKA helper with time-axis centering matrix:
  - `J = I_L - (1/L) 11^T`
- Add helper to compute per-sample CKA features:
  - `s_rec_b = CKA(H_b, Hc_hat_b)`
  - `s_cls_b = CKA(H'_b, Hd'_hat_b)`
- Add two gate MLP modules:
  - `MLP_cls: R^2 -> R`
  - `MLP_rec: R^2 -> R`
- Add sigmoid outputs:
  - `alpha_b` and `beta_b`.
- Add per-sample fusion implementation with broadcast over `[L, H]`.

### Explicit edit content
- Enforce route constraints in Exp2 path:
  - `H` queries continuous branch only.
  - `H'` queries discrete branch only.
- Produce head-specific fused tensors:
  - `H_cls_b = alpha_b * Hd'_hat_b + (1-alpha_b) * Hc_hat_b`
  - `H_rec_b = beta_b * Hd'_hat_b + (1-beta_b) * Hc_hat_b`

### Interface and contract definitions
- Head input tensors maintain expected shapes used by reconstruction and classification heads.
- `alpha_b` and `beta_b` shall be logged as scalar means and optional dispersion statistics in `aux/log`.

### Design pattern application
- **Composition**: CKA feature extractor and gate MLPs are independent submodules composed into model forward.
- **Single responsibility**: CKA helper computes similarity only; fusion helper computes fusion only.

### Risk mitigation
- **Fusion collapse risk**: log `alpha_b` and `beta_b` distributions by stage and epoch.
- **Prototype redundancy risk**: preserve existing optional diversity/usage diagnostics and compare against gate statistics.

### Acceptance criteria
1. Exp2 path produces `alpha_b` and `beta_b` with shape `[B]` and values in `(0,1)`.
2. Head forward passes succeed with fused per-sample tensors and no shape errors.
3. Existing non-Exp2 path still uses legacy fusion behavior unchanged.

## Phase 4 — Token-Partitioned Memory Updates

### Phase summary
This phase introduces token-level memory update partitioning aligned with Exp2 semantics while preserving train-only write behavior and eval read-only behavior.

### File-level edits
1. `src/models/thesis_multitask.py`
- Extend memory update helpers to accept optional token masks/subsets.
- In Exp2 train stage:
  - continuous update consumes only `M=0` tokens.
  - discrete update consumes only `M=1` tokens.
- Add guarded no-op paths when subset is empty.

### Interface and contract definitions
- Memory checkpoint state keys remain unchanged.
- `load_checkpoint_extra_state` and existing lifecycle flags remain backward-compatible.

### Design pattern application
- **Strategy within model stage**: partition policy is conditional by Exp2 flags.

### Risk mitigation
- **High-variance update risk from small subsets**: guarded no-op + logging counters for empty partitions.
- **Projector drift risk (online-specific)**: not modified in this offline phase; document as unaffected boundary.

### Acceptance criteria
1. Train stage updates memory without exceptions when one subset is empty.
2. Val/test stages perform read-only memory behavior exactly as before.
3. Memory lifecycle tests remain valid or are updated with explicit Exp2 partition assertions.

## Phase 5 — Trainer Integration, Metrics, and Protocol Alignment

### Phase summary
This phase ensures engine-level compatibility and makes Experiment Protocol v2 operational through configuration and logging.

### File-level edits
1. `src/engine/trainer.py`
- No interface change; only ensure new log keys are aggregated and persisted.
- Confirm scheduler/checkpoint monitor keys (`val_synth_vus_pr`) continue to exist in epoch metrics when Exp2 is enabled.

2. `configs/experiment/smd_thesis_multitask_redlamp_multiclass_window20_exp2.yaml`
- Enable Exp2 options and protocol defaults.

3. `documents/design/offline_pretraining_phase_two_view_contrastive_design.md`
- If code-level names differ from current SSOT wording, align nomenclature exactly to implemented symbols.

### Interface and contract definitions
- Trainer constructor and loop signatures remain unchanged.
- Evaluation entrypoint remains unchanged and continues to evaluate test loader from checkpoint.

### Design pattern application
- **Stable engine interfaces** are preserved; model behavior variations are controlled through configuration strategy.

### Risk mitigation
- **Evaluation metric inflation risk**: maintain existing test evaluation metric pipeline and thresholding path unchanged.

### Acceptance criteria
1. Exp2 configuration executes one complete epoch with synthetic validation stage enabled.
2. Monitor and checkpoint selection run without missing-key failures.
3. Evaluation script still loads and evaluates checkpoints from Exp2 run output.

## Phase 6 — Test and Validation Matrix

### Phase summary
This phase formalizes unit and integration verification required for implementation readiness.

### Unit tests
1. Add CKA helper tests in `tests/`:
- shape/finite checks
- identical-input sanity (high similarity)
- randomized-input sanity (bounded finite similarity).

2. Add InfoNCE helper tests:
- anchors restricted to `M=0`
- valid negative pool construction
- finite loss for normal batch sizes.

3. Add gate tests:
- input `u_b` shape `[B,2]`
- outputs `alpha_b`, `beta_b` shape `[B]`
- value range checks after sigmoid.

### Integration tests
1. Extend multitask one-step tests (`tests/test_one_multitask_train_step.py` or equivalent):
- Exp2 flags enabled
- returns valid output contract
- logs include contrastive and gate summaries.

2. Add memory partition integration test:
- verify token routing by mask into continuous/discrete memory update helpers.

3. Add synthetic validation integration test:
- verify `val_synth_contrastive_loss` exists when Exp2 enabled.

### Acceptance criteria
1. New tests pass in local CI command set used by repository.
2. Existing tests for baseline and non-Exp2 multitask modes remain passing.
3. No regression in config loading tests and scheduler/checkpoint monitor tests.

## Measurable Completion Criteria
1. Code compiles and runs for Exp2 config with no bootstrap.
2. One-epoch smoke run completes train + val + val_synth with finite metrics.
3. Checkpoint saving and loading operate with unchanged interfaces.
4. Test evaluation from `scripts/evaluate.py` completes on produced checkpoint.
5. SSOT design text and implementation symbols are synchronized.

