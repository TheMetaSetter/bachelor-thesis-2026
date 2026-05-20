---
date: 2026-05-20 21:00:53 +07
planner: Artificial Intelligence Agent
git_commit: 1e44b7b33a3b9813e62242d2bdbed004c7a0ef65
branch: dev
repository: bachelor-thesis-2026
topic: "Implementation plan for Experiment 2: offline pre-training phase two-view contrastive + CKA-gated per-sample fusion"
tags: [plan, implementation, time-series, anomaly-detection, multitask]
status: complete
last_updated: 2026-05-20
last_updated_by: Artificial Intelligence Agent
source_research_note: documents/logs/05-20-2026/research/research-exp2-current-codebase-status.md
---

# Plan: Experiment 2 Implementation (Offline Pre-Training Phase Two-View Contrastive + CKA-Gated Per-Sample Fusion)

**Date**: 2026-05-20 21:00:53 +07  
**Planner**: Artificial Intelligence Agent  
**Git Commit**: 1e44b7b33a3b9813e62242d2bdbed004c7a0ef65  
**Branch**: dev

## Plan Objective
This plan specifies implementation work for Experiment 2 while preserving the active repository contracts and engineering constraints. The target behavior is the SSOT design in `documents/design/offline_pretraining_phase_two_view_contrastive_design.md`, with strict consistency to `documents/design/idea.md` and `documents/design/design_starter.md`.

## Current State (Grounded)
1. The offline training pipeline is already complete and configuration-driven through `scripts/train.py`, registry builders, and `Trainer`.
2. The active multitask model owner is `src/models/thesis_multitask.py`, which currently contains encoder, prototype branches, fusion parameters, memory lifecycle, and stage-specific losses.
3. Synthetic anomaly augmentation and mask propagation already exist in `src/data/augment.py` and model-side batch preparation (`synthetic_anomaly_mask`, `classification_labels`).
4. Synthetic validation stage execution already exists in `Trainer` with deterministic reset support (`prepare_synthetic_validation_epoch`).
5. Exp1-ready no-bootstrap config already exists with `scheduler monitor` and `checkpoint monitor` set to `val_synth_vus_pr`.

## Selected Approach
The implementation should follow one integrated approach aligned to the current architecture and SSOT decisions:

- Keep one-model-one-file ownership in `src/models/thesis_multitask.py`.
- Implement two-view contrastive and CKA-gated fusion in the same model file rather than creating parallel model variants.
- Preserve existing batch/output contracts and trainer interfaces.
- Extend metrics/logging surfaces additively without breaking existing keys.

This approach best aligns with repository constraints, minimizes codepath divergence, and allows immediate ablation against Exp1.

## Implementation Plan

### A. Model interface and configuration extensions
1. Extend `ThesisMultitaskModelConfig` and grouped flat-key parsing to include new objective/schedule fields for Exp2:
   - `enable_two_view_contrastive` (bool)
   - `contrastive_temperature` (float, default 0.1)
   - `lambda_contrastive` (float, default 1.0)
   - `enable_cka_gated_fusion` (bool)
   - `cka_eps` (float, numerical stability)
2. Keep defaults backward-compatible so existing experiments still run unchanged when these flags are disabled.
3. Preserve contract compatibility for `build_model_from_experiment_config(...)` in train/evaluate scripts.

### B. Two-view forward path and routing constraints
1. In `src/models/thesis_multitask.py`, add a dedicated internal helper for two-view preparation in train and val_synth stages:
   - Build `(x, x')` and `M` with existing injector surface.
2. Add encoder computation for both views in Exp2-enabled stages:
   - `H = f_theta(x)`
   - `H' = f_theta(x')`
3. Enforce hard routing semantics in model internals:
   - `H` queries only continuous branch.
   - `H'` queries only discrete branch.
4. Keep existing single-view path as fallback when Exp2 flags are disabled.

### C. InfoNCE objective integration
1. Implement InfoNCE on non-injected anchor positions (`M=0`) with in-batch negatives.
2. Reuse existing mask and tensor shapes; do not alter external batch contract.
3. Compute and log stage-specific contrastive values:
   - `train_contrastive_loss`
   - `val_synth_contrastive_loss` (when synthetic validation is enabled)
4. Aggregate into total loss only when `enable_two_view_contrastive=true`:
   - `L_total = L_existing + lambda_contrastive * L_contrastive`

### D. Linear CKA and per-sample gating
1. Add linear CKA helper with time-axis centering exactly as SSOT formula specifies.
2. Compute two per-sample CKA scalars:
   - reconstruction-oriented CKA from `(H_b, Hc_hat_b)`
   - classification-oriented CKA from `(H'_b, Hd'_hat_b)`
3. Build per-sample feature `u_b = [s_rec, s_cls]` and two separate gate networks:
   - `MLP_cls` -> `alpha_b`
   - `MLP_rec` -> `beta_b`
4. Apply per-sample fusion for each head with broadcast over `(L, d_h)`:
   - `H_cls_b = alpha_b * Hd'_hat_b + (1-alpha_b) * Hc_hat_b`
   - `H_rec_b = beta_b * Hd'_hat_b + (1-beta_b) * Hc_hat_b`
5. Keep legacy fusion logits path available for backward compatibility when `enable_cka_gated_fusion=false`.

### E. Memory update policy (token-partitioned)
1. In train computational stage only, apply token-level partitioning:
   - Continuous memory update consumes only tokens with `M=0`.
   - Discrete memory update consumes only tokens with `M=1`.
2. Retain read-only memory behavior in validation and test stages.
3. Preserve existing memory initialization lifecycle and checkpoint state structures.

### F. Trainer and metric flow integration
1. Keep `Trainer` interface unchanged; rely on stage log dictionaries for new metrics.
2. Ensure scheduler/checkpoint monitor keys remain resolvable (`val_synth_vus_pr`).
3. Ensure synthetic validation stage (`val_synth`) computes and logs contrastive metrics when Exp2 is enabled.

### G. Configuration additions for Experiment 2
1. Create a dedicated Exp2 experiment config file under `configs/experiment/` with:
   - no-bootstrap override
   - Exp2 flags enabled
   - monitor settings consistent with protocol v2 (`val_synth_vus_pr` for scheduler and checkpoint)
2. Keep Exp1 config untouched as baseline reference.

## Contract Enforcement Plan
1. **Batch contract**: remain `x, point_labels, mask, timestamps, meta`; two-view artifacts are internal to model preparation path and do not break `validate_batch`.
2. **Encoder contract**: output remains hidden representation shape `[B, L, H]` (plus pooled/aux).
3. **Model output contract**: preserve required keys in `validate_model_outputs`; add new diagnostics only under `aux` and stage logs.

## Test Plan and Validation Procedures

### Unit-level tests
1. CKA helper test:
   - shape correctness
   - finite outputs
   - bounded behavior for identical vs random matrices.
2. InfoNCE helper test:
   - mask filtering (`M=0` anchors only)
   - non-empty negative set handling.
3. Gate networks test:
   - per-sample output shape `[B]`
   - output range `(0,1)` after sigmoid.

### Integration tests (model step)
1. `train_step` with Exp2 enabled returns:
   - valid output contract
   - `train_contrastive_loss` in logs
   - no NaN in total loss.
2. `validation_synthetic_step` with Exp2 enabled returns:
   - valid output contract
   - `val_synth_contrastive_loss` in logs.
3. Memory update partition test:
   - continuous updater only receives/uses `M=0` tokens.
   - discrete updater only receives/uses `M=1` tokens.

### End-to-end smoke path
1. One-epoch smoke training run with Exp2 config and small window caps.
2. Verify monitor keys exist in epoch metrics and checkpoint selection runs without missing-key errors.

## Risk and Mitigation (Implementation-facing)
1. **Risk**: branch-routing regressions break existing behavior.  
   **Mitigation**: keep feature flags and legacy path fallback.
2. **Risk**: empty token subset for memory partition on small batches.  
   **Mitigation**: implement guarded no-op update when subset is empty and log counters.
3. **Risk**: numeric instability in CKA denominator.  
   **Mitigation**: use `cka_eps` and explicit clamp.
4. **Risk**: stage metric key mismatch for scheduler/checkpoint.  
   **Mitigation**: assert presence of `val_synth_vus_pr` in Exp2 epoch logs during test.
5. **Risk**: synthetic validation path omits contrastive computation.  
   **Mitigation**: add explicit val_synth codepath checks and test coverage.

## Minimal Vertical Slice (Before Full Expansion)
1. Enable Exp2 flags and implement two-view + InfoNCE in train stage only.
2. Add CKA-gated per-sample fusion for head inputs in train stage.
3. Verify one-epoch smoke with valid metrics and checkpointing.
4. Extend to val_synth contrastive computation and memory partitioning.
5. Finalize tests and config matrix.

## Open Questions (Non-blocking for first coding pass)
1. Whether to expose `alpha_b` and `beta_b` distributions as explicit histogram artifacts in logger or retain scalar summaries only.
2. Whether to add optional gate regularization terms for CKA-driven gates in the first Exp2 implementation or defer to later ablation.
