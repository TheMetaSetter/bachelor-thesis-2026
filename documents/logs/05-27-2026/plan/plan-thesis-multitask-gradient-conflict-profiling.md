---
date: 2026-05-27 13:10:00 +0700 (+07)
planner: Artificial Intelligence Agent
git_commit: 57aeba72e81071194e6e271faab39fbc1e955c89
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed implementation plan for REDLAMP-style CNN encoder integration and gradient interference profiling in thesis_multitask"
tags: [plan, multitask, gradient-conflict, redlamp, encoder, profiling]
status: proposed
last_updated: 2026-05-27
last_updated_by: Artificial Intelligence Agent
---

# Implementation Plan: Thesis Multitask Gradient Conflict Profiling

## Current State
The current training path is `scripts/train.py -> Trainer.train(...) -> ThesisMultitaskModel.training_step(...) -> _shared_step(...)`, and optimization is performed centrally in `src/engine/trainer.py` via `loss.backward()` then `optimizer.step()`. The existing `ThesisMultitaskModel` in `src/models/thesis_multitask.py` uses an MLP encoder (`MultitaskWindowEncoder`) and computes a weighted multitask objective inside `_shared_step(...)`.

Batch contract and model output contract are already validated by `validate_batch(...)` and `validate_model_outputs(...)`, so the plan must preserve these interfaces. Current experiment monitoring and checkpoint policy use `val_synth_vus_pr` and evaluator thresholding with quantile `q=0.95`, which must remain unchanged.

## Design Options
Option A (selected): keep the existing optimization trajectory unchanged by preserving one standard backward pass for parameter updates (`L_total.backward()`), and run additional gradient-profiling passes only to measure `g_CE`, `g_MSE`, cosine similarity, and preservation ratio. This minimizes optimizer-side behavioral drift.

Option B: replace training update with manual gradient injection for all parameters from decomposed losses. This provides strict decomposition control but introduces high risk of training-path divergence and implementation complexity in a large self-contained model file.

Option C: profile only one focus layer with manual gradient injection for that layer. This is simple but creates partial-step inconsistency because non-profiled layers may receive stale or missing gradients in that step.

Rationale: Option A aligns with the approved design choice and keeps the scientific comparison fair by isolating profiling from optimization behavior.

## Risk and Mitigation
Risk: gradient profiling can unintentionally alter gradients if autograd graphs are reused incorrectly.
Mitigation: enforce strict sequence with dedicated `zero_grad` boundaries for profiling passes and a final clean `zero_grad -> L_total.backward -> step` for actual update.

Risk: high memory and runtime overhead from retaining computation graph and collecting per-layer vectors.
Mitigation: implement configurable layer scope (`all encoder` vs `focus layer only`) and lightweight scalar logging per iteration.

Risk: layer-name ambiguity in a long self-contained model file.
Mitigation: add explicit encoder-layer enumeration helper and deterministic parameter-name filtering; always include bottleneck/projection layer as mandatory focus output.

Risk: logging noise obscures trend interpretation.
Mitigation: log raw metrics plus EMA(alpha=0.1) and SMA(window=50), with EMA as the primary line and SMA as stability cross-check.

Risk: metric explosion in `metrics.jsonl` and W&B.
Mitigation: keep compact naming schema and separate focused diagnostic stream via existing focused-metrics support in logger.

## Open Questions
At this stage, previously ambiguous low-level choices have been resolved by explicit user decisions:
- profile scope default: full encoder + mandatory projection layer;
- logging cadence: per iteration (batch-level);
- smoothing: raw + EMA(alpha=0.1) + SMA(window=50);
- evaluator thresholding: keep current quantile (`q=0.95`).

Therefore, no blocking uncertainty remains before implementation.

## Detailed File-Level Plan

### 1) `src/models/thesis_multitask.py` (primary change)
Add a REDLAMP-style CNN encoder implementation and adapter-compatible switch while preserving external output contract.

Planned modifications:
- Add `ConvBlock1D` helper class (Conv1d -> BatchNorm1d -> ReLU -> Dropout(0.2)).
- Add `RedLampCnnEncoder` class with:
  - 4 convolution blocks,
  - default half-filter schedule per approved constraint (`[64, 64, 128, 128]`),
  - kernel size 4, stride 2, and REDLAMP-aligned max pooling,
  - projection bottleneck to embedding dim 128.
- Keep encoder output in thesis contract shape `hidden: Tensor[B, L, H]` by transposing back to `[B, L, H]` after CNN path.
- Extend encoder factory logic in `ThesisMultitaskModel` to choose `mlp` or `redlamp_cnn` via explicit config key (for ablation friendliness).
- Add method to expose encoder gradient target parameters deterministically, for example `_iter_encoder_named_parameters_for_profiling()`.

Contract enforcement:
- Input batch remains `x: [B, L, D]` at model boundary.
- Internal CNN path handles reshape/permutation with explicit checks and comments.
- `validate_model_outputs(...)` compatibility is preserved (`hidden`, `pooled`, `recon`, `logits`, `point_scores`, `window_scores`, `aux`).

### 2) `src/engine/trainer.py` (secondary change)
Integrate a non-intrusive profiling subroutine executed during train stage.

Planned modifications:
- Add optional `gradient_profiling_config` in trainer init.
- In train step branch, when enabled and model is `thesis_multitask`:
  - run profiling function before actual optimization step,
  - collect weighted gradients for `gamma * L_CE` and `(1-gamma) * L_MSE` over selected encoder parameters,
  - compute per-layer cosine similarity and preservation ratio,
  - log raw and smoothed metrics.
- Ensure actual update path remains:
  - `optimizer.zero_grad()`
  - `total_loss.backward()`
  - optional gradient clipping
  - `optimizer.step()`

Low-level profiling sequence (measurement-only path):
1. Forward pass to obtain `classification_loss`, `reconstruction_loss` tensors.
2. `zero_grad` -> backward on weighted CE (`retain_graph=True`) -> copy encoder grads.
3. `zero_grad` -> backward on weighted MSE -> copy encoder grads.
4. Compute:
   - `cos(g_ce, g_mse)`
   - `R = ||g_total|| / (||g_ce|| + ||g_mse||)`
5. Optional validation check:
   - compare `g_ce + g_mse` against gradients from backward on weighted total (assert/tolerance in debug mode).
6. `zero_grad` and execute the normal optimization backward/step.

Important implementation note:
- Do not inject `.grad` for only one layer then call global `optimizer.step()` in production path. That pattern is used only in isolated sanity checks and not in the final train trajectory.

### 3) `src/engine/logger.py` (small extension)
Use existing focused-metrics pipeline for gradient-conflict diagnostics.

Planned modifications:
- Add helper to flatten nested gradient-profile metric dict into scalar keys.
- Route high-priority keys (focus projection layer) to `focused_metrics.jsonl` and W&B focused log when enabled.

### 4) `src/core/config.py` (schema/validation changes)
Add and validate new config groups for encoder and profiling.

Planned config keys:
- model:
  - `encoder_type: "mlp" | "redlamp_cnn"`
  - `cnn_kernel_size: 4`
  - `cnn_stride: 2`
  - `cnn_dropout: 0.2`
  - `cnn_filter_schedule: [64, 64, 128, 128]`
  - `cnn_projection_dim: 128`
- task or experiment logging block:
  - `enable_gradient_conflict_profiling: bool`
  - `gradient_profile_primary_focus_layer: str` (default projection layer)
  - `gradient_profile_scope: "all_encoder" | "focus_only"`
  - `gradient_profile_gamma: float` (default 0.1)
  - `gradient_profile_log_every_n_steps: int` (default 1)
  - `gradient_profile_ema_alpha: float` (default 0.1)
  - `gradient_profile_sma_window: int` (default 50)

Validation rules:
- enforce allowed enum values,
- numeric range checks,
- if enabled, ensure focus layer key exists in encoder parameter names.

### 5) Config files under `configs/model/` and `configs/experiment/`
Add one minimal vertical-slice experiment config to activate profiling with current threshold policy unchanged.

Planned files:
- New model config variant enabling `encoder_type: redlamp_cnn`.
- New experiment config variant enabling gradient profiling logs, while keeping:
  - `checkpoint_monitor_metric: val_synth_vus_pr`
  - evaluator thresholding unchanged (`q=0.95`, current default in evaluator).

### 6) Notebook under `notebooks/` (implementation companion)
Create/update a compact notebook with small cells for:
- listing encoder layers available for profiling,
- one-batch profiling demo,
- raw vs EMA vs SMA visualization,
- assertion demo for gradient-sum consistency.

## Interface Contracts to Enforce
Batch contract (`src/core/contracts.py`): unchanged.

Encoder contract (internal):
- input to encoder adapter from model: `x` in `[B, L, D]`.
- encoder returns `hidden` in `[B, L, H]` for both MLP and CNN variants.

Model output contract: unchanged keys and semantics, preserving compatibility with trainer/evaluator/metrics pipeline.

## Engineering Principles Applied
Separation of concerns:
- model file handles architecture + loss decomposition signals,
- trainer handles step orchestration and profiling lifecycle,
- logger handles persistence and optional W&B routing.

Single responsibility:
- dedicated helper methods for gradient extraction, smoothing update, and metric assembly.

Stable interfaces:
- no change to script entrypoints or evaluator API.

Design patterns:
- composition over inheritance in model internals,
- adapter-style encoder switch (`mlp` vs `redlamp_cnn`),
- strategy-like profiling scope selection by config,
- existing registry/factory flow in `scripts/train.py` retained.

## Test Plan
Add minimal, focused `pytest` tests:
- `tests/test_thesis_multitask_redlamp_cnn_shapes.py`
  - verifies input `[B, L, D]` -> hidden/recon/logits shape invariants.
- `tests/test_gradient_profile_metrics_one_step.py`
  - one train step with profiling enabled,
  - checks cosine and R keys exist for all encoder layers + focus projection layer.
- `tests/test_gradient_profile_sum_consistency.py`
  - checks `g_total` approximately equals `g_ce + g_mse` under tolerance.
- `tests/test_config_gradient_profile_validation.py`
  - validates config parsing and rejection of invalid profiling parameters.

Runtime validation procedure:
1. Run one smoke experiment (few epochs) with profiling enabled.
2. Confirm `metrics.jsonl` contains raw, EMA, SMA keys.
3. Confirm focused metrics stream contains projection-layer primary diagnostics.
4. Confirm training still checkpoints on `val_synth_vus_pr` and evaluator behavior remains consistent with current threshold protocol.

## Minimal Vertical Slice Before Advanced Extensions
Phase 1 (must complete first):
- integrate encoder switch + profiling metrics for train stage only,
- verify one-batch and one-epoch smoke.

Phase 2:
- add full encoder-layer metric logging + focused metric stream.

Phase 3:
- add notebook visual diagnostics and compact plotting.

Out of scope for this slice:
- online adaptation changes,
- new evaluator thresholding strategy,
- multi-dataset expansion.

## Deliverables
- Updated `src/models/thesis_multitask.py` with REDLAMP-style CNN encoder option.
- Updated `src/engine/trainer.py` with measurement-only gradient profiling flow and stable optimization path.
- Updated config validation and new experiment/model config variants.
- New minimal tests for shape, profiling metrics, and gradient sum consistency.
- Notebook in `notebooks/` with small-cell demonstration for interpretability.
