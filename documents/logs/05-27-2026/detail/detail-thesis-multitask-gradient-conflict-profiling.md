---
date: 2026-05-27 13:24:00 +0700 (+07)
author: Artificial Intelligence Agent
git_commit: 57aeba72e81071194e6e271faab39fbc1e955c89
branch: dev
repository: bachelor-thesis-2026
source_plan: documents/logs/05-27-2026/plan/plan-thesis-multitask-gradient-conflict-profiling.md
topic: "Detailed programming plan for REDLAMP-style CNN encoder and gradient-conflict profiling in thesis_multitask"
tags: [detail, multitask, redlamp, gradient-profiling, contracts, testing]
status: proposed
last_updated: 2026-05-27
last_updated_by: Artificial Intelligence Agent
---

# Detailed Plan: Thesis Multitask Gradient Conflict Profiling

## 1. Scope and Thesis Alignment
This detailed plan implements a minimal but complete vertical slice to test the hypothesis of gradient interference between classification and reconstruction objectives at the encoder level. The implementation preserves the current training/evaluation protocol, including checkpoint monitoring (`val_synth_vus_pr`) and evaluator thresholding (`quantile = 0.95`), to ensure experiment comparability.

The implementation follows repository constraints in `codebase_preferences.md`: readability-first, one model per file, explicit contracts, minimal codepaths, and configuration-driven ablations.

## 2. Fixed Contracts and Interfaces
Batch contract remains unchanged and enforced by `src/core/contracts.py::validate_batch`.

Model output contract remains unchanged and enforced by `validate_model_outputs`, with required fields unchanged: `hidden`, `pooled`, `recon`, `logits`, `point_scores`, `window_scores`, and `aux`.

Encoder contract is extended internally but stable externally:
- model boundary input: `x` with shape `[B, L, D]`.
- encoder output presented to the model pipeline: `hidden` with shape `[B, L, H]` for both `mlp` and `redlamp_cnn` encoder types.

Task and engine contract remains stable:
- `training_step/validation_step/synthetic_validation_step` signatures unchanged.
- trainer still drives optimization through standard `zero_grad -> backward -> step` path.

## 3. Design Pattern Mapping
Composition over inheritance is preserved in `thesis_multitask.py` by adding CNN components as composable modules without changing base model hierarchy.

Adapter pattern is applied to encoder selection via explicit `encoder_type` switch in model configuration while preserving the same encoder output contract.

Strategy pattern is applied to profiling scope via configuration (`all_encoder` versus `focus_only`) and smoothing policy (raw, EMA, SMA).

Registry/factory usage remains unchanged in script entrypoints (`scripts/train.py`, `scripts/evaluate.py`) and runtime registries.

## 4. Phase-by-Phase Implementation Detail

### Phase 1: REDLAMP-style CNN Encoder Integration in `thesis_multitask.py`
Phase objective: introduce REDLAMP-style feature extractor and projection bottleneck while preserving current multitask forward/loss contracts.

File-level edits:
- `src/models/thesis_multitask.py`

Edit content:
1. Add `ConvBlock1D` module:
   - `nn.Conv1d`, `nn.BatchNorm1d`, `nn.ReLU`, `nn.Dropout(p=0.2)`.
   - default kernel size `4`, stride `2`, padding consistent with current REDLAMP reference behavior.
2. Add `RedLampCnnEncoder` module:
   - four ConvBlocks with approved half-filter schedule `[64, 64, 128, 128]`.
   - `nn.MaxPool1d(kernel_size=2, stride=2)` after block stack.
   - final projection `nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)` as bottleneck/projection layer.
3. Add encoder selection in model initialization:
   - `encoder_type="mlp"` keeps current behavior.
   - `encoder_type="redlamp_cnn"` activates CNN path.
4. Convert tensor layout in CNN path explicitly:
   - from `[B, L, D]` to `[B, D, L]` before Conv1d.
   - back to `[B, L', H]` after projection.
5. Maintain existing downstream interface by reusing pooled representation logic and preserving key names.

Acceptance criteria:
- A forward pass with valid batch returns all expected output keys.
- `validate_model_outputs(...)` passes without modification to contract checker.
- Encoder selection can switch between MLP and CNN entirely by config.

### Phase 2: Gradient Profiling Mechanics in Trainer (Measurement-Only)
Phase objective: compute `g_CE`, `g_MSE`, cosine similarity, and preservation ratio per iteration without altering natural optimization trajectory.

File-level edits:
- `src/engine/trainer.py`
- `src/models/thesis_multitask.py` (small helper additions only)

Edit content:
1. Add model helper methods for deterministic encoder parameter enumeration:
   - e.g., `_iter_encoder_named_parameters_for_profiling()`.
   - expose mandatory focus parameter key for projection layer.
2. Add trainer-side profiling function sequence for training stage only:
   - run when `enable_gradient_conflict_profiling=true`.
   - compute weighted CE and weighted MSE gradients with separate backward passes.
   - copy/detach per-layer gradients before next pass.
3. Compute metrics per selected encoder parameter:
   - cosine similarity of `g_CE` and `g_MSE`.
   - preservation ratio `R = ||g_total|| / (||g_CE|| + ||g_MSE||)`.
4. Add optional analytical consistency assertion path:
   - verify `g_CE + g_MSE` approximately equals autograd gradient from weighted total loss under tolerance.
5. Preserve production optimization path:
   - after profiling, clear grads and run normal `total_loss.backward()` then `optimizer.step()`.

Explicit non-goal:
- no production use of partial `.grad` injection for only one layer.

Acceptance criteria:
- Profiling path logs gradients for all selected encoder parameters and mandatory projection layer.
- Standard training update order remains unchanged.
- One-step debug assertion confirms gradient-sum consistency within tolerance.

### Phase 3: Logging, Smoothing, and Focused Metric Stream
Phase objective: persist interpretable raw and smoothed conflict metrics per iteration for trend analysis.

File-level edits:
- `src/engine/trainer.py`
- `src/engine/logger.py`

Edit content:
1. Add raw metric logging keys for each profiled layer:
   - `grad_conflict/<layer>/cosine_raw`
   - `grad_conflict/<layer>/r_ratio_raw`
2. Add smoothing state and update logic:
   - EMA primary: `alpha=0.1`.
   - SMA stability check: `window=50`.
3. Log both smoothed traces:
   - `.../cosine_ema`, `.../cosine_sma`, `.../r_ratio_ema`, `.../r_ratio_sma`.
4. Route primary focus-layer metrics to focused metrics stream via logger’s existing focused channel.

Acceptance criteria:
- `metrics.jsonl` contains raw + EMA + SMA values each logged iteration.
- focused metrics file contains projection-layer primary diagnostics.
- W&B receives the same keys when enabled.

### Phase 4: Configuration Schema and Validation
Phase objective: make encoder/profiling behavior ablation-friendly and explicitly validated.

File-level edits:
- `src/core/config.py`
- `configs/model/*.yaml`
- `configs/experiment/*.yaml`

Edit content:
1. Extend model config schema:
   - `encoder_type`, `cnn_kernel_size`, `cnn_stride`, `cnn_dropout`, `cnn_filter_schedule`, `cnn_projection_dim`.
2. Extend logging/experiment schema:
   - `enable_gradient_conflict_profiling`
   - `gradient_profile_scope`
   - `gradient_profile_primary_focus_layer`
   - `gradient_profile_gamma`
   - `gradient_profile_log_every_n_steps`
   - `gradient_profile_ema_alpha`
   - `gradient_profile_sma_window`
3. Add validation checks:
   - enum checks and numeric range checks.
   - if profiling enabled, non-empty focus-layer string and valid scope.
4. Add one minimal experiment config variant enabling profiling while preserving:
   - `checkpoint_monitor_metric: val_synth_vus_pr`
   - existing evaluator thresholding behavior (`q=0.95`, unchanged code path).

Acceptance criteria:
- invalid profiling config fails fast with explicit error messages.
- valid profiling config resolves and trains without API changes to scripts.

### Phase 5: Notebook Companion and Verification Artifacts
Phase objective: provide small-cell, readable artifact for methodology transparency and rapid sanity-checking.

File-level edits:
- `notebooks/redlamp_gradient_conflict_demo.ipynb` (or update existing equivalent notebook in `notebooks/` only)

Edit content:
1. Cell: list encoder parameters and highlight projection layer.
2. Cell: run one profiling step and print raw metrics.
3. Cell: apply EMA and SMA updates and display side-by-side.
4. Cell: run gradient-sum consistency assertion.

Acceptance criteria:
- notebook executes end-to-end on one dummy batch.
- each cell is short and isolated, aligned with readability constraint.

## 5. Risk Mitigation Matrix (Execution-Time Controls)
Prototype redundancy risk is controlled by keeping profiling instrumentation orthogonal to model branches and not introducing parallel objective pathways beyond existing CE+MSE decomposition.

Fusion collapse risk is monitored by correlating gradient conflict metrics with existing fusion-related stage logs, without changing fusion logic in this slice.

Adaptation contamination risk is prevented by scope isolation: this work does not alter online adaptation modules and does not introduce adaptation-state updates.

Projector drift risk is bounded by preserving existing training update semantics and focusing diagnostics on encoder/projection gradients instead of modifying projector-style residual modules.

Evaluation metric inflation risk is mitigated by keeping evaluator threshold protocol unchanged (`q=0.95`) and by reporting conflict diagnostics as additional observability metrics, not as replacement primary metrics.

## 6. Test and Validation Plan
Unit tests:
- `tests/test_thesis_multitask_redlamp_cnn_shapes.py`: verifies shape and contract invariants for CNN encoder mode.
- `tests/test_gradient_profile_metrics_one_step.py`: verifies expected profiling metric keys for full encoder scope and focus layer.
- `tests/test_gradient_profile_sum_consistency.py`: verifies `g_total ≈ g_CE + g_MSE`.
- `tests/test_config_gradient_profile_validation.py`: verifies schema validation and failure modes.

Integration tests:
- one-batch train step with profiling enabled and disabled to ensure no regression in update path.
- one short smoke run (few epochs) to verify persisted metric streams and checkpoint behavior.

Validation checklist:
1. Model forward/backward pass succeeds in both `mlp` and `redlamp_cnn` encoder types.
2. Profiling metrics appear per iteration in `metrics.jsonl`.
3. EMA and SMA traces are both logged.
4. Focus-layer metrics are present in focused metrics stream.
5. Best checkpoint selection still uses `val_synth_vus_pr`.
6. Offline evaluator remains on current threshold protocol (`keep-current-thresholding`, `q=0.95`).

## 7. Programming Order (Low-Level Execution Sequence)
1. Implement encoder modules and encoder switch in `thesis_multitask.py`.
2. Add encoder parameter enumeration helper for profiling target selection.
3. Implement trainer profiling routine with strict autograd pass ordering.
4. Add smoothing state and logging integration.
5. Add config schema and config examples.
6. Add/adjust tests.
7. Update notebook demonstration cells.
8. Run smoke validation and archive outputs under experiment output directory.

## 8. Measurable Completion Criteria
The detailed plan is considered successfully executed when all statements below hold:
- The repository can run a multitask smoke experiment with `encoder_type=redlamp_cnn` and profiling enabled.
- Per-iteration raw/EMA/SMA conflict metrics exist for all encoder layers and the projection focus layer.
- Gradient sum consistency checks pass in debug test mode.
- No regression is observed in trainer API, script entrypoints, checkpoint selection metric, or evaluator thresholding behavior.
- All new tests pass with `pytest` for the added files.

