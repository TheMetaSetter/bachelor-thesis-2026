---
date: 2026-09-12 20:20:39 +07:00
researcher: OpenAI Codex
topic: Core computational flows for a minimal standalone SMD codebase
status: complete-with-explicit-unknowns
revision: b8fff201cdfdb27aa9cec377d607487c4c2755cb
branch: dev
---

# Research: Minimal SMD runtime

## Summary

The computational core can be organized around data preparation, two-stage offline training, calibration, causal online processing, and evaluation.
The strongest simplification opportunity is replacing configuration generators, nested subprocesses, model mixins, and historical compatibility paths with direct calls.
Removing memory initialization, stochastic retrieval, verification, or method-specific adaptation would change the experiment rather than merely simplify its implementation.

This report inspects the current working tree, including existing uncommitted changes.
It follows `prompts/1_research_prompt.md` and extends the earlier SMD readiness research with computational evidence.
It is a source trace with focused tests, not a full training run or a complete repository audit.

## Execution paths

### 1. Data preparation

Implemented: `SMDDatasetParser.parse` reads one entity as `[T,D]`, splits the end of the training sequence into validation, and loads test labels separately (`src/data/datasets/smd.py:62`).
The loader cleans sequences, fits the scaler on training only, transforms each split, and creates `[B,L,D]` windows (`src/data/loaders.py:150`).
The active THESIS model config selects 38 channels, length 20, a CNN encoder, 32-dimensional tokens, and 12 classes (`configs/model/thesis_multitask_two_stage_window20.yaml:1`).
Synthetic training creates window classification labels and a point-level injection mask; these are different objects.

### 2. THESIS offline training

Implemented call chain: `run_thesis_offline_benchmark` → `execute_two_stage_plan` → training CLI → `Trainer.train` → `ThesisMultitaskModel.training_step`.
The orchestrator launches Stage A, initializes memory from its best checkpoint, launches Stage B, then evaluates Stage B best (`scripts/experiments/run_two_stage_offline_pretraining.py:372`).

Stage A encodes clean and injected views, computes reconstruction and classification losses, and adds two-view contrastive loss when active.
O1 additionally computes Balanced Point-Score Loss; the current classification branch combines classification and score loss with a factor of 0.5 when the score loss is available (`src/models/thesis_multitask_impl/thesis_multitask_loss_step_mixin.py:16`, `:218`).
Therefore “O1 adds a loss” alone is insufficient to reproduce its exact objective.
`Trainer.train` owns epoch iteration, backward, clipping, optimizer steps, validation, and checkpoint selection (`src/engine/trainer.py:607`).

Memory initialization reloads Stage A best and encodes training windows (`scripts/experiments/run_two_stage_offline_pretraining.py:258`).
The model constructs continuous prototypes, a class-stratified discrete codebook, and anomaly-verification metadata.
Current code collects normal positions from normal-class windows for continuous memory but all positions of each class window for discrete memory, with a fallback pool for absent classes (`src/models/thesis_multitask_impl/thesis_multitask_state_memory_mixin.py:234`, `:317`).
The offline ontology explicitly records a conflict with the earlier injected-token-only specification.

Stage B uses memory retrieval with a frozen encoder.
The current base config selects concatenation projections; direct routing already exists as another path.
Direct routing sends continuous context to reconstruction and discrete context to classification, including sampled representations (`src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:14`).
Its Stage B freezes legacy fusion modules (`src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py:157`).
The matrix requests direct routing for the new benchmark; it does not require preserving obsolete fusion parameters in new checkpoints.

Configured stochastic retrieval uses 10 Monte Carlo samples (`configs/model/thesis_multitask_two_stage_window20.yaml:25`).
The forward path constructs sampled memory contexts and prediction summaries (`src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py:131`).
The order of squared error and sample reduction matters: mean sample error is not generally error of mean reconstruction (`src/models/thesis_multitask.py:109`).

### 3. Calibration and offline evaluation

The benchmark reloads Stage B best, evaluates clean validation and test, reconstructs point timelines, and exports thresholds (`scripts/benchmarks/run_thesis_offline_benchmark.py:581`, `:841`, `:985`).
The evaluator maps window predictions back to absolute point positions and tracks coverage (`src/engine/evaluator.py:146`).

Configured protocol: `raw_input`, identity score transform, offline evaluation stride 20 with end-aligned tail, online stride 1, clean-validation q99 thresholds, and EWMA weights 0.9/0.1 (`configs/protocol/smd_window20_cleanval_q99_ewma09.yaml:1`).
Raw-input error restores original sensor units with the training scaler.
Historical sigmoid calibration is a separate executable branch, not the active raw-input protocol.
The matrix's generic stride 1 statement conflicts with the protocol's offline evaluation stride 20; these must be named by operation in the new contract.

The threshold builder calibrates online EWMA by replaying clean validation with the online scoring path (`scripts/benchmarks/run_thesis_offline_benchmark.py:841`).
Offline point threshold, online EWMA point threshold, input-window threshold, and latent thresholds are distinct values.
The new implementation must preserve their score spaces and must never estimate them from the test stream.

Correction to the earlier readiness report: `thresholds.json` does not itself prove a V4 mismatch.
Schema selection depends on the score protocol (`src/protocols/threshold_artifact.py:601`).
The unresolved issue is the matrix's historical V4 wording versus the active raw-input contract, not merely the filename.

### 4. THESIS online processing

Implemented entry: `run_thesis_online_benchmark` calls `run_thesis_online_tta_experiment` (`scripts/benchmarks/run_thesis_online_benchmark.py:284`).
Each event reads a causal window, scores it, updates EWMA by absolute point index, classifies its region, optionally adapts, and records the event (`src/engine/online_tta/online_engine_window_core.py:43`).
Prediction uses the scored event; adaptation affects subsequent scoring.
The EWMA map retains only current-window indices; points stop changing naturally when later windows no longer contain them (`src/engine/online_tta/point_ewma.py:8`).

Current A0 does inference only.
Current A1 updates on verified PNN reconstruction only.
Current A2 updates on accepted hard-old windows or verified PNN, adding online-to-source contrastive loss (`src/engine/online_tta/online_engine_step.py:119`).
The matrix's revised A1 means hard-old plus contrastive; A2 adds PNN.
This is an intentional algorithm-policy change for the proposed codebase, not a behavior-preserving rename.

Gray-zone windows enter `VerificationBuffer`; capacity and new-entry state trigger verification, followed by adapted-status and TTL updates (`src/engine/online_tta/verification_cycle.py:24`).
Verification consumes frozen source codewords, anomaly radii, and recurrent continuous-prototype signatures to select `pnn_mask` (`src/engine/online_tta/signature_verification.py:13`).
Hard-old admission uses a separate non-overlap guard.
Only the projector may train; accepted events use a fresh AdamW optimizer (`src/engine/online_tta/online_optimizer.py:24`).
Keeping an optimizer across events would change this algorithm.

### 5. Baselines and evaluation

Traditional offline runners fit a reference model, calibrate validation scores, score test data, and compute metrics (`scripts/benchmarks/run_offline_benchmark.py:285`).
Frozen online Stumpy, KMeansAD, and Isolation Forest retain their reference model (`src/baselines/online/frozen.py:83`, `:184`).
Their native scores and endpoint smoothing are not aliases of THESIS raw point-MSE/vector-EWMA scores.

CANDI and M2N2 load a matching RedLamp encoder and use different update rules.
CANDI selects hard/moderate pools and adapts their reconstruction loss (`src/baselines/online/candi.py:261`, `:307`).
M2N2 uses its own masked reconstruction update (`src/baselines/online/m2n2.py:111`).
The baseline loop records scores before adaptation (`src/baselines/online/adaptive.py:441`).
A shared scheduler and result format are appropriate; imposing the THESIS projector update on these methods is not.

`compute_pointwise_metrics` calls VUS, budgeted VUS, and affiliation metrics (`src/metrics/pointwise.py:727`).
Budgeted VUS traverses threshold and range-buffer values and reduces them at fixed FPR budgets (`src/metrics/pointwise.py:423`).
Metric definitions and boundary behavior require numerical tests; changing metric formulas to shorten code would invalidate comparison.

## Readiness implications

The current generator still excludes three entities, enumerates O0/O1, and creates `onl-` run identifiers (`scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:33`, `:184`).
A single manifest with explicit dependency paths can remove the need to reconstruct A0 paths from name prefixes.
The target remains 588 offline and 1,176 online logical units, totaling 1,764; target W&B stage runs total 2,352.
The matrix's later 504/924 counts describe the smaller O0/O1 expansion and must not govern the new plan.

## PDF evidence supplied during research

Source: `../T826_KL_KHMT06_BaoCao.pdf`, relative to the repository root.
The PDF has 96 pages; the relevant discussion is printed pages 27–31 and 48, corresponding to PDF pages 60–64 and 81.
Text was extracted and the complete formula, continuation, and variant pages 63, 64, and 81 were visually inspected.
Other chapters were searched for relevant terms, not audited in full.

Equation (3.25) takes a clean token as anchor and assigns the corresponding augmented token to the positive set when its injection mask is zero, or the negative set when its injection mask is one.
Equation (3.28) gives:

\[
\mathcal L_{\mathrm{con}}
=-\frac{1}{|Q|}\sum_{q\in Q}
\log\frac{\sum_{p\in P(q)}\exp(\cos(q,p)/\tau_{\mathrm{con}})}
{\sum_{p\in P(q)}\exp(\cos(q,p)/\tau_{\mathrm{con}})
+\sum_{n\in N(q)}\exp(\cos(q,n)/\tau_{\mathrm{con}})}.
\]

Equation (3.29) combines reconstruction, classification, and this contrastive loss with separate weights.
The formula is now sourced; it is no longer an unknown loss definition.
However, (3.25) does not completely enumerate cross-position/cross-window membership of the sets or the empty-positive case for an injected-position anchor.
Those executable details still need an explicit rule.

Implemented comparison: `_compute_two_view_contrastive_loss` first drops all injected positions, normalizes the remaining clean/augmented tokens, forms their similarity matrix, and uses diagonal cross-entropy targets (`src/models/thesis_multitask_impl/thesis_multitask_routing_mixin.py:307`).
Thus off-diagonal non-injected tokens act as negatives in current code; injected anomaly tokens never enter this loss.
This is not established as equivalent to equations (3.25)/(3.28).

The PDF's printed page 48 assigns Section 3.2 losses to O0 and adds Balanced Point-Score Loss for O1; it does not define O2.
The new matrix instead reserves its point-level contrastive component for O2 while retaining the existing two-view objective in its O0 flow.
Use the PDF as formula evidence, the matrix as the requested experiment policy, and record this semantic difference explicitly.
Whether the new O2 replaces the old contrastive term or adds a second term is not settled by naming alone.

The PDF's printed page 31 also calls for normal-point continuous prototypes, per-class point selection for discrete prototypes, and a frozen encoder and memories after initialization.
This strengthens the need to reconcile current all-window discrete token collection with the intended point-level selection.

## Verification and unknowns

Executed with `.venv/bin/python -m pytest -q`: `tests/models/test_direct_branch_routing.py`, `tests/online/test_full_spec_online_losses.py`, `tests/online/test_online_signature_verification.py`, and `tests/evaluation/test_budgeted_vus_metrics.py`.
Result: 22 passed in 3.31 seconds.
This confirms the selected tests, not full SMD readiness or complete mathematical equivalence.

The PDF establishes the proposed point-level loss formula, but O2's relationship to the existing loss, complete set construction, empty-set behavior, and selected numerical weight/temperature remain to be recorded.
The PDF's O0 name and the new matrix's O2 name must not be silently treated as aliases.
The raw-score threshold contract, split-specific strides, and memory-pool conflict require an explicit decision before dependent implementation.
No remote hardware, W&B connection, or complete training execution was tested.
