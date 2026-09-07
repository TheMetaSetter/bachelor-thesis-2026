---
date: 2026-09-06 Asia/Ho_Chi_Minh
planner: OpenAI Codex
topic: "Pro-reconstruction offline matrix with per-run VUS-PR FPR-budget checkpoint monitoring"
status: ready
revision: cc42b4e3faa433d9d81e33719dfab0cbbe1cee66
branch: dev
related_research: documents/logs/2026-09-06/research/research-pro-reconstruction-code-modification-surfaces.md
---

# Implementation Plan: Pro-reconstruction VUS-budget matrix

## Summary

The implementation will train the main method for 54 combinations and select each Stage A and Stage B `best.pt` with that run's one `VUS-PR@FPR-budget` metric.

The matrix is `O0/O1 × machine-1-6/machine-3-4/machine-3-9 × seed 6/8/36 × FPR budget 0.001/0.005/0.01`, so it produces 108 W&B training runs.

The implementation will evaluate RedLamp and the three traditional baselines under the same synthetic-validation q99 threshold rule, but baseline evaluation will calculate all three VUS budgets once per checkpoint or baseline configuration and will not retrain or multiply baseline runs by FPR budget.

## Request

- Use normalized-input reconstruction MSE for main-method and RedLamp training-score evaluation.
- Keep `lambda_recon: 0.75`, `lambda_cls: 0.25`, O0 without point-score loss, O1 with balanced point-score loss, direct branch routing in both stages, and 12 workers.
- Select point q99 from finite synthetic-validation normal points, including normal points inside anomalous windows.
- Select window q99 from finite synthetic-validation normal windows.
- Keep iForest negative decision score, KMeans-AD nearest-centroid distance, and StumPy AB-join distance as native baseline scores.
- Compute both VUS curves at all three FPR budgets.
- Monitor one `VUS-PR@FPR-budget` per main-method run, with the budget included in its run identity.
- Do not retrain RedLamp or traditional baselines, and do not include M2N2 or CANDI.

## Current state

`src/protocols/reconstruction_scores.py:53-63` already computes normalized-input MSE, while `src/engine/evaluator.py:588-722` uses raw-input MSE when a reconstruction score space is selected.

`src/engine/trainer.py:384-405` accepts only a fixed monitor-name list, and `src/engine/trainer.py:513-569` leaves budgeted VUS values nested rather than exposing one scalar monitor value.

`scripts/benchmarks/generate_full_direct_recon075_cls025_matrix.py:26-28` generates 18 main-method cells, and `scripts/benchmarks/run_full_direct_recon075_cls025_offline_matrix.sh:15-40` runs them sequentially.

`scripts/benchmarks/run_offline_benchmark.py:275-350` calibrates traditional baselines from clean validation and persists only a point threshold.

## Desired end state

Every main-method run has an immutable identity that includes one FPR budget, uses normalized-input MSE in synthetic validation and final offline evaluation, and saves the best checkpoint by its configured `VUS-PR@FPR-budget` scalar.

Every evaluated method writes point and window thresholds with explicit score definitions and synthetic-validation-normal provenance.

Evaluation outputs retain VUS-PR and VUS-ROC mappings for all three budgets without test-label use during calibration.

## Scope

### In scope

- Protocol validation, reconstruction evaluation, threshold selection, threshold artifacts, checkpoint monitoring, main-matrix generation, RedLamp checkpoint evaluation, traditional-baseline evaluation, focused tests, smoke validation, and remote rollout preparation.

### Out of scope

- New model architectures, hyperparameter tuning, online TTA behavior, M2N2, CANDI, edits to historical output trees, and launching the remote matrix during implementation.

## Implementation approach

The implementation will extend the existing protocol, evaluator, trainer, and benchmark runners rather than create a parallel experiment framework.

It will preserve raw-input and clean-validation protocols as existing historical behavior, while a new isolated normalized-input synthetic-normal protocol drives this experiment.

The trainer will publish flat scalar names for every available budget so the existing single-float checkpoint selection path can remain intact.

## Phase 1: Establish the normalized synthetic-normal protocol contract

### Goal

The codebase accepts and records a normalized-input MSE protocol with separate synthetic-normal point and window thresholds without changing historical raw-input artifacts.

### Changes

- Add the new protocol YAML and its validator rules.
- Add the shared synthetic-normal window threshold selector beside the existing point selector.
- Extend threshold-artifact validation and schema output so the artifact states normalized-input MSE, both offline thresholds, and both source splits.
- Add contract tests for accepted normalized protocol input and rejected malformed threshold inputs.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest -q tests/evaluation/test_thresholding_helpers.py tests/engine/test_threshold_artifact.py` — expected result: new protocol and artifact tests pass with historical tests unchanged.

### Risks

- A schema change could make old artifacts unreadable, so the new artifact version must be additive and old raw-input artifact validation must remain covered.

## Phase 2: Make normalized-input MSE operational for reconstruction evaluation

### Goal

The evaluator, THESIS offline runner, and RedLamp checkpoint evaluator use normalized-input point and window MSE for calibration, predictions, metrics, and provenance.

### Changes

- Allow `normalized_input` score space in the evaluator and select its MSE arrays as operational scores.
- Build synthetic-validation point and window thresholds from all scored windows before normal-label filtering.
- Apply the two thresholds independently to point and window predictions.
- Reuse the same protocol-driven flow in THESIS and RedLamp evaluation-only paths with isolated outputs.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest -q tests/evaluation/test_evaluator_thresholding.py tests/benchmarks/test_evaluator_trace_payload.py` — expected result: normalized MSE selects both prediction types and synthetic labels drive calibration.

### Risks

- A raw/normalized array mix would invalidate threshold meaning, so each output record and artifact must state the score-space identity.

## Phase 3: Monitor one budget per main-method training run and materialize 54 cells

### Goal

The trainer exposes a scalar VUS-PR monitor for each budget, and the generator and runner create isolated O0/O1/entity/seed/budget executions.

### Changes

- Flatten each synthetic-validation budgeted VUS-PR value into a stable scalar epoch metric name and allow it in checkpoint-monitor validation.
- Calculate the training synthetic-validation monitor from normalized-input reconstruction MSE rather than model-output score values.
- Generate three experiment YAMLs for every existing base combination, set one budget-specific checkpoint monitor per YAML, and include the budget in config name, W&B name, tags, and output path.
- Add preflight validation for all 54 generated YAMLs and retain sequential execution after a successful smoke run.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest -q tests/runtime/test_trainer_checkpoint_fallback.py tests/runtime/test_learning_rate_scheduler_trainer.py tests/benchmarks/test_offline_benchmark_config_generation.py tests/benchmarks/test_two_stage_orchestration_dry_run.py` — expected result: each allowed monitor is scalar, maximized, and present in both materialized stage configurations.

### Risks

- A missing budget in run identity could overwrite checkpoints or merge W&B history, so generation tests must enumerate exactly 54 unique names and 108 stage names.

## Phase 4: Evaluate baselines with native scores and the shared threshold contract

### Goal

RedLamp checkpoints and traditional baselines are evaluated fairly without changing their trained model or native score definitions.

### Changes

- Keep RedLamp checkpoint weights and model YAML unchanged while routing its evaluation through the normalized synthetic-normal protocol.
- Expose native point and window scores for iForest, KMeans-AD, and StumPy without replacing their existing historical calibrated-score API.
- Change only the new benchmark evaluation path to derive point and window q99 thresholds from synthetic validation, then calculate test metrics and all budgeted VUS mappings.
- Write isolated baseline outputs with explicit method-specific score definitions and unavailable-cell reporting.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest -q tests/models/test_traditional_baseline_contracts.py tests/benchmarks/test_run_offline_benchmark.py tests/models/test_redlamp_baseline_runtime.py` — expected result: fitting remains train-only, calibration uses synthetic normal values, and native score directions remain unchanged.

### Risks

- Reusing the old baseline output directories could overwrite report evidence, so all new evaluation outputs must use a new root.

## Phase 5: Verify the full flow and prepare controlled rollout

### Goal

A CPU or reduced smoke run proves the complete two-stage path before any remote 54-cell launch, and the rollout artifacts make every run auditable.

### Changes

- Run focused regression tests and configuration dry runs before executing a single smoke combination.
- Run one end-to-end main-method smoke combination for one explicit FPR budget and inspect Stage A, Stage B, checkpoint monitor, point/window thresholds, and VUS mappings.
- Run baseline evaluation smoke checks from existing checkpoints or lightweight test fixtures.
- Prepare the remote command only after rereading `ssh-gpu.txt`, verifying the remote revision, and choosing an isolated remote output root.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest -q tests/evaluation tests/engine/test_threshold_artifact.py tests/runtime/test_trainer_checkpoint_fallback.py tests/benchmarks/test_run_offline_benchmark.py` — expected result: focused protocol, monitor, and benchmark tests pass.

#### Manual

- [ ] Inspect the smoke manifest and W&B run metadata — expected result: the run identity has one budget, both stages use direct branch routing, and `best.pt` records the matching monitor value.

### Risks

- A full matrix is expensive and irreversible in consumed GPU time, so do not launch it until the one-combination smoke artifact passes the stated checks.

## Testing strategy

Unit tests cover selector inputs, protocol validation, schema identity, metric flattening, and native baseline score preservation.

Integration tests cover evaluator point/window prediction, THESIS two-stage monitor propagation, RedLamp checkpoint evaluation, and traditional-baseline artifact output.

The smoke run covers the actual Stage A → Stage-B initialization → Stage B → evaluation path for one budget-specific combination.

## Migration and rollback

The implementation will not migrate historical artifacts or outputs.

The new protocol and output root isolate the experiment, and rollback consists of not invoking the new generator or runner while old configurations and artifacts remain usable.

## Final verification

- [ ] Verify 54 generated main-method YAML files and 108 unique W&B stage names.
- [ ] Verify every selected `best.pt` records one VUS-PR budget monitor value.
- [ ] Verify all outputs contain VUS-PR and VUS-ROC keys for `0.001`, `0.005`, and `0.01`.
- [ ] Verify point and window thresholds use synthetic-validation-normal provenance and match their score definitions.
