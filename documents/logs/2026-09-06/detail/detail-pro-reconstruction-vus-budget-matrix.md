---
date: 2026-09-06 Asia/Ho_Chi_Minh
topic: "Pro-reconstruction offline matrix with per-run VUS-PR FPR-budget checkpoint monitoring"
status: implemented_locally_smoke_preflight_complete_gpu_matrix_deferred
revision: cc42b4e3faa433d9d81e33719dfab0cbbe1cee66
source_structure: documents/logs/2026-09-06/structure/structure-pro-reconstruction-vus-budget-matrix.md
related_documents:
  - documents/notes/pro-reconstruction-offline-fairness-decisions.md
  - documents/logs/2026-09-06/research/research-pro-reconstruction-code-modification-surfaces.md
  - documents/logs/2026-09-06/plan/plan-pro-reconstruction-vus-budget-matrix.md
---

# Detailed Implementation: Pro-reconstruction VUS-budget matrix

## Summary

This document expands the approved structure into atomic implementation steps for 54 budget-specific main-method combinations, synthetic-normal dual thresholds, normalized-input MSE reconstruction scoring, and fair baseline evaluation.

## Source structure

The work follows five dependency-ordered phases: protocol contract, reconstruction evaluation, training monitor and matrix, baseline evaluation, then smoke and remote rollout preparation.

## Current state

`score_reconstruction` returns normalized MSE at `src/protocols/reconstruction_scores.py:53-63`, but `Evaluator.evaluate` allows only `model_output` and `raw_input` at `src/engine/evaluator.py:537-610`.

The trainer saves a checkpoint only from one scalar metric at `src/engine/trainer.py:896-919`.

The existing matrix generator emits 18 configurations at `scripts/benchmarks/generate_full_direct_recon075_cls025_matrix.py:113-126`.

The traditional runner calibrates from clean validation at `scripts/benchmarks/run_offline_benchmark.py:275-315`.

## Desired end state

The main method executes three isolated runs for every O0/O1/entity/seed base cell, and each run uses its own VUS-PR FPR budget to choose Stage A and Stage B checkpoints.

All reconstruction and RedLamp offline scores use normalized-input MSE, while traditional baselines retain their native method-specific score.

All methods derive point and window q99 thresholds from synthetic-validation normal values and report VUS-PR and VUS-ROC at every locked budget.

## Scope

### In scope

- `normalized_input` protocol support, synthetic-normal threshold functions, evaluator behavior, trainer monitor metrics, 54-cell matrix generation, RedLamp evaluation, traditional baselines, threshold artifacts, and tests.

### Out of scope

- Historical protocol rewrites, architecture or hyperparameter changes, baseline retraining, online paths, M2N2, CANDI, and remote execution.

## Phase 1: Establish normalized synthetic-normal protocol contract

### Goal

The new experiment has a valid and auditable protocol before any runner consumes it.

### Dependencies

- The agreed score space is normalized-input MSE.
- The agreed calibration sources are synthetic-normal point and window values.

### Detailed changes

#### 1. Add an isolated protocol YAML

- **File:** Proposed new file `configs/protocol/smd_window20_synthnormal_q99_normalized_input_mse_ewma09.yaml`.
- **Current responsibility:** `configs/protocol/smd_window20_synthnormal_q99_ewma09.yaml` declares raw-input MSE and a clean-validation window threshold.
- **Change:** Declare `score_space: normalized_input`, `point_score_transform: identity`, `offline_point_threshold_source_split: synthetic_validation_normal`, `offline_window_threshold_source_split: synthetic_validation_normal`, and q99 values for both levels.
- **Reason:** A separate config preserves historical raw-input runs and makes the new score contract visible in one place.
- **Inputs:** The existing window, stride, test-label, and online fields remain at their current values unless the new offline contract explicitly replaces them.
- **Outputs:** The file is the protocol argument for all new main-method, RedLamp, and traditional-baseline evaluation runs.
- **Errors:** The validator must reject a missing score space, mismatched threshold sources, or a non-q99 quantile.
- **Compatibility:** Existing raw protocol YAML files remain unchanged.

#### 2. Extend protocol validation

- **File:** `src/protocols/smd_benchmark_protocol.py`.
- **Symbol:** `OFFLINE_POINT_THRESHOLD_SOURCES` and `validate_protocol_config`.
- **Current responsibility:** The validator forces `score_space == raw_input` and `offline_threshold_split == clean_validation`.
- **Change:** Accept the existing raw protocol unchanged and add the normalized-input synthetic-normal contract, including the new window-source key.
- **Reason:** This is the existing entry gate used before benchmark execution.
- **Inputs:** `score_space` is `raw_input` or `normalized_input`; each offline source is `clean_validation` or `synthetic_validation_normal` only when its score-space contract allows it.
- **Outputs:** Valid protocol config or a precise `ValueError` naming the invalid key.
- **Errors:** Reject test as a calibration split and reject point/window source disagreement in the new normalized protocol.
- **Compatibility:** The existing `require_score_identity=False` baseline call stays usable.

#### 3. Add synthetic-normal window threshold selection

- **File:** `src/engine/thresholding.py`.
- **Symbol:** Add a window selector adjacent to `select_synthetic_validation_normal_point_threshold`.
- **Current responsibility:** The point selector filters finite scores with point label `0` and returns a requested quantile.
- **Change:** Add a selector that takes one-dimensional window scores, one-dimensional window labels, and a quantile; retain only finite scores whose window label is `0`; then return q99.
- **Reason:** The point selector cannot implement the required normal-window calibration rule because its labels have another granularity.
- **Inputs:** Equal-length score and label arrays and a quantile in `(0, 1]`.
- **Outputs:** One finite `float` threshold.
- **Errors:** Raise `ValueError` for unequal lengths or no finite normal windows.
- **Compatibility:** Keep the existing point selector behavior unchanged.

#### 4. Extend threshold artifact identity

- **File:** `src/protocols/threshold_artifact.py`.
- **Symbol:** `build_threshold_artifact` and artifact validation helpers.
- **Current responsibility:** The builder recognizes raw-input identity and persists an offline point threshold plus optional `input_window` threshold.
- **Change:** Add a new additive artifact version for normalized-input identity, an explicit `offline_window` threshold, point/window source splits, and exact score definitions `normalized_input_point_mse` and `normalized_input_window_mse`.
- **Reason:** The artifact must let a reader distinguish normalized reconstruction MSE from historical raw MSE and native baseline scores.
- **Inputs:** Point threshold, window threshold, both source splits, q99, score space, and score definitions.
- **Outputs:** A versioned artifact whose two offline thresholds state their values, sources, quantiles, and score rules.
- **Errors:** Reject unknown score space, missing threshold values, or a threshold source unsupported by the selected artifact version.
- **Compatibility:** Continue reading and validating existing raw artifact versions without migration.

### Tests

#### Protocol and threshold contract

- **Location:** `tests/evaluation/test_thresholding_helpers.py` and `tests/engine/test_threshold_artifact.py`.
- **Level:** Unit and schema-contract tests.
- **Setup:** Use finite and non-finite score arrays with mixed normal/anomalous point and window labels.
- **Action:** Select q99 thresholds and build both normalized and historical raw artifacts.
- **Expected result:** Point selection includes normal points inside anomalous windows; window selection includes only normal windows; normalized artifacts state both MSE definitions; raw artifacts retain their old schema meaning.
- **Edge cases:** Empty normal support, mismatched lengths, and unsupported score-space values raise informative errors.

### Atomic steps

- [ ] Read the existing synthetic-normal protocol YAML.
- [ ] Create the new normalized-input protocol YAML with every required existing protocol key.
- [ ] Add the window threshold source key to protocol validation.
- [ ] Add the window synthetic-normal selector signature.
- [ ] Add finite normal-window filtering.
- [ ] Add the q99 calculation for filtered window scores.
- [ ] Add failing tests for normal-window filtering and invalid inputs.
- [ ] Run the threshold-helper tests and confirm failure before implementation completion.
- [ ] Add normalized-input artifact fields and a new additive schema version.
- [ ] Add artifact tests for new fields and historical raw compatibility.
- [ ] Run `.venv/bin/python -m pytest -q tests/evaluation/test_thresholding_helpers.py tests/engine/test_threshold_artifact.py`.

### Complete when

The new protocol loads, both selectors produce q99 from their correct synthetic-normal populations, and an artifact records both normalized MSE threshold levels.

## Phase 2: Make normalized MSE and dual thresholds operational

### Goal

The evaluator and reconstruction checkpoint paths use the new protocol without raw-MSE leakage.

### Dependencies

- Phase 1 protocol validation and threshold selectors.

### Detailed changes

#### 1. Select normalized MSE in the evaluator

- **File:** `src/engine/evaluator.py`.
- **Symbol:** `_build_window_score_records` and `Evaluator.evaluate`.
- **Current responsibility:** The evaluator computes both MSE spaces but selects raw MSE for operational point scores, predictions, and window predictions.
- **Change:** Accept `normalized_input` score space; set operational point scores to `normalized_input_point_mse`; set operational window predictions to `normalized_input_window_mse`; retain both arrays only as diagnostics when needed.
- **Reason:** The evaluator owns the score arrays passed to metrics and predictions.
- **Inputs:** Reconstruction samples, scaled input, fitted scaler where the existing helper needs it, selected score space, point threshold, and window threshold.
- **Outputs:** Point records, window records, metrics, and trace payloads whose operational score identity matches the selected score space.
- **Errors:** Reject absent scaler when the reconstruction scorer needs one and reject an unsupported score space.
- **Compatibility:** Preserve `model_output` and `raw_input` behavior exactly.

#### 2. Calibrate both THESIS thresholds from synthetic validation

- **File:** `scripts/benchmarks/run_thesis_offline_benchmark.py`.
- **Symbol:** `_evaluate_offline_benchmark_splits`, `_evaluate_named_split`, `_evaluation_outputs_to_score_payload`, and `_build_thresholds`.
- **Current responsibility:** The raw path derives a synthetic point threshold only, while `clean_window_threshold` always uses clean validation raw MSE.
- **Change:** Evaluate the full `val_synth` split once without thresholds; extract normalized point scores, point labels, normalized window scores, and window labels; select both synthetic-normal q99 thresholds; then evaluate synthetic and test splits with those two thresholds.
- **Reason:** This runner owns two-stage evaluation and final threshold artifacts.
- **Inputs:** New protocol fields, stage-B checkpoint scaler state, synthetic-validation outputs, and test outputs.
- **Outputs:** Separate `offline_point_threshold`, `offline_window_threshold`, two source fields, normalized score payloads, predictions, and artifact metadata.
- **Errors:** Raise a clear error when the loader lacks `val_synth` rather than silently falling back to clean validation for the new protocol.
- **Compatibility:** Retain the old fallback and raw behavior only for the historical protocol path.

#### 3. Calibrate RedLamp checkpoints through the same protocol

- **File:** `scripts/cli/evaluate.py`.
- **Symbol:** Replace the raw-only checkpoint helper with protocol-driven reconstruction checkpoint evaluation and extend CLI arguments only as necessary for protocol and output override selection.
- **Current responsibility:** `evaluate_raw_checkpoint` calibrates both thresholds from clean validation raw MSE, and the generic path evaluates test directly.
- **Change:** For normalized reconstruction evaluation, score `val_synth`, select two synthetic-normal q99 thresholds, evaluate test, and write output to an explicitly supplied isolated directory.
- **Reason:** Existing RedLamp configurations and checkpoints can be reused without retraining.
- **Inputs:** Existing RedLamp experiment YAML, checkpoint path, new protocol config path, and isolated output directory.
- **Outputs:** Evaluation records, metrics, curves, and threshold provenance under the new output root.
- **Errors:** Reject a normalized protocol when `val_synth` data is absent or a checkpoint scaler cannot rebuild the dataset.
- **Compatibility:** Existing CLI default behavior remains unchanged when no new protocol argument is selected.

### Tests

#### Evaluator score-space contract

- **Location:** `tests/evaluation/test_evaluator_thresholding.py` and `tests/benchmarks/test_evaluator_trace_payload.py`.
- **Level:** Integration tests.
- **Setup:** Use a deterministic reconstruction batch whose raw and normalized MSE values differ.
- **Action:** Evaluate in `normalized_input` mode with supplied thresholds.
- **Expected result:** Metrics and point predictions use normalized scores, window predictions use normalized window scores, and records retain score-space identity.
- **Edge cases:** Test a window label with at least one anomalous point and a normal point inside that same window.

#### THESIS and RedLamp calibration path

- **Location:** Extend existing benchmark and RedLamp runtime tests under `tests/benchmarks/` and `tests/models/`.
- **Level:** Integration tests.
- **Setup:** Supply a synthetic loader with normal points in anomalous windows and normal/anomalous windows.
- **Action:** Run evaluation-only code paths.
- **Expected result:** Both thresholds come from synthetic normal support, test labels affect only metrics, and no clean-validation score is used for new-protocol threshold selection.

### Atomic steps

- [ ] Add `normalized_input` to evaluator score-space validation.
- [ ] Select normalized point MSE as `point_scores` in that branch.
- [ ] Select normalized window MSE for window prediction in that branch.
- [ ] Record `normalized_input` in metric and record provenance.
- [ ] Write a failing evaluator test with differing raw and normalized values.
- [ ] Run that evaluator test and confirm it fails before the implementation is complete.
- [ ] Add a threshold-free `val_synth` evaluation call in the THESIS runner.
- [ ] Extract normalized point and window payload arrays from that call.
- [ ] Select synthetic-normal point q99.
- [ ] Select synthetic-normal window q99.
- [ ] Pass both values to synthetic and test evaluator calls.
- [ ] Include both values and sources in THESIS artifact inputs.
- [ ] Replace RedLamp's clean-validation raw helper branch with the protocol-driven branch.
- [ ] Add CLI handling for explicit protocol path and isolated output path if the existing arguments cannot supply them.
- [ ] Run `.venv/bin/python -m pytest -q tests/evaluation/test_evaluator_thresholding.py tests/benchmarks/test_evaluator_trace_payload.py`.

### Complete when

A reconstruction checkpoint evaluation produces normalized-MSE test predictions with two synthetic-normal thresholds and all budgeted VUS mappings.

## Phase 3: Bind one VUS-PR budget to each main-method run

### Goal

The trainer exposes one scalar synthetic-validation VUS-PR value per configured budget and the matrix materializes 54 collision-free configurations.

### Dependencies

- Phase 2 normalized reconstruction score behavior.
- `FPR_BUDGETS == (0.001, 0.005, 0.01)` at `src/metrics/pointwise.py:23`.

### Detailed changes

#### 1. Produce normalized synthetic-validation monitor scores

- **File:** `src/engine/trainer.py`.
- **Symbol:** `_validation_point_scores`, `_aggregate_reconstructed_pointwise_metrics`, and `_resolve_best_checkpoint_monitor`.
- **Current responsibility:** Normalized-loss training returns model `point_scores` for synthetic validation, budgeted VUS remains a nested mapping, and monitor validation accepts only old names.
- **Change:** Make synthetic-validation reconstruction point scores equal normalized-input MSE for the new protocol; flatten each budget mapping to stable scalar keys such as `val_synth_vus_pr_at_fpr_budget_0_001`; permit these names with `max` monitor mode.
- **Reason:** The existing checkpoint code already selects one float, so flattening is the smallest change that connects all three budgets to that path.
- **Inputs:** Synthetic batch reconstruction, synthetic anomaly mask, VUS metric mapping, and one configured monitor name.
- **Outputs:** Numeric epoch metrics for all three VUS-PR and VUS-ROC budgets, plus one selected scalar used by `best.pt`.
- **Errors:** Reject an unsupported budget monitor name and raise when the requested scalar is absent or non-finite.
- **Compatibility:** Preserve old monitor names and their modes.

#### 2. Generate the 54 budget-specific configurations

- **File:** `scripts/benchmarks/generate_full_direct_recon075_cls025_matrix.py`.
- **Symbol:** Matrix constants, generated name helpers, `build_config`, and `generate_configs`.
- **Current responsibility:** The generator loops O0/O1, three entities, and three seeds.
- **Change:** Add the three locked FPR budgets as the fourth loop axis; encode them as `fpr001`, `fpr005`, and `fpr01`; set one matching `checkpoint_monitor_metric`; set the new protocol path; add budget tags; and make `seed{seed}_fpr{compact}` the isolated output leaf.
- **Reason:** The generator already owns naming, model overrides, worker count, output roots, and W&B metadata.
- **Inputs:** Variant, entity, seed, FPR budget, base source YAML, and proposed new matrix root `outputs/benchmark_pro_reconstruction_vus_budget`.
- **Outputs:** 54 valid experiment YAML files with non-overlapping output and W&B names.
- **Errors:** Fail generation on unknown budget or duplicate generated path.
- **Compatibility:** The existing 18-cell generator stays untouched only if a separate new generator is required; otherwise update its name and callers together so no script points at stale semantics.

#### 3. Preflight and execute the expanded matrix

- **File:** `scripts/benchmarks/run_full_direct_recon075_cls025_offline_matrix.sh` or a new clearly named sibling script if the old script must retain historical behavior.
- **Symbol:** Generation command and two nested execution loops.
- **Current responsibility:** The script generates, preflights, and runs 18 cells sequentially with a clean-validation protocol.
- **Change:** Iterate the budget axis, use the new protocol, place preflight and run logs under the budget-specific root, and validate exactly 54 generated config paths before executing any training run.
- **Reason:** A script-level count guard prevents an accidental partial expansion or output collision.
- **Inputs:** Repository-root `.venv/bin/python`, generated config root, protocol path, and `fpr001/fpr005/fpr01` identifiers.
- **Outputs:** One preflight log and one run log per budget-specific combination.
- **Errors:** Exit before training when generation count, config validation, or required monitor identity is wrong.
- **Compatibility:** Do not alter the old full-run script if users still need its historical 18-cell behavior.

### Tests

#### Trainer monitor contract

- **Location:** `tests/runtime/test_trainer_checkpoint_fallback.py` and `tests/runtime/test_learning_rate_scheduler_trainer.py`.
- **Level:** Runtime unit tests.
- **Setup:** Use a trainer epoch metric dictionary containing distinct values for the three flattened VUS-PR keys.
- **Action:** Configure each budget monitor in turn.
- **Expected result:** The trainer accepts the name, uses `max`, saves the checkpoint with the matching float, and rejects an unknown budget string.
- **Edge cases:** Nested mapping values must never reach `float()` directly.

#### Generator and two-stage identity contract

- **Location:** `tests/benchmarks/test_offline_benchmark_config_generation.py` and `tests/benchmarks/test_two_stage_orchestration_dry_run.py`.
- **Level:** Configuration and dry-run integration tests.
- **Setup:** Generate the matrix in a temporary root.
- **Action:** Materialize both stage configs for every generated YAML.
- **Expected result:** There are 54 unique root configs, 108 unique stage W&B names, both stages retain direct branch routing, and each stage keeps the parent budget monitor.

### Atomic steps

- [ ] Add a deterministic compact identifier for `0.001`.
- [ ] Add a deterministic compact identifier for `0.005`.
- [ ] Add a deterministic compact identifier for `0.01`.
- [ ] Add scalar epoch metric names for the three VUS-PR budget values.
- [ ] Add scalar epoch metric names for the three VUS-ROC budget values.
- [ ] Add the three VUS-PR scalar names to the checkpoint-monitor mode map with `max`.
- [ ] Make new-protocol synthetic validation use normalized-input reconstruction MSE.
- [ ] Add a failing trainer test for each supported budget monitor.
- [ ] Run the trainer test and confirm it fails before monitor implementation completion.
- [ ] Add the FPR-budget loop to matrix generation.
- [ ] Add the FPR budget to config names.
- [ ] Add the FPR budget to W&B names and tags.
- [ ] Add the FPR budget to output paths without exceeding the existing result-path depth.
- [ ] Set the matching monitor metric in each generated YAML.
- [ ] Count generated files and assert `54`.
- [ ] Materialize one Stage A and one Stage B config for every budget identifier.
- [ ] Assert `108` unique stage run names.
- [ ] Add a preflight count guard to the runner script.
- [ ] Run `.venv/bin/python -m pytest -q tests/runtime/test_trainer_checkpoint_fallback.py tests/benchmarks/test_offline_benchmark_config_generation.py tests/benchmarks/test_two_stage_orchestration_dry_run.py`.

### Complete when

Every generated Stage A and Stage B config exposes one matching scalar VUS-PR budget monitor, and the matrix has exactly 54 root configurations.

## Phase 4: Evaluate baselines under the fairness contract

### Goal

RedLamp and traditional baselines apply the same calibration rule while retaining their existing model and score semantics.

### Dependencies

- Phase 1 synthetic-normal selectors and artifact schema.
- Phase 2 normalized reconstruction evaluation conventions.

### Detailed changes

#### 1. Expose native baseline scores without breaking legacy behavior

- **Files:** `src/baselines/traditional/base.py`, `src/baselines/traditional/iforest.py`, `src/baselines/traditional/kmeans_ad.py`, and `src/baselines/traditional/stumpy_channel_ab.py`.
- **Symbols:** `TraditionalBaselineProtocol`, each method's private window scorer, and each `score_sequence` method.
- **Current responsibility:** iForest and KMeans-AD calibrate from clean validation then transform point scores; StumPy calibrates channel scores from clean validation before aggregation.
- **Change:** Add explicit public native point/window scoring methods for the new evaluation runner, while preserving the existing `calibrate` and `score_sequence` API for historical callers.
- **Reason:** The new runner needs raw method-specific scores to apply synthetic-normal q99 without changing old benchmark behavior.
- **Inputs:** Train-fitted baseline state and scaled query sequence.
- **Outputs:** Native window scores, pointified native point scores, covered-point mask, and method score-definition metadata.
- **Errors:** Require `fit` before native scoring and reject invalid sequence dimensions as current helpers do.
- **Compatibility:** Do not remove current clean-validation calibration methods or transformed `score_sequence` behavior.

#### 2. Use synthetic normal calibration in the traditional runner

- **File:** `scripts/benchmarks/run_offline_benchmark.py`.
- **Symbol:** `_build_split_payload`, `_score_validation_split`, `run_offline_benchmark`, threshold artifact creation, and test metric construction.
- **Current responsibility:** The runner calls `baseline.calibrate(clean_validation)`, writes only point-score arrays, and uses the calibration threshold.
- **Change:** In the new protocol path, fit only on train, collect native synthetic-validation point/window scores, select two synthetic-normal q99 thresholds, score test with native values, and persist point/window predictions plus all VUS mappings.
- **Reason:** The runner owns baseline data split order and artifacts.
- **Inputs:** Existing baseline YAML, new protocol YAML, train/val_synth/test sequences, and method-native score methods.
- **Outputs:** Isolated score files, threshold artifact, metrics, report provenance, and unavailable-cell record when a checkpoint or baseline cannot be evaluated.
- **Errors:** Reject absent synthetic-validation split for the new protocol, no finite normal support, or a baseline missing the native-score contract.
- **Compatibility:** Retain the old clean-validation path for historical protocol configs.

#### 3. Evaluate RedLamp checkpoints without retraining

- **Files:** `scripts/cli/evaluate.py` and existing RedLamp benchmark configs under `configs/experiment/benchmark/baseline/`.
- **Symbols:** Protocol-driven checkpoint evaluation branch and CLI output override.
- **Current responsibility:** Existing RedLamp configs define trained checkpoint locations and output directories.
- **Change:** Reuse these configs and checkpoints, pass the new protocol to the extended evaluation CLI, and write to the new isolated baseline-evaluation root.
- **Reason:** The user excludes RedLamp retraining but requires normalized-MSE fairness evaluation.
- **Inputs:** Existing checkpoint, existing config, new protocol, and new output root.
- **Outputs:** New evaluation artifacts only.
- **Errors:** Record a missing checkpoint as unavailable without substituting another method.
- **Compatibility:** Never mutate the existing checkpoint or its historical output directory.

### Tests

#### Native-score preservation

- **Location:** `tests/models/test_traditional_baseline_contracts.py`.
- **Level:** Unit and method-contract tests.
- **Setup:** Fit each baseline on a fixed train sequence.
- **Action:** Request native window and point scores.
- **Expected result:** iForest uses negative decision function, KMeans-AD uses centroid distance, and StumPy uses AB-join distance aggregation without clean-validation calibration transforms.
- **Edge cases:** Empty usable windows return the existing finite-or-NaN representation consistently with the current method.

#### Traditional runner calibration

- **Location:** `tests/benchmarks/test_run_offline_benchmark.py`.
- **Level:** Runner integration test.
- **Setup:** Provide a fake native-score baseline and a synthetic split with normal points inside anomalous windows and mixed window labels.
- **Action:** Run the new protocol path.
- **Expected result:** The runner calls `fit` once on train, does not call historical clean calibration, writes two synthetic-normal thresholds, and metrics include every VUS budget.

### Atomic steps

- [ ] Add native-score methods to the traditional baseline protocol.
- [ ] Add iForest native window-score output from negative decision function.
- [ ] Add iForest native pointification output and covered mask.
- [ ] Add KMeans-AD native window-score output from nearest-centroid distance.
- [ ] Add KMeans-AD native pointification output and covered mask.
- [ ] Add StumPy native AB-join window aggregation without clean-validation channel normalization.
- [ ] Add StumPy native pointification output and covered mask.
- [ ] Write failing native-score contract tests.
- [ ] Run the native-score tests and confirm they fail before the new methods are complete.
- [ ] Add new-protocol synthetic split collection in the baseline runner.
- [ ] Derive synthetic window labels from synthetic point labels using the existing window-label convention.
- [ ] Select native point q99 from finite normal points.
- [ ] Select native window q99 from finite normal windows.
- [ ] Write point and window predictions for the test split.
- [ ] Add the two thresholds and score-definition provenance to the baseline artifact.
- [ ] Add a RedLamp evaluation-only command path with isolated output override.
- [ ] Run `.venv/bin/python -m pytest -q tests/models/test_traditional_baseline_contracts.py tests/benchmarks/test_run_offline_benchmark.py tests/models/test_redlamp_baseline_runtime.py`.

### Complete when

Every in-scope baseline can produce an isolated evaluation artifact with native or normalized-MSE score provenance and two synthetic-normal q99 thresholds.

## Phase 5: Verify smoke flow and prepare remote rollout

### Goal

The implementation earns permission to consume remote GPU time only after one local full-flow combination proves the contract.

### Dependencies

- Phases 1 through 4.

### Detailed changes

#### 1. Run focused automated verification

- **Files:** Existing test files named in Phases 1 through 4.
- **Current responsibility:** Tests cover parts of the current raw and clean-validation contracts.
- **Change:** Run the changed focused suites together after all code changes land.
- **Reason:** The combined command catches broken interfaces between thresholding, evaluator, trainer, artifact, and runner layers.
- **Outputs:** Passing test output and no whitespace errors.

#### 2. Run one main-method budget smoke combination

- **Files:** The new budget-aware smoke script or the updated matrix runner in smoke mode.
- **Current responsibility:** Existing smoke and full runners do not encode the FPR budget axis.
- **Change:** Run one explicit combination, for example O0, `machine_1_6`, seed `6`, `fpr001`, through Stage A, Stage-B initialization, Stage B, and offline evaluation.
- **Reason:** One real combination detects broken interaction among training, checkpoint monitoring, and calibration before the matrix starts.
- **Outputs:** Two W&B stage runs, stage manifest, `best.pt` metadata, threshold artifact, metrics, and report in the new isolated root.
- **Errors:** Stop before a remote full launch if the monitor name, threshold source, score space, or generated count is wrong.

#### 3. Prepare remote rollout without launching it

- **File:** `ssh-gpu.txt` and the budget-aware runner script.
- **Current responsibility:** Remote endpoint and output details are operational state, not stable source facts.
- **Change:** Re-read `ssh-gpu.txt`, verify remote checkout revision and available disk, then construct a tmux command that uses the new isolated output root.
- **Reason:** The remote details may have changed since prior work.
- **Compatibility:** Do not overwrite historical output roots or stop jobs not created by this experiment.

### Tests

#### Full-flow smoke acceptance

- **Location:** Smoke output manifest, threshold artifact, evaluation metrics, and W&B metadata.
- **Level:** End-to-end manual acceptance after automated tests.
- **Setup:** One reduced or smoke configuration with one explicit budget identity.
- **Action:** Run both offline stages and evaluation.
- **Expected result:** `best.pt` stores the configured VUS-PR budget monitor; point and window threshold sources are synthetic validation normal; metrics contain both VUS curve mappings for all three budgets.
- **Edge cases:** The smoke evaluation may be truncated or single-class, so treat it as protocol evidence rather than benchmark quality evidence.

### Atomic steps

- [ ] Run the focused threshold, evaluator, trainer, artifact, generator, and baseline pytest files.
- [ ] Run `git diff --check`.
- [ ] Generate the smoke configuration for one explicit `fpr001`, `fpr005`, or `fpr01` identity.
- [ ] Run the smoke preflight.
- [ ] Inspect the resolved Stage A monitor name.
- [ ] Inspect the resolved Stage B monitor name.
- [ ] Run the one-combination smoke flow.
- [ ] Inspect Stage A `best.pt` monitor metadata.
- [ ] Inspect Stage B `best.pt` monitor metadata.
- [ ] Inspect point threshold source and score definition.
- [ ] Inspect window threshold source and score definition.
- [ ] Inspect all six VUS metric mappings in evaluation metrics.
- [ ] Re-read `ssh-gpu.txt` before any remote connection.
- [ ] Verify remote repository revision before upload or launch.
- [ ] Verify remote output root does not overlap historical experiments.
- [ ] Prepare the tmux command without starting the 54-cell matrix.

### Complete when

One budget-specific smoke run has auditable checkpoint, score, threshold, and metric provenance, and the remote launch prerequisites are documented.

## Interface and data changes

The new reconstruction protocol accepts `score_space: normalized_input` and an explicit offline window threshold source.

The trainer adds scalar keys `val_synth_vus_pr_at_fpr_budget_0_001`, `val_synth_vus_pr_at_fpr_budget_0_005`, and `val_synth_vus_pr_at_fpr_budget_0_01` as proposed canonical monitor names.

The threshold artifact adds a versioned `offline_window` entry and score-space-specific point/window score definitions.

Traditional baselines add public native-score access for the new fairness path while retaining their current historical APIs.

## Deployment and rollout

Run tests first, then one local or reduced smoke combination, then inspect its artifacts, then re-check remote state, and only then start the 54-cell remote matrix.

No historical config, checkpoint, or output directory is migrated or overwritten.

## Documentation changes

- Update `documents/notes/pro-reconstruction-offline-fairness-decisions.md` after implementation approval to replace the old 18-combination and unresolved-monitor statements with the confirmed 54-combination, 108-W&B-run contract.
- Add a runbook beside the new script only after the exact remote command and output root are verified.

## Final verification

- [ ] The generator returns 54 unique configurations.
- [ ] Two-stage materialization returns 108 unique W&B stage names.
- [ ] Each stage selects `best.pt` with the VUS-PR budget named by its parent run.
- [ ] Every main-method and RedLamp evaluation uses normalized-input MSE.
- [ ] Every baseline report retains its native score definition.
- [ ] Every new artifact records synthetic-normal point and window threshold provenance.
- [ ] Every evaluation metrics file contains VUS-PR and VUS-ROC at `0.001`, `0.005`, and `0.01`.

## Assumptions and non-blocking uncertainties

- The implementation uses `fpr001`, `fpr005`, and `fpr01` as compact run identifiers; tests will make this mapping explicit.
- The final remote output root and tmux command remain intentionally deferred until the smoke run passes and `ssh-gpu.txt` is reread.
