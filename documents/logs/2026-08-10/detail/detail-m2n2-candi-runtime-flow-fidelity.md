---
date: 2026-08-10 Asia/Ho_Chi_Minh
topic: "Detailed implementation: M2N2 and CANDI runtime-flow fidelity corrections"
status: implemented_smoke_verified_matrix_pending
revision: aebf99382af8508f2dd0f809a531a1bd72dba4c1
source_structure: documents/logs/2026-08-10/research/research-m2n2-candi-method-fidelity.md#follow-up-implementation-structure-sequential-stages
related_documents:
  - documents/logs/2026-08-10/research/research-m2n2-candi-method-fidelity.md
  - documents/spec/online_benchmark_contract.md
  - prompts/1_research_prompt.md
  - prompts/2_plan_prompt.md
  - prompts/3_structure_prompt.md
  - prompts/4_detail_prompt.md
---

# Detailed Implementation: M2N2 and CANDI runtime-flow fidelity corrections

## Summary

This document expands the approved seven-phase structure into sequential atomic
steps. The target is the reference Predictor and adapter lifecycle from
`bsc-thesis-ref-codebases/CANDI-main`, executed on the approved RedLamp encoder
checkpoint variant `reference_adapter_redlamp_encoder`.

The plan corrects the common Predictor order, batch-causal state visibility,
clean-validation q99.5 threshold profile, reference TTA optimizer, M2N2 update
timing, CANDI FPM representation input, SANA topology boundary, configuration
identity, tests and smoke verification. It does not claim native MLP/TimesNet
model equivalence.

## Source structure

The source structure is the follow-up section in the research report:

`documents/logs/2026-08-10/research/research-m2n2-candi-method-fidelity.md#follow-up-implementation-structure-sequential-stages`

Its phases are preserved in this order:

1. Freeze the runtime-fidelity acceptance contract.
2. Rebuild the common Predictor and batch-causal lifecycle.
3. Align threshold and optimizer semantics.
4. Correct M2N2 reference update execution.
5. Correct CANDI FPM and SANA runtime parity.
6. Migrate configurations, tests and artifacts.
7. Validate one real end-to-end run and approve expansion.

## Current state

- `scripts/benchmarks/run_online_streaming_benchmark.py:297-405` loads the
  dataset, constructs a baseline, calibrates validation and calls
  `run_sequence()`.
- `src/baselines/online/adaptive.py:343-355` scores a window, calls adaptation,
  and then computes the prediction/record from the saved raw score.
- `src/baselines/online/base.py:92-98` fixes the helper batch size to `1`.
- `src/baselines/online/adaptive.py:281-307` calibrates the active raw-window
  threshold from q99 configuration values.
- `src/baselines/online/m2n2.py:73-77` and `src/baselines/online/candi.py:138-142`
  always construct AdamW optimizers.
- `src/baselines/online/candi.py:233-235` uses `x + sana_in(x)` for current FPM
  representations.
- Focused method tests and the full `tests/online` suite pass, but they do not
  yet prove exact reference threshold, optimizer, batch and representation
  parity.

## Desired end state

The active runtime must satisfy all of the following observable conditions:

- Each test batch is scored completely before its adapter update.
- Prediction and record fields are created from the pre-update score before the
  update is called.
- Only the next test batch sees updated model, optimizer or method state.
- Thresholds come from clean-validation raw scores under the selected q99.5
  reference profile.
- M2N2 uses reference Detrender, timestep masking, masked loss and one selected
  TTA optimizer step.
- CANDI uses raw-input FPM representations, reference hard/moderate selection,
  separate pools, minimum-sample gates and SANA reconstruction MSE.
- Every corrected artifact carries method, model, checkpoint, threshold,
  optimizer, batch and comparability identity.

## Scope

### In scope

- The benchmark runner and adaptive baseline lifecycle.
- M2N2 and CANDI method-owned update state.
- Threshold, optimizer, configuration and metadata contracts.
- Fidelity tests, generated artifacts and one real smoke per method.

### Out of scope

- Replacing the RedLamp checkpoint with an unverified native MLP/TimesNet model.
- Changing `src/engine/online_tta/` THESIS behavior.
- Adding THESIS triage, PNN verification, projector updates or uncertainty
  ablations to M2N2/CANDI.
- Overwriting legacy approximation outputs or mixing their metrics with the
  corrected adapter-variant results.

## Evidence

- `scripts/benchmarks/run_online_streaming_benchmark.py:297-405` — active
  runner, calibration and test-stream entry path.
- `src/baselines/online/adaptive.py:268-355` — active threshold and test loop.
- `src/baselines/online/base.py:80-98` — stride-one batch helper.
- `src/baselines/online/m2n2.py:67-149` — M2N2 state and update.
- `src/baselines/online/candi.py:105-333` — CANDI state, selection and update.
- `src/models/online_adapter_modules.py:46-136` — current SANA module.
- `bsc-thesis-ref-codebases/CANDI-main/predictor.py:21-103` — reference
  Predictor initialization and causal test loop.
- `bsc-thesis-ref-codebases/CANDI-main/tta/m2n2/adapter_m2n2.py:13-39` —
  reference M2N2 update.
- `bsc-thesis-ref-codebases/CANDI-main/tta/candi/adapter_candi.py:133-379` —
  reference CANDI setup, FPM selection, pools and SANA loss.
- `bsc-thesis-ref-codebases/CANDI-main/config.py:69-97` — reference threshold,
  CANDI and TTA solver defaults.

## Phase 0: Freeze the runtime-fidelity acceptance contract

### Goal

Define one configuration and artifact identity for the corrected RedLamp
adapter variant before changing executable behavior.

### Dependencies

- Follow-up runtime audit in the research report.
- Approved RedLamp simple 1D-CNN checkpoint decision.
- Existing `documents/spec/online_benchmark_contract.md`.

### Detailed atomic steps

#### Step 0.1: Read the active contract before changing names

- **File:** `documents/spec/online_benchmark_contract.md`
- **Symbol:** online method identity, checkpoint and update-policy sections
- **Current responsibility:** Defines the active online model/checkpoint,
  threshold and adaptation metadata contract.
- **Change:** Identify the exact sections that describe `main`, RedLamp,
  threshold source, trainable surface and update policy. Treat them as the
  contract to revise, not as evidence that the runtime already matches it.
- **Reason:** The implementation must not create a second undocumented contract.
- **Inputs:** Current specification and research follow-up audit.
- **Outputs:** A checked list of contract fields that later code and generated
  artifacts must emit.
- **Errors:** Stop if the specification still conflates native MLP/TimesNet and
  `reference_adapter_redlamp_encoder`.
- **Dependencies:** None beyond the source structure.
- **Compatibility:** Preserve unrelated frozen baselines and THESIS contracts.
- **Verification:** Compare each field with current constructor kwargs and one
  existing benchmark report.

#### Step 0.2: Define the corrected runtime profile

- **File:** `documents/spec/online_benchmark_contract.md`
- **Symbol:** method profile and threshold/update tables
- **Current responsibility:** Records adapter mechanisms but still permits q99,
  AdamW and batch-one behavior without a single reference profile.
- **Change:** Add one explicit corrected profile containing the reference
  `score -> prediction/record -> adapt` order, clean-validation raw threshold,
  q99.5 ratio, selected TTA optimizer, M2N2 gamma/steps, CANDI FPM/SANA
  settings, selected batch size and checkpoint role.
- **Reason:** The same profile must drive configuration, metadata and tests.
- **Inputs:** Reference `config.py`, `predictor.py`, M2N2 adapter and CANDI
  adapter values.
- **Outputs:** A named profile that the generator and constructors can validate.
- **Errors:** Reject a profile that leaves optimizer, threshold or batch size
  implicit.
- **Dependencies:** Step 0.1.
- **Compatibility:** Keep the existing RedLamp checkpoint name and mark native
  MLP/TimesNet equivalence as unsupported.
- **Verification:** Manually compare the profile table with the reference files
  and the follow-up audit list.

#### Step 0.3: Define the migration identity for old configurations

- **File:** `configs/experiment/online_benchmark/m2n2/` and
  `configs/experiment/online_benchmark/candi/`
- **Symbol:** legacy `main`, `A0`, `A1`, `A2` and corrected variant names
- **Current responsibility:** Contains multiple generations of online configs.
- **Change:** Specify which configs are historical approximation inputs and
  which configs are corrected `reference_adapter_redlamp_encoder` inputs.
- **Reason:** A benchmark run must not silently reuse a stale runtime contract.
- **Inputs:** Config filenames, `online_variant`, baseline kwargs and output
  roots.
- **Outputs:** A migration mapping used by generator and config tests.
- **Errors:** Fail validation when a corrected config has no checkpoint, profile
  identity or method-native settings.
- **Dependencies:** Step 0.2.
- **Compatibility:** Do not delete or overwrite historical configs in this plan.
- **Verification:** Inspect one config from each generation and compare their
  metadata/output identities.

#### Step 0.4: Define the contract-validation failure boundary

- **File:** `tests/online/test_online_method_fidelity.py` and existing config
  validation tests
- **Symbol:** proposed profile/config contract assertions in existing test files
- **Current responsibility:** Checks method mechanisms and basic config identity.
- **Change:** Specify rejection cases for missing threshold profile, missing
  optimizer, wrong checkpoint role, legacy variant leakage and native-model
  claims attached to the RedLamp variant.
- **Reason:** Configuration drift must fail before a benchmark produces artifacts.
- **Inputs:** Corrected profile and migration mapping.
- **Outputs:** Deterministic contract failures with the offending field named.
- **Errors:** Report the first mismatched profile field and do not fall back to a
  legacy default.
- **Dependencies:** Steps 0.2-0.3.
- **Compatibility:** Existing focused tests remain valid unless they assert the
  old approximation identity.
- **Verification:** Run the focused method/config test group after implementation.

### Tests

#### Runtime profile contract

- **Location:** `tests/online/test_online_method_fidelity.py` and the existing
  online config/artifact contract tests.
- **Level:** Unit/contract.
- **Setup:** Build one corrected M2N2 config, one corrected CANDI config and one
  legacy config fixture.
- **Action:** Load metadata and validate each profile.
- **Expected result:** Corrected configs pass; legacy configs are identified as
  non-comparable and cannot masquerade as corrected configs.
- **Edge cases:** Missing checkpoint, q99 profile, AdamW-only profile and
  `online_variant: main`.

### Verification

#### Automated

- [ ] `.venv/bin/pytest -q tests/online/test_online_method_fidelity.py` — profile
  and identity assertions pass after implementation.
- [ ] Existing config/artifact contract tests — corrected configs expose all
  required profile fields.

#### Manual

- [ ] Compare one corrected YAML, its method metadata and its threshold artifact;
  all three must carry the same profile identity.

### Risks and recovery

- **Risk:** A stale default silently reintroduces q99, AdamW or legacy variant
  behavior.
- **Mitigation:** Validate profile fields after constructor instantiation and
  reject mismatches.
- **Verification:** Use the negative contract fixtures from Step 0.4.
- **Recovery:** Keep the prior artifacts untouched and repair the profile/config
  source before running any benchmark.

### Complete when

- The corrected profile is documented.
- Legacy and corrected artifact identities are distinct.
- Negative profile/config tests fail before runtime execution.

## Phase 1: Rebuild the common Predictor and batch-causal lifecycle

### Goal

Make the runner and shared adaptive base implement the reference batch lifecycle
without allowing a current update to affect the current batch's records.

### Dependencies

- Phase 0 runtime profile and config identity.
- Existing `run_online_streaming_benchmark()` entry point.
- Existing stride-one stream and absolute-index behavior.

### Detailed atomic steps

#### Step 1.1: Record the current runner boundaries before refactoring

- **File:** `scripts/benchmarks/run_online_streaming_benchmark.py`
- **Symbol:** `run_online_streaming_benchmark`, baseline construction,
  calibration and test loop
- **Current responsibility:** Loads splits, constructs a baseline, calibrates,
  selects a test range and calls `run_sequence()`.
- **Change:** Map the current ownership of split loading, threshold creation,
  stream selection, record aggregation and report writing before moving any
  lifecycle operation.
- **Reason:** The reference flow must be added without duplicating dataset or
  artifact ownership.
- **Inputs:** Current runner and benchmark report schema.
- **Outputs:** A stable boundary list used by the next steps.
- **Errors:** Stop if a proposed pre-TTA stage would consume test labels or
  overwrite the existing threshold/report artifact contract.
- **Dependencies:** Phase 0.
- **Compatibility:** Preserve CLI/config entry point and selected absolute range.
- **Verification:** Trace one existing smoke configuration from YAML to report.

#### Step 1.2: Define the batch input/output contract

- **File:** `src/baselines/online/base.py`
- **Symbol:** `OnlineStreamingBaselineProtocol`, `build_stride1_batcher`
- **Current responsibility:** Exposes `calibrate()` and `run_sequence()` and
  provides a helper with `batch_size=1`.
- **Change:** Define the internal batch contract as `[B,L,D]` input with one
  score/prediction record per window and one adaptation call per batch.
- **Reason:** The reference adapter receives a batch, not one independently
  updated window at a time.
- **Inputs:** Stride-one stream windows, absolute indices and selected batch size.
- **Outputs:** Ordered batch records plus method update diagnostics.
- **Errors:** Reject wrong tensor rank, inconsistent batch lengths or non-causal
  index ordering.
- **Dependencies:** Step 1.1 and `src/data/stream.py`.
- **Compatibility:** Preserve `calibrate()` and `run_sequence()` as the external
  runner boundary while keeping the new batch behavior internal.
- **Verification:** Add a multi-window fixture and assert shape/index contracts.

#### Step 1.3: Preserve stream windows and absolute indices while batching

- **File:** `src/data/stream.py` and `src/baselines/online/base.py`
- **Symbol:** `SMDOnlineStream`, `OnlineWindowBatcher`,
  `build_stride1_batcher`
- **Current responsibility:** Produces stride-one windows and can construct a
  batcher, but adaptive `run_sequence()` bypasses it.
- **Change:** Route the corrected adaptive test path through the stream/batcher
  boundary or implement the same cursor semantics in the baseline loop. Keep
  `window_start_index`, `window_end_index`, `point_index` and `stream_step`.
- **Reason:** The reference loop consumes batches in order, while the current
  path creates all windows directly from the full sequence.
- **Inputs:** Selected sequence, window size, stride and configured batch size.
- **Outputs:** `[B,L,D]` batches with stable entity-global indices.
- **Errors:** Reject gaps, overlap changes, out-of-range indices or a batch that
  crosses unrelated entity sequences.
- **Dependencies:** Step 1.2 and existing absolute-range selector.
- **Compatibility:** Preserve the current half-open selected range and stride-one
  alignment.
- **Verification:** Existing stream tests plus a fixture comparing expected
  starts `[0,1,2,...]` with emitted metadata.

#### Step 1.4: Add the reference-ordered pre-TTA stage

- **File:** `scripts/benchmarks/run_online_streaming_benchmark.py`
- **Symbol:** `run_online_streaming_benchmark` around construction/calibration
  and test execution
- **Current responsibility:** Constructs the adapter baseline before entering the
  adaptive test loop and does not compute the reference pre-TTA score sets.
- **Change:** Add the selected pre-TTA score/calibration boundary before adapter
  updates. Keep train/validation/test score ownership explicit and ensure the
  adapter receives only the data allowed by the method contract.
- **Reason:** The reference Predictor computes pre-TTA scores before creating
  the TTA adapter and threshold result.
- **Inputs:** Trained/checkpoint model, train/validation/test loaders and labels
  only for final metrics.
- **Outputs:** Pre-TTA score artifacts or report fields required by the profile,
  then the adapter-ready threshold state.
- **Errors:** Fail when a score set is computed with post-adaptation state or
  when test labels enter threshold/adapter inputs.
- **Dependencies:** Steps 1.1-1.3 and Phase 0 profile.
- **Compatibility:** Preserve the runner's output schema; add provenance fields
  rather than replacing existing common fields.
- **Verification:** Integration test records pre-TTA state before any optimizer
  or pool update is possible.

#### Step 1.5: Move prediction and record construction before adaptation

- **File:** `src/baselines/online/adaptive.py`
- **Symbol:** `AdaptiveStreamingBaselineBase.run_sequence`
- **Current responsibility:** Scores one window, calls `_adapt_tensor()`, then
  creates `prediction` and record fields.
- **Change:** For each batch, compute raw scores, EWMA reporting values,
  predictions and all pre-update record fields. Append the records before
  invoking the method adaptation function.
- **Reason:** The reference side-effect order is
  `score -> prediction/record -> adapt`.
- **Inputs:** Current batch tensor, threshold and protocol EWMA weights.
- **Outputs:** Records containing pre-update score/prediction and a later update
  result attached without rewriting the prediction.
- **Errors:** Reject a record if its prediction was computed after state mutation
  or if score count does not match batch size.
- **Dependencies:** Steps 1.2-1.4.
- **Compatibility:** Keep existing record keys and add/update diagnostics only
  after the method returns.
- **Verification:** Monkeypatch score and adapt callbacks and assert exact call
  order.

#### Step 1.6: Apply one update after scoring the whole batch

- **File:** `src/baselines/online/adaptive.py`, `src/baselines/online/m2n2.py`
  and `src/baselines/online/candi.py`
- **Symbol:** `_adapt_tensor()` implementations and shared test loop
- **Current responsibility:** Receives one `[1,L,D]` window and may update before
  the next loop iteration.
- **Change:** Pass the complete `[B,L,D]` batch to one adapter update after all
  batch predictions are recorded. Do not call the adapter separately for each
  item in the same batch.
- **Reason:** Reference state is batch-causal, not item-causal within a batch.
- **Inputs:** Current batch and its pre-update scores; M2N2 uses its threshold
  internally and CANDI uses batch candidates.
- **Outputs:** One method update summary and post-update state for the next batch.
- **Errors:** Reject partial batch updates or state changes between batch items.
- **Dependencies:** Steps 1.2 and 1.5.
- **Compatibility:** Keep `batch_size=1` behavior as a valid degenerate case;
  do not label it as the only supported reference behavior.
- **Verification:** Multi-window batch test compares parameters before/after the
  update and confirms all current-batch predictions use the old state.

#### Step 1.7: Verify causal state serialization boundary

- **File:** `src/baselines/online/base.py` and method owners
- **Symbol:** method runtime state and metadata/report serialization
- **Current responsibility:** Emits scalar diagnostics but does not expose a
  complete batch-causal state snapshot.
- **Change:** Identify the minimum state needed to audit a run: model parameters,
  optimizer state, M2N2 Detrender state, CANDI references and candidate pools.
- **Reason:** A later-batch effect cannot be reproduced if method state is not
  identifiable.
- **Inputs:** Runtime state after each batch or at artifact finalization.
- **Outputs:** JSON-compatible metadata plus tensor/checkpoint state where the
  existing artifact contract permits it.
- **Errors:** Fail report validation when update state is claimed but omitted.
- **Dependencies:** Steps 1.5-1.6 and artifact contract.
- **Compatibility:** Do not persist every forward output; keep selected
  diagnostics only.
- **Verification:** State round-trip or deterministic snapshot test where the
  current artifact system supports it.

### Tests

#### Batch-causal call order

- **Location:** `tests/online/test_online_method_fidelity.py`
- **Level:** Unit/integration.
- **Setup:** Two batches with at least two windows in the first batch and a
  deterministic mutable model/update callback.
- **Action:** Run the corrected lifecycle and capture score, record and adapt
  events.
- **Expected result:** All first-batch scores and predictions precede one update;
  the second batch uses updated state.
- **Edge cases:** `batch_size=1`, final incomplete batch and empty sequence.

### Verification

#### Automated

- [ ] `.venv/bin/pytest -q tests/online/test_online_method_fidelity.py` — exact
  call order and multi-window causal tests pass.
- [ ] `.venv/bin/pytest -q tests/online/test_online_streaming_baseline_contracts.py`
  — existing record/index contracts remain valid.

#### Manual

- [ ] Inspect the first two batches of one smoke trace and verify that no first
  batch prediction changes after its adaptation call.

### Risks and recovery

- **Risk:** A batcher change may alter selected range or point alignment.
- **Mitigation:** compare emitted absolute indices against the existing stream
  tests before threshold recalibration.
- **Verification:** Run stream/index integration tests and inspect one report.
- **Recovery:** Keep the old output root and restore the prior stream profile;
  do not mix records from different batch contracts.

### Complete when

- The runner emits pre-update batch records.
- One update occurs per batch.
- The next batch, and only the next batch, observes updated state.

## Phase 2: Align threshold and optimizer semantics

### Goal

Make the threshold and TTA optimizer explicit reference-profile inputs and make
their identities visible in the benchmark artifacts.

### Dependencies

- Phase 0 runtime profile.
- Phase 1 batch-causal score lifecycle.
- Existing threshold artifact builder and generated-config workflow.

### Detailed atomic steps

#### Step 2.1: Add the q99.5 protocol identity

- **File:** `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml` and the
  protocol configuration loader
- **Symbol:** `online_threshold_quantile`, protocol name and threshold fields
- **Current responsibility:** Selects clean-validation q99 for the current
  online benchmark.
- **Change:** Add or rename a protocol identity for the reference TTA ratio
  `ANOMALY_RATIO=0.5`, which maps to raw validation percentile `0.995`.
- **Reason:** The reference TTA path forces ratio thresholding and does not use
  the current q99 default.
- **Inputs:** Clean-validation raw-window scores and anomaly ratio `0.5` percent.
- **Outputs:** A protocol config whose quantile and name identify q99.5.
- **Errors:** Reject a protocol whose declared ratio and quantile disagree.
- **Dependencies:** Phase 0 profile.
- **Compatibility:** Keep the existing q99 protocol for historical runs; do not
  overwrite its artifacts.
- **Verification:** Load both protocol files and assert different names and
  quantiles.

#### Step 2.2: Calculate the adapter threshold from raw validation scores

- **File:** `src/baselines/online/adaptive.py`
- **Symbol:** `AdaptiveStreamingBaselineBase.calibrate`
- **Current responsibility:** Calculates raw and EWMA validation quantiles and
  returns the raw q99 threshold for the current runner.
- **Change:** Use the corrected profile's clean-validation raw score quantile
  for the adapter decision threshold. Keep EWMA threshold only where the
  artifact/report contract explicitly requires it as a diagnostic.
- **Reason:** Reference TTA compares raw anomaly scores with the ratio threshold.
- **Inputs:** Finite validation window scores, quantile `0.995`, EWMA weights.
- **Outputs:** `threshold_value`, `threshold_source` and artifact fields that
  distinguish raw threshold from EWMA diagnostics.
- **Errors:** Fail on no finite validation scores, invalid quantile or mismatched
  threshold source.
- **Dependencies:** Step 2.1 and Phase 1 calibration lifecycle.
- **Compatibility:** Preserve validation-only threshold split and existing
  threshold artifact schema fields where possible.
- **Verification:** Deterministic quantile fixture and threshold artifact test.

#### Step 2.3: Add explicit optimizer method configuration

- **File:** `scripts/benchmarks/generate_online_streaming_benchmark_configs.py`
  and M2N2/CANDI baseline kwargs
- **Symbol:** `BENCHMARK_METHOD_VARIANTS` and method-specific kwargs
- **Current responsibility:** Emits learning rate and weight decay while the
  constructors silently choose AdamW.
- **Change:** Add an optimizer method field to corrected configs. Set the
  reference profile default to `sgd`, and retain learning rate, weight decay,
  momentum, dampening and Nesterov fields only when the selected optimizer uses
  them.
- **Reason:** Optimizer choice changes adaptation dynamics and must not be hidden.
- **Inputs:** Reference TTA solver defaults and selected profile.
- **Outputs:** YAML fields consumed by both method constructors.
- **Errors:** Reject unsupported optimizer names or missing required optimizer
  parameters.
- **Dependencies:** Phase 0 profile and config generator.
- **Compatibility:** Legacy configs retain their historical identity and are not
  silently rewritten.
- **Verification:** Generator test compares emitted fields with the profile.

#### Step 2.4: Construct the selected optimizer without changing parameter scope

- **File:** `src/baselines/online/m2n2.py` and `src/baselines/online/candi.py`
- **Symbol:** `_initialize_method_state`
- **Current responsibility:** Creates AdamW over the current method parameter
  list.
- **Change:** Select the configured optimizer class. For M2N2, include the
  parameters allowed by the RedLamp adapter variant. For CANDI with SANA,
  include only `sana_in` and `sana_out`; keep the pretrained backbone frozen.
- **Reason:** Optimizer identity and parameter scope are separate contracts.
- **Inputs:** Optimizer kwargs, model parameters and CANDI `USE_SANA` setting.
- **Outputs:** Optimizer with deterministic parameter groups and metadata.
- **Errors:** Reject an empty parameter group, an unknown optimizer or a frozen
  parameter attached to a CANDI SANA optimizer.
- **Dependencies:** Step 2.3 and existing method state initialization.
- **Compatibility:** Preserve the selected RedLamp model surface and method
  metadata keys; add optimizer identity rather than replacing method names.
- **Verification:** Parameter-group and optimizer-class tests for both methods.

#### Step 2.5: Record threshold and optimizer provenance

- **File:** `src/baselines/online/adaptive.py`, `m2n2.py`, `candi.py` and
  `src/protocols/threshold_artifact.py`
- **Symbol:** `_method_metadata`, `build_threshold_artifact`
- **Current responsibility:** Records checkpoint and adaptation settings but not
  all reference-profile fields.
- **Change:** Add threshold score space, quantile, anomaly ratio, optimizer
  method, optimizer hyperparameters, batch size and trainable surface.
- **Reason:** Reports must explain why corrected results are not comparable to
  q99/AdamW historical results.
- **Inputs:** Runtime profile, instantiated optimizer and calibration output.
- **Outputs:** JSON-serializable metadata in threshold artifact and report.
- **Errors:** Fail if metadata says q99.5 while the actual threshold is q99 or if
  optimizer metadata differs from the instantiated class.
- **Dependencies:** Steps 2.2-2.4.
- **Compatibility:** Preserve common artifact keys consumed by existing report
  readers; add a profile/version identity.
- **Verification:** Artifact round-trip and metadata consistency test.

### Tests

#### Threshold and optimizer profile

- **Location:** `tests/online/test_threshold_artifact.py`,
  `tests/online/test_online_method_fidelity.py` and config generator tests.
- **Level:** Unit/contract.
- **Setup:** Use deterministic clean-validation scores, corrected configs and
  both method constructors.
- **Action:** Calibrate thresholds and inspect optimizer classes/parameter groups.
- **Expected result:** Threshold equals q99.5 raw validation percentile; both
  methods expose the selected optimizer and correct trainable scope.
- **Edge cases:** Empty scores, invalid quantile, unsupported optimizer and CANDI
  `USE_SANA=False`.

### Verification

#### Automated

- [ ] `.venv/bin/pytest -q tests/online/test_threshold_artifact.py tests/online/test_online_method_fidelity.py`
  — threshold/profile assertions pass.
- [ ] Generator/config validation — every corrected YAML carries the selected
  quantile and optimizer.

#### Manual

- [ ] Compare one threshold artifact with the actual `threshold_value` and one
  method report with the instantiated optimizer class.

### Risks and recovery

- **Risk:** q99.5 and optimizer changes invalidate historical metrics.
- **Mitigation:** use separate protocol/output identities and preserve old files.
- **Verification:** artifact comparability test rejects mixed identities.
- **Recovery:** rerun only under the prior historical config into its original
  historical output root; never overwrite corrected artifacts.

### Complete when

- Corrected configs select q99.5 and the reference optimizer explicitly.
- Runtime metadata matches actual threshold and optimizer state.
- Historical q99/AdamW artifacts remain separate.

## Phase 3: Correct M2N2 reference update execution

### Goal

Ensure M2N2 preserves the reference Detrender, timestep pseudo-anomaly mask,
masked reconstruction loss and one optimizer step after the common batch order
has been corrected.

### Dependencies

- Phase 1 batch input and record timing.
- Phase 2 optimizer profile.
- Existing `Detrender` and RedLamp reconstruction model contract.

### Detailed atomic steps

#### Step 3.1: Validate Detrender state shape and initialization

- **File:** `src/models/online_adapter_modules.py`
- **Symbol:** `Detrender.__init__`, `update_statistics`, `normalize`,
  `denormalize`
- **Current responsibility:** Stores a mean-only EMA state and normalizes input
  by subtracting/adding the mean.
- **Change:** Confirm the state is shaped `[1,1,C]`, initialized at zero and
  updated with `mean_new = gamma * mean_old + (1-gamma) * batch_mean` over batch
  and time dimensions.
- **Reason:** M2N2 reference uses mean-only Detrender state, not mean/std state.
- **Inputs:** `[B,L,C]` batch and `gamma=0.99999` in the reference profile.
- **Outputs:** Updated mean state and normalized/denormalized tensors with the
  original shape.
- **Errors:** Reject wrong feature count, invalid gamma or non-finite batch mean.
- **Dependencies:** Phase 0 profile and existing Detrender module.
- **Compatibility:** Do not add gradient updates to Detrender mean.
- **Verification:** Deterministic EMA fixture compares exact floating-point
  update against the reference equation.

#### Step 3.2: Keep scoring separate from adaptation

- **File:** `src/baselines/online/m2n2.py`
- **Symbol:** `_score_tensor`, `_score_tensor_batch`, `_adapt_tensor`
- **Current responsibility:** Scores with current Detrender state and adapts
  after the shared runner calls the method.
- **Change:** Ensure scoring never updates Detrender or optimizer state. Ensure
  `_adapt_tensor` receives the complete current batch and updates Detrender only
  after the prediction/record stage.
- **Reason:** The reference prediction score is pre-adaptation, while the
  adaptation forward uses the newly updated Detrender statistics.
- **Inputs:** Current batch and pre-update raw score/threshold metadata.
- **Outputs:** Stable score from old state; update loss/diagnostics from new
  Detrender state.
- **Errors:** Reject accidental gradient graph retention in score methods or
  score calls that mutate Detrender state.
- **Dependencies:** Phase 1 lifecycle.
- **Compatibility:** Keep the score reduction `mean` over time and features.
- **Verification:** Snapshot Detrender and model state before scoring and assert
  no mutation; then assert mutation occurs only during adaptation.

#### Step 3.3: Compute the exact timestep mask

- **File:** `src/baselines/online/m2n2.py`
- **Symbol:** `M2N2StreamingBaseline._adapt_tensor`
- **Current responsibility:** Computes timestep error, `>= threshold` mask and
  masked loss for a one-window input.
- **Change:** For `[B,L,C]`, compute `E=(recon-x)^2`, `A=mean(E, dim=-1)`,
  `ytilde=(A >= threshold)`, `normal_mask=(ytilde == 0)` and
  `loss=(A * normal_mask).mean()` across all `B*L` positions.
- **Reason:** The reference removes pseudo-anomalous timesteps from the loss but
  still divides by the complete batch-time count.
- **Inputs:** Adaptation reconstruction, raw batch, threshold.
- **Outputs:** Scalar loss, normal mask count and pseudo-anomaly count.
- **Errors:** Reject shape mismatch, non-finite reconstruction, loss or mask
  diagnostics.
- **Dependencies:** Steps 3.1-3.2 and Phase 2 threshold profile.
- **Compatibility:** Do not use test labels or the passed `scores` argument to
  create the mask.
- **Verification:** Hand-computed fixture with both normal and pseudo-anomaly
  timesteps checks mask and loss exactly.

#### Step 3.4: Execute one configured M2N2 optimizer step

- **File:** `src/baselines/online/m2n2.py`
- **Symbol:** `_adapt_tensor`
- **Current responsibility:** Runs zero-grad, backward and optimizer step with
  the current method optimizer.
- **Change:** Run exactly one configured step after the masked loss, using the
  Phase 2 optimizer and the allowed RedLamp adapter parameter surface.
- **Reason:** Reference adapter factory sets M2N2 TTA steps to one.
- **Inputs:** Finite masked loss and configured optimizer.
- **Outputs:** Updated model parameters, finite `loss_total` and diagnostics.
- **Errors:** Fail before state reporting on non-finite loss or missing optimizer;
  restore train/eval mode on failure where possible.
- **Dependencies:** Steps 3.3 and Phase 2 optimizer construction.
- **Compatibility:** Keep `m2n2_steps=1` as the corrected profile and reject
  unsupported alternative values until explicitly specified.
- **Verification:** Compare trainable parameter state before/after one update;
  frozen/disallowed state must remain unchanged.

#### Step 3.5: Restore model mode and publish update diagnostics

- **File:** `src/baselines/online/m2n2.py` and adaptive record schema
- **Symbol:** `_adapt_tensor`, `_method_metadata`, record construction
- **Current responsibility:** Restores eval mode and emits loss/mask metadata.
- **Change:** Preserve the mode present before adaptation, set eval mode for the
  next score, and attach loss/mask/update fields to the already-created current
  batch record without changing its prediction.
- **Reason:** Reference adapter temporarily enters train mode and later scores
  with the updated model in eval mode.
- **Inputs:** Prior mode, update result and current batch records.
- **Outputs:** Auditable record and method metadata.
- **Errors:** Report mode-restoration failure and do not claim a successful update
  if optimizer step did not complete.
- **Dependencies:** Steps 3.4 and Phase 1 record order.
- **Compatibility:** Keep existing `loss_total`, `adaptation_mask_count` and
  `did_update` keys.
- **Verification:** Mode-restoration and record-timing tests.

### Tests

#### M2N2 masked update

- **Location:** `tests/online/test_online_method_fidelity.py`
- **Level:** Unit/integration.
- **Setup:** Deterministic RedLamp adapter model, Detrender state, batch with
  known reconstruction errors and threshold.
- **Action:** Score the batch, record prediction, adapt once and score the next
  batch.
- **Expected result:** Score is pre-update; mask and loss match reference;
  allowed parameters update; next batch can observe the update.
- **Edge cases:** All timesteps masked, no timesteps masked, one-window batch,
  non-finite reconstruction and invalid gamma.

### Verification

#### Automated

- [ ] `.venv/bin/pytest -q tests/online/test_online_method_fidelity.py` —
  Detrender, mask, loss, optimizer and causal tests pass.
- [ ] `.venv/bin/python -m compileall -q src tests` — changed Python files
  compile successfully.

#### Manual

- [ ] Review one M2N2 record and verify raw score/prediction precede update,
  while loss and mask diagnostics describe the update.

### Risks and recovery

- **Risk:** Batch mean and normalization order differ from checkpoint training.
- **Mitigation:** preserve the selected RedLamp contract and recalibrate after
  the full lifecycle is fixed.
- **Verification:** finite score/loss smoke and Detrender equation fixture.
- **Recovery:** retain the last validated adapter variant and mark failed
  threshold artifacts incomplete.

### Complete when

- M2N2 equations and one-step optimizer behavior pass deterministic tests.
- M2N2 records show pre-update scores and post-update diagnostics.
- The next batch, not the current batch, observes updated state.

## Phase 4: Correct CANDI FPM and SANA runtime parity

### Goal

Make CANDI use the reference representation and selection rules, preserve the
frozen/trainable boundary, and update separate candidate pools only after the
reference gate is satisfied.

### Dependencies

- Phase 1 batch-causal lifecycle.
- Phase 2 threshold and optimizer profile.
- Phase 0 decision about whether layer-level SANA parity is required. This
  detail plan selects the reference topology as the implementation target while
  retaining the RedLamp model identity.

### Detailed atomic steps

#### Step 4.1: Freeze the detector before creating trainable SANA modules

- **File:** `src/baselines/online/candi.py`
- **Symbol:** `CANDIStreamingBaseline._initialize_method_state`
- **Current responsibility:** Creates SANA modules, freezes the backbone when
  `candi_use_sana=True`, and creates AdamW.
- **Change:** Preserve the order `freeze detector -> create sana_in/sana_out ->
  mark SANA parameters trainable -> create selected optimizer`. Make the
  `USE_SANA=False` branch explicit and separate.
- **Reason:** Reference CANDI freezes pretrained model parameters when SANA is
  enabled and optimizes only SANA parameters.
- **Inputs:** `candi_use_sana`, model parameters and optimizer profile.
- **Outputs:** Frozen backbone, trainable SANA modules and valid optimizer groups.
- **Errors:** Reject an optimizer containing a frozen detector parameter or an
  empty SANA parameter group.
- **Dependencies:** Phase 2 optimizer construction.
- **Compatibility:** Preserve current method metadata and RedLamp checkpoint
  identity; do not unfreeze the detector in the SANA profile.
- **Verification:** Snapshot all parameter `requires_grad` flags and optimizer
  parameter IDs.

#### Step 4.2: Port the reference SANA temporal embedding

- **File:** `src/models/online_adapter_modules.py`
- **Symbol:** `SANA`, proposed internal temporal-embedding component
- **Current responsibility:** Uses one Conv1d/ReLU embedding per variable before a
  PyTorch `TransformerEncoder`.
- **Change:** Match the reference `TemporalEmbedding` behavior: input
  `[B,1,L]`, one temporal Conv1d stack per variable, mean over time to `[B,D]`,
  and variable-wise stacking to `[B,N,D]`.
- **Reason:** The reference SANA builds variable-specific temporal embeddings
  before cross-variable attention.
- **Inputs:** `[B,L,N]`, `d_model`, kernel size, dilation and dropout settings.
- **Outputs:** `[B,N,d_model]` embeddings with the same variable ordering.
- **Errors:** Reject wrong window length, feature count or incompatible model/head
  dimensions.
- **Dependencies:** Phase 0 SANA profile and existing SANA public forward API.
- **Compatibility:** Keep `SANA.forward(x)` input/output shape `[B,L,N]` so CANDI
  callers do not need a new tensor contract.
- **Verification:** Shape test and deterministic comparison against a small
  reference-equivalent fixture.

#### Step 4.3: Port the reference cross-variable encoder and projection

- **File:** `src/models/online_adapter_modules.py` and existing reference code
  under `bsc-thesis-ref-codebases/CANDI-main/layers/`
- **Symbol:** `SANA.forward`, internal encoder and variable projections
- **Current responsibility:** Uses `nn.TransformerEncoder`, then one linear
  projection per variable.
- **Change:** Implement the reference `Encoder`, `EncoderLayer`, attention and
  LayerNorm ordering used by `TCN_iTrans`, followed by one projection from
  `[B,d_model]` to `[B,L]` per variable.
- **Reason:** Exact runtime parity requires the reference cross-variable SANA
  topology, not only a name-compatible transformer.
- **Inputs:** `[B,N,d_model]`, SANA attention settings and window size.
- **Outputs:** `[B,L,N]` residual output before gating.
- **Errors:** Reject unsupported SANA type, invalid head divisibility or missing
  projection dimensions; fail rather than silently switching to Linear.
- **Dependencies:** Step 4.2 and reference layer definitions.
- **Compatibility:** Keep explicit `Linear` only when it is an intentional
  configured ablation; do not use it as the default reference profile.
- **Verification:** Module shape, finite-forward and parameter-name tests.

#### Step 4.4: Preserve reference gating semantics

- **File:** `src/models/online_adapter_modules.py`
- **Symbol:** `SANA.gating` and `SANA.forward`
- **Current responsibility:** Multiplies decoded residual by `tanh(gating)`.
- **Change:** Keep one trainable gate per variable, initialize it from the
  profile (`0.0`), and apply `output * tanh(gating)` after projection.
- **Reason:** Zero initialization makes the initial SANA residual zero while
  allowing later adaptation.
- **Inputs:** Decoded `[B,L,N]` output and gating vector `[N]`.
- **Outputs:** Gated residual with original shape.
- **Errors:** Reject gate/input feature mismatch or non-finite output.
- **Dependencies:** Steps 4.2-4.3.
- **Compatibility:** Preserve CANDI `x + sana_in(x)` and
  `reconstruction - sana_out(reconstruction)` paths.
- **Verification:** Zero-gate test returns zero residual; nonzero-gate test
  confirms feature-wise scaling.

#### Step 4.5: Build raw-input validation representations

- **File:** `src/baselines/online/candi.py`
- **Symbol:** `_calibration_complete`
- **Current responsibility:** Computes validation representations from raw
  validation windows and stores covariance, hard and moderate references.
- **Change:** Keep validation representation construction on raw windows and
  ensure it runs before any test adaptation. Compute covariance and
  `torch.linalg.pinv` exactly once per calibration.
- **Reason:** FPM reference statistics come from clean validation only and must
  not drift with SANA updates.
- **Inputs:** Clean validation windows and raw validation scores.
- **Outputs:** Validation representations, covariance inverse, hard references,
  moderate references and quartiles.
- **Errors:** Fail on fewer than two windows, no moderate references, wrong
  representation dimension or non-finite covariance inverse.
- **Dependencies:** Phase 1 calibration and model representation API.
- **Compatibility:** Keep validation data separate from test labels and test
  windows.
- **Verification:** Calibration fixture asserts references never change during
  test-stream updates.

#### Step 4.6: Match hard top-k and moderate reference sets

- **File:** `src/baselines/online/candi.py`
- **Symbol:** `_calibration_complete`
- **Current responsibility:** Selects `max(1, int(...))` hard references and
  Q1/Q3 moderate references.
- **Change:** Apply the selected reference top-k integer rule directly, use the
  reference anomaly ratio, and preserve strict `(q1 < score < q3)` moderate
  reference selection. Handle zero top-k according to the explicit profile
  instead of silently changing it with a local guard.
- **Reason:** The local guard changes behavior for small validation sets.
- **Inputs:** Validation score count, anomaly ratio and score distribution.
- **Outputs:** Hard and moderate representation banks with stable indices.
- **Errors:** Raise a profile/data error when the reference rule yields no valid
  hard or moderate bank; do not silently substitute a different rule.
- **Dependencies:** Step 4.5 and Phase 0 profile.
- **Compatibility:** Preserve the reference bank ordering used by Mahalanobis
  selection and diagnostics.
- **Verification:** Small-validation-set tests compare direct integer behavior;
  normal-size fixture checks expected bank sizes.

#### Step 4.7: Select candidates from raw current representations

- **File:** `src/baselines/online/candi.py`
- **Symbol:** `CANDIStreamingBaseline._collect_candidates`
- **Current responsibility:** Computes FPM representation from
  `self._candi_input(x)`, which includes `sana_in(x)`.
- **Change:** Compute `representation = backbone_.get_representations(x)` for
  hard and moderate Mahalanobis selection. Keep `_candi_input(x)` only inside
  reconstruction score/loss calculations.
- **Reason:** Reference FPM selection uses the current raw model input, not the
  SANA-transformed reconstruction input.
- **Inputs:** Raw current batch, raw score, threshold and fixed validation banks.
- **Outputs:** Hard/moderate boolean masks and selected raw windows.
- **Errors:** Reject representation shape mismatch or missing covariance/reference
  state.
- **Dependencies:** Steps 4.5-4.6 and Phase 1 score timing.
- **Compatibility:** Keep test labels out of masks and selection.
- **Verification:** Fixture changes SANA output while holding raw backbone
  representation fixed; selection must remain determined by raw representation.

#### Step 4.8: Apply exact hard/moderate score conditions

- **File:** `src/baselines/online/candi.py`
- **Symbol:** `_collect_candidates`
- **Current responsibility:** Uses hard `score > threshold` and moderate
  `score < threshold` plus Mahalanobis similarity.
- **Change:** Preserve these actual reference conditions, including the reference
  code's overwritten Q1/Q3 moderate score predicate. Keep chi-square percentile
  `0.05` and latent dimension as degrees of freedom.
- **Reason:** The executable reference uses `score < threshold` for moderate
  candidates after overwriting the earlier Q1/Q3 mask.
- **Inputs:** Raw batch scores, threshold, latent representations and covariance
  inverse.
- **Outputs:** Candidate masks without label dependence.
- **Errors:** Reject non-finite distances, invalid covariance or missing reference
  banks.
- **Dependencies:** Step 4.7.
- **Compatibility:** Keep `USE_FPM`, `USE_HARD` and `USE_MODERATE` explicit; do
  not map THESIS triage labels into these masks.
- **Verification:** Deterministic hard/moderate Mahalanobis fixtures with scores
  on both sides of threshold.

#### Step 4.9: Accumulate and gate separate candidate pools

- **File:** `src/baselines/online/candi.py`
- **Symbol:** `_hard_pool`, `_moderate_pool`, `_adapt_tensor`
- **Current responsibility:** Appends selected windows and updates each pool at
  `candi_min_samples`.
- **Change:** Append selected raw windows separately, update hard first and
  moderate second when each pool reaches `MIN_SAMPLES=16`, then clear only the
  pool that was updated. Report remaining pool sizes after the update.
- **Reason:** Reference CANDI has independent hard and moderate accumulation
  state and can perform both updates in one incoming batch.
- **Inputs:** Selected batch windows and current pool state.
- **Outputs:** Pool sizes, update count, loss and decision diagnostics.
- **Errors:** Reject empty-pool adaptation, failed concatenation or non-finite
  loss; preserve other pool state if one branch fails before its update.
- **Dependencies:** Steps 4.7-4.8 and Phase 2 optimizer.
- **Compatibility:** Preserve the existing record keys for pool diagnostics and
  keep update state local to CANDI.
- **Verification:** Pool fixture reaches 15 then 16 samples; both-pool fixture
  verifies hard-before-moderate order and independent reset.

#### Step 4.10: Compute SANA reconstruction loss on selected pools

- **File:** `src/baselines/online/candi.py`
- **Symbol:** `_candi_input`, `_candi_reconstruction`, `_adapt_pool`
- **Current responsibility:** Computes `x + sana_in(x)`, backbone reconstruction,
  subtracts `sana_out`, and uses full reconstruction MSE.
- **Change:** Preserve the reference reconstruction path and run one configured
  optimizer step on the concatenated selected pool. Do not apply M2N2 timestep
  masks or labels to the CANDI loss.
- **Reason:** CANDI selects samples before adaptation and optimizes full MSE.
- **Inputs:** Concatenated hard or moderate pool `[B,L,N]`.
- **Outputs:** Finite MSE, updated SANA parameters and cleared updated pool.
- **Errors:** Reject empty pool, wrong shape, non-finite residual or loss.
- **Dependencies:** Steps 4.4 and 4.9, Phase 2 optimizer.
- **Compatibility:** Keep the frozen backbone unchanged when `USE_SANA=True`.
- **Verification:** Parameter snapshot proves only SANA changes; loss equals
  `F.mse_loss(reconstruction, batch)` on a deterministic fixture.

#### Step 4.11: Keep optional label diagnostics outside adaptation

- **File:** `scripts/benchmarks/run_online_streaming_benchmark.py`,
  `src/baselines/online/candi.py` and report schema
- **Symbol:** runner label plumbing and CANDI metadata
- **Current responsibility:** Declares metrics-only label usage but does not
  reproduce reference pool anomaly counters.
- **Change:** If the corrected report requires reference pool statistics, pass
  labels only to a metrics collector indexed by absolute batch position. Keep
  labels out of representation, mask, loss and optimizer calls.
- **Reason:** Reference CANDI reads labels for statistics, not selection or loss.
- **Inputs:** Test labels and selected candidate indices after adaptation logic.
- **Outputs:** Optional hard/moderate anomaly-count diagnostics.
- **Errors:** Reject any code path where labels are read before selection or used
  to decide an update.
- **Dependencies:** Phase 1 record/index contract.
- **Compatibility:** Keep `test_label_usage: metrics_only` explicit; omit the
  counters if the active artifact schema does not require them.
- **Verification:** Label permutation test leaves selection, loss and optimizer
  state unchanged while changing only optional metric counters.

### Tests

#### CANDI FPM and pool lifecycle

- **Location:** `tests/online/test_online_method_fidelity.py`
- **Level:** Unit/integration.
- **Setup:** Deterministic validation bank, covariance inverse, hard/moderate
  references, SANA modules and two candidate pools.
- **Action:** Select candidates, cross pool gates, adapt and score the next batch.
- **Expected result:** Raw representation controls FPM; pools update separately;
  SANA-only parameters change; next batch sees updated SANA state.
- **Edge cases:** No moderate references, zero hard top-k, exactly 15/16 pool
  samples, both pools reaching the gate, labels permuted and `USE_FPM=False`.

### Verification

#### Automated

- [ ] `.venv/bin/pytest -q tests/online/test_online_method_fidelity.py` — FPM,
  SANA, pool and causal tests pass.
- [ ] `.venv/bin/python -m compileall -q src tests` — SANA and CANDI code compile.

#### Manual

- [ ] Inspect one CANDI trace showing raw representation source, hard/moderate
  pool sizes, update order, loss and next-batch score.

### Risks and recovery

- **Risk:** Porting exact SANA layers changes score distribution and CPU cost.
- **Mitigation:** compare module shapes/finite outputs first, then run one smoke
  before recalibrating matrix artifacts.
- **Verification:** deterministic SANA fixture and one method smoke.
- **Recovery:** retain the previous variant output root, mark failed artifacts
  incomplete and do not aggregate them with corrected results.

### Complete when

- FPM uses raw current representations.
- Hard/moderate pools and score predicates match reference behavior.
- SANA topology/scope and full MSE update pass tests.
- Labels affect only optional metrics diagnostics.

## Phase 5: Migrate configurations, tests and artifacts

### Goal

Regenerate corrected experiment inputs and prove that tests and artifacts
describe the same runtime profile without overwriting historical outputs.

### Dependencies

- Phases 0-4 implemented and focused tests passing.
- Corrected threshold and optimizer profile.
- Stable batch and method metadata contracts.

### Detailed atomic steps

#### Step 5.1: Update the configuration generator source

- **File:** `scripts/benchmarks/generate_online_streaming_benchmark_configs.py`
- **Symbol:** `BENCHMARK_METHOD_VARIANTS` and method baseline kwargs
- **Current responsibility:** Generates RedLamp adapter configs with q99 and
  method settings that may still select AdamW implicitly.
- **Change:** Emit explicit corrected profile fields: adapter variant, selected
  batch size, q99.5 protocol, optimizer, learning rate/weight decay and
  method-native M2N2/CANDI/SANA settings.
- **Reason:** Generated YAML is the source of truth for benchmark runs.
- **Inputs:** Phase 0 profile and constructor schemas.
- **Outputs:** Deterministic main/smoke config mappings for both methods.
- **Errors:** Fail generation or validation when a method lacks checkpoint,
  optimizer, threshold or method-native settings.
- **Dependencies:** Phases 0-4.
- **Compatibility:** Preserve entity, seed, window size, absolute range and
  metrics-only policy unless the corrected profile explicitly changes batch size.
- **Verification:** Generator count, schema and identity tests.

#### Step 5.2: Generate corrected main and smoke YAML files

- **File:** `configs/experiment/online_benchmark/m2n2/` and
  `configs/experiment/online_benchmark/candi/`
- **Symbol:** generated corrected config files
- **Current responsibility:** Contains corrected and legacy generations together.
- **Change:** Generate corrected files under the explicit
  `reference_adapter_redlamp_encoder` identity and preserve legacy files as
  historical/non-comparable inputs.
- **Reason:** A benchmark operator must be able to select the corrected profile
  without guessing from a legacy filename.
- **Inputs:** Updated generator and checkpoint inventory.
- **Outputs:** Valid YAML files for one entity/seed smoke and official main runs.
- **Errors:** Reject missing paths, stale `online_variant`, q99 protocol or
  approximation-only kwargs in corrected files.
- **Dependencies:** Step 5.1.
- **Compatibility:** Do not delete or overwrite old output roots.
- **Verification:** Load every generated YAML with the project's existing config
  validation path and compare it with generator output.

#### Step 5.3: Add call-order and profile regression tests

- **File:** `tests/online/test_online_method_fidelity.py` and
  `tests/online/test_online_streaming_baseline_contracts.py`
- **Symbol:** method-fidelity, shared lifecycle and metadata tests
- **Current responsibility:** Proves core adapter mechanics, but not all strict
  runtime-flow differences.
- **Change:** Add assertions for prediction-before-adapt, one update per batch,
  next-batch state visibility, q99.5 threshold, optimizer class, raw CANDI
  representation and legacy config rejection.
- **Reason:** Passing tests must cover the remaining audit findings.
- **Inputs:** Deterministic model/method fixtures and corrected configs.
- **Outputs:** Focused failures that identify one violated contract.
- **Errors:** Test should fail if update occurs before record, if any current-batch
  item sees an update, or if labels influence adaptation.
- **Dependencies:** Phases 1-4.
- **Compatibility:** Keep shared record/index tests separate from method-specific
  fidelity tests.
- **Verification:** Focused pytest command and full `tests/online` suite later.

#### Step 5.4: Recalibrate threshold artifacts after the final flow

- **File:** benchmark threshold artifact output and runner report generation
- **Symbol:** `build_threshold_artifact`, calibration result and report metadata
- **Current responsibility:** Writes artifacts for the current q99 adapter score
  protocol.
- **Change:** Recompute clean-validation threshold artifacts only after the
  corrected batch, score, optimizer and model flow passes focused tests. Record
  q99.5, optimizer, batch size, checkpoint SHA-256, SANA identity and selected
  range.
- **Reason:** Thresholds are distribution-dependent and cannot be reused after
  score/update changes.
- **Inputs:** Corrected calibration scores, profile and checkpoint identity.
- **Outputs:** New threshold JSON, report provenance and comparability identity.
- **Errors:** Reject non-finite values, wrong quantile, missing checkpoint or
  profile mismatch.
- **Dependencies:** Steps 5.1-5.3 and one passing smoke prerequisite.
- **Compatibility:** Keep historical q99 artifacts unchanged and non-comparable.
- **Verification:** Artifact schema, finite-value, identity and threshold-source
  tests.

#### Step 5.5: Update normative documentation

- **File:** `documents/spec/online_benchmark_contract.md` and the research report
- **Symbol:** runtime profile, method metadata, migration and verification
  sections
- **Current responsibility:** Documents the adapter variant and earlier
  implementation evidence.
- **Change:** Record the corrected profile, exact remaining/non-equivalence
  boundary, generated config identity and test/smoke status.
- **Reason:** Documentation is the project source of truth for later benchmark
  interpretation.
- **Inputs:** Corrected code behavior, configs and artifacts.
- **Outputs:** Consistent contract and research evidence.
- **Errors:** Do not mark exact fidelity before Phase 6 acceptance passes.
- **Dependencies:** Steps 5.1-5.4.
- **Compatibility:** Preserve historical audit sections but label them as
  pre-implementation evidence.
- **Verification:** Manual cross-check of spec, config, metadata and report.

### Tests

#### Configuration and artifact migration

- **Location:** existing generator/config/artifact tests and online method tests.
- **Level:** Contract/integration.
- **Setup:** Corrected and legacy config fixtures plus one corrected threshold
  artifact.
- **Action:** Generate, load, validate and compare identities.
- **Expected result:** Corrected artifacts pass; mixed-generation aggregation is
  rejected or explicitly marked non-comparable.
- **Edge cases:** Missing checkpoint, stale q99 protocol, stale optimizer,
  legacy variant and incomplete threshold metadata.

### Verification

#### Automated

- [ ] `.venv/bin/python -m scripts.benchmarks.generate_online_streaming_benchmark_configs --print-count`
  — generator completes with corrected method entries.
- [ ] Existing config/artifact tests — generated files load and identity checks
  pass.
- [ ] `.venv/bin/pytest -q tests/online/test_online_method_fidelity.py tests/online/test_online_streaming_baseline_contracts.py`
  — focused suite passes.
- [ ] `git diff --check` — no whitespace errors in the plan's implementation
  changes or generated documentation updates.

#### Manual

- [ ] Compare one corrected M2N2 and one corrected CANDI YAML with their
  threshold artifact and report metadata.
- [ ] Confirm old and corrected output roots are not mixed in reports.

### Risks and recovery

- **Risk:** Generator changes leave stale fields in hand-edited YAML files.
- **Mitigation:** regenerate from source and validate every corrected config.
- **Verification:** compare generated content with a fresh generator run.
- **Recovery:** restore historical configs from their existing files and rerun
  generation into a new corrected identity.

### Complete when

- Corrected YAML, metadata and artifacts agree.
- Focused regression tests cover every follow-up audit mismatch.
- Historical outputs remain intact and non-comparable.

## Phase 6: Validate one real end-to-end run and approve expansion

### Goal

Prove the corrected runtime through the actual benchmark entry point before
running a full matrix.

### Dependencies

- Phases 0-5 complete.
- One corrected M2N2 smoke config and one corrected CANDI smoke config.
- Required RedLamp checkpoints available at the configured paths.

### Detailed atomic steps

#### Step 6.1: Run the focused tests before smoke execution

- **File:** online test suite and corrected configs
- **Symbol:** method-fidelity, lifecycle, config and artifact tests
- **Current responsibility:** Existing tests establish basic adapter behavior.
- **Change:** Run the focused tests after all corrected runtime code and configs
  are present; do not start a smoke with a failing fidelity test.
- **Reason:** A real smoke can pass shape checks while using the wrong method.
- **Inputs:** Corrected source, tests and YAML.
- **Outputs:** Passing focused test evidence.
- **Errors:** Stop and return to the responsible phase on any failure.
- **Dependencies:** Phase 5.
- **Compatibility:** Do not modify output artifacts during this check.
- **Verification:** Use the focused `.venv/bin/pytest -q` commands already defined
  in Phases 0-5.

#### Step 6.2: Run one corrected M2N2 smoke

- **File:** corrected M2N2 smoke YAML and
  `scripts/benchmarks/run_online_streaming_benchmark.py`
- **Symbol:** `run_online_streaming_benchmark`
- **Current responsibility:** Executes the existing online benchmark entry path.
- **Change:** Run one real M2N2 entity/seed through the corrected profile and
  retain its report, records, threshold artifact and selected-range metadata.
- **Reason:** This verifies loader-to-calibration-to-causal-adaptation flow.
- **Inputs:** Real RedLamp checkpoint, dataset config, corrected protocol and
  smoke range.
- **Outputs:** Completed benchmark report with finite score/loss and provenance.
- **Errors:** Stop on missing checkpoint, profile mismatch, non-finite value,
  incomplete range or unexpected update order.
- **Dependencies:** Step 6.1.
- **Compatibility:** Write into a new corrected smoke output root.
- **Verification:** Inspect report status, threshold profile, records and M2N2
  mask/loss diagnostics.

#### Step 6.3: Run one corrected CANDI smoke

- **File:** corrected CANDI smoke YAML and the same benchmark entry point
- **Symbol:** `run_online_streaming_benchmark`
- **Current responsibility:** Executes CANDI with configured FPM/SANA state.
- **Change:** Run one real CANDI entity/seed through the corrected profile and
  retain validation-reference, pool, SANA and threshold provenance.
- **Reason:** This verifies actual FPM selection and pool-gated adaptation.
- **Inputs:** Real RedLamp checkpoint, clean validation data and corrected CANDI
  settings.
- **Outputs:** Completed report with finite scores/losses and candidate/update
  diagnostics.
- **Errors:** Stop on empty reference bank, wrong candidate representation,
  frozen-parameter mutation, non-finite loss or incomplete range.
- **Dependencies:** Step 6.1 and corrected SANA implementation.
- **Compatibility:** Write into a new corrected CANDI smoke output root.
- **Verification:** Inspect first two batches, pool gate transitions, SANA
  parameter scope and next-batch score.

#### Step 6.4: Verify report and artifact acceptance conditions

- **File:** generated `online_records.json`, threshold artifact and benchmark
  report for both smoke runs
- **Symbol:** method metadata, record order and stream-selection metadata
- **Current responsibility:** Stores benchmark outputs and provenance.
- **Change:** Check that each record contains pre-update score/prediction,
  correct batch/stream indices, update diagnostics and profile identity.
- **Reason:** The final claim depends on observable artifacts, not only process
  completion.
- **Inputs:** Two smoke artifact bundles.
- **Outputs:** Manual acceptance checklist with pass/fail evidence.
- **Errors:** Mark run provisional/incomplete when any identity, finite-value,
  batch-order or label-isolation check fails.
- **Dependencies:** Steps 6.2-6.3.
- **Compatibility:** Do not rewrite failed artifacts; retain them for diagnosis.
- **Verification:** Compare the first two batches against the expected reference
  trace `score -> prediction/record -> adapt -> next batch`.

#### Step 6.5: Run the full relevant online test suite

- **File:** `tests/online/`
- **Symbol:** full online test collection
- **Current responsibility:** Covers stream, contracts, artifacts and method
  behavior across the online package.
- **Change:** Run the full suite only after both corrected smokes pass.
- **Reason:** Full-suite results should correspond to the corrected runtime
  generation, not an unverified intermediate state.
- **Inputs:** Corrected source and test suite.
- **Outputs:** Full test result and warnings recorded in the research log.
- **Errors:** Distinguish pre-existing warnings from failures; do not approve
  matrix expansion on a failing test.
- **Dependencies:** Step 6.4.
- **Compatibility:** Keep unrelated tests and historical artifacts unchanged.
- **Verification:** `.venv/bin/pytest -q tests/online` completes successfully.

#### Step 6.6: Decide whether to approve matrix expansion

- **File:** research report and benchmark approval record
- **Symbol:** final verification and matrix gate
- **Current responsibility:** Keeps full matrix execution blocked until evidence
  is reviewed.
- **Change:** Mark matrix approval only when profile, focused tests, both smokes,
  full online tests and artifact review all pass.
- **Reason:** A full matrix multiplies any remaining method-flow error.
- **Inputs:** Phase 6 test results, smoke reports and manual checklist.
- **Outputs:** Explicit approved or blocked status with reasons.
- **Errors:** Keep status blocked/provisional and return to the responsible phase
  when evidence conflicts.
- **Dependencies:** Steps 6.1-6.5.
- **Compatibility:** Never aggregate provisional and approved generations.
- **Verification:** Manual sign-off against the final verification checklist.

### Tests

#### End-to-end corrected profile

- **Location:** benchmark smoke outputs and `tests/online/`.
- **Level:** End-to-end plus manual artifact review.
- **Setup:** One corrected M2N2 and one corrected CANDI config with real
  checkpoint paths.
- **Action:** Run each through the real entry point, then inspect first two
  batches and final artifacts.
- **Expected result:** Both complete with finite values, correct profile identity,
  pre-update records and next-batch feedback.
- **Edge cases:** Missing checkpoint, empty CANDI moderate bank, incomplete
  final batch, non-finite loss and profile mismatch.

### Verification

#### Automated

- [ ] Focused method/lifecycle/config tests pass.
- [ ] One corrected M2N2 smoke completes successfully.
- [ ] One corrected CANDI smoke completes successfully.
- [ ] `.venv/bin/pytest -q tests/online` — full relevant online suite passes.

#### Manual

- [ ] Review the first two batches of both smoke reports.
- [ ] Confirm threshold, optimizer, checkpoint, batch, selected range and label
  provenance fields.
- [ ] Approve matrix expansion only after all evidence is consistent.

### Risks and recovery

- **Risk:** A shape-valid smoke hides a wrong update order or score source.
- **Mitigation:** Require artifact-level order evidence and focused tests before
  approval.
- **Verification:** Manual first-two-batch trace plus automated call-order test.
- **Recovery:** Preserve failed outputs, mark them provisional and return to the
  phase identified by the failure.

### Complete when

- Both corrected smokes complete with finite values and complete provenance.
- Full online tests pass.
- Manual review confirms the exact selected runtime flow.
- Matrix approval is explicitly recorded.

## Interface and data changes

### Baseline lifecycle interface

- The external runner-facing `calibrate()` and `run_sequence()` names remain
  stable.
- The internal adaptive contract changes from one-window update calls to one
  update per `[B,L,D]` test batch.
- Each batch produces one record per input window, preserving absolute indices.
- Method update diagnostics remain separate from prediction fields so the
  pre-update prediction cannot be overwritten by post-update state.

### Configuration interface

- Corrected configs require explicit profile identity, threshold quantile/ratio,
  optimizer method and batch size.
- M2N2 requires gamma and one-step settings.
- CANDI requires FPM/SANA, hard/moderate, minimum-sample and SANA settings.
- Checkpoint role remains `pretrained_encoder` for the RedLamp variant.

### Artifact compatibility

- Existing q99/AdamW and legacy approximation artifacts remain readable but are
  non-comparable to corrected profile artifacts.
- Corrected reports must carry profile, checkpoint, threshold, optimizer, batch,
  SANA and selected-range identity.
- No broad deletion or overwrite is part of this plan.

## Deployment and rollout

1. Update the contract/profile and generator definitions.
2. Implement and test the common lifecycle.
3. Implement threshold/optimizer and method corrections.
4. Regenerate corrected configs and artifacts in new output roots.
5. Run focused tests, then one M2N2 and one CANDI smoke.
6. Run the full online suite and record manual acceptance.
7. Expand to the full matrix only after explicit approval.

Mixed-version runs must remain separated by configuration/profile and output
identity. If a phase fails, retain its diagnostics and roll back by selecting the
last validated profile, not by deleting historical outputs.

## Documentation changes

- Update `documents/spec/online_benchmark_contract.md` with the corrected
  runtime profile and RedLamp-variant boundary.
- Update the research report with implementation evidence only after each phase
  passes its gate.
- Record threshold source, optimizer, batch size, SANA identity, checkpoint role
  and comparability identity in benchmark reports.

## Final verification

- [ ] Phase 0 profile and migration identity are explicit.
- [ ] Phase 1 proves `score -> prediction/record -> adapt -> next batch`.
- [ ] Phase 2 proves q99.5 and selected optimizer semantics.
- [ ] Phase 3 proves M2N2 Detrender, mask, loss and update behavior.
- [ ] Phase 4 proves CANDI raw FPM, pool gates, SANA and causal behavior.
- [ ] Phase 5 proves generated config and artifact consistency.
- [ ] Phase 6 proves both real smokes, full online tests and manual acceptance.
- [ ] Only corrected artifacts are considered for matrix expansion.

## Assumptions and non-blocking uncertainties

- The RedLamp encoder checkpoint remains the selected model surface; native
  MLP/TimesNet equivalence is outside this implementation boundary.
- This detail plan targets the reference SANA topology. If implementation
  evidence later proves that the current SANA module must remain, the result
  identity must stay variant-level and the structure must be reviewed before
  changing the fidelity claim.
- The current project may retain an official `batch_size=1` benchmark profile.
  The internal lifecycle must still preserve reference batch semantics, and the
  selected profile must be recorded explicitly.
