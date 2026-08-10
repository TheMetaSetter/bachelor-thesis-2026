---
date: 2026-08-10 Asia/Ho_Chi_Minh
researcher: OpenAI Codex
topic: "Xác định các đoạn code cần điều chỉnh để M2N2 và CANDI đúng phương pháp"
status: corrected_runtime_profile_smoke_verified_matrix_pending
revision: aebf99382af8508f2dd0f809a531a1bd72dba4c1
branch: dev
---

# Research: Xác định các đoạn code cần điều chỉnh để M2N2 và CANDI đúng phương pháp

## Summary

Implementation cũ chưa faithful với M2N2 và CANDI trong
`bsc-thesis-ref-codebases/CANDI-main`. Runtime mới đã chuyển sang adapter có
Detrender, pseudo-anomaly mask, FPM hard/moderate pools, minimum-sample gate và
SANA MSE. Tuy nhiên, audit runtime hiện tại cho thấy flow vẫn chưa tương đương
đầy đủ với Predictor reference.

Khoảng cách này không chỉ nằm ở hai điều kiện `_should_update`. Muốn claim
M2N2/CANDI đúng phương pháp, cần điều chỉnh model/checkpoint contract,
adaptation state, thứ tự score-update trong stream, configuration và tests.

Có một khác biệt contract đã được ghi rõ: repo hiện có RedLamp simple 1D-CNN
encoder checkpoints nhưng không có native MLP/TimesNet checkpoints của
reference codebase. Vì vậy target được đặt tên
`reference_adapter_redlamp_encoder`; đây là implementation áp dụng các cơ chế
adapter M2N2/CANDI trên encoder RedLamp, không phải native MLP/TimesNet. Không
được gọi target này là exact reference implementation cho đến khi các lệch
runtime trong follow-up audit được xử lý.

## Research question

Đọc `prompts/1_research_prompt.md` và xác định các đoạn code cần điều chỉnh để
M2N2 và CANDI được lập trình chính xác về mặt phương pháp theo
`bsc-thesis-ref-codebases/CANDI-main/tta/m2n2/` và
`bsc-thesis-ref-codebases/CANDI-main/tta/candi/`.

Phần đầu của report là research audit. Implementation record ở cuối report ghi
các thay đổi đã thực hiện, test và smoke evidence.

## System context

Entry point của hai baseline là `scripts/benchmarks/run_online_streaming_benchmark.py`.
Runner load train/validation/test sequences, khởi tạo baseline bằng train
sequence, calibrate trên clean validation, cắt test stream theo absolute range,
rồi gọi `baseline.run_sequence()`.

Sau implementation, `CANDIStreamingBaseline` và `M2N2StreamingBaseline` đã có
state và update logic riêng. `AdaptiveStreamingBaselineBase` vẫn sở hữu model
loading, validation calibration, window traversal và record serialization.

THESIS online runtime là một path khác dưới `src/engine/online_tta/`. Baseline
không được nhận THESIS Gumbel sampling, four-region triage, PNN verification,
projector update hoặc uncertainty ablation. Đây là ranh giới cần giữ khi sửa
baseline.

## Execution path

```text
benchmark YAML
  -> run_online_streaming_benchmark.run_online_streaming_benchmark
  -> build_dataset
  -> train sequence + clean validation + test sequence
  -> CANDIStreamingBaseline / M2N2StreamingBaseline
  -> load RedLamp encoder checkpoint + adapter-owned state
  -> calibrate(clean validation)
       -> batched stride-1 validation windows
       -> raw-window scores
       -> EWMA for reporting/artifact metadata
       -> q99.5 raw threshold for the reference adapter profile
  -> run_sequence(test stream)
       -> score the whole batch
       -> compute prediction and append records from pre-update scores
       -> adapt once for the whole batch
       -> continue to the next batch
```

The current test path is sequential and therefore an update can affect the next
window. It is configured with one window per batch, not with the general
reference test-batch contract. The source call order is currently
`score -> EWMA -> adapt -> prediction/record`; the reference call order is
`score -> prediction/record -> adapt`.

The reference predictor processes one test batch, records its score, calls
`adapter.adapt(inputs, scores)`, and only then processes the next batch. This
causal score-update order is part of the method behavior.

## Historical pre-implementation audit

The following sections describe the implementation state before the
encoder-checkpoint adapter was added. They are retained as historical evidence;
the current-state conclusions are in the follow-up audit near the end of this
report.

## Detailed findings

### 1. Current shared adaptive implementation

Implemented behavior:

- `fit()` computes per-feature train mean and standard deviation.
- `_fit_backbone()` creates `SimpleWindowCnnAutoencoder`.
- The loader reads only `encoder.*` from the RedLamp checkpoint.
- The encoder is frozen.
- `_update_reference()` updates both mean and standard deviation with a custom
  EMA.
- `run_sequence()` emits `loss_total: None`; it never performs optimizer,
  gradient, or parameter updates.

Evidence:

- `src/baselines/online/adaptive.py:122-161` — fit, backbone creation,
  checkpoint loading and encoder freezing.
- `src/baselines/online/adaptive.py:230-241` — custom mean/std reference update.
- `src/baselines/online/adaptive.py:370-459` — current score, triage, update and
  record flow.
- `src/models/simple_window_cnn_autoencoder.py:25-35` — decoder initialization
  and architecture.
- `src/baselines/online/redlamp_encoder_checkpoint.py:90-113` — encoder-only
  checkpoint loading.

Adjustment surface:

- `src/baselines/online/adaptive.py:60-461` is not a faithful shared core for
  both methods. Shared window traversal and record serialization can remain,
  but method-specific normalization, trainable parameters, sample selection,
  loss and update state must move into method-owned logic.
- `src/models/simple_window_cnn_autoencoder.py:11-40` and
  `src/baselines/online/redlamp_encoder_checkpoint.py:43-105` must be reviewed
  together with the checkpoint decision. The current code initializes a fresh
  decoder but loads only the encoder. The available evidence does not show a
  trained M2N2/CANDI decoder or native MLP/TimesNet checkpoint compatible with
  the reference adapter.
- `src/baselines/online/adaptive.py:382-426` must change from whole-sequence
  precomputation to score-current-window, adapt-current-state, then continue
  with the next window/batch.

### 2. M2N2 reference versus current code

Reference behavior:

1. Attach `Detrender(num_features, gamma)` to the model and enable model
   normalization.
2. For each incoming batch, update the Detrender statistics.
3. Forward the model and calculate timestep reconstruction error
   `A = mean((recon - x)^2, dim=-1)`.
4. Create pseudo-anomaly labels with `A >= threshold`.
5. Mask those positions out of the adaptation loss.
6. Backpropagate the remaining reconstruction loss and perform one optimizer
   step. The adapter factory explicitly sets M2N2 TTA steps to one.

Evidence:

- `bsc-thesis-ref-codebases/CANDI-main/tta/m2n2/adapter_m2n2.py:13-39` — M2N2
  adapter setup and update objective.
- `bsc-thesis-ref-codebases/CANDI-main/models/normalizer.py:7-43` — Detrender
  state, mean-only normalization and EMA statistic update.
- `bsc-thesis-ref-codebases/CANDI-main/tta/adapter.py:5-12,39-42` — optimizer
  construction and M2N2 step count.

Current behavior:

- `src/baselines/online/m2n2.py:51-62` only checks whether raw and EWMA scores
  are below the threshold.
- `src/baselines/online/adaptive.py:230-241` updates feature mean and standard
  deviation, not a reference Detrender mean used by the model.
- `src/baselines/online/adaptive.py:431-459` writes `loss_total: None`; no
  reconstruction loss, timestep mask, backward pass or optimizer step exists.
- `src/baselines/online/adaptive.py:159-161` freezes the encoder, so the
  current path cannot reproduce the reference model adaptation surface.

Required adjustment areas:

- `src/baselines/online/m2n2.py`: replace the predicate-only policy with a
  method-owned M2N2 adapter state containing Detrender state, thresholded
  timestep mask, reconstruction loss and optimizer.
- `src/baselines/online/adaptive.py`: expose a per-window/per-batch forward and
  adapt lifecycle, or stop using this shared base for M2N2.
- `src/models/`: provide the model surface that the selected M2N2 checkpoint
  actually initializes. The reference adapter supports MLP and TimesNet; the
  current RedLamp encoder-only checkpoint does not establish that contract.
- `scripts/benchmarks/run_online_streaming_benchmark.py`: pass the current
  window/batch to M2N2, collect its score before adaptation, then let the next
  window use updated model state.
- Configuration must expose M2N2-native `gamma`, optimizer, learning rate,
  trainable parameter scope and adaptation step count. The current
  `adaptation_momentum` is not a substitute for M2N2 `gamma` plus optimizer
  learning rate.

### 3. CANDI reference versus current code

Reference behavior:

- CANDI optionally freezes the pretrained detector and adds trainable `sana_in`
  and `sana_out` residual modules.
- It computes validation representations and their covariance inverse.
- It stores top-k validation representations as the hard reference set and
  Q1-Q3 validation representations as the moderate reference set.
- For each test batch, it computes current representations and anomaly scores.
- Hard candidates require both high anomaly score and small Mahalanobis distance
  to the hard reference set.
- Moderate candidates require a non-anomalous score and small Mahalanobis
  distance to the moderate reference set.
- It accumulates candidate samples separately and adapts only when each pool
  reaches `MIN_SAMPLES`.
- Adaptation uses reconstruction MSE through SANA and optimizer steps.

Evidence:

- `bsc-thesis-ref-codebases/CANDI-main/tta/candi/adapter_candi.py:47-130` —
  SANA module.
- `bsc-thesis-ref-codebases/CANDI-main/tta/candi/adapter_candi.py:133-180` —
  trainable surface and validation reference statistics.
- `bsc-thesis-ref-codebases/CANDI-main/tta/candi/adapter_candi.py:192-299` —
  hard/moderate FPM selection and accumulation.
- `bsc-thesis-ref-codebases/CANDI-main/tta/candi/adapter_candi.py:301-367` —
  minimum-sample update and reconstruction-MSE objective.
- `bsc-thesis-ref-codebases/CANDI-main/predictor.py:83-91` — causal
  score-before-adapt loop.
- `bsc-thesis-ref-codebases/CANDI-main/config.py:77-95` — reference defaults
  for FPM, SANA, hard/moderate selection, minimum samples and optimizer.
- `bsc-thesis-ref-codebases/CANDI-main/README.md:8-10` — CANDI's stated
  false-positive reference set and lightweight residual adaptation.

Current behavior:

- `src/baselines/online/candi.py:57-66` updates on the repository's
  `gray_zone` or `pnn_candidate` triage labels.
- The current CANDI path has no validation representation bank, covariance,
  Mahalanobis distance, top-k false-positive set, moderate reference set, SANA,
  candidate pools, minimum-sample gate, reconstruction loss or optimizer.
- `src/engine/online_tta/triage.py:44-64` defines the legacy baseline triage
  labels used by current CANDI. Those labels are not the CANDI reference FPM
  selection algorithm.
- `src/baselines/online/adaptive.py:422-426` updates reference statistics
  immediately for one window rather than accumulating candidate batches.

Required adjustment areas:

- `src/baselines/online/candi.py`: replace the triage predicate with CANDI
  adapter state and method-native hard/moderate selection.
- Add a CANDI-owned representation/reference-statistics component for validation
  representations, covariance pseudoinverse, top-k hard reference set and
  moderate reference set. Its inputs must come from clean validation only.
- Add SANA input/output residual modules and an optimizer. The pretrained model
  must remain frozen when the selected CANDI configuration uses SANA.
- Add separate hard and moderate accumulation buffers, `MIN_SAMPLES` gate,
  reset behavior after each update, and `loss_total` reporting from the actual
  MSE objective.
- Change the runner/stream lifecycle so scores are emitted before the current
  adaptation and updated parameters affect only subsequent windows/batches.
- Replace the current `gray_zone`/`pnn_candidate` configuration surface with
  explicit CANDI-native settings. Do not reuse THESIS four-region semantics as
  an undocumented alias.

### 4. Model and checkpoint contract conflict

The active repository contract says that both M2N2 and CANDI use the same
RedLamp CNN encoder checkpoint with latent dimension 128 and that runtime keeps
the baseline decoder. It also says that the checkpoint is selected to avoid
retraining each combination.

Evidence:

- `documents/spec/online_benchmark_contract.md:14-24` — RedLamp encoder
  decision.
- `documents/spec/online_benchmark_contract.md:191-241` — checkpoint path,
  architecture and encoder-only loading contract.
- `documents/spec/online_benchmark_contract.md:243-264` — current baseline
  update policy contract.
- `scripts/benchmarks/generate_online_streaming_benchmark_configs.py:98-133` —
  generated M2N2/CANDI kwargs.
- `configs/experiment/online_benchmark/m2n2/smd__m2n2__online_main__machine_3_4__w20__seed6__main.yaml:11-23`
  and the corresponding CANDI config — active RedLamp checkpoint settings.

The reference code instead constructs adapters around MLP or TimesNet models;
the adapter factory selects those model-specific classes. Evidence:

- `bsc-thesis-ref-codebases/CANDI-main/tta/adapter.py:37-51` — reference
  adapter/model selection.
- `bsc-thesis-ref-codebases/CANDI-main/models/mlp/modeling_mlp.py:8-35` —
  reference MLP reconstruction model.

This is a specification conflict, not a small missing function. Before coding,
the project must choose and document one of these meanings:

1. **Reference-faithful baselines:** use model/checkpoint artifacts compatible
   with the reference M2N2/CANDI adapter mechanisms and revise the current
   RedLamp-only contract accordingly.
2. **RedLamp-based repository variants:** retain the RedLamp backbone contract,
   but name the methods as repository-specific M2N2-style/CANDI-style variants
   and document the methodological deviations. They must not be reported as
   exact reference implementations.

The available files do not establish that the existing RedLamp encoder
checkpoint can initialize the reference MLP/TimesNet adapter without changing
the model contract.

### 5. Runner and stream lifecycle

The current runner is structurally correct for loading splits, selecting the
absolute test range and invoking the baseline protocol, but it assumes that a
baseline can score a complete `T x D` sequence in one call.

Evidence:

- `scripts/benchmarks/run_online_streaming_benchmark.py:297-324` — dataset
  loading and baseline construction.
- `scripts/benchmarks/run_online_streaming_benchmark.py:335-339` — calibration.
- `scripts/benchmarks/run_online_streaming_benchmark.py:364-404` — test range
  selection and one `run_sequence()` call per selected sequence.
- `src/baselines/online/base.py:80-98` — an online batcher helper exists, but
  the M2N2/CANDI `run_sequence()` path does not use it.

Adjustment areas:

- `src/baselines/online/base.py:60-77`: extend the protocol if it must expose
  `initialize_from_validation()`, `score_window()`, `adapt_window()` and
  state serialization. Do not force CANDI's candidate-pool state into a scalar
  `did_update` boolean only.
- `scripts/benchmarks/run_online_streaming_benchmark.py:360-405`: use a
  sequential window/batch lifecycle for adaptive baselines while preserving the
  same absolute stream range and threshold split.
- `src/data/stream.py:38-177` and `build_stride1_batcher()` can provide the
  sequential window contract, but the code must verify that window shape,
  batch size and stream cursor match the reference adapter's expected
  `[B, L, D]` inputs.

### 6. Configuration and generated artifacts

The current generator exposes `adaptation_momentum`, CNN dimensions and a
RedLamp encoder checkpoint for both methods. Those fields describe the current
approximation, not the reference algorithms.

Adjustment areas:

- `scripts/benchmarks/generate_online_streaming_benchmark_configs.py:98-133` —
  replace or extend kwargs with method-native M2N2 and CANDI settings after the
  model/checkpoint decision.
- `configs/experiment/online_benchmark/m2n2/` and
  `configs/experiment/online_benchmark/candi/` — regenerate official main and
  smoke configs from the updated generator; do not hand-edit generated files as
  the source of truth.
- `documents/spec/online_benchmark_contract.md:191-264` — update the locked
  checkpoint, score and update contract if reference-faithful implementation is
  selected.
- Threshold artifacts must be recalibrated after changing the score model or
  inference protocol. Existing thresholds cannot be reused silently because the
  score distribution and trainable surface will change.

### 7. Tests that currently pass without proving method fidelity

`tests/online/test_online_streaming_baseline_contracts.py:71-146` checks that
both classes can instantiate, calibrate, emit records and expose metadata. It
does not prove:

- M2N2 Detrender semantics;
- M2N2 timestep masking and optimizer update;
- CANDI validation FPM statistics;
- Mahalanobis hard/moderate selection;
- SANA trainable parameter scope;
- minimum-sample buffering;
- score-before-update causal feedback;
- the actual reconstruction loss value.

`tests/online/test_redlamp_encoder_checkpoint.py:25-48` explicitly locks
encoder-only checkpoint loading. This test is valid for the current RedLamp
contract, but it must be revised or separated if reference-faithful native
M2N2/CANDI checkpoints replace that contract.

Adjustment areas:

- Extend or split `tests/online/test_online_streaming_baseline_contracts.py`
  into shared stream-contract tests and method-fidelity tests.
- Add M2N2 tests for Detrender update, pseudo-anomaly mask, masked loss,
  parameter change and one-step causal effect.
- Add CANDI tests for validation-only reference construction, hard/moderate
  selection, candidate accumulation, `MIN_SAMPLES`, SANA parameter scope,
  optimizer update and next-window feedback.
- Add a regression test proving that no test labels enter scoring, selection or
  adaptation. Labels may remain metrics-only.

## Evidence

- `prompts/1_research_prompt.md:30-41` — research-only scope, evidence rules
  and prohibition on presenting intended behavior as implementation.
- `prompts/1_research_prompt.md:106-136` — execution tracing and evidence
  classification requirements.
- `prompts/1_research_prompt.md:245-273` — required research report location
  and structure.
- `src/baselines/online/adaptive.py:122-161` — current backbone and frozen
  encoder path.
- `src/baselines/online/adaptive.py:370-459` — current adaptive runtime.
- `src/baselines/online/m2n2.py:51-75` — current M2N2 policy.
- `src/baselines/online/candi.py:57-79` — current CANDI policy.
- `bsc-thesis-ref-codebases/CANDI-main/tta/m2n2/adapter_m2n2.py:13-39` —
  reference M2N2 update.
- `bsc-thesis-ref-codebases/CANDI-main/tta/candi/adapter_candi.py:133-367` —
  reference CANDI setup, selection and update.
- `bsc-thesis-ref-codebases/CANDI-main/predictor.py:83-91` — reference causal
  score/adapt order.
- `documents/spec/full-spec-v3.md:1183-1194` — baseline-native protocol
  boundary.
- `documents/spec/full-spec-v3.md:1405-1425` — fairness and leakage rules.

## Configuration observed

| Setting | Active value | Evidence | Scope |
| --- | --- | --- | --- |
| `baseline_name` | `m2n2` or `candi` | `configs/experiment/online_benchmark/m2n2/...yaml:1-3`, `configs/experiment/online_benchmark/candi/...yaml:1-3` | Online baseline runner |
| `online_variant` | `main` | Same configs: `3` | Baseline label, not THESIS A-variant |
| `adaptation_momentum` | `0.01` M2N2; `0.02` CANDI | Same configs: `15` | Current custom reference-statistics update |
| `encoder_family` | `cnn_simple` | Same configs: `17` | Current RedLamp-based approximation |
| `encoder_dim` | `128` | Same configs: `18` | Current RedLamp checkpoint contract |
| `pretrained_encoder_checkpoint` | RedLamp `best.pt` | Same configs: `23` | Current checkpoint source |
| M2N2 `gamma` | `0.99999` in reference | `bsc-thesis-ref-codebases/CANDI-main/config.py:75-76` | Reference Detrender |
| CANDI `MIN_SAMPLES` | `16` in reference | `bsc-thesis-ref-codebases/CANDI-main/config.py:77-86` | Reference candidate-pool update gate |
| CANDI `USE_FPM` / `USE_SANA` | `true` in reference | Same config: `78-79` | Reference CANDI mechanisms |

## Conflicts and uncertainties

1. The current benchmark contract intentionally chooses RedLamp CNN encoder
   reuse, while the reference adapter code is model-native for MLP/TimesNet.
   The available files do not establish a compatible native checkpoint for a
   reference-faithful port using the current RedLamp artifact.
2. The reference repository contains both the adapter path and a separate
   `TSB-AD` implementation. This report uses the user-named adapter paths as the
   primary reference and treats the TSB-AD files as supporting evidence only.
3. The current contract calls the RedLamp reconstruction head a baseline-owned
   decoder, but the current online class creates a fresh decoder and loads only
   encoder tensors. The available current tests confirm encoder loading, not
   decoder training or reconstruction quality.
4. The current `online_variant: main` label is a configuration label. It cannot
   be interpreted as an M2N2/CANDI method variant or as THESIS A0/A1/A2.

## Open questions

1. Should the project revise the locked RedLamp checkpoint contract and create
   native M2N2/CANDI-compatible model checkpoints, or retain RedLamp and report
   repository-specific method variants?
2. If native checkpoints are selected, should the project port the reference
   MLP/TimesNet model or adapt the reference objectives to an existing thesis
   model while documenting the resulting method deviation?
3. Which exact CANDI reference configuration is the target for the thesis:
   `USE_FPM=true`, `USE_SANA=true`, hard and moderate selection, and
   `MIN_SAMPLES=16`, or an explicitly documented ablation?
4. What exact threshold protocol should M2N2/CANDI use after their score model
   and update timing change? Existing threshold artifacts need recalibration.

## Implementation structure: sequential stages (draft pending Phase 0 decision)

### Summary

The work is structured as a sequence of gates. First, the project must decide
which model and checkpoint contract represents a method-faithful M2N2/CANDI
implementation. The implementation can then establish the model/state boundary,
make the stream causal, add M2N2 and CANDI native adaptation, update the
configuration and artifacts, and finally validate one real end-to-end run.

This structure is intentionally high-level. The detailed file-level actions
remain in the implementation plan below and must not start until the Phase 0
contract decision is confirmed.

### Request

Create sequential stages for every implementation phase so that each stage has
one observable result, a dependency on earlier work, and an explicit completion
gate. Keep the common SMD stream range, clean-validation threshold split,
metrics-only label usage and isolation from THESIS-specific online mechanics.

### Confirmed context

- The current M2N2/CANDI path shares one RedLamp-based adaptive implementation
  and updates reference statistics instead of model parameters.
- The reference M2N2 path requires Detrender state, timestep masking and a
  masked reconstruction optimization step.
- The reference CANDI path requires validation-derived representations,
  hard/moderate candidate pools, FPM selection, SANA and a minimum-sample gate.
- The current stream path scores the selected sequence before applying updates;
  the reference path scores, records and adapts one batch before the next.
- The current RedLamp checkpoint contract and the reference-native MLP/TimesNet
  model contract are not proven compatible.

### Scope

#### In scope

- Resolve the model/checkpoint identity required for a method-fidelity claim.
- Establish model, optimizer and method-state ownership.
- Enforce causal score-then-adapt processing.
- Implement the reference M2N2 and CANDI update mechanisms.
- Align configuration, threshold artifacts, provenance and fidelity tests.
- Validate one real checkpoint smoke run before a full benchmark matrix.

#### Out of scope

- Reusing THESIS Gumbel sampling, four-region triage, PNN verification,
  projector updates or uncertainty ablations inside M2N2/CANDI.
- Mixing results from the current approximation with results from the native
  implementation.
- Launching the full benchmark matrix before the single-combination smoke run
  and method-fidelity tests pass.
- Adding support for an unselected model family or an unconfirmed checkpoint.

## Proposed phases and sequential stages

### Phase 0: Resolve the model and checkpoint contract

**Result:** The project has one explicit model/checkpoint decision for the
method-fidelity target, with compatible terminology, configuration fields and
artifact identity.

**Stages:**

1. **Compare the two contract options.** Document the consequences of selecting
   the reference-native model/checkpoint path versus retaining RedLamp as an
   explicitly repository-specific variant.
2. **Verify checkpoint availability and compatibility.** Confirm the selected
   model family, input/output shapes, state-dict coverage, trainable modules and
   threshold-score contract from available project evidence.
3. **Record the decision and terminology mapping.** Update the benchmark
   contract so “method-faithful M2N2/CANDI” and any retained RedLamp variant are
   separate, non-interchangeable objects.
4. **Freeze the implementation boundary.** Reject any implementation path that
   silently loads an incompatible encoder-only checkpoint or silently calls a
   repository-specific variant method-faithful.

**Depends on:** Existing research findings and the user's decision on the
model/checkpoint contract.

**Verification:** Automated checkpoint/schema validation and a manual review of
the selected model, checkpoint, method name and artifact identity.

**Risks:** The available RedLamp artifact may not support the reference adapter
contract. Keep the phase open until the project explicitly selects native
checkpoints or accepts a documented deviation.

**Complete when:** No model/checkpoint conflict remains open and later phases
can name one stable method-native model surface.

### Phase 1: Establish the method-native model and state boundary

**Result:** Each baseline owns a complete model state and can distinguish frozen
detector parameters, trainable adaptation parameters, optimizer state and
method-specific runtime state.

**Stages:**

1. **Define the model ownership boundary.** Specify which model produces scores,
   which parameters may change and which parameters must remain frozen.
2. **Load and validate the complete selected state.** Ensure the checkpoint
   covers every required component and that the model accepts the stream tensor
   shape without creating an unverified fresh reconstruction head.
3. **Create method-owned runtime state.** Allocate the M2N2 Detrender state or
   the CANDI validation references, candidate pools and SANA modules only after
   the selected model has been validated.
4. **Prove the state boundary.** Run a minimal baseline initialization and
   confirm that disallowed parameters remain unchanged while allowed state is
   represented explicitly.

**Depends on:** Phase 0 model/checkpoint decision.

**Verification:** Deterministic state-dict checks, tensor-shape checks and a
manual inspection of trainable/frozen parameter names.

**Risks:** A partial checkpoint can make the code run while producing an
unverified decoder or adapter. Fail early on missing or unexpected required
state.

**Complete when:** The selected checkpoint, model and trainable surface pass
the state-boundary checks for both baselines.

### Phase 2: Establish causal score-then-adapt streaming

**Result:** The online runtime processes windows in stream order, records the
score for window `t`, adapts from the permitted information for `t`, and only
then scores window `t+1`.

**Stages:**

1. **Define the lifecycle boundary.** Give the runner a small method-owned flow
   for initialization, scoring, adaptation and state reporting while preserving
   the existing calibration and report interface.
2. **Process one window or batch at a time.** Preserve absolute indices,
   stride-one windows, selected ranges and metrics-only label handling while
   removing full-sequence precomputation from the adaptive path.
3. **Persist reproducibility state.** Capture the model parameters, optimizer
   state, Detrender state or CANDI pools/references needed to resume or audit a
   causal run.
4. **Validate causal feedback.** Use a deterministic two-window fixture to
   prove that an accepted update from window 1 can affect window 2, while the
   score recorded for window 1 remains pre-update.

**Depends on:** Phase 1 state ownership and the existing stream/index contract.

**Verification:** Automated two-window ordering and state-round-trip tests;
manual inspection of a short `receive -> score -> record -> adapt` trace.

**Risks:** Sequential processing can change score distributions and runtime
cost. Recalibrate thresholds after the exact lifecycle is fixed.

**Complete when:** The runner and both baselines satisfy the causal lifecycle
without changing stream range, window alignment or label ownership.

### Phase 3: Implement M2N2 native adaptation

**Result:** M2N2 updates its method-native state using Detrender statistics, a
timestep pseudo-anomaly mask, masked reconstruction loss and the configured
optimizer step.

**Stages:**

1. **Implement Detrender updates.** Update the mean-only Detrender with the
   reference `gamma` semantics before the model forward pass.
2. **Compute the reference adaptation quantities.** Produce timestep
   reconstruction errors, the pseudo-anomaly mask and the normal-position mask
   without using test labels.
3. **Run the masked optimization step.** Compute the masked reconstruction loss
   and execute the configured zero-gradient, backward and optimizer-step flow.
4. **Separate prediction from adaptation.** Keep the prediction threshold as a
   scoring/reporting decision and do not use the current raw/EWMA predicate as a
   substitute for the M2N2 loss mask.
5. **Expose auditable update state.** Record finite score, threshold, loss,
   mask count, update status and method configuration needed to explain each
   update.

**Depends on:** Phases 1-2 and the selected native model/checkpoint.

**Verification:** Deterministic Detrender, mask, masked-loss, parameter-change
and causal-feedback tests; manual review of one M2N2 trace.

**Risks:** The selected checkpoint may use a different normalization order.
Verify the training/inference convention before accepting the update sequence.

**Complete when:** The M2N2 tests prove the reference update equations and the
baseline changes only the permitted state in causal order.

### Phase 4: Implement CANDI native adaptation

**Result:** CANDI selects candidates from validation-derived latent references,
maintains separate hard/moderate pools, and updates SANA only after the
reference minimum-sample condition is met.

**Stages:**

1. **Build clean-validation references.** Compute the representations,
   covariance pseudoinverse and configured hard/moderate reference sets from
   clean validation data only.
2. **Select hard and moderate candidates.** Apply score and Mahalanobis
   conditions to current test representations without reading test labels.
3. **Accumulate and gate candidate pools.** Maintain separate pools and trigger
   adaptation only when the configured `MIN_SAMPLES` condition is satisfied.
4. **Adapt SANA residual modules.** Keep the pretrained detector frozen, update
   the allowed SANA modules with reconstruction MSE and clear each pool after
   its update.
5. **Remove the THESIS triage shortcut.** Ensure CANDI metadata and tests no
   longer interpret `gray_zone` or `pnn_candidate` as the native CANDI update
   rule.

**Depends on:** Phases 1-2 and the selected CANDI FPM/SANA configuration.

**Verification:** Deterministic reference-set, Mahalanobis, pool-gate, SANA
parameter-boundary, label-isolation and causal-feedback tests; manual review of
one CANDI candidate/update trace.

**Risks:** Singular covariance or insufficient validation references can make
selection undefined. Validate the data contract and fail explicitly instead of
silently changing the selection rule.

**Complete when:** CANDI tests prove FPM selection, SANA adaptation,
minimum-sample gating and label-independent causal updates.

### Phase 5: Align configuration, thresholds and artifacts

**Result:** Generated configurations and persisted artifacts describe the
selected native methods and are not confused with historical approximation
runs.

**Stages:**

1. **Define method-native configuration fields.** Replace shared approximation
   settings with explicit M2N2 and CANDI update settings tied to the selected
   model contract.
2. **Regenerate and validate benchmark configurations.** Preserve entity, seed,
   stream range, window size and metric settings while removing stale
   approximation-only fields.
3. **Recalibrate thresholds.** Recompute thresholds from clean validation after
   the final native score and causal update protocol is fixed.
4. **Write provenance and migration identity.** Record checkpoint, model,
   method settings, threshold source, selected range and update policy; mark old
   approximation outputs non-comparable instead of overwriting them.

**Depends on:** Phases 0-4 and a stable causal score protocol.

**Verification:** Configuration/schema checks, threshold finite-value and
identity checks, and manual comparison of one generated configuration with the
selected reference settings.

**Risks:** Any score-protocol change invalidates old thresholds and tables.
Create new result identities and keep historical artifacts separate.

**Complete when:** The generator, YAML files, threshold artifacts and manifests
pass validation and identify the native method unambiguously.

### Phase 6: Validate fidelity and one real end-to-end smoke

**Result:** Method-level evidence and one real checkpoint run show that the
native implementations work through the actual benchmark entry point before
the full matrix is considered.

**Stages:**

1. **Run deterministic method tests.** Verify equations, masks, parameter
   boundaries, candidate pools, minimum-sample gates and finite losses.
2. **Run causal integration tests.** Verify calibration, selected absolute
   range, sequential records, threshold use, update order and artifact schema.
3. **Run one configured real checkpoint smoke.** Execute one entity/seed/method
   combination and inspect the produced records, threshold artifact and
   benchmark report.
4. **Review acceptance evidence.** Confirm checkpoint identity, complete range,
   finite metrics, causal update evidence and metrics-only label usage.
5. **Approve or stop before matrix expansion.** Expand to the full benchmark
   only if all preceding evidence passes; otherwise retain the failing evidence
   and return to the responsible phase.

**Depends on:** Phases 0-5.

**Verification:** Focused online tests, one real end-to-end smoke and manual
artifact review. The full relevant online suite runs only after the smoke passes.

**Risks:** Shape or artifact success can hide method misimplementation. The
smoke is acceptable only when the method-fidelity tests already pass.

**Complete when:** The focused tests, one real smoke and provenance review pass;
the project can justify a full matrix without mixing implementation generations.

### Dependency summary

| Phase | Requires | Enables |
| --- | --- | --- |
| 0 | Research evidence and model/checkpoint decision | Stable method identity and implementation boundary |
| 1 | Phase 0 | Complete model and method-state ownership |
| 2 | Phase 1 | Causal scoring/adaptation for both methods |
| 3 | Phases 1-2 | Native M2N2 update evidence |
| 4 | Phases 1-2 | Native CANDI update evidence |
| 5 | Phases 0-4 | Reproducible configs, thresholds and artifacts |
| 6 | Phases 0-5 | Acceptance of one real smoke and later matrix execution |

### Decisions confirmed

- The common stream range, window alignment, clean-validation threshold split
  and metrics-only label usage remain in scope.
- M2N2 and CANDI must retain their reference-native update mechanisms and must
  not inherit THESIS-specific triage or adaptation components.
- Historical approximation outputs must remain separate from native-method
  results.

### Blocking decision

The model/checkpoint decision in Phase 0 is still open. The structure remains a
draft until the project selects either reference-native checkpoints for a
method-fidelity claim or an explicitly named RedLamp-based variant with its
deviation documented. This decision changes the model boundary, configuration,
threshold and verification stages.

### Feedback requested

- Does this phase order match the intended delivery path?
- Should the project select reference-native checkpoints, or retain RedLamp as
  a separately named repository-specific variant?
- Is any stage too broad, too narrow or missing before the structure is expanded
  into implementation instructions?

## Implementation plan: draft pending Phase 0 decision

### Summary

The implementation must replace the current shared reference-statistics
approximation with method-native M2N2 and CANDI adaptation while preserving the
common SMD stream range, clean-validation threshold split, metrics-only label
usage and baseline isolation from THESIS-specific mechanics.

The plan cannot become implementation-ready until Phase 0 resolves the
RedLamp-only checkpoint contract versus the reference-native model contract.
That decision changes the model files, checkpoint artifacts, configuration
schema and threshold recalibration path.

### Request

Read `prompts/2_plan_prompt.md` and write concrete implementation phases for
this research report so M2N2 and CANDI can later be implemented accurately
according to the user-named reference adapters.

No source code, configuration, test or benchmark implementation belongs to this
planning update.

### Current state

The active runner uses `fit() -> calibrate() -> run_sequence()` and passes a
complete selected test sequence to one baseline call. M2N2 and CANDI share a
CNN autoencoder and update only feature mean/std. Their tests establish basic
instantiation, calibration, records and checkpoint metadata, but not reference
method fidelity.

### Desired end state

- M2N2 performs reference-style online detrending, masked reconstruction-loss
  adaptation and optimizer updates.
- CANDI performs reference-style validation-derived FPM selection, hard/moderate
  candidate buffering, SANA residual adaptation and optimizer updates.
- Both methods score the current causal window/batch before adaptation, and the
  updated state affects only subsequent windows/batches.
- The selected model/checkpoint contract is explicit and compatible with the
  chosen reference architecture.
- Thresholds, provenance, metrics and tests describe the actual score and update
  protocol.

### Scope

#### In scope

- Resolve and document the model/checkpoint contract.
- Implement method-native M2N2 and CANDI adaptation state.
- Replace whole-sequence precomputed scoring with causal score/update order.
- Update configs, threshold artifacts, tests, reports and method metadata.
- Verify no test labels enter scoring, selection or adaptation.

#### Out of scope

- Changing `src/engine/online_tta/` THESIS behavior.
- Adding THESIS A0/A1/A2 variants to M2N2 or CANDI.
- Adding new uncertainty, PNN, projector or verification behavior to baselines.
- Running the full 99-run online matrix before one end-to-end smoke passes.
- Claiming exact reference fidelity for a RedLamp-based variant without
  documenting its deviations.

### Implementation approach

Keep common stream selection, absolute indices, clean-validation threshold
ownership and artifact serialization in the existing benchmark path. Keep
M2N2-specific and CANDI-specific adaptation logic in their existing baseline
owners unless a reusable helper is proven necessary. Do not use the current
`gray_zone`/`pnn_candidate` triage as a substitute for CANDI FPM, and do not use
`adaptation_momentum` as a substitute for M2N2's Detrender plus optimizer.

The implementation should preserve the current repository boundary that
baselines use native protocols, but it must first replace or explicitly revise
the incompatible RedLamp-only contract if exact reference fidelity is the goal.

## Phase 0: Resolve the model and checkpoint contract

### Goal

Choose one executable contract before source implementation begins:

1. reference-faithful M2N2/CANDI using compatible native model checkpoints; or
2. RedLamp-based repository variants with explicit non-equivalence and different
   method names.

Only option 1 satisfies the current user request for methodologically accurate
M2N2 and CANDI.

### Changes

#### 1. Lock the selected native model surface

- **File:** `documents/spec/online_benchmark_contract.md`
- **Symbol:** Sections 2.4 and 2.5, checkpoint and adaptation contract
- **Change:** Record the chosen reference model family, checkpoint role,
  trainable parameter boundary, input shape, window size, threshold source and
  adaptation step configuration. If the native MLP/TimesNet checkpoint is not
  available, record that artifact as a prerequisite instead of mapping the
  RedLamp checkpoint by name.
- **Reason:** Current config and reference adapter use different model contracts.
- **Dependencies:** Online generator, baseline constructors, threshold
  artifacts, checkpoint inventory and final report naming.

#### 2. Preserve terminology identity

- **File:** `documents/spec/online_benchmark_contract.md`
- **Symbol:** Terminology changes section
- **Change:** Map current `m2n2`/`candi` runtime objects to the selected native
  adapter objects, or introduce explicit `m2n2_style`/`candi_style` names if
  RedLamp is retained.
- **Reason:** Similar method names do not prove semantic equivalence.
- **Dependencies:** Config keys, output directories, metadata and table labels.

### Verification

#### Automated

- [ ] Contract/config validation tests reject a checkpoint whose architecture,
  input dimension or state keys do not match the selected adapter.

#### Manual

- [ ] Inspect one selected checkpoint and confirm its model state, trainable
  surface and method identity match the revised contract.

### Risks

- **Risk:** Existing RedLamp checkpoints cannot initialize the native reference
  model. **Mitigation:** stop before Phase 1 and record the missing artifact;
  verify with strict state-dict and shape checks.

## Phase 1: Establish the method-native model and state boundary

### Goal

Make the model and checkpoint loaded by each adapter explicit before adding
online updates.

### Changes

#### 1. Separate model initialization from adaptation

- **File:** `src/baselines/online/adaptive.py:60-161`
- **Symbol:** `AdaptiveStreamingBaselineBase.__init__`, `fit`, `_fit_backbone`
- **Change:** Remove the assumption that both methods share one RedLamp
  autoencoder and one frozen-encoder-only loader. Keep only common validated
  window/device/metadata utilities. Let each method own its model and optimizer
  initialization.
- **Reason:** M2N2 and CANDI have different trainable surfaces and state.
- **Dependencies:** `m2n2.py`, `candi.py`, checkpoint loader and config kwargs.

#### 2. Validate the selected model state

- **File:** `src/models/simple_window_cnn_autoencoder.py` and
  `src/baselines/online/redlamp_encoder_checkpoint.py`
- **Symbol:** `SimpleWindowCnnAutoencoder`, `load_redlamp_encoder_checkpoint`
- **Change:** Keep these files only if the Phase 0 contract explicitly retains
  RedLamp. Otherwise add the selected native model loader at a clearly named
  proposed path and test the complete model state, not only `encoder.*`.
- **Reason:** A fresh untrained decoder cannot provide a valid reconstruction
  score for a reference adapter.
- **Dependencies:** checkpoint artifact, offline model training and threshold
  calibration.

### Verification

#### Automated

- [ ] Model construction and checkpoint round-trip tests verify every required
  state key, tensor shape and trainable/frozen parameter boundary.
- [ ] A single forward pass returns `[B, L, D]` reconstruction-compatible
  outputs and a finite scalar score per sample/window.

#### Manual

- [ ] Inspect parameter names before and after adaptation and confirm only the
  contract-approved parameters can change.

### Risks

- **Risk:** Sharing the current autoencoder hides a wrong decoder or wrong
  latent contract. **Mitigation:** fail fast on checkpoint/model mismatch and
  retain method-specific metadata in every report.

## Phase 2: Replace the stream lifecycle with causal score-then-adapt

### Goal

Ensure that each method receives one causal window/batch, emits its score before
adaptation, and lets only later windows/batches observe the updated state.

### Changes

#### 1. Define the adaptive lifecycle

- **File:** `src/baselines/online/base.py:60-98`
- **Symbol:** `OnlineStreamingBaselineProtocol`, `build_stride1_batcher`
- **Change:** Preserve the stable calibration/report interface, but add an
  explicit method-owned lifecycle for sequential scoring and adaptation, or
  define a small internal adapter protocol with `initialize`, `score_batch`,
  `adapt_batch` and `state_metadata`.
- **Reason:** CANDI requires candidate-pool state and M2N2 requires optimizer
  state; `did_update: bool` alone is insufficient.
- **Dependencies:** runner, stream batcher, record schema and state tests.

#### 2. Process windows sequentially

- **File:** `scripts/benchmarks/run_online_streaming_benchmark.py:360-405`
- **Symbol:** test-stream loop
- **Change:** Use the existing absolute-range selection and stride-1 window
  contract, but invoke score and adaptation in causal order. Keep labels out of
  the baseline call; retain them only for final metrics.
- **Reason:** The reference predictor scores one batch and calls `adapt()` before
  consuming the next batch.
- **Dependencies:** `src/data/stream.py`, threshold state, record aggregation
  and output ordering.

#### 3. Define update-state serialization

- **File:** `src/baselines/online/base.py` and method owners
- **Symbol:** method runtime state
- **Change:** Persist the minimum state needed to reproduce a run: model
  parameters, optimizer state, M2N2 Detrender state, or CANDI candidate buffers
  and validation reference statistics.
- **Reason:** A causal online method is not reproducible if update state is
  discarded.
- **Dependencies:** artifact manifest and checkpoint/state round-trip tests.

### Verification

#### Automated

- [ ] A two-window fixture proves window 2's score changes only when window 1's
  accepted update is enabled.
- [ ] A test proves the score recorded for window `t` was computed before the
  adaptation triggered by window `t`.
- [ ] Absolute start/end indices and `stream_step` remain unchanged.

#### Manual

- [ ] Inspect one short stream trace and verify the order
  `receive -> score -> record -> adapt -> next window`.

### Risks

- **Risk:** Changing from vectorized full-sequence scoring changes score
  distribution and runtime cost. **Mitigation:** recalibrate thresholds and run
  one smoke combination before any matrix run.

## Phase 3: Implement M2N2 native adaptation

### Goal

Reproduce the reference M2N2 update: online Detrender statistics, timestep
pseudo-anomaly mask, masked reconstruction loss and one optimizer step.

### Changes

#### 1. Add Detrender state

- **File:** `src/baselines/online/m2n2.py`
- **Symbol:** `M2N2StreamingBaseline` and method-owned adapter state
- **Change:** Maintain a mean-only Detrender with configured `gamma`; update it
  from the current input batch before model forward, matching the reference
  `mean.lerp_(mu, 1 - gamma)` semantics.
- **Reason:** Current `reference_mean/std` EMA is not the reference M2N2
  normalizer.
- **Dependencies:** selected native model, batch shape and state serialization.

#### 2. Add masked adaptation objective

- **File:** `src/baselines/online/m2n2.py`
- **Symbol:** proposed `adapt_batch` method
- **Change:** Compute per-timestep `A = mean((recon - x)^2, dim=-1)`, create
  `ytilde = A >= threshold`, use `mask = ytilde == 0`, compute `(A * mask).mean()`,
  then run `zero_grad -> backward -> step` for the configured number of steps.
- **Reason:** This is the reference M2N2 adaptation objective.
- **Dependencies:** optimizer config, trainable parameter boundary and record
  `loss_total`.

#### 3. Remove the predicate-only update path

- **File:** `src/baselines/online/m2n2.py` and
  `src/baselines/online/adaptive.py`
- **Symbol:** `_should_update`, `_update_reference`
- **Change:** Do not use raw/EWMA dual threshold as the adaptation mechanism.
  Keep prediction thresholding separate from M2N2's timestep loss mask.
- **Reason:** Current predicate has no reference gradient update.
- **Dependencies:** runner lifecycle and tests.

### Verification

#### Automated

- [ ] Detrender statistics match the reference EMA update on a deterministic
  batch.
- [ ] The pseudo-anomaly mask excludes only `A >= threshold` positions.
- [ ] One optimizer step changes an allowed trainable parameter and does not
  change frozen parameters.
- [ ] `loss_total` is finite and equals the masked reconstruction objective for a
  deterministic fixture.

#### Manual

- [ ] Review one logged M2N2 record containing score, threshold, mask count,
  loss and update status.

### Risks

- **Risk:** Updating Detrender before forward can cause train/test protocol
  mismatch if the selected checkpoint was trained with a different normalization
  convention. **Mitigation:** verify the native training path and calibrate
  threshold after the exact inference sequence is fixed.

## Phase 4: Implement CANDI native adaptation

### Goal

Reproduce CANDI's validation-derived false-positive mining and SANA residual
adaptation without using THESIS triage semantics.

### Changes

#### 1. Build validation reference sets

- **File:** `src/baselines/online/candi.py`
- **Symbol:** calibration/initialization lifecycle
- **Change:** From clean validation only, compute representations, covariance
  pseudoinverse, top-k hard reference representations and moderate reference
  representations selected from the configured score range.
- **Reason:** CANDI FPM needs validation-derived latent reference sets.
- **Dependencies:** model `get_representations`, threshold and clean-validation
  data contract.

#### 2. Implement hard/moderate candidate selection

- **File:** `src/baselines/online/candi.py`
- **Symbol:** proposed method-owned selection and buffer methods
- **Change:** For each test batch, compute current representations and scores.
  Select hard candidates using high score plus Mahalanobis proximity to the hard
  reference set. Select moderate candidates using non-anomalous score plus
  Mahalanobis proximity to the moderate reference set. Accumulate the two pools
  separately.
- **Reason:** This replaces the current `gray_zone`/`pnn_candidate` predicate.
- **Dependencies:** covariance inverse, chi-square threshold, candidate state
  and no-label adaptation rule.

#### 3. Add SANA and minimum-sample update

- **File:** `src/baselines/online/candi.py`
- **Symbol:** proposed SANA modules, `adapt_batch`
- **Change:** Freeze the pretrained detector when SANA is enabled, create
  trainable `sana_in` and `sana_out`, adapt each eligible pool only after
  `MIN_SAMPLES`, compute reconstruction MSE, run configured optimizer steps and
  clear that pool after update.
- **Reason:** This is the reference CANDI trainable surface and update gate.
- **Dependencies:** selected model, optimizer config, loss reporting and state
  serialization.

#### 4. Remove THESIS triage dependency

- **File:** `src/baselines/online/candi.py` and
  `src/engine/online_tta/triage.py`
- **Symbol:** current `_should_update` and `classify_legacy_baseline_window`
- **Change:** Stop using legacy THESIS-style labels for CANDI adaptation. Keep
  `classify_legacy_baseline_window` only if another confirmed baseline path uses
  it; otherwise mark its CANDI use removed in tests and metadata.
- **Reason:** Full-spec-v3 explicitly isolates baseline-native protocols from
  THESIS four-region triage.
- **Dependencies:** metadata, tests and report schema.

### Verification

#### Automated

- [ ] Validation references use clean validation inputs only.
- [ ] Hard and moderate masks match deterministic Mahalanobis fixtures.
- [ ] Candidate pools update only after `MIN_SAMPLES` and reset after update.
- [ ] Only SANA parameters change when SANA is enabled.
- [ ] CANDI MSE and optimizer update are finite and affect the next window only.
- [ ] Test labels do not affect candidate selection or adaptation.

#### Manual

- [ ] Inspect a short CANDI trace showing validation references, candidate pool
  sizes, update trigger, loss and next-window score.

### Risks

- **Risk:** Covariance can be singular or validation reference sets can be empty.
  **Mitigation:** use the reference pseudoinverse behavior, validate minimum
  reference sizes, and fail with an explicit configuration/data error.
- **Risk:** CANDI candidate accumulation increases memory use. **Mitigation:**
  retain only the configured pools and reset them after update; do not persist
  every forward output.

## Phase 5: Update configuration, thresholds and artifacts

### Goal

Make configuration and persisted artifacts describe the actual native methods.

### Changes

#### 1. Generate method-native configurations

- **File:** `scripts/benchmarks/generate_online_streaming_benchmark_configs.py:98-133`
- **Symbol:** `_baseline_kwargs`
- **Change:** Replace the current shared CNN/momentum kwargs with explicit M2N2
  fields (`gamma`, optimizer, learning rate, steps, trainable scope) and CANDI
  fields (`USE_FPM`, `USE_SANA`, hard/moderate selection, `MIN_SAMPLES`, SANA
  architecture, optimizer and steps).
- **Reason:** Current YAML describes the approximation, not the reference
  methods.
- **Dependencies:** Phase 0 model contract and method constructors.

#### 2. Regenerate and validate YAML

- **File:** `configs/experiment/online_benchmark/m2n2/` and
  `configs/experiment/online_benchmark/candi/`
- **Symbol:** generated `main` and `smoke` configurations
- **Change:** Regenerate from the source generator. Keep the same entity/seed,
  absolute range, window size and metrics-only label policy.
- **Reason:** Generated files are not the source of truth.
- **Dependencies:** config schema, preflight checks and checkpoint inventory.

#### 3. Recalibrate thresholds and manifests

- **File:** online threshold artifact path selected by the runner
- **Symbol:** threshold artifact generation and report metadata
- **Change:** Recompute clean-validation thresholds after the native model,
  window order and score protocol are fixed. Record checkpoint identity, model
  identity, method settings, stream range and update policy.
- **Reason:** Existing thresholds are not valid for a changed score protocol.
- **Dependencies:** one successful smoke run and artifact integrity checks.

### Verification

#### Automated

- [ ] Generator emits exactly one official `main` config per method/entity/seed.
- [ ] Config validation rejects missing native method settings.
- [ ] Threshold artifacts pass schema, checkpoint identity and finite-value
  checks.

#### Manual

- [ ] Compare one generated config with the selected reference configuration and
  confirm no stale `adaptation_momentum`-only path remains.

### Risks

- **Risk:** Changing score protocol invalidates old result tables. **Mitigation:**
  mark old artifacts non-comparable, write new output identities and do not mix
  them in aggregation.

## Phase 6: Add fidelity tests and run one end-to-end smoke

### Goal

Prove method fidelity before full benchmark execution.

### Changes

#### 1. Add method-level tests

- **File:** `tests/online/test_online_streaming_baseline_contracts.py`
- **Symbol:** existing shared contract tests
- **Change:** Split shared stream checks from M2N2/CANDI method-fidelity tests.
  Add deterministic fixtures for loss, masks, parameter changes, buffers,
  minimum-sample gates and causal feedback.
- **Reason:** Existing tests prove only instantiation and record shape.
- **Dependencies:** Phases 1-4.

#### 2. Revise checkpoint tests

- **File:** `tests/online/test_redlamp_encoder_checkpoint.py`
- **Symbol:** encoder-only checkpoint assertions
- **Change:** Keep them only for a retained RedLamp compatibility path. For the
  native contract, add complete model-state and trainable-surface tests.
- **Reason:** Current test locks a contract that may be replaced.
- **Dependencies:** Phase 0 decision.

### Verification

#### Automated

- [ ] `pytest -q tests/online/test_online_streaming_baseline_contracts.py tests/online/test_redlamp_encoder_checkpoint.py` — relevant tests pass after their contract is updated.
- [ ] `pytest -q tests/online/test_online_streaming_benchmark_wrapper.py` — runner still preserves calibration, range selection and artifact writing.
- [ ] Full relevant online test suite — all method, stream, threshold and artifact checks pass.

#### Manual

- [ ] Run exactly one configured real checkpoint smoke combination.
- [ ] Inspect `online_records.json`, threshold artifact and benchmark report.
- [ ] Confirm the report contains causal update evidence, checkpoint identity,
  finite losses, complete selected range and metrics-only label usage.

### Risks

- **Risk:** A smoke test can pass shape contracts while still using the wrong
  method. **Mitigation:** require the method-fidelity assertions from Phases 3
  and 4 before accepting the smoke result.

## Testing strategy

Use deterministic unit tests for each method's state transition, then one
end-to-end smoke on one entity/seed/config. Test clean validation isolation,
finite scores and losses, parameter boundaries, causal update order, absolute
indices, checkpoint identity and metrics-only label usage. Run the full online
test suite only after the smoke combination passes.

## Migration and rollback

Do not overwrite old approximation outputs. Use new method/config identities or
mark old outputs `not_comparable` in manifests. Preserve the current RedLamp
loader and configs until Phase 0 decides whether they remain an explicit
compatibility path. If native implementation fails, rollback means selecting
the prior config/code revision and retaining old artifacts as historical, not
mixing their metrics with native-method results.

## Documentation

- Update `documents/spec/online_benchmark_contract.md` after Phase 0.
- Add the selected method/checkpoint terminology mapping.
- Update baseline method metadata and report schema descriptions.
- Record exact reference deviations if any RedLamp compatibility path remains.
- Update the research report with implementation evidence after each phase;
  do not mark a method faithful before the fidelity tests pass.

## Final verification

- [ ] Phase 0 model/checkpoint decision is explicit and no blocking contract
  conflict remains.
- [ ] M2N2 passes Detrender, masked-loss, optimizer and causal-update tests.
- [ ] CANDI passes FPM, Mahalanobis, SANA, minimum-sample and causal-update
  tests.
- [ ] One real smoke combination completes with finite metrics and complete
  provenance.
- [ ] Threshold artifacts are recalibrated from the exact native score protocol.
- [ ] Full relevant online tests pass before the matrix is launched.

## Assumptions and non-blocking uncertainties

- The common SMD absolute-range, window-size and metrics-only label contracts
  remain unchanged unless Phase 0 explicitly revises them.
- The current `src/data/stream.py` window shapes can support `[B, L, D]`, but the
  exact batch size and state serialization should be verified in Phase 2.
- The precise native model checkpoint path remains unknown until the blocking
  Phase 0 decision is made.

## Detailed implementation instructions: draft pending Phase 0 decision

### Status and gating

This section expands the approved-by-request phase structure into atomic
implementation steps. It is still a draft because the model/checkpoint contract
in Phase 0 is open. An implementer must not begin the model, constructor or
configuration steps that depend on that contract until the project records the
decision in `documents/spec/online_benchmark_contract.md`.

The steps below describe the confirmed current code and the smallest changes
needed to reach the reference mechanisms. They do not authorize source-code,
configuration, test or benchmark edits during this documentation task.

### Atomic-step rules

- Execute steps within a phase in numeric order.
- Do not skip a phase's completion gate because a later test happens to pass.
- Treat a step marked **conditional on Phase 0** as blocked until the selected
  model/checkpoint contract is recorded.
- Keep test labels out of baseline calibration, candidate selection, loss
  construction and parameter updates. The runner may keep labels for final
  metrics only.
- Preserve the selected absolute half-open stream range, stride-one window
  alignment, entity-global indices, clean-validation threshold split and
  existing output directory identity unless a documented contract decision
  changes one of them.

## Phase 0: Resolve the model and checkpoint contract

### Goal

Choose and record the model/checkpoint contract that makes the later M2N2 and
CANDI implementation claims precise. The implementation must distinguish a
reference-faithful method from a RedLamp-based repository variant.

### Dependencies

- Research evidence in this report.
- Current contract in `documents/spec/online_benchmark_contract.md`.
- Reference adapter/model code under
  `bsc-thesis-ref-codebases/CANDI-main/tta/` and
  `bsc-thesis-ref-codebases/CANDI-main/models/`.
- A human decision when the available artifacts cannot establish compatibility.

### Detailed atomic steps

#### Step 0.1: Enumerate the two candidate contracts

- **File:** `documents/spec/online_benchmark_contract.md`
- **Symbol:** M2N2/CANDI model, checkpoint and baseline-update contract sections
- **Current responsibility:** The document defines the online baseline's
  RedLamp CNN encoder, encoder-only checkpoint role and shared adaptation
  settings.
- **Change:** Write two explicit contract entries for comparison:
  `reference-native M2N2/CANDI` and `RedLamp-based repository variant`.
- **Reason:** The current RedLamp contract does not prove compatibility with the
  reference adapter's MLP/TimesNet model surface.
- **Inputs:** Existing contract text, current RedLamp checkpoint role and the
  reference model/adapter selection rules.
- **Outputs:** A terminology table that names model family, checkpoint role,
  reconstruction head, trainable parameters, normalizer, score function and
  adaptation objective for each contract.
- **Errors:** Stop if the table treats `cnn_simple` as an alias for MLP/TimesNet
  without state-dict, input-shape and forward-contract evidence.
- **Dependencies:** Research findings sections 2–4 in this report.
- **Compatibility:** Do not rename the current path to “faithful” merely to
  preserve existing configuration names. Historical names remain historical.
- **Verification:** Manually compare every table row with the current source and
  the user-named reference files.

#### Step 0.2: Inventory the reference model surface

- **File:** `bsc-thesis-ref-codebases/CANDI-main/tta/adapter.py`
- **Symbol:** `construct_adapter`
- **Current responsibility:** The reference selects M2N2/CANDI adapters by
  method and then selects `MLPAdapter` or `TimesNetAdapter` by model name.
- **Change:** Use this file as the authoritative inventory of the required
  model-facing methods: forward reconstruction, `get_anomaly_scores`, and for
  CANDI `get_representations`.
- **Reason:** The later port must not assume that the current CNN encoder-only
  object supplies the reference model API.
- **Inputs:** Selected model name and reference TTA method.
- **Outputs:** A verified list of required model methods and trainable modules.
- **Errors:** Mark Phase 0 blocked if the selected current checkpoint cannot
  provide the required methods or if a wrapper would change the method's score
  or loss semantics.
- **Dependencies:** Step 0.1.
- **Compatibility:** Keep the thesis runner's stable `calibrate()` and
  `run_sequence()` boundary; only the method-owned model surface is selected.
- **Verification:** Read the model definitions used by the reference adapter,
  including `models/mlp/modeling_mlp.py`, and record the required tensor shapes.

#### Step 0.3: Verify current checkpoint inventory

- **File:** `src/baselines/online/redlamp_encoder_checkpoint.py`
- **Symbol:** `load_redlamp_encoder_checkpoint`
- **Current responsibility:** Loads only matching `encoder.*` tensors and
  returns `RedLampEncoderCheckpoint` identity metadata.
- **Change:** Define the validation required for the selected contract. For the
  retained RedLamp variant, validate the encoder-only role explicitly. For a
  native contract, require a complete model checkpoint loader owned by the
  selected model surface.
- **Reason:** A fresh decoder in
  `src/models/simple_window_cnn_autoencoder.py` is not evidence of a trained
  reference reconstruction model.
- **Inputs:** Checkpoint path, expected model family, required state keys and
  expected input/output dimensions.
- **Outputs:** Validated checkpoint identity containing role, resolved path,
  SHA-256 and model/checkpoint contract name.
- **Errors:** Raise a clear validation error for a missing path, incompatible
  state key, wrong shape, wrong role or unsupported contract. Do not silently
  fall back to a randomly initialized component.
- **Dependencies:** Steps 0.1–0.2.
- **Compatibility:** Preserve the existing loader only as a named compatibility
  path if the RedLamp variant is retained.
- **Verification:** Extend the existing loader tests only after the contract is
  selected; current tests prove encoder loading, not native full-model loading.

#### Step 0.4: Decide the fidelity claim

- **File:** `documents/logs/2026-08-10/research/research-m2n2-candi-method-fidelity.md`
- **Symbol:** `Open questions` and `Blocking decision` sections
- **Current responsibility:** Records the unresolved RedLamp/native contract
  conflict.
- **Change:** Record one of the following decisions explicitly:
  reference-native checkpoints for a method-fidelity claim, or a separately
  named RedLamp-based variant whose deviation is reported.
- **Reason:** The decision changes model files, constructor inputs, checkpoint
  artifacts, configuration keys, thresholds and acceptance tests.
- **Inputs:** Evidence from Steps 0.1–0.3 and the project decision.
- **Outputs:** One stable implementation target and one stable result identity.
- **Errors:** Do not mark the phase complete if the decision says “use RedLamp”
  but does not specify whether the result is a faithful method or a variant.
- **Dependencies:** Steps 0.1–0.3.
- **Compatibility:** Preserve old approximation outputs as historical and
  non-comparable; do not overwrite them.
- **Verification:** Manual review confirms that method name, model name,
  checkpoint role and result identity are unambiguous.

#### Step 0.5: Lock terminology and migration boundary

- **File:** `documents/spec/online_benchmark_contract.md`
- **Symbol:** baseline method identity, `online_variant`, checkpoint provenance
  and update-policy sections
- **Current responsibility:** Describes the existing online benchmark contract.
- **Change:** Add the selected old/new object mapping. Mark objects as unchanged,
  renamed, split or deprecated; do not map a RedLamp variant to native M2N2 or
  CANDI by similar name alone.
- **Reason:** Results, checkpoints and threshold artifacts must remain
  comparable only within the same method/model contract.
- **Inputs:** Step 0.4 decision.
- **Outputs:** A contract that later code and artifact tests can validate.
- **Errors:** Reject an artifact when its recorded contract name and loaded
  checkpoint role disagree.
- **Dependencies:** Step 0.4.
- **Compatibility:** Existing unrelated baselines and THESIS online paths keep
  their own contracts.
- **Verification:** Inspect one old and one future artifact identity and confirm
  they cannot be aggregated as the same method when their contracts differ.

### Tests

#### Contract decision and checkpoint identity

- **Location:** `tests/online/test_redlamp_encoder_checkpoint.py` and a clearly
  named native-checkpoint test in the same online test area if Phase 0 selects
  native checkpoints.
- **Level:** Contract/unit.
- **Setup:** Create a fixture containing the exact selected model state and one
  fixture containing an incomplete or wrong-role state.
- **Action:** Load each fixture through the selected checkpoint loader.
- **Expected result:** The valid fixture returns the expected contract and
  identity; the invalid fixture raises a clear error.
- **Edge cases:** Missing encoder key, missing decoder/adapter key, wrong input
  dimension, empty path and unsupported model family.

### Verification

#### Automated

- [ ] Existing checkpoint tests pass for any retained RedLamp compatibility path.
- [ ] The selected native/variant checkpoint fixture passes complete state and
  contract validation.
- [ ] The contract metadata contains model family, checkpoint role and SHA-256.

#### Manual

- [ ] Review the terminology mapping with the method-fidelity claim.

### Risks and recovery

- **Risk:** The available checkpoint cannot support the reference adapter.
- **Mitigation:** Keep the native path blocked and choose either a native
  checkpoint or an explicitly named variant.
- **Verification:** Compare required state keys and forward methods before
  writing implementation code.
- **Recovery:** Retain the old approximation code/configuration and mark its
  historical outputs non-comparable to future native outputs.

### Complete when

- The project records one model/checkpoint contract.
- The fidelity claim and variant naming are explicit.
- The selected checkpoint role and required model surface are validated.
- No later phase needs to guess whether RedLamp is a native model or a variant.

## Phase 1: Establish the method-native model and state boundary

### Goal

Make the selected baseline load a complete, shape-compatible model state and
make the trainable/frozen boundary visible to both adaptation code and tests.

### Dependencies

- Phase 0 contract decision.
- Existing `AdaptiveStreamingBaselineBase.fit()` and `_fit_backbone()` flow.
- Selected model/checkpoint loader.

### Detailed atomic steps

#### Step 1.1: Separate common stream configuration from method model configuration

- **File:** `src/baselines/online/adaptive.py`
- **Symbol:** `AdaptiveStreamingBaselineBase.__init__`
- **Current responsibility:** Validates window, threshold, CNN dimensions and
  `adaptation_momentum`, then stores RedLamp encoder settings shared by M2N2 and
  CANDI.
- **Change:** Keep only configuration that is common to the selected online
  stream contract in the base constructor. Move method-owned optimizer,
  normalizer and adaptation fields to the method constructors or method-owned
  initialization path.
- **Reason:** M2N2's `gamma` and optimizer are not CANDI's FPM/SANA settings;
  shared `adaptation_momentum` currently describes neither reference method.
- **Inputs:** `train_sequence`, `input_dim`, `window_size`, threshold settings,
  selected model settings and checkpoint identity.
- **Outputs:** A validated common baseline object with no hidden
  method-specific update policy.
- **Errors:** Reject non-positive dimensions, invalid threshold weights, absent
  required checkpoint or unsupported model family before `fit()` mutates state.
- **Dependencies:** Phase 0 and Step 1.2.
- **Compatibility:** Preserve `calibrate()` and `run_sequence()` caller keyword
  names until the runner/config migration is complete.
- **Verification:** Constructor tests cover valid configuration and each
  invalid-value branch.

#### Step 1.2: Make model construction contract-driven

- **File:** `src/baselines/online/adaptive.py`
- **Symbol:** `AdaptiveStreamingBaselineBase._fit_backbone`
- **Current responsibility:** Creates `SimpleWindowCnnAutoencoder`, loads only
  RedLamp encoder tensors and freezes the encoder while leaving a fresh decoder.
- **Change:** Construct the model selected in Phase 0. Load every required
  checkpoint component. Set `requires_grad` and train/eval mode according to
  the selected method contract, not according to the current shared encoder-only
  path.
- **Reason:** M2N2 and CANDI adapt model outputs; a randomly initialized decoder
  or an unverified native wrapper changes the score and loss semantics.
- **Inputs:** `[T, D]` train sequence, `window_size`, selected model settings,
  checkpoint path and seed.
- **Outputs:** A model object with validated forward output and latent/repr
  access where CANDI requires it; a checkpoint identity record.
- **Errors:** Raise when the train sequence is shorter than one window, feature
  dimension disagrees, required model methods are absent or checkpoint state is
  incomplete.
- **Dependencies:** Step 0.3 and selected model implementation.
- **Compatibility:** Keep device placement controlled by the existing runner
  argument; do not introduce a second device lifecycle.
- **Verification:** Instantiate both methods with deterministic fixtures and
  assert output shape `[B, L, D]` plus required representation shape.

#### Step 1.3: Resolve the current autoencoder head explicitly

- **File:** `src/models/simple_window_cnn_autoencoder.py`
- **Symbol:** `SimpleWindowCnnAutoencoder.__init__`, `forward`
- **Current responsibility:** Builds a shared CNN encoder and a newly initialized
  1x1 convolution decoder.
- **Change:** If Phase 0 retains this model as an explicitly named variant,
  document and load the complete encoder/decoder state required for scoring and
  adaptation. If Phase 0 selects native MLP/TimesNet, do not silently extend
  this class to impersonate the reference model; use the selected native model
  or a clearly labeled wrapper.
- **Reason:** Similar reconstruction outputs do not prove identical method
  semantics.
- **Inputs:** `[B, L, D]` tensor and selected state dict.
- **Outputs:** Reconstruction `[B, L, D]`; latent/repr only if the selected
  contract defines it.
- **Errors:** Reject a state dict that initializes only the encoder when the
  selected score requires a trained decoder.
- **Dependencies:** Phase 0 decision.
- **Compatibility:** Preserve this class for callers that explicitly use the
  simple CNN variant; do not change unrelated model users.
- **Verification:** Full-state round trip and deterministic reconstruction test.

#### Step 1.4: Define model train/eval transitions

- **File:** `src/baselines/online/adaptive.py` and the method owner selected in
  Phase 0
- **Symbol:** model initialization, scoring and adaptation entry points
- **Current responsibility:** Calls `self.backbone_.eval()` once and scores under
  `torch.no_grad()`.
- **Change:** Keep scoring in evaluation/no-gradient mode. Enter training mode
  only around the method's optimizer update, then restore the previous mode.
- **Reason:** CANDI reference adaptation temporarily trains the selected model;
  M2N2 also requires gradient computation for its update step.
- **Inputs:** Current model mode and adaptation batch.
- **Outputs:** Same model mode after adaptation as before the update, with only
  allowed parameters changed.
- **Errors:** Fail if an update is requested without an optimizer or if a frozen
  parameter receives a gradient when the contract forbids it.
- **Dependencies:** Steps 1.2–1.3 and Phase 2 lifecycle.
- **Compatibility:** Scoring remains deterministic and does not retain a graph.
- **Verification:** Test mode restoration and parameter gradient boundaries.

#### Step 1.5: Emit complete model/checkpoint metadata

- **File:** `src/baselines/online/adaptive.py`
- **Symbol:** `_backbone_metadata`, `_method_metadata`
- **Current responsibility:** Emits RedLamp encoder fields, checkpoint SHA-256
  and shared `adaptation_momentum`.
- **Change:** Emit the selected model family, complete checkpoint role,
  checkpoint SHA-256, trainable parameter scope, optimizer identity and native
  method settings. Keep legacy RedLamp fields only for a retained variant.
- **Reason:** Threshold artifacts and benchmark reports must prove which model
  produced the score.
- **Inputs:** Validated model/checkpoint and method state.
- **Outputs:** JSON-serializable method metadata.
- **Errors:** Raise if required identity fields are missing instead of emitting
  partial provenance.
- **Dependencies:** Steps 0.4, 1.2 and method constructors.
- **Compatibility:** Existing report consumers may continue reading common keys;
  new native keys must be additive or versioned.
- **Verification:** Metadata test checks finite values, stable names and exact
  checkpoint identity.

### Tests

#### Complete state and trainable boundary

- **Location:** `tests/online/test_redlamp_encoder_checkpoint.py` for the
  retained loader and a proposed method-model test beside the online baseline
  contract tests for the native path.
- **Level:** Unit/contract.
- **Setup:** Build a deterministic selected model and checkpoint with one
  frozen detector component and one allowed adaptation component.
- **Action:** Load, score, adapt once and compare state dicts before/after.
- **Expected result:** Required state loads; only allowed parameters/state change;
  scoring output has the expected shape.
- **Edge cases:** Missing component, wrong feature dimension, no window and
  optimizer accidentally attached to a frozen parameter.

### Verification

#### Automated

- [ ] Constructor validation rejects invalid selected model/checkpoint settings.
- [ ] Full model state round trip preserves deterministic score output.
- [ ] Frozen/trainable parameter test passes for both methods.
- [ ] Method metadata contains the selected model and checkpoint contract.

#### Manual

- [ ] Inspect one metadata object and confirm it does not call an approximation
  “method-faithful”.

### Risks and recovery

- **Risk:** Reusing a partial RedLamp state gives plausible but invalid scores.
- **Mitigation:** Validate required state keys and model methods before scoring.
- **Verification:** State-dict and deterministic forward tests.
- **Recovery:** Revert to the retained historical variant without mixing its
  outputs with native results.

### Complete when

- Both baselines have an explicit model/checkpoint owner.
- The complete state and trainable boundary are tested.
- Method metadata identifies the exact model contract.

## Phase 2: Replace full-sequence scoring with causal score-then-adapt

### Goal

Make each baseline process one stride-one window in order so that the score for a
window is recorded before that same window can update the method state used by
the next window.

### Dependencies

- Phase 1 model/state boundary.
- `OnlineStreamingBaselineProtocol` and `build_stride1_batcher` in
  `src/baselines/online/base.py`.
- Absolute-range selection in `src/protocols/online_stream_range.py`.
- Existing `calibrate()` threshold interface.

### Detailed atomic steps

#### Step 2.1: Define the minimum lifecycle interface

- **File:** `src/baselines/online/base.py`
- **Symbol:** `OnlineStreamingBaselineProtocol`
- **Current responsibility:** Requires only `calibrate()` and one
  full-sequence `run_sequence()` call.
- **Change:** Add the smallest method-owned lifecycle needed by the causal path,
  such as initialization, one-window scoring, adaptation and state metadata, or
  keep those operations private to `run_sequence()` if the public protocol can
  remain stable. Choose the smaller boundary after checking callers.
- **Reason:** A boolean `did_update` cannot represent optimizer state, candidate
  pools, Detrender state or loss output.
- **Inputs:** One window/batch `[B, L, D]`, threshold and protocol settings.
- **Outputs:** Score state before update, adaptation result after update and
  serializable state metadata.
- **Errors:** Reject scoring before initialization/calibration, malformed tensor
  shape, missing threshold or non-finite score/loss.
- **Dependencies:** Phase 1.
- **Compatibility:** Preserve the runner's `calibrate()`/`run_sequence()` call
  boundary unless a confirmed caller requires the internal lifecycle directly.
- **Verification:** Protocol conformance test for all online baseline classes.

#### Step 2.2: Decide whether to reuse the existing batcher

- **File:** `src/baselines/online/base.py`
- **Symbol:** `build_stride1_batcher`
- **Current responsibility:** Builds an `SMDOnlineStream` with stride 1 and a
  batch size of 1, without view noise or legacy views.
- **Change:** Verify whether its emitted fields and shapes are sufficient for the
  deep baseline score/adapt path. Reuse it if they are; otherwise keep the
  existing direct window slicing and document why the helper cannot be reused.
- **Reason:** The project already has a sequential stream helper; adding a
  second incompatible windowizer would create unnecessary paths.
- **Inputs:** Selected sequence, window size and stream metadata.
- **Outputs:** Ordered windows with absolute/global index information.
- **Errors:** Reject missing labels/masks only when the selected contract truly
  requires them; never use labels to select an update.
- **Dependencies:** Existing `SMDOnlineStream` and `OnlineWindowBatcher`.
- **Compatibility:** Preserve stride-one and point-index alignment.
- **Verification:** Compare emitted starts/ends with
  `window_scores_to_causal_endpoint_scores` on a short sequence.

#### Step 2.3: Replace precomputation in `run_sequence`

- **File:** `src/baselines/online/adaptive.py`
- **Symbol:** `AdaptiveStreamingBaselineBase.run_sequence`
- **Current responsibility:** Computes all window, point and EWMA scores at
  lines 382–399 before entering the update loop.
- **Change:** For each valid endpoint in order, construct the current window,
  compute its raw and latent/representation score with the current model state,
  compute the EWMA from the previous accepted score, record the prediction, then
  call the method-owned adaptation operation before the next endpoint.
- **Reason:** The reference predictor scores and records one batch, calls
  `adapter.adapt(inputs, scores)`, and then advances.
- **Inputs:** Selected sequence `[T, D]`, threshold, EWMA weights and method
  state.
- **Outputs:** Ordered metric history and records with one record per valid
  endpoint.
- **Errors:** Skip only the existing warm-up points whose EWMA is undefined;
  raise on non-finite model score, invalid threshold or invalid window slice.
- **Dependencies:** Steps 2.1–2.2 and method adaptation implementations in
  Phases 3–4.
- **Compatibility:** Preserve `point_index = absolute_start + endpoint - 1`,
  `window_start_index`, `window_end_index`, selected range and prediction rule
  `score > threshold`.
- **Verification:** Two-window test shows window 2 sees window 1's update only
  when adaptation is enabled; window 1's recorded score is unchanged.

#### Step 2.4: Implement incremental EWMA state

- **File:** `src/protocols/point_scores.py` and/or the method-owned streaming
  state in `src/baselines/online/adaptive.py`
- **Symbol:** existing `ewma_scores` semantics and proposed one-step EWMA state
- **Current responsibility:** `ewma_scores` transforms a complete score array and
  uses the previous smoothed value, with the first finite score unchanged.
- **Change:** Reproduce that exact recurrence one point at a time rather than
  recomputing an array before adaptation.
- **Reason:** The causal loop must preserve the established calibration and
  prediction semantics.
- **Inputs:** Current finite raw point score, previous EWMA or `None`, current
  and previous weights.
- **Outputs:** Current EWMA and updated previous-EWMA state.
- **Errors:** Reject weights whose sum is not 1.0 or non-finite current score.
- **Dependencies:** Step 2.3.
- **Compatibility:** The first finite point must equal the raw point score, as in
  `ewma_scores`.
- **Verification:** Compare incremental output with existing vectorized
  `ewma_scores` on deterministic arrays containing NaN warm-up points.

#### Step 2.5: Separate baseline-native update reason from THESIS triage

- **File:** `src/baselines/online/adaptive.py`, `src/baselines/online/base.py`
  and `scripts/benchmarks/run_online_streaming_benchmark.py`
- **Symbol:** `classify_legacy_baseline_window`, `_should_update`,
  `build_online_record_schema`, `_normalize_online_records`
- **Current responsibility:** The adaptive path classifies every score through
  THESIS legacy triage and passes the result to M2N2/CANDI policy predicates.
- **Change:** Remove triage from the native update decision. Preserve the shared
  record shape only as required by existing consumers; if the string field must
  remain, emit an explicit non-native sentinel such as `not_used` and add a
  method-owned update reason/state field after checking consumers.
- **Reason:** CANDI FPM and M2N2 pseudo-anomaly masking are not THESIS
  four-region triage.
- **Inputs:** Native score, threshold, method state and method selection result.
- **Outputs:** Native update decision, reason, loss and pool/buffer counts.
- **Errors:** Do not call `classify_legacy_baseline_window` from native M2N2/CANDI
  adaptation. Fail tests if labels or legacy triage affect the decision.
- **Dependencies:** Phase 0 terminology and Phase 3/4 method rules.
- **Compatibility:** Leave `src/engine/online_tta/` and frozen baselines that
  genuinely use legacy triage unchanged.
- **Verification:** Monkeypatch the legacy classifier to raise and prove native
  M2N2/CANDI execution does not call it.

#### Step 2.6: Preserve the shared online record contract

- **File:** `src/baselines/online/base.py`
- **Symbol:** `build_online_record_schema`
- **Current responsibility:** Emits entity/global indices, score fields,
  threshold, prediction, `triage_decision`, `did_update`, loss and verification
  buffer size.
- **Change:** Keep the common fields needed by the runner and reports. Add only
  method-owned fields that are required to audit M2N2/CANDI, such as update
  reason, adaptation pool size, mask count or method state version. Define their
  absent-value representation and validate finite numeric values.
- **Reason:** A method-fidelity claim needs evidence of the native update while
  existing artifact consumers need stable common fields.
- **Inputs:** Common score state and method-specific update result.
- **Outputs:** JSON-serializable record.
- **Errors:** Reject a record with mismatched indices, non-finite score/loss or
  a prediction inconsistent with `ewma_point_score > threshold`.
- **Dependencies:** Steps 2.3–2.5 and the artifact schema.
- **Compatibility:** Do not remove existing fields until all consumers and tests
  are checked; do not repurpose `triage_decision` to mean a CANDI candidate
  class.
- **Verification:** Record-schema tests cover one no-update and one update
  record for each method.

#### Step 2.7: Define state serialization boundary

- **File:** `src/baselines/online/adaptive.py` and method owners
- **Symbol:** proposed method-owned state export/import functions, only if a
  checkpoint/resume caller exists after repository search
- **Current responsibility:** `run_sequence()` keeps state in memory and emits
  no optimizer, Detrender or candidate-pool state.
- **Change:** If the benchmark requires resume/audit state, serialize model
  parameters, optimizer state, Detrender state, CANDI references and pools. If
  no caller requires resume, emit a state summary in metadata without adding a
  new persistence format.
- **Reason:** Add state only for a concrete reproducibility requirement.
- **Inputs:** Current method state.
- **Outputs:** Validated state dictionary or report metadata.
- **Errors:** Reject state with missing method version or incompatible model
  contract.
- **Dependencies:** Steps 1.5 and 2.3.
- **Compatibility:** Do not persist every forward output; retain only reportable
  state and selected diagnostics.
- **Verification:** Round-trip state test when serialization is in scope.

### Tests

#### Causal order and index alignment

- **Location:** `tests/online/test_online_streaming_baseline_contracts.py` and a
  focused proposed test module in `tests/online/` if the shared test becomes too
  broad.
- **Level:** Unit/integration.
- **Setup:** Use a deterministic two- or three-window sequence, a model whose
  parameter visibly changes after adaptation and a fixed threshold.
- **Action:** Run once with adaptation enabled and once disabled.
- **Expected result:** Window `t` uses pre-update parameters; later windows may
  differ only in the enabled run; absolute indices and stream steps are equal.
- **Edge cases:** Sequence shorter than one window, NaN warm-up score, selected
  absolute start greater than zero and final endpoint.

### Verification

#### Automated

- [ ] Incremental EWMA matches `ewma_scores` on the same raw sequence.
- [ ] Native M2N2/CANDI execution does not invoke legacy THESIS triage.
- [ ] Record indices and prediction rule remain correct.
- [ ] State round-trip passes if resume/audit persistence is selected.

#### Manual

- [ ] Inspect one short trace in this exact order: receive window, score, record,
  adapt, advance to next window.

### Risks and recovery

- **Risk:** Sequential scoring changes thresholds and runtime cost.
- **Mitigation:** Recalibrate after the loop is fixed and run one smoke before
  scaling.
- **Verification:** Compare old/new record counts and index ranges; do not
  compare old/new metric values as if they used the same protocol.
- **Recovery:** Restore the historical runner/configuration identity and keep
  new outputs under a separate non-comparable identity.

### Complete when

- The adaptive path is causal.
- EWMA and index semantics remain stable.
- Native update decisions do not depend on THESIS triage or labels.
- Records expose enough state to audit the update.

## Phase 3: Implement M2N2 native adaptation

### Goal

Replace M2N2's current threshold predicate and reference mean/std update with the
reference Detrender, timestep pseudo-anomaly mask, masked reconstruction loss and
optimizer update.

### Dependencies

- Phases 0–2.
- Selected model's reconstruction output `[B, L, D]`.
- Reference `Detrender` semantics in
  `bsc-thesis-ref-codebases/CANDI-main/models/normalizer.py` and
  `bsc-thesis-ref-codebases/CANDI-main/tta/m2n2/adapter_m2n2.py`.
- M2N2 gamma/step settings selected in the contract.

### Detailed atomic steps

#### Step 3.1: Replace the M2N2 constructor settings

- **File:** `src/baselines/online/m2n2.py`
- **Symbol:** `M2N2StreamingBaseline.__init__`
- **Current responsibility:** Passes `adaptation_momentum`, CNN settings and
  encoder-only checkpoint to the shared base.
- **Change:** Add and validate method-native settings: Detrender `gamma`,
  optimizer name/learning rate/weight decay, adaptation step count and explicit
  trainable scope. Remove `adaptation_momentum` from the native path; retain it
  only in a separately named compatibility variant.
- **Reason:** The reference M2N2 update is gradient-based and uses Detrender
  gamma, not feature mean/std momentum.
- **Inputs:** Values from the method config; `gamma` must be in the range the
  selected reference contract permits, optimizer settings must be finite and
  step count must be positive.
- **Outputs:** Validated M2N2 method object with optimizer configuration stored.
- **Errors:** Raise on missing gamma, invalid gamma, unsupported optimizer,
  non-positive steps or missing selected checkpoint.
- **Dependencies:** Phase 0 decision and Step 1.1.
- **Compatibility:** Keep the old constructor only if it is explicitly exposed
  under a non-faithful variant name.
- **Verification:** Constructor validation tests for default/reference values
  and each invalid branch.

#### Step 3.2: Add mean-only Detrender state

- **File:** `src/baselines/online/m2n2.py`; reference:
  `bsc-thesis-ref-codebases/CANDI-main/models/normalizer.py`
- **Symbol:** proposed `Detrender` state owner or an explicitly named adapter
  component; reference `_update_statistics`, `_normalize`, `_denormalize`
- **Current responsibility:** The current base updates NumPy `reference_mean_`
  and `reference_std_` with `adaptation_momentum`.
- **Change:** Maintain a tensor mean with shape `[1, 1, D]`. At each adaptation
  batch, compute `mu = mean(x, dim=all dimensions except feature, keepdim=True)`
  without gradient and update `mean.lerp_(mu, 1 - gamma)` before the model's
  normalized forward path, matching the reference.
- **Reason:** M2N2 reference normalizes by a mean-only Detrender; it does not
  update a feature standard deviation with the current custom EMA.
- **Inputs:** Float tensor `[B, L, D]`, finite gamma and current Detrender mean.
- **Outputs:** Updated Detrender state and normalized input `x - mean` if the
  selected model uses it.
- **Errors:** Reject rank other than 3, feature mismatch, non-finite input or
  mean update.
- **Dependencies:** Selected model normalization contract from Phase 0.
- **Compatibility:** Do not mutate NumPy reference std in the native M2N2 path.
- **Verification:** Deterministic batch test compares the updated mean with the
  reference `lerp_` formula and checks no gradient is attached.

#### Step 3.3: Define M2N2 score computation

- **File:** `src/baselines/online/m2n2.py` and selected model owner
- **Symbol:** proposed M2N2 score method; reference adapter forward/reconstruction
  path
- **Current responsibility:** Base `_score_backbone_windows` computes one mean
  MSE score over `[L, D]` and a latent absolute-value score from the shared CNN.
- **Change:** Use the selected model's reconstruction and the exact score space
  agreed in Phase 0. Keep per-timestep reconstruction errors available for the
  adaptation mask rather than reducing immediately to one scalar.
- **Reason:** The scalar prediction score and timestep adaptation mask serve
  different purposes.
- **Inputs:** Current window batch `[B, L, D]` and current model/Detrender state.
- **Outputs:** Scalar score per sample, reconstruction tensor and per-timestep
  error `A` with shape `[B, L]` after feature reduction.
- **Errors:** Raise on shape mismatch or non-finite reconstruction/error.
- **Dependencies:** Steps 1.2, 2.3 and 3.2.
- **Compatibility:** The reported point score must remain finite and align to the
  window endpoint; do not use test labels.
- **Verification:** Hand-computed `[B, L, D]` fixture confirms `A = mean((recon -
  x)^2, dim=-1)` and the reported score uses the selected reduction.

#### Step 3.4: Construct the pseudo-anomaly mask

- **File:** `src/baselines/online/m2n2.py`; reference:
  `bsc-thesis-ref-codebases/CANDI-main/tta/m2n2/adapter_m2n2.py`
- **Symbol:** reference `get_mask`; proposed method-owned mask operation
- **Current responsibility:** No M2N2 loss mask exists; `_should_update` only
  checks raw and EWMA threshold predicates.
- **Change:** Compute `ytilde = A >= threshold` and `mask = (ytilde == 0)` using
  the method's configured threshold in the same score space as `A`.
- **Reason:** Reference M2N2 excludes pseudo-anomalous timesteps from the update
  objective rather than deciding adaptation with a whole-window predicate.
- **Inputs:** Per-timestep error `A` and scalar threshold.
- **Outputs:** Boolean mask `[B, L]`, masked count and optional diagnostics.
- **Errors:** Reject threshold with incompatible score space, non-finite values
  or a mask shape different from `A`.
- **Dependencies:** Step 3.3 and threshold contract.
- **Compatibility:** `mask` is internal adaptation state; it must not read
  `point_labels`.
- **Verification:** Fixture with values below, equal to and above threshold
  proves equality uses the reference `>=` rule.

#### Step 3.5: Implement masked reconstruction loss

- **File:** `src/baselines/online/m2n2.py`
- **Symbol:** proposed `calculate_loss`/`adapt_batch`; reference M2N2 adapter
  loss construction
- **Current responsibility:** `loss_total` is always `None`; no backward pass
  occurs.
- **Change:** Compute the reference masked objective from `A` and `mask`,
  specifically the mean of `A * mask` over the selected timestep/feature
  reduction. Keep the loss attached to the graph until backward.
- **Reason:** This is the central M2N2 method update, and it cannot be replaced
  by updating NumPy reference statistics.
- **Inputs:** Current window tensor, reconstruction and boolean mask.
- **Outputs:** Scalar finite torch loss plus mask count.
- **Errors:** Define and test the all-masked case. Use an explicit finite policy
  selected from the reference behavior; do not silently divide by zero.
- **Dependencies:** Step 3.4.
- **Compatibility:** Report detached numeric `loss_total` only after the update;
  never serialize a live tensor.
- **Verification:** Compare deterministic loss with a hand-computed masked MSE;
  test no-mask and all-mask edge cases.

#### Step 3.6: Create the optimizer and perform configured steps

- **File:** `src/baselines/online/m2n2.py` and
  `src/baselines/online/adaptive.py`
- **Symbol:** proposed M2N2 optimizer owner and `adapt_batch`; reference
  `Adapter.optimizer`, `zero_grad`, `backward`, `step`
- **Current responsibility:** No optimizer exists in the current baseline.
- **Change:** Build the optimizer over exactly the selected trainable parameters.
  For each configured step, execute `zero_grad()`, compute the masked loss,
  call `backward()`, optionally apply only the selected gradient policy, and call
  `step()`. Restore evaluation mode after adaptation.
- **Reason:** Reference M2N2 sets `STEPS=1` in its adapter construction and then
  performs a gradient update.
- **Inputs:** Current batch, threshold, optimizer and step settings.
- **Outputs:** `did_update`, detached loss, mask count and updated model/
  Detrender state.
- **Errors:** Raise on missing optimizer, empty trainable parameter list,
  non-finite loss/gradient or optimizer failure. Do not claim an update when no
  step ran.
- **Dependencies:** Steps 1.4, 3.1 and 3.5.
- **Compatibility:** The update occurs after the current score is recorded and
  before the next window score.
- **Verification:** Parameter-delta test proves allowed parameters change,
  frozen parameters do not, and a second window can observe the changed model.

#### Step 3.7: Remove the predicate-only M2N2 path

- **File:** `src/baselines/online/m2n2.py` and
  `src/baselines/online/adaptive.py`
- **Symbol:** `M2N2StreamingBaseline._should_update`,
  `AdaptiveStreamingBaselineBase._update_reference`
- **Current responsibility:** M2N2 updates reference mean/std when raw and EWMA
  scores are both below threshold.
- **Change:** Remove these functions from the native M2N2 update decision. Route
  the decision through `adapt_batch` and keep any legacy predicate only behind
  an explicitly named compatibility variant.
- **Reason:** The current predicate performs no reference gradient update and is
  not part of the reference M2N2 algorithm.
- **Inputs:** Current score/reconstruction state and M2N2 mask.
- **Outputs:** Native update result.
- **Errors:** Fail a method-fidelity test if `_update_reference` changes native
  M2N2 state or if `_should_update` is called by the native path.
- **Dependencies:** Steps 2.5, 3.4–3.6.
- **Compatibility:** Other frozen baselines may retain their own update policy.
- **Verification:** Monkeypatch or spy on `_update_reference` and assert it is
  unused by native M2N2.

#### Step 3.8: Record M2N2-specific diagnostics

- **File:** `src/baselines/online/m2n2.py`,
  `src/baselines/online/base.py` and benchmark record writer
- **Symbol:** method metadata, metric history and record creation
- **Current responsibility:** Emits `loss_total: None`, a triage decision and
  fixed zero verification-buffer size.
- **Change:** Emit M2N2 update reason, optimizer-step count, mask count, loss,
  Detrender state/version and trainable-scope metadata. Keep common score and
  index fields unchanged.
- **Reason:** Reviewers must be able to verify that an observed update was the
  M2N2 masked optimization step.
- **Inputs:** Detached update result and current stream indices.
- **Outputs:** JSON-safe diagnostics with finite numeric values.
- **Errors:** Reject live tensors, NaN/Inf loss or inconsistent `did_update` and
  step count.
- **Dependencies:** Steps 2.6 and 3.6.
- **Compatibility:** Do not expose labels or use them to populate adaptation
  diagnostics.
- **Verification:** One record fixture checks update/no-update consistency.

### Tests

#### M2N2 Detrender update

- **Location:** Existing M2N2-focused test location under `tests/online/`, or a
  clearly named addition to `tests/online/test_online_streaming_baseline_contracts.py`.
- **Level:** Unit.
- **Setup:** Fixed tensor `[B, L, D]`, fixed initial mean and gamma.
- **Action:** Apply one Detrender update.
- **Expected result:** Mean equals the reference `lerp_` result and has no grad.
- **Edge cases:** Batch size one, constant features, non-finite input and wrong
  feature dimension.

#### M2N2 mask and loss

- **Location:** Same M2N2-focused test location.
- **Level:** Unit.
- **Setup:** Hand-computed reconstruction errors around the threshold.
- **Action:** Construct mask and masked loss.
- **Expected result:** Values `>= threshold` are excluded; loss equals the
  reference masked objective and is finite under the selected all-masked policy.
- **Edge cases:** All positions masked, no positions masked and threshold equal
  to an error.

#### M2N2 optimizer boundary and causality

- **Location:** M2N2-focused online test plus shared causal test.
- **Level:** Integration.
- **Setup:** One allowed trainable parameter, one frozen parameter and two
  sequential windows.
- **Action:** Score/adapt first window, then score second window.
- **Expected result:** First score is pre-update; allowed state changes; frozen
  state does not; second score reflects the update when the update is accepted.
- **Edge cases:** Zero accepted mask, optimizer step count greater than one and
  update failure.

### Verification

#### Automated

- [ ] Constructor and gamma/optimizer validation pass.
- [ ] Detrender update matches the reference formula.
- [ ] Pseudo-anomaly mask uses the exact `>=` threshold rule.
- [ ] Masked loss matches a deterministic hand calculation.
- [ ] Optimizer changes only the permitted parameters.
- [ ] Native M2N2 never calls `_should_update` or `_update_reference`.
- [ ] `loss_total` is finite whenever `did_update` is true.

#### Manual

- [ ] Inspect one M2N2 record and trace score, mask count, loss, optimizer step
  and next-window effect.

### Risks and recovery

- **Risk:** Normalizing before/after the model differs from the selected native
  training contract.
- **Mitigation:** Confirm the selected model's normalizer and test the exact
  order before calibrating thresholds.
- **Verification:** Deterministic forward test with Detrender state inspection.
- **Recovery:** Stop the smoke run, retain old artifacts, and return to Phase 0
  if the checkpoint cannot support the selected normalization.

### Complete when

- M2N2 uses Detrender, pseudo-mask, masked loss and optimizer steps.
- No reference-statistics predicate remains in the native path.
- Unit/integration tests prove parameter boundary and causal feedback.

## Phase 4: Implement CANDI native adaptation

### Goal

Implement validation-derived false-positive mining (FPM), separate hard and
moderate candidate pools, SANA residual modules and the reference minimum-sample
adaptation gate without using test labels or THESIS triage.

### Dependencies

- Phases 0–2.
- Selected model must expose reconstruction and `get_representations`-equivalent
  behavior for the CANDI contract.
- Reference `adapter_candi.py`, CANDI config defaults and selected FPM/SANA
  settings.

### Detailed atomic steps

#### Step 4.1: Define CANDI method settings

- **File:** `src/baselines/online/candi.py`
- **Symbol:** `CANDIStreamingBaseline.__init__`
- **Current responsibility:** Passes shared CNN and `adaptation_momentum` settings
  to the base.
- **Change:** Add explicit settings for `USE_FPM`, `USE_SANA`, `USE_HARD`,
  `USE_MODERATE`, `MIN_SAMPLES`, SANA gating/architecture, optimizer and TTA
  steps. Validate combinations before fitting.
- **Reason:** The current constructor cannot express the reference CANDI
  protocol.
- **Inputs:** Method config and selected checkpoint/model contract.
- **Outputs:** Validated CANDI object with empty candidate state before
  calibration.
- **Errors:** Reject negative/zero `MIN_SAMPLES`, disabled selection with no
  valid fallback explicitly defined, invalid covariance/reference settings or
  SANA optimizer values.
- **Dependencies:** Phase 0 decision and Step 1.1.
- **Compatibility:** Keep old `adaptation_momentum` only in a named variant.
- **Verification:** Constructor tests cover reference defaults and invalid
  combinations.

#### Step 4.2: Build validation representations

- **File:** `src/baselines/online/candi.py`
- **Symbol:** proposed CANDI calibration/initialization method; reference
  `CANDIAdapter.__init__` representation setup
- **Current responsibility:** Base calibration scores clean validation but does
  not retain representations or candidate reference sets.
- **Change:** During calibration/initialization, compute representations from
  clean validation inputs only and retain the minimum state required for FPM.
  Do not pass test sequence or test labels into this initialization.
- **Reason:** CANDI uses validation-derived latent references to identify likely
  false positives during test-time adaptation.
- **Inputs:** Clean validation windows, score array and model representation API.
- **Outputs:** Validation representations, mean/covariance inputs and method
  reference state.
- **Errors:** Reject empty validation set, representation dimension mismatch,
  non-finite representation or missing representation method.
- **Dependencies:** Steps 1.2, 2.2 and Phase 0 model contract.
- **Compatibility:** Threshold calibration remains clean-validation-only and
  uses the same selected stream protocol.
- **Verification:** Test that changing test labels or test inputs does not
  change validation reference state.

#### Step 4.3: Compute covariance pseudoinverse and validate reference size

- **File:** `src/baselines/online/candi.py`
- **Symbol:** proposed CANDI reference-state builder; reference
  `val_representations_cov` and `torch.linalg.pinv`
- **Current responsibility:** No covariance or Mahalanobis reference state.
- **Change:** Compute representation covariance and its pseudoinverse using the
  selected reference behavior. Validate finite dimensions and minimum sample
  requirements before enabling FPM.
- **Reason:** Candidate selection depends on Mahalanobis proximity in latent
  space.
- **Inputs:** Validation representation matrix `[N, R]`.
- **Outputs:** Mean/covariance/pseudoinverse and representation dimension.
- **Errors:** Raise an explicit data/configuration error when `N` is too small,
  covariance shape is invalid or pseudoinverse is non-finite.
- **Dependencies:** Step 4.2.
- **Compatibility:** Do not silently replace pseudoinverse with an unrelated
  diagonal or identity covariance.
- **Verification:** Singular deterministic covariance fixture confirms the
  selected pseudoinverse policy and finite output.

#### Step 4.4: Construct hard validation references

- **File:** `src/baselines/online/candi.py`
- **Symbol:** proposed hard-reference builder; reference `topk`, `topk_indices`,
  `topk_representations`
- **Current responsibility:** No top-k validation representation set.
- **Change:** Compute top-k count from the selected anomaly-ratio configuration,
  select the highest validation scores and retain their representations for hard
  proximity testing.
- **Reason:** Reference CANDI's hard pool requires high-score candidates that
  are close to validation hard references.
- **Inputs:** Validation scores, validation representations and configured
  anomaly ratio.
- **Outputs:** Hard reference representation matrix and selected count.
- **Errors:** Reject empty reference set, invalid ratio or a top-k count that
  becomes zero unless the selected contract defines a minimum of one.
- **Dependencies:** Steps 4.2–4.3 and Phase 0 configuration decision.
- **Compatibility:** Store only references, not all forward outputs, unless the
  existing artifact policy explicitly requires them.
- **Verification:** Known score ordering fixture returns the expected top-k
  rows.

#### Step 4.5: Construct moderate validation references

- **File:** `src/baselines/online/candi.py`
- **Symbol:** proposed moderate-reference builder; reference Q1/Q3 selection
- **Current responsibility:** No moderate reference set.
- **Change:** Select validation representations in the configured moderate score
  interval and retain their representations separately from hard references.
- **Reason:** Hard and moderate candidates have different score and proximity
  rules and must not share one pool or one reference set.
- **Inputs:** Validation scores and representations, selected interval rule.
- **Outputs:** Moderate reference representation matrix and interval metadata.
- **Errors:** Fail explicitly when the interval produces no references; do not
  silently use hard references as a moderate fallback.
- **Dependencies:** Step 4.2 and selected reference configuration.
- **Compatibility:** No test label is allowed in the selection.
- **Verification:** Deterministic quartile fixture proves strict/inclusive
  boundary behavior according to the selected reference rule.

#### Step 4.6: Select hard candidates online

- **File:** `src/baselines/online/candi.py`
- **Symbol:** proposed `select_candidates`/hard-selection operation; reference
  `get_samples_to_adapt` hard branch
- **Current responsibility:** `_should_update` returns true for `gray_zone` or
  `pnn_candidate` and then updates feature reference statistics.
- **Change:** For each current test batch, compute representations. For each
  representation, calculate squared Mahalanobis distance to hard references,
  reduce across reference rows using the selected `any` rule, combine it with
  the high-score threshold rule and append only accepted inputs to the hard
  pool.
- **Reason:** This is CANDI FPM hard candidate selection.
- **Inputs:** Current batch `[B, L, D]`, current scores, hard references,
  covariance pseudoinverse and chi-square/configured threshold.
- **Outputs:** Hard boolean mask, selected inputs and updated hard-pool count.
- **Errors:** Reject shape mismatch, non-finite distance or missing hard
  references when `USE_HARD` is enabled.
- **Dependencies:** Steps 4.3–4.4 and Phase 2 causal loop.
- **Compatibility:** Never read `test_labels`; keep current score recorded before
  adding candidates for adaptation.
- **Verification:** Hand-computed distance fixture checks mask and pool contents.

#### Step 4.7: Select moderate candidates online

- **File:** `src/baselines/online/candi.py`
- **Symbol:** proposed moderate-selection operation; reference moderate branch
  in `get_samples_to_adapt`
- **Current responsibility:** No moderate candidate pool or Mahalanobis test.
- **Change:** Apply the selected non-anomalous/moderate score rule and
  Mahalanobis proximity to moderate references. Append accepted inputs only to
  the moderate pool.
- **Reason:** Moderate FPM candidates must not be merged with hard candidates.
- **Inputs:** Current batch, scores, moderate references, covariance
  pseudoinverse and configured thresholds.
- **Outputs:** Moderate boolean mask, selected inputs and pool count.
- **Errors:** Reject missing moderate references when enabled and non-finite
  distance; define no-candidate behavior as an empty pool.
- **Dependencies:** Steps 4.3, 4.5 and 4.6.
- **Compatibility:** Preserve separate hard/moderate diagnostics and do not
  reuse THESIS gray-zone labels.
- **Verification:** Fixture tests score boundaries and reference proximity
  independently.

#### Step 4.8: Implement `MIN_SAMPLES` pool gates

- **File:** `src/baselines/online/candi.py`
- **Symbol:** proposed pool state and adaptation gate; reference
  `n_samples_to_adapt_hard`, `n_samples_to_adapt_moderate`
- **Current responsibility:** Every accepted CANDI predicate immediately updates
  reference mean/std; no candidate accumulation exists.
- **Change:** Keep hard and moderate pools separately. Trigger hard adaptation
  only when the hard pool count reaches `MIN_SAMPLES`; apply the same rule to the
  moderate pool. Do not adapt partial pools.
- **Reason:** The reference uses a minimum-sample gate before each pool update.
- **Inputs:** Selected candidate tensors and `MIN_SAMPLES`.
- **Outputs:** `did_update`, pool count before/after, selected pool identity and
  reset state.
- **Errors:** Reject a non-positive gate or a pool with inconsistent count and
  tensor length.
- **Dependencies:** Steps 4.6–4.7.
- **Compatibility:** If both pools reach the gate in one batch, apply the
  selected reference order and report both updates separately or as one record
  with explicit counts; do not silently merge pools.
- **Verification:** Accumulate `MIN_SAMPLES-1`, assert no update; add one sample,
  assert update and pool reset.

#### Step 4.9: Add SANA input/output modules

- **File:** `src/baselines/online/candi.py`; reference `SANA`, `sana_in` and
  `sana_out` in `bsc-thesis-ref-codebases/CANDI-main/tta/candi/adapter_candi.py`
- **Symbol:** proposed method-owned SANA modules
- **Current responsibility:** No trainable SANA residual modules.
- **Change:** When `USE_SANA` is enabled, freeze the pretrained detector and add
  trainable input/output residual modules with the selected reference type,
  gating initialization and shape. Attach only their parameters to the CANDI
  optimizer.
- **Reason:** Reference CANDI adapts SANA residuals rather than the current
  feature mean/std.
- **Inputs:** Selected model dimensions, SANA config and input `[B, L, D]`.
- **Outputs:** Adapted input/reconstruction path with `sana_in` and `sana_out`.
- **Errors:** Reject unsupported SANA type, dimension mismatch or empty trainable
  parameter set.
- **Dependencies:** Phase 0 model surface and Step 4.1.
- **Compatibility:** Do not change detector parameters when SANA is enabled;
  if SANA is disabled, follow the selected reference full-model adaptation rule
  explicitly and record it.
- **Verification:** Parameter-scope test and forward-shape test for both SANA
  enabled/disabled settings.

#### Step 4.10: Implement CANDI reconstruction MSE

- **File:** `src/baselines/online/candi.py`
- **Symbol:** proposed `calculate_loss`/`adapt_batch`; reference `MLPAdapter` and
  `TimesNetAdapter.calculate_loss`
- **Current responsibility:** No CANDI loss; `loss_total` remains `None`.
- **Change:** Apply SANA input residual, run the selected model, apply SANA output
  residual and compute reconstruction MSE against the candidate input. Keep the
  graph through backward and detach only the recorded loss.
- **Reason:** This is the native CANDI SANA adaptation objective.
- **Inputs:** One complete hard or moderate candidate pool `[N, L, D]`.
- **Outputs:** Finite scalar MSE and gradients for allowed SANA parameters.
- **Errors:** Reject empty pool, wrong shape, non-finite reconstruction/loss or
  missing SANA module when enabled.
- **Dependencies:** Step 4.9 and pool gates.
- **Compatibility:** The loss is computed only on method-selected inputs and
  never on labels.
- **Verification:** Hand-computed MSE fixture and gradient-scope test.

#### Step 4.11: Run CANDI optimizer steps and reset pools

- **File:** `src/baselines/online/candi.py`
- **Symbol:** proposed `adapt_batch`; reference `CANDIAdapter.adapt`
- **Current responsibility:** No optimizer or pool reset.
- **Change:** For each eligible pool, temporarily enter train mode, run the
  configured number of optimizer steps, restore mode, record detached loss and
  clear only the pool that was adapted. Keep the other pool intact.
- **Reason:** Reference CANDI adapts hard and moderate pools independently.
- **Inputs:** Eligible pool, optimizer, SANA state and step count.
- **Outputs:** Update result containing pool name, before/after count, loss and
  parameter-delta summary.
- **Errors:** Raise on optimizer failure or non-finite loss; do not clear a pool
  when its update failed unless the recovery policy explicitly records it.
- **Dependencies:** Steps 4.8–4.10.
- **Compatibility:** Run after current window record and before next window score.
- **Verification:** Pool-reset and failure-recovery tests.

#### Step 4.12: Remove THESIS triage dependency and test-label leakage

- **File:** `src/baselines/online/candi.py`,
  `src/engine/online_tta/triage.py` and native online tests
- **Symbol:** `CANDIStreamingBaseline._should_update`,
  `classify_legacy_baseline_window`
- **Current responsibility:** CANDI updates when triage says `gray_zone` or
  `pnn_candidate`.
- **Change:** Remove native CANDI dependence on both labels and triage. Keep
  `classify_legacy_baseline_window` only for confirmed THESIS/frozen callers.
- **Reason:** CANDI's FPM selection is based on validation references, score and
  Mahalanobis proximity, not THESIS four-region names.
- **Inputs:** Current score, representation and CANDI reference state only.
- **Outputs:** Candidate masks, pool updates and native update reason.
- **Errors:** Test failure if changing test labels changes masks, pools, losses or
  parameters.
- **Dependencies:** Steps 2.5 and 4.6–4.11.
- **Compatibility:** Preserve unrelated `src/engine/online_tta/` behavior.
- **Verification:** Run the same stream twice with different labels and compare
  all adaptation state and score outputs.

#### Step 4.13: Record CANDI diagnostics

- **File:** `src/baselines/online/candi.py` and shared record/metadata owners
- **Symbol:** `_method_metadata`, metric history and record construction
- **Current responsibility:** Emits `update_on_gray_zone_and_pnn_candidate`,
  shared `adaptation_momentum` and zero verification-buffer size.
- **Change:** Emit FPM/SANA flags, reference counts, covariance dimension,
  hard/moderate pool counts, `MIN_SAMPLES`, selected pool, update reason, loss,
  step count and trainable scope.
- **Reason:** The report must show that CANDI used its own candidate mining and
  SANA update gate.
- **Inputs:** Detached method state and current record indices.
- **Outputs:** JSON-safe metadata and per-step diagnostics.
- **Errors:** Reject inconsistent counts, non-finite distances/losses or
  `did_update=true` without an optimizer step.
- **Dependencies:** Steps 2.6 and 4.8–4.11.
- **Compatibility:** Keep common record fields and label-free adaptation.
- **Verification:** One hard update, one moderate update and one no-update record
  fixture.

### Tests

#### CANDI validation isolation

- **Location:** CANDI-focused test location under `tests/online/`.
- **Level:** Unit/integration.
- **Setup:** Clean validation windows, fixed validation scores and two test
  sequences with identical inputs but different labels.
- **Action:** Build CANDI reference state and process both test sequences.
- **Expected result:** Validation references are identical; candidate masks,
  pools, losses and parameters are identical.
- **Edge cases:** Empty validation set, one validation representation and
  singular covariance.

#### CANDI selection and pool gate

- **Location:** CANDI-focused test location.
- **Level:** Unit.
- **Setup:** Hand-computed hard/moderate references, covariance inverse and
  scores around the selected thresholds.
- **Action:** Select candidates over multiple batches.
- **Expected result:** Hard/moderate masks follow the selected Mahalanobis and
  score rules; no update occurs before `MIN_SAMPLES`; the correct pool resets
  after update.
- **Edge cases:** Both pools reach the gate simultaneously, empty moderate
  reference set and disabled hard/moderate selection.

#### CANDI SANA parameter boundary

- **Location:** CANDI-focused test location.
- **Level:** Unit/integration.
- **Setup:** Model with identifiable detector and SANA parameters.
- **Action:** Run one eligible hard and moderate adaptation.
- **Expected result:** Only allowed SANA parameters change; MSE is finite; mode
  is restored; pool reset follows the configured result.
- **Edge cases:** SANA disabled, optimizer failure and non-finite loss.

### Verification

#### Automated

- [ ] Clean validation alone builds all FPM references.
- [ ] Hard and moderate masks match deterministic Mahalanobis fixtures.
- [ ] Test labels do not change selection or adaptation.
- [ ] Pools gate at `MIN_SAMPLES` and reset only after successful update.
- [ ] SANA parameter boundary and MSE tests pass.
- [ ] Native CANDI never calls THESIS triage or `_should_update`.
- [ ] All recorded distances, losses and counts are finite/consistent.

#### Manual

- [ ] Inspect a short CANDI trace showing validation references, candidate
  selection, pool growth, update trigger, loss and next-window score.

### Risks and recovery

- **Risk:** Singular covariance or empty reference set makes candidate selection
  undefined.
- **Mitigation:** Validate references and use only the selected reference
  pseudoinverse policy; fail explicitly when requirements are unmet.
- **Verification:** Singular/empty fixtures and clear error assertions.
- **Recovery:** Stop before adaptation, retain state for diagnosis and do not
  emit a method-faithful result.

### Complete when

- FPM, hard/moderate pools, SANA and `MIN_SAMPLES` are implemented.
- Test labels and THESIS triage do not affect native CANDI.
- Tests prove selection, update scope, pool reset and causal feedback.

## Phase 5: Update configurations, thresholds and artifacts

### Goal

Make the generated experiment configurations, threshold artifacts and output
metadata describe the selected native implementation and keep old approximation
results separate.

### Dependencies

- Phase 0 contract decision.
- Completed method constructors and lifecycle from Phases 1–4.
- Existing generator and threshold artifact validation.

### Detailed atomic steps

#### Step 5.1: Update the source configuration generator

- **File:** `scripts/benchmarks/generate_online_streaming_benchmark_configs.py`
- **Symbol:** `_baseline_kwargs`, `build_online_streaming_benchmark_config`
- **Current responsibility:** Emits CNN dimensions, RedLamp checkpoint and
  `adaptation_momentum` for both CANDI and M2N2.
- **Change:** Emit explicit M2N2 fields (`gamma`, optimizer, learning rate,
  weight decay, steps, trainable scope) and CANDI fields (FPM/SANA flags,
  hard/moderate flags, `MIN_SAMPLES`, SANA settings, optimizer and steps).
- **Reason:** The generator is the source of truth; hand-editing generated YAML
  would be overwritten.
- **Inputs:** Method, entity, seed and smoke/main mode.
- **Outputs:** A deterministic `baseline_kwargs` mapping with no stale
  approximation-only fields in the selected native path.
- **Errors:** Raise on unknown method, missing checkpoint mapping or invalid
  native setting.
- **Dependencies:** Phases 0, 3 and 4.
- **Compatibility:** Preserve entity IDs, seeds, window size, absolute ranges,
  output roots and smoke step limit.
- **Verification:** Generator test inspects one M2N2 and one CANDI main/smoke
  config.

#### Step 5.2: Regenerate generated YAML files

- **File:** `configs/experiment/online_benchmark/m2n2/` and
  `configs/experiment/online_benchmark/candi/`
- **Symbol:** Generated `main` and `smoke` YAML configurations
- **Current responsibility:** Store the current RedLamp/`adaptation_momentum`
  configuration.
- **Change:** Regenerate all affected files from the source generator after the
  constructor schema is final. Do not manually edit generated outputs.
- **Reason:** Every benchmark combination must use the same native settings and
  provenance.
- **Inputs:** Generator output and selected checkpoint inventory.
- **Outputs:** Valid YAML files with method-native kwargs.
- **Errors:** Stop if generated YAML contains stale native/variant names, an
  absent required key or a checkpoint path that does not exist in the selected
  inventory.
- **Dependencies:** Step 5.1.
- **Compatibility:** Do not rewrite unrelated stumpy/kmeans/iforest configs.
- **Verification:** Parse generated YAML and compare against the generator's
  returned mapping.

#### Step 5.3: Extend threshold artifact provenance

- **File:** `src/protocols/threshold_artifact.py` and
  `src/baselines/online/adaptive.py`
- **Symbol:** `build_threshold_artifact`, `validate_threshold_artifact`,
  `AdaptiveStreamingCalibration`
- **Current responsibility:** Validates clean-validation split, checkpoint SHA,
  threshold values, EWMA settings and generic provenance.
- **Change:** Add method/model contract, update protocol/version and native
  method settings to provenance only if the artifact schema can accept additive
  fields. If a schema version is required, increment it deliberately and update
  validators/tests together.
- **Reason:** A threshold calibrated before causal/native changes is not valid
  for the new score protocol.
- **Inputs:** Final native score protocol, clean validation scores, config path,
  resolved config identity and checkpoint identity.
- **Outputs:** Valid artifact with finite thresholds and complete provenance.
- **Errors:** Reject missing contract identity, mismatched checkpoint/config,
  non-finite threshold or non-clean calibration split.
- **Dependencies:** Steps 2.4, 3.3/4.2 and schema consumers.
- **Compatibility:** Preserve schema v3/v4 behavior for unrelated existing
  artifacts unless a deliberate version migration is approved.
- **Verification:** Existing artifact integrity tests plus native artifact
  fixtures.

#### Step 5.4: Recalibrate after native score/lifecycle completion

- **File:** `src/baselines/online/adaptive.py` and
  `scripts/benchmarks/run_online_streaming_benchmark.py`
- **Symbol:** `calibrate`, `run_online_streaming_benchmark`
- **Current responsibility:** Scores all clean-validation windows with the
  current model and computes q-quantile point/EWMA thresholds.
- **Change:** Recompute thresholds after the selected native model, Detrender or
  CANDI calibration references and causal score protocol are final. Keep test
  labels out of threshold computation.
- **Reason:** A changed model or update timing changes the score distribution.
- **Inputs:** Clean validation sequences, protocol EWMA weights, native model
  and threshold quantile.
- **Outputs:** New threshold artifact and calibration metadata.
- **Errors:** Reject empty validation, no finite scores, mismatched checkpoint
  SHA or method/config identity.
- **Dependencies:** Phases 2–4 and Step 5.3.
- **Compatibility:** Preserve `clean_validation_stride1_ewma` only when the
  exact native protocol still uses that score source; otherwise record the new
  source explicitly.
- **Verification:** Threshold artifact passes validator and is reproduced by a
  second calibration with the same seed/state.

#### Step 5.5: Define output identity and migration handling

- **File:** `scripts/benchmarks/run_online_streaming_benchmark.py`, output report
  writer and `documents/spec/online_benchmark_contract.md`
- **Symbol:** report `baseline_name`, `online_variant`, `method_metadata`,
  `online_execution` and artifact paths
- **Current responsibility:** Writes `online_metrics.json`,
  `online_records.json`, threshold artifact and report under the current output
  identity.
- **Change:** Include native/variant contract identity, method configuration,
  checkpoint identity, threshold source and update protocol. Use a new output
  identity or explicit `not_comparable` marker for old approximation runs.
- **Reason:** Old and new scores cannot be aggregated as one method.
- **Inputs:** Calibration metadata, generated config and selected stream
  metadata.
- **Outputs:** Auditable report and artifact paths.
- **Errors:** Refuse to write a native result when provenance is incomplete.
- **Dependencies:** Steps 5.2–5.4.
- **Compatibility:** Keep absolute stream range and record ordering unchanged.
- **Verification:** Artifact integrity test checks all paths and identity fields.

### Tests

#### Configuration generation

- **Location:** Existing benchmark/config tests if present; otherwise a focused
  test in the existing online benchmark test area.
- **Level:** Unit/contract.
- **Setup:** Generate one main and one smoke config for each method.
- **Action:** Parse YAML and inspect `baseline_kwargs`.
- **Expected result:** Native fields are present, stale approximation fields are
  absent from the native path, and shared stream settings are unchanged.
- **Edge cases:** Unsupported method, missing checkpoint, invalid seed and
  smoke/main differences.

#### Threshold and artifact provenance

- **Location:** `tests/online/test_artifact_integrity.py` plus native threshold
  tests where needed.
- **Level:** Contract/integration.
- **Setup:** Valid and invalid native threshold artifacts.
- **Action:** Validate, write and reload artifacts.
- **Expected result:** Valid artifacts round-trip; identity, split, finite-value
  and schema errors are rejected.
- **Edge cases:** Old schema artifact, missing checkpoint SHA and mismatched
  resolved config SHA.

### Verification

#### Automated

- [ ] Generator output matches parsed YAML for affected methods.
- [ ] Native constructor keys are accepted and stale keys are rejected or
  explicitly routed to a compatibility variant.
- [ ] Threshold artifacts validate with complete native provenance.
- [ ] Artifact writer preserves expected output paths and record schema.

#### Manual

- [ ] Compare one generated M2N2 and CANDI config with the selected reference
  settings and checkpoint contract.
- [ ] Confirm old approximation outputs are not overwritten or aggregated.

### Risks and recovery

- **Risk:** Regenerating configs invalidates old threshold/result tables.
- **Mitigation:** Recalibrate and write new identities; preserve old outputs.
- **Verification:** Manifest/report comparison before and after generation.
- **Recovery:** Re-run the prior generator revision/configuration only for the
  historical path; never mix its values with native results.

### Complete when

- Generator and YAML describe the selected native methods.
- Thresholds are recalibrated after the final score/lifecycle protocol.
- Artifacts identify model, checkpoint, method settings and comparability.

## Phase 6: Add fidelity tests and run one end-to-end smoke

### Goal

Prove the native method mechanisms through focused tests and one real benchmark
entry point before any full matrix execution.

### Dependencies

- Phases 0–5.
- One selected entity/seed smoke configuration.
- Checkpoint and data artifacts required by that configuration.

### Detailed atomic steps

#### Step 6.1: Split shared contract tests from method-fidelity tests

- **File:** `tests/online/test_online_streaming_baseline_contracts.py`
- **Symbol:** `test_online_streaming_baselines_calibrate_and_run`,
  `test_online_streaming_baselines_emit_entity_global_indices`
- **Current responsibility:** Proves basic calibration, records, metadata and
  global indices for five online baselines, but does not prove M2N2/CANDI
  algorithm fidelity.
- **Change:** Keep shared stream/record assertions. Move method-specific
  expectations into focused M2N2/CANDI tests and replace encoder-only metadata
  assertions with the selected contract assertions.
- **Reason:** A shared shape test can pass while both methods remain
  predicate-only approximations.
- **Inputs:** Deterministic train/validation/test fixtures and selected model
  checkpoint fixture.
- **Outputs:** Clear failures identifying stream contract versus method fidelity.
- **Errors:** Fail when a method claims native behavior but emits no optimizer
  loss, no method state or wrong model contract.
- **Dependencies:** Phases 1–5.
- **Compatibility:** Keep tests for unrelated frozen baselines and shared index
  behavior.
- **Verification:** Run the focused test file after each method-level test is
  added.

#### Step 6.2: Add M2N2 fidelity test group

- **File:** Existing `tests/online/` M2N2-focused test location, clearly named
  after repository conventions if a new file is required.
- **Symbol:** Detrender, mask, loss, optimizer and causal tests defined in Phase 3
- **Current responsibility:** No test proves any M2N2 reference equation.
- **Change:** Add deterministic tests for Steps 3.2–3.8.
- **Reason:** Each reference mechanism needs an isolated assertion.
- **Inputs:** Fixed tensors, fixed threshold and fixed optimizer/model seed.
- **Outputs:** Exact expected mean/mask/loss/state changes.
- **Errors:** Test failure on label leakage, triage invocation, reference-statistics
  updates or missing gradient step.
- **Dependencies:** Phase 3.
- **Compatibility:** Keep current loader test only for an explicitly retained
  RedLamp variant.
- **Verification:** Run the M2N2-focused test group with `pytest` and require all
  tests to pass before smoke execution.

#### Step 6.3: Add CANDI fidelity test group

- **File:** Existing `tests/online/` CANDI-focused test location, clearly named
  after repository conventions if a new file is required.
- **Symbol:** Validation references, Mahalanobis masks, pools, SANA, MSE and
  causal tests defined in Phase 4
- **Current responsibility:** No test proves CANDI FPM or SANA behavior.
- **Change:** Add deterministic tests for Steps 4.2–4.13.
- **Reason:** Candidate selection and adaptation state are the method identity.
- **Inputs:** Fixed validation representations/scores, covariance fixture,
  candidate batches and labels with controlled changes.
- **Outputs:** Exact masks, pool counts, update triggers, losses and parameter
  deltas independent of labels.
- **Errors:** Fail on triage invocation, test-label dependence, pool merge or
  update before `MIN_SAMPLES`.
- **Dependencies:** Phase 4.
- **Compatibility:** Preserve tests for unrelated THESIS online engine behavior.
- **Verification:** Run the CANDI-focused test group with `pytest` before smoke.

#### Step 6.4: Revise checkpoint tests

- **File:** `tests/online/test_redlamp_encoder_checkpoint.py`
- **Symbol:** `test_loader_reads_encoder_and_ignores_redlamp_heads`,
  `test_loader_rejects_missing_encoder_key`
- **Current responsibility:** Locks the encoder-only RedLamp loader behavior.
- **Change:** Keep these tests only if Phase 0 retains that compatibility path.
  Add complete native state/contract tests for the native path; do not weaken the
  current missing-key test merely to make an incompatible checkpoint load.
- **Reason:** The current test is useful for the variant but cannot certify a
  complete native model.
- **Inputs:** Selected checkpoint fixtures.
- **Outputs:** Explicit pass/fail for each checkpoint role.
- **Errors:** Wrong-role checkpoint must fail with a clear message.
- **Dependencies:** Phase 0 and Phase 1.
- **Compatibility:** Historical RedLamp loader contract remains testable when
  retained.
- **Verification:** Focused checkpoint test command passes.

#### Step 6.5: Run the existing online integration checks

- **File:** `tests/online/test_online_streaming_benchmark_wrapper.py`,
  `tests/online/test_online_entrypoint.py` and artifact tests
- **Symbol:** Existing benchmark/entrypoint/artifact tests
- **Current responsibility:** Checks runner calibration, output writing,
  threshold/report paths and wrapper behavior.
- **Change:** Update only assertions made invalid by the confirmed native record
  or provenance schema; preserve calibration, selected range and output checks.
- **Reason:** Method-level tests do not prove the actual benchmark entry point.
- **Inputs:** Fake baselines for wrapper tests and real selected baseline for
  method integration tests.
- **Outputs:** Passing runner/report/artifact integration evidence.
- **Errors:** Fail on missing records, wrong output paths, non-finite values,
  invalid threshold artifact or changed absolute range.
- **Dependencies:** Phases 2 and 5.
- **Compatibility:** Do not change THESIS wrapper expectations as part of this
  baseline task.
- **Verification:** Run the existing focused online integration test files.

#### Step 6.6: Run one real smoke configuration

- **File:** `scripts/benchmarks/run_online_streaming_benchmark.py`
- **Symbol:** `main`, `run_online_streaming_benchmark`
- **Current responsibility:** Loads one benchmark config, calibrates, selects the
  test range, runs the baseline and writes threshold/metrics/records/report.
- **Change:** No new behavior in this step; execute the already generated native
  smoke configuration after Phases 1–5 pass.
- **Reason:** The smoke must exercise the real entry point and real checkpoint,
  not only synthetic fixtures.
- **Inputs:** One generated native smoke YAML and its protocol config.
- **Outputs:** `online_thresholds.json`, `online_metrics.json`,
  `online_records.json` and benchmark report in the configured smoke output.
- **Errors:** Stop on checkpoint mismatch, non-finite score/loss, incomplete
  selected range, missing provenance or failure to write artifacts.
- **Dependencies:** Steps 5.2–5.5 and all fidelity tests.
- **Compatibility:** Use one combination only before any matrix expansion.
- **Verification:** Inspect artifact identity, range, records, causal update
  fields, finite metrics and metrics-only label policy.

#### Step 6.7: Decide whether the full matrix is allowed

- **File:** This research report and benchmark result manifest/documentation
  owner selected by project conventions
- **Symbol:** Final verification and run-approval record
- **Current responsibility:** No native method acceptance record exists.
- **Change:** Record pass/fail evidence for all required gates. Approve matrix
  expansion only when focused tests, integration checks, one smoke and manual
  provenance review pass.
- **Reason:** A full matrix can produce many invalid results when one method
  mechanism is still wrong.
- **Inputs:** Test reports, smoke artifacts and manual review.
- **Outputs:** Explicit acceptance or a return-to-phase decision.
- **Errors:** Keep the task blocked if any fidelity or provenance condition fails.
- **Dependencies:** Step 6.6.
- **Compatibility:** Do not aggregate results from failed/native and historical
  approximation generations.
- **Verification:** Checklist in the final verification section is complete.

### Tests

#### Focused online tests

- **Location:** Existing online test files and clearly named method-focused test
  files only where the current shared file becomes unclear.
- **Level:** Unit, integration and contract according to the behavior.
- **Setup:** Deterministic fixtures for equations; one real checkpoint for smoke.
- **Action:** Run method groups, shared lifecycle checks and the real entry point.
- **Expected result:** All method, causal, artifact and index assertions pass.
- **Edge cases:** Empty/short streams, NaNs, singular covariance, all-masked
  M2N2 batch, failed optimizer step and altered labels.

### Verification commands

#### Automated

- [ ] `pytest -q tests/online/test_online_streaming_baseline_contracts.py tests/online/test_redlamp_encoder_checkpoint.py` — shared contract and selected checkpoint tests pass.
- [ ] `pytest -q tests/online/test_online_streaming_benchmark_wrapper.py tests/online/test_online_entrypoint.py tests/online/test_artifact_integrity.py` — runner, output and artifact checks pass.
- [ ] The repository's focused M2N2 and CANDI test files — every method-fidelity assertion passes.
- [ ] The full relevant online test suite — only after one smoke combination passes.
- [ ] Run the existing benchmark config generator entry point for the selected
  smoke/main scope and verify generated YAML matches source output.
- [ ] Run the existing benchmark entry point with one generated smoke config and
  the repository protocol config; expect completed status and all required
  artifacts.

#### Manual

- [ ] Inspect one M2N2 trace and one CANDI trace.
- [ ] Inspect threshold, metrics, records and report provenance.
- [ ] Confirm the selected stream range and global index alignment.
- [ ] Confirm test labels appear only in final metric calculation, not baseline
  selection/update state.

### Risks and recovery

- **Risk:** A shape/integration test passes while method logic is still the old
  predicate-only approximation.
- **Mitigation:** Require the isolated equation, state-boundary and causal tests
  before the smoke.
- **Verification:** Inspect method-specific loss, optimizer, mask/pool and state
  diagnostics in the smoke artifacts.
- **Recovery:** Stop matrix expansion, retain outputs for diagnosis but mark
  them non-comparable, and return to the failed phase.

### Complete when

- All focused fidelity tests pass.
- Existing online integration checks pass after only justified contract updates.
- One real native smoke run completes with finite values and complete
  provenance.
- The project explicitly approves or rejects full matrix expansion.

## Interface and data changes

### Baseline lifecycle

- **Current interface:** `OnlineStreamingBaselineProtocol` exposes
  `calibrate()` and `run_sequence()`; the latter receives a complete selected
  sequence.
- **Target interface:** The runner still calls the stable public methods unless
  source inspection shows a smaller internal lifecycle can remain private. The
  method owner must expose enough internal state for score-before-adapt,
  optimizer/pool diagnostics and causal testing.
- **Compatibility rule:** Do not change unrelated frozen baselines or the THESIS
  `src/engine/online_tta/` protocol.

### Record schema

- Preserve entity ID, global point/window indices, `stream_step`, raw score,
  EWMA score, threshold, prediction, `did_update` and `loss_total`.
- Preserve `prediction == int(ewma_point_score > threshold)`.
- Do not use `triage_decision` as a native M2N2/CANDI method label. Retain a
  compatibility sentinel or version the field only after checking consumers.
- Add method-specific diagnostics only when they are JSON-safe, finite and
  required to audit the native update.

### Threshold artifacts

- Keep `clean_validation` as the calibration split unless Phase 0 explicitly
  changes it.
- Record the exact score protocol, EWMA settings, model/checkpoint contract,
  resolved config identity and method update settings.
- Recalibrate after every change to model score space, normalization, window
  order or adaptation timing.

### Output migration

- Do not overwrite old approximation outputs.
- Use a new method/variant/contract identity or mark old artifacts
  `not_comparable`.
- Do not mix old and native rows in seed aggregation or final tables.
- Keep only report-ready summaries, provenance, selected diagnostics, stage
  initialization checkpoint and stage-best checkpoint according to the project
  artifact-retention rules.

## Deployment and rollout

This task has no production deployment. The rollout boundary is the benchmark
matrix.

1. Complete Phase 0 and freeze the contract.
2. Implement and test Phases 1–4 in dependency order.
3. Regenerate configs and thresholds in Phase 5.
4. Run focused tests and one real smoke in Phase 6.
5. Expand to the full matrix only after the smoke and provenance gates pass.

If any gate fails, stop the rollout, retain failing artifacts for diagnosis,
mark them non-comparable and return to the phase that owns the failure. Rollback
means selecting the historical approximation configuration/code revision; it
does not mean deleting artifacts or silently reusing native thresholds.

## Documentation changes

- `documents/spec/online_benchmark_contract.md`: record the selected model,
  checkpoint, method identity, update protocol and migration mapping.
- This research report: record implementation evidence and test results after
  each phase; do not change the method status to faithful before Phase 6 passes.
- Generated benchmark configuration provenance: record the generator source,
  method settings, checkpoint identity and smoke/main mode.
- Benchmark report/manifest: record comparability identity, selected range,
  threshold source, update state and artifact paths.
- If a RedLamp variant remains, document its exact deviation from reference
  M2N2/CANDI and keep the variant name separate from the native method name.

## Final verification

- [x] Phase 0 contract decision is recorded and no model/checkpoint identity
  conflict remains.
- [x] The selected encoder checkpoint validates strictly; the adapter-owned
  reconstruction head is explicitly initialized and updated by the method.
- [x] Frozen/trainable parameter boundaries pass.
- [x] A two-window test proves that updated state can affect the next window;
  strict source order still needs correction because prediction/record is
  currently assembled after `_adapt_tensor()` returns.
- [x] M2N2 Detrender, mask, masked loss, optimizer and diagnostics pass.
- [x] CANDI validation references, Mahalanobis selection, pools, SANA,
  `MIN_SAMPLES` and diagnostics pass for the adapter variant; strict reference
  parity still needs the raw-input FPM representation and reference optimizer.
- [x] No native baseline uses test labels or THESIS triage for adaptation.
- [x] Config generator and generated YAML pass schema/identity checks.
- [x] Threshold artifacts are recalibrated from the selected adapter score
  protocol; they are currently q99 and therefore do not yet match the reference
  TTA default q99.5.
- [x] Two real smoke combinations complete with finite metrics and complete
  provenance.
- [x] The full relevant online test suite passes after the smoke.
- [ ] Only then is the full benchmark matrix approved.

## Blocking questions before implementation — resolved

1. The project selected the available RedLamp simple 1D-CNN encoder checkpoint
   and the separate identity `reference_adapter_redlamp_encoder`. The result is
   not labeled native MLP/TimesNet.
2. The available checkpoint inventory is the RedLamp `best.pt` path per entity
   and seed under `outputs/benchmark/smd/redlamp_baseline/<entity>/seed<seed>/`.
3. The target CANDI configuration is the reference default:
   `USE_FPM=true`, `USE_SANA=true`, `USE_HARD=true`, `USE_MODERATE=true`,
   `MIN_SAMPLES=16`, `STEPS=1`.

The third decision is also resolved as the reference default. The detailed
instructions are retained as the implementation trace; the matrix remains
blocked only on manual report review and explicit approval.

## Implementation record: encoder-checkpoint adapter variant

The user resolved the Phase 0 model decision during implementation: use the
available RedLamp simple 1D-CNN encoder checkpoint. The selected identity is
`reference_adapter_redlamp_encoder`. It is separate from historical
`main` approximation artifacts and separate from a native MLP/TimesNet
implementation.

The implementation now covers the following atomic-step results:

- `src/models/online_redlamp_reconstruction.py` loads and validates the
  encoder state and owns the adapter reconstruction head.
- `src/models/online_adapter_modules.py` implements the mean-only M2N2
  Detrender and the CANDI SANA residual module.
- `src/baselines/online/adaptive.py` processes one test window at a time. The
  raw score is computed before adaptation and updated state affects later
  windows, but the source call order is `score, EWMA, adapt, prediction/record`.
  This is numerically score-before-update but not the exact reference call order.
- `src/baselines/online/m2n2.py` implements gamma-based Detrender updates,
  timestep pseudo-anomaly masks, masked reconstruction loss and one optimizer
  step.
- `src/baselines/online/candi.py` implements validation representations,
  covariance pseudoinverse, hard/moderate Mahalanobis selection, separate
  candidate pools, `MIN_SAMPLES` gating and SANA MSE updates. Its current FPM
  selection representation is computed from `x + sana_in(x)`, whereas the
  reference selection uses `model.get_representations(x)`.
- `scripts/benchmarks/generate_online_streaming_benchmark_configs.py` emits
  explicit `reference_adapter_redlamp_encoder` configs for M2N2/CANDI with the
  reference settings. Existing `main` artifacts are not overwritten or
  reclassified.
- `documents/spec/online_benchmark_contract.md` records the encoder-checkpoint
  variant, trainable surfaces and update equations.

Verification passed:

- Method/checkpoint and causal tests: `8 passed`.
- Wrapper/config/artifact tests: `9 passed`.
- Full `tests/online` suite: `89 passed`, one pre-existing joblib core-count
  warning.
- One real CANDI smoke through the benchmark entry point completed for
  `machine-1-6`, seed `6`, absolute range `[146,2200)`, with smoke truncation
  to 16 online steps. The report contains finite scores, checkpoint SHA-256,
  variant identity, threshold artifact and metrics-only label provenance.
- One real M2N2 smoke through the same entry point completed for the same
  entity, seed and range. Its 16 records contain finite masked losses,
  `did_update=true` and a mask count of 20.

The remaining gate is not only manual report review. A follow-up runtime audit
must first correct the strict Predictor-flow mismatches recorded below. The
method status remains provisional for matrix reporting until those corrections,
focused tests and report review pass.

## Follow-up audit: remaining runtime-flow corrections

This section is the current-state audit requested after comparing the implemented
adapter variant with the user-provided `CANDI-main` Predictor flow. The local
method-fidelity tests passing does not remove these differences because those
tests validate the repository contract, not exact reference parity.

### A. Shared Predictor and stream lifecycle

1. **Prediction/record is assembled after adaptation.**

   - **Current code:** `src/baselines/online/adaptive.py:343-355` computes the
     raw score, calls `_adapt_tensor()`, and only then computes `prediction`.
   - **Reference:** `score -> prediction/record -> adapter.adapt()`.
   - **Required correction area:** `AdaptiveStreamingBaselineBase.run_sequence()`.
   - **Why it matters:** The prediction value currently comes from the saved
     pre-update raw score, so the numeric result is usually unchanged. The
     runtime side-effect order is nevertheless not the reference order.

2. **The current test stream is one-window causal, not general batch-causal.**

   - **Current code:** `src/baselines/online/base.py:92-98` fixes the helper
     batch size to `1`, and `adaptive.py:343-352` wraps every window as
     `[1, L, D]`.
   - **Reference:** all windows in one test batch are scored before one adapter
     update; the next batch sees the updated state.
   - **Required correction area:** `src/baselines/online/base.py:80-98`,
     `src/baselines/online/adaptive.py:321-404` and the runner loop.
   - **Current classification:** equivalent only for the explicitly configured
     special case `batch_size=1`; not a general implementation of the
     reference batch lifecycle.

3. **The outer Predictor pre-TTA phase is absent.**

   - **Current code:** `scripts/benchmarks/run_online_streaming_benchmark.py:297-405`
     constructs the baseline, calibrates validation and enters the adaptive test
     loop. It does not deep-copy a trained model or compute train, validation and
     pre-TTA test scores before constructing the adapter.
   - **Reference:** `predictor.py` computes those score sets first, creates the
     threshold, then constructs the TTA adapter.
   - **Required correction area:**
     `scripts/benchmarks/run_online_streaming_benchmark.py:297-405` and the
     model/checkpoint initialization boundary.
   - **Current classification:** the repository benchmark runner is a reduced
     online protocol, not the full reference Predictor flow.

### B. Threshold and optimizer contract

4. **Threshold quantile does not match the reference TTA default.**

   - **Current code:** `src/baselines/online/adaptive.py:281-307` uses the
     configured raw-window quantile; active protocol/config values are `0.99`.
   - **Reference:** TTA forces ratio thresholding with `ANOMALY_RATIO=0.5`,
     hence validation percentile `99.5`.
   - **Required correction area:**
     `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml`, generated
     baseline configs, threshold artifact generation and related tests.
   - **Current classification:** threshold split is clean-validation-only, but
     its active quantile is not reference-equivalent.

5. **Optimizer identity does not match the reference default.**

   - **Current code:** M2N2 and CANDI always create `torch.optim.AdamW` in
     `src/baselines/online/m2n2.py:73-77` and `src/baselines/online/candi.py:138-142`.
   - **Reference:** `TEST.TTA.SOLVER.OPTIMIZING_METHOD` defaults to `sgd` in
     `bsc-thesis-ref-codebases/CANDI-main/config.py:89-97`, and the adapter uses
     that TTA solver.
   - **Required correction area:** both method constructors, the generator and
     method configuration schema.

### C. CANDI-specific parity

6. **FPM selection uses the wrong current representation after SANA updates.**

   - **Current code:** `src/baselines/online/candi.py:233-235` calls
     `get_representations(self._candi_input(x))`.
   - **Reference:** `adapter_candi.py:get_samples_to_adapt()` calls
     `model.get_representations(x)`; the SANA-transformed input is used in the
     reconstruction loss, not in FPM representation selection.
   - **Required correction area:** `CANDIStreamingBaseline._collect_candidates()`
     and its direct test fixture.

7. **Hard-reference top-k has an extra local guard.**

   - **Current code:** `src/baselines/online/candi.py:200` forces
     `topk = max(1, int(...))`.
   - **Reference:** uses the direct integer result of
     `int(len(val_scores) * anomaly_ratio / 100)`.
   - **Required correction area:** CANDI validation-reference construction and
     the small-validation-set behavior test.
   - **Current classification:** the guard may prevent a runtime error, but it
     changes the reference behavior for validation sets with fewer than 200
     windows at anomaly ratio `0.5`.

8. **SANA is topology-compatible but not layer-identical.**

   - **Current code:** `src/models/online_adapter_modules.py:46-136` uses
     PyTorch `TransformerEncoder`.
   - **Reference:** uses its own `Encoder`, `EncoderLayer`, `AttentionLayer` and
     `FullAttention` stack.
   - **Required correction area:** only if the claim is exact SANA fidelity;
     otherwise the method metadata must continue to identify this as the
     `reference_adapter_redlamp_encoder` variant.

9. **Reference CANDI reads test labels for pool statistics; current adapter does
   not.**

   - **Current code:** metadata declares `metrics_only`, but no test labels are
     passed into `CANDIStreamingBaseline` for the reference pool counters.
   - **Reference:** `CANDIAdapter` receives `test_labels` and uses them only to
     report anomaly counts in selected pools.
   - **Required correction area:** runner-to-adapter metrics plumbing, if those
     reference diagnostics are part of the required output. This is not a
     selection or loss leak.

### D. Model/checkpoint identity

10. **The model surface remains a RedLamp variant, not native MLP/TimesNet.**

   - **Current code:** `src/models/online_redlamp_reconstruction.py` loads the
     encoder and creates an adapter-owned decoder.
   - **Reference:** `construct_adapter()` selects MLP or TimesNet model-specific
     adapters.
   - **Required correction area:** model/checkpoint contract and method naming,
     not a silent alias.
   - **Current classification:** accepted as the user-selected checkpoint
     variant, but it cannot support an unqualified claim of native reference
     model fidelity.

## Required correction order

For a strict runtime-flow claim, the remaining work should be evaluated in this
order:

1. Correct `prediction/record -> adapt` ordering and add a direct regression
   test for the call sequence.
2. Decide whether the benchmark target is the reference's general batch-causal
   lifecycle or the repository's explicitly fixed `batch_size=1` special case.
3. Align threshold quantile and optimizer identity with the selected reference
   configuration, then regenerate/recalibrate artifacts.
4. Change CANDI FPM selection to use raw current-window representations and
   decide whether exact top-k and SANA layer parity are required.
5. Keep the RedLamp checkpoint identity separate from native MLP/TimesNet claims
   in metadata, configs and result tables.

Until these items are resolved, the correct status is **adapter mechanisms
implemented, exact reference runtime flow not yet established**.

## Follow-up implementation structure: sequential stages

### Summary

The follow-up implementation is divided into seven dependent phases. Each phase
has a small number of observable stages and must remain verifiable before the
next phase starts. The structure targets reference Predictor control flow on the
approved RedLamp adapter variant; it does not rename that variant as native
MLP/TimesNet.

### Request

Read `prompts/3_structure_prompt.md` and organize the runtime-flow correction
plan into sequential stages. Keep the stages high-level; file-level atomic
instructions belong to a later detail phase.

### Confirmed context

- The current adapter mechanisms pass focused method tests, but strict runtime
  parity still differs in call order, batch lifecycle, threshold quantile,
  optimizer identity and CANDI FPM representation input.
- The approved checkpoint is the RedLamp simple 1D-CNN encoder, with identity
  `reference_adapter_redlamp_encoder`.
- Historical `main`, `A0`, `A1` and `A2` approximation artifacts remain separate.
- The reference control flow is batch-causal:
  `score -> prediction/record -> adapt -> next batch`.

### Scope

#### In scope

- Runtime profile and artifact identity.
- Predictor pre-TTA and batch-causal stream lifecycle.
- Threshold and optimizer alignment.
- M2N2 and CANDI method-state execution.
- Config, test, artifact and smoke validation.

#### Out of scope

- Replacing the approved RedLamp checkpoint with an unverified native model.
- THESIS-specific online adaptation components.
- Full benchmark matrix execution before the single-method gates pass.
- Mixing historical approximation results with corrected adapter-variant results.

### Proposed phases

#### Phase 0: Freeze the runtime-fidelity acceptance contract

**Result:** The project has one explicit runtime profile for the corrected
RedLamp adapter variant, including batch size, threshold ratio, optimizer,
method settings and non-equivalence boundary.

**Stages:**

1. **Define the acceptance profile.** Record the selected reference flow,
   q99.5 threshold, TTA optimizer, M2N2 and CANDI defaults, and the RedLamp
   variant identity.
2. **Separate configuration generations.** Mark legacy approximation configs and
   artifacts as non-comparable to corrected adapter-variant outputs.
3. **Validate the profile boundary.** Confirm that later phases can read the
   profile from configuration and reproduce it in method metadata.

**Depends on:** Follow-up audit and approved RedLamp checkpoint decision.

**Verification:** Contract/config validation and manual comparison of one legacy
and one corrected configuration.

**Risks:** A stale config may silently select q99, AdamW or legacy behavior.
Reject profile/method metadata mismatches before implementation continues.

**Complete when:** Method identity, runtime settings and comparability boundary
are explicit and unambiguous.

#### Phase 1: Rebuild the common Predictor and batch-causal lifecycle

**Result:** The runner processes the selected test stream in reference order and
only exposes an update to the next batch.

**Stages:**

1. **Establish the pre-TTA stage.** Separate model initialization, score
   calibration and any required pre-adapter score aggregation from adapter
   construction.
2. **Establish the batch boundary.** Represent test inputs as `[B,L,D]` and
   ensure every window in one batch uses one pre-update model state.
3. **Fix record timing.** Emit score, prediction and pre-update record fields
   before invoking adaptation.
4. **Verify next-batch feedback.** Confirm that updates from batch `t` affect
   batch `t+1`, while batch `t` records remain unchanged.

**Depends on:** Phase 0 runtime profile.

**Verification:** Call-order test, multi-window batch test, absolute-index test
and one short manual stream trace.

**Risks:** Batch boundaries may change score distributions and artifact sizes.
Keep range/index contracts stable and defer threshold regeneration to Phase 2.

**Complete when:** The runner and both adapters satisfy the reference
`score -> prediction/record -> adapt` lifecycle.

#### Phase 2: Align threshold and optimizer semantics

**Result:** Threshold and optimizer behavior are explicit, reproducible and
match the selected reference profile.

**Stages:**

1. **Align threshold calibration.** Use clean-validation raw scores and the
   selected 0.5% anomaly ratio, producing q99.5.
2. **Align optimizer construction.** Select the configured TTA optimizer,
   defaulting to reference SGD, and expose its parameter groups.
3. **Align provenance.** Record threshold source, quantile, optimizer and
   trainable surface in artifacts and method metadata.
4. **Verify score-space consistency.** Confirm prediction uses the intended raw
   score and threshold rather than an unintended EWMA decision score.

**Depends on:** Phase 0 profile and Phase 1 lifecycle.

**Verification:** Threshold fixture, optimizer parameter-group test, metadata
check and threshold artifact schema validation.

**Risks:** Existing q99 artifacts become non-comparable. Create new output
identities and do not overwrite historical artifacts.

**Complete when:** Active configs, instantiated methods and threshold artifacts
agree on quantile, optimizer and score semantics.

#### Phase 3: Correct M2N2 reference update execution

**Result:** M2N2 performs the reference Detrender, timestep mask, masked loss and
one optimizer update at the corrected batch boundary.

**Stages:**

1. **Verify Detrender timing.** Update the mean-only state before the adaptation
   forward pass for each batch.
2. **Verify the adaptation objective.** Compute timestep error, pseudo-anomaly
   mask and masked reconstruction loss without test labels.
3. **Verify the optimizer transition.** Perform the configured update and restore
   the prior train/eval mode.
4. **Verify causal state visibility.** Confirm only the next batch sees the
   updated M2N2 parameters and Detrender state.

**Depends on:** Phases 1 and 2.

**Verification:** Deterministic equation/mask/loss tests, parameter-boundary
test and two-batch feedback test.

**Risks:** Changing batch size changes the Detrender mean. Record the selected
batch profile and calibrate/evaluate under the same profile.

**Complete when:** M2N2 tests prove reference equations, optimizer behavior and
batch-causal feedback.

#### Phase 4: Correct CANDI FPM and SANA runtime parity

**Result:** CANDI selects candidates from raw current representations, maintains
separate pools and updates SANA only according to the reference gates.

**Stages:**

1. **Correct FPM representation input.** Use `get_representations(x)` for
   selection and reserve SANA-transformed inputs for reconstruction.
2. **Verify reference banks and masks.** Align hard top-k, moderate Q1/Q3,
   covariance, Mahalanobis and chi-square behavior.
3. **Verify SANA trainable scope.** Confirm frozen detector parameters and the
   selected SANA implementation boundary.
4. **Verify pool lifecycle.** Accumulate hard and moderate samples separately,
   update at `MIN_SAMPLES`, update hard before moderate, then clear each pool.
5. **Verify next-batch feedback.** Confirm the current batch is scored before a
   SANA update can affect the next batch.

**Depends on:** Phases 1 and 2, plus the Phase 0 SANA fidelity boundary.

**Verification:** Deterministic FPM selection, Mahalanobis, pool-gate, SANA
parameter-scope and causal-feedback tests, followed by one manual trace.

**Risks:** An exact SANA port may change scores and runtime cost. Keep variant
identity explicit and require a method-level smoke before matrix regeneration.

**Complete when:** CANDI tests prove raw-representation selection, separate pool
gates, SANA scope and causal updates.

#### Phase 5: Migrate configurations, tests and artifacts

**Result:** Corrected configs and artifacts describe one reproducible runtime
profile and remain separate from historical approximation outputs.

**Stages:**

1. **Regenerate method configs.** Emit the selected batch, threshold, optimizer,
   M2N2, CANDI, SANA and checkpoint settings from the generator.
2. **Extend regression coverage.** Add call-order, batch-causal, threshold,
   optimizer, raw-representation and legacy-config checks.
3. **Recalibrate artifacts.** Generate thresholds and reports only after the
   corrected score/update path passes focused tests.
4. **Verify comparability metadata.** Confirm corrected and historical output
   roots cannot be aggregated as one method.

**Depends on:** Phases 0-4.

**Verification:** Generator/schema tests, focused online suite, artifact
identity checks and manual report review.

**Risks:** Generated files or tables may retain stale fields. Treat the generator
as the source of truth and reject stale profile fields in validation.

**Complete when:** Configs, tests, threshold artifacts and reports agree with
the corrected runtime profile.

#### Phase 6: Validate one real end-to-end run and approve expansion

**Result:** One real M2N2 and one real CANDI run demonstrate the corrected flow
through the benchmark entry point before matrix execution.

**Stages:**

1. **Run focused tests.** Confirm method equations, lifecycle, threshold,
   optimizer and artifact checks pass.
2. **Run one M2N2 smoke.** Inspect finite scores/losses, pre-update records,
   checkpoint identity and next-batch feedback.
3. **Run one CANDI smoke.** Inspect candidate pools, update gates, SANA
   diagnostics, finite values and next-batch feedback.
4. **Review acceptance evidence.** Compare the first two batches with the
   expected reference trace and verify metrics-only label usage.
5. **Approve or stop.** Expand to the full matrix only when all evidence passes;
   otherwise retain the failing artifact and return to its responsible phase.

**Depends on:** Phases 0-5.

**Verification:** Focused tests, two real smoke runs and manual artifact review.

**Risks:** Shape-valid smoke output can still hide method drift. Require exact
call-order and method-fidelity tests before accepting either smoke.

**Complete when:** Both smokes and provenance review pass, and the full matrix
has explicit approval.

### Dependency summary

| Phase | Requires | Enables |
| --- | --- | --- |
| 0 | Audit and checkpoint decision | Stable runtime profile and identity |
| 1 | Phase 0 | Common Predictor and batch-causal lifecycle |
| 2 | Phases 0-1 | Reproducible threshold and optimizer semantics |
| 3 | Phases 1-2 | Correct M2N2 state transition |
| 4 | Phases 1-2 and SANA boundary | Correct CANDI state transition |
| 5 | Phases 0-4 | Comparable configs, tests and artifacts |
| 6 | Phases 0-5 | Approval for matrix execution |

### Decisions confirmed

- The RedLamp encoder checkpoint remains the selected model surface for this
  adapter variant.
- The result identity remains `reference_adapter_redlamp_encoder` and must not
  be reported as native MLP/TimesNet.
- Historical approximation artifacts remain non-comparable.
- Test labels remain outside calibration, selection, loss and optimizer updates;
  they may be used for final metrics or explicitly marked diagnostics only.

### Non-blocking uncertainties

- The exact SANA layer port determines whether the result can claim layer-level
  SANA parity. The detailed plan must either implement the reference topology or
  preserve variant-level metadata.
- The reference Predictor's aggregate train/test score reports are required only
  if the active benchmark contract consumes them; the per-batch causal order is
  required in either case.

### Feedback requested

- Does this phase order match the intended correction path?
- Should exact SANA layer parity remain in scope, or should the project keep the
  current SANA topology and report only adapter-level parity?
- Is the proposed batch profile appropriate for the final benchmark, or should
  the corrected runtime support both `batch_size=1` and the reference batch size?

## Follow-up implementation plan: runtime-flow fidelity corrections

### Summary

This plan corrects the remaining differences between the active
`reference_adapter_redlamp_encoder` runtime and the user-provided
`CANDI-main` Predictor flow. The plan keeps the approved RedLamp encoder
checkpoint contract. It aligns the control flow, threshold protocol, optimizer
identity, CANDI representation selection and verification evidence without
claiming native MLP/TimesNet model equivalence.

### Request

Read `prompts/2_plan_prompt.md` and define ordered implementation phases for the
runtime-flow corrections identified in the follow-up audit above.

### Current state

- `scripts/benchmarks/run_online_streaming_benchmark.py:297-405` constructs the
  adapter baseline, calibrates validation and calls `run_sequence()` directly.
- `src/baselines/online/adaptive.py:343-355` scores one window, adapts, and
  only then computes the prediction/record from the saved raw score.
- `src/baselines/online/base.py:92-98` and the active runner use one window per
  batch, so the current stream is a special case of the reference batch-causal
  lifecycle.
- `src/baselines/online/adaptive.py:281-307` and the active protocol use q99,
  while the reference TTA default uses ratio `0.5`, or q99.5.
- M2N2 and CANDI use AdamW, while the reference TTA solver defaults to SGD.
- CANDI uses `get_representations(x + sana_in(x))` for FPM selection after SANA
  updates, while the reference uses `get_representations(x)`.

### Desired end state

- The active adapter variant follows `score -> prediction/record -> adapt` for
  every test batch, and only the next batch observes the update.
- The runner can represent the reference batch-causal lifecycle explicitly;
  the official batch size is recorded in configuration and metadata.
- The threshold and optimizer settings are explicit and match the selected
  reference profile.
- CANDI uses raw current-window representations for FPM selection and uses SANA
  only in the reconstruction/adaptation path.
- Tests prove the call order, threshold source, optimizer identity, CANDI
  selection input and next-batch feedback.
- RedLamp-based results remain separate from native MLP/TimesNet claims.

### Scope

#### In scope

- Common Predictor and test-stream lifecycle.
- Clean-validation threshold and reference TTA ratio semantics.
- M2N2/CANDI optimizer configuration and trainable-surface metadata.
- CANDI FPM representation input, top-k behavior and SANA parity boundary.
- Method-fidelity tests, generated configurations, threshold artifacts and
  report provenance.

#### Out of scope

- Replacing the approved RedLamp encoder with an unverified native MLP/TimesNet
  checkpoint.
- Changing THESIS online behavior under `src/engine/online_tta/`.
- Adding THESIS triage, PNN verification, projector updates or uncertainty
  ablations to M2N2/CANDI.
- Mixing old `main`/`A0`/`A1`/`A2` approximation artifacts with the corrected
  adapter-variant artifacts.

### Evidence

- `scripts/benchmarks/run_online_streaming_benchmark.py:297-405` — active
  runner construction, calibration and test execution.
- `src/baselines/online/adaptive.py:281-307,321-355` — active threshold and
  test-stream order.
- `src/baselines/online/base.py:80-98` — active batch helper and fixed batch size.
- `src/baselines/online/m2n2.py:67-129` — M2N2 state, optimizer and update.
- `src/baselines/online/candi.py:105-305` — CANDI SANA, FPM, pools and updates.
- `bsc-thesis-ref-codebases/CANDI-main/predictor.py:21-103` — reference
  Predictor initialization, pre-TTA scoring and score/predict/adapt loop.
- `bsc-thesis-ref-codebases/CANDI-main/tta/candi/adapter_candi.py:133-379` —
  reference CANDI selection, pools and SANA loss.
- `bsc-thesis-ref-codebases/CANDI-main/config.py:69-97` — reference threshold,
  TTA and solver defaults.

### Implementation approach

Keep `calibrate()` and `run_sequence()` as the stable benchmark-facing boundary,
but make the internal adapter lifecycle explicit. The runner will own split
loading, threshold artifacts and record aggregation. M2N2 will own Detrender and
model-parameter updates. CANDI will own FPM references, candidate pools and
SANA updates. The implementation will use one selected reference profile rather
than silently combining the current q99/AdamW/batch-one choices with reference
method names.

## Phase 0: Freeze the runtime-fidelity acceptance contract

### Goal

Define one unambiguous target before changing code: exact reference adapter flow
on the approved RedLamp model surface, with separate identity from native
MLP/TimesNet implementations.

### Changes

#### 1. Record the reference runtime profile

- **File:** `documents/spec/online_benchmark_contract.md`
- **Symbol:** online method identity, threshold and update-policy sections
- **Change:** Record `reference_adapter_redlamp_encoder` as a RedLamp model
  variant with reference adapter control flow. Record the selected test batch
  size, q99.5 ratio threshold, TTA optimizer, M2N2 gamma, CANDI defaults and
  the fact that native MLP/TimesNet equivalence is not claimed.
- **Reason:** Later code and artifacts must use one stable interpretation of
  “reference flow”.
- **Dependencies:** Existing contract and follow-up audit.

#### 2. Mark historical configurations non-comparable

- **File:** `configs/experiment/online_benchmark/m2n2/` and
  `configs/experiment/online_benchmark/candi/`
- **Symbol:** legacy `main`, `A0`, `A1` and `A2` configurations
- **Change:** Define the migration identity that separates legacy approximation
  configs from corrected `reference_adapter_redlamp_encoder` configs.
- **Reason:** A config label must not silently select another runtime contract.
- **Dependencies:** Generator and artifact metadata.

### Verification

#### Automated

- [ ] Contract tests reject missing runtime-profile fields and mismatched
  checkpoint/method identities.

#### Manual

- [ ] Review one corrected config and one legacy config and confirm they cannot
  be aggregated as the same method contract.

### Risks

- **Risk:** A reference profile is defined but active configs still select q99 or
  AdamW. **Mitigation:** require config validation to compare every profile
  field with the instantiated baseline metadata.

## Phase 1: Rebuild the common Predictor and batch-causal lifecycle

### Goal

Make the active runner follow the reference lifecycle: initialize the selected
model state, perform the required pre-TTA score/calibration stage, then process
each test batch as `score -> prediction/record -> adapt`.

### Changes

#### 1. Separate pre-TTA scoring from adapter construction

- **File:** `scripts/benchmarks/run_online_streaming_benchmark.py:297-405`
- **Symbol:** `run_online_streaming_benchmark`
- **Change:** Add the reference-ordered pre-TTA stage required by the selected
  profile, keeping train/validation/test score ownership separate from the
  post-adapter test loop. Do not pass test labels into scoring or adaptation.
- **Reason:** The current runner constructs the adapter path without the
  reference Predictor's pre-TTA score phase.
- **Dependencies:** model checkpoint identity, threshold artifact schema and
  selected batch-size contract.

#### 2. Move prediction/record before adaptation

- **File:** `src/baselines/online/adaptive.py:321-404`
- **Symbol:** `AdaptiveStreamingBaselineBase.run_sequence`
- **Change:** Compute raw score, prediction and record fields first; call
  `_adapt_tensor()` only after the record has captured the pre-update state.
  Update `did_update`, loss and pool diagnostics in the same record without
  changing the recorded prediction.
- **Reason:** This is the explicit reference side-effect order.
- **Dependencies:** record schema and method update return values.

#### 3. Process configurable batches without leaking future state

- **File:** `src/baselines/online/base.py:80-98` and
  `src/baselines/online/adaptive.py:321-404`
- **Symbol:** `build_stride1_batcher`, `run_sequence`
- **Change:** Represent a test batch as `[B,L,D]`, score all its windows before
  one adapter update, and expose each window's pre-update score in output order.
  Preserve stride-one alignment and absolute indices.
- **Reason:** The reference is batch-causal, not timestep-causal.
- **Dependencies:** `src/data/stream.py`, record aggregation and selected
  batch-size configuration.

### Verification

#### Automated

- [ ] A call-order test observes `score -> prediction/record -> adapt`.
- [ ] A multi-window batch test proves every window in one batch uses the same
  state and the next batch observes the update.
- [ ] Existing absolute range and stream-step tests remain unchanged.

#### Manual

- [ ] Inspect one short trace and confirm batch boundaries, indices and update
  visibility match the reference flow.

### Risks

- **Risk:** Batch processing changes score distributions and artifact sizes.
  **Mitigation:** regenerate thresholds and compare selected-range metadata
  1:1 before approving a benchmark run.

## Phase 2: Align threshold and optimizer semantics

### Goal

Make threshold creation and optimizer construction explicit reference-profile
settings instead of hidden q99 and AdamW defaults.

### Changes

#### 1. Align the TTA threshold source

- **File:** `src/baselines/online/adaptive.py:268-307` and
  `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml`
- **Symbol:** `calibrate`, `online_threshold_quantile`
- **Change:** Use clean-validation raw scores and the selected reference ratio
  (`ANOMALY_RATIO=0.5`, percentile `99.5`) for the adapter threshold. Keep EWMA
  as a reporting/artifact diagnostic unless the selected profile explicitly
  makes it the decision score.
- **Reason:** Current q99 is not the reference TTA threshold.
- **Dependencies:** Phase 0 profile, threshold artifact schema and generated
  protocol/config files.

#### 2. Expose and select the TTA optimizer

- **File:** `src/baselines/online/m2n2.py:67-77`,
  `src/baselines/online/candi.py:105-142`, and
  `scripts/benchmarks/generate_online_streaming_benchmark_configs.py`
- **Symbol:** method initialization and baseline kwargs
- **Change:** Add an explicit optimizer method field and instantiate the
  selected reference TTA optimizer, defaulting to SGD for reference parity.
  Record method, learning rate, weight decay and trainable parameter scope in
  metadata.
- **Reason:** Current code always selects AdamW and therefore hides a method
  setting that changes parameter updates.
- **Dependencies:** Phase 0 config schema and method metadata.

### Verification

#### Automated

- [ ] Threshold fixture equals the clean-validation percentile expected from the
  configured anomaly ratio.
- [ ] Optimizer test asserts the selected optimizer class and parameter groups
  for M2N2 and CANDI.
- [ ] Prediction remains `raw_score > threshold` and does not use test labels.

#### Manual

- [ ] Inspect one threshold artifact and one method metadata object and confirm
  the score space, quantile, optimizer and trainable surface agree.

### Risks

- **Risk:** Changing q99 to q99.5 invalidates existing reports. **Mitigation:**
  write new threshold/output identities and mark old artifacts non-comparable.

## Phase 3: Correct M2N2 reference update execution

### Goal

Preserve the already implemented M2N2 equations while making their batch
boundary, optimizer and record timing exactly match the reference adapter.

### Changes

#### 1. Keep Detrender update before adaptation forward

- **File:** `src/baselines/online/m2n2.py:107-129`
- **Symbol:** `M2N2StreamingBaseline._adapt_tensor`
- **Change:** Apply the current batch mean with reference gamma semantics before
  reconstruction, build `A`, `ytilde`, `mask` and `(A * mask).mean()`, then run
  one configured optimizer step.
- **Reason:** This is the reference M2N2 update equation and is already close to
  correct; the phase prevents lifecycle changes from altering it.
- **Dependencies:** Phase 1 batch input and Phase 2 optimizer profile.

#### 2. Preserve metrics-only label isolation

- **File:** `src/baselines/online/m2n2.py` and record metadata
- **Symbol:** `_adapt_tensor`, `_method_metadata`
- **Change:** Keep `scores` unused by the loss and keep test labels outside the
  mask, loss and optimizer path.
- **Reason:** The reference adapter accepts `scores` but does not use it to
  construct the M2N2 mask.
- **Dependencies:** Phase 1 runner interface.

### Verification

#### Automated

- [ ] Deterministic batch test matches Detrender mean, timestep mask and masked
  loss values.
- [ ] Optimizer update changes only the configured M2N2 trainable surface.
- [ ] A two-batch test proves the first update affects only the next batch.

#### Manual

- [ ] Inspect one M2N2 record and confirm score/prediction are pre-update while
  loss and mask diagnostics describe the subsequent update.

### Risks

- **Risk:** Changing batch size changes the Detrender batch mean. **Mitigation:**
  make the selected batch size explicit and calibrate/evaluate under the same
  profile.

## Phase 4: Correct CANDI FPM and SANA runtime parity

### Goal

Make CANDI select candidates from the same representation and score conditions as
the reference, then adapt separate pools through the permitted SANA parameters.

### Changes

#### 1. Use raw input for FPM representations

- **File:** `src/baselines/online/candi.py:225-251`
- **Symbol:** `CANDIStreamingBaseline._collect_candidates`
- **Change:** Compute FPM representations from `backbone_.get_representations(x)`.
  Keep `_candi_input(x)` only in the reconstruction score/loss path.
- **Reason:** SANA changes reconstruction input, not the reference FPM selection
  representation.
- **Dependencies:** Phase 1 batch lifecycle and validation representation bank.

#### 2. Match hard/moderate reference-set behavior

- **File:** `src/baselines/online/candi.py:187-211`
- **Symbol:** `_calibration_complete`
- **Change:** Align hard top-k integer behavior, Q1/Q3 moderate reference
  construction, covariance pseudoinverse and chi-square percentile with the
  selected reference profile. Keep separate hard and moderate pools.
- **Reason:** A local `max(1, ...)` guard and any changed representation source
  alter candidate selection on small validation sets.
- **Dependencies:** Phase 0 profile and validation-data contract.

#### 3. Decide and implement exact SANA topology boundary

- **File:** `src/models/online_adapter_modules.py:46-136`
- **Symbol:** `SANA`
- **Change:** Either port the reference `TemporalEmbedding`/`Encoder`/
  `FullAttention` topology or keep the current compatible implementation and
  explicitly retain variant-level rather than layer-identical claims. The
  selected outcome must be recorded before artifact generation.
- **Reason:** The current module uses PyTorch `TransformerEncoder`, not the
  reference attention stack.
- **Dependencies:** Phase 0 fidelity claim and model/checkpoint contract.

#### 4. Preserve pool gate and update ordering

- **File:** `src/baselines/online/candi.py:253-305`
- **Symbol:** `_adapt_pool`, `_adapt_tensor`
- **Change:** Score/predict the whole incoming batch first, append selected
  samples, update hard then moderate pools only at `MIN_SAMPLES`, clear each pool
  after its own update and report post-update pool sizes.
- **Reason:** This matches the reference CANDI lifecycle and prevents same-batch
  state leakage.
- **Dependencies:** Phase 1 ordering and Phase 2 optimizer.

### Verification

#### Automated

- [ ] FPM fixture proves selection is invariant to SANA input transformation
  when the backbone representation of raw `x` is unchanged.
- [ ] Hard/moderate masks match deterministic Mahalanobis fixtures.
- [ ] Pool gate, reset, hard-before-moderate update and SANA parameter-boundary
  tests pass.
- [ ] Labels do not affect selection, loss or optimizer state; optional pool
  anomaly counters are tested separately as metrics-only diagnostics.

#### Manual

- [ ] Inspect one CANDI trace containing raw representation source, candidate
  pool sizes, update trigger, loss and next-batch score.

### Risks

- **Risk:** Exact SANA port changes output and runtime cost. **Mitigation:** keep
  the RedLamp variant identity explicit and run one method-level smoke before
  regenerating the matrix.

## Phase 5: Migrate configurations, tests and artifacts

### Goal

Make generated configurations, tests and persisted reports describe the
corrected runtime without mixing historical approximation outputs.

### Changes

#### 1. Regenerate method configurations

- **File:** `scripts/benchmarks/generate_online_streaming_benchmark_configs.py`
  and `configs/experiment/online_benchmark/{m2n2,candi}/`
- **Symbol:** `BENCHMARK_METHOD_VARIANTS`, method baseline kwargs and generated
  YAML files
- **Change:** Emit the selected batch size, q99.5 threshold profile, optimizer,
  M2N2 gamma/steps, CANDI FPM/SANA settings and checkpoint identity. Remove
  stale approximation-only fields from corrected configs.
- **Reason:** Generated YAML is the active experiment contract.
- **Dependencies:** Phases 0-4.

#### 2. Add regression and integration tests

- **File:** `tests/online/test_online_method_fidelity.py`,
  `tests/online/test_online_streaming_baseline_contracts.py` and runner wrapper
  tests
- **Symbol:** method-fidelity, lifecycle and artifact tests
- **Change:** Add assertions for prediction-before-adapt, batch-causal state,
  q99.5 threshold, optimizer identity, raw CANDI representation, exact pool
  behavior and legacy-config rejection.
- **Reason:** Existing tests pass while several exact-reference mismatches
  remain untested.
- **Dependencies:** Phases 1-4.

#### 3. Recalibrate and preserve artifact identity

- **File:** threshold artifact output, benchmark report and online records
- **Symbol:** threshold/provenance metadata and output roots
- **Change:** Recalibrate clean-validation thresholds after the final score
  protocol. Record profile, batch size, optimizer, checkpoint role, SANA choice,
  selected range and comparability identity. Keep old artifacts unchanged.
- **Reason:** Score distributions and parameter-update behavior change when the
  runtime flow changes.
- **Dependencies:** one passing smoke run under the corrected profile.

### Verification

#### Automated

- [ ] Generator and schema validation pass for all corrected main/smoke configs.
- [ ] Focused method/lifecycle tests pass.
- [ ] `git diff --check` passes and no stale corrected config silently selects
  `main`, `A0`, `A1` or `A2` behavior.

#### Manual

- [ ] Review one M2N2 and one CANDI threshold artifact and benchmark report.
- [ ] Confirm old and corrected output roots remain non-comparable.

### Risks

- **Risk:** Existing tables or reports mix q99/AdamW and reference-profile
  results. **Mitigation:** use new result identities and mark old artifacts
  non-comparable.

## Phase 6: Validate one real end-to-end run and approve expansion

### Goal

Prove the corrected runtime through the real benchmark entry point before any
full matrix execution.

### Changes

#### 1. Run focused tests and one smoke per method

- **File:** existing benchmark entry point and corrected smoke configs
- **Symbol:** `run_online_streaming_benchmark`
- **Change:** Run one M2N2 and one CANDI configuration through the selected
  reference profile and inspect records, threshold artifact and report.
- **Reason:** Unit tests cannot prove the complete loader-calibration-stream-
  artifact path.
- **Dependencies:** Phases 0-5.

#### 2. Review acceptance evidence

- **File:** generated `online_records.json`, threshold artifact and benchmark
  report
- **Symbol:** method metadata, update records and stream selection
- **Change:** Confirm finite scores/losses, pre-update prediction records,
  correct batch boundaries, updated-next-batch feedback, checkpoint identity and
  metrics-only label usage.
- **Reason:** These are the acceptance conditions for the runtime-flow claim.
- **Dependencies:** smoke outputs.

### Verification

#### Automated

- [ ] Focused method/lifecycle tests pass.
- [ ] One real M2N2 smoke completes with finite output and expected metadata.
- [ ] One real CANDI smoke completes with finite output, pool diagnostics and
  expected metadata.
- [ ] Full relevant `tests/online` suite passes after both smokes.

#### Manual

- [ ] Trace the first two batches in each report and verify
  `score -> prediction/record -> adapt -> next batch`.
- [ ] Approve matrix expansion only after the corrected artifacts are clearly
  separate from historical approximation artifacts.

### Risks

- **Risk:** A smoke can pass shapes while using the wrong contract. **Mitigation:**
  require the exact-order, threshold, optimizer and CANDI representation tests
  before accepting the smoke.

## Testing strategy

Use deterministic unit tests for each state transition, then runner-level
integration tests and one real smoke per method. Test both `batch_size=1` and the
selected reference batch profile if the runner supports both. Keep test labels
outside calibration, FPM selection, M2N2 masks, losses and optimizer updates.

## Migration and rollback

Do not overwrite existing q99/AdamW or legacy approximation artifacts. Use a new
runtime-profile and output identity for corrected runs. If the exact SANA or
batch-profile change fails, retain the RedLamp adapter variant as historical,
mark the attempted artifacts incomplete, and return to the last validated
profile without mixing results.

## Documentation

- Update `documents/spec/online_benchmark_contract.md` with the selected runtime
  profile and exact RedLamp-variant boundary.
- Keep this report's follow-up audit and implementation evidence synchronized
  with the actual code and test results.
- Record threshold source, optimizer, batch size, SANA implementation identity,
  checkpoint role and comparability identity in generated reports.

## Final verification

- [x] The active runner follows the selected Predictor flow and call order.
- [x] Threshold and optimizer metadata match the selected reference profile.
- [x] M2N2 passes Detrender, masked-loss, optimizer and causal-batch tests.
- [x] CANDI passes raw-representation FPM, pool-gate, SANA and causal-batch
  tests.
- [x] One real smoke per method passes with separate provenance.
- [x] The full relevant online test suite passes before matrix expansion.

## Assumptions and non-blocking uncertainties

- The approved RedLamp encoder checkpoint remains the model surface for this
  variant; this plan does not claim native MLP/TimesNet equivalence.
- The exact SANA layer port is a fidelity boundary. If the project keeps the
  current compatible SANA module, the metadata must retain variant-level naming.
- The reference Predictor's aggregate train/test metrics are only required in
  the active report if the benchmark contract consumes them; the adapter's
  score/predict/adapt lifecycle remains required regardless.

## Follow-up implementation record: corrected runtime profile

The follow-up implementation has applied the remaining runtime corrections
within the RedLamp adapter variant. The code still does not claim native
MLP/TimesNet checkpoint equivalence.

### Implemented changes

- `AdaptiveStreamingBaselineBase` now accepts an adaptation batch and appends
  every batch record before calling the adapter update. The update runs once for
  that batch, so only the next batch sees the new state.
- The benchmark runner now collects a pre-TTA test-score summary before the
  test stream can update the adapter. The summary stores counts and finite-score
  ranges, not every forward-pass tensor.
- The corrected adapter protocol uses clean-validation raw-score quantile
  `0.995`, which corresponds to reference `ANOMALY_RATIO=0.5` percent. The old
  q99 profile remains available for historical and traditional runs.
- M2N2 and CANDI now use explicit optimizer settings. The corrected profile uses
  SGD with learning rate `1e-4`, weight decay `1e-4`, momentum `0.9`, zero
  dampening and Nesterov enabled.
- M2N2 keeps the reference Detrender update, timestep mask and masked loss.
- CANDI FPM selection now computes current representations from raw `x`, keeps
  separate hard/moderate pools, uses the reference score predicates and records
  optional pool-label counters without using labels for selection or loss.
- CANDI SANA now uses the reference temporal embedding, full attention,
  convolutional feed-forward layer, final normalization and feature gating.
- Corrected YAML generation points to
  `smd_window20_cleanval_q995_ewma09.yaml` and records optimizer and batch
  fields explicitly.

### Verification evidence

- Focused method/config tests: `8 passed`.
- Full online suite: `91 passed`.
- Corrected M2N2 smoke: completed for `machine-1-6`, seed `6`, range
  `[146,2200)`, 16 steps; report records q99.5, SGD, checkpoint role/SHA-256,
  pre-TTA summary and finite masked losses.
- Corrected CANDI smoke: completed for the same entity, seed, range and step
  limit; report records q99.5, SGD, raw-FPM/SANA profile, checkpoint identity,
  pre-TTA summary and pool diagnostics. The moderate pool contains 13 samples
  at the end of this 16-step smoke, so the `MIN_SAMPLES=16` update gate does not
  fire in that short run.
- `git diff --check` passed.

Repository-wide `.venv/bin/pytest -q` was also run. It produced `484 passed`,
`1 skipped` and `6 failed`. The failures are in pre-existing dirty offline,
compliance snapshot and multitask-model files outside this online adapter
change. The required `tests/online` suite remains green with `91 passed`.

### Current approval boundary

The corrected profile is ready for manual artifact review. The full benchmark
matrix remains unapproved until the reviewer confirms that corrected output
roots are separated from legacy q99/AdamW artifacts and accepts the two smoke
reports. No matrix run was started in this task.
