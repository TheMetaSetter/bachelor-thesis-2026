---
date: 2026-07-09
researcher: Artificial Intelligence Agent
repository: bachelor-thesis-2026
topic: "Structure outline for THESIS-first full-spec-v2 implementation, fair offline/online benchmarks, and demo"
source_research: documents/logs/07-08-2026/research/research-fair-baseline-map-for-full-spec-v2.md
source_plan: documents/logs/07-08-2026/plan/plan-thesis-first-full-spec-v2-offline-online-benchmark-demo.md
prompt: prompts/3_structure_prompt.md
status: draft_for_feedback
---

# Structure: THESIS-first full-spec-v2 implementation

## Overview

This structure turns the locked research and plan documents into a staged programming outline. The implementation should complete THESIS first, then force every baseline through the same protocol boundary for SMD `machine-1-6`, `machine-3-4`, and `machine-3-9`.

The central rule is simple: data protocol, point-score protocol, threshold artifacts, and benchmark records are shared. Model-specific code sits behind small adapters. This keeps the codebase useful for research, teaching, and repeated experiments.

## Locked Protocol Summary

### Offline protocol

Clean validation, synthetic validation, and test must use non-overlapping windows. If a sequence has a final leftover segment shorter than `window_size`, create one final full window ending at the sequence end. Only that final window may overlap the previous window. Overlapped points average their anomaly scores.

```text
₍^. .^₎⟆ offline score path

scaled sequence
  -> non-overlap windows
  -> optional final tail-overlap window
  -> model point scores
  -> average overlapped tail points
  -> point-level clean-validation threshold
  -> offline metrics
```

### Online protocol

Online test-time adaptation uses sliding windows with stride `1`. The online threshold is not reused directly from the offline evaluator. It is calibrated by simulating the online stride-1 stream on clean validation, computing online anomaly scores, applying EWMA, and then selecting a point-level threshold from that clean-validation EWMA score stream.

```text
( ˶˘ ³˘)♡ online threshold path

clean validation sequence
  -> sliding windows, stride = 1
  -> score at current window endpoint
  -> EWMA score
  -> point-level online threshold
  -> online TTA evaluation
```

### Code quality protocol

New code should be easy to read for a high-school student. Each file should have one clear reason to exist. Protocol-heavy files should include a short ASCII diagram showing how the local functions fit into the larger pipeline. Function names should explain behavior directly, such as `build_nonoverlap_tail_windows` or `calibrate_online_ewma_threshold`.

## Implementation Phases

### Phase 1: Shared Protocol Foundation

This phase creates the shared protocol layer before changing model behavior. It is the minimal vertical slice because every later THESIS, baseline, and demo path depends on the same windowing, score, and threshold contracts.

Create:

```text
src/protocols/
    __init__.py
    smd_benchmark_protocol.py
    point_scores.py
    threshold_artifact.py
    synthetic_profile.py

configs/protocol/
    smd_window20_cleanval_q99_ewma09.yaml
    synthetic_redlamp12_visible_window20.yaml
```

Responsibilities:

- `smd_benchmark_protocol.py` stores entity ids, seeds, window size, threshold quantile, EWMA weights, and benchmark split policy.
- `point_scores.py` owns non-overlap window starts, tail-overlap window starts, overlap score averaging, causal endpoint pointification, and EWMA.
- `threshold_artifact.py` serializes offline point threshold, online EWMA point threshold, input-window threshold, latent-window band, and provenance.
- `synthetic_profile.py` names the 12-class synthetic anomaly profile and records class-specific intensity settings.

Design principles:

- Separation of concerns: protocol math lives outside models.
- Stable interface: every method returns point scores and threshold artifacts with the same schema.
- Pedagogical clarity: add one ASCII diagram at the top of `point_scores.py` and `threshold_artifact.py`.

Primary tests:

```text
tests/test_benchmark_protocol_config.py
tests/test_nonoverlap_tail_windowing.py
tests/test_point_score_contracts.py
tests/test_threshold_artifact.py
tests/test_online_ewma_threshold.py
```

Acceptance:

- Non-overlap validation/test windows add at most one tail-overlap window.
- Overlapped tail points average exactly two score sources.
- Offline threshold uses clean validation point scores only.
- Online threshold uses clean validation stride-1 simulation plus EWMA.

### Phase 2: Synthetic Anomaly Visibility Profile

This phase improves synthetic anomaly class quality without changing the 12-class label contract. It should happen before the final THESIS benchmark so O0/O1 train on the intended visible synthetic classes.

Modify:

```text
src/data/augment.py
configs/task/multitask_tsad_redlamp_multiclass_window20_benchmark_fixed_synth.yaml
```

Add or expose class-specific controls for:

- stronger `spike`
- longer `flip`
- clearer `speedup`
- clearer `cutoff`
- clearer `average`
- stronger `scale`
- modestly stronger `wander`, `contextual`, and `upsidedown`
- unchanged `mixture` first

Design principles:

- Backward compatibility: keep the old global `spike_scale` and `anomaly_visibility_boost`.
- Strategy-style dispatch: keep each anomaly family isolated in its own injection function.
- Single-meaning config: visible profile values must be explicit and not inferred from test labels.

Primary tests:

```text
tests/test_synthetic_anomaly_injection.py
tests/test_synthetic_anomaly_visibility_profile.py
tests/test_synthetic_anomaly_visualization.py
```

Acceptance:

- Class names and label ids remain unchanged.
- Fixed synthetic seeds produce deterministic masks and metadata.
- Weak classes change enough points or magnitude to be visibly different in side-by-side plots.

### Phase 3: THESIS Offline O0/O1 Completion

This phase completes the offline THESIS artifact path before any baseline is added. O0 and O1 must become the standard that all baselines later match.

Modify:

```text
src/models/thesis_multitask.py
src/models/thesis_multitask_loss_mixin.py
src/models/thesis_multitask_routing_mixin.py
scripts/run_two_stage_offline_pretraining.py
scripts/evaluate.py
src/engine/evaluator.py
src/engine/thresholding.py
```

Create:

```text
scripts/run_thesis_offline_benchmark.py
configs/experiment/offline_benchmark/thesis/
```

Key contracts:

- O0 is two-stage base.
- O1 adds Point-wise Balanced Reconstruction-Score Loss.
- Total epoch budget is `30 = 25 Stage A + 5 Stage B`.
- Stage B keeps encoder and memories frozen as already specified by the active two-stage configs.
- Evaluation uses official non-overlap plus tail-overlap protocol.
- Offline point threshold is calibrated from clean validation only.

Design principles:

- Composition over rewrite: reuse the existing two-stage runner and evaluator where possible.
- Fail-fast validation: config mismatch between protocol window size and model/data window size should raise early.
- Artifact-first design: thresholds, point scores, metrics, and resolved protocol JSON must be written before online work starts.

Primary tests:

```text
tests/test_thesis_multitask_point_score_loss.py
tests/test_thesis_offline_artifact_exports.py
tests/test_benchmark_two_method_configs.py
tests/test_config_loading.py
```

Acceptance:

- O0/O1 configs load for all three entities and seeds.
- One smoke run on `machine-1-6` writes checkpoint, point scores, metrics, and threshold artifact.
- Threshold artifact records clean-validation provenance and offline point rule.

### Phase 4: THESIS Online A0/A1/A2 Completion

This phase implements the full `full-spec-v2` online engine. It must happen after offline artifacts exist because online A0/A1/A2 load an offline THESIS checkpoint and thresholds.

Create:

```text
src/engine/online_tta/
    __init__.py
    online_engine.py
    online_losses.py
    online_optimizer.py
    triage.py
    verification_buffer.py
    ttl_buffer.py

scripts/run_thesis_online_benchmark.py
configs/experiment/online_benchmark/thesis/
```

Modify:

```text
src/models/online_adaptation.py
src/engine/online_loop.py
src/data/stream.py
```

Key contracts:

- A0 performs no online optimizer step.
- A1 updates only `online_mlp_projector` with PNN reconstruction loss.
- A2 updates only `online_mlp_projector` with hard-old reconstruction or PNN reconstruction plus online contrastive regularizer.
- Source encoder, prototype banks, codebook, reconstruction heads, and classification path remain frozen.
- Online threshold is recalibrated from clean validation stride-1 EWMA simulation.

Design principles:

- One file, one concept: losses, triage, buffer, optimizer, and stream engine stay separated.
- Runtime assertion: trainable parameter names must be checked before every online optimizer step.
- Educational diagrams: `online_engine.py` should show the stream-to-score-to-update flow.

Core online flow:

```text
₍₍⚞(˶˃ ꒳ ˂˶)⚟⁾⁾ online THESIS flow

test point stream
  -> latest sliding window
  -> frozen THESIS forward
  -> point score + EWMA
  -> A0: no update
  -> A1/A2: triage + projector-only update
  -> record finalized prediction
```

Primary tests:

```text
tests/test_online_tta_trainable_surface.py
tests/test_online_tta_variants.py
tests/test_online_tta_triage.py
tests/test_online_verification_buffer.py
tests/test_online_ewma_threshold.py
```

Acceptance:

- A0 changes no parameters.
- A1 and A2 change only `online_mlp_projector`.
- Frozen modules have zero gradients or unchanged parameters after online updates.
- Online records include point index, raw score, EWMA score, threshold, prediction, triage decision, update flag, and losses.

### Phase 5: THESIS Benchmark Config Matrix

This phase materializes the full THESIS config set for the three target entities. It should not introduce new behavior; it only makes the locked behavior runnable and repeatable.

Create:

```text
scripts/generate_smd_benchmark_configs.py
configs/experiment/offline_benchmark/thesis/
configs/experiment/online_benchmark/thesis/
```

Required variants:

```text
offline: O0, O1
online: O0-A0, O0-A1, O0-A2, O1-A0, O1-A1, O1-A2
entities: machine-1-6, machine-3-4, machine-3-9
seeds: 6, 8, 36
```

Design principles:

- Generated configs should be boring and explicit.
- Shared protocol config should avoid duplicated hidden constants.
- Existing `configs/experiment/benchmark/` configs stay readable until replacement is verified.

Primary tests:

```text
tests/test_benchmark_config_generation.py
tests/test_config_loading.py
tests/test_config_loading_additional.py
```

Acceptance:

- Every generated config loads with duplicate-key validation.
- Every generated config resolves to `window_size: 20`, `epochs: 30`, and the locked protocol.

### Phase 6: Offline Baseline Integration

This phase starts only after THESIS offline artifacts are stable. Baselines must adapt to the THESIS protocol, not the other way around.

Create:

```text
src/baselines/
    __init__.py
    traditional/
        __init__.py
        base.py
        stumpy_channel_ab.py
        kmeans_ad.py
        iforest.py
    neural/
        __init__.py
        redlamp.py

scripts/run_offline_benchmark.py
configs/experiment/offline_benchmark/
```

Baseline contracts:

- RedLamp reuses the existing in-repo `src/models/redlamp_baseline.py` path but writes shared benchmark artifacts.
- STUMPY main variant is `STUMPY-ChannelAB-FrozenTrainRef`.
- KMeansAD and IForest use train-fit state and clean-validation thresholding.
- All methods output point scores where larger means more anomalous.
- No method uses test labels for threshold selection.

Design principles:

- Adapter pattern: baseline code translates method-native output into the shared artifact schema.
- No double aggregation: if a baseline already returns point scores, do not pass them through another window-to-point mapper.
- Runtime transparency: log fit time, calibration time, inference time, threshold source, and point rule.

Primary tests:

```text
tests/test_stumpy_channel_ab_contract.py
tests/test_traditional_baseline_contracts.py
tests/test_redlamp_baseline_active_benchmark_config.py
```

Acceptance:

- One offline smoke for each baseline on `machine-1-6` produces compatible artifacts.
- STUMPY does not use full-test self-join as the main fair baseline.
- Baseline summaries can be joined with THESIS summaries without special-case columns.

### Phase 7: Online Baseline Integration

This phase adds CANDI, M2N2, and traditional ML frozen streaming scorers to the online benchmark. It depends on the shared online stream and threshold protocol from Phase 4.

Create:

```text
src/baselines/
    online/
        __init__.py
        candi.py
        m2n2.py

scripts/run_online_streaming_benchmark.py
configs/experiment/online_benchmark/
```

Baseline contracts:

- CANDI and M2N2 keep their native online adaptation logic, but consume the same stream, scaler, window size, threshold policy, and metric code.
- STUMPY, KMeansAD, and IForest are frozen streaming scorers in the main online benchmark.
- Adaptive traditional variants, if added later, must be separate rows or tables.

Design principles:

- Honest comparability: native update surfaces are logged, not hidden.
- Shared stream interface: every method receives windows in the same order.
- Causal scoring: no future test windows can affect current predictions.

Primary tests:

```text
tests/test_online_entrypoint.py
tests/test_online_stream.py
tests/test_traditional_baseline_contracts.py
```

Acceptance:

- Online benchmark emits one shared record schema for THESIS, CANDI, M2N2, STUMPY, KMeansAD, and IForest.
- Warm-up points are masked or marked consistently.
- Online metrics use the online EWMA threshold artifact.

### Phase 8: Reporting, Audit, and Demo

This phase is deliberately last. It should consume stable benchmark artifacts instead of duplicating metric logic.

Create:

```text
scripts/summarize_benchmark_results.py

demo/
    app.py
    demo_state.py
    loaders.py
    offline_replay.py
    online_replay.py
    plotting.py

configs/experiment/demo/
    demo_offline_replay.yaml
    demo_online_stream.yaml
```

Reporting contracts:

- Main score-based table uses raw point scores.
- Main label-based table uses clean-validation thresholds.
- Online table uses online EWMA thresholds.
- Oracle or non-causal variants are clearly marked and excluded from main claims.

Demo contracts:

- Demo loads existing checkpoints, point-score artifacts, and threshold artifacts.
- Demo never becomes the official metric pipeline.
- Labels are optional overlays after prediction, not tuning inputs.

Design principles:

- Consumer-only demo: demo code reads artifacts and displays them.
- No duplicated evaluation logic: metrics remain in benchmark/evaluator code.
- Teaching-first UI: show the signal, score, threshold, and prediction path with minimal text.

Primary tests:

```text
tests/test_demo_state.py
tests/test_evaluation_metrics_audit.py
tests/test_evaluation_protocol_audit.py
```

Acceptance:

- Summary tables include method, variant, entity, seed, split, threshold source, point rule, smoothing, and test-label usage.
- Demo can replay one offline THESIS run and one online THESIS run from saved artifacts.

## Dependency Order

```text
Phase 1 protocol
  -> Phase 2 synthetic profile
  -> Phase 3 THESIS offline
  -> Phase 4 THESIS online
  -> Phase 5 THESIS config matrix
  -> Phase 6 offline baselines
  -> Phase 7 online baselines
  -> Phase 8 reporting + demo
```

This order keeps the fairness standard anchored in THESIS and avoids making baseline-specific behavior leak back into the main method.

## First Vertical Slice

The first runnable slice should be intentionally small:

```text
⸜(｡˃ ᵕ ˂ )⸝♡ first slice

machine-1-6, seed 6
  -> protocol helpers
  -> visible synthetic profile smoke
  -> THESIS O0 offline smoke
  -> offline threshold artifact
  -> THESIS O0-A0 online smoke
  -> online EWMA threshold artifact
```

This slice proves the shared protocol, checkpoint loading, score export, and threshold export before A1/A2 and baselines add complexity.

## Resolved Structure Decision

Keep both the legacy synthetic profile and the visible synthetic profile. Run one short comparison before selecting the main benchmark profile. Everything else in this structure follows the locked protocol from the research and plan documents.

## Feedback Request

Does this phase order and granularity look right before moving to `prompts/4_detail_prompt.md`? The next detail step should turn each phase into concrete edit order, file-level contracts, and verification commands.
