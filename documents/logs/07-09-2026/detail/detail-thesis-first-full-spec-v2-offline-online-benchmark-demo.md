---
date: 2026-07-09
researcher: Artificial Intelligence Agent
repository: bachelor-thesis-2026
topic: "Detailed programming plan for THESIS-first full-spec-v2 implementation, fair offline/online benchmarks, and demo"
source_research: documents/logs/07-08-2026/research/research-fair-baseline-map-for-full-spec-v2.md
source_plan: documents/logs/07-08-2026/plan/plan-thesis-first-full-spec-v2-offline-online-benchmark-demo.md
source_structure: documents/logs/07-09-2026/structure/structure-thesis-first-full-spec-v2-offline-online-benchmark-demo.md
prompt: prompts/4_detail_prompt.md
status: detailed_plan_for_review
---

# Detail: THESIS-first full-spec-v2 implementation

## 0. Implementation Contract

This detailed plan is the implementation contract before code changes. It preserves the locked workflow:

```text
1_research -> 2_plan -> 3_structure -> 4_detail -> implementation
```

The implementation must complete THESIS first. Baselines are added only after THESIS artifacts define the shared protocol boundary.

Target entities:

```text
machine-1-6
machine-3-4
machine-3-9
```

Main epoch budget:

```yaml
epochs: 30
two_stage:
  expected_total_training_epochs: 30
  stage_a_multitask_epochs: 25
  stage_b_fusion_finetuning_epochs: 5
```

Required THESIS online variants:

```text
O0-A0
O0-A1
O0-A2
O1-A0
O1-A1
O1-A2
```

Synthetic profile decision:

```text
Keep legacy and visible profiles.
Run one short synthetic-profile comparison before selecting the main benchmark profile.
```

## 1. Global Code Quality Rules

Every implementation phase must follow these rules from `codebase_preferences.md`.

- Keep functions and class methods at or below 50 lines.
- Keep Python files at or below 500 lines.
- Prefer explicit names over abbreviations.
- Keep one runtime concept per file.
- Add short ASCII diagrams in protocol-heavy files.
- Explain code as if the reader is a high-school student.
- Keep model-specific forward and loss behavior inside the model file or the existing model mixins.
- Put non-model protocol helpers outside model files.

Recommended comment style:

```text
₍^. .^₎⟆ Where this helper fits

raw sequence
  -> window starts
  -> window records
  -> model scores
  -> point-level threshold
```

Use cute markers sparingly. They should make navigation easier, not distract from the code.

## 2. Shared Interfaces

### 2.1 Window Record Contract

A window record must keep the current repository shape:

```python
window = {
    "x": Tensor[L, D],
    "point_labels": Tensor[L] | None,
    "mask": Tensor[L] | None,
    "timestamps": Tensor[L] | None,
    "meta": {
        "dataset_name": str,
        "entity_id": str,
        "split": str,
        "start_index": int,
        "end_index": int,
        "window_size": int,
        "series_id": str,
        "absolute_start_index": int,
        "absolute_end_index": int,
        "source_sequence_length": int,
        "tail_policy": str,
        "is_tail_window": bool,
    },
}
```

The new fields `tail_policy` and `is_tail_window` must be additive. Existing code that ignores them should keep working.

### 2.2 Point Score Contract

All methods must export point scores as:

```python
point_scores: np.ndarray | torch.Tensor  # shape [T]
point_labels: np.ndarray | torch.Tensor  # shape [T]
covered_point_mask: np.ndarray | torch.Tensor  # shape [T]
```

Score polarity is fixed:

```text
larger score = more anomalous
```

### 2.3 Threshold Artifact Contract

Threshold artifacts must serialize as JSON-compatible dictionaries:

```python
{
    "artifact_version": 1,
    "method_name": str,
    "variant_name": str,
    "entity_id": str,
    "seed": int,
    "window_size": 20,
    "thresholds": {
        "offline_point": {
            "value": float,
            "source_split": "clean_validation",
            "score_rule": "nonoverlap_tail_average",
            "quantile": 0.99,
        },
        "online_ewma_point": {
            "value": float,
            "source_split": "clean_validation",
            "score_rule": "stride1_causal_endpoint_ewma",
            "quantile": 0.99,
            "ewma_current_weight": 0.9,
            "ewma_previous_weight": 0.1,
        },
    },
    "provenance": {
        "test_label_usage": "metrics_only",
        "created_by": str,
        "config_path": str,
    },
}
```

Window input and latent thresholds can be added under the same `thresholds` key after the basic offline/online point thresholds pass tests.

### 2.4 Online Record Contract

Each online record must include:

```python
{
    "entity_id": str,
    "point_index": int,
    "window_start_index": int,
    "window_end_index": int,
    "raw_point_score": float,
    "ewma_point_score": float,
    "threshold": float,
    "prediction": int,
    "online_variant": "A0" | "A1" | "A2",
    "triage_decision": str | None,
    "did_update": bool,
    "loss_total": float | None,
}
```

A0 must still write records with `did_update: false`.

## 3. Phase 1: Shared Protocol Foundation

### 3.1 Summary

This phase adds the shared protocol layer. It must be implemented and tested before model or baseline changes.

### 3.2 File-Level Edits

Add:

```text
src/protocols/__init__.py
src/protocols/smd_benchmark_protocol.py
src/protocols/point_scores.py
src/protocols/threshold_artifact.py
src/protocols/synthetic_profile.py
configs/protocol/smd_window20_cleanval_q99_ewma09.yaml
configs/protocol/synthetic_redlamp12_legacy_window20.yaml
configs/protocol/synthetic_redlamp12_visible_window20.yaml
```

Modify:

```text
src/data/window.py
src/core/config.py
src/core/config_experiment_validation.py
src/engine/evaluator.py
src/engine/thresholding.py
```

### 3.3 Concrete Edit Content

In `src/protocols/smd_benchmark_protocol.py`, define:

```python
SMD_BENCHMARK_ENTITIES = ("machine-1-6", "machine-3-4", "machine-3-9")
SMD_BENCHMARK_SEEDS = (6, 8, 36)

def validate_protocol_config(config: dict[str, Any]) -> None:
    ...
```

Required protocol keys:

```yaml
protocol_name: smd_window20_cleanval_q99_ewma09
window_size: 20
offline_tail_policy: end_align
offline_threshold_split: clean_validation
offline_threshold_quantile: 0.99
online_window_stride: 1
online_threshold_split: clean_validation
online_threshold_quantile: 0.99
online_ewma_current_weight: 0.9
online_ewma_previous_weight: 0.1
test_label_usage: metrics_only
point_adjustment: false
```

In `src/protocols/point_scores.py`, define:

```python
def build_nonoverlap_tail_window_starts(
    sequence_length: int,
    window_size: int,
) -> list[int]:
    ...

def average_overlapping_point_scores(
    sequence_length: int,
    window_scores: Sequence[np.ndarray],
    window_starts: Sequence[int],
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    ...

def window_scores_to_causal_endpoint_scores(
    window_scores: Sequence[float],
    sequence_length: int,
    window_size: int,
) -> np.ndarray:
    ...

def ewma_scores(
    point_scores: np.ndarray,
    current_weight: float,
    previous_weight: float,
) -> np.ndarray:
    ...
```

Implementation notes:

- If `sequence_length < window_size`, return an empty start list for offline evaluation.
- Regular starts are `0, window_size, 2 * window_size, ...`.
- If the final regular window does not end at `sequence_length`, append `sequence_length - window_size`.
- Do not append a duplicate start.
- In overlap averaging, use sum and count arrays.
- Points with count zero remain `nan` and are excluded by `covered_point_mask`.

In `src/data/window.py`, keep the old `slice_sequence_into_windows` signature working, and add optional `tail_policy: str = "drop"`. The existing train path should keep current behavior unless configs opt into `end_align`.

Suggested signature:

```python
def slice_sequence_into_windows(
    raw_sequence: dict[str, Any],
    window_size: int = 100,
    stride: int = 10,
    tail_policy: str = "drop",
) -> list[dict[str, Any]]:
    ...
```

In `src/engine/thresholding.py`, keep current helpers and add:

```python
def select_clean_validation_point_threshold(
    clean_validation_point_scores: np.ndarray,
    quantile: float,
) -> float:
    ...

def select_online_ewma_threshold(
    clean_validation_ewma_scores: np.ndarray,
    quantile: float,
) -> float:
    ...
```

Use `np.nanquantile` and ignore `nan` warm-up points.

In `src/protocols/threshold_artifact.py`, define:

```python
def build_threshold_artifact(...) -> dict[str, Any]:
    ...

def write_threshold_artifact(artifact: dict[str, Any], output_path: Path) -> None:
    ...

def load_threshold_artifact(path: Path) -> dict[str, Any]:
    ...
```

### 3.4 Tests

Add:

```text
tests/test_nonoverlap_tail_windowing.py
tests/test_point_score_contracts.py
tests/test_threshold_artifact.py
tests/test_online_ewma_threshold.py
tests/test_benchmark_protocol_config.py
```

Important cases:

- `sequence_length = 100`, `window_size = 20` gives starts `[0, 20, 40, 60, 80]`.
- `sequence_length = 95`, `window_size = 20` gives starts `[0, 20, 40, 60, 75]`.
- Tail overlap points `75..79` average scores from starts `60` and `75`.
- Online EWMA ignores the first `window_size - 1` warm-up points.
- Threshold artifact round-trips through JSON.

### 3.5 Acceptance Criteria

- New tests pass.
- Existing windowizer tests still pass.
- No existing train config changes behavior unless it sets the new tail policy.
- New protocol config loads through the config system.

## 4. Phase 2: Synthetic Profile Comparison

### 4.1 Summary

This phase keeps both synthetic profiles, compares them briefly, and only then selects the main benchmark profile.

### 4.2 File-Level Edits

Modify:

```text
src/data/augment.py
src/core/config_model_validation.py
scripts/visualize_synthetic_anomalies.py
```

Add:

```text
scripts/compare_synthetic_profiles.py
configs/protocol/synthetic_redlamp12_legacy_window20.yaml
configs/protocol/synthetic_redlamp12_visible_window20.yaml
tests/test_synthetic_anomaly_visibility_profile.py
```

### 4.3 Concrete Edit Content

In `src/data/augment.py`, keep the existing family registry. Add optional class-specific settings without changing class ids:

```python
family_intensity: dict[str, dict[str, float | int | list[float]]]
```

Supported initial keys:

```yaml
family_intensity:
  spike:
    spike_scale_multiplier: 1.2
    max_spikes: 2
  flip:
    min_segment_fraction: 0.35
    max_segment_fraction: 0.5
  speedup:
    factors: [2.0, 3.0, 4.0]
  cutoff:
    force_mode: hold_or_zero
  average:
    min_segment_fraction: 0.35
    max_segment_fraction: 0.5
  scale:
    factors: [0.2, 0.4, 1.8, 2.4]
  wander:
    drift_multiplier: 1.3
  contextual:
    offset_multiplier: 1.3
  upsidedown:
    visibility_multiplier: 1.2
  mixture:
    keep_legacy_behavior: true
```

Keep legacy config equal to the current values:

```yaml
min_segment_fraction: 0.2
max_segment_fraction: 0.3
spike_scale: 3.0
anomaly_visibility_boost: 1.5
```

`scripts/compare_synthetic_profiles.py` should:

- load one SMD data config;
- sample a small fixed batch;
- apply legacy and visible profiles with the same seed;
- save a small JSON summary;
- save side-by-side figures using existing visualization helpers.

Suggested output:

```text
outputs/synthetic_profile_comparison/
    machine_1_6_seed7_summary.json
    machine_1_6_seed7_12_class_grid.png
```

### 4.4 Tests

Add tests for:

- 12-class order unchanged;
- deterministic metadata under fixed seed;
- visible profile preserves `mixture` behavior unless explicitly enabled;
- profile config validation rejects unknown family names.

### 4.5 Acceptance Criteria

- Both profiles can be loaded.
- Comparison script runs on `machine-1-6` smoke data.
- The detail report records which profile is selected for the main benchmark.

## 5. Phase 3: THESIS Offline O0/O1 Completion

### 5.1 Summary

This phase completes THESIS offline artifacts before any baseline is integrated.

### 5.2 File-Level Edits

Modify:

```text
scripts/run_two_stage_offline_pretraining.py
scripts/evaluate.py
src/engine/evaluator.py
src/engine/thresholding.py
src/models/thesis_multitask.py
src/models/thesis_multitask_loss_mixin.py
src/models/thesis_multitask_routing_mixin.py
```

Add:

```text
scripts/run_thesis_offline_benchmark.py
configs/experiment/offline_benchmark/thesis/
tests/test_thesis_offline_artifact_exports.py
```

### 5.3 Concrete Edit Content

`scripts/run_thesis_offline_benchmark.py` should wrap the existing two-stage runner:

```text
load experiment config
  -> load protocol config
  -> run two-stage offline pretraining
  -> evaluate clean validation for offline threshold
  -> evaluate synthetic validation for synthetic diagnostics
  -> evaluate test for final metrics
  -> write threshold artifact
  -> write score artifacts
```

The script should not duplicate training logic from `scripts/run_two_stage_offline_pretraining.py`.

Expected output tree:

```text
outputs/benchmark/smd/thesis/{variant}/{entity}/seed{seed}/
    two_stage/
    thresholds/thresholds.json
    scores/clean_validation_point_scores.npz
    scores/synthetic_validation_point_scores.npz
    scores/test_point_scores.npz
    metrics/offline_metrics.json
    protocol/resolved_protocol.json
```

`src/engine/evaluator.py` should support a protocol-driven point reconstruction mode:

```python
point_reconstruction_rule: "legacy_overlap_average" | "nonoverlap_tail_average"
```

For the new benchmark path, use `nonoverlap_tail_average`. Keep the legacy rule available for old tests and historical configs.

O1 checks:

- `score_loss_type: pointwise_balanced_bce_logits`;
- `score_loss_target: synthetic_anomaly_mask`;
- Stage A uses score loss;
- Stage B does not use point-score BCE by default unless config explicitly enables it.

### 5.4 Tests

Run or add:

```text
tests/test_thesis_multitask_point_score_loss.py
tests/test_multitask_shapes.py
tests/test_one_multitask_train_step.py
tests/test_thesis_offline_artifact_exports.py
tests/test_config_loading.py
```

Smoke command after implementation:

```bash
.venv/bin/python scripts/run_thesis_offline_benchmark.py \
  --experiment-config configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O0__machine_1_6__w20__seed6__smoke.yaml
```

### 5.5 Acceptance Criteria

- O0 and O1 smoke configs load.
- O0 smoke writes threshold and score artifacts.
- O1 smoke confirms point-score loss path.
- Threshold artifact says `source_split: clean_validation`.

## 6. Phase 4: THESIS Online A0/A1/A2 Completion

### 6.1 Summary

This phase implements the full THESIS online test-time adaptation path. It must load offline artifacts from Phase 3.

### 6.2 File-Level Edits

Create:

```text
src/engine/online_tta/__init__.py
src/engine/online_tta/online_engine.py
src/engine/online_tta/online_losses.py
src/engine/online_tta/online_optimizer.py
src/engine/online_tta/triage.py
src/engine/online_tta/verification_buffer.py
src/engine/online_tta/ttl_buffer.py
scripts/run_thesis_online_benchmark.py
configs/experiment/online_benchmark/thesis/
```

Modify:

```text
src/models/online_adaptation.py
src/engine/online_loop.py
src/data/stream.py
```

### 6.3 Concrete Edit Content

`src/data/stream.py` should support two stream modes:

```python
stream_window_mode: "sliding_stride_1" | "nonoverlap_tail"
```

Online TTA uses `sliding_stride_1`. Offline threshold calibration uses non-overlap plus tail-overlap through Phase 1 helpers.

`src/engine/online_tta/online_optimizer.py` should expose:

```python
def collect_projector_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    ...

def assert_only_projector_is_trainable(model: torch.nn.Module) -> None:
    ...
```

`src/engine/online_tta/online_losses.py` should expose:

```python
def compute_a1_pnn_reconstruction_loss(...) -> torch.Tensor:
    ...

def compute_a2_hard_old_reconstruction_loss(...) -> torch.Tensor:
    ...

def compute_a2_online_contrastive_loss(...) -> torch.Tensor:
    ...
```

`src/engine/online_tta/triage.py` should expose:

```python
def classify_online_window(
    input_window_score: float,
    latent_window_score: float,
    thresholds: dict[str, Any],
) -> str:
    ...
```

Use explicit return names:

```text
hard_old_normality
gray_zone
strong_anomaly
pnn_candidate
```

`src/engine/online_tta/online_engine.py` should own the high-level loop:

```text
load checkpoint and thresholds
  -> calibrate online threshold on clean validation stride-1 EWMA
  -> stream test windows
  -> score endpoint
  -> update EWMA
  -> A0/A1/A2 branch
  -> write online records
```

### 6.4 Online Variant Contracts

A0:

```text
forward only
no optimizer
no parameter updates
write online records
```

A1:

```text
forward
triage
PNN reconstruction-only loss
projector-only optimizer step
write online records and loss
```

A2:

```text
forward
triage
hard-old or PNN reconstruction loss
online contrastive regularizer
projector-only optimizer step
write online records and loss
```

### 6.5 Tests

Add or update:

```text
tests/test_online_tta_trainable_surface.py
tests/test_online_tta_variants.py
tests/test_online_tta_triage.py
tests/test_online_verification_buffer.py
tests/test_online_ewma_threshold.py
tests/test_online_entrypoint.py
```

Smoke command:

```bash
.venv/bin/python scripts/run_thesis_online_benchmark.py \
  --experiment-config configs/experiment/online_benchmark/thesis/smd__thesis__online__O0_A0__machine_1_6__w20__seed6__smoke.yaml
```

### 6.6 Acceptance Criteria

- A0 changes no parameters.
- A1 changes only `online_mlp_projector`.
- A2 changes only `online_mlp_projector`.
- Online threshold artifact is separate from offline threshold artifact.
- Online records contain raw score, EWMA score, threshold, prediction, update flag, and loss fields.

## 7. Phase 5: THESIS Benchmark Config Matrix

### 7.1 Summary

This phase materializes configs after core THESIS behavior is stable.

### 7.2 File-Level Edits

Add:

```text
scripts/generate_smd_benchmark_configs.py
configs/experiment/offline_benchmark/thesis/
configs/experiment/online_benchmark/thesis/
tests/test_benchmark_config_generation.py
```

### 7.3 Config Matrix

Generate:

```text
offline variants: O0, O1
online variants: O0-A0, O0-A1, O0-A2, O1-A0, O1-A1, O1-A2
entities: machine-1-6, machine-3-4, machine-3-9
seeds: 6, 8, 36
```

Main config naming:

```text
smd__thesis__offline__O0__machine_1_6__w20__seed6__main.yaml
smd__thesis__online__O1_A2__machine_3_9__w20__seed36__main.yaml
```

### 7.4 Acceptance Criteria

- Every generated config loads.
- Configs point to protocol files instead of duplicating hidden constants.
- Old `configs/experiment/benchmark/` remains read-supported.

## 8. Phase 6: Offline Baselines

### 8.1 Summary

Baselines are added after THESIS offline artifacts define the reference protocol.

### 8.2 File-Level Edits

Create:

```text
src/baselines/__init__.py
src/baselines/traditional/__init__.py
src/baselines/traditional/base.py
src/baselines/traditional/stumpy_channel_ab.py
src/baselines/traditional/kmeans_ad.py
src/baselines/traditional/iforest.py
src/baselines/neural/__init__.py
src/baselines/neural/redlamp.py
scripts/run_offline_benchmark.py
configs/experiment/offline_benchmark/redlamp/
configs/experiment/offline_benchmark/stumpy/
configs/experiment/offline_benchmark/kmeans_ad/
configs/experiment/offline_benchmark/iforest/
```

### 8.3 Adapter Interface

`src/baselines/traditional/base.py` should define a simple protocol:

```python
class TraditionalBaselineProtocol(Protocol):
    def fit(self, train_sequence: np.ndarray) -> None:
        ...

    def calibrate(self, clean_validation_sequence: np.ndarray) -> dict[str, Any]:
        ...

    def score_sequence(self, query_sequence: np.ndarray) -> np.ndarray:
        ...
```

All adapters must return point scores with shape `[T]`.

### 8.4 STUMPY Main Baseline

Implement only the primary fair variant first:

```text
STUMPY-ChannelAB-FrozenTrainRef
```

Contract:

- reference split is train only;
- query split is clean validation or test;
- process channels independently;
- robust channel calibration uses clean validation;
- channel aggregation is max after robust validation z-score;
- online reference update is disabled.

### 8.5 KMeansAD and IForest

Use the reference behavior but prevent protocol drift:

- no test-label threshold tuning;
- no contamination from test labels;
- avoid double window-to-point aggregation;
- log any internal z-score/window normalization as method-specific metadata.

### 8.6 Tests

Add:

```text
tests/test_stumpy_channel_ab_contract.py
tests/test_traditional_baseline_contracts.py
tests/test_redlamp_baseline_active_benchmark_config.py
```

Acceptance:

- Each offline baseline smoke produces shared score and threshold artifacts.
- Result summarizer can read THESIS and baseline artifacts through one schema.

## 9. Phase 7: Online Baselines

### 9.1 Summary

Online baselines are added after the THESIS online stream and online threshold artifacts are stable.

### 9.2 File-Level Edits

Create:

```text
src/baselines/online/__init__.py
src/baselines/online/candi.py
src/baselines/online/m2n2.py
scripts/run_online_streaming_benchmark.py
configs/experiment/online_benchmark/candi/
configs/experiment/online_benchmark/m2n2/
configs/experiment/online_benchmark/stumpy/
configs/experiment/online_benchmark/kmeans_ad/
configs/experiment/online_benchmark/iforest/
```

### 9.3 Contracts

CANDI and M2N2:

- use native online adaptation logic;
- consume same stream order;
- use same scaler state;
- report native trainable/update surface;
- output the shared online record schema.

STUMPY, KMeansAD, IForest:

- run as frozen streaming scorers;
- no test stream state update in main table;
- use causal endpoint scores and online EWMA threshold.

### 9.4 Tests

Add or update:

```text
tests/test_online_stream.py
tests/test_online_entrypoint.py
tests/test_traditional_baseline_contracts.py
```

Acceptance:

- Every online method emits records with the same schema.
- Warm-up points are masked consistently.
- Online threshold source is `clean_validation_stride1_ewma`.

## 10. Phase 8: Reporting and Demo

### 10.1 Summary

Reporting and demo consume stable artifacts. They must not duplicate official metric logic.

### 10.2 File-Level Edits

Create:

```text
scripts/summarize_benchmark_results.py
demo/app.py
demo/demo_state.py
demo/loaders.py
demo/offline_replay.py
demo/online_replay.py
demo/plotting.py
configs/experiment/demo/demo_offline_replay.yaml
configs/experiment/demo/demo_online_stream.yaml
```

### 10.3 Reporting Contracts

Summary rows must include:

```text
method
variant
entity_id
seed
benchmark_type
threshold_source
point_rule
smoothing_rule
test_label_usage
runtime_seconds
metrics
```

### 10.4 Demo Contracts

Demo mode 1:

```text
offline replay
  -> load signal
  -> load point scores
  -> load threshold
  -> display predictions and optional labels
```

Demo mode 2:

```text
online replay
  -> load online records
  -> animate point stream
  -> display raw score, EWMA score, threshold, TTA mode
```

### 10.5 Tests

Add:

```text
tests/test_demo_state.py
tests/test_evaluation_metrics_audit.py
tests/test_evaluation_protocol_audit.py
```

Acceptance:

- Summary table marks oracle or non-causal rows.
- Demo can replay one saved offline run and one saved online run.
- Demo code does not compute official benchmark metrics.

## 11. Risk Mitigation Checklist

Prototype redundancy:

- Keep continuous and discrete branch diagnostics in THESIS metrics.
- Add summary fields for branch usage and discrete codeword usage.

Fusion collapse:

- Preserve existing fusion metrics.
- Log reconstruction branch contributions if already available.

Adaptation contamination:

- Use clean-validation thresholds only.
- Use projector-only online updates.
- Keep strong anomalies out of online update targets.

Projector drift:

- Assert only projector parameters are trainable.
- Log projector gradient norm.
- Keep online optimizer state reset policy explicit.

Metric inflation:

- No test threshold tuning.
- No point adjustment in main table.
- Mark oracle and self-join variants separately.
- Keep score-based and threshold-based results separate.

## 12. Validation Order

Run focused tests first:

```bash
.venv/bin/python -m pytest \
  tests/test_nonoverlap_tail_windowing.py \
  tests/test_point_score_contracts.py \
  tests/test_threshold_artifact.py \
  tests/test_online_ewma_threshold.py
```

Then synthetic profile checks:

```bash
.venv/bin/python -m pytest \
  tests/test_synthetic_anomaly_injection.py \
  tests/test_synthetic_anomaly_visibility_profile.py
```

Then THESIS model checks:

```bash
.venv/bin/python -m pytest \
  tests/test_multitask_shapes.py \
  tests/test_one_multitask_train_step.py \
  tests/test_thesis_multitask_point_score_loss.py
```

Then online checks:

```bash
.venv/bin/python -m pytest \
  tests/test_online_tta_trainable_surface.py \
  tests/test_online_tta_variants.py \
  tests/test_online_ewma_threshold.py
```

Then one vertical smoke:

```bash
.venv/bin/python scripts/run_thesis_offline_benchmark.py \
  --experiment-config configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O0__machine_1_6__w20__seed6__smoke.yaml

.venv/bin/python scripts/run_thesis_online_benchmark.py \
  --experiment-config configs/experiment/online_benchmark/thesis/smd__thesis__online__O0_A0__machine_1_6__w20__seed6__smoke.yaml
```

## 13. First Implementation Slice

Implement the first slice in this order:

1. Add protocol helpers and tests.
2. Add threshold artifact helpers and tests.
3. Add non-overlap tail window support behind a config key.
4. Add legacy and visible synthetic profile configs.
5. Add short synthetic profile comparison script.
6. Add THESIS offline wrapper and artifact export.
7. Run O0 offline smoke for `machine-1-6`, seed `6`.
8. Add online threshold calibration from clean validation stride-1 EWMA.
9. Add A0 online benchmark wrapper.
10. Run O0-A0 online smoke for `machine-1-6`, seed `6`.

This slice proves the protocol before A1, A2, and baselines increase the number of moving parts.

## 14. Accepted Review Gate Before Coding

These two implementation gates are accepted by the user:

1. The synthetic-profile short comparison is visual and statistical, not a full 30-epoch model benchmark.
2. The first implementation slice stops after O0-A0 smoke before adding A1/A2 and baselines.

Implementation may start from tests for Phase 1.

The first coding checkpoint is therefore:

```text
tests first
  -> protocol helpers
  -> threshold artifact helpers
  -> non-overlap tail window support
  -> synthetic profile comparison
  -> THESIS O0 offline smoke
  -> THESIS O0-A0 online smoke
```
