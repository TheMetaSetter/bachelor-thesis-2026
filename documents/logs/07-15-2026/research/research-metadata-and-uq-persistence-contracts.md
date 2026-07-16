---
date: 2026-07-15 16:32:50 +07
researcher: TheMetaSetter
git_commit: b9f98e329401455a97d981dff3a4eafe509f9d47
branch: dev
repository: bachelor-thesis-2026
topic: "How to ensure checkpoint metadata is meaningful and how to persist full UQ fields"
tags: [research, time-series, anomaly-detection, multi-class]
status: complete
last_updated: 2026-07-15
last_updated_by: TheMetaSetter
---

# Research: How to ensure checkpoint metadata is meaningful and how to persist full UQ fields

**Date**: 2026-07-15 16:32:50 +07
**Researcher**: TheMetaSetter
**Git Commit**: b9f98e329401455a97d981dff3a4eafe509f9d47
**Branch**: dev

## Research Question
Làm sao để đảm bảo checkpoint có lưu metadata, metadata đó là dữ liệu có nghĩa chứ không phải rỗng, và làm sao để đảm bảo tương tự cho các field UQ trong `detail-thesis-uq-field-inventory.md`?

## Summary
Checkpoint metadata hiện đã có lớp kiểm tra cấu trúc và đối chiếu với config, nhưng chưa có lớp kiểm tra “semantic non-empty” cho mọi field provenance. Phần UQ runtime đã có contract về schema và shape cho `stochastic_query` và `uncertainty`, nhưng benchmark output hiện tại chỉ lưu summary records/metrics, không persist đầy đủ trace payload UQ.

## Detailed Findings

### Data / Metadata Preparation
- `src/core/config_model_validation.py` đã chặn các giá trị đầu vào rỗng hoặc vô hiệu cho `monte_carlo_samples`, `sample_retention_policy`, `return_mc_samples`, `anomaly_families`, và các nhiệt độ UQ.
- `src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py` đặt `verification_metadata_source` mặc định là `"uninitialized"` hoặc `"disabled"` khi branch chưa được khởi tạo.
- `src/models/thesis_multitask_impl/thesis_multitask_state_memory_mixin.py` gán `verification_metadata_source = "train_anomaly_tokens_q99"` khi memory đã khởi tạo từ synthetic verification path.

### Modeling and Checkpointing
- `src/models/thesis_multitask_impl/thesis_multitask_state_serialization_mixin.py` đã normalize provenance khi save/load checkpoint:
  - nếu `memory_initialized` và mask/radii tồn tại nhưng `verification_metadata_source` rỗng, `uninitialized`, hoặc `disabled`, nó rewrite thành `"train_anomaly_tokens_q99"`.
  - `get_checkpoint_extra_state()` lưu `anomalous_codeword_mask`, `anomaly_radii`, codeword class ids, contributing token counts, and UQ control fields.
- `src/engine/checkpoint.py` lưu `checkpoint_metadata` với digest fields và control metadata; validation hiện chủ yếu kiểm tra:
  - key presence
  - type/shape consistency at config level
  - sha256 equality against config
  - matching `memory_label_source`, `stochastic_inference`, `monte_carlo_samples`, temperatures, `variance_correction`, `return_mc_samples`, and `sample_retention_policy`
- Current gap: checkpoint metadata validation does not yet explicitly enforce semantic non-emptiness for every provenance field beyond config equivalence.

### UQ Runtime and Trace Persistence
- `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py` constructs the full UQ runtime payload:
  - `outputs["aux"]["stochastic_query"]`
  - `outputs["aux"]["uncertainty"]`
- `src/core/contracts.py` validates schema and tensor ranks for those fields.
- `src/engine/evaluator.py` builds trace payloads containing:
  - `uncertainty_history`
  - `stochastic_query`
  - `sample_retention_policy`
  - `mc_sample_histories`
- However, the Stage-B output tree inspected on the remote GPU contained only:
  - `evaluation_curves.json`
  - `evaluation_metrics.json`
  - `evaluation_protocol_audit.json`
  - `evaluation_records.json`
  - `metrics.jsonl`
  - `resolved_experiment_config.json`
  and no separate persisted trace file with the full UQ payload.
- Therefore, the current runtime computes UQ traces, but Stage-B benchmark artifacts do not yet persist the full payload in a durable trace artifact.

### Threshold / Artifact Provenance
- `src/protocols/threshold_artifact.py` validates threshold artifact provenance and requires:
  - non-empty `created_by`
  - non-empty `config_path`
  - `thresholds` mapping
  - optional `checkpoint_sha256` / `resolved_config_sha256` only if present and non-null
- `scripts/benchmarks/run_thesis_online_benchmark.py` writes an `online_artifact_manifest.json` with:
  - artifact checksums
  - `resolved_experiment_config_sha256`
  - `threshold_artifact_sha256`
  - `threshold_artifact_path`
- In the remote smoke run, `online_artifact_manifest.json` was valid, but `threshold_artifact.checkpoint_sha256` inside the threshold artifact remained `None`, so provenance is still incomplete for the threshold layer.

## Code References
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/thesis_multitask_impl/thesis_multitask_state_serialization_mixin.py`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/checkpoint.py`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/core/contracts.py`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/evaluator.py`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/protocols/threshold_artifact.py`

## Pipeline Documentation
Checkpoint metadata should be checked at two levels:

1. **Structural validity**: keys exist, types match, sha256 matches config, and checkpoint load/save round-trips.
2. **Semantic validity**: provenance fields should be non-empty and point to a real initialization path, not just a placeholder string.

For UQ fields, the pipeline should distinguish:

1. **Runtime contract**: `stochastic_query` and `uncertainty` tensors exist and have valid shapes.
2. **Persistence contract**: the evaluator or benchmark runner writes trace artifacts that retain the runtime payload, not only summary metrics.

## Historical Context (from documents/)
- `documents/spec/full-spec-v3.md` defines checkpoint provenance and UQ runtime expectations.
- `documents/inventories/detail-thesis-uq-field-inventory.md` lists the exact UQ fields expected in checkpoint metadata, `stochastic_query`, and `uncertainty`.

## Open Questions
- Should Stage-B benchmark output persist the full `trace_payload` directly under the run tree, or only inside a retention bundle?
- Should checkpoint save/load reject `verification_metadata_source` values that are still placeholder-like even when the tensor payload is present?
- Should `threshold_artifact.provenance.checkpoint_sha256` be mandatory for online runs, not optional?

## Follow-up: Stage-B UQ persistence map

For the current codebase, the UQ-related fields split into three groups:

1. **Meaningful checkpoint metadata**
   - `verification_metadata_source`
   - `verification_metadata_schema_version`
   - `verification_metadata_split`
   - `verification_metadata_initialization_seed`
   - `verification_codeword_class_ids`
   - `verification_contributing_token_counts`
   - `stochastic_inference`
   - `monte_carlo_samples`
   - `continuous_temperature`
   - `discrete_temperature`
   - `variance_correction`
   - `return_mc_samples`
   - `sample_retention_policy`
   These are meaningful when they are non-placeholder and align with the loaded model state.

2. **Runtime UQ payloads**
   - `outputs["aux"]["stochastic_query"]`
   - `outputs["aux"]["uncertainty"]`
   - `outputs["aux"]["deterministic_geometry"]`
   These are produced during evaluation and are not persisted inside the checkpoint itself.

3. **Post-stage-B artifact files**
   - `evaluation_traces.json` in `scripts/cli/evaluate.py`
   - `traces/clean_validation_traces.json`
   - `traces/synthetic_validation_traces.json`
   - `traces/test_traces.json`
   These are the files that should carry the trace payload after Stage B finishes.
