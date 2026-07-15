---
date: 2026-07-15 15:05:39 +0700
researcher: TheMetaSetter
git_commit: 3005102f0c8e8a5f62a842c3da3d9a1a71118ed0
branch: dev
repository: bachelor-thesis-2026
topic: "UQ persistence patch points after Stage A and Stage B"
tags: [research, thesis, uncertainty-quantification, checkpoint, traces]
status: complete
last_updated: 2026-07-15
last_updated_by: TheMetaSetter
---

# Research: UQ persistence patch points after Stage A and Stage B

**Date**: 2026-07-15 15:05:39 +0700  
**Researcher**: TheMetaSetter  
**Git Commit**: 3005102f0c8e8a5f62a842c3da3d9a1a71118ed0  
**Branch**: dev

## Research Question
Làm sao để bảo đảm các field UQ trong `documents/inventories/detail-thesis-uq-field-inventory.md` vẫn được lưu đúng sau khi chạy Stage A và Stage B?

## Summary
Checkpoint hiện tại chỉ nên giữ provenance và control metadata UQ. Payload UQ đầy đủ như `stochastic_query` và `uncertainty` phải được export ra trace artifact riêng trong evaluation. Vì vậy, phần cần vá không phải là “nhét thêm tensor vào checkpoint”, mà là làm rõ 3 điểm: checkpoint provenance, trace export, và fail-fast validation.

## Detailed Findings

### Data Preparation
- Không có thay đổi dữ liệu cần thiết cho mục tiêu này.
- Trọng tâm là vòng đời của artifact sau khi model forward và evaluation chạy xong.

### Modeling and Training
- `src/models/thesis_multitask_impl/thesis_multitask_state_serialization_mixin.py` đã lưu:
  - `verification_metadata_source`
  - `verification_codeword_class_ids`
  - `verification_contributing_token_counts`
  - `anomalous_codeword_mask`
  - `anomaly_radii`
  - các field UQ control như `monte_carlo_samples`, `return_mc_samples`, `sample_retention_policy`
- `src/models/thesis_multitask_impl/thesis_multitask_state_memory_mixin.py` là nơi Stage A gán:
  - `verification_metadata_source = "train_anomaly_tokens_q99"`
- `src/engine/trainer.py` đã có nhánh refresh `best.pt` sau khi memory đã initialized, nên đây là điểm cần giữ để tránh checkpoint bị chốt khi provenance còn `uninitialized`.

### Evaluation
- `src/engine/evaluator.py` đã build trace payload chứa:
  - `stochastic_query`
  - `uncertainty_history`
  - `mc_sample_histories`
- `scripts/benchmarks/run_thesis_offline_benchmark.py` đã gom trace payloads và summary metrics.
- `src/engine/checkpoint.py` chỉ nên tiếp tục lưu UQ control/provenance metadata, không nên giữ raw MC tensor payload.

## Code References
- `src/models/thesis_multitask_impl/thesis_multitask_state_serialization_mixin.py:13-50` - checkpoint extra state UQ/provenance fields
- `src/models/thesis_multitask_impl/thesis_multitask_state_memory_mixin.py:340-365` - Stage A provenance assignment
- `src/engine/trainer.py:817-902` - best/final checkpoint save flow
- `src/engine/evaluator.py:365-405` - trace payload construction
- `scripts/benchmarks/run_thesis_offline_benchmark.py:209-337` - trace summarization and artifact collection
- `src/engine/checkpoint.py:36-69` - checkpoint metadata fields
- `documents/inventories/detail-thesis-uq-field-inventory.md:11-109` - UQ field inventory

## Pipeline Documentation
The clean split is:

1. Stage A and Stage B save checkpoint state with provenance and UQ control metadata.
2. Evaluation forward passes emit runtime UQ payloads.
3. Trace artifacts persist those runtime payloads for audit and analysis.
4. Metrics files persist summary statistics only.

## Historical Context
- Inventory note confirms checkpoint stores metadata/control, not full Monte Carlo payload.
- Offline benchmark already has trace payload collection, so the main gap is making persistence explicit and verified.

## Open Questions
- Should trace export remain JSON only, or add a binary sidecar for large MC tensors.
- Should validation fail immediately when `verification_metadata_source == "uninitialized"` after Stage A is expected to have initialized memories.
- Should the offline and online paths share one common trace writer to avoid drift.

## Patch Points
1. Add a post-save assertion in the trainer for initialized-memory checkpoints.
2. Add a trace persistence contract for `stochastic_query` and `uncertainty`.
3. Add targeted tests for checkpoint provenance and trace export.
4. Keep `checkpoint_metadata` and `extra_state` small and stable.
