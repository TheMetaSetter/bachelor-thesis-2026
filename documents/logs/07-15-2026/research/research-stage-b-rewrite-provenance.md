---
date: 2026-07-15 15:05:39 +0700
researcher: TheMetaSetter
git_commit: 3005102f0c8e8a5f62a842c3da3d9a1a71118ed0
branch: dev
repository: bachelor-thesis-2026
topic: "Stage B rewrite provenance when reusing an old checkpoint"
tags: [research, thesis, checkpoint, provenance, online-adaptation]
status: complete
last_updated: 2026-07-15
last_updated_by: TheMetaSetter
---

# Research: Stage B rewrite provenance when reusing an old checkpoint

**Date**: 2026-07-15 15:05:39 +0700  
**Researcher**: TheMetaSetter  
**Git Commit**: 3005102f0c8e8a5f62a842c3da3d9a1a71118ed0  
**Branch**: dev

## Research Question
Liệt kê những chỗ cần vá để Stage B có thể nạp checkpoint cũ rồi tự rewrite provenance trước khi save lại, thay vì bắt buộc rerun Stage A.

## Summary
Luồng hiện tại đã có đủ data để Stage B cứu checkpoint cũ và tái lưu checkpoint mới, nhưng chưa có bước chuẩn hóa provenance khi `verification_metadata_source` vẫn là `uninitialized`. Điểm vá tối thiểu là sau khi Stage B load checkpoint cũ, nếu tensor metadata đã hợp lệ thì phải rewrite provenance về giá trị chuẩn trước khi save `best.pt` hoặc `final.pt`. Sau đó cần thêm test fail-fast cho contract này.

## Detailed Findings

### Data Preparation
- Không thay đổi dữ liệu.
- Bài toán chỉ nằm ở vòng đời checkpoint và provenance metadata.

### Modeling and Training
- `src/models/thesis_multitask_impl/thesis_multitask_state_serialization_mixin.py` hiện load:
  - `anomalous_codeword_mask`
  - `anomaly_radii`
  - `verification_codeword_class_ids`
  - `verification_contributing_token_counts`
  - `verification_metadata_source`
- Nhưng nếu checkpoint cũ có mask/radii hợp lệ mà source là `uninitialized`, thì code chỉ gán lại source từ `extra_state` hiện có, không tự sửa về `"train_anomaly_tokens_q99"`.
- `src/engine/trainer.py` đã có nhánh refresh `best.pt` sau khi memories được initialized, nên đây là điểm tự nhiên nhất để rewrite provenance trước khi save lại.

### Evaluation
- `src/engine/online_tta/signature_verification.py` là contract chặn online strict.
- `PrototypeVerificationMetadata.from_model()` fail nếu source là `""`, `"uninitialized"`, hoặc `"disabled"`.
- Vì vậy, nếu muốn Stage B rerun dùng checkpoint cũ nhưng vẫn qua online strict sau đó, provenance phải được rewrite trước khi checkpoint mới được lưu.

## Code References
- `src/models/thesis_multitask_impl/thesis_multitask_state_serialization_mixin.py:90-182` - checkpoint extra-state load path
- `src/engine/trainer.py:853-902` - best/final checkpoint re-save path
- `src/engine/online_tta/signature_verification.py:64-87` - online provenance gate
- `src/models/thesis_multitask_impl/thesis_multitask_state_memory_mixin.py:356-363` - canonical Stage A provenance source assignment

## Pipeline Documentation
The practical rewrite flow should be:

1. Load old Stage A checkpoint into Stage B.
2. Validate that mask, radii, codeword class ids, and token counts are present and shape-correct.
3. If those tensors are valid and memory has been initialized, rewrite `verification_metadata_source` to the canonical source.
4. Save `best.pt` and `final.pt` with the rewritten provenance.
5. Run online verification against the rewritten Stage B checkpoint.

## Historical Context
- The remote inventory already shows that the current Stage A smoke checkpoints have tensor metadata but still `verification_metadata_source = uninitialized`.
- The online verifier is intentionally strict and fails closed on that state.

## Open Questions
- Should the rewrite happen in `load_checkpoint_extra_state()` or in `Trainer` right before save.
- Should Stage B rewrite only when loaded tensors are valid, or also emit a warning when repairing provenance.
- Should the rewritten source always be `train_anomaly_tokens_q99`, or derive from a stricter canonical mapping helper.

## Patch Points
1. Add a small helper that normalizes verification provenance when mask/radii are present and memory is initialized.
2. Call that helper in the Stage B save path before `save_checkpoint()` for `best.pt` and `final.pt`.
3. Keep the rewrite gated so invalid or missing tensors still fail hard.
4. Add a test that loads an old checkpoint with `uninitialized` source, rewrites it, saves it, and verifies online signature acceptance.
