# Checklist vá lỗi: metadata checkpoint và UQ persistence

## 0. Chốt phạm vi trước khi vá
- [ ] Xác nhận mục tiêu là checkpoint metadata phải không rỗng và có nghĩa.
- [ ] Xác nhận mục tiêu là UQ fields phải được sinh ra, lưu lại, và kiểm được.
- [ ] Xác nhận mục tiêu là artifact xuất ra phải có provenance đầy đủ.
- [ ] Giữ đúng SSOT ở `documents/spec/full-spec-v3.md`.
- [ ] Giữ đúng inventory ở `documents/inventories/detail-thesis-uq-field-inventory.md`.

## 1. Vá lớp config validation
- [ ] Siết `src/core/config_model_validation.py` để các field UQ không thể đi vào runtime ở trạng thái vô nghĩa.
- [ ] Bảo đảm các field sau không rỗng và không vô hiệu:
  - `monte_carlo_samples`
  - `continuous_temperature`
  - `discrete_temperature`
  - `variance_correction`
  - `return_mc_samples`
  - `sample_retention_policy`
- [ ] Thêm test cho config lỗi:
  - `monte_carlo_samples < 1`
  - `sample_retention_policy = none` nhưng `return_mc_samples = true`
  - nhiệt độ không dương

## 2. Thêm runtime semantic validation
- [ ] Tách validator riêng cho:
  - `stochastic_query`
  - `uncertainty`
  - `threshold_artifact`
  - `checkpoint metadata`
- [ ] Trong `src/core/contracts.py`, không chỉ kiểm schema và shape mà còn kiểm:
  - `stochastic_query is not None`
  - `stochastic_query["num_samples"] == model.monte_carlo_samples`
  - sample tensors có đúng batch/sample rank
  - variance tensors finite
- [ ] Nếu `return_mc_samples = true`, bắt buộc artifact runtime phải giữ sample payload thật.
- [ ] Nếu `sample_retention_policy != retain_for_eda`, thì không được giả vờ là có trace đầy đủ.

## 3. Vá checkpoint serialization
- [ ] Trong `src/models/thesis_multitask_impl/thesis_multitask_state_serialization_mixin.py`, giữ normalize provenance khi save/load.
- [ ] Không cho phép `verification_metadata_source` còn là `""`, `"uninitialized"`, hoặc `"disabled"` khi `memory_initialized = True`.
- [ ] Bổ sung check semantic cho checkpoint metadata:
  - `anomalous_codeword_mask` phải phản ánh dữ liệu thật
  - `anomaly_radii` phải finite
  - `verification_codeword_class_ids` phải khớp shape
  - `verification_contributing_token_counts` phải khớp shape
- [ ] Nếu checkpoint chỉ có tensor shape đúng nhưng nội dung rỗng hoặc all-zero không hợp contract, cho fail closed hoặc buộc rerun Stage A.

## 4. Vá lưu trace/artifact
- [ ] Trong `src/engine/evaluator.py`, persist trace payload thật chứ không chỉ summary.
- [ ] Lưu riêng:
  - `stochastic_query`
  - `uncertainty`
  - `mc_sample_histories`
  - `sample_retention_policy`
- [ ] Nếu chưa muốn nhúng vào checkpoint, thì export ra file trace riêng trong Stage B và online output tree.
- [ ] Thêm manifest checksum cho trace artifact như đã làm với checkpoint và threshold artifact.

## 5. Vá threshold provenance
- [ ] Trong `src/protocols/threshold_artifact.py`, giữ validate `checkpoint_sha256`.
- [ ] Trong online path, điền thật `threshold_artifact.checkpoint_sha256`, không để `None`.
- [ ] Bắt buộc `threshold_artifact.provenance` có:
  - `checkpoint_sha256`
  - `resolved_config_sha256`
  - `created_by`
  - `config_path`
- [ ] Nếu không thể chứng minh provenance, fail trước khi chạy online dài hơi.

## 6. Thêm test bắt lỗi tinh vi
- [ ] Test checkpoint roundtrip:
  - save/load vẫn giữ provenance
  - placeholder provenance bị rewrite đúng
- [ ] Test UQ contract:
  - `stochastic_query` và `uncertainty` đủ field
  - shape và rank đúng
  - tensors finite
- [ ] Test artifact provenance:
  - manifest checksum khớp
  - threshold artifact có `checkpoint_sha256`
- [ ] Test runtime smoke trên đúng 1 tổ hợp đại diện trước khi scale batch.

## 7. Xác minh trên remote GPU
- [ ] Chạy lại 1 tổ hợp đại diện thật trên remote.
- [ ] Kiểm tra:
  - Stage A checkpoint provenance
  - Stage B checkpoint provenance
  - trace artifact có thật sự chứa UQ payload
  - online threshold artifact có checkpoint hash
- [ ] Chỉ khi pass hết mới chạy toàn bộ combination.

## Ưu tiên thực hiện
- [ ] Ưu tiên 1: checkpoint provenance
- [ ] Ưu tiên 2: UQ runtime validation
- [ ] Ưu tiên 3: trace export
- [ ] Ưu tiên 4: threshold artifact provenance
- [ ] Ưu tiên 5: test và smoke remote
