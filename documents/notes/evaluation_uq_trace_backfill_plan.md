# Evaluation UQ Trace Backfill Plan

Date: 2026-07-20

Mục tiêu:
- Giữ lại `uncertainty_history` trong evaluation traces.
- Giữ `uq_summary.json` có số thật cho 2 bảng report.
- Không giữ raw MC sample payload nếu không cần.
- Chỉ rerun evaluation-only, không rerun Stage A hay Stage B.

## 1. Chốt lại contract cần giữ

Giữ lại tối thiểu:
- `point_score_history`
- `window_score_history`
- `uncertainty_history`
- `sample_retention_policy`
- metadata cần cho audit và backfill

Không bắt buộc giữ lâu dài:
- `mc_sample_histories`
- raw `stochastic_query` payload nếu nó chỉ chứa sample tensors nặng
- raw MC sample tensors

Đích cần đạt:
- trace vẫn còn đủ dữ liệu để aggregate variance
- file summary vẫn nhỏ để có thể dọn artifact nặng sau đó

## 2. Sửa code ở evaluation path

Ưu tiên kiểm tra và chỉnh:
- `src/engine/evaluator.py`
- `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py`
- `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py`

Mục tiêu kỹ thuật:
- evaluation vẫn gọi được nhánh MC/uncertainty
- `uncertainty_history` không bị rơi về `None`
- trace không cần persist toàn bộ sample payload
- `sample_retention_policy=none` không được làm mất summary UQ nếu summary đã được tính

Kiểm tra thêm:
- `src/core/uq_summary.py`
- `scripts/ops/backfill_uq_summary.py`

Mục tiêu:
- summary được aggregate từ `uncertainty_history`
- backfill không cần train lại
- backfill có thể chạy từ evaluation-only artifact cũ nếu trace còn đủ dữ liệu

## 3. Viết test khóa contract mới

Cần có test chứng minh:
- trace vẫn chứa `uncertainty_history`
- summary variance không còn `null`
- raw sample payload có thể vắng mặt mà vẫn qua

Nên bổ sung hoặc chỉnh các test liên quan:
- `tests/benchmarks/test_evaluator_trace_payload.py`
- `tests/core/test_uq_summary.py`
- `tests/ops/test_backfill_uq_summary.py`

Tiêu chí pass:
- `uncertainty_history` có field variance thật
- `uq_summary.json` sinh ra các field:
  - `point_anomaly_score_variance_mean`
  - `window_anomaly_score_variance_mean`
  - `classification_probability_variance_mean`
  - `reconstruction_variance_point_mean`
  - `reconstruction_variance_window_mean`
  - `continuous_retrieval_variance_point_mean`
  - `continuous_retrieval_variance_window_mean`
  - `discrete_retrieval_variance_point_mean`
  - `discrete_retrieval_variance_window_mean`

## 4. Rerun evaluation-only rồi dọn artifact

Chỉ chạy lại:
- evaluation-only trên checkpoint Stage B hiện có

Không chạy lại:
- Stage A
- Stage B fine-tune

Sau khi evaluation-only xong:
- backfill lại `uq_summary.json`
- kiểm tra summary có số thật
- kiểm tra report cuối cùng có đủ metric cần báo cáo
- nếu ổn thì xoá raw trace và raw sample payload nặng để giảm dung lượng

Thứ tự an toàn:
1. sửa code
2. chạy test cục bộ
3. rerun evaluation-only
4. backfill summary
5. kiểm tra lại summary/report
6. prune artifact nặng

## Điều kiện hoàn tất

Chỉ xem là xong khi:
- evaluation traces còn `uncertainty_history`
- `uq_summary.json` không còn `null` ở các field variance cần báo cáo
- checkpoint Stage A/B không bị đụng lại
- raw artifact nặng đã có thể xoá an toàn sau khi summary đã được lưu
