---
date: 2026-05-21 00:00:00 +07
author: Artificial Intelligence Agent
topic: "Summary so far: Exp2 CKA logging and reconstruction-loss tracing"
status: complete
---

# Summary So Far

## 1) Yêu cầu và triển khai Exp2 CKA logging
- Đã thống nhất phạm vi logging CKA cho `train` và `val_synth`.
- Đã triển khai thêm các metric CKA vào pipeline log:
  - `*_cka_reconstruction_mean`
  - `*_cka_reconstruction_std`
  - `*_cka_classification_mean`
  - `*_cka_classification_std`
- Đã giữ nguyên công thức huấn luyện và monitor/checkpoint policy hiện tại.
- Đã bổ sung fallback rõ ràng khi `enable_cka_gated_fusion=False`: metric CKA vẫn tồn tại và về `0.0`.

## 2) Kiểm thử sau khi triển khai
- Đã cập nhật test integration cho train-step Exp2 để assert đầy đủ 4 metric CKA.
- Đã thêm test cho `synthetic_validation_step` để assert metric CKA cho `val_synth`.
- Đã thêm test fallback khi tắt CKA gate.
- Kết quả test gần nhất cho nhóm liên quan: pass.

## 3) Làm rõ “4 CKA metrics”
- Có 2 đại lượng CKA nội bộ:
  - CKA giữa `h` và `\hat{h_c}` (nhánh reconstruction-side signal)
  - CKA giữa `h'` và `\hat{h'_d}` (nhánh classification-side signal)
- Mỗi đại lượng log 2 thống kê (`mean`, `std`) theo batch -> tổng 4 metric.

## 4) Script visualize training metrics
- Đã thêm script:
  - `scripts/visualize_training_metrics.py`
- Script đọc `metrics.jsonl` và vẽ timeline theo epoch, gồm cả các metric CKA Exp2 (mặc định).

## 5) Truy vết hiện tượng loss dao động (chưa đề xuất giải pháp)
Bạn yêu cầu chỉ debug/tracing, chưa cần solution. Nội dung truy vết đã khoanh vùng:
- Đường đi tạo batch train và synthetic augmentation.
- Nhánh two-view trong train step.
- Forward path tạo `recon` và điểm lỗi theo timestep.
- Công thức `reconstruction_loss` (bao gồm nhánh `reconstruction_normal_only` + `synthetic_anomaly_mask`).
- Điểm ghi `train_reconstruction_loss` vào stage log.
- Điểm trainer aggregate theo epoch (trung bình qua train batches).
- Nơi `synthetic_anomaly_mask` được sinh trong injector.

## 6) Các vị trí code mấu chốt đã chỉ ra
- `src/models/thesis_multitask.py`
  - chuẩn bị batch / augment
  - `_compute_reconstruction_loss(...)`
  - `_build_normal_time_step_mask(...)`
  - `_build_stage_log(...)`
  - `_shared_step(...)`
- `src/engine/trainer.py`
  - vòng lặp train batch và aggregate epoch metrics
- `src/data/augment.py`
  - sinh `synthetic_anomaly_mask`

## 7) Trạng thái hiện tại
- Bạn đã xác nhận mục tiêu là tiếp tục truy vết nguyên nhân dao động mạnh của loss, đặc biệt là reconstruction loss.
- Chưa đi vào đề xuất fix theo đúng yêu cầu hiện tại.
