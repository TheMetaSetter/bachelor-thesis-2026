# Checklist hôm nay - SMD 2-1, 300 epochs

Ngày: 06-08-2026

Mục tiêu: khóa thiết kế low-level và chuẩn bị đường chạy cho 3 thí nghiệm trên SMD machine `2-1`.

## Checklist

- [ ] Chạy lại baseline RedLamp với `simple CNN encoder` và `train_balance_classes=true`, để batch train cân bằng theo `normal + 11 anomaly classes`.
- [ ] Thiết kế và triển khai thí nghiệm 2 với `simple CNN backbone`, giữ nguyên contrastive design cũ trong `documents/design/offline_pretraining_phase_two_view_contrastive_design.md`.
- [ ] Với thí nghiệm 2, ép `DMTRL-LAF` dùng `K = 2` vì chỉ có 2 task chính: reconstruction và classification.
- [ ] Log `SVD energy retention`, `SVD residual`, và `factorization drift` cho từng conv layer khi khởi tạo và trong quá trình train.
- [ ] Thiết kế và triển khai `soft hinge consensus loss` giữa reconstruction head và classification head.
- [ ] Chạy pilot ngắn để đo scale của `per-window MSE` và gap `Delta = mu_A - mu_N`, rồi mới chốt margin `m`.
- [ ] Thiết kế và triển khai thí nghiệm 3: `simple CNN encoder + discrete prototypes + continuous prototypes + CKA-gated fusion head`.
- [ ] Lập trình flow pretrain cho thí nghiệm 3 theo 2 phase single-task, rồi một phase multi-task, trước khi gắn prototype branches.
- [ ] Chốt lại tất cả config knobs, tensor shapes, và logging metrics để có thể chạy reproducibly.

## Ghi chú cần làm rõ

- Contrastive option của thí nghiệm 2 giữ nguyên design cũ, không đổi semantics positive/negative.
- `m` của consensus loss chưa chốt cố định, cần dựa trên pilot statistics.
- `SVD energy loss` với `K = 2` chủ yếu là sanity check; metric hữu ích hơn là `factorization drift`.
