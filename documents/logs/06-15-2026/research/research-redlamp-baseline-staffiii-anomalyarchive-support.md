---
date: 2026-06-15 12:20:39 +07
researcher: TheMetaSetter
git_commit: 866c225ed69e4c618718551558077669090411e4
branch: dev
repository: bachelor-thesis-2026
topic: "Can the current codebase run a RedLamp baseline experiment on 219_UCR_Anomaly_STAFFIIIDatabase_41612_64632_64852.txt?"
tags: [research, time-series, anomaly-detection, multi-class]
status: complete
last_updated: 2026-06-15
last_updated_by: TheMetaSetter
---

# Research: Can the current codebase run a RedLamp baseline experiment on 219_UCR_Anomaly_STAFFIIIDatabase_41612_64632_64852.txt?

**Date**: 2026-06-15 12:20:39 +07
**Researcher**: TheMetaSetter
**Git Commit**: 866c225ed69e4c618718551558077669090411e4
**Branch**: dev

## Research Question
Sử dụng `1_research_prompt.md` để kiểm tra xem tình hình codebase hiện tại có chạy được thí nghiệm baseline RedLamp hay không, với chuỗi mục tiêu là `219_UCR_Anomaly_STAFFIIIDatabase_41612_64632_64852.txt`.

## Summary
Kết luận hiện tại là: codebase có thể chạy baseline RedLamp theo đường chuẩn hiện có, nhưng chỉ đối với dataset `smd`. Với chuỗi `219_UCR_Anomaly_STAFFIIIDatabase_41612_64632_64852.txt` thuộc `data/AnomalyArchive`, repo hiện tại chưa có data loader hay config path trực tiếp để chạy baseline ngay. Bản chất blocker nằm ở chỗ runtime và validator vẫn khóa `dataset_name` vào `smd`, còn `scripts/train.py` chỉ đăng ký loader `smd`.

## Detailed Findings

### Data Preparation
- `src/data/loaders.py` hiện là builder SMD duy nhất. Nó parse SMD, clean, fit scaler trên train, rồi window hóa với `WindowDataset` dùng slice nửa mở `[start_index:end_index)` và stride cấu hình.
- `src/core/config.py` chỉ chấp nhận `dataset_name` nằm trong `{"smd"}`. Vì vậy bất kỳ experiment config nào đổi sang một dataset khác sẽ bị fail fast.
- `scripts/train.py` chỉ đăng ký `register_dataset("smd", build_smd_dataset_bundle)` và không có registry entry cho `AnomalyArchive` hoặc custom single-file loader.
- `configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml` trỏ về `configs/data/smd_rtx3090_machine_2_1_20.yaml`, tức baseline hiện hữu được cấu hình cho SMD chứ không phải `AnomalyArchive`.

### Modeling and Training
- `scripts/train.py` đăng ký `redlamp_mlp_baseline` và tạo model từ config qua registry.
- `src/models/redlamp_mlp_baseline.py` là baseline model hiện tại; nó nhận batch chuẩn `[B, L, D]` và dùng `classification_label_mode="redlamp_multiclass"`.
- Baseline training hiện được thiết kế cho synthetic anomaly augmentation cùng taxonomy RedLamp, không phải cho raw single-file STAFFIII evaluation trực tiếp.

### Evaluation
- Eval path trong repo hiện có thể chạy trên các dataset/batch contract do SMD builder cung cấp.
- Không thấy loader/evaluator sẵn có cho `AnomalyArchive` raw file trước khi đưa vào trainer.

## Code References
- `src/core/config.py:171-181` - dataset validator chỉ cho phép `smd`
- `src/data/loaders.py:118-226` - SMD dataset builder và window dataset
- `src/data/loaders.py:64-115` - window slice contract `[start_index:end_index)`
- `scripts/train.py:41-48` - registry chỉ đăng ký `smd`
- `scripts/train.py:51-78` - model build path cho RedLamp baseline
- `configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml:1-22` - baseline config hiện tại

## Pipeline Documentation
Pipeline hiện tại cho baseline RedLamp là:

1. load experiment config
2. validate `dataset_name == smd`
3. register SMD dataset builder
4. build SMD sequences
5. clean, scale, windowize với `window_size=20`
6. build `redlamp_mlp_baseline`
7. train/evaluate qua trainer and evaluator hiện có

Đường chạy này chưa có nhánh cho `data/AnomalyArchive`, nên chuỗi `219_UCR_Anomaly_STAFFIIIDatabase_41612_64632_64852.txt` chưa thể dùng trực tiếp trong baseline hiện tại.

## Historical Context (from documents/)
- `documents/design/idea.md` và `documents/design/design_starter.md` xác nhận active thesis experiments đang dùng window length `20` và batch contract `[B, L, D]`.
- Các nghiên cứu trước đó trong `documents/logs/` cũng cho thấy repo đang ở trạng thái SMD-first, còn custom dataset support phải được thêm rõ ràng qua loader/config.

## Open Questions
- Nếu muốn chạy baseline trên `219...STAFFIIIDatabase...`, cần thêm loader riêng cho `AnomalyArchive` hay tạm thời xuất file này sang một adapter theo batch contract hiện tại.
- Chưa có config chuẩn sẵn để map file STAFFIII đơn lẻ vào train/val/test splits theo cách nhất quán với trainer hiện tại.

