---
date: 2026-05-21 17:54:14 +0700
researcher: TheMetaSetter
git_commit: a1bd17b08a633a91b4871f803e441dccfb49990a
branch: dev
repository: bachelor-thesis-2026
topic: "Duyệt codebase hiện tại cho thí nghiệm alternating-freeze giữa alpha/beta và phần còn lại trước khi lập kế hoạch"
tags: [research, time-series, anomaly-detection, multi-class]
status: complete
last_updated: 2026-05-21
last_updated_by: TheMetaSetter
---

# Research: Duyệt codebase hiện tại cho thí nghiệm alternating-freeze giữa alpha/beta và phần còn lại trước khi lập kế hoạch

**Date**: 2026-05-21 17:54:14 +0700  
**Researcher**: TheMetaSetter  
**Git Commit**: a1bd17b08a633a91b4871f803e441dccfb49990a  
**Branch**: dev

## Research Question
Thế thì tạo riêng một config khác cho thí nghiệm này nhé. Nhớ đặt tên để phân biệt. Ghi chú các training config quan trọng ngay trong tên để tôi nhận ra ngay như learning rate, learning rate scheduler, v.v. hoặc là để trong tag, các tag đầu tiên.

Duyệt codebase để xem tình hình trước khi lên bất kì kế hoạch nào. Sử dụng `prompts/1_research_prompt.md`.

## Summary
Codebase hiện tại đã có cơ chế warm-up cho fusion thông qua `freeze_fusion_for_epochs` cùng hai giá trị override `warmup_alpha_value` và `warmup_beta_value`. Tuy nhiên, cơ chế này chỉ cố định giá trị trộn của `alpha` và `beta` trong giai đoạn warm-up ban đầu, sau đó trả lại chế độ học bình thường, và không có cơ chế luân phiên hai pha huấn luyện giữa việc freeze `alpha/beta` và freeze phần tham số còn lại theo chu kỳ.

Do đó, ở thời điểm nghiên cứu này, một file config mới chỉ có thể cấu hình các tham số hiện có của lịch warm-up fusion, optimizer, scheduler, checkpoint monitor, và logging. Nó chưa thể kích hoạt một alternating-freeze scheduler thực sự nếu không bổ sung logic mới trong model hoặc trainer.

## Detailed Findings

### Data Preparation
- Dữ liệu vẫn đi theo pipeline SMD hiện hành và không có thay đổi liên quan đến alternating-freeze.
- Cấu hình task multiclass window20 đang dùng:
  - `freeze_fusion_for_epochs: 0`
  - `warmup_alpha_value: 0.5`
  - `warmup_beta_value: 0.5`
  (`configs/task/multitask_tsad_redlamp_multiclass_window20.yaml`)

### Modeling and Training
- `ThesisMultitaskModel` có các trường schedule cho warm-up fusion:
  - `freeze_fusion_for_epochs`, `warmup_alpha_value`, `warmup_beta_value`.
- Hàm `set_epoch_context` chỉ làm:
  1. xác định `warmup_active = epoch_index < freeze_fusion_for_epochs`,
  2. gán `active_alpha_override` và `active_beta_override` khi warm-up đang bật,
  3. sau warm-up đặt override về `None` để quay lại học từ `alpha_logit`, `beta_logit`.
- Trong `_compute_fusion_outputs`, khi override là `None`, mô hình dùng `torch.sigmoid(self.alpha_logit)` và `torch.sigmoid(self.beta_logit)` như bình thường.
- Không thấy cơ chế `requires_grad` theo nhóm tham số để freeze-unfreeze luân phiên giữa các nhóm tham số.
- Optimizer hiện khởi tạo theo toàn bộ `model.parameters()` trong `scripts/train.py`; không thấy param-group policy riêng để thực thi alternating-freeze giữa nhóm fusion-logit và nhóm còn lại.

### Evaluation
- Luồng đánh giá không có logic đặc thù alternating-freeze; chỉ đánh giá checkpoint theo experiment config và checkpoint path.
- Monitor metric checkpoint trong nhánh thesis multitask vẫn dùng `val_synth_vus_pr` theo policy hiện hành.

## Code References
- `configs/task/multitask_tsad_redlamp_multiclass_window20.yaml:6` - `freeze_fusion_for_epochs`.
- `configs/task/multitask_tsad_redlamp_multiclass_window20.yaml:7` - `warmup_alpha_value`.
- `configs/task/multitask_tsad_redlamp_multiclass_window20.yaml:8` - `warmup_beta_value`.
- `src/models/thesis_multitask.py:759` - `set_epoch_context` điều khiển warm-up override.
- `src/models/thesis_multitask.py:770` - điều kiện `warmup_active`.
- `src/models/thesis_multitask.py:771` - gán `active_alpha_override`.
- `src/models/thesis_multitask.py:772` - gán `active_beta_override`.
- `src/models/thesis_multitask.py:1335` - dùng `alpha_logit` khi không override.
- `src/models/thesis_multitask.py:1339` - dùng `beta_logit` khi không override.
- `scripts/train.py:84` - đọc optimizer config.
- `scripts/train.py:87` - map learning rate từ config.
- `scripts/train.py:91` - optimizer trên `model.parameters()`.
- `src/engine/trainer.py:496` - trainer gọi `set_epoch_context` mỗi epoch.

## Pipeline Documentation
1. Train script nạp experiment config và build model theo registry.
2. `Trainer` gọi `set_epoch_context` đầu mỗi epoch.
3. Model áp warm-up override cho `alpha/beta` nếu epoch còn nằm trong `freeze_fusion_for_epochs`.
4. Sau warm-up, fusion quay về scalar học được từ `alpha_logit` và `beta_logit`.
5. Không có pha alternating-freeze giữa nhóm fusion và nhóm tham số còn lại trong runtime hiện tại.

## Historical Context (from documents/)
- `documents/design/idea.md` và `documents/design/design_starter.md` mô tả hợp đồng hidden representation, fusion theo `alpha`/`beta`, và objective modular cho offline multitask.
- Hai tài liệu này không tự động đồng nghĩa với việc alternating-freeze đã được lập trình; hiện trạng triển khai runtime vẫn theo warm-up override một chiều như đã nêu.

## Open Questions
- Nếu thí nghiệm alternating-freeze yêu cầu freeze luân phiên theo chu kỳ epoch hoặc theo step, cần xác định chính xác policy chuyển pha (chu kỳ, tiêu chí chuyển pha, và nhóm tham số cụ thể).
- Cần xác định alternating-freeze sẽ được điều khiển từ config field nào để giữ nguyên nguyên tắc config-driven và dễ ablation.
