---
date: 2026-05-23 15:15:28 +07 +0700
researcher: Artificial Intelligence Agent
git_commit: 57aeba72e81071194e6e271faab39fbc1e955c89
branch: dev
repository: bachelor-thesis-2026
topic: "Audit vòng lặp for dài và đa trách nhiệm theo codebase_preferences"
tags: [research, readability, maintainability, loops, anomaly-detection]
status: complete
last_updated: 2026-05-23
last_updated_by: Artificial Intelligence Agent
---

# Research: Audit vòng lặp for dài và đa trách nhiệm theo codebase_preferences

**Date**: 2026-05-23 15:15:28 +07 +0700  
**Researcher**: Artificial Intelligence Agent  
**Git Commit**: 57aeba72e81071194e6e271faab39fbc1e955c89  
**Branch**: dev

## Research Question
Vòng lặp qua từng epoch trong `src/engine/trainer.py` đang không tuân theo nguyên tắc độ dài/readability/maintainability trong `codebase_preferences.md`. Cần duyệt lại codebase để xác định các vị trí vòng lặp for tương tự, theo phương pháp nghiên cứu của `prompts/1_research_prompt.md`.

## Summary
Kết quả rà soát cho thấy vấn đề chính là các vòng lặp chưa được tách theo trách nhiệm nghiệp vụ (scheduling, forward/backward, diagnostics, metric aggregation, checkpoint policy, logging) trong cùng một khối lặp. Vị trí nặng nhất là vòng `for epoch_index in range(epochs)` trong `Trainer.train`, tiếp theo là một số vòng trong `Trainer._run_validation_epoch`, `OnlineLoop.run`, và khối thu thập token để khởi tạo memory trong `ThesisMultiTaskModel`. Ngoài ra có một số vòng lặp ở mức tiện ích/tính toán thuần (config validation, metric computation) không phải rủi ro chính về khả năng bảo trì.

## Detailed Findings

### Data Preparation
- Vòng lặp inject anomaly theo batch tại `src/data/augment.py` là tuyến dữ liệu chính của multiclass synthetic labels.
- `augment_batch` đang giữ nhiều bước trong một loop mẫu (`for batch_index in range(batch_size)`), gồm quyết định inject, mutate window, cập nhật mask, label, metadata và hậu xử lý log cuối batch.
- Mặc dù số dòng vừa phải, loop này là điểm coupling cao giữa augmentation và logging, đặc biệt khi cộng thêm logic thống kê multiclass.

### Modeling and Training
- `Trainer.train` chứa một vòng epoch dài với nhiều trách nhiệm hợp nhất:
  - điều khiển context epoch và hook khởi tạo memory,
  - vòng train-batch đầy đủ (scheduler batch-level, move device, forward, backward, clip, optimizer step, log),
  - chạy `val` và `val_synth`,
  - tổng hợp nhiều nhóm metric (loss, diagnostics, classification, pointwise, evaluator),
  - policy chọn best checkpoint và refresh checkpoint cuối cùng.
- Điều này trực tiếp va chạm với nguyên tắc: “Do not write overly long for loops... split into small helper functions” trong `codebase_preferences.md`.
- `Trainer._run_validation_epoch` có một vòng batch gom cả xử lý, log, lưu logits/labels history, forward timing và pointwise payload; đây là loop đa trách nhiệm cấp batch.
- `ThesisMultiTaskModel._collect_memory_initialization_token_pool_from_loader` có vòng lấy batch để bootstrap memory, trộn clean-token path và synthetic-token path trong cùng loop.
- `OnlineLoop.run` có vòng online-step gánh cả pre-forward đo score, snapshot tham số, train step, post-forward đo score, đo update norm, ghi record, checkpoint định kỳ.

### Evaluation
- `Evaluator.evaluate` đã tách helper tốt hơn (`_move_batch_to_device`, `_run_model_on_batch`, `_remember_forward_pass_seconds`, `_log_batch_outputs`), nhưng vẫn giữ một loop batch có nhiều side effect (append payload + metrics context).
- `scripts/run_multiseed_experiments.py` có loop điều phối process parallel lồng nhau; đây là loop điều phối hệ thống, không phải loop mô hình, nhưng vẫn là vùng có độ phức tạp điều khiển cao.

## Code References
- `src/engine/trainer.py:556` - vòng epoch chính `for epoch_index in range(epochs)`.
- `src/engine/trainer.py:585` - vòng train batch bên trong epoch.
- `src/engine/trainer.py:678` - khối aggregate metrics lớn trong cùng epoch loop.
- `src/engine/trainer.py:786` - logic monitor/checkpoint nằm trong cùng epoch loop.
- `src/engine/trainer.py:408` - vòng validation batch trong `_run_validation_epoch`.
- `src/models/thesis_multitask.py:1017` - vòng thu thập token bootstrap memory.
- `src/engine/online_loop.py:78` - vòng online adaptation step.
- `src/data/augment.py:752` - vòng inject anomaly theo từng sample trong batch.
- `scripts/run_multiseed_experiments.py:179` - vòng điều phối tiến trình song song.
- `src/engine/evaluator.py:216` - vòng evaluate batch.

## Pipeline Documentation
Trong pipeline hiện tại, engine đóng vai trò điều phối từ train/val đến checkpoint và logging. Tuy nhiên tại `Trainer.train`, nhiều bước nghiệp vụ được thực hiện trực tiếp trong epoch loop thay vì thông qua helper mức “một trách nhiệm”. Điều này làm tăng chi phí đọc top-to-bottom và tăng rủi ro regression khi thay đổi một thành phần (ví dụ diagnostics hoặc checkpoint policy) vì phạm vi chỉnh sửa đụng trực tiếp vòng epoch trung tâm.

## Historical Context (from documents/)
- `prompts/1_research_prompt.md` yêu cầu mô tả hiện trạng code, không nhảy sang đề xuất tối ưu.
- `documents/design/design_starter.md` nhấn mạnh engine nên giữ vai trò loop/checkpoint/logging và model giữ logic đặc thù.
- `codebase_preferences.md` nhấn mạnh readability-first và cấm for-loop quá dài khi ôm nhiều trách nhiệm.

## Open Questions
- Có cần đặt một ngưỡng định lượng nội bộ cho “loop too long” (ví dụ số nhánh điều kiện, số side effects, số khối trách nhiệm) để review nhất quán hơn giữa các file?
- Các comment giải thích sư phạm mong muốn ưu tiên tiếng Anh, tiếng Việt, hay song ngữ cho các loop lõi (`trainer.py`, `online_loop.py`, `thesis_multitask.py`)?
- Phạm vi refactor ưu tiên chỉ `src/engine/trainer.py` trước hay bao gồm cả `online_loop.py` và `augment.py` trong cùng đợt?
