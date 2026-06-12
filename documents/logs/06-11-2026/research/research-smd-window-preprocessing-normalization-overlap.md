---
date: 2026-06-11 15:29:04 +07
researcher: Artificial Intelligence Agent
git_commit: 4a4e23939b0b8961fa27919282c1622e44840d66
branch: dev
repository: bachelor-thesis-2026
topic: "SMD preprocessing: normalization order and window overlap"
tags: [research, time-series, preprocessing, windowing]
status: complete
last_updated: 2026-06-11
last_updated_by: Artificial Intelligence Agent
---

# Research: SMD preprocessing: normalization order and window overlap

**Date**: 2026-06-11 15:29:04 +07  
**Researcher**: Artificial Intelligence Agent  
**Git Commit**: 4a4e23939b0b8961fa27919282c1622e44840d66  
**Branch**: dev

## Research Question
Xác định quy trình pre-processing từ time-series đến window trong codebase hiện tại: normalize trước hay sau khi cắt window, và window đang là overlapping hay non-overlapping.

## Summary
Trong pipeline SMD hiện tại, code normalize theo toàn bộ sequence trước, rồi mới cắt thành window. Cụ thể, `SequenceStandardScaler.fit(...)` gom toàn bộ điểm train theo trục thời gian để tính mean/std trên từng channel, sau đó `transform_sequences(...)` áp dụng scale lên từng full sequence. Chỉ sau bước đó, `WindowDataset` và `slice_sequence_into_windows(...)` mới cắt sequence thành window cố định.

Windowing hiện tại là sliding window có stride cấu hình được. Mặc định `stride=10` với `window_size=100`, nên đây là overlapping window. Tuy nhiên, repo cũng có config riêng đặt `stride=100`, nên non-overlapping là một cấu hình khả dụng chứ không phải mặc định toàn cục.

## Detailed Findings

### Data Preparation
- `src/data/datasets/smd.py` đọc raw SMD files thành full sequences theo entity, tách train/val/test theo entity, và gán `point_labels` ở cấp timestep.
- `src/data/loaders.py` thực hiện cleaning, fit scaler trên `cleaned_sequences["train"]`, rồi transform toàn bộ `cleaned_sequences` trước khi window hóa.
- `src/data/scalers.py` tính mean/std từ `torch.cat([sequence["x"] for sequence in sequences], dim=0)`, tức là trên toàn bộ time points của các train sequences đã làm sạch.
- `src/data/window.py` và `src/data/loaders.py` đều dùng `range(0, sequence_length - window_size + 1, stride)` để cắt cửa sổ.
- `src/data/collate.py` ghép các window thành batch `[B, L, D]` cùng metadata dạng list.

### Windowing
- Mặc định ở API là `window_size=100`, `stride=10` trong `src/data/api.py`.
- `configs/data/smd.yaml` và `configs/data/smd_rtx3090_smoke.yaml` cũng dùng `stride: 10`.
- Một số config riêng như `configs/data/smd_rtx3090_machine_2_1_64.yaml` đặt `stride: 100`, nên chạy non-overlapping chỉ là một lựa chọn cấu hình.
- `WindowDataset` lưu các index triple `(sequence_index, start_index, end_index)` và materialize window on-demand.

### Evaluation Link
- `src/engine/evaluator.py` ghi rõ evaluator “merges overlapping window scores back to entity timelines”.
- Khi reconstruct pointwise records, code cộng dồn score theo từng window rồi chia cho số lần phủ của mỗi timestep, tức là overlap được average lại ở cấp timeline.

## Code References
- `src/data/api.py:13-18, 66-70` - default `window_size` và `stride`
- `src/data/loaders.py:64-115, 156-190` - `WindowDataset` và DataLoader
- `src/data/loaders.py:156-161` - fit scaler trước khi window hóa
- `src/data/scalers.py:16-27` - mean/std trên toàn train sequence points
- `src/data/window.py:16-53` - slicing logic theo stride
- `src/data/datasets/smd.py:71-162` - raw SMD parsing và split
- `src/engine/evaluator.py:3-7, 44-128, 199-245` - merge window overlap về timeline
- `configs/data/smd.yaml:1-4` - default overlap config
- `configs/data/smd_rtx3090_machine_2_1_64.yaml:1-8` - non-overlap config example

## Open Questions
- Không có điểm mơ hồ lớn trong pipeline này. Nếu cần, bước tiếp theo có thể rà thêm các experiment config nào đang được dùng thật sự để xác nhận run mặc định trong từng notebook/script.
