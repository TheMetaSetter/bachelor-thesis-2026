---
date: 2026-06-10 13:20:16 +07
researcher: OpenAI Codex
git_commit: 1a2825cbfe400a9f3ce280d83f2da8f26c39daba
branch: dev
repository: bachelor-thesis-2026
topic: "Dòng code cần sửa để visualization phản ánh đúng slice nửa mở [start_index, end_index)"
tags: [research, time-series, anomaly-detection, visualization, python-slice-semantics]
status: complete
last_updated: 2026-06-10
last_updated_by: OpenAI Codex
---

# Research: Dòng code cần sửa để visualization phản ánh đúng slice nửa mở [start_index, end_index)

**Date**: 2026-06-10 13:20:16 +07
**Researcher**: OpenAI Codex
**Git Commit**: 1a2825cbfe400a9f3ce280d83f2da8f26c39daba
**Branch**: dev

## Research Question

Cần sửa ở những dòng code nào để visualization phản ánh đúng semantics Python slice nửa mở của injection, tức là anomaly nằm trên interval `[start_index, end_index)`, chứ không phải hiểu nhầm `end_index` là một điểm được bao gồm.

## Summary

Code injector hiện tại là đúng. Slice trong Python được áp dụng theo nửa mở: `start_index` được tính, còn `end_index` là exclusive. Phần cần sửa nằm chủ yếu ở visualization helper, nơi vùng tô đỏ và chú thích hiện tại làm người đọc dễ hiểu nhầm rằng `end_index` cũng thuộc anomaly region.

## Detailed Findings

### Data Preparation

Không cần sửa ở data loader hay injector cho vấn đề này. `WindowDataset` đã trả về window đúng contract, `SyntheticAnomalyInjector` đã ghi `start_index` và `end_index` đúng theo semantics nửa mở.

### Modeling and Training

Không cần sửa model logic. Model chỉ tiêu thụ `batch["x"]` sau khi injector đã sửa xong.

### Evaluation

Chỗ cần sửa nằm ở visualization, cụ thể là file:

- `scripts/visualize_synthetic_anomalies.py`

Các dòng liên quan trực tiếp:

- `scripts/visualize_synthetic_anomalies.py:92-109`
- `scripts/visualize_synthetic_anomalies.py:106-109`

`axvspan(metadata["start_index"], metadata["end_index"], ...)` đang tô vùng với cách hiểu trực quan không khớp hoàn toàn với interval nửa mở của slice Python. Nếu mục tiêu là phản ánh đúng semantics nửa mở, nên sửa chú thích và cách highlight để làm rõ rằng `end_index` không thuộc vùng bị tiêm.

### Code References

- `src/data/augment.py:266-274` - slice nửa mở khi gán `updated_segment` vào `anomalous_channel_window`
- `src/data/augment.py:322-343` - định nghĩa `flip`
- `scripts/visualize_synthetic_anomalies.py:92-109` - title và vùng tô đỏ của subplot, đây là nơi cần sửa
- `tests/test_synthetic_anomaly_visualization.py` - có thể cần bổ sung test nếu đổi rule hiển thị

## Pipeline Documentation

Semantics đúng trong code hiện tại là:

```python
anomalous_channel_window[start_index:end_index, 0] = updated_segment
channel_mask[start_index:end_index] = local_mask.long()
```

Điều đó có nghĩa:

- phần tử tại `start_index` được tính vào anomaly
- phần tử tại `end_index` không được tính vào anomaly

Do đó, phần trực quan nên ghi rõ vùng là `[start_index, end_index)` hoặc vẽ marker riêng cho `end_index - 1` thay vì khiến người đọc hiểu nhầm `end_index` là inclusive.

## Historical Context (from documents/)

Các design notes dùng tên `start_index` và `end_index` để mô tả đoạn contiguous subsequence. Nhưng code injector là nguồn sự thật cuối cùng, và ở đây nó đang dùng slice Python nửa mở.

## Open Questions

Nếu muốn sửa visualization cho đúng tuyệt đối với semantics nửa mở, cần chọn một trong hai cách:

1. giữ `axvspan` nhưng đổi nhãn sang `[start_index, end_index)` để tránh hiểu nhầm
2. đổi cách tô vùng để marker trực quan khớp hơn với `end_index - 1`
