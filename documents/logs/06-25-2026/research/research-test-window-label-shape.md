---
date: 2026-06-25 15:08:24 +0700
researcher: TheMetaSetter
git_commit: c66927b06d3b94f3505792cd3aaf66c0fc6b1064
branch: dev
repository: bachelor-thesis-2026
topic: "Trong codebase này, một input window trong test set sẽ được đi kèm với nhãn là một vec-tơ hay một ma trận?"
tags: [research, time-series, anomaly-detection, labels, windowing]
status: complete
last_updated: 2026-06-25
last_updated_by: TheMetaSetter
---

# Research: Trong codebase này, một input window trong test set sẽ được đi kèm với nhãn là một vec-tơ hay một ma trận?

**Date**: 2026-06-25 15:08:24 +0700
**Researcher**: TheMetaSetter
**Git Commit**: c66927b06d3b94f3505792cd3aaf66c0fc6b1064
**Branch**: dev

## Research Question

Trong codebase hiện tại, một test window mang nhãn ở dạng nào tại runtime: vector hay ma trận?

## Summary

Một test window riêng lẻ mang nhãn `point_labels` ở dạng vector 1 chiều với độ dài đúng bằng window size, tức `[L]`. Khi nhiều window được collate thành một batch thì nhãn trở thành ma trận 2 chiều `[B, L]`. Codebase không gắn nhãn test gốc dưới dạng ma trận cho từng window đơn lẻ.

## Detailed Findings

### Data Preparation

- SMD parser nạp `test_labels` cho toàn bộ chuỗi test dưới dạng vector 1 chiều và gán vào `raw_sequence["point_labels"]` của split `test`: `src/data/datasets/smd.py:120-164`.
- Runtime contract cho raw sequence cũng ép `point_labels` phải có rank 1 và cùng chiều dài với chuỗi: `src/core/contracts.py:49-58`.
- Khi cắt một sequence thành một window, `WindowDataset.__getitem__` lấy lát cắt `sequence["point_labels"][start_index:end_index]`, nên nhãn của một window riêng lẻ là vector `[L]`: `src/data/loaders.py:204-244`.

### Batch and Evaluation Semantics

- `validate_window(...)` xác nhận window là một object độc lập, còn `validate_batch(...)` xác nhận `batch["point_labels"]` phải có rank 2: `src/core/contracts.py:73-104`.
- `collate_windows(...)` stack các vector nhãn của từng window theo chiều batch, nên batch label có dạng `[B, L]`: `src/data/collate.py:12-34`.
- `point_labels_to_window_labels(...)` chỉ là phép suy diễn phụ để đổi từ `[B, L]` sang `[B]` bằng quy tắc “window anomalous nếu có ít nhất một anomalous point”; đây không phải nhãn test gốc được lưu sẵn cùng window: `src/data/api.py:179-184`.
- Evaluator cũng giả định payload pointwise có dạng `[B, L]`, sau đó mới merge chồng lấp về lại timeline gốc: `src/engine/evaluator.py:50-123` và `src/engine/evaluator.py:237-268`.

### Modeling and Training

- Ở multitask model, `classification_labels` và `synthetic_anomaly_mask` là các trường được thêm vào batch đã chuẩn bị, không phải nhãn test gốc từ dataset: `src/models/thesis_multitask.py:2220-2270`.
- Baseline cũng giữ cùng semantics: nhãn gốc theo điểm là `point_labels`, còn nhãn phân loại đa lớp là trường riêng được thêm vào ở bước prepare batch: `src/models/redlamp_mlp_baseline.py:320-372`.

## Code References

- `src/data/datasets/smd.py:120-164` - gán `test_labels` vào `point_labels` của test sequence
- `src/core/contracts.py:49-58` - raw sequence yêu cầu `point_labels` là tensor rank 1
- `src/data/loaders.py:204-244` - một window lấy lát cắt `point_labels[start:end]`
- `src/data/collate.py:12-34` - batch hóa thành `point_labels` shape `[B, L]`
- `src/data/api.py:179-184` - suy ra `window_labels` từ `point_labels`
- `src/engine/evaluator.py:50-123` - evaluator yêu cầu pointwise labels `[B, L]`

## Pipeline Documentation

Với test set hiện tại, semantics chuẩn là:

- một test sequence có `point_labels` toàn chuỗi ở dạng vector `[T]`;
- một test window cắt ra từ sequence đó có `point_labels` ở dạng vector `[L]`;
- một test batch gồm nhiều window có `point_labels` ở dạng ma trận `[B, L]`;
- nếu cần nhãn mức window thì code suy ra từ `[B, L]` sang `[B]`, chứ không lưu nhãn đó như ground truth chính.

## Historical Context (from documents/)

`documents/design/design_starter.md` và `documents/design/idea.md` đều mô tả batch contract chuẩn là `point_labels: Optional[Tensor[B, L]]`, tức design intent cũng khớp với runtime implementation hiện tại.

## Open Questions

- Không có mơ hồ lớn cho câu hỏi này. Điểm duy nhất dễ gây nhầm là `classification_labels` trong multitask pipeline không phải ground-truth test label gốc của dataset.
