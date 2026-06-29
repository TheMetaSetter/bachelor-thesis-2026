---
date: 2026-06-29 15:55:30 +07
researcher: TheMetaSetter
git_commit: ad1fce42fc447fcf8a2180153b0c8590edeec541
branch: dev
repository: bachelor-thesis-2026
topic: "Current runtime order of synthetic anomaly injection versus windowization"
tags: [research, time-series, anomaly-detection, synthetic-anomaly, windowization]
status: complete
last_updated: 2026-06-29
last_updated_by: TheMetaSetter
---

# Research: Current runtime order of synthetic anomaly injection versus windowization

**Date**: 2026-06-29 15:55:30 +07  
**Researcher**: TheMetaSetter  
**Git Commit**: `ad1fce42fc447fcf8a2180153b0c8590edeec541`  
**Branch**: `dev`

## Research Question

Trong codebase hiện tại, synthetic anomaly được tiêm vào chuỗi train trước rồi mới cắt thành window, hay là cắt thành window trước rồi mới tiêm? Nói ngắn gọn là:

- `tiêm trước rồi cắt sau`
- hay `cắt trước rồi tiêm sau`

Ngoài ra, câu hỏi này còn quan tâm tới việc liệu implementation hiện tại có đang giới hạn độ dài anomaly span trong một window hay không.

## Summary

Codebase hiện tại đang chạy theo thứ tự:

1. parse chuỗi gốc thành các raw sequence theo split
2. clean sequence
3. fit scaler trên train sequence
4. transform sequence
5. cắt sequence thành fixed-length windows
6. collate windows thành batch có dạng `[B, L, D]`
7. chỉ tới lúc này mới tiêm synthetic anomaly vào từng window trong batch

Vì vậy, **implementation hiện tại là `cắt trước rồi tiêm sau`**, không phải `tiêm trước rồi cắt sau`.

Synthetic anomaly hiện tại không được tạo trên full train series rồi mới mang đi windowize. Thay vào đó, nó được tạo **online ở runtime**, ngay trước forward/training step, trên từng mini-batch window đã được cắt sẵn.

Điểm này dẫn tới ba hệ quả kỹ thuật rất rõ:

1. độ dài synthetic anomaly span bị ràng buộc bởi `window_size`
2. vị trí và hình dạng anomaly hiện tại chỉ được quyết định từ ngữ cảnh nằm bên trong window đang xét
3. các window chồng lắp lên nhau không được buộc phải chia sẻ cùng một anomaly span nhất quán trên cùng một timeline gốc

## Detailed Findings

### Data Preparation

#### 1. Loader path làm việc trên full sequences trước

`src/data/loaders.py` cho thấy pipeline dữ liệu chính bắt đầu từ `cleaned_sequences`, sau đó fit scaler trên train split, rồi transform tất cả split, rồi mới gọi hàm build window dataset:

- `scaler.fit(cleaned_sequences["train"])` tại `src/data/loaders.py:169-170`
- `scaled_sequences = { ... scaler.transform_sequences(...) ... }` tại `src/data/loaders.py:171-174`
- `_build_window_datasets(...)` tại `src/data/loaders.py:176-179`

Điều này cho thấy ở tầng loader, dữ liệu vẫn đang là **sequence-level object** cho tới trước bước windowization.

#### 2. WindowDataset mới là nơi cắt sequence thành windows

`WindowDataset` trong `src/data/loaders.py:196-263` là nơi thực sự cắt chuỗi thành cửa sổ.

Các dòng quan trọng:

- duyệt từng sequence: `src/data/loaders.py:210-218`
- cắt theo `start_index`, `end_index`: `src/data/loaders.py:214-216`
- trong `__getitem__`, trả ra:
  - `x = sequence["x"][start_index:end_index]`
  - `point_labels = sequence["point_labels"][start_index:end_index]`
  - metadata chứa `start_index`, `end_index`, `absolute_start_index`, `absolute_end_index`

Nói ngắn gọn, đến đây dữ liệu đã bị chuyển từ **chuỗi dài** sang **window độc lập**.

#### 3. DataLoader chỉ batch các windows đã cắt sẵn

`_build_loaders_from_datasets` trong `src/data/loaders.py:105-147` dùng `collate_windows` cho train, val, test.

Trong `src/data/collate.py:11-36`, `collate_windows` stack các `window["x"]` thành:

```python
batch["x"] = torch.stack([...], dim=0)
```

và `validate_batch(batch)` ở `src/data/collate.py:35` sau đó kiểm tra batch contract.

`validate_batch` trong `src/core/contracts.py:95-103` yêu cầu:

- `batch["x"]` phải có rank 3
- nghĩa là batch có dạng `[B, L, D]`

Đây là bằng chứng rất quan trọng: model không nhận raw full sequence, mà nhận **batch of windows**.

### Modeling and Training

#### 4. Trainer đưa batch window vào model training_step

`src/engine/trainer.py:621-637` cho thấy trainer duyệt `train_loader`, move batch lên device, rồi gọi:

```python
step_output = self.model.training_step(batch_on_device)
```

`train_loader` ở đây đã là loader của các windows. Nói cách khác, synthetic injection nếu có thì phải xảy ra **sau bước windowization**, trừ khi loader đã inject trước. Nhưng loader không hề gọi injector.

### Synthetic Augmentation

#### 5. RedLamp baseline inject trên batch window, không inject trên raw sequence

Trong `src/models/redlamp_mlp_baseline.py:308-321`:

- nếu `stage_name == "train"` và bật synthetic augmentation, model gọi:

```python
self.synthetic_anomaly_injector.augment_batch(batch)
```

Sau đó `training_step` đi qua `_shared_step`, và `_shared_step` gọi `_prepare_batch` trước khi forward:

- `_shared_step(...)` tại `src/models/redlamp_mlp_baseline.py:623-684`
- `training_step(...)` tại `src/models/redlamp_mlp_baseline.py:686-687`

Thứ tự runtime ở baseline là:

1. nhận `batch` từ `train_loader`
2. `batch` này đã là batch window `[B, L, D]`
3. `_prepare_batch` gọi injector
4. injector sửa window trong batch
5. model forward trên batch đã bị tiêm anomaly

#### 6. Thesis multitask model cũng inject trên batch window

Trong `src/models/thesis_multitask.py`, logic tương tự cũng xuất hiện.

Ở `src/models/thesis_multitask.py:2239-2240`, nếu đang ở stage train và bật synthetic augmentation, `_prepare_clean_batch` sẽ gọi:

```python
self.synthetic_anomaly_injector.augment_batch(batch)
```

Ở `src/models/thesis_multitask.py:2284-2290`, `_prepare_batch` dùng `synthetic_validation_injector` cho `val_synth` và `val_realistic`.

Ở `src/models/thesis_multitask.py:3145-3166`, `_shared_step` ghi rất rõ bằng comment rằng chuẩn bị batch là nơi “tiêm bất thường nhân tạo vào nếu cần”, rồi mới forward.

`training_step(...)` chỉ đơn giản gọi `_shared_step(...)`:

- `src/models/thesis_multitask.py:3249-3255`

Vì vậy, cả hai model chính trong codebase đều thống nhất cùng một cơ chế:

- **window được cắt trước**
- **synthetic anomaly được tiêm sau**

### Exact Injection Semantics

#### 7. Injector chỉ chấp nhận batch `[B, L, D]`

Trong `src/data/augment.py:865-872`, `augment_batch` kiểm tra:

```python
if batch["x"].ndim != 3:
    raise ValueError("batch['x'] must have shape [B, L, D]")
```

Đây là bằng chứng trực tiếp nhất cho thấy injector đang được thiết kế để làm việc trên **window batch**, không phải trên full sequence.

#### 8. Injector lấy `window_size` từ chính batch window

Trong `src/data/augment.py:873-876`:

```python
clean_windows = batch["x"]
batch_size, window_size, _ = clean_windows.shape
```

Tức là injector không biết gì về phần chuỗi nằm ngoài window. Toàn bộ quyết định của nó được neo vào `window_size` hiện tại.

#### 9. Độ dài anomaly span được lấy từ `window_size`

Trong `src/data/augment.py:197-217`, `_sample_segment_bounds(window_size, ...)` tính:

- `min_segment_length = int(window_size * self.min_segment_fraction)`
- `max_segment_length = int(window_size * self.max_segment_fraction)`

Điều này có nghĩa là **độ dài synthetic anomaly segment hiện tại bị ràng buộc trực tiếp bởi chiều dài của window**.

Nếu window dài `L = 20`, thì segment được lấy theo một phần của 20 timestep đó, chứ không thể tự nhiên kéo dài ra ngoài window.

#### 10. Injector sửa từng window độc lập

Trong `src/data/augment.py:889-915`, injector duyệt:

```python
for batch_index in range(batch_size):
    ...
    augmented_window, anomaly_mask, window_metadata = self._inject_single_window(...)
```

Tức là từng window trong batch được xử lý riêng.

Không có đoạn code nào ở đây:

- nối nhiều window kề nhau lại thành một chuỗi lớn
- duy trì một anomaly span chung băng qua nhiều window
- ép hai window overlap phải có cùng synthetic anomaly trong phần chồng lắp

Do đó, từ implementation hiện tại có thể kết luận rằng các window được augment **độc lập theo từng window**.

#### 11. Synthetic label cũng được cập nhật ở mức window hiện tại

Sau khi inject, `augment_batch` cập nhật:

- `augmented_batch["point_labels"]`
- `augmented_batch["classification_labels"]`
- `augmented_batch["synthetic_anomaly_mask"]`
- `augmented_batch["augmentation_metadata"]`

Các dòng chính là `src/data/augment.py:917-930`.

Đáng chú ý là:

```python
augmented_batch["point_labels"] = torch.maximum(
    original_point_labels.clone(), anomaly_masks
)
```

ở `src/data/augment.py:920-923`.

Nghĩa là synthetic anomaly mask chỉ được gộp với **point label của chính window hiện tại**, không phải point label của full sequence trước khi cắt.

## Practical Answer

### Câu trả lời ngắn gọn

**Hiện tại codebase đang làm theo kiểu `cắt trước rồi tiêm sau`.**

Không phải:

1. lấy full train sequence
2. tiêm synthetic anomaly lên full sequence
3. rồi mới cắt sequence đã bị tiêm thành các windows

Mà là:

1. lấy full train sequence
2. clean và scale
3. cắt thành windows
4. tạo batch `[B, L, D]`
5. ngay trước bước forward/training_step mới tiêm synthetic anomaly vào từng window trong batch

### Diễn giải rất cụ thể

Nếu có hai window overlap nhau trên cùng một đoạn timeline gốc, code hiện tại **không** đảm bảo rằng phần overlap đó sẽ chứa cùng một synthetic anomaly.

Lý do là vì:

- mỗi window đã bị tách ra thành một object riêng từ trước
- injector xử lý từng window riêng
- injector không nhìn thấy raw full sequence
- injector cũng không giữ một trạng thái anomaly span chung cho nhiều window

Vì vậy, synthetic anomaly hiện tại là **window-local augmentation**, không phải **sequence-level augmentation**.

## Implementation Consequences Visible Today

Các hệ quả dưới đây không phải là đề xuất hay phê bình. Đây chỉ là những gì có thể đọc ra trực tiếp từ implementation hiện tại.

### 1. Synthetic anomaly span bị chặn bởi window

Vì segment bounds được lấy từ `window_size` trong `src/data/augment.py:197-217`, nên anomaly span hiện tại không thể dài vượt ra ngoài window đang xét.

### 2. Synthetic anomaly không dùng ngữ cảnh rộng hơn một window

Vì injector nhận đầu vào là `batch["x"]` với shape `[B, L, D]` tại `src/data/augment.py:865-876`, nên mọi quyết định injection hiện tại đều dựa trên thông tin của chính window đó.

Không có chỗ nào trong runtime path hiện tại cho phép injector nhìn sang:

- prefix trước window
- suffix sau window
- toàn bộ raw series của entity

### 3. Tính nhất quán theo timeline gốc không được đảm bảo giữa các window overlap

Vì injector xử lý từng `batch_index` độc lập tại `src/data/augment.py:889-915`, nên hai window overlap không buộc phải chia sẻ cùng một anomaly process trong vùng thời gian chồng lắp.

### 4. Synthetic anomaly không được materialize thành dataset mới ở tầng loader

`src/data/loaders.py` chỉ:

- clean
- scale
- windowize
- build dataloader

Nó không inject synthetic anomaly vào `raw_sequences` hay `scaled_sequences`.

Nghĩa là synthetic anomaly hiện tại là **runtime augmentation**, không phải **persistent augmented dataset artifact**.

## Code References

- `src/data/loaders.py:77-102` - build `WindowDataset` objects from scaled sequences
- `src/data/loaders.py:169-179` - fit scaler on train sequences, transform all splits, then build windows
- `src/data/loaders.py:196-263` - `WindowDataset` slices full sequences into windows
- `src/data/collate.py:11-36` - collate windows into batch
- `src/core/contracts.py:95-103` - batch contract requires `batch["x"]` to have rank 3
- `src/engine/trainer.py:621-637` - trainer sends loader batches into `model.training_step(...)`
- `src/models/redlamp_mlp_baseline.py:308-321` - baseline injects synthetic anomalies in `_prepare_batch`
- `src/models/redlamp_mlp_baseline.py:623-687` - baseline `_shared_step` and `training_step`
- `src/models/thesis_multitask.py:2239-2240` - thesis model injects synthetic anomalies in train preparation path
- `src/models/thesis_multitask.py:2284-2290` - thesis model injects synthetic anomalies for synthetic validation path
- `src/models/thesis_multitask.py:3145-3166` - thesis `_shared_step` prepares batch and then forwards
- `src/models/thesis_multitask.py:3249-3278` - thesis training and validation step entry points
- `src/data/augment.py:197-217` - anomaly segment length is sampled from current `window_size`
- `src/data/augment.py:865-950` - injector augments a batch of windows with shape `[B, L, D]`

## Pipeline Documentation

Current offline pipeline for synthetic anomaly training:

1. raw dataset parser returns full split sequences
2. cleaning pipeline processes split sequences
3. scaler is fit on clean train sequences only
4. scaled split sequences are produced
5. `WindowDataset` slices scaled sequences into fixed windows
6. `DataLoader + collate_windows` build a batch of windows
7. model-side `_prepare_batch(...)` calls `SyntheticAnomalyInjector.augment_batch(...)`
8. forward pass and loss computation run on the already-windowized, already-augmented batch

This is the active implementation path for both:

- `src/models/redlamp_mlp_baseline.py`
- `src/models/thesis_multitask.py`

## Historical Context (from documents/)

The design documents under `documents/design/` are consistent with a fixed-window training view.

Two signals are relevant:

1. `documents/design/idea.md` states the active thesis experiments around fixed-length windows with `L = 20`.
2. The design language around RedLamp-style subsequence corruption is compatible with local corruption inside a window.

However, the decisive evidence for injection order is still the runtime code path, not the design documents.

## Open Questions

1. The repository currently implements window-local synthetic augmentation. It does not currently implement sequence-level augmentation before windowization.
2. The question of whether sequence-level injection would be more realistic is not answered by the current implementation. That would be a separate design decision, not a fact about the current code.
3. If future work wants anomaly spans that stay consistent across overlapping windows or use broader temporal context than one window, that behavior would require a different augmentation surface than the current `augment_batch(batch)` API.
