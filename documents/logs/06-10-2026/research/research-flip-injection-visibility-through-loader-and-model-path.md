---
date: 2026-06-10 13:20:16 +07
researcher: OpenAI Codex
git_commit: 1a2825cbfe400a9f3ce280d83f2da8f26c39daba
branch: dev
repository: bachelor-thesis-2026
topic: "Khả năng phép flip không được áp dụng lên cửa sổ và mô hình không nhìn thấy được"
tags: [research, time-series, anomaly-detection, synthetic-anomaly, flip]
status: complete
last_updated: 2026-06-10
last_updated_by: OpenAI Codex
---

# Research: Khả năng phép flip không được áp dụng lên cửa sổ và mô hình không nhìn thấy được

**Date**: 2026-06-10 13:20:16 +07
**Researcher**: OpenAI Codex
**Git Commit**: 1a2825cbfe400a9f3ce280d83f2da8f26c39daba
**Branch**: dev

## Research Question

Có khả năng nào phép `flip` chưa được áp dụng lên cửa sổ và mô hình không nhìn thấy được hay không? Cần kiểm tra toàn bộ pipeline từ khâu load mẫu dữ liệu, collate batch, injection, cho tới chỗ model tiêu thụ `batch["x"]`, với chú ý đặc biệt tới các toán tử gán thực sự thay đổi tensor.

## Summary

Trong pipeline chuẩn của codebase, khi một mẫu được quyết định là synthetic anomaly với family `flip`, phép `flip` thực sự được áp dụng vào `augmented_batch["x"]`, và tensor đã bị sửa này được chuyển trực tiếp vào `forward(...)` của model. Không có bước nào trong đường đi chuẩn từ loader tới model ghi đè `x` trở lại clean window sau khi injection đã xảy ra.

Có ba nhóm tình huống cần phân biệt rõ. Thứ nhất, có những nhánh mà injection không được gọi, nên model nhìn thấy clean window là đúng thiết kế, không phải lỗi. Thứ hai, có những nhánh mà injection được gọi và phép gán được thực thi, nhưng giá trị trước và sau trùng nhau, khiến `flip` trở thành một no-op về mặt số học. Thứ ba, có khả năng một caller bên ngoài tự đưa vào một batch đã gắn cờ là pre-augmented nhưng dữ liệu bên trong lại không phản ánh augmentation; model sẽ tin batch đó và không augment lại. Tình huống thứ ba không xuất hiện trong đường đi loader -> collate -> model chuẩn của repository, nhưng về mặt lập trình Python thì vẫn có thể xảy ra nếu caller tự tạo batch sai.

## Detailed Findings

### Data Preparation

`WindowDataset.__getitem__` materialize từng cửa sổ bằng cách clone slice của chuỗi nguồn:

```python
return {
    "x": sequence["x"][start_index:end_index].clone(),
    ...
}
```

Điều này nằm tại `src/data/loaders.py:91-115`. Nghĩa là mỗi sample window đi ra từ dataset đã là một tensor riêng, không phải view dùng chung với full sequence.

Sau đó `collate_windows(...)` stack các window thành batch:

```python
batch = {
    "x": torch.stack([window["x"] for window in windows], dim=0),
    ...
}
```

Điều này nằm tại `src/data/collate.py:11-36`. Như vậy `batch["x"]` là tensor mới có shape `[B, L, D]`.

### Augmentation

Đối với `flip`, registry ánh xạ family name `"flip"` sang `_inject_flip_family` tại `src/data/augment.py:88-106`.

Segment để inject được lấy từ `_sample_segment_bounds(...)`:

```python
segment_length = int(
    self._randint(
        min_segment_length, max_segment_length + 1, (1,), device=device
    ).item()
)
start_index = int(
    self._randint(0, max_start_index + 1, (1,), device=device).item()
)
end_index = start_index + segment_length
```

Điều này nằm tại `src/data/augment.py:190-210`. Với config đang dùng cho SMD window-20 balanced, `min_segment_fraction=0.1` và `max_segment_fraction=0.2`, nên segment length hiện hành là 2-4 time-step, không phải 1 (`configs/task/multitask_tsad_redlamp_multiclass_window20_balanced.yaml:14-15`).

Hàm `flip` thật sự là:

```python
segment = self._extract_channel_segment(
    clean_channel_window, start_index, end_index
)
updated_segment = torch.flip(segment, dims=(0,))
local_mask = torch.ones(
    segment.shape[0], dtype=torch.long, device=segment.device
)
anomalous_channel_window, channel_mask = self._apply_segment_update(
    clean_channel_window,
    start_index,
    end_index,
    updated_segment,
    local_mask,
)
family_parameters = {"operation": "reverse_subsequence"}
```

Điều này nằm tại `src/data/augment.py:322-343`.

Phép gán thay đổi tensor thực sự diễn ra trong `_apply_segment_update(...)`:

```python
anomalous_channel_window = clean_channel_window.clone()
anomalous_channel_window[start_index:end_index, 0] = updated_segment
channel_mask[start_index:end_index] = local_mask.long()
```

Điều này nằm tại `src/data/augment.py:258-274`. Về semantics của Python/PyTorch, đây là gán slice in-place vào tensor clone. Nếu `updated_segment` khác segment gốc, giá trị trong tensor kết quả sẽ đổi thật.

Sau đó `_inject_single_window(...)` ghi channel đã sửa trở lại window đã augment:

```python
augmented_window[:, channel_index] = anomalous_channel_window.squeeze(-1)
anomaly_mask = torch.maximum(anomaly_mask, channel_mask)
family_parameters_by_channel[str(channel_index)] = family_parameters
```

Điều này nằm tại `src/data/augment.py:678-689`.

Cuối cùng `augment_batch(...)` ghi window đã augment vào batch đầu ra:

```python
augmented_batch["x"][batch_index] = augmented_window
anomaly_masks[batch_index] = anomaly_mask
classification_labels[batch_index] = target_class_label
augmentation_metadata.append(window_metadata)
```

Điều này nằm tại `src/data/augment.py:800-826`.

### Modeling and Training

Ở RedLamp baseline, `_prepare_batch(...)` sẽ gọi injector trong `train` hoặc `val_synth` nếu synthetic augmentation/validation đang bật:

```python
if stage_name == "train" and self.use_synthetic_augmentation:
    return self.synthetic_anomaly_injector.augment_batch(batch)
if stage_name == "val_synth" and self.use_synthetic_validation:
    return self.synthetic_validation_injector.augment_batch(batch)
```

Điều này nằm tại `src/models/redlamp_mlp_baseline.py:278-289`.

Sau đó `_shared_step(...)` dùng thẳng `prepared_batch`:

```python
prepared_batch = self._prepare_batch(batch, stage_name)
outputs = self.forward(prepared_batch)
reconstruction_loss = F.mse_loss(outputs["recon"], prepared_batch["x"])
```

Điều này nằm tại `src/models/redlamp_mlp_baseline.py:592-597`. Trong `forward(...)`, model đọc trực tiếp:

```python
x_tensor = batch["x"]
hidden = self.encoder(x_tensor)
```

Điều này nằm tại `src/models/redlamp_mlp_baseline.py:322-331`.

Ở thesis multitask model, logic tương tự. `_prepare_batch(...)` dùng synthetic validation injector cho `val_synth` và `val_realistic`, còn `_prepare_clean_batch(...)` dùng train injector cho `train`:

```python
if stage_name == "train" and self.use_synthetic_augmentation:
    return self.synthetic_anomaly_injector.augment_batch(batch)
...
if stage_name in {"val_synth", "val_realistic"} and self.use_synthetic_validation:
    return self.synthetic_validation_injector.augment_batch(batch)
```

Điều này nằm tại `src/models/thesis_multitask.py:1642-1644` và `src/models/thesis_multitask.py:1687-1690`.

Sau đó `_shared_step(...)` cũng dùng `prepared_batch` trực tiếp trong forward:

```python
prepared_batch = self._prepare_batch(batch, stage_name)
...
outputs = self.forward(prepared_batch, stage_name=stage_name)
```

Điều này nằm tại `src/models/thesis_multitask.py:2340-2358`. Encoder của model đọc thẳng:

```python
hidden = self.network(batch["x"])
```

Điều này nằm tại `src/models/thesis_multitask.py:187-194`.

### Evaluation

Phần visualization hiện tại chỉ đọc `augmented_batch["x"]`, `synthetic_anomaly_mask`, và `augmentation_metadata` để vẽ, nên nó không thể làm mất augmentation. Xem `scripts/visualize_synthetic_anomalies.py:54-113`.

## Code References

- `src/data/loaders.py:91` - `WindowDataset.__getitem__` clone slice thành window riêng
- `src/data/collate.py:17` - stack window thành `batch["x"]`
- `src/data/augment.py:322` - `_inject_flip_family`
- `src/data/augment.py:266` - gán `updated_segment` vào `anomalous_channel_window[start_index:end_index, 0]`
- `src/data/augment.py:687` - gán channel đã augment trở lại `augmented_window`
- `src/data/augment.py:823` - gán `augmented_window` vào `augmented_batch["x"][batch_index]`
- `src/models/redlamp_mlp_baseline.py:286` - gọi `synthetic_anomaly_injector.augment_batch(batch)`
- `src/models/redlamp_mlp_baseline.py:594` - `forward(prepared_batch)`
- `src/models/redlamp_mlp_baseline.py:324` - `x_tensor = batch["x"]`
- `src/models/thesis_multitask.py:1643` - gọi `synthetic_anomaly_injector.augment_batch(batch)`
- `src/models/thesis_multitask.py:1689` - gọi `synthetic_validation_injector.augment_batch(batch)`
- `src/models/thesis_multitask.py:2358` - `forward(prepared_batch, stage_name=stage_name)`
- `src/models/thesis_multitask.py:188` - `hidden = self.network(batch["x"])`

## Pipeline Documentation

Đường đi chuẩn của dữ liệu là:

`full sequence -> WindowDataset.__getitem__ clone -> collate_windows stack -> model._prepare_batch -> SyntheticAnomalyInjector.augment_batch -> model.forward(prepared_batch)`

Trong đường đi chuẩn này, không có bước nào ghi đè `prepared_batch["x"]` trở lại clean window sau khi augmentation đã được áp dụng.

Runtime check cục bộ trên một batch toy xác nhận hành vi này. Với `anomaly_families=("flip",)`, `anomaly_probability=1.0`, `min_segment_fraction=max_segment_fraction=0.5`, injector trả metadata:

```python
{
    'is_synthetic_anomaly': True,
    'anomaly_family': 'flip',
    'anomaly_family_index': 0,
    'start_index': 2,
    'end_index': 5,
    'affected_channels': [2],
    'family_parameters_by_channel': {'2': {'operation': 'reverse_subsequence'}}
}
```

và biến đổi channel bị ảnh hưởng từ:

```python
clean = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
```

thành:

```python
augmented = [0.0, 1.0, 4.0, 3.0, 2.0, 5.0]
```

Điều này xác nhận `flip` đã đi vào `augmented_batch["x"]`.

## Historical Context (from documents/)

`documents/design/idea.md` và `documents/design/design_starter.md` đều mô tả synthetic anomaly injection là một phần của offline multitask objective và model-owned batch preparation. Điều này khớp với implementation hiện tại: augmentation không được làm ở dataset layer, mà được thực hiện trong model preparation step.

## Open Questions

Trong pipeline chuẩn, không có bằng chứng cho thấy `flip` bị tính toán xong rồi mất trước khi vào model. Tuy nhiên, vẫn có các trường hợp mà model không quan sát được thay đổi hữu hiệu:

1. Mẫu được gán class `normal`, nên `should_inject` là `False` và không có flip.
2. Stage hiện tại không bật synthetic augmentation theo config, ví dụ `val` hoặc `test`.
3. `torch.flip(...)` được gọi nhưng segment là no-op về mặt số học.

Nhóm thứ ba gồm hai khả năng:

- Segment length bằng 1 nếu config cho phép.
- Segment có đối xứng/palindrome hoặc toàn giá trị giống nhau, nên `torch.flip(segment, dims=(0,)) == segment`.

Trong config balanced window-20 đang dùng, segment length hiện hành là 2-4, nên trường hợp length 1 không xảy ra ở run này. Tuy vậy, palindrome hoặc constant subsequence vẫn có thể xảy ra, và khi đó phép gán vẫn chạy nhưng giá trị trước/sau không đổi, khiến model không nhìn thấy một biến đổi thực sự.
