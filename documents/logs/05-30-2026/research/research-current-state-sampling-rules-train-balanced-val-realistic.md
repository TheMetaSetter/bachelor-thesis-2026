# Research: Current Codebase State vs. Confirmed Sampling Rules (Train Balanced + Val Realistic)

Date: 2026-05-30
Scope: Trace current implementation status for decisions in `documents/logs/05-30-2026/detail/detail-sampling-rules-train-balanced-val-realistic.md`
Method: Repository-grounded read following `prompts/1_research_prompt.md` (no speculation)

## Research Question
Codebase hiện tại đã triển khai tới đâu cho các quyết định sampling mới: 
- train per-batch balancing cho 12 lớp,
- policy round-robin phần dư,
- realistic validation source từ test prior,
- override anomaly rate,
- strict config semantics và fail-fast validation.

## Executive Status
Tại thời điểm hiện tại, các semantics mới trong file detail **chưa được triển khai** trong code/config/test. Codebase đang chạy theo semantics cũ:
- Có synthetic augmentation + synthetic validation (`val_synth`) dựa trên cùng `val_loader`.
- Có tùy chọn cân bằng **nhị phân** trong batch (`balance_binary_classes_within_batch`).
- Không có khái niệm `val_realistic_source`, `val_anomaly_rate_override`, hay `train_balance_classes`.

## Findings

### 1) Train per-batch balance (12 lớp) 
Hiện tại injector chỉ có cơ chế sampling nhị phân anomaly/clean thông qua cờ `balance_binary_classes_within_batch`, không có allocator đa lớp 12-class.

- `src/data/augment.py:46` nhận `balance_binary_classes_within_batch`.
- `src/data/augment.py:713-726` chỉ lấy quyết định inject boolean theo số lượng anomaly window trong batch.
- `src/data/augment.py:743-765` gán `classification_labels` theo từng window sau khi inject; không có quota theo từng class index.
- `configs/task/multitask_tsad_redlamp_multiclass_window20.yaml:10` đang đặt `balance_binary_classes_within_batch: false`.
- `configs/task/multitask_tsad_window10_binary.yaml:9` có bật `balance_binary_classes_within_batch: true` cho bài toán binary/window10.

Kết luận: chưa có `task.train_balance_classes` và chưa có logic cân bằng tương đối cho 12 lớp.

### 2) Remainder round-robin allocation
Không tìm thấy bất kỳ implementation hoặc test nào cho round-robin phần dư theo class index khi `batch_size % 12 != 0`.

- Không có match cho các token liên quan (`round-robin`, `round_robin`, `train_balance_classes`) trong `src/`, `configs/`, `tests/`.

Kết luận: chưa triển khai.

### 3) Val realistic source (`test_same_scope` / `test_smd_all`)
Pipeline hiện tại chỉ có hai validation stage:
- `val`: clean validation,
- `val_synth`: synthetic validation trên chính `val_loader`.

Không có stage `val_realistic` và không có source switch từ test prior.

- `src/engine/trainer.py:658-663` chạy `validation_step` trên `val_loader`.
- `src/engine/trainer.py:678-684` chạy `synthetic_validation_step` cũng trên `val_loader`.
- `src/models/thesis_multitask.py:2307-2321` tách rõ `validation_step` và `synthetic_validation_step`.
- `tests/test_multitask_validation_alignment.py:143-192` xác nhận metric tách `val` và `val_synth`, không có `val_realistic`.

Kết luận: chưa có `task.val_realistic_source` và chưa có route `test_same_scope|test_smd_all`.

### 4) Window-level anomaly definition
Định nghĩa “window anomalous nếu có >=1 điểm anomaly” **đã tồn tại** trong API helper:

- `src/data/api.py:102-107`:
  - kiểm tra shape `[B, L]`,
  - trả về `(point_labels.sum(dim=1) > 0).long()`.

Định nghĩa này khớp với quyết định mới ở file detail, nhưng hiện chưa được nối vào một `val_realistic` pipeline riêng.

### 5) Override behavior (`val_anomaly_rate_override`)
Không có field cấu hình hoặc logic runtime nào cho override anomaly rate ở validation realistic.

- Không có match cho `val_anomaly_rate_override` trong `configs/src/tests`.

Kết luận: chưa triển khai.

### 6) Uniform anomaly-family distribution (11 families)
Injector đang chọn family bằng random đều trên danh sách `anomaly_families` đang bật:

- `src/data/augment.py:657-663` chọn index random trong `[0, len(self.anomaly_families))`.

Điểm cần lưu ý: đây là uniform trên tập families được cấu hình, không phải một cơ chế explicit “val realistic family prior module”.

### 7) Minimal config fields và strict semantics
Config validator hiện chỉ cho phép khóa cũ của `multitask_tsad`, và sẽ reject khóa lạ.

- `src/core/config.py:295-310` whitelist task keys hiện hành (không có 4 field mới).
- `src/core/config.py:326-333` unknown task keys => raise ValueError fail-fast.
- `src/core/config.py:673-679` validate boolean cho `balance_binary_classes_within_batch`.
- `src/core/config.py:825-850` validate range cho anomaly params hiện có.

Kết luận:
- Có fail-fast validation theo semantics cũ.
- Nếu thêm field mới ngay bây giờ vào YAML thì sẽ fail vì unknown keys.

## Direct Gap Mapping to Confirmed Decisions

1. `task.train_balance_classes`: **Missing**.
2. Round-robin remainder over 12 classes: **Missing**.
3. `task.val_realistic`: **Missing**.
4. `task.val_realistic_source`: **Missing**.
5. `task.val_anomaly_rate_override`: **Missing**.
6. Window anomaly rule (`>=1` anomalous point): **Present helper**, chưa wired vào realistic-validation branch.
7. Uniform 11-family sampling: **Partially present** trong injector random family choice, nhưng chưa có dedicated realistic-validation prior controller.
8. Backward-compatibility policy “remove old incompatible configs”: **Chưa áp dụng**; codebase vẫn đang vận hành với config schema cũ.

## Historical Context Alignment
Tài liệu design hiện tại trong `documents/design/` mô tả pipeline multitask + synthetic validation đang chạy, và code path thực tế khớp với hướng đó. File detail ngày 2026-05-30 là tập quyết định mới hơn, nhưng chưa được phản ánh vào implementation.

## Open Questions
1. `train_balance_classes` có cần áp cho cả `classification_label_mode=binary` hay chỉ `redlamp_multiclass`?
2. Với `val_realistic_source=test_smd_all`, code có cần parser riêng cho toàn bộ test entities hay tái sử dụng parser hiện tại kèm chế độ entity scope?
3. Khi `val_anomaly_rate_override` được set, family distribution có vẫn giữ uniform 11 families hay cần bám prior từ source?
4. “Remove outdated configs” sẽ remove ở mức file YAML nào (task-only hay cả experiment presets liên quan)?

## Key Code References
- `src/data/augment.py:46`
- `src/data/augment.py:713-726`
- `src/data/augment.py:743-765`
- `src/data/augment.py:657-663`
- `src/models/thesis_multitask.py:200-210`
- `src/models/thesis_multitask.py:2307-2321`
- `src/engine/trainer.py:658-684`
- `src/core/config.py:295-310`
- `src/core/config.py:326-333`
- `src/core/config.py:673-679`
- `src/core/config.py:825-850`
- `src/data/api.py:102-107`
- `configs/task/multitask_tsad_redlamp_multiclass_window20.yaml:1-14`
- `configs/task/multitask_tsad_window10_binary.yaml:1-13`
- `tests/test_multitask_validation_alignment.py:98-109`
- `tests/test_multitask_validation_alignment.py:143-192`
