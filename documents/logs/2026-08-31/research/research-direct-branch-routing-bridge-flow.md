---
date: 2026-08-31
researcher: OpenAI Codex
topic: "Thiết kế runtime flow tối giản cho bridge của Hướng 2"
status: implemented
revision: 6cd32ac94cc20876828f5c62fd2df5c3ac557587
branch: dev
---

# Research: Thiết kế runtime flow tối giản cho bridge của Hướng 2

## Summary

Bridge chỉ làm một việc: chuyển một checkpoint Stage A `best.pt` đã có thành checkpoint khởi tạo cho Stage B direct routing.

Luồng tối thiểu gồm sáu thao tác: đọc checkpoint nguồn, dựng model Stage B direct, nạp trọng số với allowlist tương thích, nạp trạng thái phụ, khởi tạo các **memory banks**, rồi lưu `stage_b_init.pt`. Bridge không chạy lại Stage A và không gọi hai khối fusion.

## Research question

Thiết kế runtime flow tối giản cho bridge trong Hướng 2, với kiến trúc direct routing không dùng hai khối fusion.

## System context

Model direct routing dùng `training_phase: stage_b_fusion_finetuning` và `fusion_mode: direct_branch_routing`.

Trong forward path, continuous latent đi thẳng vào reconstruction head. Discrete latent đi thẳng vào classification head. Code không thực hiện phép fusion trong nhánh này tại `thesis_multitask_routing_geometry_helpers.py:369-392`.

Model Stage B với `discrete_query_mode: cosine_topk` không tạo `discrete_assignment` tại `thesis_multitask_setup_mixin.py:216-229`. Vì vậy checkpoint Stage A có thể chứa hai khóa mà model Stage B không dùng.

## Execution path

Luồng được duyệt cho bridge là:

```text
Stage A best.pt đã có
        │
        ▼
Đọc checkpoint trên CPU
        │
        ▼
Dựng model Stage B direct
        │
        ▼
Nạp model_state_dict với strict=False
và chỉ cho phép hai khóa discrete_assignment.*
        │
        ▼
Nạp extra_state của Stage A
        │
        ▼
Dựng train loader và khởi tạo memory banks một lần
        │
        ▼
Lưu stage_b_init.pt của direct routing
        │
        ▼
Stage B direct training nạp checkpoint bằng strict=True
```

Bridge chỉ đọc train loader. Bridge không chạy validation, test, optimizer, scheduler hoặc đánh giá metric.

## Detailed findings

### 1. Các thao tác bridge đã có trong code

`_load_stage_a_state_into_stage_b_model()` đã dùng `strict=False`, cho phép đúng hai khóa `discrete_assignment.weight` và `discrete_assignment.bias`, rồi báo lỗi nếu có khóa thiếu hoặc khóa thừa khác.

`_prepare_stage_b_initialization_checkpoint()` vẫn giữ adapter cho flow two-stage cũ. Hàm `prepare_stage_b_initialization_checkpoint()` và CLI riêng thực hiện cùng contract cho direct routing.

Đây là phần code hiện có. Thiết kế Hướng 2 chỉ cần đặt các thao tác này sau một entry point bridge riêng cho direct routing.

### 2. Memory banks là bước bắt buộc

Hook `maybe_initialize_memories_from_loader()` dừng nếu không có token normal. Khi chạy thành công, hook cập nhật các memory banks và metadata xác minh, sau đó đánh dấu model đã khởi tạo memory.

Bridge phải lưu `extra_state` sau bước này. Nếu bỏ qua bước khởi tạo, checkpoint direct có thể chưa sẵn sàng cho Stage B.

### 3. Direct routing không gọi fusion

Bridge dựng model với `fusion_mode: direct_branch_routing` để checkpoint đích có đúng cấu hình Stage B.

Model vẫn có thể giữ các module fusion cũ trong `state_dict` để tương thích khóa của checkpoint Stage A. Đây chỉ là tương thích lưu trữ. Bridge và Stage B direct không gọi các module đó và không huấn luyện chúng.

### 4. Stage B training dùng checkpoint bridge

Loader chung hiện gọi `load_checkpoint()` với `strict=True` mặc định tại `scripts/cli/train.py:81-92` và `src/engine/checkpoint.py:277-313`.

Vì vậy Stage B training không nên nạp raw Stage A `best.pt`. Nó phải nạp `stage_b_init.pt` do bridge tạo. Checkpoint này đã có cấu trúc của model Stage B, nên không còn mismatch `discrete_assignment.*`.

## Minimal bridge contract

| Thành phần | Nội dung |
| --- | --- |
| Input | Một Stage A `best.pt` đã có và cấu hình Stage B direct tương ứng |
| Model đích | `training_phase=stage_b_fusion_finetuning`, `fusion_mode=direct_branch_routing` |
| Loader cần dùng | Chỉ train loader, `shuffle_train=false`, `num_workers=0` |
| Compatibility rule | Chỉ cho phép `discrete_assignment.weight` và `discrete_assignment.bias` là khóa thừa |
| Memory step | Khởi tạo memory banks đúng một lần từ train loader |
| Output | `.../offline/stage_b/initializations/stage_b_init.pt` |
| Sau bridge | Stage B direct training nạp output bằng `strict=True` |

## Configuration observed

| Setting | Active value | Evidence | Scope |
| --- | --- | --- | --- |
| `training_phase` | `stage_b_fusion_finetuning` | `scripts/run_direct_branch_routing_full.py:76-83` | Direct Stage B model |
| `fusion_mode` | `direct_branch_routing` | `scripts/run_direct_branch_routing_full.py:76-83` | Direct forward path |
| Stage A source của dynamic runner | Stage A `best.pt` | `scripts/run_direct_branch_routing_full.py:48-56` | Chỉ được dùng làm input cho bridge |
| Direct initialization output | `.../offline/stage_b/initializations/stage_b_init.pt` | `scripts/run_direct_branch_routing_full.py:61-67` | Input strict cho Stage B |
| Loader strictness sau bridge | `true` | `src/engine/checkpoint.py:277-304` | Stage B training |
| Memory initialization | Từ train loader | `scripts/experiments/run_two_stage_offline_pretraining.py:275-290` | Bridge |

## Conflicts and uncertainties

- Dynamic runner đã tách Stage A source khỏi Stage B initialization output và gọi bridge trước Stage B training.
- `run_two_stage_offline_pretraining.py` hiện luôn chạy Stage A khi dùng entry point orchestration đầy đủ. Hướng 2 không được gọi entry point đó để chạy trực tiếp; bridge cần dùng riêng phần chuẩn bị checkpoint.
- CLI `scripts.experiments.run_direct_branch_routing_bridge` đã tạo entry point riêng chỉ làm bridge, không chạy Stage A.

## Open questions

Không còn câu hỏi thiết kế mở cho luồng tối giản này. Việc còn lại là triển khai entry point bridge và kiểm thử rằng checkpoint đầu ra nạp được bằng `strict=True`.

## Evidence

- `scripts/experiments/run_two_stage_offline_pretraining.py:224-241` — bridge kiểm tra mismatch giữa Stage A và Stage B.
- `scripts/experiments/run_two_stage_offline_pretraining.py:244-337` — bridge dựng model, nạp trạng thái, khởi tạo memory banks và lưu checkpoint.
- `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:369-392` — direct routing trả latent trực tiếp cho hai task head.
- `src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py:216-229` — Stage B cosine-topk không tạo `discrete_assignment`.
- `src/models/thesis_multitask_impl/thesis_multitask_state_memory_init_helpers.py:18-43` — hook khởi tạo memory banks từ train loader.
- `scripts/cli/train.py:81-92` — training loader nạp checkpoint khởi tạo.
- `src/engine/checkpoint.py:277-313` — `strict=True` là mặc định khi nạp checkpoint.
- `src/engine/trainer.py:568-589` — Trainer cũng kiểm tra hook khởi tạo memory banks ở đầu epoch.
