---
date: 2026-08-31T08:24:02+07:00
researcher: OpenAI Codex
topic: "Rà soát lỗi checkpoint khi chạy direct branch routing và đề xuất hướng xử lý tối giản"
status: complete
revision: 670468a378c0a084e327b90f8d4eee02a8a47b4e
branch: dev
---

# Research: Lỗi checkpoint của direct branch routing

## Summary

Smoke run dừng trước forward pass vì runner nạp sai loại checkpoint.

Model direct routing của Stage B không tạo `discrete_assignment` khi dùng `discrete_query_mode: cosine_topk`. Checkpoint Stage A vẫn chứa hai khóa của lớp này. `scripts.cli.train` đang gọi `load_checkpoint()` với `strict=True`, nên PyTorch dừng ở hai khóa thừa.

Hướng đơn giản nhất là dùng `stage_b_init.pt` tương ứng với cùng offline variant, entity và seed. Đây là checkpoint đã qua bước chuyển Stage A sang Stage B và đã khởi tạo memory. Codebase đã có đúng cơ chế này. Không cần chạy lại Stage A.

## Research question

Rà soát codebase để xác định nguyên nhân lỗi khi smoke run direct branch routing không load được checkpoint, sau đó đề xuất các cách xử lý đơn giản nhất mà không chạy lại Stage A.

## System context

Smoke runner tạo một cấu hình Stage-B-only từ baseline O0, `machine_1_6`, seed 6. Nó đặt `fusion_mode` thành `direct_branch_routing`, giữ `training_phase` là `stage_b_fusion_finetuning`, rồi gọi `run_training_experiment()`.

Direct routing đã được cài trong model. Nó trả latent của continuous branch cho reconstruction head và latent của discrete branch cho classification head. Nhánh này không gọi hai concat projection. Như vậy, lỗi hiện tại xảy ra trước phần direct routing.

## Execution path

1. `scripts.run_direct_branch_routing_smoke` tạo config bằng `build_direct_experiment_config()`.
2. Runner đặt `initialization_checkpoint_path` tới Stage A `best.pt`.
3. `run_training_experiment()` đăng ký runtime, tạo dataset, tạo model và gọi `maybe_initialize_model_from_checkpoint()`.
4. Hàm này gọi `CheckpointManager.load_checkpoint()` mà không truyền `strict`, nên dùng mặc định `strict=True`.
5. Model Stage B direct được tạo với `discrete_assignment = None` vì `training_phase` là Stage B và `discrete_query_mode` là `cosine_topk`.
6. Checkpoint Stage A chứa `discrete_assignment.weight` và `discrete_assignment.bias`.
7. `model.load_state_dict(..., strict=True)` báo hai khóa thừa và dừng chương trình.
8. Trainer chưa kịp chạy batch đầu tiên, nên chưa có forward, backward, đo runtime hoặc đo memory.

## Detailed findings

### 1. Lỗi nằm ở cặp checkpoint-model

**Implemented:** `ThesisMultitaskSetupMixin._build_prototype_memory()` bỏ lớp `discrete_assignment` trong Stage B khi query mode là `cosine_topk`. Đây là lựa chọn có chủ ý của runtime hiện tại.

**Configured:** model config dùng `training_phase: stage_a_multitask_pretraining`, `fusion_mode: task_specific_concat_projection` và `discrete_query_mode: cosine_topk`. Direct runner thay hai giá trị phase và fusion mode sau khi load baseline.

**Runtime evidence:** log smoke run cho thấy model có `discrete_assignment` với 0 tham số, sau đó `strict=True` báo hai khóa `discrete_assignment.*` là unexpected. Điều này khớp với source code và không phải lỗi dữ liệu.

### 2. `best.pt` của Stage A không phải checkpoint khởi tạo hoàn chỉnh cho Stage B

Stage A `best.pt` lưu trạng thái của model Stage A. Sau Stage A, two-stage runner phải tạo một model Stage B, nạp state tương thích, thu latent token pool, khởi tạo continuous prototype bank, discrete codebook và verification metadata, rồi mới lưu `stage_b_init.pt`.

Trainer cũng có hook khởi tạo memory ở đầu epoch nếu memory chưa sẵn sàng. Tuy nhiên, `stage_b_init.pt` vẫn là artifact đúng của ranh giới Stage A → Stage B. Nó giữ rõ provenance và tránh biến Stage B smoke run thành một đường khởi tạo khác.

### 3. Codebase đã có cầu nối tương thích

`_load_stage_a_state_into_stage_b_model()` đã dùng `strict=False`, nhưng chỉ cho phép đúng hai khóa thừa là `discrete_assignment.weight` và `discrete_assignment.bias`. Sau đó `_prepare_stage_b_initialization_checkpoint()` gọi `maybe_initialize_memories_from_loader()` và lưu payload Stage-B-compatible.

Đây là bằng chứng rằng project không nên đổi loader chung thành `strict=False` một cách im lặng. Cơ chế tương thích đã có phạm vi hẹp và kiểm tra `missing_keys`/`unexpected_keys`.

### 4. Runner hiện chọn checkpoint không đúng với config Stage-B-only đã thiết kế

Config ablation độc lập trỏ tới:

```text
outputs/benchmark/smd/thesis/O0/machine_1_6/seed6/two_stage/initializations/stage_b_init.pt
```

Nhưng `scripts/run_direct_branch_routing_full.py` và smoke runner hiện xây config trỏ tới:

```text
outputs/benchmark/smd/thesis/O0/machine_1_6/seed6/two_stage/stage_a_multitask_pretraining/checkpoints/best.pt
```

Hai lựa chọn này không tương đương. Lựa chọn thứ hai gây đúng lỗi trong log.

## Simplest solution options

| Ưu tiên | Hướng | Thay đổi cần có | Rủi ro |
| --- | --- | --- | --- |
| 1 | Dùng `stage_b_init.pt` tương ứng | Đổi đường dẫn khởi tạo trong smoke/full config hoặc builder | Thấp nhất. Đúng lifecycle Stage B và không chạy Stage A |
| 2 | Tạo `stage_b_init.pt` từ Stage A `best.pt` bằng helper hiện có | Gọi bridge hiện có một lần cho từng combination còn thiếu, sau đó runner dùng file mới | Thấp nếu kiểm tra đủ path; có thêm bước ghi artifact |
| 3 | Cho phép load Stage A bằng compatibility mode có kiểm tra khóa | Mở rộng helper nhỏ ở runner, cho phép đúng hai khóa thừa và vẫn khởi tạo memory | Trung bình. Dễ tạo đường chạy khác với artifact `stage_b_init.pt` |
| 4 | Giữ `discrete_assignment` trong model direct | Đổi query mode hoặc tạo lớp không dùng | Không nên chọn. Tăng state không dùng và làm sai ý nghĩa ablation |

### Khuyến nghị

Chọn hướng 1. Với mỗi run, ánh xạ:

```text
{offline_variant}/{entity}/seed{seed}/
  two_stage/initializations/stage_b_init.pt
```

Ví dụ smoke O0 / `machine_1_6` / seed6 dùng file `stage_b_init.pt` cùng identity. Sau khi sửa builder, chỉ cần kiểm tra file tồn tại, load strict và chạy một smoke run. Không gọi two-stage orchestrator và không chạy Stage A.

Nếu một file `stage_b_init.pt` thật sự thiếu, chọn hướng 2. Không tự động rơi xuống Stage A `best.pt` bằng `strict=False`; fallback đó có thể che một mismatch khác.

## Configuration observed

| Setting | Active value | Evidence | Scope |
| --- | --- | --- | --- |
| `training_phase` | `stage_b_fusion_finetuning` | `scripts/run_direct_branch_routing_full.py:76-83` | Direct smoke/full model |
| `fusion_mode` | `direct_branch_routing` | `scripts/run_direct_branch_routing_full.py:76-83` | Direct smoke/full model |
| `discrete_query_mode` | `cosine_topk` | `configs/model/thesis_multitask_two_stage_window20.yaml:77-83` | Shared model config |
| loader strictness | `True` | `src/engine/checkpoint.py:277-304` | Generic checkpoint load |
| smoke initialization | Stage A `best.pt` | `scripts/run_direct_branch_routing_full.py:57-61` | Current smoke/full builder |
| standalone ablation initialization | `stage_b_init.pt` | `configs/experiment/offline_ablation/thesis/smd__thesis__offline__direct_branch_routing__machine_1_6__w20__seed6__stage_b.yaml:7-15` | Existing explicit ablation config |

## Tests and validation evidence

**Tested:** direct routing tests verify branch identity, MC routing, frozen legacy fusion modules, and forward/backward behavior. The checkpoint compatibility test covers a legacy Stage-B model whose keys match the direct model.

**Not yet tested:** a test that loads a real Stage-A `best.pt` into the direct Stage-B model. The current failing smoke run is the first evidence for this exact mismatch.

**Historical artifact evidence:** the remote checkpoint inventory records `stage_b_init.pt`, Stage A checkpoints, and Stage B checkpoints for all 18 O0/O1 × entity × seed combinations. The inventory is historical and should be revalidated read-only before a new remote run.

## Conflicts and uncertainties

- The current full/smoke builder uses Stage A `best.pt`, while the explicit standalone ablation YAML and the earlier plan use `stage_b_init.pt`.
- The pasted log proves the two unexpected keys, but it does not prove the current remote copy of every `stage_b_init.pt` is readable. The inventory says those files existed in an earlier scan; re-check exact files before the full run.
- `strict=False` alone may let the first load pass, and the trainer can initialize memory at epoch start. Nevertheless, that path is less explicit than using the prepared Stage-B initialization artifact.

## Open questions

1. Does the current remote checkout still contain all 18 `stage_b_init.pt` files with the expected run identity?
2. Were the current smoke/full runner changes pushed to the same remote revision that produced the pasted log?
3. Should the runner use the checked-in standalone ablation YAMLs, or keep generating configs from the 18 baseline YAMLs?

## Conclusion

The smallest safe correction is configuration-level: make smoke/full direct runs consume the matching `stage_b_init.pt`. The direct routing implementation itself is not the failing part. If a Stage-B initialization artifact is missing, use the existing Stage-A-to-Stage-B bridge once and then run direct Stage B. Avoid changing the global checkpoint loader to silently ignore unexpected keys.
