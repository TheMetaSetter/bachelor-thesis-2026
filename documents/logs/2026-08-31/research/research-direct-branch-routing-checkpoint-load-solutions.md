---
date: 2026-08-31 08:29:27 +0700
researcher: OpenAI Codex
topic: "Rà soát codebase và đề xuất hướng xử lý lỗi checkpoint cho direct branch routing"
status: complete
revision: 670468a378c0a084e327b90f8d4eee02a8a47b4e
branch: dev
---

# Research: Rà soát codebase và đề xuất hướng xử lý lỗi checkpoint cho direct branch routing

## Summary

Lỗi xảy ra vì runner direct routing đang nạp trực tiếp checkpoint Stage A vào mô hình Stage B bằng `strict=True`. Checkpoint Stage A còn có hai khóa `discrete_assignment.weight` và `discrete_assignment.bias`. Mô hình Stage B dùng `cosine_topk` không tạo lớp này, nên PyTorch báo `Unexpected key(s)` trước khi chạy forward.

Codebase đã có sẵn đường đi đúng hơn: `stage_b_init.pt`. Checkpoint này được tạo sau khi bridge Stage A sang Stage B và khởi tạo memory. Kiểm tra read-only trên GPU cloud cho thấy đủ 18 file `stage_b_init.pt` cho tổ hợp O0/O1 × 3 entity × 3 seed.

## Research question

Rà soát codebase để xác định nguyên nhân lỗi load checkpoint của thí nghiệm direct branch routing và đề xuất ba hướng giải quyết đơn giản nhất.

## System context

Thí nghiệm đặt mô hình ở `training_phase: stage_b_fusion_finetuning` và `fusion_mode: direct_branch_routing`. Runtime vẫn tính output của hai prototype branch, nhưng hàm fusion trả thẳng continuous latent cho reconstruction head và discrete latent cho classification head. Hai fusion projection cũ vẫn tồn tại trong `state_dict` để giữ tương thích checkpoint, nhưng không được dùng trong đường forward direct.

## Execution path

1. `scripts/run_direct_branch_routing_smoke.py` gọi `build_smoke_experiment_config()`.
2. Hàm này gọi `build_direct_experiment_config()` trong `scripts/run_direct_branch_routing_full.py`.
3. Runner đặt `training_phase` thành Stage B, đặt `fusion_mode` thành `direct_branch_routing`, rồi đặt checkpoint khởi tạo thành `stage_a_multitask_pretraining/checkpoints/best.pt`.
4. `run_training_experiment()` dựng mô hình Stage B.
5. `maybe_initialize_model_from_checkpoint()` gọi `CheckpointManager.load_checkpoint()` với mặc định `strict=True`.
6. `model.load_state_dict()` dừng vì checkpoint có hai khóa `discrete_assignment.*` nhưng mô hình Stage B cosine-topk không có lớp tương ứng.
7. Vì lỗi xảy ra ở bước 6, runtime chưa chạy memory initialization, forward, loss hoặc optimizer step.

## Detailed findings

### 1. Mô hình direct routing không có `discrete_assignment` ở Stage B cosine-topk

`_build_prototype_memory()` đặt `self.discrete_assignment = None` khi phase là Stage B và query mode là `cosine_topk`. Đây là thiết kế có chủ ý vì cosine-topk dùng codebook để truy vấn, không dùng assignment head Gumbel.

`_compute_fusion_outputs()` kiểm tra `fusion_mode`. Với `direct_branch_routing`, hàm trả `continuous_hidden` cho `hidden_reconstruction` và `discrete_hidden` cho `hidden_classification`, đồng thời đặt alpha và beta bằng zero.

### 2. Loader mặc định yêu cầu state-dict khớp tuyệt đối

`maybe_initialize_model_from_checkpoint()` không truyền đối số `strict`, nên dùng giá trị mặc định của `CheckpointManager.load_checkpoint()`. Loader này gọi `model.load_state_dict(..., strict=True)`. Vì vậy mọi khóa thừa hoặc thiếu đều làm run dừng.

### 3. Runner và cấu hình standalone đang không cùng hợp đồng checkpoint

`configs/experiment/offline_ablation/thesis/smd__thesis__offline__direct_branch_routing__machine_1_6__w20__seed6__stage_b.yaml` đã trỏ tới:

`outputs/benchmark/smd/thesis/O0/machine_1_6/seed6/two_stage/initializations/stage_b_init.pt`

Ngược lại, `build_direct_experiment_config()` trong full runner trỏ tới Stage A `best.pt`. Smoke runner dùng lại hàm này, nên cũng nhận Stage A `best.pt`. Đây là xung đột giữa cấu hình standalone và runner sinh cấu hình động.

### 4. Stage A best và Stage B init có trạng thái khác nhau

Kiểm tra read-only checkpoint O0/machine-1-6/seed6 trên GPU cloud cho thấy:

| Checkpoint | Có `discrete_assignment.*` | `memory_initialized` | Phase trong config |
| --- | --- | --- | --- |
| Stage A `best.pt` | Có | `false` | `stage_a_multitask_pretraining` |
| `stage_b_init.pt` | Không | `true` | `stage_b_fusion_finetuning` |

Stage B init vì vậy không chỉ giải quyết tên khóa. Nó còn chứa trạng thái memory đã được khởi tạo và metadata xác minh.

### 5. Codebase đã có bridge Stage A → Stage B

`_load_stage_a_state_into_stage_b_model()` dùng `strict=False`, nhưng chỉ cho phép đúng hai khóa thừa là `discrete_assignment.weight` và `discrete_assignment.bias`. Sau đó `_prepare_stage_b_initialization_checkpoint()` gọi `load_checkpoint_extra_state()`, khởi tạo memory từ train loader, rồi lưu `stage_b_init.pt`.

Đây là bằng chứng rằng việc dùng `strict=False` tùy tiện trong loader chung không phải hợp đồng hiện tại. Bridge hiện có đã giới hạn mismatch và xử lý thêm vòng đời memory.

## Three simplest solution directions

### Hướng 1 — Dùng `stage_b_init.pt` đã có (khuyến nghị)

Đổi `initialization_checkpoint_path` trong full/smoke runner từ Stage A `best.pt` sang checkpoint tương ứng dưới `two_stage/initializations/stage_b_init.pt`. Không cần đổi model và không cần chạy lại Stage A.

Đây là cách ít thay đổi nhất vì nó khớp với file cấu hình direct ablation hiện có và với loader `strict=True`. Kiểm tra cloud đã xác nhận đủ 18 file cho toàn bộ ma trận chạy.

Giới hạn: thí nghiệm bắt đầu từ trạng thái Stage B đã được bridge và khởi tạo memory. Nó không trực tiếp kiểm tra khả năng nạp raw Stage A vào mô hình direct.

### Hướng 2 — Tạo bridge checkpoint riêng cho direct routing từ Stage A best

Giữ Stage A `best.pt` làm nguồn. Trước khi chạy Stage B direct, dựng mô hình Stage B direct, nạp Stage A bằng bridge allowlist hiện có, khởi tạo memory từ train loader, rồi lưu một checkpoint init riêng dưới output của direct routing. Sau đó runner nạp checkpoint init này bằng `strict=True`.

Hướng này giữ rõ nguồn gốc “bắt đầu từ Stage A best” và vẫn không chạy lại Stage A. Chi phí là thêm một bước chuẩn bị checkpoint trước training.

### Hướng 3 — Cho phép compatibility load có phạm vi hẹp

Thêm một nhánh loader chỉ dành cho trường hợp `training_phase` là Stage B, `fusion_mode` là `direct_branch_routing`, và checkpoint nguồn là Stage A. Nhánh này dùng `strict=False`, kiểm tra allowlist đúng hai khóa `discrete_assignment.*`, rồi để Trainer gọi hook `maybe_initialize_memories_from_loader()` ở đầu epoch.

Hướng này ít file thay đổi hơn hướng 2, nhưng phải kiểm tra kỹ `memory_initialized`. Nếu Stage A checkpoint báo memory chưa khởi tạo, Trainer phải thực sự chạy bước khởi tạo trước forward đầu tiên. Không nên đổi loader chung thành `strict=False` cho mọi checkpoint vì sẽ che giấu mismatch không liên quan.

## Comparison

| Hướng | Thay đổi | Dùng lại Stage A best | Rủi ro chính | Mức đơn giản |
| --- | --- | --- | --- | --- |
| 1. Dùng `stage_b_init.pt` | Chỉ sửa đường dẫn runner/config | Gián tiếp, qua bridge có sẵn | Phụ thuộc artifact Stage B init | Cao nhất |
| 2. Bridge riêng cho direct | Thêm bước tạo init checkpoint | Có | Thêm bước và artifact mới | Trung bình |
| 3. Compatibility load hẹp | Sửa loader + kiểm tra memory | Có, trực tiếp | Dễ bỏ sót lifecycle memory | Trung bình |

## Evidence

- `scripts/run_direct_branch_routing_full.py:43-83` — runner đặt phase Stage B, direct routing và đường dẫn Stage A `best.pt`.
- `scripts/run_direct_branch_routing_smoke.py:25-78` — smoke runner dùng lại cấu hình động của full runner rồi gọi `run_training_experiment()`.
- `scripts/cli/train.py:81-92` — loader khởi tạo gọi `CheckpointManager.load_checkpoint()` mà không truyền `strict`.
- `scripts/cli/train.py:222-228` — mô hình được dựng rồi nạp checkpoint trước optimizer và training loop.
- `src/engine/checkpoint.py:277-317` — `strict=True` là mặc định và được truyền vào `load_state_dict()`.
- `src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py:216-229` — Stage B cosine-topk không tạo `discrete_assignment`.
- `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:361-392` — direct routing bỏ qua hai fusion block trong output task.
- `src/engine/trainer.py:568-589` — Trainer kiểm tra và có thể khởi tạo memory ở đầu mỗi epoch.
- `scripts/experiments/run_two_stage_offline_pretraining.py:224-241` — bridge chỉ cho phép hai khóa assignment là unexpected.
- `scripts/experiments/run_two_stage_offline_pretraining.py:244-337` — bridge khởi tạo memory và lưu `stage_b_init.pt`.
- `configs/experiment/offline_ablation/thesis/smd__thesis__offline__direct_branch_routing__machine_1_6__w20__seed6__stage_b.yaml:7-15` — cấu hình standalone đã dùng `stage_b_init.pt` và direct routing.
- GPU cloud, `/root/bachelor-thesis-2026`, revision `670468a378c0a084e327b90f8d4eee02a8a47b4e` — kiểm tra read-only xác nhận 18 `stage_b_init.pt` không rỗng.

## Configuration observed

| Setting | Active value | Evidence | Scope |
| --- | --- | --- | --- |
| `training_phase` | `stage_b_fusion_finetuning` | `scripts/run_direct_branch_routing_full.py:76-83` | Direct runner |
| `fusion_mode` | `direct_branch_routing` | `scripts/run_direct_branch_routing_full.py:76-83` | Direct runner |
| `discrete_query_mode` | `cosine_topk` | `configs/model/thesis_multitask_two_stage_window20.yaml:77-82` | Model default |
| Loader strictness | `true` | `src/engine/checkpoint.py:277-304` | Checkpoint load |
| Stage A source | `.../stage_a_multitask_pretraining/checkpoints/best.pt` | `scripts/run_direct_branch_routing_full.py:57-61` | Current dynamic runner |
| Standalone Stage B source | `.../two_stage/initializations/stage_b_init.pt` | `configs/experiment/offline_ablation/thesis/smd__thesis__offline__direct_branch_routing__machine_1_6__w20__seed6__stage_b.yaml:9` | Existing direct config |

## Conflicts and uncertainties

- Code and tests for the dynamic full runner currently assert Stage A `best.pt`, while the standalone direct ablation config uses `stage_b_init.pt`. The available evidence does not establish whether the dynamic runner change was intentional or accidental.
- The live cloud check confirms that 18 `stage_b_init.pt` files exist and are non-empty. It does not by itself prove that every file has correct metric quality or matching data provenance.
- Hướng 3 is technically plausible because Trainer has a memory-init hook, but the current code does not define a generic compatibility mode for this case. It therefore needs a focused test before implementation.

## Open questions

- Có muốn thí nghiệm định nghĩa điểm bắt đầu là `stage_b_init.pt` đã bridge, hay bắt buộc lấy raw Stage A `best.pt` làm đầu vào trực tiếp?
- Nếu chọn Hướng 2 hoặc 3, artifact init riêng có cần giữ nguyên trong output canonical của từng run không?
