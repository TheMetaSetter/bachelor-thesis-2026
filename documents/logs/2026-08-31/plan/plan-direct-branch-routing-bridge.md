---
date: 2026-08-31 Asia/Ho_Chi_Minh
planner: OpenAI Codex
topic: "Implement direct branch routing checkpoint bridge"
status: implemented
revision: 6cd32ac94cc20876828f5c62fd2df5c3ac557587
branch: dev
related_research: documents/logs/2026-08-31/research/research-direct-branch-routing-bridge-flow.md
---

# Implementation Plan: Direct Branch Routing Checkpoint Bridge

**Goal:** Tạo checkpoint khởi tạo Stage B direct từ Stage A `best.pt` đã có, rồi chạy Stage B direct mà không chạy lại Stage A.

**Architecture:** Bridge đọc checkpoint Stage A trên CPU, dựng model Stage B với `fusion_mode: direct_branch_routing`, nạp state với allowlist tương thích, nạp `extra_state`, khởi tạo memory banks từ train loader và lưu `stage_b_init.pt`. Runtime direct không gọi hai fusion head; các module cũ chỉ còn để giữ state-dict keys.

**Tech Stack:** Python, PyTorch, YAML, pytest, `.venv/bin/python`.

**Spec:** `documents/logs/2026-08-31/research/research-direct-branch-routing-bridge-flow.md`

## Global Constraints

- Không chạy Stage A.
- Không gọi hoặc cập nhật hai fusion head trong direct Stage B.
- Chỉ cho phép `discrete_assignment.weight` và `discrete_assignment.bias` là khóa thừa khi chuyển checkpoint.
- Khởi tạo memory banks đúng một lần từ train loader không shuffle.
- Stage B training nạp checkpoint bridge bằng `strict=True` mặc định.
- Giữ nguyên loss, preprocessing, metric, stochastic inference và output schema.
- Không chạy full benchmark hoặc SSH trong phiên lập trình local.

## Current state

- `scripts/experiments/run_two_stage_offline_pretraining.py` đã có logic chuyển Stage A sang Stage B nhưng logic này chỉ nhận manifest của two-stage runner.
- `scripts/run_direct_branch_routing_full.py` hiện đưa raw Stage A `best.pt` vào `initialization_checkpoint_path` của Stage B direct.
- `scripts/cli/train.py` nạp checkpoint với `strict=True`, nên raw Stage A có thể gây mismatch `discrete_assignment.*`.
- Config standalone đã dùng `two_stage/initializations/stage_b_init.pt`, nhưng runner động chưa tự tạo checkpoint direct riêng.

## Desired end state

- Có hàm bridge dùng được cho một cấu hình Stage B và một checkpoint Stage A.
- Có CLI bridge riêng, không đi qua two-stage orchestrator.
- Full runner và smoke runner tạo hoặc tái sử dụng đúng checkpoint bridge trước khi gọi `run_training_experiment()`.
- Output bridge nằm trong stage output direct:
  `outputs/benchmark/smd/<entity>/seed<seed>/thesis_direct_branch_routing_<offline_variant>/offline/stage_b/initializations/stage_b_init.pt`.
- Một test chứng minh output bridge load được bằng `strict=True` và đã có memory banks.

## Scope

### In scope

- Tách logic bridge hiện có thành hàm dùng chung.
- Thêm CLI bridge Stage A → Stage B direct.
- Đổi full/smoke runner sang checkpoint bridge.
- Thêm test cho mismatch, memory banks, strict load và không chạy Stage A.

### Out of scope

- Sửa Stage A hoặc chạy lại Stage A.
- Sửa hai model fusion cũ.
- Thay đổi loss, metric, threshold, preprocessing hoặc online adaptation.
- Chạy GPU cloud, full matrix hoặc benchmark kết quả.

## Evidence

- `scripts/experiments/run_two_stage_offline_pretraining.py:224-337` — logic allowlist, nạp `extra_state`, khởi tạo memory banks và lưu checkpoint.
- `src/models/thesis_multitask_impl/thesis_multitask_state_memory_init_helpers.py:18-43` — contract khởi tạo memory banks từ train loader.
- `src/engine/checkpoint.py:277-313` — checkpoint loader dùng `strict=True` mặc định.
- `scripts/run_direct_branch_routing_full.py:43-83` — runner tạo cấu hình direct nhưng còn trỏ raw Stage A.
- `scripts/run_direct_branch_routing_smoke.py:25-78` — smoke runner dùng lại cấu hình động của full runner.
- `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:369-392` — direct route không thực hiện fusion.

## Implementation approach

Mở rộng `scripts/experiments/run_two_stage_offline_pretraining.py` bằng một hàm bridge nhận trực tiếp `stage_b_config`, `stage_a_checkpoint_path` và `initialization_checkpoint_path`. Giữ `_prepare_stage_b_initialization_checkpoint()` làm adapter cho flow cũ. Thêm `scripts/experiments/run_direct_branch_routing_bridge.py` làm CLI mỏng gọi hàm này. Runner full/smoke dùng cùng hàm trước khi train và đổi `initialization_checkpoint_path` sang output bridge.

## Phase 1: Chốt contract bridge bằng test

### Goal

Test mô tả input, output và các mismatch được phép trước khi sửa runtime.

### Changes

- **File:** `tests/experiments/test_direct_branch_routing_bridge.py` (đề xuất mới)
- **Change:** Tạo fixture nhỏ cho Stage A payload, Stage B direct config và train loader; kiểm tra allowlist, memory banks và strict reload.
- **Reason:** Khóa hành vi bridge trước khi tách code dùng chung.

### Verification

- **Automated:** `.venv/bin/python -m pytest -q tests/experiments/test_direct_branch_routing_bridge.py` — test mới ban đầu phải thất bại vì chưa có entry point bridge.

## Phase 2: Tách và triển khai bridge

### Goal

Bridge tạo được `stage_b_init.pt` Stage-B-compatible mà không gọi Stage A.

### Changes

- **File:** `scripts/experiments/run_two_stage_offline_pretraining.py`
- **Symbols:** `_load_stage_a_state_into_stage_b_model`, `_prepare_stage_b_initialization_checkpoint`, hàm mới `prepare_stage_b_initialization_checkpoint`.
- **Change:** Đưa sáu thao tác bridge vào hàm nhận config và hai path trực tiếp; adapter manifest gọi lại hàm mới.
- **Reason:** Dùng lại code đã có, không tạo flow chuyển checkpoint thứ hai.

- **File:** `scripts/experiments/run_direct_branch_routing_bridge.py` (đề xuất mới)
- **Change:** Nhận `--stage-b-config`, `--stage-a-checkpoint`, `--output-checkpoint`; load config; gọi hàm bridge; in output path.
- **Reason:** Có entry point riêng, không kích hoạt two-stage orchestrator.

### Verification

- **Automated:** Test bridge pass, output load strict pass, mismatch ngoài allowlist bị từ chối.
- **Manual:** Đọc log bridge và xác nhận không có lệnh Stage A, validation, test hoặc optimizer.

## Phase 3: Nối runner và config

### Goal

Full/smoke direct luôn dùng checkpoint bridge tương ứng trước khi train.

### Changes

- **File:** `scripts/run_direct_branch_routing_full.py`
- **Symbols:** `build_direct_experiment_config`, `main`, helper path identity.
- **Change:** Tách path Stage A source khỏi path output bridge; tạo bridge cho từng combination; đặt output bridge làm `initialization_checkpoint_path` trước `run_training_experiment()`.
- **Reason:** Không đưa raw Stage A vào loader strict của Stage B.

- **File:** `scripts/run_direct_branch_routing_smoke.py`
- **Change:** Gọi cùng bridge path và giữ output dưới `outputs/benchmark_smoke`.
- **Reason:** Smoke phải kiểm tra đúng flow full nhưng chỉ với một combination nhỏ.

- **Files:** các YAML direct standalone hiện có (chỉ khi cần)
- **Change:** Đồng nhất `initialization_checkpoint_path` với output bridge direct.
- **Reason:** Một config chỉ có một nguồn khởi tạo rõ ràng.

### Verification

- **Automated:** Test 18 config có source và output bridge duy nhất; test smoke không chứa `two_stage`.
- **Manual:** Kiểm tra path output không ghi đè Stage A và mỗi identity có một `stage_b_init.pt`.

## Phase 4: Kiểm tra flow hoàn chỉnh local

### Goal

Xác nhận bridge và một bước Stage B direct hoạt động cùng nhau trên fixture local, không chạy Stage A.

### Changes

- Không thêm behavior mới. Chỉ bổ sung test nếu Phase 3 phát hiện mismatch.

### Verification

- **Automated:** `.venv/bin/python -m pytest -q tests/experiments/test_direct_branch_routing_bridge.py tests/models/test_direct_branch_routing.py tests/runtime/test_checkpoint_roundtrip.py` — pass.
- **Automated:** `.venv/bin/python -m pytest -q tests/benchmarks/test_direct_branch_routing_full_runner.py tests/benchmarks/test_direct_branch_routing_smoke_runner.py` — pass.
- **Manual:** Chạy bridge trên fixture; kiểm tra payload có `memory_initialized=true`, `verification_metadata_source` và path đúng.

## Testing strategy

Test theo thứ tự: contract mismatch → bridge serialization → strict reload → runner path → một forward/backward Stage B. Không dùng GPU cloud trong phiên local. Không gọi two-stage runner trong bất kỳ test direct nào.

## Migration and rollback

Bridge ghi file mới dưới output direct, không sửa hoặc xóa Stage A. Nếu bridge lỗi, xóa riêng file init direct chưa hoàn tất và sửa code; Stage A artifact vẫn giữ nguyên. Runner chỉ train sau khi file bridge tồn tại và load strict thành công.

## Documentation

- Cập nhật comment/CLI help trong entry point bridge để nói rõ Stage A chỉ là input và Stage A không chạy lại.
- Cập nhật research/plan reference nếu path output hoặc tên hàm thay đổi.

## Final verification

- [ ] Bridge tạo `stage_b_init.pt` từ một Stage A `best.pt` đã có.
- [ ] `CheckpointManager.load_checkpoint(..., strict=True)` load được output bridge.
- [ ] Memory banks đã khởi tạo trước Stage B training.
- [ ] Full/smoke runner không gọi Stage A.
- [ ] Direct forward vẫn route continuous latent vào reconstruction head và discrete latent vào classification head.

## Assumptions and non-blocking uncertainties

- `stage_b_init.pt` có thể tồn tại sẵn trên cloud; runner sẽ tái sử dụng nếu artifact đọc được, nếu không thì tạo lại từ Stage A source.
- Tên CLI mới là đề xuất; implementer phải giữ module form `python -m` theo quy ước repository.
