---
date: 2026-07-30T18:00:00+07:00
topic: "Sửa memory lifecycle giữa Stage A và Stage B của THESIS offline"
status: approved
revision: 1a570cda6eb7976255add3fc5d4f7f385d40dea3
branch: dev
related_documents:
  - documents/logs/30-07-2026/research/research-stage-memory-lifecycle-mismatch.md
  - documents/logs/30-07-2026/plan/plan-fix-stage-memory-lifecycle.md
---

# Implementation Structure: Sửa memory lifecycle giữa Stage A và Stage B

## Summary

Implementation sẽ sửa phase boundary của THESIS để Stage A không khởi tạo hoặc
truy xuất memory. Stage B sẽ tải checkpoint Stage A, khởi tạo hai memory bank,
freeze chúng rồi mới train fusion/prediction heads.

Quá trình gồm bốn pha theo dependency: kiểm tra an toàn và baseline, sửa phase
gate, đồng bộ test/checkpoint contract, rồi xác minh toàn bộ pipeline bằng smoke.

## Request

Dựa trên [implementation plan](../plan/plan-fix-stage-memory-lifecycle.md),
triển khai theo thứ tự các pha và stage dưới đây. Không sửa source/config/test
trong lúc viết structure hoặc detail plan; không tạo commit.

## Confirmed context

- `Trainer.train()` gọi model memory hook trước batch đầu tiên của mỗi epoch
  [tại epoch loop](../../../../src/engine/trainer.py#L565-L596).
- Phase gate hiện cho cả Stage A và Stage B đi vào prototype path
  [tại `_phase_uses_prototype_path()`](../../../../src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py#L50-L58).
- Forward đã có nhánh passthrough khi phase không dùng prototype path
  [tại `forward()`](../../../../src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py#L135-L234)
  và [passthrough outputs](../../../../src/models/thesis_multitask_impl/thesis_multitask_state_passthrough_mixin.py#L9-L60).
- Stage-B runner tải Stage-A state, khôi phục extra state rồi gọi memory
  initializer [tại preparation flow](../../../../scripts/experiments/run_two_stage_offline_pretraining.py#L254-L291).
- Spec v3 yêu cầu Stage A chạy trước memory construction và không dùng memory
  retrieval [tại lifecycle](../../../spec/full-spec-v3.md#L61-L73) và
  [Stage A contract](../../../spec/full-spec-v3.md#L430-L464).

## Scope

### In scope

- Giới hạn phase gate để chỉ Stage B dùng prototype retrieval.
- Giữ Trainer và Stage-B orchestration, không thêm codepath mới.
- Cập nhật fixture/test cho Stage A bypass, Stage B initialization và
  checkpoint transition.
- Chạy focused tests, compile check và O1 two-stage smoke.

### Out of scope

- Không đổi thuật toán xây continuous bank hoặc discrete codebook.
- Không đổi loss formula, config budget, checkpoint schema hoặc online TTA.
- Không xoá memory buffers khỏi `state_dict`.
- Không sửa specification lịch sử.
- Không chạy full benchmark matrix trước smoke pass.

## Proposed phases

### Phase 1: Boundary và baseline được khóa

**Result:** Có inventory chính xác, preservation list và test baseline trước khi
chỉnh code.

**Stages:**

1. **Stage 1.1 — Kiểm tra change boundary:** xác nhận các caller của phase
   gate, initializer, passthrough path, checkpoint lifecycle và các test trực
   tiếp gọi initializer.
2. **Stage 1.2 — Chạy baseline:** chạy nhóm test memory, checkpoint và
   orchestration; ghi nhận failure có sẵn.

**Depends on:** Research note và implementation plan.

**Verification:** Exact search, `git status`, focused pytest.

**Risks:** Dirty worktree có thay đổi khác; phải giữ nguyên các thay đổi đó.

**Complete when:** Không còn ambiguity về file/caller cần sửa và có baseline
   test result.

### Phase 2: Stage A bypass memory, Stage B giữ prototype path

**Result:** Một phase gate duy nhất phản ánh đúng lifecycle: Stage A dùng
   passthrough; Stage B dùng prototype path.

**Stages:**

1. **Stage 2.1 — Sửa phase gate:** `_phase_uses_prototype_path()` chỉ trả
   `True` cho `stage_b_fusion_finetuning`.
2. **Stage 2.2 — Kiểm tra model behavior:** xác nhận Stage A không collect
   token pool, không retrieval, không optional prototype loss; xác nhận Stage B
   vẫn initialize được.

**Depends on:** Phase 1 hoàn tất.

**Verification:** Model unit tests và state/bank assertions.

**Risks:** Test cũ dùng model mặc định Stage A nhưng lại mong initializer hoặc
   memory update; phải phân loại fixture theo mục tiêu test.

**Complete when:** Stage A initializer trả `False` và bank không đổi; Stage B
   initializer trả `True` trên loader nhỏ.

### Phase 3: Checkpoint và test contract đồng bộ

**Result:** Checkpoint Stage A giữ lifecycle chưa initialized và không chặn
   Stage-B initialization.

**Stages:**

1. **Stage 3.1 — Cập nhật unit fixtures:** memory initialization/update tests
   chạy dưới Stage B; Stage-A tests kiểm tra bypass.
2. **Stage 3.2 — Thêm checkpoint transition regression:** tạo Stage-A
   checkpoint, load vào Stage-B model, gọi initializer và kiểm tra state sau init.
3. **Stage 3.3 — Chạy regression groups:** memory, checkpoint, model shape,
   point-score loss và two-stage orchestration tests.

**Depends on:** Phase 2.

**Verification:** `memory_initialized=False` trong Stage-A checkpoint; Stage-B
   init thành công; output keys/loss contract không đổi.

**Risks:** Test checkpoint round-trip cũ có thể kiểm tra memory buffer đã init;
   giữ test đó và thêm test lifecycle mới, không thay thế bằng assertion yếu hơn.

**Complete when:** Ba hành vi được test độc lập: Stage A không init, Stage B
   init được, checkpoint transition không fail.

### Phase 4: End-to-end verification và artifact audit

**Result:** Offline two-stage pipeline O1 chạy hết Stage A -> Stage-B init ->
   Stage B -> evaluation với epoch budget smoke.

**Stages:**

1. **Stage 4.1 — Static/local gate:** compile, focused tests, diff/whitespace
   check và kiểm tra không có source change ngoài scope.
2. **Stage 4.2 — O1 smoke:** chạy Stage A hai epoch, Stage B một epoch trên
   config `machine_1_6`, seed 6; local không có CUDA thì chạy remote theo
   `ssh-gpu.txt`.
3. **Stage 4.3 — Artifact audit:** kiểm tra Stage-A extra state, `stage_b_init.pt`,
   Stage-B best checkpoint, freeze state và execution report.

**Depends on:** Phase 3 pass.

**Verification:** Không còn lỗi `Stage B initialization checkpoint could not
initialize memories`; artifact có state đúng lifecycle.

**Risks:** Smoke có thể dùng output cũ hoặc nhầm config; dùng output directory
mới và xác minh manifest/config/checkpoint provenance.

**Complete when:** Smoke pass, Stage-A checkpoint chưa initialized, Stage-B init
checkpoint đã initialized/frozen và Stage-B training hoàn tất.

## Dependency summary

| Phase | Requires | Enables |
| --- | --- | --- |
| Phase 1 | Research, plan, live worktree | Boundary và baseline đáng tin cậy |
| Phase 2 | Phase 1 | Model lifecycle đúng spec |
| Phase 3 | Phase 2 | Checkpoint transition và regression evidence |
| Phase 4 | Phase 3 | Concrete end-to-end smoke và artifact acceptance |

## Decisions confirmed

- Sửa `_phase_uses_prototype_path()` thay vì thêm guard vào `Trainer`, vì model
  đã sở hữu semantics phase và đã có passthrough branch.
- Không thêm flag/config path mới.
- Không đổi `maybe_initialize_memories_from_loader()` ngoài việc để phase guard
  hiện có nhận đúng kết quả từ predicate mới.
- Không đổi checkpoint schema; chỉ sửa lifecycle state được ghi vào checkpoint.

## Non-blocking uncertainties

- Một số test cũ dùng model mặc định Stage A cho memory primitive. Detail plan
  phải xác định rõ test nào chuyển sang Stage B và test nào trở thành Stage-A
  bypass regression.
- Smoke remote cần xác nhận đường dẫn checkout và thiết bị CUDA từ `ssh-gpu.txt`
  tại thời điểm triển khai.

## Feedback requested

Structure này đã được mở rộng tiếp thành detail plan theo yêu cầu của anh; không
còn chờ một quyết định làm thay đổi phase order.
