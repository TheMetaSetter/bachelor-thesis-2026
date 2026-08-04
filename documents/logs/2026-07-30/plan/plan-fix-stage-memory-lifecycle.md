---
date: 2026-07-30T00:00:00+07:00
planner: OpenAI Codex
topic: "Sửa memory lifecycle giữa Stage A và Stage B của THESIS offline"
status: ready
revision: 1a570cda6eb7976255add3fc5d4f7f385d40dea3
branch: dev
related_research: documents/logs/30-07-2026/research/research-stage-memory-lifecycle-mismatch.md
---

# Implementation Plan: Sửa memory lifecycle giữa Stage A và Stage B

## Summary

Kế hoạch sửa lỗi bằng cách giới hạn `_phase_uses_prototype_path()` chỉ còn
`stage_b_fusion_finetuning`. Stage A sẽ dùng nhánh passthrough đã có, không
retrieval và không gọi memory initializer. Stage-B orchestration vẫn tải
checkpoint Stage A, gọi initializer một lần trên train loader, rồi lưu
`stage_b_init.pt` trước khi train fusion/prediction heads.

Đây là kế hoạch triển khai, chưa sửa source, test, config hoặc specification và
chưa tạo commit.

## Request and acceptance criteria

### Current behavior

`Trainer.train()` gọi `maybe_initialize_memories_from_loader()` trước batch đầu
tiên của mỗi epoch. Với Stage A, model hiện được xem là đang dùng prototype path,
nên initializer xây dựng hai memory bank ngay trong Stage A. Checkpoint Stage A
sau đó lưu `memory_initialized: true`; Stage-B initializer khôi phục flag này và
trả về `False`, làm runner ném `RuntimeError`.

Các kết luận này được ghi trong [research note](../research/research-stage-memory-lifecycle-mismatch.md)
và được kiểm tra lại ở [Trainer epoch order](../../../../src/engine/trainer.py#L565-L596),
[model phase gate](../../../../src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py#L50-L58),
[memory initializer](../../../../src/models/thesis_multitask_impl/thesis_multitask_state_memory_init_helpers.py#L18-L44)
và [Stage-B preparation](../../../../scripts/experiments/run_two_stage_offline_pretraining.py#L254-L291).

### Desired behavior

```text
Stage A
  -> encoder + task heads/losses
  -> no memory initialization
  -> no continuous/discrete retrieval
  -> checkpoint extra_state.memory_initialized = false

Stage B preparation
  -> load Stage-A model state and extra_state
  -> initialize continuous bank and discrete codebook from train loader
  -> mark memory initialized and freeze memory updates
  -> save stage_b_init.pt
  -> train Stage-B fusion/prediction heads
```

Acceptance criteria:

1. Stage A không gọi token-pool collection hoặc `mark_memories_initialized()`.
2. Stage A forward dùng `phase_direct_passthrough`; output contract vẫn giữ
   `continuous_branch`, `discrete_branch`, `fusion`, `recon`, `logits`,
   `point_scores` và `window_scores`.
3. Stage A không dùng continuous/discrete retrieval và các optional prototype
   losses không chạy trong Stage A.
4. Stage-A best checkpoint có `extra_state["memory_initialized"] is False`.
5. Stage B initialization gọi thành công đúng một lần, tạo đủ hai memory bank,
   đặt `memory_initialized=True` và tôn trọng
   `freeze_memories_after_initialization=True`.
6. Stage-B training tiếp tục dùng prototype path và encoder/memory freeze
   contract hiện tại.
7. Smoke run O1 với Stage A hai epoch và Stage B một epoch đi qua toàn bộ
   offline two-stage pipeline trên CPU hoặc remote CUDA tùy môi trường.

## Constraints and exclusions

### In scope

- phase gate của THESIS multitask model;
- kiểm tra initializer caller trong `Trainer` và Stage-B runner;
- cập nhật test fixtures đang gọi initializer trực tiếp;
- thêm regression test cho checkpoint transition Stage A -> Stage B;
- chạy focused tests, compile check và một two-stage smoke.

### Out of scope

- không đổi schema hoặc thuật toán xây continuous bank/discrete codebook;
- không đổi `Trainer` thành nơi chứa logic Stage A/Stage B;
- không đổi Stage-A loss công thức, synthetic anomaly generation hoặc metric;
- không đổi config budget, checkpoint manager, output layout hoặc online TTA;
- không xoá memory buffers khỏi `state_dict`; chúng vẫn được model đăng ký từ
  lúc construction và sẽ được Stage-B initializer ghi đè;
- không sửa `full-spec-v2.md` hoặc `full-spec-v3.md` trong fix này;
- không chạy full benchmark matrix trước khi concrete smoke pass.

## Design decision

| Option | Decision | Lý do và evidence |
| --- | --- | --- |
| Đổi `_phase_uses_prototype_path()` để chỉ trả `True` cho Stage B | **Chọn** | Nhánh passthrough đã tồn tại ở [state passthrough](../../../../src/models/thesis_multitask_impl/thesis_multitask_state_passthrough_mixin.py#L9-L60). Cùng predicate đang điều khiển forward retrieval, optional losses, memory initializer và trainable fusion modules, nên một phase gate duy nhất giảm số codepath. |
| Thêm check riêng trong `Trainer` để bỏ qua Stage A | Không chọn | Đưa ý nghĩa phase vào engine loop, trong khi [Trainer](../../../../src/engine/trainer.py#L565-L596) hiện chỉ gọi model hook. Cách này vẫn để model Stage A có thể retrieval nếu caller khác gọi forward. |
| Chỉ đổi `maybe_initialize_memories_from_loader()` nhưng giữ Stage A prototype path | Không chọn | Stage A sẽ không build bank ở đầu epoch nhưng vẫn có thể đi vào retrieval path với các buffer tồn tại; điều này không chứng minh được spec “Stage A does not use retrieval”. |
| Tắt memory bằng config riêng cho Stage A | Không chọn | Tạo thêm config path và không giải quyết rõ lifecycle của generated Stage-A/Stage-B configs trong [two-stage runner](../../../../scripts/experiments/run_two_stage_offline_pretraining.py#L120-L171). |

### Consequence of the chosen decision

`_phase_uses_prototype_path()` sẽ có nghĩa đúng với tên: chỉ phase có memory
retrieval mới trả về `True`. Stage A chuyển sang output passthrough; điều này
đã được code hỗ trợ, không cần thêm facade hoặc nhánh mới. Stage B giữ nguyên
prototype path, nên initializer hiện tại tự động cho phép Stage B và chặn Stage
A qua cùng một guard.

`Trainer` vẫn gọi model hook ở đầu mỗi epoch. Trong Stage A, hook trả `False`
ngay tại phase guard và không đọc `train_loader`; trong Stage B, hook ở đầu epoch
sẽ trả `False` nếu memory đã được chuẩn bị trước đó, còn Stage-B orchestration
vẫn là caller chính thực hiện initialization.

## Verified implementation boundary

| File | Symbol/section | Vai trò hiện tại | Hành động dự kiến |
| --- | --- | --- | --- |
| `src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py` | `_phase_uses_prototype_path()` | Cho cả Stage A và B đi vào memory path | Chỉ cho Stage B đi vào memory path; cập nhật comment nếu cần |
| `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py` | `forward()` | Prototype retrieval khi predicate true, passthrough khi false | Không đổi cấu trúc; kiểm chứng Stage A đi vào `else` |
| `src/models/thesis_multitask_impl/thesis_multitask_state_passthrough_mixin.py` | `_build_phase_passthrough_outputs()` | Cung cấp hidden trực tiếp cho fusion và đánh dấu bypass | Giữ nguyên; chỉ bổ sung assertion/test nếu cần |
| `src/models/thesis_multitask_impl/thesis_multitask_state_memory_init_helpers.py` | `maybe_initialize_memories_from_loader()` | Guard phase trước token-pool collection | Không cần thêm phase branch; predicate mới làm guard đúng |
| `src/engine/trainer.py` | epoch loop | Gọi model hook trước batch | Không đổi |
| `scripts/experiments/run_two_stage_offline_pretraining.py` | `_prepare_stage_b_initialization_checkpoint()` | Load Stage A, init memory, lưu Stage-B init checkpoint | Không đổi orchestration; regression test phải chứng minh call thành công |
| `src/models/thesis_multitask_impl/thesis_multitask_state_serialization_mixin.py` | lifecycle load/save | Serialize và restore `memory_initialized` | Không đổi schema; kiểm tra Stage-A false và Stage-B true |

## Phase 0 — Baseline and working-tree safety

### Goal

Ghi nhận trạng thái trước sửa và xác nhận test baseline, đồng thời bảo toàn các
thay đổi đang có trong worktree.

### Changes

Không sửa file.

### Verification

Chạy từng lệnh riêng:

```bash
git status --short
.venv/bin/python -m pytest -q tests/models/test_multitask_memory_bootstrap.py tests/models/test_multitask_memory_initialization.py tests/models/test_multitask_memory_updates.py
.venv/bin/python -m pytest -q tests/runtime/test_checkpoint_roundtrip.py tests/benchmarks/test_two_stage_orchestration_dry_run.py
```

Ghi nhận test nào đang fail trước patch. Không dùng `git reset`, `git checkout`
hoặc thay nguyên file để làm sạch worktree.

### Exit criteria

Có baseline test result và danh sách dirty files; scope của patch không chạm các
thay đổi timing/online đang có.

## Phase 1 — Make Stage A memory-free through the existing phase boundary

### Goal

Đưa Stage A vào nhánh passthrough hiện có và giữ Stage B là phase duy nhất dùng
prototype retrieval.

### Changes

#### `src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py`

- **Symbol:** `_phase_uses_prototype_path()`.
- **Change:** trả `True` chỉ khi `training_phase == TWO_STAGE_B_PHASE_NAME`.
- **State effect:** Stage A không còn đủ điều kiện để gọi initializer, update
  memory, retrieval prototype hoặc chạy optional prototype losses.
- **Forward effect:** `forward()` chọn `_build_phase_passthrough_outputs(hidden)`;
  `hidden` đi trực tiếp vào reconstruction/classification heads.
- **Parameter effect:** `_configure_trainable_parameters_for_phase()` dùng cùng
  predicate để giữ các fusion/prototype modules không train trong Stage A; Stage
  B behavior hiện tại không đổi.

#### `src/models/thesis_multitask_impl/thesis_multitask_state_memory_init_helpers.py`

- **Symbol:** `maybe_initialize_memories_from_loader()`.
- **Change:** không thêm codepath mới. Xác nhận phase guard hiện tại trả `False`
  trước `_collect_memory_initialization_token_pool_from_loader()` ở Stage A.
- **Error behavior:** Stage-B initializer vẫn raise nếu không tạo được memory;
  không nuốt lỗi dữ liệu train rỗng.

#### `src/engine/trainer.py`

- **Symbol:** epoch loop gọi `maybe_initialize_memories_from_loader()`.
- **Change:** không đổi.
- **Reason:** trainer giữ trách nhiệm loop mechanics; model giữ semantics của
  phase và memory lifecycle.

### Verification

- Stage-A model forward trả `memory_bypass_active=True` ở cả hai branch, không có
  `active_continuous_memory_bank` hoặc `active_discrete_codebook`.
- Stage-A direct initializer trả `False`, `memory_initialized` vẫn `False`,
  `memory_ready_for_initialization` vẫn `False`, bank tensors không đổi.
- Stage-B model vẫn trả `_phase_uses_prototype_path() == True` và direct
  initializer với loader nhỏ trả `True`.

### Risks and rollback

- **Risk:** test cũ đang gọi initializer trên model mặc định Stage A sẽ fail vì
  fixture không nói rõ phase. Chuyển fixture của các test initializer/memory
  update sang Stage B; không đổi production default ngoài phase gate.
- **Risk:** Stage-A test có thể mong prototype branch metrics. Cập nhật expectation
  theo contract passthrough, nhưng giữ output keys và loss metrics cần thiết.
- **Rollback:** nếu focused tests cho thấy một caller hợp lệ ngoài two-stage cần
  memory trong Stage A, dừng và kiểm tra lại specification/caller trước khi mở
  thêm phase. Không thêm alias phase theo suy đoán.

## Phase 2 — Align unit and checkpoint-transition tests

### Goal

Test trực tiếp lifecycle mới thay vì chỉ test initializer độc lập hoặc dry-run
ordering.

### Changes

#### `tests/models/test_multitask_memory_bootstrap.py`

- Thay test `test_trainer_initializes_memory_after_bootstrap_window` bằng test
  xác nhận Trainer không khởi tạo memory trong Stage A, kể cả khi
  `bootstrap_encoder_epochs` đã hết.
- Thêm test Stage-B model có thể gọi initializer sau khi bắt đầu ở trạng thái
  `memory_initialized=False`.
- Giữ test forward bootstrap để chứng minh memory tensors không bị mutate.

#### `tests/models/test_multitask_memory_initialization.py`

- Đặt fixture model vào `stage_b_fusion_finetuning` vì các test này kiểm tra
  hành vi build bank, K-means token pool, synthetic class split và freeze sau
  initialization.
- Thêm assertion rằng initializer trên Stage A trả `False` trước khi test
  Stage-B initialization.

#### `tests/models/test_multitask_memory_updates.py`

- Đặt fixture update vào Stage B với
  `freeze_memories_after_initialization=False`, vì đây là test legacy/update
  primitive chứ không phải contract của active Stage-B config.
- Giữ nguyên assertion bank và EMA update; thêm assertion Stage A không update
  nếu test cần phân biệt phase.

#### `tests/runtime/test_checkpoint_roundtrip.py`

- Cập nhật model fixture gọi initializer trực tiếp sang Stage B.
- Bổ sung regression test: tạo Stage-A model chưa initialized, lưu checkpoint
  với `get_checkpoint_extra_state()`, tạo Stage-B model, load model state và
  extra state, gọi initializer trên loader nhỏ, assert thành công và extra state
  sau initialization có `memory_initialized=True`.
- Giữ test round-trip đã xác nhận memory buffers có thể serialize/restore; test
  mới kiểm tra đúng thứ tự lifecycle, không thay thế test cũ.

#### `tests/benchmarks/test_two_stage_orchestration_dry_run.py`

- Giữ dry-run ordering tests.
- Nếu mock boundary hiện tại cho phép, thêm một test gọi helper chuẩn bị Stage-B
  init với fixture checkpoint/data nhỏ; nếu không, đặt test transition ở
  `tests/runtime/test_checkpoint_roundtrip.py` và ghi rõ orchestration helper
  vẫn được kiểm tra bằng smoke ở Phase 3.

### Exit criteria

Test suite chứng minh ba điều độc lập: Stage A không init, Stage B init được,
checkpoint extra state không chặn Stage-B init.

## Phase 3 — End-to-end verification and artifact checks

### Automated verification

Chạy từng nhóm theo thứ tự:

```bash
.venv/bin/python -m pytest -q tests/models/test_multitask_memory_bootstrap.py tests/models/test_multitask_memory_initialization.py tests/models/test_multitask_memory_updates.py
.venv/bin/python -m pytest -q tests/runtime/test_checkpoint_roundtrip.py tests/benchmarks/test_two_stage_orchestration_dry_run.py
.venv/bin/python -m pytest -q tests/models/test_multitask_point_score_loss.py tests/models/test_one_multitask_train_step.py tests/models/test_model_shapes.py
.venv/bin/python -m compileall -q src/engine src/models scripts/experiments
git diff --check
```

Sau đó chạy một smoke O1 đã nêu trong research context, với Stage A hai epoch
và Stage B một epoch. Nếu local không có CUDA, chạy smoke trên remote GPU theo
`ssh-gpu.txt`; không bật `debug_timing` trong run dùng để báo cáo performance.

### Manual verification

Đọc log/artifact của một run và xác nhận:

1. Stage-A log không có message `Marked prototype memories as initialized`.
2. Stage-A checkpoint extra state ghi `memory_initialized: false`.
3. Stage-B preparation có message `Stage B initialization checkpoint ready`.
4. `stage_b_init.pt` có hai bank đã được khởi tạo, metadata initialization và
   `memory_training_enabled: false` khi config freeze là `true`.
5. Stage-B best checkpoint được tạo và runner không còn lỗi
   `Stage B initialization checkpoint could not initialize memories`.

### Artifact and rollback checks

- Chỉ giữ các checkpoint/artifact theo output retention policy hiện hành.
- Không xoá checkpoint Stage A cũ trong plan này; run mới phải có output directory
  riêng hoặc cơ chế skip/resume hiện hành.
- Nếu smoke fail sau Phase 1, so sánh checkpoint extra state và branch metrics
  trước khi rollback. Rollback chỉ là revert patch phase gate và test patch; không
  dùng destructive Git command trên dirty worktree.

## Open questions and non-blocking uncertainty

Không còn blocking question cho patch tối thiểu: `full-spec-v3.md` đã quy định
Stage A không retrieval và memory construction diễn ra sau Stage A
([lifecycle](../../../spec/full-spec-v3.md#L61-L73),
[Stage A](../../../spec/full-spec-v3.md#L430-L464),
[Stage B](../../../spec/full-spec-v3.md#L466-L492)).

Điểm cần kiểm chứng trong implementation nhưng không chặn plan: các test cũ
đang dùng model mặc định Stage A để kiểm tra memory update/initialization. Khi
đổi fixture sang Stage B, phải giữ đúng mục tiêu của từng test và không biến
test primitive thành bằng chứng cho behavior Stage A.

## Out-of-scope follow-up

Sau khi smoke pass, có thể lập detail plan riêng cho việc làm rõ tên
`_phase_uses_prototype_path()` nếu muốn tách “prototype branch” khỏi “memory
retrieval”. Việc đó không thuộc patch tối thiểu hiện tại vì predicate hiện đang
được dùng nhất quán cho forward, initializer, optional losses và trainable
parameter setup.
