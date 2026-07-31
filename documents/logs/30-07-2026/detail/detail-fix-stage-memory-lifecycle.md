---
date: 2026-07-30T18:00:00+07:00
topic: "Sửa memory lifecycle giữa Stage A và Stage B của THESIS offline"
status: ready
revision: 1a570cda6eb7976255add3fc5d4f7f385d40dea3
branch: dev
source_structure: documents/logs/30-07-2026/structure/structure-fix-stage-memory-lifecycle.md
related_documents:
  - documents/logs/30-07-2026/research/research-stage-memory-lifecycle-mismatch.md
  - documents/logs/30-07-2026/plan/plan-fix-stage-memory-lifecycle.md
---

# Detailed Implementation: Sửa memory lifecycle giữa Stage A và Stage B

## Summary

Implementation sẽ sửa một phase gate trong THESIS multitask model. Stage A sẽ
dùng encoder và task heads qua nhánh passthrough, không khởi tạo hoặc truy xuất
continuous/discrete memory. Stage B sẽ tiếp tục dùng prototype path và khởi tạo
memory sau khi tải checkpoint Stage A.

Tài liệu này mở rộng [approved structure](../structure/structure-fix-stage-memory-lifecycle.md)
và giữ nguyên dependency order của [implementation plan](../plan/plan-fix-stage-memory-lifecycle.md).
Nhiệm vụ này chỉ viết hướng dẫn triển khai; không sửa source, config, test hoặc
specification và không tạo commit.

## Source structure

Structure gồm bốn pha:

1. khóa boundary và baseline;
2. sửa phase gate để Stage A bypass memory;
3. đồng bộ test và checkpoint transition;
4. chạy local gates, O1 smoke và artifact audit.

Phase 2 phụ thuộc Phase 1. Phase 3 chỉ sửa test sau khi behavior boundary đã rõ.
Phase 4 chỉ chạy sau khi checkpoint transition test pass.

## Current state

`Trainer.train()` gọi `maybe_initialize_memories_from_loader()` trước vòng lặp
batch ở mỗi epoch [tại Trainer](../../../../src/engine/trainer.py#L565-L596).

Model hiện coi cả Stage A và Stage B là prototype path
[tại phase predicate](../../../../src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py#L50-L58).
Vì vậy initializer đi qua ba guard, collect token pool, build hai bank và đánh
dấu model initialized [tại initializer](../../../../src/models/thesis_multitask_impl/thesis_multitask_state_memory_init_helpers.py#L18-L44).

Forward chỉ gọi prototype lookup khi predicate trả `True`; nếu predicate trả
`False`, forward dùng `_build_phase_passthrough_outputs(hidden)`
[tại forward](../../../../src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py#L135-L234)
và [tại passthrough helper](../../../../src/models/thesis_multitask_impl/thesis_multitask_state_passthrough_mixin.py#L9-L60).

Stage-B runner tải Stage-A model state và extra state, rồi gọi initializer
[tại preparation flow](../../../../scripts/experiments/run_two_stage_offline_pretraining.py#L254-L291).
Extra state khôi phục `memory_initialized` từ checkpoint
[tại lifecycle restore](../../../../src/models/thesis_multitask_impl/thesis_multitask_state_serialization_mixin.py#L172-L228).

Spec v3 yêu cầu Stage A chạy trước memory construction và không dùng memory
retrieval [tại lifecycle](../../../spec/full-spec-v3.md#L61-L73) và
[Stage A contract](../../../spec/full-spec-v3.md#L430-L464). Config smoke hiện
dùng hai epoch cho Stage A và một epoch cho Stage B
[tại O1 smoke config](../../../../configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O1__machine_1_6__w20__seed6__smoke.yaml#L45-L50).

## Desired end state

```text
Stage A model
  training_phase = stage_a_multitask_pretraining
  _phase_uses_prototype_path() = False
  forward -> phase_direct_passthrough
  initializer -> False before token-pool collection
  checkpoint extra_state.memory_initialized = False

Stage B model
  training_phase = stage_b_fusion_finetuning
  load Stage-A state and extra_state
  initializer -> collect train tokens -> build two banks -> mark initialized
  freeze memory updates according to config
  save stage_b_init.pt
  train Stage-B fusion/prediction heads
```

Observable contracts:

- Stage-A model vẫn trả các output keys hiện hành, gồm `recon`, `logits`,
  `point_scores`, `window_scores` và các branch dictionaries.
- Stage-A `active_continuous_memory_bank` và `active_discrete_codebook` là
  `None`; branch aux đánh dấu `memory_bypass_active=True`.
- Stage-A initializer không đọc `train_loader` sau khi phase guard được kiểm tra.
- Stage-B initializer vẫn tạo continuous bank và discrete codebook từ train
  loader, sau đó đặt `memory_initialized=True`.
- Stage-B initialization không còn bị chặn bởi Stage-A `extra_state`.

## Scope

### In scope

- `ThesisMultitaskSetupMixin._phase_uses_prototype_path()`.
- Các test trực tiếp kiểm tra initializer, memory update, checkpoint round-trip
  và two-stage orchestration.
- Local/remote verification và artifact lifecycle audit.

### Out of scope

- `Trainer` epoch-loop semantics.
- `maybe_initialize_memories_from_loader()` algorithm và token-pool selection.
- K-means, codebook construction, loss formula, synthetic augmentation và
  Stage-B freeze implementation.
- Config files, checkpoint schema, online TTA và full benchmark matrix.
- Xóa memory buffers khỏi model `state_dict`.

## Evidence

- [Trainer calls the hook before training batches](../../../../src/engine/trainer.py#L565-L596) — engine không tự phân biệt Stage A/B.
- [Setup initializes lifecycle flags](../../../../src/models/thesis_multitask_impl/thesis_multitask_setup_helpers.py#L147-L159) — two-stage model bắt đầu với `memory_initialized=False`.
- [Current phase predicate](../../../../src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py#L50-L58) — lỗi hiện tại nằm ở việc cho Stage A vào prototype path.
- [Initializer guards and state mutation](../../../../src/models/thesis_multitask_impl/thesis_multitask_state_memory_init_helpers.py#L18-L44) — phase predicate là guard trước token-pool collection.
- [Passthrough branch](../../../../src/models/thesis_multitask_impl/thesis_multitask_state_passthrough_mixin.py#L9-L60) — repository đã có output path cho phase không retrieval.
- [Checkpoint extra-state construction](../../../../src/models/thesis_multitask_impl/thesis_multitask_state_serialization_mixin.py#L77-L87) — lifecycle fields được ghi vào checkpoint.
- [Stage-B initializer caller](../../../../scripts/experiments/run_two_stage_offline_pretraining.py#L275-L291) — runner yêu cầu initializer trả `True`.
- [Spec lifecycle](../../../spec/full-spec-v3.md#L61-L73) — memory construction nằm sau Stage A.

## Phase 1: Boundary và baseline được khóa

### Goal

Xác định chính xác các file/caller sẽ thay đổi và ghi nhận test baseline trước
khi sửa behavior.

### Dependencies

- Research note đã hoàn tất.
- [Implementation plan](../plan/plan-fix-stage-memory-lifecycle.md) đã chọn
  phase gate tại model.
- Live worktree phải được bảo toàn; không dùng destructive Git command.

### Detailed changes

#### 1. Tạo inventory của phase và memory lifecycle symbols

- **File:** `src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py`
- **Symbol:** `_phase_uses_prototype_path()`.
- **Current responsibility:** Chọn prototype path cho Stage A và Stage B.
- **Change:** Chưa sửa. Xác nhận definition và tất cả callers trước patch.
- **Reason:** Đây là boundary trung tâm được forward, initializer, loss và
  trainable-parameter setup dùng chung.
- **Inputs:** `training_phase`.
- **Outputs:** Boolean phase decision.
- **Errors:** Nếu có caller ngoài inventory, dừng và phân loại trước khi sửa.
- **Dependencies:** Forward, initializer, optional losses và setup.
- **Compatibility:** Không đổi tên phase constants hoặc config keys.
- **Verification:**

```bash
rg -n "_phase_uses_prototype_path|maybe_initialize_memories_from_loader|memory_initialized|training_phase" src tests scripts configs
```

Expected: caller graph khớp research note; không có phase thứ ba cần giữ.

#### 2. Bảo toàn dirty worktree

- **File:** repository worktree.
- **Symbol:** Các thay đổi có sẵn về online timing, stream range và TTLBuffer.
- **Current responsibility:** Đây là công việc ngoài scope của memory lifecycle.
- **Change:** Chưa sửa; ghi lại `git status --short` và các diff liên quan.
- **Reason:** Không để plan này vô tình yêu cầu overwrite thay đổi của anh.
- **Inputs:** Live checkout.
- **Outputs:** Danh sách dirty files dùng cho final scope audit.
- **Errors:** Nếu source memory files đã bị sửa ngoài phạm vi research, đọc diff
  trước khi tiếp tục.
- **Dependencies:** Tất cả phase sau.
- **Compatibility:** Giữ nguyên các thay đổi không thuộc memory lifecycle.
- **Verification:**

```bash
git status --short
```

#### 3. Chạy baseline tests

- **File:** Không sửa file.
- **Symbol:** Memory initialization, memory updates, checkpoint và dry-run
  orchestration tests.
- **Current responsibility:** Xác lập behavior trước patch.
- **Change:** Chạy test, không chỉnh failure trong phase này.
- **Reason:** Phân biệt regression mới với failure có sẵn.
- **Inputs:** Current source và current test fixtures.
- **Outputs:** Baseline test result.
- **Errors:** Ghi rõ failure; không mở rộng scope để làm suite xanh.
- **Dependencies:** Không có.
- **Compatibility:** Không thay output hoặc checkpoint.
- **Verification:**

```bash
.venv/bin/python -m pytest -q tests/models/test_multitask_memory_bootstrap.py tests/models/test_multitask_memory_initialization.py tests/models/test_multitask_memory_updates.py
```

```bash
.venv/bin/python -m pytest -q tests/runtime/test_checkpoint_roundtrip.py tests/benchmarks/test_two_stage_orchestration_dry_run.py
```

### Tests

Không thêm test ở phase này. Các test mới thuộc Phase 3 sau khi behavior gate
đã được sửa.

### Verification

#### Automated

- [ ] Exact symbol search hoàn tất và caller list khớp research note.
- [ ] Baseline pytest result được ghi nhận.

#### Manual

- [ ] Đọc diff của các file dirty để chắc chắn implementation sau này không
  overwrite timing/stream-range/TTLBuffer work.

### Risks and recovery

- **Risk:** Baseline fail sẵn làm kết quả sau patch bị hiểu sai.
- **Mitigation:** Ghi riêng baseline failure và chỉ so sánh cùng test command.
- **Verification:** Chạy lại đúng command sau mỗi phase.
- **Recovery:** Dừng trước Phase 2 nếu source boundary không còn khớp research.

### Complete when

- Inventory có definition và use của phase gate, initializer, passthrough,
  serialization và Stage-B caller.
- Dirty worktree được ghi nhận.
- Baseline tests có trạng thái rõ ràng.

## Phase 2: Stage A bypass memory, Stage B giữ prototype path

### Goal

Đổi một phase predicate để Stage A không đi vào retrieval hoặc initialization,
trong khi Stage B giữ nguyên prototype flow.

### Dependencies

- Phase 1 đã xác nhận không có caller/phase khác ngoài Stage A và Stage B.
- Passthrough output path đã tồn tại trong model.

### Detailed changes

#### 1. Giới hạn prototype path ở Stage B

- **File:** `src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py`
- **Symbol:** `ThesisMultitaskSetupMixin._phase_uses_prototype_path()`.
- **Current responsibility:** Trả `True` cho cả
  `TWO_STAGE_A_PHASE_NAME` và `TWO_STAGE_B_PHASE_NAME`.
- **Change:** Trả `True` chỉ khi `self.training_phase == TWO_STAGE_B_PHASE_NAME`.
  Giữ nguyên phase constants và kiểu trả về `bool`.
- **Reason:** Spec đặt memory construction sau Stage A; predicate này được dùng
  làm boundary chung cho forward, initializer và optional prototype losses.
- **Inputs:** `self.training_phase` đã được lưu từ model config
  [tại setup helper](../../../../src/models/thesis_multitask_impl/thesis_multitask_setup_helpers.py#L120-L138).
- **Outputs:**
  - Stage A: `False`;
  - Stage B: `True`.
- **Errors:** Không thêm fallback phase hoặc alias mới. Config phase không hợp lệ
  vẫn bị `ActiveRuntimeConfig` từ chối
  [tại phase validation](../../../../src/models/thesis_multitask_impl/thesis_multitask_components.py#L210-L232).
- **Dependencies:** `forward()`, `_should_bypass_memory_for_stage()`,
  `maybe_initialize_memories_from_loader()` và
  `_configure_trainable_parameters_for_phase()` dùng predicate này.
- **Compatibility:** Stage-B retrieval, Stage-B freeze và output schema phải giữ.

#### 2. Không thay Trainer và initializer algorithm

- **File:** `src/engine/trainer.py`.
- **Symbol:** Epoch loop tại `Trainer.train()`.
- **Current responsibility:** Gọi model hook trước training batches.
- **Change:** Không sửa.
- **Reason:** Trainer chỉ điều phối loop; model quyết định phase semantics.
- **Inputs:** Model hook và train loader như hiện tại.
- **Outputs:** Stage A hook trả `False` trước khi đọc loader; Stage B runner vẫn
  gọi initializer rõ ràng.
- **Errors:** Không nuốt `ValueError` do train loader không có normal token.
- **Dependencies:** Phase predicate.
- **Compatibility:** Các model khác có hook memory vẫn dùng Trainer contract cũ.

- **File:** `src/models/thesis_multitask_impl/thesis_multitask_state_memory_init_helpers.py`.
- **Symbol:** `maybe_initialize_memories_from_loader()`.
- **Current responsibility:** Trả `False` nếu initialized/bootstrap/non-prototype;
  nếu không thì collect token pool và mark initialized.
- **Change:** Không thêm phase branch. Predicate mới phải làm Stage A dừng ở
  `if not model._phase_uses_prototype_path(): return False`.
- **Reason:** Tránh duplicate lifecycle rule.
- **Inputs:** Stage A/B model, train loader, device.
- **Outputs:** Stage A `False`; Stage B `True` sau khi build bank.
- **Errors:** Stage B vẫn báo lỗi nếu pool không có normal token.
- **Dependencies:** Model phase predicate và `mark_memories_initialized()`.
- **Compatibility:** Memory initialization data source và freeze behavior không đổi.

#### 3. Giữ passthrough output path

- **File:** `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py`.
- **Symbol:** `forward()`.
- **Current responsibility:** Prototype lookup khi predicate true; passthrough khi
  predicate false.
- **Change:** Không đổi nhánh forward. Chỉ kiểm chứng Stage A đi vào `else`.
- **Reason:** Repository đã có implementation phù hợp với Stage A memory-free.
- **Inputs:** Encoder hidden state.
- **Outputs:** `continuous_outputs`, `discrete_outputs`, `fusion_outputs` từ
  `_build_phase_passthrough_outputs()`; heads vẫn tạo recon/logits/scores.
- **Errors:** Giữ output validation hiện tại.
- **Dependencies:** `ThesisMultitaskStatePassthroughMixin`.
- **Compatibility:** Output keys và shape contract không đổi.

### Tests

#### Stage A phase behavior

- **Location:** `tests/models/test_multitask_memory_bootstrap.py`.
- **Level:** Unit.
- **Setup:** Build model với `training_phase=stage_a_multitask_pretraining`,
  `bootstrap_encoder_epochs=0`, loader nhỏ và clone memory tensors.
- **Action:** Gọi `maybe_initialize_memories_from_loader()` và một
  `training_step()`.
- **Expected result:** Initializer trả `False`; bank tensors và lifecycle flags
  không đổi; forward aux đánh dấu bypass; active banks là `None`.
- **Edge cases:** `bootstrap_encoder_epochs > 0` vẫn bypass; Stage A không đọc
  loader khi phase guard đã trả `False`.

#### Stage B phase behavior

- **Location:** `tests/models/test_multitask_memory_initialization.py`.
- **Level:** Unit.
- **Setup:** Build model với `training_phase=stage_b_fusion_finetuning`, loader
  chứa normal tokens, `freeze_memories_after_initialization=True`.
- **Action:** Gọi initializer.
- **Expected result:** Trả `True`, tạo hai bank, đặt `memory_initialized=True`,
  `memory_training_enabled=False`.
- **Edge cases:** Loader thiếu normal tokens vẫn raise `ValueError` như hiện tại.

### Verification

#### Automated

- [ ] Chạy hai test behavior ở trên.
- [ ] Chạy `compileall` cho model và Trainer.

```bash
.venv/bin/python -m pytest -q tests/models/test_multitask_memory_bootstrap.py tests/models/test_multitask_memory_initialization.py
```

```bash
.venv/bin/python -m compileall -q src/engine/trainer.py src/models/thesis_multitask_impl
```

#### Manual

- [ ] Đọc model output trong test failure/debug nếu có để xác nhận Stage A dùng
  `phase_direct_passthrough`, không chỉ vì một flag tên `memory_initialized`.

### Risks and recovery

- **Risk:** Đổi predicate làm Stage-A fusion/prototype modules có
  `requires_grad` khác trước.
- **Mitigation:** Kiểm tra `_configure_trainable_parameters_for_phase()` và giữ
  Stage A trainable surface phù hợp với passthrough; không thêm module mới.
- **Verification:** One-train-step test và gradient/parameter assertions hiện có.
- **Recovery:** Nếu Stage-A loss shape lỗi, dừng và xem lại output contract của
  passthrough; không bật lại memory retrieval để che lỗi.

### Complete when

Stage A bypass hoàn toàn memory path, Stage B vẫn init/retrieve được và focused
model tests pass.

## Phase 3: Checkpoint và test contract đồng bộ

### Goal

Chứng minh checkpoint Stage A truyền đúng lifecycle chưa initialized sang Stage B
và Stage-B initializer hoạt động sau khi load.

### Dependencies

- Phase 2 đã đổi phase gate và focused model tests pass.
- Checkpoint manager vẫn giữ state/extra-state contracts hiện hành.

### Detailed changes

#### 1. Phân loại lại memory initialization fixtures

- **File:** `tests/models/test_multitask_memory_initialization.py`.
- **Symbols:** `_build_initialization_model()` và các test initializer.
- **Current responsibility:** Fixture không chỉ rõ phase nhưng gọi initializer
  trực tiếp.
- **Change:** Set `training_phase=stage_b_fusion_finetuning` vì test mục tiêu là
  xây memory bank, K-means và freeze sau initialization.
- **Reason:** Sau fix, Stage A direct initializer phải bị chặn.
- **Inputs:** Existing raw/synthetic batches.
- **Outputs:** Existing bank shapes, norms, source labels và lifecycle state.
- **Errors:** Test phải tiếp tục phát hiện empty normal token pool.
- **Dependencies:** Phase predicate.
- **Compatibility:** Không làm yếu assertion về bank values hoặc metadata.

#### 2. Chuyển memory-update primitive tests sang Stage B

- **File:** `tests/models/test_multitask_memory_updates.py`.
- **Symbol:** `_build_initialized_model()`.
- **Current responsibility:** Build default-phase model rồi gọi initializer để
  test continuous/discrete EMA updates.
- **Change:** Set `training_phase=stage_b_fusion_finetuning` và
  `freeze_memories_after_initialization=False` trong fixture.
- **Reason:** Test update primitive không còn phù hợp với Stage-A contract;
  Stage B là phase có prototype path.
- **Inputs:** Existing initialization batch and update batches.
- **Outputs:** Existing assertions that bank/EMA state changes in train and not
  in validation/test.
- **Errors:** Không đổi update gating hoặc EMA algorithm.
- **Dependencies:** Phase 2 predicate.
- **Compatibility:** Giữ nguyên mục tiêu test memory update.

#### 3. Sửa bootstrap/trainer tests theo lifecycle mới

- **File:** `tests/models/test_multitask_memory_bootstrap.py`.
- **Symbols:** `_build_model()`, `test_trainer_initializes_memory_after_bootstrap_window`.
- **Current responsibility:** Test cũ mong Trainer tự initialize sau bootstrap.
- **Change:**
  1. Cho `_build_model()` nhận `training_phase` khi test cần phân biệt phase.
  2. Thay test cũ bằng Stage-A test xác nhận Trainer không initialize memory dù
     bootstrap window đã hết.
  3. Thêm Stage-B test xác nhận initializer trực tiếp hoạt động với
     `memory_initialized=False`.
- **Reason:** Active two-stage spec không cho Stage A khởi tạo memory.
- **Inputs:** Existing lightweight loader and `Trainer` fixture.
- **Outputs:** Stage-A state unchanged; Stage-B init succeeds.
- **Errors:** Không coi `False` của Stage-A hook là test failure.
- **Dependencies:** `Trainer.train()` và phase predicate.
- **Compatibility:** Giữ test forward bootstrap và memory tensor immutability.

#### 4. Thêm checkpoint transition regression

- **File:** `tests/runtime/test_checkpoint_roundtrip.py`.
- **Symbols:** Existing checkpoint tests and a new lifecycle regression test in
  the same file.
- **Current responsibility:** Test round-trip memory state và generic checkpoint
  metadata; chưa test Stage-A checkpoint -> Stage-B initializer.
- **Change:**
  1. Những test gọi initializer trực tiếp phải dùng Stage-B fixture.
  2. Tạo Stage-A model với `training_phase=stage_a_multitask_pretraining` và
     chưa initialized.
  3. Lưu checkpoint với `model.get_checkpoint_extra_state()`.
  4. Tạo Stage-B model, load `model_state_dict` và `extra_state` bằng cùng
     checkpoint contract.
  5. Gọi initializer trên loader nhỏ.
  6. Assert initializer trả `True`, hai bank có shape đúng và lifecycle state
     sau init là initialized/frozen theo config.
- **Reason:** Đây là regression cho lỗi cross-stage đã xảy ra thật.
- **Inputs:** Small deterministic batch, CPU device, temporary checkpoint path.
- **Outputs:** Stage-A extra state false; Stage-B extra state true after init.
- **Errors:** Nếu Stage-B init trả `False`, test phải fail với lifecycle context.
- **Dependencies:** `CheckpointManager`, `load_checkpoint_extra_state()` và
  initializer.
- **Compatibility:** Giữ test cũ xác nhận memory buffers có thể round-trip.

#### 5. Giữ orchestration dry-run tests và bổ sung boundary nếu phù hợp

- **File:** `tests/benchmarks/test_two_stage_orchestration_dry_run.py`.
- **Symbols:** Existing `execute_two_stage_plan()` and command-plan tests.
- **Current responsibility:** Kiểm tra thứ tự stage và module invocation, không
  chạy checkpoint transition thật.
- **Change:** Giữ test dry-run. Chỉ thêm mock/integration test tại đây nếu có thể
  gọi `_prepare_stage_b_initialization_checkpoint()` với fixture nhỏ mà không
  phụ thuộc dataset lớn; nếu không, dùng checkpoint regression ở bước 4 và
  để end-to-end runner smoke kiểm tra orchestration.
- **Reason:** Không làm dry-run test thành test phụ thuộc CUDA/dataset thật.
- **Inputs:** Existing manifest fixtures.
- **Outputs:** Stage order và command construction không đổi.
- **Errors:** Không sửa expected stage names để che lỗi runner.
- **Dependencies:** Stage-B preparation function.
- **Compatibility:** Dry-run không tạo checkpoint hoặc thay đổi output.

### Tests

#### Checkpoint lifecycle transition

- **Location:** `tests/runtime/test_checkpoint_roundtrip.py`.
- **Level:** Integration-style unit test.
- **Setup:** Temporary checkpoint directory; Stage-A and Stage-B model configs;
  deterministic loader.
- **Action:** Save Stage-A payload, load state/extra state into Stage-B, call
  initializer.
- **Expected result:** Stage-A extra state false; Stage-B init true; bank tensors
  initialized; freeze flag respected.
- **Edge cases:** Missing/empty train tokens must continue raising the existing
  initialization error.

### Verification

#### Automated

- [ ] Memory initialization, bootstrap và update tests pass.
- [ ] Checkpoint round-trip và two-stage dry-run tests pass.

```bash
.venv/bin/python -m pytest -q tests/models/test_multitask_memory_bootstrap.py tests/models/test_multitask_memory_initialization.py tests/models/test_multitask_memory_updates.py
```

```bash
.venv/bin/python -m pytest -q tests/runtime/test_checkpoint_roundtrip.py tests/benchmarks/test_two_stage_orchestration_dry_run.py
```

### Risks and recovery

- **Risk:** Một fixture được chuyển sang Stage B nhưng vô tình đổi mục tiêu test.
- **Mitigation:** Giữ tên test và assertion gốc; chỉ thêm phase/freeze values cần
  cho mục tiêu test.
- **Verification:** Review diff theo từng fixture và chạy test file riêng.
- **Recovery:** Tách fixture Stage-A bypass và Stage-B memory primitive, không
  khôi phục initializer vào Stage A.

### Complete when

Test suite chứng minh độc lập: Stage A không init, Stage B init được và
checkpoint extra state không chặn Stage-B initialization.

## Phase 4: End-to-end verification và artifact audit

### Goal

Chứng minh pipeline O1 two-stage chạy hết sau khi sửa lifecycle và artifact phản
ánh đúng boundary giữa hai stage.

### Dependencies

- Phase 3 tests pass.
- Có config O1 smoke hợp lệ và quyền chạy remote GPU nếu local thiếu CUDA.
- Không bật `debug_timing: true` trong run dùng để báo cáo performance.

### Detailed changes

Không thêm source change ở phase này. Chỉ chạy verification và đọc artifact.

#### 1. Static/local gate

- **File:** `src/engine/trainer.py`, `src/models/thesis_multitask_impl/`,
  `scripts/experiments/run_two_stage_offline_pretraining.py`.
- **Symbol:** Import graph và lifecycle call graph.
- **Current responsibility:** Runtime source sau Phase 2/3.
- **Change:** Không sửa; kiểm tra compile, whitespace và diff scope.
- **Reason:** Phân biệt implementation failure với environment/smoke failure.
- **Inputs:** Current worktree.
- **Outputs:** Clean compile and test evidence.
- **Errors:** Dừng trước remote nếu import/targeted tests fail.
- **Dependencies:** Phase 3.
- **Compatibility:** Không chạm dirty worktree ngoài patch memory lifecycle.
- **Verification:**

```bash
.venv/bin/python -m pytest -q tests/models/test_multitask_point_score_loss.py tests/models/test_one_multitask_train_step.py tests/models/test_model_shapes.py
```

```bash
.venv/bin/python -m compileall -q src/engine src/models scripts/experiments
```

```bash
git diff --check
```

#### 2. O1 two-stage smoke

- **File:** `configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O1__machine_1_6__w20__seed6__smoke.yaml`.
- **Symbol:** Two-stage epoch budget and output directory.
- **Current responsibility:** O1 smoke config đặt Stage A hai epoch, Stage B một
  epoch [tại config](../../../../configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O1__machine_1_6__w20__seed6__smoke.yaml#L45-L50).
- **Change:** Không đổi config. Dùng config hiện có để chạy smoke.
- **Reason:** Smoke phải kiểm tra đúng pipeline đã gây lỗi.
- **Inputs:** O1 config, remote checkout từ `ssh-gpu.txt`, CUDA device nếu có.
- **Outputs:** Stage-A best checkpoint, `stage_b_init.pt`, Stage-B best checkpoint,
  manifest, execution report và evaluation artifacts.
- **Errors:** Nếu Stage-B initializer vẫn trả false, giữ log và dừng; không tự
  sửa config hoặc bỏ qua stage.
- **Dependencies:** Phase 3 pass và remote environment preflight.
- **Compatibility:** Không dùng `debug_timing` cho performance report.
- **Verification command:** Dùng entrypoint hiện hành với config trên:

```bash
.venv/bin/python scripts/experiments/run_two_stage_offline_pretraining.py --experiment-config configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O1__machine_1_6__w20__seed6__smoke.yaml
```

Nếu local không có CUDA, chạy cùng command trên remote host sau khi đọc và xác
nhận `ssh-gpu.txt`. Không mở rộng thành full benchmark matrix.

#### 3. Artifact lifecycle audit

- **File:** Generated stage manifest, Stage-A best checkpoint,
  `two_stage/initializations/stage_b_init.pt`, Stage-B best checkpoint.
- **Symbol:** `extra_state`, lifecycle fields, stage names và execution report.
- **Current responsibility:** Persist stage provenance and checkpoint state.
- **Change:** Không sửa artifact bằng tay; chỉ đọc và đối chiếu.
- **Reason:** Xác nhận runtime behavior chứ không chỉ unit test.
- **Inputs:** Smoke output directory.
- **Outputs:** Audit result.
- **Errors:** Artifact thiếu hoặc sai provenance là smoke failure; không dùng
  artifact cũ để kết luận pass.
- **Dependencies:** Smoke completed.
- **Compatibility:** Giữ output layout và checkpoint naming hiện hành.
- **Verification:** Kiểm tra:
  1. Stage-A checkpoint có `memory_initialized=False`.
  2. Log Stage A không có `Marked prototype memories as initialized`.
  3. Stage-B initialization log có `Stage B initialization checkpoint ready`.
  4. `stage_b_init.pt` có hai bank, metadata initialization và
     `memory_training_enabled=False` khi freeze config là `true`.
  5. Stage-B best checkpoint và execution report tồn tại.
  6. Không còn lỗi `Stage B initialization checkpoint could not initialize memories`.

### Tests

- **Location:** Remote smoke output and local regression groups.
- **Level:** End-to-end.
- **Setup:** O1 `machine_1_6`, seed 6, Stage A 2 epochs, Stage B 1 epoch.
- **Action:** Run two-stage runner and inspect generated artifacts.
- **Expected result:** Full offline flow completes.
- **Edge cases:** Local CUDA unavailable; run remotely. Existing output may be
  present; use a new exact output target or an explicitly verified skip/resume
  path, never mix old Stage-A checkpoint with new source without provenance.

### Verification

#### Automated

- [ ] Focused model/checkpoint tests pass.
- [ ] Model, Trainer và experiment script compile.
- [ ] O1 smoke exits successfully.

#### Manual

- [ ] Compare stage names, config path, checkpoint paths and lifecycle fields in
  manifest/report.
- [ ] Confirm no unrelated dirty-worktree file changed.

### Risks and recovery

- **Risk:** Smoke accidentally resumes an old failed run.
- **Mitigation:** Verify output directory, manifest timestamp, config path and
  checkpoint provenance before accepting result.
- **Verification:** Read execution report and checkpoint metadata.
- **Recovery:** Stop the run, isolate a new output directory, and rerun only the
  exact O1 smoke. Do not delete broad output trees.

### Complete when

O1 smoke completes Stage A, Stage-B initialization, Stage B and evaluation; the
artifacts prove Stage A was memory-free and Stage B initialized/froze memory.

## Interface and data changes

### Model phase contract

| Phase | `_phase_uses_prototype_path()` | Forward path | Initializer |
| --- | --- | --- | --- |
| Stage A | `False` | `phase_direct_passthrough` | Returns `False` before loader collection |
| Stage B | `True` | Continuous/discrete prototype retrieval | Builds banks and returns `True` once |

No public method signature changes. `Trainer` continues to call the same model
hook, and the hook remains responsible for phase-specific meaning.

### Checkpoint contract

No schema field is added or removed. The intended state changes are:

- Stage-A best checkpoint: `memory_initialized=False`;
- Stage-B initialization checkpoint: `memory_initialized=True`;
- Stage-B freeze config: `memory_training_enabled=False` after
  `mark_memories_initialized()`.

The model buffers still exist in `state_dict` at construction. Stage B initializer
overwrites their values from train-token statistics; this plan does not remove
those buffers from serialization.

## Deployment and rollout

1. Run Phase 1 baseline locally.
2. Apply only the phase-gate source change in Phase 2.
3. Run Phase 2 focused model tests.
4. Update and run Phase 3 tests.
5. Run local static/regression gates.
6. Run one O1 smoke on remote GPU if local CUDA is unavailable.
7. Only after smoke pass may a broader benchmark be considered in a separate
   task.

There is no feature flag or mixed-version migration. Old checkpoints should not
be used as scientific evidence unless their lifecycle fields and source/config
provenance are explicitly inspected.

## Documentation changes

- This structure file records phase/stage order.
- This detail file records file-level implementation instructions.
- No specification history or runtime documentation is changed by the code fix.

## Final verification

- [ ] Stage A never calls memory initialization or prototype retrieval.
- [ ] Stage A checkpoint records `memory_initialized=False`.
- [ ] Stage B initializes both banks after loading Stage-A state.
- [ ] Stage B freezes encoder/memory behavior according to existing config.
- [ ] Existing output keys, loss path and checkpoint schema remain compatible.
- [ ] Focused tests, compile check and O1 smoke pass.
- [ ] No unrelated worktree change is overwritten.

## Assumptions and non-blocking uncertainties

- The current active model accepts only Stage A and Stage B phase names, so no
  third phase needs a compatibility decision
  [tại config validation](../../../../src/models/thesis_multitask_impl/thesis_multitask_components.py#L210-L232).
- Some existing memory primitive tests use the default Stage-A phase implicitly;
  the implementer must make the intended phase explicit in each fixture.
- Remote host details, CUDA visibility and output root must be re-read from
  `ssh-gpu.txt` at execution time; this detail document does not hard-code them.
