---
date: 2026-08-31 Asia/Ho_Chi_Minh
topic: "Implement direct branch routing checkpoint bridge"
status: implemented
revision: 6cd32ac94cc20876828f5c62fd2df5c3ac557587
source_structure: documents/logs/2026-08-31/structure/structure-direct-branch-routing-bridge.md
related_documents:
  - documents/logs/2026-08-31/plan/plan-direct-branch-routing-bridge.md
  - documents/logs/2026-08-31/research/research-direct-branch-routing-bridge-flow.md
---

# Detailed Implementation: Direct Branch Routing Checkpoint Bridge

## Summary

Implementer sẽ thêm một bridge riêng cho Stage A `best.pt` → Stage B direct `stage_b_init.pt`, sau đó nối full/smoke runner vào output này. Mọi bước bên dưới chạy tuần tự; bước sau chỉ bắt đầu khi bước trước đã pass.

## Source structure

Tài liệu structure định nghĩa bốn pha: khóa contract bằng test, triển khai bridge, nối runner, rồi kiểm tra local end-to-end. Tài liệu này giữ nguyên thứ tự đó.

## Current state

- `_load_stage_a_state_into_stage_b_model()` đã dùng `strict=False` với allowlist hai khóa `discrete_assignment.*`.
- `_prepare_stage_b_initialization_checkpoint()` đã nạp `extra_state`, khởi tạo memory banks và lưu payload, nhưng nhận manifest two-stage.
- `run_direct_branch_routing_full.py` còn đưa Stage A `best.pt` trực tiếp vào Stage B.
- `CheckpointManager.load_checkpoint()` mặc định dùng `strict=True`.

## Desired end state

Bridge nhận đúng ba đầu vào logic: config Stage B direct, Stage A source checkpoint và output checkpoint. Bridge tạo model Stage B, nạp state tương thích, nạp extra state, khởi tạo memory banks từ train loader, lưu output và để Stage B load strict.

## Scope

### In scope

- `scripts/experiments/run_two_stage_offline_pretraining.py`
- `scripts/experiments/run_direct_branch_routing_bridge.py` (file mới)
- `scripts/run_direct_branch_routing_full.py`
- `scripts/run_direct_branch_routing_smoke.py`
- `tests/experiments/test_direct_branch_routing_bridge.py` (file mới)
- Test runner/config liên quan khi assertion path thay đổi.

### Out of scope

- `src/models` direct routing đã chạy đúng.
- Stage A training, two-stage orchestration trong direct flow, online flow và cloud execution.

## Evidence

- `scripts/experiments/run_two_stage_offline_pretraining.py:224-241` — allowlist mismatch.
- `scripts/experiments/run_two_stage_offline_pretraining.py:244-337` — các thao tác bridge hiện có.
- `src/models/thesis_multitask_impl/thesis_multitask_state_memory_init_helpers.py:18-43` — memory banks cần token normal và được đánh dấu initialized.
- `src/engine/checkpoint.py:277-313` — strict load và extra state.
- `scripts/run_direct_branch_routing_full.py:43-83` — identity và path hiện tại.

## Phase 1: Contract bridge được kiểm thử

### Goal

Test fail trước, rồi test pass sau khi bridge được thêm.

### Dependencies

- Research note bridge và source symbols ở phần Evidence.

### Sequential stages and atomic steps

#### Stage 1.1 — Tạo fixture payload

- [ ] Tạo file test mới `tests/experiments/test_direct_branch_routing_bridge.py`.
- [ ] Tạo model Stage A nhỏ trên CPU trong fixture.
- [ ] Tạo model Stage B với `training_phase=stage_b_fusion_finetuning`.
- [ ] Tạo payload có `model_state_dict`, `extra_state`, `config` và `checkpoint_metadata`.
- [ ] Ghi payload vào file dưới `tmp_path`.
- [ ] Tạo train loader fixture có ít nhất một normal token.

#### Stage 1.2 — Viết test mismatch

- [ ] Thêm test gọi bridge với `discrete_assignment.weight` là key thừa.
- [ ] Thêm test gọi bridge với `discrete_assignment.bias` là key thừa.
- [ ] Thêm test gọi bridge với `unexpected.weight` là key thừa.
- [ ] Khẳng định test key ngoài allowlist nhận `RuntimeError`.

#### Stage 1.3 — Chạy test đỏ

- [ ] Chạy `.venv/bin/python -m pytest -q tests/experiments/test_direct_branch_routing_bridge.py`.
- [ ] Ghi nhận failure do hàm bridge chưa tồn tại.
- [ ] Xác nhận test không gọi `execute_two_stage_plan()`.

Stage 1.2 phụ thuộc Stage 1.1. Stage 1.3 phụ thuộc Stage 1.2.

### Detailed changes

#### 1. Tạo fixture checkpoint tối giản

- **File:** `tests/experiments/test_direct_branch_routing_bridge.py`
- **Symbol:** fixture/helper nội bộ của test.
- **Current responsibility:** Chưa có.
- **Change:** Tạo Stage A model payload bằng `ThesisMultitaskModel`, `model_state_dict`, `extra_state` và config Stage B direct tối giản.
- **Reason:** Test phải dùng payload cùng schema thật.
- **Inputs:** `tmp_path`, model dimensions nhỏ, CPU.
- **Outputs:** source checkpoint tạm và config dict.
- **Errors:** Test fail rõ nếu payload thiếu `model_state_dict` hoặc `extra_state`.
- **Dependencies:** `CheckpointManager`, `load_experiment_config` không cần gọi trong unit fixture.
- **Compatibility:** Không sửa payload trong repo.

#### 2. Viết test mismatch allowlist

- **File:** `tests/experiments/test_direct_branch_routing_bridge.py`
- **Symbol:** `test_bridge_rejects_unexpected_checkpoint_keys`.
- **Current responsibility:** Chưa có.
- **Change:** Gọi hàm bridge với một key thừa không phải `discrete_assignment.weight/bias`.
- **Reason:** Không để compatibility mode che lỗi checkpoint.
- **Inputs:** State dict có key `unexpected.weight`.
- **Outputs:** `RuntimeError` chứa `unexpected_keys`.
- **Errors:** Không được chấp nhận key ngoài allowlist.
- **Dependencies:** Hàm bridge dự kiến ở Phase 2.
- **Compatibility:** Hai key assignment vẫn được phép.

#### 3. Chạy test để xác nhận failure đầu tiên

- **File:** Không đổi.
- **Symbol:** Test mới ở trên.
- **Current responsibility:** Chưa có entry point bridge.
- **Change:** Chạy test targeted.
- **Reason:** Xác nhận test đang kiểm tra behavior mới.
- **Inputs:** Repository local.
- **Outputs:** Test fail vì hàm bridge chưa tồn tại hoặc chưa có contract.
- **Errors:** Nếu test pass trước implementation, sửa fixture/test vì nó chưa chứng minh behavior mới.
- **Dependencies:** Bước 1 và 2.
- **Compatibility:** Không chạy Stage A.

### Tests

- **Location:** `tests/experiments/test_direct_branch_routing_bridge.py`
- **Level:** Unit.
- **Setup:** Payload Stage A tạm và config Stage B direct.
- **Action:** Gọi bridge với key hợp lệ và key không hợp lệ.
- **Expected result:** Key hợp lệ được chuyển; key khác bị reject.
- **Edge cases:** Thiếu `model_state_dict`, thiếu normal token, output parent chưa tồn tại.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest -q tests/experiments/test_direct_branch_routing_bridge.py` — fail đúng vì bridge chưa implement.

#### Manual

- [ ] Đọc assertion và xác nhận test không import hoặc gọi `execute_two_stage_plan()`.

### Risks and recovery

- **Risk:** Test vô tình chạy dataset thật.
- **Mitigation:** Dùng `tmp_path`, model nhỏ và loader fixture.
- **Verification:** Không có path dưới `outputs/` trong test.
- **Recovery:** Sửa fixture trước khi sang Phase 2.

### Complete when

- Contract test đã fail vì thiếu implementation và không có side effect ngoài thư mục tạm.

## Phase 2: Bridge độc lập tạo checkpoint Stage B

### Goal

Tái sử dụng logic hiện có bằng một hàm trực tiếp và một CLI mỏng.

### Dependencies

- Phase 1 contract tests phải tồn tại.

### Sequential stages and atomic steps

#### Stage 2.1 — Tách hàm bridge dùng chung

- [ ] Đổi tên hoặc giữ nguyên helper mismatch hiện có để tái sử dụng allowlist.
- [ ] Tạo hàm `prepare_stage_b_initialization_checkpoint(...)` với ba input path/config đã nêu.
- [ ] Đưa bước build Stage B direct vào hàm mới.
- [ ] Đưa bước load Stage A trên CPU vào hàm mới.
- [ ] Đưa bước load `extra_state` vào hàm mới.
- [ ] Đưa bước build train loader với `shuffle_train=False` và `num_workers=0` vào hàm mới.
- [ ] Đưa bước khởi tạo memory banks vào hàm mới.
- [ ] Đưa bước lưu payload và checkpoint metadata vào hàm mới.
- [ ] Sửa `_prepare_stage_b_initialization_checkpoint()` để gọi hàm mới.

#### Stage 2.2 — Thêm CLI bridge

- [ ] Tạo `scripts/experiments/run_direct_branch_routing_bridge.py`.
- [ ] Thêm đối số bắt buộc `--stage-b-config`.
- [ ] Thêm đối số bắt buộc `--stage-a-checkpoint`.
- [ ] Thêm đối số bắt buộc `--output-checkpoint`.
- [ ] Load config bằng `load_experiment_config`.
- [ ] Reject config không có direct Stage B hoặc có `two_stage`.
- [ ] Gọi hàm bridge dùng chung.
- [ ] In path output sau khi lưu thành công.

#### Stage 2.3 — Kiểm tra output bridge

- [ ] Bổ sung test gọi hàm bridge với loader fixture có normal token.
- [ ] Bổ sung test kiểm tra `memory_initialized=True` trong `extra_state`.
- [ ] Bổ sung test load output bằng `CheckpointManager.load_checkpoint(..., strict=True)`.
- [ ] Bổ sung test loader không có normal token phải raise `ValueError`.
- [ ] Chạy lại targeted bridge tests.

Stage 2.2 phụ thuộc Stage 2.1. Stage 2.3 phụ thuộc Stage 2.2.

### Detailed changes

#### 1. Tách hàm bridge dùng chung

- **File:** `scripts/experiments/run_two_stage_offline_pretraining.py`
- **Symbol:** đề xuất `prepare_stage_b_initialization_checkpoint(*, stage_b_config: dict[str, Any], stage_a_checkpoint_path: Path, initialization_checkpoint_path: Path) -> Path`.
- **Current responsibility:** `_prepare_stage_b_initialization_checkpoint(manifest)` lấy config/path từ manifest rồi làm toàn bộ bridge.
- **Change:** Chuyển sáu thao tác vào hàm mới: register components; build Stage B direct model; load CPU payload; allowlist state; load `extra_state`; build train loader không shuffle; initialize memory banks; save payload.
- **Reason:** CLI direct cần bridge mà không tạo manifest và không chạy Stage A.
- **Inputs:** Config có `training_phase=stage_b_fusion_finetuning`, `fusion_mode=direct_branch_routing`; source file; target file.
- **Outputs:** `Path` của target có `model_state_dict`, `extra_state`, `config`, `checkpoint_metadata`.
- **Errors:** `FileNotFoundError` cho source; `RuntimeError` cho mismatch ngoài allowlist; `RuntimeError` nếu memory banks không khởi tạo; lỗi save nếu target không ghi được.
- **Dependencies:** `build_model_from_experiment_config`, `build_dataset`, `CheckpointManager`.
- **Compatibility:** Giữ nguyên `_prepare_stage_b_initialization_checkpoint()` làm adapter gọi hàm mới; không đổi two-stage output cũ.

#### 2. Viết CLI bridge

- **Proposed new file:** `scripts/experiments/run_direct_branch_routing_bridge.py`
- **Symbol:** `parse_args()`, `main()`.
- **Current responsibility:** Chưa có.
- **Change:** Nhận `--stage-b-config`, `--stage-a-checkpoint`, `--output-checkpoint`; load config bằng `load_experiment_config`; kiểm tra direct Stage B; gọi hàm dùng chung; in target path.
- **Reason:** Có entry point riêng để chạy một bridge, không gọi two-stage runner.
- **Inputs:** Ba path tồn tại/được phép tạo parent.
- **Outputs:** Một `stage_b_init.pt`.
- **Errors:** Reject config không direct hoặc có `two_stage`; không fallback âm thầm sang Stage A khác.
- **Dependencies:** Hàm Phase 2.1.
- **Compatibility:** Chạy bằng `.venv/bin/python -m scripts.experiments.run_direct_branch_routing_bridge`.

#### 3. Hoàn thiện test bridge

- **File:** `tests/experiments/test_direct_branch_routing_bridge.py`
- **Symbols:** `test_bridge_initializes_memory_banks`, `test_bridge_output_loads_strictly`.
- **Current responsibility:** Test mới đang fail.
- **Change:** Gọi hàm bridge với loader fixture có normal token; load output vào model Stage B bằng `CheckpointManager(...).load_checkpoint(..., strict=True)`.
- **Reason:** Chứng minh output đã đúng schema Stage B và memory banks đã sẵn sàng.
- **Inputs:** CPU fixture, output dưới `tmp_path`.
- **Outputs:** `memory_initialized=True`, strict load không mismatch.
- **Errors:** Fail nếu không lưu `extra_state` sau memory initialization.
- **Dependencies:** Hàm bridge và fixture Phase 1.
- **Compatibility:** Không dùng `strict=False` ở bước kiểm tra output.

### Tests

- **Location:** `tests/experiments/test_direct_branch_routing_bridge.py`
- **Level:** Integration local.
- **Setup:** Source Stage A payload, direct config, train loader fixture.
- **Action:** Gọi bridge, sau đó strict load output.
- **Expected result:** Output tồn tại, strict load pass, memory banks và verification metadata có trong extra state.
- **Edge cases:** Loader không có normal token phải raise `ValueError`.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest -q tests/experiments/test_direct_branch_routing_bridge.py` — tất cả bridge tests pass.
- [ ] `.venv/bin/python -m pytest -q tests/runtime/test_checkpoint_roundtrip.py` — checkpoint contract cũ không regression.

#### Manual

- [ ] Chạy `--help` chỉ sau khi CLI đã tồn tại và kiểm tra ba đối số là bắt buộc; help không thay thế test runtime.

### Risks and recovery

- **Risk:** Tách helper làm đổi payload two-stage cũ.
- **Mitigation:** Giữ adapter cũ và chạy checkpoint roundtrip.
- **Verification:** Test two-stage helper hiện có vẫn pass.
- **Recovery:** Revert adapter change, giữ hàm mới độc lập rồi sửa sau bằng test.

### Complete when

- CLI tạo được output fixture và output load strict vào Stage B direct.

## Phase 3: Runner direct dùng bridge

### Goal

Runner full/smoke dùng output bridge cho từng identity.

### Dependencies

- Phase 2 bridge function và CLI đã pass.

### Sequential stages and atomic steps

#### Stage 3.1 — Định nghĩa source/target path

- [ ] Tạo helper trả Stage A source path từ offline variant, entity và seed.
- [ ] Tạo helper trả direct target path dưới `stage_b/initializations/stage_b_init.pt`.
- [ ] Đặt target làm `initialization_checkpoint_path` của direct config.
- [ ] Giữ `two_stage` bị loại khỏi direct config.
- [ ] Thêm kiểm tra source và target không trùng nhau.

#### Stage 3.2 — Nối full runner

- [ ] Duyệt 18 direct configs hiện có.
- [ ] Kiểm tra source Stage A tồn tại trước bridge.
- [ ] Gọi hàm bridge khi target chưa tồn tại.
- [ ] Kiểm tra target strict-load được.
- [ ] Gọi `run_training_experiment()` sau khi target pass.
- [ ] Không gọi `execute_two_stage_plan()`.

#### Stage 3.3 — Nối smoke runner

- [ ] Đặt target smoke dưới `outputs/benchmark_smoke/.../initializations/stage_b_init.pt`.
- [ ] Gọi bridge cho O0, `machine_1_6`, seed6 khi target thiếu.
- [ ] Kiểm tra target strict-load được.
- [ ] Gọi training smoke sau bridge.

#### Stage 3.4 — Cập nhật assertions runner

- [ ] Cập nhật assertion full runner để kiểm tra target bridge.
- [ ] Giữ assertion 18 path là duy nhất.
- [ ] Cập nhật assertion smoke target thuộc `benchmark_smoke`.
- [ ] Assert direct configs không có `two_stage`.
- [ ] Chạy test runner targeted.

Stage 3.2 phụ thuộc Stage 3.1. Stage 3.3 chỉ bắt đầu sau khi Stage 3.2 pass. Stage 3.4 phụ thuộc Stage 3.2 và 3.3.

### Detailed changes

#### 1. Định nghĩa source/target path rõ ràng

- **File:** `scripts/run_direct_branch_routing_full.py`
- **Symbols:** `build_direct_experiment_config`, `_parse_baseline_identity`, helper path mới nếu cần.
- **Current responsibility:** Builder đặt Stage A `best.pt` làm initialization path.
- **Change:** Giữ source path theo `O0/O1`, entity, seed; đặt target dưới `.../offline/stage_b/initializations/stage_b_init.pt`; không trộn source với target.
- **Reason:** Source chỉ là input bridge; target mới là input Stage B.
- **Inputs:** Baseline config path.
- **Outputs:** Direct config với target initialization path.
- **Errors:** Reject identity không parse được hoặc source/target trùng nhau.
- **Dependencies:** Phase 2 target contract.
- **Compatibility:** Giữ `two_stage` bị loại khỏi direct config.

#### 2. Gọi bridge trước training

- **File:** `scripts/run_direct_branch_routing_full.py`
- **Symbol:** `main()`.
- **Current responsibility:** Build 18 configs rồi gọi `run_training_experiment()`.
- **Change:** Với từng config, kiểm tra target; nếu target chưa có thì gọi hàm bridge với source Stage A; chỉ sau strict preflight mới gọi `run_training_experiment()`.
- **Reason:** Train loader không thể nạp raw Stage A strict.
- **Inputs:** Một direct config và source checkpoint tương ứng.
- **Outputs:** Stage B training nhận target bridge.
- **Errors:** Dừng ngay nếu source thiếu hoặc strict load target fail.
- **Dependencies:** Phase 2 bridge function.
- **Compatibility:** Không gọi `execute_two_stage_plan()`.

#### 3. Nối smoke runner

- **File:** `scripts/run_direct_branch_routing_smoke.py`
- **Symbol:** `main()`.
- **Current responsibility:** Build một config smoke rồi train.
- **Change:** Dùng target bridge dưới `outputs/benchmark_smoke/.../initializations/stage_b_init.pt`; gọi bridge trước train khi target thiếu.
- **Reason:** Smoke phải kiểm tra đúng checkpoint boundary.
- **Inputs:** O0, `machine_1_6`, seed6.
- **Outputs:** Smoke Stage B direct.
- **Errors:** Dừng nếu target strict load không pass.
- **Dependencies:** Full runner path helper và Phase 2 bridge.
- **Compatibility:** Giữ `use_wandb=False`, epoch và window limits của smoke.

#### 4. Cập nhật assertions runner

- **Files:** `tests/benchmarks/test_direct_branch_routing_full_runner.py`, `tests/benchmarks/test_direct_branch_routing_smoke_runner.py`
- **Symbols:** path assertions.
- **Current responsibility:** Một số test còn kỳ vọng raw Stage A path.
- **Change:** Assert source path riêng và target bridge path riêng; assert direct config dùng target.
- **Reason:** Ngăn regression quay lại raw Stage A.
- **Inputs:** 18 configs và một smoke config.
- **Outputs:** Path uniqueness và identity matching.
- **Errors:** Fail nếu target không theo canonical output tree.
- **Dependencies:** Phase 3.1–3.3.
- **Compatibility:** `fusion_mode`, `training_phase`, `epochs` và output schema giữ nguyên.

### Tests

- **Location:** runner tests nêu trên.
- **Level:** Unit/config integration.
- **Setup:** Baseline config paths, không load GPU.
- **Action:** Build configs và inspect source/target paths.
- **Expected result:** 18 target paths unique, smoke target nằm dưới `benchmark_smoke`, không có `two_stage` trong direct config.
- **Edge cases:** Sai variant, entity hoặc seed phải bị reject.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest -q tests/benchmarks/test_direct_branch_routing_full_runner.py tests/benchmarks/test_direct_branch_routing_smoke_runner.py` — path/config tests pass.
- [ ] `.venv/bin/python -m pytest -q tests/models/test_direct_branch_routing.py` — direct routing behavior không đổi.

#### Manual

- [ ] Đọc call order trong `main()` và xác nhận bridge xảy ra trước `run_training_experiment()`.

### Risks and recovery

- **Risk:** Target đã tồn tại nhưng thuộc identity khác.
- **Mitigation:** Đọc metadata/config trong target trước khi tái sử dụng.
- **Verification:** Assert offline variant/entity/seed trong payload config.
- **Recovery:** Không ghi đè; dừng và báo path mismatch.

### Complete when

- Full và smoke runner chỉ truyền target bridge vào Stage B training.

## Phase 4: Local end-to-end verification

### Goal

Chứng minh toàn bộ flow local tối thiểu chạy được từ source fixture đến một bước Stage B direct.

### Dependencies

- Phase 3 runner/config tests pass.

### Sequential stages and atomic steps

#### Stage 4.1 — Chạy targeted suite

- [ ] Chạy bridge tests.
- [ ] Chạy direct routing tests.
- [ ] Chạy checkpoint roundtrip tests.
- [ ] Chạy full/smoke runner tests.
- [ ] Ghi lại từng kết quả pass/fail.

#### Stage 4.2 — Kiểm tra payload metadata

- [ ] Đọc payload bằng `torch.load(..., map_location="cpu")`.
- [ ] Kiểm tra `training_phase` là `stage_b_fusion_finetuning`.
- [ ] Kiểm tra `fusion_mode` là `direct_branch_routing`.
- [ ] Kiểm tra `memory_initialized` là true.
- [ ] Kiểm tra `verification_metadata_source` không rỗng.
- [ ] Xác nhận log không có Stage A training.

Stage 4.2 chỉ bắt đầu sau Stage 4.1.

### Detailed changes

#### 1. Chạy nhóm test cuối

- **File:** Không sửa nếu test pass.
- **Symbol:** Các test bridge, direct routing, checkpoint roundtrip và runner.
- **Current responsibility:** Kiểm tra từng boundary riêng.
- **Change:** Chạy nhóm test theo lệnh verification.
- **Reason:** Kiểm tra cả bridge và behavior direct hiện có.
- **Inputs:** Local `.venv`.
- **Outputs:** Test report.
- **Errors:** Phân loại failure liên quan bridge với failure cũ không liên quan.
- **Dependencies:** Tất cả phase trước.
- **Compatibility:** Không chạy Stage A, không SSH.

#### 2. Kiểm tra payload metadata

- **File:** Output tạm dưới `tmp_path`.
- **Symbol:** `extra_state` và `checkpoint_metadata`.
- **Current responsibility:** Payload do bridge tạo.
- **Change:** Đọc bằng `torch.load(..., map_location="cpu")` và kiểm tra `memory_initialized`, verification source, config phase/mode.
- **Reason:** Strict load không tự chứng minh memory banks đã sẵn sàng.
- **Inputs:** `stage_b_init.pt` fixture.
- **Outputs:** Các trường metadata đúng.
- **Errors:** Fail nếu thiếu extra state hoặc phase/mode không direct Stage B.
- **Dependencies:** Bridge output.
- **Compatibility:** Không sửa artifact thật.

### Tests

- **Location:** Nhóm test ở Phase 4.
- **Level:** Integration local.
- **Setup:** Fixture checkpoint và direct model.
- **Action:** Bridge → strict load → one forward/backward.
- **Expected result:** Loss finite; reconstruction/classification heads nhận đúng branch; fusion modules không có gradient.
- **Edge cases:** Single batch, CPU, stochastic inference bật/tắt.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest -q tests/experiments/test_direct_branch_routing_bridge.py tests/models/test_direct_branch_routing.py tests/runtime/test_checkpoint_roundtrip.py` — pass.
- [ ] `.venv/bin/python -m pytest -q tests/benchmarks/test_direct_branch_routing_full_runner.py tests/benchmarks/test_direct_branch_routing_smoke_runner.py` — pass.

#### Manual

- [ ] Kiểm tra log không có Stage A training và output target nằm trong `initializations/` của direct stage.

### Risks and recovery

- **Risk:** Full repository có failure không liên quan.
- **Mitigation:** Chỉ kết luận theo targeted tests và ghi rõ failure ngoài scope.
- **Verification:** Lưu command và output test.
- **Recovery:** Không chạy remote cho tới khi targeted tests pass.

### Complete when

- Bridge output strict-load được, memory banks đã initialized, direct one-step pass và runner không gọi Stage A.

## Interface and data changes

Hàm bridge mới nhận `stage_b_config: dict[str, Any]`, `stage_a_checkpoint_path: Path`, `initialization_checkpoint_path: Path` và trả `Path`. Payload giữ schema checkpoint hiện có; chỉ thay `model_state_dict`, `extra_state`, `config` và metadata của model Stage B direct. Không có migration cho Stage A artifact.

## Deployment and rollout

Local implementation chỉ tạo fixture. Khi triển khai remote sau này, kiểm tra read-only từng source và target trước, chạy một smoke combination, rồi mới mở rộng matrix. Không xóa Stage A hoặc checkpoint Stage B cũ.

## Documentation changes

- CLI help phải nói rõ bridge không chạy Stage A.
- Comment trong runner phải phân biệt `stage_a_source_checkpoint` và `stage_b_initialization_checkpoint`.
- Nếu tên hàm hoặc path thay đổi, cập nhật research note và plan reference.

## Final verification

- [ ] Contract tests pass.
- [ ] Bridge CLI tạo output và output load strict.
- [ ] Memory banks được lưu sau initialization.
- [ ] Full/smoke runner dùng target bridge.
- [ ] Không có direct path gọi two-stage orchestrator hoặc fusion head.

## Assumptions and non-blocking uncertainties

- Artifact bridge có thể được tái sử dụng nếu metadata identity khớp; nếu không khớp, runner phải dừng thay vì fallback âm thầm.
- Việc kiểm tra 18 artifact thật trên cloud là bước remote sau khi local implementation hoàn tất.

## Implementation result

- Bridge CLI, full runner và smoke runner đã dùng checkpoint Stage B trung gian.
- Target được strict-load trước Stage B training.
- Test liên quan bridge, direct routing, checkpoint và runner: `27 passed`.
- Toàn bộ pytest repository: `505 passed, 1 skipped, 6 failed`. Sáu lỗi nằm ở các test artifact/snapshot/MLP ngoài phạm vi thay đổi lần này.
- Không chạy Stage A và không chạy GPU cloud.
