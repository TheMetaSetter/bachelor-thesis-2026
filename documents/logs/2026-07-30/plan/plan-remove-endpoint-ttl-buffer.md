---
date: 2026-07-30T16:47:12+07:00
planner: OpenAI Codex
topic: "Loại bỏ endpoint TTLBuffer khỏi THESIS online TTA"
status: ready
revision: 1a570cda6eb7976255add3fc5d4f7f385d40dea3
branch: dev
related_research: documents/logs/30-07-2026/research/research-remove-endpoint-ttl-buffer.md
---

# Implementation Plan: Loại bỏ endpoint `TTLBuffer`

## Summary

Kế hoạch sẽ loại bỏ class endpoint `TTLBuffer`, toàn bộ plumbing của object này
trong THESIS online TTA, unit test chỉ dành cho object đó và hai field telemetry
`online/ttl_buffer_size`/`ttl_buffer_size`.

Kế hoạch giữ nguyên `VerificationBuffer`, `VerificationCycleController`,
`ttl_remaining`, `verification_entries` và toàn bộ TTL semantics theo
verification cycle. Kết quả mong muốn là main online behavior không đổi, nhưng
implementation không còn một endpoint list ngoài ý tưởng gốc và không còn tên
`TTLBuffer` gây nhầm với verification-entry TTL.

## Request

Dùng kết quả research theo `prompts/1_research_prompt.md` để lập kế hoạch sơ
thảo theo `prompts/2_plan_prompt.md`. Mục tiêu là loại bỏ những đoạn code liên
quan đến `TTLBuffer` để implementation gần hơn với ý tưởng gốc.

Nhiệm vụ hiện tại chỉ tạo plan. Nhiệm vụ này không sửa source, test, config,
specification hoặc commit.

## Current state

Runtime tạo một `TTLBuffer` với `ttl_steps=window_size`
[trong context](../../../../src/engine/online_tta/online_engine_run.py#L180-L208),
truyền object qua nhiều function signatures
[trong sequence loop](../../../../src/engine/online_tta/online_engine_run.py#L211-L230)
và thêm `window_end-1` vào object cho mỗi window không phải
`strong_anomaly`
[trong buffer update](../../../../src/engine/online_tta/online_engine_window_metrics.py#L195-L227).

Production runtime không gọi `expire()`, `contains()` hoặc `clear()` của class
[API hiện tại](../../../../src/engine/online_tta/ttl_buffer.py#L7-L33). Nó chỉ
đọc `len()` để phát một metric
[trong window outputs](../../../../src/engine/online_tta/online_engine_window_metrics.py#L230-L278)
và một flat checkpoint field
[trong final checkpoint](../../../../src/engine/online_tta/online_engine_run.py#L399-L424).

TTL theo spec đang nằm trong `VerificationBuffer`
[tại admission và cleanup](../../../../src/engine/online_tta/verification_buffer.py#L30-L76).
`VerificationCycleController` là caller giảm TTL sau completed cycle
[tại cycle orchestration](../../../../src/engine/online_tta/verification_cycle.py#L21-L36).

Chi tiết caller graph và terminology mapping nằm trong
[research note](../research/research-remove-endpoint-ttl-buffer.md).

## Desired end state

Sau implementation:

1. Source và tests không còn import, class, argument, context key hoặc fixture
   mang tên `TTLBuffer`/`ttl_buffer`.
2. THESIS, M2N2 và CANDI metrics không còn
   `online/ttl_buffer_size`.
3. Checkpoint mới không còn flat field `ttl_buffer_size`.
4. `VerificationBuffer` vẫn admitted gray-zone window theo non-overlap rule.
5. Unresolved verification entry vẫn bắt đầu với `ttl_remaining=2`, giảm đúng
   một lần sau mỗi completed verification cycle và bị loại ở zero.
6. Model score, EWMA, triage decision, admission, verification, adaptation,
   prediction và performance metrics không đổi.
7. Các thay đổi debug timing và stream-range đang tồn tại trong worktree được
   bảo toàn.

## Terminology contract

| Khái niệm | Canonical name sau thay đổi | Hành động |
| --- | --- | --- |
| Buffer chứa admitted gray-zone windows và TTL theo verification cycle | `VerificationBuffer` | Giữ nguyên |
| TTL còn lại của một verification entry | `ttl_remaining` | Giữ nguyên |
| Controller kết thúc cycle và giảm entry TTL | `VerificationCycleController` | Giữ nguyên |
| List chứa endpoint của non-strong windows | Không còn runtime object | Xoá |
| Size metric của endpoint list | Không còn output field | Xoá |

Không đổi tên `VerificationBuffer` thành `TTLBuffer`. Không diễn giải biến
`ttl_buffer` trong pseudocode spec-v2 là class endpoint hiện tại. Schema và
lifecycle cho thấy pseudocode đó tương ứng với canonical `VerificationBuffer`.

## Scope

### In scope

- Xoá `src/engine/online_tta/ttl_buffer.py`.
- Xoá import, construction, context key, function argument và forwarding chỉ
  phục vụ endpoint `TTLBuffer`.
- Xoá side effect thêm endpoint.
- Xoá `online/ttl_buffer_size` khỏi THESIS và baseline metric outputs.
- Xoá `ttl_buffer_size` khỏi final checkpoint `extra_state`.
- Cập nhật tests và fake results theo function/output contract mới.
- Chạy targeted tests, toàn bộ online tests và một remote smoke ngắn.

### Out of scope

- Không thay đổi triage thresholds hoặc triage truth table.
- Không thay đổi `VerificationBuffer` schema, capacity, admission hoặc TTL.
- Không thay đổi PNN, signature recurrence hoặc adaptation loss.
- Không đổi online runtime state schema ngoài việc xác nhận nó vốn không chứa
  endpoint TTL entries.
- Không sửa nội dung lịch sử của `full-spec-v2.md` hoặc `full-spec-v3.md`
  trong cùng refactor.
- Không tối ưu CUDA hoặc timing instrumentation trong refactor này.
- Không chạy full benchmark matrix trước khi một concrete remote smoke pass.

## Evidence

- [`TTLBuffer` definition](../../../../src/engine/online_tta/ttl_buffer.py#L7-L33)
  — object lưu `item` và `expires_at`.
- [`TTLBuffer` construction](../../../../src/engine/online_tta/online_engine_run.py#L203-L208)
  — context dùng `window_size` làm `ttl_steps`.
- [`TTLBuffer.add()` production call](../../../../src/engine/online_tta/online_engine_window_metrics.py#L222-L226)
  — caller lưu endpoint point index.
- [`online/ttl_buffer_size`](../../../../src/engine/online_tta/online_engine_window_metrics.py#L276-L277)
  — output runtime duy nhất ở cấp window.
- [`ttl_buffer_size` checkpoint field](../../../../src/engine/online_tta/online_engine_run.py#L418-L420)
  — checkpoint chỉ lưu size, không lưu endpoint entries.
- [`VerificationBuffer` TTL](../../../../src/engine/online_tta/verification_buffer.py#L30-L76)
  — implementation TTL đúng theo verification cycle.
- [`full-spec-v2` TTL policy](../../../../documents/spec/full-spec-v2.md#L1299-L1326)
  — không giảm TTL theo stream step.
- [`full-spec-v3` TTL policy](../../../../documents/spec/full-spec-v3.md#L876-L878)
  — adapted entry bị loại; unresolved entry giảm TTL khi cycle hoàn tất.

## Implementation approach

Chọn một refactor nhỏ, trực tiếp:

1. Gỡ endpoint state khỏi production flow từ root owner đi xuống.
2. Gỡ telemetry chỉ tồn tại vì state đó.
3. Gỡ tests của object đã xoá và chỉnh tests còn lại theo signature mới.
4. Chạy parity tests tập trung vào behavior cần giữ.

Không thay `TTLBuffer` bằng class khác hoặc feature flag. Replacement sẽ giữ
thêm một codepath không có trong ý tưởng gốc. Không giữ
`online/ttl_buffer_size=0` để tương thích nội bộ vì repository không có consumer
đọc field này; giữ field zero sẽ tiếp tục tạo ghost terminology.

Đây là behavior-preserving simplification đối với main experiment flow. Nó có
một thay đổi contract có chủ đích: artifact mới bỏ hai field telemetry không
thuộc spec.

## Phase 1: Remove production component and wiring

### Goal

Online runtime load và xử lý window mà không tạo hoặc truyền endpoint
`TTLBuffer`.

### Changes

#### 1. Delete the endpoint component

- **File:** `src/engine/online_tta/ttl_buffer.py`
- **Symbol:** `TTLBuffer`
- **Change:** Xoá toàn bộ file.
- **Reason:** Production không thực hiện expiry và không dùng stored endpoint
  cho quyết định nào.
- **Dependencies:** Gỡ tất cả import trước khi verification.

#### 2. Remove root ownership and sequence plumbing

- **File:** `src/engine/online_tta/online_engine_run.py`
- **Symbols:** import `TTLBuffer`, `_build_runtime_online_context()`,
  `_run_online_sequence()`, `_run_online_execution_sequences()`,
  `run_thesis_online_tta_experiment()`
- **Change:** Xoá import, `context["ttl_buffer"]`, parameter `ttl_buffer` và ba
  call arguments truyền object.
- **Reason:** File này tạo và sở hữu object phụ.
- **Dependencies:** Cập nhật signatures trong window core ở cùng phase để
  source luôn import được sau patch.
- **Preservation rule:** Không đụng `debug_timing`, stream-range selection,
  `verification_buffer`, `hard_old_guard` hoặc `signature_history`.

#### 3. Remove window-core plumbing

- **File:** `src/engine/online_tta/online_engine_window_core.py`
- **Symbols:** import `TTLBuffer`, `_process_online_window()`,
  `_build_event_window_outputs()`, `_admit_and_verify_online_window()`
- **Change:** Xoá type import, parameters và forwarding arguments của
  `ttl_buffer`.
- **Reason:** Core không đọc nội dung object; nó chỉ chuyển object sang helper.
- **Dependencies:** Cập nhật helper signatures trong metrics module.
- **Preservation rule:** Giữ nguyên event order:
  `prepare_event -> admission/verification -> adaptation_step -> outputs`.

#### 4. Remove endpoint mutation

- **File:** `src/engine/online_tta/online_engine_window_metrics.py`
- **Symbols:** import `TTLBuffer`, `_update_online_window_buffers()`
- **Change:** Xoá parameter `ttl_buffer` và block
  `if triage_decision != "strong_anomaly": ttl_buffer.add(...)`.
- **Reason:** Đây là side effect duy nhất ghi endpoint state.
- **Dependencies:** Không đổi gray-zone branch gọi
  `verification_buffer.try_admit()`.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest -q tests/online/test_online_engine_max_steps.py tests/online/test_entity_threshold_runtime.py` — online loop signatures vẫn chạy.
- [ ] `.venv/bin/python -m pytest -q tests/online/test_verification_cycle.py tests/online/test_online_runtime_state.py` — canonical verification TTL vẫn chạy.
- [ ] `.venv/bin/python -m compileall -q src/engine/online_tta` — không còn import hoặc signature lỗi.

#### Manual

- [ ] Đọc diff của ba online engine files — các thay đổi debug timing và
  stream-range hiện có vẫn còn nguyên.

### Risks

- **Import break:** Xoá file trước khi gỡ import làm toàn module không load.
  Patch phải đồng bộ component và caller rồi mới chạy test.
- **Accidental TTL deletion:** Tìm kiếm tên `ttl` có thể chạm
  `VerificationBuffer.default_ttl`. Chỉ xoá exact endpoint symbols đã liệt kê.
- **Dirty-worktree collision:** Ba caller files đang có thay đổi chưa commit.
  Dùng patch hẹp; không checkout hoặc thay nguyên file.

## Phase 2: Remove obsolete telemetry and align tests

### Goal

Artifact/schema mới không còn field gợi ý rằng endpoint `TTLBuffer` vẫn là một
runtime concept.

### Changes

#### 1. Remove THESIS telemetry

- **File:** `src/engine/online_tta/online_engine_window_metrics.py`
- **Symbol:** `_build_online_window_outputs()`
- **Change:** Xoá parameter `ttl_buffer` và metric
  `online/ttl_buffer_size`.
- **Reason:** Không còn state để đo; spec không yêu cầu field này.
- **Preserve:** Giữ `online/verification_buffer_size`.

- **File:** `src/engine/online_tta/online_engine_run.py`
- **Symbol:** `_finalize_online_execution()`
- **Change:** Xoá flat checkpoint field `ttl_buffer_size`.
- **Reason:** Field chỉ là độ dài của object ngoài spec.
- **Preserve:** Giữ `verification_buffer_size`,
  `verification_buffer_entries` và `online_runtime_state`.

#### 2. Remove baseline zero placeholders

- **File:** `src/baselines/online/base.py`
- **Symbol:** `build_online_record_schema()`
- **Change:** Xoá optional argument `ttl_buffer_size` và output key tương ứng.

- **File:** `src/baselines/online/frozen.py`
- **Change:** Xoá `online/ttl_buffer_size: 0` khỏi metric dictionary.

- **File:** `src/baselines/online/adaptive.py`
- **Change:** Xoá `online/ttl_buffer_size: 0` khỏi metric dictionary.

- **Reason:** M2N2 và CANDI không có endpoint `TTLBuffer`; zero placeholder chỉ
  giữ ghost terminology.
- **Preserve:** Không đổi score, prediction, update hoặc baseline method logic.

#### 3. Update tests and fixtures

- **File:** `tests/online/test_online_verification_buffer.py`
- **Change:** Xoá `TTLBuffer` import và
  `test_ttl_buffer_expires_old_items()`. Giữ overlap test của
  `VerificationBuffer`.
- **Reason:** Test này chỉ bảo vệ class bị xoá; nó không bảo vệ main online
  behavior.

- **File:** `tests/online/test_online_engine_max_steps.py`
- **Change:** Xoá import và hai constructor/call arguments `TTLBuffer(...)`.
- **Reason:** Khớp sequence signature mới; assertions về max steps giữ nguyên.

- **File:** `tests/online/test_entity_threshold_runtime.py`
- **Change:** Xoá fake context key `ttl_buffer`.
- **Reason:** Runtime context không còn đọc key.

- **File:** `tests/online/test_online_streaming_benchmark_wrapper.py`
- **Change:** Xoá obsolete metric key khỏi fake result.
- **Reason:** Khớp artifact schema mới.

### Verification

#### Automated

- [ ] `rg -n "\bTTLBuffer\b|ttl_buffer|ttl_buffer_size|online/ttl_buffer_size" src tests` — không còn match.
- [ ] `.venv/bin/python -m pytest -q tests/online/test_online_verification_buffer.py tests/online/test_verification_cycle.py tests/online/test_online_runtime_state.py` — verification TTL contract còn nguyên.
- [ ] `.venv/bin/python -m pytest -q tests/online/test_online_streaming_baseline_contracts.py tests/online/test_online_streaming_benchmark_wrapper.py` — M2N2/CANDI/THESIS wrapper contracts vẫn chạy.
- [ ] `.venv/bin/python -m pytest -q tests/online` — toàn bộ online suite pass.
- [ ] `git diff --check` — không có whitespace error.

#### Manual

- [ ] Kiểm tra một generated `online_metrics.json`: có
  `online/verification_buffer_size`, không có `online/ttl_buffer_size`.
- [ ] Kiểm tra `online_final.pt` bằng `.venv/bin/python`: `extra_state` có
  verification state nhưng không có `ttl_buffer_size`.

### Risks

- **Artifact schema drift:** Artifact mới thiếu một field có trong artifact cũ.
  Không cần data migration vì checkpoint cũ chỉ có scalar size và resume code
  không đọc field này. Ghi thay đổi vào run provenance hoặc implementation note.
- **External consumer chưa thấy:** Repository không có consumer, nhưng notebook
  ngoài repo có thể phụ thuộc exact key. Trước full benchmark, search các
  notebook/script mà team dùng để tạo bảng nếu chúng nằm ngoài tracked source.

## Phase 3: Verify one real online flow before wider experiments

### Goal

Xác nhận refactor không đổi runtime flow trên một concrete THESIS combination
trước khi chạy benchmark rộng.

### Changes

Không thay đổi code ở phase này. Chỉ chạy verification.

### Verification

#### Automated

- [ ] Local: `.venv/bin/python -m pytest -q tests/online tests/benchmarks/test_thesis_online_benchmark_wrapper.py` — online and wrapper suite pass.
- [ ] Remote CUDA: chạy đúng một O1/A2 THESIS smoke cho
  `machine-1-6` trên đoạn `[5608,5909)`, dùng checkpoint và config diagnostic
  đã được xác minh trong task trước.
- [ ] So sánh trước/sau trên cùng seed, checkpoint và stream range:
  số processed windows, triage counts, adaptation count, predictions và
  performance metrics phải giống nhau; chỉ hai TTLBuffer telemetry fields được
  phép biến mất.

#### Manual

- [ ] Xem log remote để xác nhận không có import error, missing argument hoặc
  checkpoint serialization error.
- [ ] Không bật full benchmark matrix sau smoke nếu parity chưa đạt.

### Risks

- **CUDA environment:** Máy local không có CUDA, nên local tests không thay thế
  real online smoke. Dùng remote chỉ sau khi source và focused tests pass.
- **Uncontrolled comparison:** Hai run khác config/checkpoint/range không chứng
  minh parity. Comparison phải khóa cùng provenance.

## Testing strategy

Testing chia thành ba lớp:

1. **Static contract:** `rg`, compileall và diff check phát hiện dead import,
   parameter hoặc field còn sót.
2. **Behavior unit/integration:** verification-cycle tests bảo vệ
   `ttl_remaining`; online loop và baseline wrapper tests bảo vệ signatures và
   output flow.
3. **Real runtime parity:** một remote O1/A2 short-stream run bảo vệ CUDA path,
   serialization và experiment behavior.

Test bị xoá duy nhất là test của endpoint class đã xoá. Các test bảo vệ main
behavior không được làm yếu đi để khiến refactor pass.

## Migration and rollback

Không có model-weight, config hoặc runtime-state migration.

Checkpoint cũ vẫn có thể chứa `extra_state["ttl_buffer_size"]`; resume code hiện
không đọc field này nên loader có thể bỏ qua field thừa. Checkpoint mới chỉ
không tạo field đó.

Rollback an toàn là hoàn tác đúng refactor patch. Không cần khôi phục artifact
đã tạo. Nếu external reporting tool thật sự yêu cầu field cũ, ưu tiên sửa tool
để không phụ thuộc ghost field; không tái tạo endpoint class chỉ để phát một
scalar.

## Documentation

- Giữ research note này làm evidence về mapping:
  spec-v2 pseudocode `ttl_buffer` → canonical `VerificationBuffer`;
  live endpoint `TTLBuffer` → object bổ sung đã loại bỏ.
- Nếu sau này tạo `full-spec-v4.md`, thêm terminology-change section nói rõ
  canonical object là `VerificationBuffer` và không có endpoint `TTLBuffer`.
- Không sửa lịch sử của spec-v2/spec-v3 trong refactor code này.

## Final verification

- [ ] `src/engine/online_tta/ttl_buffer.py` không còn tồn tại.
- [ ] Không còn exact source/test reference đến endpoint `TTLBuffer`.
- [ ] Không còn `online/ttl_buffer_size` hoặc checkpoint
  `ttl_buffer_size` trong output mới.
- [ ] `VerificationBuffer.default_ttl`, `ttl_remaining` và cycle cleanup còn
  nguyên.
- [ ] Focused tests và toàn bộ online suite pass.
- [ ] Một remote CUDA O1/A2 short-stream smoke pass với parity ngoài hai field
  telemetry đã xoá.
- [ ] Diff không chứa thay đổi ngoài removal boundary.

## Assumptions and non-blocking uncertainties

- Repository search hiện tại không tìm thấy consumer của hai telemetry fields.
  Consumer bên ngoài repository vẫn là unknown.
- Plan giả định artifact schema mới được phép bỏ các field ngoài spec. Đây là
  hệ quả trực tiếp của yêu cầu loại bỏ component, không ảnh hưởng performance
  calculation trong source hiện tại.
- Line reference dựa trên live dirty worktree ngày 30-07-2026. Implementation
  agent phải re-run `rg` trước khi patch nếu line numbers đã dịch chuyển.
