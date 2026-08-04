---
date: 2026-07-30T18:25:15+07:00
topic: "Loại bỏ endpoint TTLBuffer khỏi THESIS online TTA"
status: ready
revision: 1a570cda6eb7976255add3fc5d4f7f385d40dea3
branch: dev
source_structure: documents/logs/30-07-2026/structure/structure-remove-endpoint-ttl-buffer.md
related_documents:
  - documents/logs/30-07-2026/research/research-remove-endpoint-ttl-buffer.md
  - documents/logs/30-07-2026/plan/plan-remove-endpoint-ttl-buffer.md
---

# Detailed Implementation: Loại bỏ endpoint `TTLBuffer`

## Summary

Implementation phải xóa endpoint `TTLBuffer` bằng một refactor nhỏ và trực
tiếp. Refactor không được tạo replacement class, compatibility flag hoặc một
endpoint list mới dưới tên khác.

Thay đổi được chia thành ba pha:

1. xóa đồng bộ endpoint state, THESIS telemetry, final checkpoint field và
   các test import trực tiếp component;
2. xóa ghost terminology khỏi shared baseline schema, M2N2, CANDI, frozen
   baselines và remaining fixtures;
3. chứng minh main online behavior không đổi bằng local gates và một remote
   CUDA parity smoke.

`VerificationBuffer`, `VerificationCycleController`, `ttl_remaining` và
`verification_entries` phải giữ nguyên.

## Source structure

Tài liệu này mở rộng
[approved implementation structure](../structure/structure-remove-endpoint-ttl-buffer.md).

Structure ban đầu đặt việc xóa endpoint component trước việc xóa THESIS
telemetry. Source evidence cho thấy hai việc này phụ thuộc trực tiếp nhau:
`_build_online_window_outputs()` vẫn cần `TTLBuffer` để tạo metric
[tại output builder](../../../../src/engine/online_tta/online_engine_window_metrics.py#L230-L278),
và `_finalize_online_execution()` vẫn đọc context object để tạo checkpoint
field
[tại finalization](../../../../src/engine/online_tta/online_engine_run.py#L379-L424).

Ngày 30-07-2026, anh đã phê duyệt correction sau:

- Phase 1 xóa endpoint runtime, THESIS telemetry và directly-coupled tests
  trong cùng atomic change.
- Phase 2 xử lý shared baseline schema và remaining fixtures.
- Phase 3 chỉ chạy sau khi source và tests đã dùng final contract.

## Current state

### Runtime flow

1. `_build_runtime_online_context()` tạo `TTLBuffer` với
   `ttl_steps=window_size`
   [trong context](../../../../src/engine/online_tta/online_engine_run.py#L124-L208).
2. `_run_online_sequence()` nhận object này rồi truyền nó đến
   `_process_online_window()`
   [trong sequence API](../../../../src/engine/online_tta/online_engine_run.py#L211-L289).
3. Window core chuyển cùng object đến buffer-update helper và output helper
   [trong orchestration](../../../../src/engine/online_tta/online_engine_window_core.py#L54-L142).
4. `_update_online_window_buffers()` lưu `window_end-1` cho mọi window không
   thuộc `strong_anomaly`
   [tại endpoint mutation](../../../../src/engine/online_tta/online_engine_window_metrics.py#L195-L227).
5. `_build_online_window_outputs()` chỉ đọc `len(ttl_buffer)` để tạo
   `online/ttl_buffer_size`
   [tại metric dictionary](../../../../src/engine/online_tta/online_engine_window_metrics.py#L230-L278).
6. Finalization chỉ đọc độ dài lần nữa để tạo flat checkpoint field
   `ttl_buffer_size`
   [tại checkpoint extra state](../../../../src/engine/online_tta/online_engine_run.py#L399-L424).

Production không gọi `expire()`, `contains()` hoặc `clear()` của class
[trong API](../../../../src/engine/online_tta/ttl_buffer.py#L7-L33).

### Canonical verification TTL

`VerificationBuffer.try_admit()` khởi tạo `ttl_remaining=2`
[tại admission](../../../../src/engine/online_tta/verification_buffer.py#L30-L48).
`finish_verification_cycle()` giảm TTL đúng một lần sau completed cycle
[tại cleanup](../../../../src/engine/online_tta/verification_buffer.py#L61-L76).
`VerificationCycleController.maybe_run()` là caller của cleanup này
[tại cycle orchestration](../../../../src/engine/online_tta/verification_cycle.py#L21-L36).

`OnlineRuntimeState` serialize `verification_entries`
[tại payload](../../../../src/engine/online_tta/runtime_state.py#L70-L89)
và restore chúng vào `VerificationBuffer`
[tại restore path](../../../../src/engine/online_tta/runtime_state.py#L188-L216).
Runtime state không serialize endpoint entries `item`/`expires_at`.

### Baseline contract

Shared baseline record schema nhận `ttl_buffer_size` và tạo
`online/ttl_buffer_size`
[trong base schema](../../../../src/baselines/online/base.py#L156-L190).
Adaptive và frozen baselines phát constant zero
[trong adaptive metrics](../../../../src/baselines/online/adaptive.py#L300-L312),
[trong frozen metrics](../../../../src/baselines/online/frozen.py#L234-L246).

M2N2 và CANDI kế thừa `AdaptiveStreamingBaselineBase`
[tại M2N2 class](../../../../src/baselines/online/m2n2.py#L9-L32),
[tại CANDI class](../../../../src/baselines/online/candi.py#L15-L38).
Hai class này không tạo hoặc sử dụng endpoint `TTLBuffer`.

## Desired end state

Sau implementation:

- `src/engine/online_tta/ttl_buffer.py` không còn tồn tại.
- Production và tests không còn import `TTLBuffer`.
- Online context và function signatures không còn `ttl_buffer`.
- Runtime không còn lưu endpoint point index hoặc `expires_at`.
- THESIS, M2N2, CANDI và frozen baseline outputs không còn
  `online/ttl_buffer_size`.
- Checkpoint mới không còn flat field `ttl_buffer_size`.
- Checkpoint mới vẫn giữ `verification_buffer_size`,
  `verification_buffer_entries`, `verification_history` và
  `online_runtime_state`.
- Old checkpoint có extra flat `ttl_buffer_size` vẫn load được vì resume path
  chỉ chọn các canonical fields
  [tại legacy fallback](../../../../src/engine/online_tta/runtime_state.py#L219-L269).
- Scoring, EWMA, triage, verification admission, verification cycle,
  adaptation, prediction và report performance giữ nguyên.

## Scope

### In scope

- Endpoint component và production wiring.
- THESIS metric/checkpoint fields chỉ phụ thuộc component này.
- Shared baseline argument, metric placeholders và output contract.
- Direct imports, call arguments, context fixtures và fake metrics trong tests.
- Focused compatibility test cho old extra field.
- Local regression gates và một remote O1/A2 CUDA parity smoke.

### Out of scope

- `VerificationBuffer` schema, admission, capacity hoặc TTL lifecycle.
- Triage thresholds và truth table.
- PNN mask, recurrent signatures và adaptation loss.
- Debug timing implementation và stream-range implementation.
- Experiment configuration changes.
- Specification history edits.
- Full benchmark matrix.
- External notebooks hoặc reporting tools chưa nằm trong repository.

## Implementation invariants

Các invariant sau là gate bắt buộc:

| Invariant | Phải giữ |
| --- | --- |
| Event order | `prepare_event -> admission/verification -> adaptation_step -> outputs` |
| Gray-zone owner | `VerificationBuffer.try_admit()` |
| Verification TTL start | `ttl_remaining=2` |
| TTL tick | Một lần sau completed verification cycle |
| Adapted-entry cleanup | Loại ngay khi cycle kết thúc |
| Unresolved cleanup | Giảm TTL rồi loại khi `ttl_remaining <= 0` |
| Runtime serialization | `verification_entries` nằm trong `OnlineRuntimeState` |
| Dirty-worktree features | Giữ debug timing và stream-range selection |
| Observable schema delta | Chỉ bỏ `online/ttl_buffer_size` và flat `ttl_buffer_size` |

## Phase 1: Xóa atomically endpoint `TTLBuffer` khỏi THESIS

### Goal

Kết thúc Phase 1 với THESIS source importable, directly-coupled tests pass và
không còn endpoint state hoặc THESIS output field dựa trên state đó.

### Dependencies

- Approved structure và terminology mapping.
- Live worktree chưa bị overwrite.
- Có pre-change parity snapshot với đủ provenance.

### Stage 1.1: Khóa boundary, dirty worktree và pre-change evidence

#### 1. Tạo exact removal inventory

- **Files:** `src/`, `tests/`
- **Symbols:** `TTLBuffer`, `ttl_buffer`, `ttl_buffer_size`,
  `online/ttl_buffer_size`
- **Current responsibility:** Các tên này trải từ component, runtime plumbing,
  telemetry đến test fixtures.
- **Change:** Chưa sửa file. Chạy exact search trên live worktree và lưu output
  vào implementation log.
- **Reason:** Line numbers có thể dịch chuyển. Exact symbol inventory tránh
  xóa nhầm `VerificationBuffer` hoặc `ttl_remaining`.
- **Inputs:** Live checkout tại thời điểm implementation.
- **Outputs:** Danh sách definition, import, construction, argument, call,
  mutation, metric, checkpoint field và test reference.
- **Errors:** Nếu xuất hiện caller mới ngoài inventory trong research note,
  dừng patch và phân loại caller trước.
- **Dependencies:** Toàn bộ Phase 1 và Phase 2.
- **Compatibility:** Search positive cho `VerificationBuffer`,
  `ttl_remaining`, `verification_entries` phải được lưu làm preservation list.
- **Verification:**

```bash
rg -n "\bTTLBuffer\b|ttl_buffer|ttl_buffer_size|online/ttl_buffer_size" src tests
```

Expected: Chỉ có endpoint surfaces đã ghi trong research/plan hoặc caller mới
được phân loại rõ.

#### 2. Ghi nhận dirty-worktree guard

- **Files:**
  - `src/engine/online_tta/online_engine_run.py`
  - `src/engine/online_tta/online_engine_window_core.py`
  - `src/engine/online_tta/online_engine_window_metrics.py`
- **Symbols phải giữ:** `_select_online_stream_sequence`,
  `OnlineTtaTimingLogger`, `debug_timing`, `timing_logger.measure`,
  `_forward_online_window`, `_extract_online_window_scores`
- **Current responsibility:** Live files chứa thay đổi chưa commit về stream
  slicing và debug timing.
- **Change:** Chưa sửa nội dung. Lưu diff trước refactor; patch sau đó chỉ chạm
  exact TTL lines.
- **Reason:** Khôi phục file từ `HEAD` sẽ làm mất công việc ngoài scope.
- **Inputs:** `git diff` của đúng ba file.
- **Outputs:** Pre-change diff dùng cho final scope audit.
- **Errors:** Nếu diff có thay đổi khác ngoài timing/range đã biết, dừng và
  phân loại trước khi patch.
- **Dependencies:** Stage 1.2 và Stage 3.4.
- **Compatibility:** Mọi timing label và stream slicing behavior giữ nguyên.
- **Verification:**

```bash
git status --short
```

```bash
git diff -- src/engine/online_tta/online_engine_run.py
```

```bash
git diff -- src/engine/online_tta/online_engine_window_core.py
```

```bash
git diff -- src/engine/online_tta/online_engine_window_metrics.py
```

Expected: Implementer biết chính xác hunk nào thuộc user changes và hunk nào
sắp thêm cho TTL removal.

#### 3. Capture pre-change parity snapshot

- **Config:**
  `configs/experiment/online_diagnostic/thesis/smd__thesis__online__O1_A2__machine_1_6__w20__seed6__transfer_timing_5608_5909.yaml`
- **Entrypoint:** `scripts/benchmarks/run_thesis_online_benchmark.py::main`
- **Current responsibility:** Config khóa O1/A2, `machine-1-6`, seed 6, CUDA,
  range `[5608,5909)` và `debug_timing=true`
  [tại diagnostic config](../../../../configs/experiment/online_diagnostic/thesis/smd__thesis__online__O1_A2__machine_1_6__w20__seed6__transfer_timing_5608_5909.yaml#L1-L69).
- **Change:** Không đổi config. Xác minh một artifact pre-change có cùng source,
  config, protocol và reference checkpoint. Nếu artifact trước đó thiếu một
  trong bốn checksum/path này, chạy pre-change diagnostic trước Stage 1.2.
- **Reason:** Post-change smoke không tự chứng minh parity.
- **Inputs:**
  - source revision và live diff fingerprint;
  - diagnostic config;
  - protocol config;
  - resolved Stage B checkpoint;
  - seed 6.
- **Outputs cần giữ:**
  - `online_metrics.json`;
  - `online_records.json`;
  - `online_final.pt` metadata hoặc selected extra-state keys;
  - report/manifest;
  - config, protocol và checkpoint checksums.
- **Errors:** Artifact thiếu provenance không được dùng làm parity baseline.
- **Dependencies:** Stage 3.3.
- **Compatibility:** Snapshot chỉ dùng chẩn đoán. Không dùng run với
  `debug_timing=true` để báo cáo performance chính thức.
- **Verification:** Segment dài 301 points. Stride-1 stream với window size 20
  phải tạo `301 - 20 + 1 = 282` windows
  [tại stream builder](../../../../src/engine/online_tta/online_calibration.py#L18-L39),
  [tại window enumeration](../../../../src/data/stream.py#L69-L103).

Remote command sau khi kết nối đến verified repository:

```bash
.venv/bin/python scripts/benchmarks/run_thesis_online_benchmark.py --experiment-config configs/experiment/online_diagnostic/thesis/smd__thesis__online__O1_A2__machine_1_6__w20__seed6__transfer_timing_5608_5909.yaml --protocol-config configs/protocol/smd_window20_cleanval_q99_ewma09.yaml --online-variant A2
```

Expected: `processed_windows=282`, artifact integrity pass và không có runtime
error. Không ghi password hoặc secret vào implementation log.

#### Stage 1.1 tests

Chạy focused tests trước patch để biết baseline local status:

```bash
.venv/bin/python -m pytest -q tests/online/test_online_verification_buffer.py tests/online/test_online_engine_max_steps.py tests/online/test_verification_cycle.py tests/online/test_online_runtime_state.py
```

Expected: Pass. Nếu fail trước patch, ghi rõ failure là pre-existing; không
được sửa ngoài scope chỉ để làm suite xanh.

#### Stage 1.1 complete when

- Removal inventory đầy đủ.
- Preservation list đầy đủ.
- Dirty-worktree diff được lưu.
- Pre-change parity artifact có provenance đầy đủ và 282 windows.

### Stage 1.2: Gỡ đồng bộ runtime, THESIS telemetry và component

Stage 1.2 và Stage 1.3 tạo một atomic Phase 1 patch. Không chạy toàn bộ test
suite ở trạng thái nằm giữa hai stage.

#### 1. Gỡ root ownership và sequence plumbing

- **File:** `src/engine/online_tta/online_engine_run.py`
- **Symbols:** import `TTLBuffer`, `_build_runtime_online_context`,
  `_run_online_sequence`, `_run_online_execution_sequences`,
  `_finalize_online_execution`, `run_thesis_online_tta_experiment`
- **Current responsibility:** File này tạo endpoint object, truyền nó qua hai
  execution paths và đọc độ dài khi save checkpoint
  [tại context và sequence](../../../../src/engine/online_tta/online_engine_run.py#L124-L289),
  [tại multi-sequence call](../../../../src/engine/online_tta/online_engine_run.py#L323-L376),
  [tại direct call](../../../../src/engine/online_tta/online_engine_run.py#L493-L548).
- **Change:**
  1. Xóa import `TTLBuffer`.
  2. Xóa `context["ttl_buffer"]`.
  3. Xóa keyword-only parameter `ttl_buffer` khỏi `_run_online_sequence()`.
  4. Xóa argument forwarding ở `_process_online_window()`.
  5. Xóa `context["ttl_buffer"]` khỏi multi-sequence và direct experiment calls.
  6. Xóa flat checkpoint field `"ttl_buffer_size"`.
- **Reason:** Đây là root owner và outermost contract của endpoint state.
- **Inputs:** Experiment config, protocol config, selected stream, canonical
  `verification_buffer`.
- **Outputs:** Metric history, records và checkpoint không còn endpoint TTL
  field.
- **Errors:** Không thêm fallback `context.get("ttl_buffer")`; fallback sẽ giữ
  dead compatibility path.
- **Dependencies:** Window-core signatures phải đổi trong cùng patch.
- **Compatibility:**
  - Giữ `debug_timing` và `OnlineTtaTimingLogger`.
  - Giữ stream range selection.
  - Giữ `verification_buffer_size`,
    `verification_buffer_entries`, `verification_history` và
    `online_runtime_state`.
  - Giữ error behavior cho invalid variant, missing threshold artifact,
    batch size và stream range.
- **Verification:** Import module thành công; exact search trong file không còn
  endpoint symbol.

#### 2. Gỡ window-core plumbing

- **File:** `src/engine/online_tta/online_engine_window_core.py`
- **Symbols:** import `TTLBuffer`, `_process_online_window`,
  `_build_event_window_outputs`, `_admit_and_verify_online_window`
- **Current responsibility:** Window core không đọc endpoint entries. Nó chỉ
  chuyển object đến buffer-update và output helpers
  [tại orchestration](../../../../src/engine/online_tta/online_engine_window_core.py#L54-L142),
  [tại admission helper](../../../../src/engine/online_tta/online_engine_window_core.py#L201-L230).
- **Change:**
  1. Xóa type import.
  2. Xóa `ttl_buffer` parameter khỏi ba functions.
  3. Xóa forwarding argument đến `_update_online_window_buffers()`.
  4. Xóa positional/keyword argument đến `_build_online_window_outputs()`.
- **Reason:** Core API chỉ nên chứa state tham gia online decisions hoặc
  observable outputs còn được hỗ trợ.
- **Inputs:** Event, canonical `VerificationBuffer`, controller và model state.
- **Outputs:** Cùng `(previous_ewma_score, metric, record)` contract, trừ một
  metric key bị bỏ có chủ đích.
- **Errors:** Không đổi exception hoặc return type của bất kỳ helper nào.
- **Dependencies:** Metrics-helper signatures phải đổi cùng patch.
- **Compatibility:** Giữ nguyên timing boundaries:
  `prepare_event`, `buffer_and_verification`, `adaptation_step`,
  `build_outputs`.
- **Verification:** Event order và timing labels không đổi trong diff.

#### 3. Gỡ endpoint mutation và THESIS metric

- **File:** `src/engine/online_tta/online_engine_window_metrics.py`
- **Symbols:** import `TTLBuffer`, `_update_online_window_buffers`,
  `_build_online_window_outputs`
- **Current responsibility:** Helper vừa cập nhật canonical
  `VerificationBuffer`, vừa ghi endpoint list ngoài spec; output helper phát
  độ dài của cả hai
  [tại buffer helper](../../../../src/engine/online_tta/online_engine_window_metrics.py#L195-L227),
  [tại output helper](../../../../src/engine/online_tta/online_engine_window_metrics.py#L230-L278).
- **Change:**
  1. Xóa type import.
  2. Xóa `ttl_buffer` parameter khỏi hai helpers.
  3. Xóa toàn bộ block
     `if triage_decision != "strong_anomaly": ttl_buffer.add(...)`.
  4. Xóa metric `"online/ttl_buffer_size"`.
- **Reason:** Endpoint mutation không có decision consumer; metric không còn
  source state.
- **Inputs:** Không đổi inputs dùng cho gray-zone admission:
  batch metadata, three scores, triage decision và `VerificationBuffer`.
- **Outputs:** `_update_online_window_buffers()` vẫn trả
  `(admitted, rejected)`. `_build_online_window_outputs()` vẫn trả
  `(record, metric)`.
- **Errors:** Không thay đổi failure behavior của
  `verification_buffer.try_admit()`.
- **Dependencies:** Core callers đổi cùng patch.
- **Compatibility:**
  - Chỉ `gray_zone` gọi `try_admit()`.
  - `normal`, `hard_old_normality` và `strong_anomaly` không tạo verification
    entry.
  - `online/verification_buffer_size` vẫn tồn tại.
  - PNN, scoring và adaptation helpers không đổi.
- **Verification:** Diff không chạm lines thuộc `_score_online_window`,
  `_build_event_pnn_mask` hoặc `_verify_and_adapt_entries`.

#### 4. Xóa endpoint component

- **File to delete:** `src/engine/online_tta/ttl_buffer.py`
- **Symbol:** `TTLBuffer`
- **Current responsibility:** Lưu dictionaries gồm `item` và `expires_at`,
  đồng thời cung cấp unused expiry/query APIs.
- **Change:** Xóa file sau khi tất cả production/test imports đã được xử lý
  trong atomic Phase 1 patch.
- **Reason:** Giữ file không caller sẽ tiếp tục tạo ghost runtime concept.
- **Inputs/outputs/errors:** Không còn public runtime contract sau refactor.
- **Dependencies:** Mọi import phải biến mất trước compile/test.
- **Compatibility:** Không tạo alias hoặc deprecation shim.
- **Verification:** File không tồn tại và module search không còn import path.

### Stage 1.3: Đồng bộ directly-coupled tests và khóa Phase 1

#### 1. Thay test của endpoint class bằng canonical verification behavior

- **File:** `tests/online/test_online_verification_buffer.py`
- **Symbols:** import `TTLBuffer`,
  `test_ttl_buffer_expires_old_items`,
  tests của `_update_online_window_buffers`
- **Current responsibility:** File có một overlap test của
  `VerificationBuffer` và một expiry test chỉ dành cho endpoint class
  [tại tests hiện tại](../../../../tests/online/test_online_verification_buffer.py#L1-L26).
- **Change:**
  1. Xóa `TTLBuffer` import.
  2. Xóa `test_ttl_buffer_expires_old_items`.
  3. Giữ nguyên overlap test.
  4. Thêm focused test: `gray_zone` tạo đúng một `VerificationBuffer` entry với
     `ttl_remaining=2`.
  5. Thêm parameterized test cho `normal`, `hard_old_normality` và
     `strong_anomaly`: không decision nào tạo verification entry.
- **Reason:** Test suite phải chuyển từ deleted helper behavior sang canonical
  online behavior cần giữ.
- **Setup:** Batch nhỏ có `x` tensor và `meta` gồm `stream_step`,
  `start_index`, `end_index`, `entity_id`.
- **Action:** Gọi `_update_online_window_buffers()` với từng triage decision.
- **Expected result:**
  - `gray_zone`: `(admitted=True, rejected=False)`, buffer length 1,
    TTL 2.
  - Non-gray decisions: `(False, False)`, buffer length 0.
- **Edge cases:** Không đưa overlapping second gray window vào test này vì
  overlap contract đã có test riêng.
- **Compatibility:** Không assert endpoint point index hoặc expiry behavior.

#### 2. Đồng bộ sequence-call tests

- **File:** `tests/online/test_online_engine_max_steps.py`
- **Symbols:** `TTLBuffer` import,
  `test_run_online_sequence_honors_max_online_steps`,
  `test_run_online_sequence_rejects_batched_causal_windows`,
  `test_build_runtime_online_context_keeps_none_max_online_steps`
- **Current responsibility:** Hai sequence calls truyền
  `TTLBuffer(ttl_steps=20)`
  [tại first call](../../../../tests/online/test_online_engine_max_steps.py#L38-L77),
  [tại batched call](../../../../tests/online/test_online_engine_max_steps.py#L79-L131).
- **Change:**
  1. Xóa import.
  2. Xóa hai `ttl_buffer=` arguments.
  3. Giữ max-step assertions và batched-window exception assertion.
  4. Sau khi build context, thêm
     `assert "ttl_buffer" not in context`.
  5. Giữ assertion `context["max_online_steps"] is None`.
- **Reason:** Test phải dùng final function/context contract.
- **Inputs/outputs:** Không đổi fake stream hoặc result tuples.
- **Errors:** Batch size 2 vẫn phải raise `ValueError` với message hiện tại.
- **Dependencies:** Production signatures ở Stage 1.2.
- **Compatibility:** Không đổi max-step semantics.
- **Verification:** File không còn endpoint import hoặc constructor.

#### Stage 1 tests

```bash
.venv/bin/python -m compileall -q src/engine/online_tta
```

Expected: Exit 0, không có missing module hoặc syntax error.

```bash
.venv/bin/python -m pytest -q tests/online/test_online_verification_buffer.py tests/online/test_online_engine_max_steps.py tests/online/test_verification_cycle.py tests/online/test_online_runtime_state.py tests/online/test_online_entrypoint.py
```

Expected: Pass. Verification-cycle test vẫn xác nhận TTL `2 -> 1`
[tại existing test](../../../../tests/online/test_verification_cycle.py#L7-L22).

#### Stage 1 risks and recovery

- **Risk:** Xóa file trước caller gây import failure.
  - **Mitigation:** Apply production, direct-test và file-deletion changes như
    một atomic patch.
  - **Verification:** Compileall và focused tests.
  - **Recovery:** Revert chỉ TTL removal hunks; không restore nguyên dirty files.
- **Risk:** Gỡ nhầm canonical verification TTL.
  - **Mitigation:** Positive preservation search trước và sau patch.
  - **Verification:** Verification-cycle/runtime-state tests.
  - **Recovery:** Khôi phục exact verification hunk; không tái tạo endpoint
    class.
- **Risk:** Mất timing/range work.
  - **Mitigation:** Compare against Stage 1.1 dirty diff.
  - **Verification:** Timing and stream-range tests trong Phase 3.
  - **Recovery:** Re-apply preserved dirty hunks từ saved diff.

### Phase 1 complete when

- THESIS source và directly-coupled tests không còn endpoint import/state.
- File endpoint đã xóa.
- THESIS metric/checkpoint no longer emits two obsolete fields.
- Compileall và focused tests pass.
- Canonical verification TTL tests pass.

## Phase 2: Xóa ghost terminology khỏi shared baseline contract

### Goal

Kết thúc Phase 2 với THESIS, M2N2, CANDI và frozen baselines cùng dùng final
output schema, trong khi method behavior không đổi.

### Dependencies

- Phase 1 pass.
- Canonical THESIS schema đã được xác lập.

### Stage 2.1: Đồng bộ shared baseline schema

#### 1. Xóa obsolete optional argument và key

- **File:** `src/baselines/online/base.py`
- **Symbol:** `build_online_record_schema`
- **Current responsibility:** Function nhận `ttl_buffer_size: int = 0` và phát
  `online/ttl_buffer_size`
  [tại schema](../../../../src/baselines/online/base.py#L156-L190).
- **Change:**
  1. Xóa optional argument `ttl_buffer_size`.
  2. Xóa output key `online/ttl_buffer_size`.
  3. Giữ optional `verification_buffer_size`.
- **Reason:** Shared schema không nên quảng bá state mà baselines không có.
- **Inputs:** Các score, indices, prediction, variant, triage, update và
  verification size hiện tại.
- **Outputs:** Cùng record dictionary trừ obsolete field.
- **Errors:** Không thêm validation hoặc đổi conversion behavior.
- **Dependencies:** Project-wide search hiện không có caller truyền argument
  này; re-run search trước edit để xác nhận.
- **Compatibility:** Tất cả existing required arguments và field types giữ
  nguyên.
- **Verification:** Function import được và exact signature không còn argument.

### Stage 2.2: Gỡ baseline placeholders

#### 1. Adaptive baseline metrics

- **File:** `src/baselines/online/adaptive.py`
- **Symbol:** `AdaptiveStreamingBaselineBase.run_sequence`
- **Current responsibility:** Mỗi window tạo metric với
  `online/ttl_buffer_size=0`
  [tại metric append](../../../../src/baselines/online/adaptive.py#L285-L314).
- **Change:** Xóa đúng một dictionary entry.
- **Reason:** M2N2 và CANDI kế thừa flow này nhưng không có endpoint buffer.
- **Inputs/outputs:** Score, prediction, triage, update và remaining metric
  fields giữ nguyên.
- **Errors:** Không đổi loop, calibration hoặc update policy.
- **Dependencies:** M2N2 và CANDI subclasses.
- **Compatibility:** Không chỉnh `M2N2StreamingBaseline` hoặc
  `CANDIStreamingBaseline`; method-specific `_should_update()` giữ nguyên.
- **Verification:** Baseline contract test cho cả M2N2 và CANDI pass.

#### 2. Frozen baseline metrics

- **File:** `src/baselines/online/frozen.py`
- **Symbol:** `_FrozenStreamingBaseline.run_sequence`
- **Current responsibility:** Frozen methods phát cùng zero placeholder
  [tại metric append](../../../../src/baselines/online/frozen.py#L220-L249).
- **Change:** Xóa đúng một dictionary entry.
- **Reason:** Shared benchmark methods phải bỏ ghost field nhất quán.
- **Inputs/outputs:** Prediction và all remaining fields giữ nguyên.
- **Errors:** Không đổi scoring hoặc threshold logic.
- **Dependencies:** Stumpy, KMeansAD và IForest frozen flows trong test matrix.
- **Compatibility:** Không đổi `did_update=False`.
- **Verification:** Baseline contract test cho toàn bộ listed baselines pass.

### Stage 2.3: Đồng bộ remaining fixtures và compatibility test

#### 1. Xóa fake runtime context key

- **File:** `tests/online/test_entity_threshold_runtime.py`
- **Symbol:** `test_execution_selects_a_distinct_artifact_for_each_entity`
- **Current responsibility:** Fake context chứa unused `ttl_buffer` object
  [tại fixture](../../../../tests/online/test_entity_threshold_runtime.py#L16-L48).
- **Change:** Xóa key này.
- **Reason:** Fixture phải phản ánh final runtime context.
- **Expected result:** Test vẫn thấy thresholds `1.0` và `9.0` cho đúng entities.
- **Compatibility:** Không đổi threshold-selection assertion.

#### 2. Xóa fake metric key

- **File:** `tests/online/test_online_streaming_benchmark_wrapper.py`
- **Symbol:** fake baseline `run_sequence`
- **Current responsibility:** Fake metric chứa obsolete key
  [tại fake result](../../../../tests/online/test_online_streaming_benchmark_wrapper.py#L76-L108).
- **Change:** Xóa key này; giữ `online/verification_buffer_size`.
- **Reason:** Fixture không được giữ schema đã bỏ.
- **Expected result:** Wrapper tests vẫn tạo reports và normalized records.
- **Compatibility:** Không đổi fake prediction/update behavior.

#### 3. Strengthen baseline output contract

- **File:** `tests/online/test_online_streaming_baseline_contracts.py`
- **Symbol:** `test_online_streaming_baselines_calibrate_and_run`
- **Current responsibility:** Test chạy CANDI, M2N2 và ba frozen methods nhưng
  chỉ kiểm tra output tồn tại và variant
  [tại baseline matrix](../../../../tests/online/test_online_streaming_baseline_contracts.py#L51-L108).
- **Change:** Trong loop hiện tại, assert:
  - `online/verification_buffer_size` vẫn có trong first metric;
  - `online/ttl_buffer_size` không có trong bất kỳ metric nào;
  - score/prediction/update fields hiện có vẫn tồn tại.
- **Reason:** Contract change phải được test trực tiếp.
- **Setup/action:** Giữ nguyên generated sequences, calibration và run.
- **Expected result:** Tất cả methods dùng final metric schema.
- **Edge cases:** Test bao phủ adaptive và frozen inheritance paths.

#### 4. Verify old-checkpoint compatibility

- **File:** `tests/online/test_online_runtime_state.py`
- **Symbols:** import `resume_online_runtime`, proposed test
  `test_resume_online_runtime_ignores_obsolete_ttl_buffer_size`
- **Current responsibility:** Existing tests bảo vệ runtime-state roundtrip,
  restore và verification-cycle parity
  [tại runtime-state tests](../../../../tests/online/test_online_runtime_state.py#L21-L138).
- **Change:** Thêm một focused compatibility test:
  1. Fake checkpoint manager trả checkpoint có
     `extra_state["ttl_buffer_size"]`.
  2. Cùng `extra_state` chứa valid `online_runtime_state`.
  3. Gọi `resume_online_runtime()` với `VerificationBuffer` và
     `NonOverlapGuard`.
  4. Assert resume thành công, identity đúng và verification entries restore.
- **Reason:** Removal làm thay đổi checkpoint writer nhưng không được làm old
  checkpoint unreadable.
- **Inputs:** Một obsolete scalar bất kỳ, ví dụ `999`; giá trị này không được
  tham gia state restore.
- **Outputs:** Valid `OnlineRuntimeState`; không có endpoint state được dựng lại.
- **Errors:** Existing entity/variant/schema mismatch errors vẫn phải giữ.
- **Dependencies:** Resume loader hiện chỉ chọn canonical fields
  [tại resume logic](../../../../src/engine/online_tta/runtime_state.py#L219-L269).
- **Compatibility:** Không tăng runtime schema version; không migration/backfill.
- **Verification:** New test pass và mismatch tests hiện tại vẫn pass.

#### Stage 2 tests

```bash
.venv/bin/python -m pytest -q tests/online/test_online_streaming_baseline_contracts.py tests/online/test_online_streaming_benchmark_wrapper.py tests/online/test_entity_threshold_runtime.py tests/online/test_online_runtime_state.py tests/online/test_verification_cycle.py
```

Expected: Pass; runtime-output fixtures và output dictionaries không chứa
ghost field. Chỉ compatibility checkpoint fixture chứa obsolete scalar.

#### Stage 2 risks and recovery

- **Risk:** External consumer ngoài repo cần exact old key.
  - **Mitigation:** Ghi schema delta trong implementation handoff; không tạo
    compatibility placeholder khi consumer chưa được chứng minh.
  - **Verification:** Search tracked notebooks/scripts nếu chúng được đưa vào
    scope sau này.
  - **Recovery:** Sửa external consumer. Không khôi phục endpoint class.
- **Risk:** Test update làm yếu behavior checks.
  - **Mitigation:** Chỉ xóa endpoint-specific assertion; thêm canonical
    verification and schema assertions.
  - **Verification:** Review test diff và full online suite.
  - **Recovery:** Khôi phục preserved assertions, không khôi phục deleted class
    test.

### Phase 2 complete when

- Shared/adaptive/frozen baseline source không còn obsolete field.
- M2N2 và CANDI behavior tests pass.
- Runtime-output fixtures dùng final contract; compatibility fixture giữ đúng
  một obsolete scalar để kiểm tra old checkpoint.
- Old checkpoint with obsolete scalar still resumes.

## Phase 3: Chứng minh main online behavior không đổi

### Goal

Chấp nhận refactor chỉ khi static checks, full local regression và one-combination
remote CUDA parity đều pass.

### Dependencies

- Phase 1 và Phase 2 hoàn tất.
- Pre-change snapshot từ Stage 1.1 có đủ provenance.
- Remote source state chỉ khác pre-change state ở approved removal patch.

### Stage 3.1: Static và focused local verification

#### 1. Negative production endpoint search

```bash
rg -n "\bTTLBuffer\b|ttl_buffer|ttl_buffer_size|online/ttl_buffer_size" src
```

Expected: Không có output. Exit code 1 của `rg` trong trường hợp no match là
expected, không phải test failure.

Test search phải chạy riêng:

```bash
rg -n "\bTTLBuffer\b|ttl_buffer|ttl_buffer_size|online/ttl_buffer_size" tests
```

Expected: Không có endpoint import, constructor, call argument hoặc
runtime-output fixture. Literal `ttl_buffer_size` chỉ được phép xuất hiện trong
focused old-checkpoint compatibility test.

#### 2. Positive canonical-state search

```bash
rg -n "VerificationBuffer|ttl_remaining|verification_entries" src/engine/online_tta tests/online
```

Expected: Vẫn có definition, callers, serialization và tests.

#### 3. Compilation

```bash
.venv/bin/python -m compileall -q src/engine/online_tta src/baselines/online
```

Expected: Exit 0.

#### 4. Focused test matrix

```bash
.venv/bin/python -m pytest -q tests/online/test_online_verification_buffer.py tests/online/test_verification_cycle.py tests/online/test_online_runtime_state.py tests/online/test_online_engine_max_steps.py tests/online/test_online_entrypoint.py tests/online/test_entity_threshold_runtime.py tests/online/test_online_streaming_baseline_contracts.py tests/online/test_online_streaming_benchmark_wrapper.py
```

Expected: Pass.

#### 5. Dirty-feature protection

```bash
.venv/bin/python -m pytest -q tests/online/test_online_stream_range.py tests/online/test_online_timing_debug.py
```

Expected: Pass. Refactor không làm mất stream slicing hoặc timing debug.

#### 6. Diff hygiene

```bash
git diff --check
```

Expected: Không có whitespace error.

### Stage 3.2: Full local online regression gate

```bash
.venv/bin/python -m pytest -q tests/online
```

Expected: Toàn bộ online suite pass.

```bash
.venv/bin/python -m pytest -q tests/benchmarks/test_thesis_online_benchmark_wrapper.py
```

Expected: Wrapper report, retention summary, runtime-state export và retention
policy tests pass
[tại wrapper tests](../../../../tests/benchmarks/test_thesis_online_benchmark_wrapper.py#L33-L176).

Không chuyển sang remote nếu một command fail. Phân loại failure theo:

1. endpoint removal regression;
2. pre-existing dirty-worktree failure;
3. environment/dependency failure.

Chỉ loại 1 được sửa trong refactor này. Loại 2 hoặc 3 phải được báo riêng,
không mở rộng patch âm thầm.

### Stage 3.3: Remote CUDA parity smoke

#### 1. Remote preflight

Trước khi ghi source lên shared GPU host:

- xác nhận exact repository root;
- chạy `git status --short`;
- xác nhận không có unrelated job hoặc dirty file sẽ bị overwrite;
- xác nhận diagnostic config, protocol config và reference checkpoint tồn tại;
- xác nhận source/config/checkpoint checksums tương ứng với pre-change snapshot.

Nếu remote có unrelated changes trên cùng target files, dừng và báo conflict.

#### 2. Transfer approved patch only

Chỉ đồng bộ:

- production files trong Phase 1;
- baseline files trong Phase 2;
- tests nếu cần chạy remote tests;
- deletion của exact `src/engine/online_tta/ttl_buffer.py`.

Không đồng bộ toàn repository bằng thao tác có thể overwrite unrelated remote
work.

#### 3. Run one diagnostic combination

```bash
.venv/bin/python scripts/benchmarks/run_thesis_online_benchmark.py --experiment-config configs/experiment/online_diagnostic/thesis/smd__thesis__online__O1_A2__machine_1_6__w20__seed6__transfer_timing_5608_5909.yaml --protocol-config configs/protocol/smd_window20_cleanval_q99_ewma09.yaml --online-variant A2
```

Expected:

- CUDA runtime pass;
- `processed_windows=282`;
- artifact integrity pass;
- no import, missing-argument, serialization or checkpoint error.

#### 4. Compare pre-change and post-change artifacts

Khóa các provenance fields:

| Field | Rule |
| --- | --- |
| Entity | Exact `machine-1-6` |
| Offline/online variant | Exact O1/A2 |
| Seed | Exact 6 |
| Stream range | Exact `[5608,5909)` |
| Window size/stride | Exact 20/1 |
| Reference checkpoint | Same path and SHA-256 |
| Protocol config | Same SHA-256 |
| Source | Same live state ngoài approved removal patch |

So sánh exact:

- processed-window count;
- metric/record count;
- ordered triage decisions;
- triage counts;
- ordered predictions;
- ordered `did_update`;
- adaptation count;
- verification admitted/rejected counts;
- verification buffer size sequence.

So sánh floating values bằng `math.isclose` với
`rel_tol=1e-6`, `abs_tol=1e-8`:

- raw point score;
- EWMA point score;
- input/latent scores;
- non-null loss values;
- final performance metrics.

Allowed schema delta:

- mỗi post-change metric không còn `online/ttl_buffer_size`;
- post-change `online_final.pt` extra state không còn `ttl_buffer_size`.

Không field nào khác được phép biến mất hoặc đổi type.

#### 5. Inspect post-change checkpoint

Load checkpoint bằng `.venv/bin/python` và xác nhận:

- absent: `extra_state["ttl_buffer_size"]`;
- present: `verification_buffer_size`;
- present: `verification_buffer_entries`;
- present: `verification_history`;
- present: `online_runtime_state`;
- `online_runtime_state["verification_entries"]` giữ đúng schema.

#### Suspicious real-world cases

Remote parity report phải ghi coverage thật, không suy diễn:

- Nếu range chứa đủ `normal`, `hard_old_normality`, `gray_zone` và
  `strong_anomaly`, so sánh ordered behavior của cả bốn.
- Nếu một triage region không xuất hiện, ghi region đó là “not exercised”; dùng
  focused unit tests thay vì tuyên bố remote coverage.
- Nếu không có verification cycle hoàn tất trong range, TTL cycle parity chỉ
  dựa vào local verification-cycle test.
- Nếu CUDA floating values lệch quá tolerance nhưng discrete decisions giống,
  chưa được kết luận parity; phải kiểm tra source/checkpoint/config provenance
  và nondeterminism trước.

### Stage 3.4: Final scope audit

#### 1. Review final diff by responsibility

Diff hợp lệ chỉ gồm:

- endpoint source deletion;
- removal of endpoint imports, state, plumbing and telemetry;
- baseline ghost-field removal;
- direct fixture updates;
- added preservation/compatibility assertions;
- approved structure/detail documents.

Diff không được gồm:

- PNN or CUDA optimization;
- triage threshold changes;
- adaptation changes;
- config edits;
- spec history edits;
- unrelated formatting.

#### 2. Re-check preserved dirty work

So sánh với Stage 1.1 saved diff. Xác nhận:

- `_select_online_stream_sequence()` còn nguyên;
- `debug_timing` config lookup còn nguyên;
- `OnlineTtaTimingLogger` wiring còn nguyên;
- timing labels còn nguyên;
- `_forward_online_window()` và `_extract_online_window_scores()` còn nguyên.

#### 3. Record artifact schema change

Implementation handoff phải ghi:

- Artifact mới bỏ hai fields nào.
- Old checkpoint vẫn load theo evidence nào.
- External consumer status vẫn unknown.
- Không có model-weight/config/runtime-schema migration.

#### 4. Final commands

```bash
rg -n "\bTTLBuffer\b|ttl_buffer|ttl_buffer_size|online/ttl_buffer_size" src
```

Expected: No match.

```bash
rg -n "\bTTLBuffer\b|ttl_buffer|ttl_buffer_size|online/ttl_buffer_size" tests
```

Expected: Chỉ có intentional old-checkpoint compatibility literal; không có
endpoint imports hoặc output fixtures.

```bash
git diff --check
```

Expected: Clean.

```bash
git status --short
```

Expected: Chỉ hiện approved changes và pre-existing user changes. Không có
unexpected artifact hoặc temporary file.

### Phase 3 risks and recovery

- **Risk:** Remote after-run overwrite pre-change artifact.
  - **Mitigation:** Capture pre-change artifact/checksums ở separate exact path
    trước source edit.
  - **Verification:** Hai artifact roots khác nhau và có manifests.
  - **Recovery:** Re-run pre-change source chỉ khi exact source state còn tái
    tạo được; nếu không thì không claim parity.
- **Risk:** CUDA nondeterminism gây float drift.
  - **Mitigation:** Same device, seed, checkpoint, config; compare discrete
    fields exactly và floats với stated tolerance.
  - **Verification:** Provenance table và comparison output.
  - **Recovery:** Re-run once trên same host; nếu drift lặp lại, report
    non-parity thay vì nới tolerance tùy ý.
- **Risk:** Remote shared checkout có unrelated edits.
  - **Mitigation:** Read-only status/diff before transfer.
  - **Verification:** Target-file diff review.
  - **Recovery:** Không overwrite; dùng isolated approved workspace hoặc yêu
    cầu người sở hữu remote changes xử lý.

### Phase 3 complete when

- Static and focused gates pass.
- Full local online suite and THESIS wrapper pass.
- One remote O1/A2 range run pass.
- Discrete parity exact and floating parity within fixed tolerance.
- Only allowed schema delta occurs.
- Final diff stays inside approved boundary.

## Interface and data changes

### Function interfaces

| Owner | Current | Target |
| --- | --- | --- |
| `_run_online_sequence()` | Nhận `ttl_buffer: TTLBuffer` | Không nhận endpoint state |
| `_process_online_window()` | Nhận và forward `ttl_buffer` | Chỉ nhận canonical verification state |
| `_admit_and_verify_online_window()` | Forward hai buffers | Chỉ forward `VerificationBuffer` |
| `_update_online_window_buffers()` | Mutate verification + endpoint buffers | Chỉ mutate `VerificationBuffer` cho gray zone |
| `_build_event_window_outputs()` | Forward endpoint buffer | Không có endpoint argument |
| `_build_online_window_outputs()` | Đọc hai buffer sizes | Chỉ đọc verification size |
| `build_online_record_schema()` | Nhận `ttl_buffer_size=0` | Không có argument này |

Đây là internal Python interface change. Repository search không cho thấy
external package consumer. Internal callers và tests phải đổi atomically.

### Metric schema

| Field | Current | Target | Compatibility |
| --- | --- | --- | --- |
| `online/ttl_buffer_size` | THESIS dynamic; baselines zero | Removed | Intentional schema change |
| `online/verification_buffer_size` | Present | Preserved | No change |

### Checkpoint schema

| Field | Current writer | Target writer | Reader behavior |
| --- | --- | --- | --- |
| Flat `ttl_buffer_size` | Present | Removed | Reader already ignores |
| `verification_buffer_size` | Present | Preserved | No change |
| `verification_buffer_entries` | Present | Preserved | Legacy resume source |
| `online_runtime_state.verification_entries` | Present | Preserved | Canonical resume source |

Không tăng `runtime_schema_version`. Không backfill. Không rewrite old
checkpoints.

## Rollout and rollback

### Rollout order

1. Capture pre-change evidence.
2. Apply Phase 1 atomically.
3. Pass Phase 1 tests.
4. Apply Phase 2.
5. Pass all local gates.
6. Transfer only approved patch to remote.
7. Run one parity smoke.
8. Stop if parity fails; không chạy benchmark rộng.

### Rollback

Rollback chỉ hoàn tác approved removal patch. Vì endpoint state không được
resume và không chứa model parameters, rollback không cần data migration.

Không dùng broad reset hoặc checkout trên dirty worktree. Recovery phải áp
dụng exact reverse patch cho TTL changes và giữ user timing/range changes.

## Documentation

Trong implementation task này:

- Giữ research, plan, structure và detail artifacts làm decision evidence.
- Không sửa lịch sử `full-spec-v2.md` hoặc `full-spec-v3.md`.
- Handoff phải dùng canonical name `VerificationBuffer`.
- Nếu tạo spec mới sau này, terminology-change section phải nói rõ endpoint
  `TTLBuffer` đã deprecated/removed và không đồng nhất nó với
  `VerificationBuffer`.

## Final verification checklist

- [ ] Pre-change artifact có đủ provenance.
- [ ] `src/engine/online_tta/ttl_buffer.py` đã xóa.
- [ ] Production source không còn endpoint symbol.
- [ ] Tests chỉ còn obsolete field literal trong old-checkpoint compatibility
      test.
- [ ] `VerificationBuffer` TTL lifecycle còn nguyên.
- [ ] THESIS metric không còn `online/ttl_buffer_size`.
- [ ] New checkpoint không còn flat `ttl_buffer_size`.
- [ ] Verification fields vẫn có trong metric/checkpoint/runtime state.
- [ ] M2N2, CANDI và frozen baselines không còn zero placeholder.
- [ ] Old checkpoint with obsolete scalar resumes.
- [ ] Focused tests pass.
- [ ] Full online suite pass.
- [ ] THESIS wrapper tests pass.
- [ ] Stream-range and timing-debug tests pass.
- [ ] Remote O1/A2 282-window run pass.
- [ ] Parity conditions pass.
- [ ] Diff giữ nguyên pre-existing user changes.
- [ ] Không full benchmark matrix được chạy.

## Assumptions and non-blocking uncertainties

- Tracked repository không có consumer đọc obsolete fields. External consumer
  vẫn unknown.
- Pre-change remote artifact có thể đã tồn tại từ timing diagnostic trước.
  Implementer vẫn phải kiểm tra provenance; không dùng artifact chỉ vì tên
  folder giống.
- Floating comparison tolerance là verification rule cho diagnostic parity,
  không phải thay đổi metric definition.
- Line anchors trong tài liệu phản ánh live dirty worktree ngày 30-07-2026.
  Implementer phải re-run exact search trước patch nếu source đã dịch chuyển.
