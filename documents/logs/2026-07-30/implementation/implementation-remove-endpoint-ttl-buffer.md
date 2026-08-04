---
date: 2026-07-30
topic: "Implementation: loại bỏ endpoint TTLBuffer"
status: complete
branch: dev
base_revision: 1a570cda6eb7976255add3fc5d4f7f385d40dea3
detail_document: ../detail/detail-remove-endpoint-ttl-buffer.md
---

# Implementation log: loại bỏ endpoint `TTLBuffer`

## Stage 1.1: boundary và pre-change evidence

### Local baseline

- Removal inventory được chốt từ live worktree bằng exact search cho
  `TTLBuffer`, `ttl_buffer`, `ttl_buffer_size` và
  `online/ttl_buffer_size`.
- Preservation inventory xác nhận các symbol canonical vẫn tồn tại:
  `VerificationBuffer`, `ttl_remaining` và `verification_entries`.
- Ba online engine file có sẵn thay đổi về stream range và debug timing. Các
  symbol cần giữ đã được ghi nhận trước patch: `_select_online_stream_sequence`,
  `OnlineTtaTimingLogger`, `debug_timing`, `timing_logger.measure`,
  `_forward_online_window` và `_extract_online_window_scores`.
- Focused pre-change tests: **13 passed**.

### Remote pre-change provenance

- Remote root: `/root/bachelor-thesis-2026`.
- Remote revision trước snapshot: `1a570cda6eb7976255add3fc5d4f7f385d40dea3`.
- Remote source được đồng bộ đúng với local dirty-worktree runtime,
  timing/range implementation, diagnostic config và test cần thiết.
- Diagnostic identity: O1/A2, SMD `machine-1-6`, seed 6, window 20, stride 1,
  stream range `[5608,5909)`.
- Pre-change artifact: `outputs/diagnostics/online/smd/thesis/O1_A2/machine_1_6/seed6/pre_change/`.
- `processed_windows=282`, `expected_windows=282`.
- `artifact_integrity_status=verified`.
- Reference checkpoint: O1 Stage B best checkpoint cho `machine_1_6`, seed 6.
- Debug timing được bật chỉ cho diagnostic này; không dùng artifact này làm
  performance report chính thức.

### Evidence boundary

Pre-change artifact đã được sao lưu riêng trước khi sửa source TTL. Không có
source, test, config hoặc specification nào được sửa ở Stage 1.1.

## Phase 1: production runtime và directly-coupled tests

- Đã gỡ `TTLBuffer` khỏi runtime context, sequence plumbing, window-core
  signatures, endpoint mutation, THESIS metric và flat checkpoint field.
- Đã xóa `src/engine/online_tta/ttl_buffer.py`.
- Đã giữ nguyên `VerificationBuffer`, gray-zone admission và
  `ttl_remaining=2`.
- Đã thay endpoint expiry test bằng focused tests cho canonical verification
  behavior: gray zone admit đúng TTL; `normal`, `hard_old_normality` và
  `strong_anomaly` không tạo verification entry.
- Đã cập nhật max-step tests để dùng context/signature mới.
- Phase 1 verification: compileall pass; **20 tests passed**.

Phase 1 hoàn tất. Chưa chạy remote post-change parity.

## Phase 2: shared baseline contract và fixtures

- Đã xóa `ttl_buffer_size` khỏi `build_online_record_schema()` và xóa
  `online/ttl_buffer_size` khỏi adaptive/frozen baseline metrics.
- M2N2 và CANDI tiếp tục dùng cùng adaptive flow; không sửa policy riêng của
  hai baseline.
- Đã xóa obsolete key khỏi entity-threshold context fixture và benchmark
  wrapper fixture.
- Đã thêm contract assertions cho M2N2, CANDI và frozen baselines: giữ
  `online/verification_buffer_size`, bỏ ghost field và giữ score/prediction/
  update fields.
- Đã thêm old-checkpoint compatibility test. Flat `ttl_buffer_size` được bỏ
  qua; canonical `online_runtime_state` và verification entry vẫn restore,
  với `ttl_remaining=2`.
- Phase 2 verification: **12 tests passed**; chỉ còn literal
  `ttl_buffer_size` trong compatibility fixture và assertion kiểm tra absence.

Phase 2 hoàn tất. Chưa chạy remote post-change parity.

## Phase 3: verification và parity

### Local gates

- Production negative search: không còn match
  `TTLBuffer`, `ttl_buffer`, `ttl_buffer_size` hoặc
  `online/ttl_buffer_size` trong `src`.
- Canonical preservation search vẫn thấy `VerificationBuffer`,
  `ttl_remaining` và `verification_entries` trong runtime/tests.
- Compile `src/engine/online_tta` và `src/baselines/online`: pass.
- Focused matrix: **25 passed**.
- Stream-range và timing-debug protection: **3 passed**.
- Toàn bộ `tests/online`: **76 passed**.
- `tests/benchmarks/test_thesis_online_benchmark_wrapper.py`:
  **2 passed**.
- `git diff --check`: clean.

### Remote CUDA parity

- Post-change O1/A2 diagnostic trên cùng remote host, seed 6, entity
  `machine-1-6`, range `[5608,5909)`, window 20/stride 1.
- Pre-change và post-change đều có **282 metrics và 282 records**;
  artifact integrity được report là `verified`.
- Ordered triage decisions, predictions, `did_update`, admitted/rejected
  counts, verification-buffer size sequence và adaptation count đều giống.
- Score floats đều pass `math.isclose(rel_tol=1e-6, abs_tol=1e-8)`.
- Pre-change metrics có 282 `online/ttl_buffer_size`; post-change có 0.
- Pre-change checkpoint có flat `ttl_buffer_size`; post-change không có.
- Post-change checkpoint vẫn có `verification_buffer_size`,
  `verification_buffer_entries`, `verification_history` và
  `online_runtime_state`; verification runtime entries giống pre-change.
- Checkpoint O1 Stage B best, protocol config và seed/range identity được giữ
  nguyên trong comparison.

## Final scope audit

Đúng phạm vi approved removal patch gồm endpoint deletion/runtime wiring,
baseline ghost-field removal, directly-coupled tests và compatibility
assertions. Các thay đổi dirty-worktree về stream range, debug timing, config
validation và online model loader được giữ nguyên; không có config/spec
history hoặc PNN/adaptation logic nào bị sửa trong removal patch.

Không chạy full benchmark matrix. Không tạo commit.
