---
date: 2026-07-10T20:57:17+0700
researcher: Codex
git_commit: 52b18d95a0f4dd83efc25f5d99e41a20263ad591
branch: dev
repository: bachelor-thesis-2026
topic: "Blocker audit for incomplete full-spec-v2 remediation"
tags: [research, full-spec-v2, blockers, online-tta, gpu]
status: complete
last_updated: 2026-07-10
last_updated_by: Codex
---

# Research: rào cản khiến full-spec-v2 chưa đạt 100%

## Research question

Xác định chính xác những mục trong `detail-full-spec-v2-gap-remediation.md`
chưa được thực thi và rào cản lớn nhất làm pipeline chưa đủ an toàn để chạy
trên GPU server thật.

## Findings

### 1. Entity calibration chưa phải runtime source of truth

`src/engine/online_tta/online_engine.py` hiện có helper
`calibrate_entity_threshold_artifacts()`, nhưng
`_build_runtime_online_context()` vẫn gọi
`calibrate_online_threshold_artifact()` một lần cho toàn bộ validation list.
Runtime vẫn lưu một `threshold_value`, ghi một artifact và dùng cùng ngưỡng
cho mọi test sequence. Đây là blocker lớn nhất vì nó làm sai contract “mỗi
máy một artifact” trước khi adaptation bắt đầu.

### 2. PNN helpers chưa được nối vào event pipeline

`signature_verification.py` chỉ cung cấp pure functions. `_process_online_window()`
không lấy codebook, anomaly radii, continuous prototypes hoặc recurrent
history; `pnn_mask` chỉ được dùng nếu caller tự đặt sẵn trong batch. Vì vậy
A1/A2 chưa thể chứng minh điều kiện “known anomaly bị loại và signature phải
lặp lại ở cửa sổ không chồng lấp”.

### 3. Verification cycle chưa điều khiển engine

`VerificationBuffer` có `try_admit()`, `should_verify()` và
`finish_verification_cycle()`, nhưng `_update_online_window_buffers()` vẫn gọi
`add()` trực tiếp và chưa gọi verification cycle ở ngưỡng 8 entry. TTL do đó
chưa được chứng minh là chỉ giảm sau cycle.

### 4. Checkpoint state mới chỉ được ghi, chưa khôi phục đầy đủ

`_finalize_online_execution()` ghi buffer entries và guard intervals vào
`extra_state`, nhưng runtime context không có loader kiểm tra entity/variant rồi
khôi phục các state này. Resume GPU job có thể bắt đầu lại với state rỗng.

### 5. Test contract và smoke command không cùng layout hiện tại

Detail cũ trỏ tới `tests/unit`, `tests/contract`, `tests/integration`, trong khi
repo hiện dùng `tests/data`, `tests/models`, `tests/online`,
`tests/evaluation`, `tests/runtime`, `tests/benchmarks`, `tests/demo`,
`tests/compliance`. Các smoke YAML O0/O1 tồn tại, nhưng chưa có một integration
test duy nhất kiểm tra toàn bộ entity artifact → triage → update → checkpoint.

### 6. GPU integrity chưa được khóa bằng preflight

Preflight matrix kiểm tra config/path, nhưng chưa xác nhận runtime invariant:
chỉ projector thay đổi, optimizer mới cho mỗi event, không restore optimizer
moments, artifact entity khớp test entity, và không có label trong adaptation
decision.

## Rào cản lớn nhất

Rào cản lớn nhất là thiếu một `OnlineRuntimeState` rõ ràng kết hợp entity,
threshold artifact, signature history, verification buffer, hard-old guard và
checkpoint resume. Khi các phần này còn nằm rời nhau, helper riêng lẻ có thể
pass unit test nhưng GPU experiment vẫn dùng nhầm threshold, bỏ qua PNN cycle
hoặc mất state sau checkpoint.

## Acceptance gap map

| Detail acceptance | Trạng thái hiện tại | Rào cản |
|---|---|---|
| artifact riêng từng entity | helper có, runtime chưa dùng | context chỉ giữ một artifact |
| PNN recurrent mask | pure helper có | chưa nối model prototype/history |
| TTL theo verification cycle | buffer API có | engine vẫn add trực tiếp |
| hard-old hinge | helper/runtime có | guard chưa gắn với successful update |
| fresh AdamW + clip | đã có | chưa có integration assertion |
| checkpoint resume | save fields có | chưa restore/validate |
| demo queue | producer/consumer có | chưa là official benchmark owner |
| full acceptance | chưa đạt | thiếu integration + GPU smoke evidence |

## Code and test references

- `src/engine/online_tta/online_engine.py` - runtime context, calibration,
  event loop and checkpoint export.
- `src/engine/online_tta/signature_verification.py` - pure PNN primitives.
- `src/engine/online_tta/verification_buffer.py` - buffer lifecycle API.
- `src/engine/checkpoint.py` - generic save/load extra-state hooks.
- `tests/online/` - current online focused tests.
- `tests/benchmarks/test_full_benchmark_matrix_preflight.py` - matrix preflight.
- `tests_archive/` - tests intentionally excluded from current collection.
