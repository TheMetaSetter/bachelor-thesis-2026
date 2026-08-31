---
date: 2026-08-31 Asia/Ho_Chi_Minh
topic: "Implement direct branch routing checkpoint bridge"
status: approved
revision: 6cd32ac94cc20876828f5c62fd2df5c3ac557587
related_documents:
  - documents/logs/2026-08-31/research/research-direct-branch-routing-bridge-flow.md
  - documents/logs/2026-08-31/plan/plan-direct-branch-routing-bridge.md
---

# Implementation Structure: Direct Branch Routing Checkpoint Bridge

## Summary

Luồng triển khai có bốn pha tuần tự. Mỗi pha tạo một kết quả kiểm tra được cho pha sau. Bridge chỉ chuyển checkpoint Stage A đã có sang Stage B direct; bridge không chạy Stage A và direct runtime không gọi hai fusion head.

## Request

Tạo bridge cho Hướng 2, dùng thuật ngữ memory banks, giữ các thiết kế còn lại, lập trình local và không chạy full benchmark hoặc SSH trong lúc lập trình.

## Confirmed context

- Logic chuyển checkpoint đã có trong `scripts/experiments/run_two_stage_offline_pretraining.py` nhưng bị gắn với manifest two-stage.
- Stage B loader dùng strict load, còn runner direct hiện trỏ raw Stage A.
- Direct routing đã bypass fusion trong model.

## Scope

### In scope

- Contract và test cho bridge.
- Hàm bridge dùng chung và CLI bridge riêng.
- Kết nối full/smoke runner với checkpoint bridge.
- Kiểm tra strict load, memory banks và một bước direct training.

### Out of scope

- Chạy hoặc sửa Stage A.
- Thay đổi fusion mode cũ, loss, metric, preprocessing hoặc online flow.
- Chạy cloud GPU hoặc full matrix.

## Proposed phases

### Phase 1: Contract bridge được kiểm thử

**Result:** Test mô tả rõ checkpoint nguồn, checkpoint đích, mismatch hợp lệ và trạng thái memory banks.

**Scope:** Fixture checkpoint/model/loader nhỏ và test mismatch.

**Depends on:** Research bridge đã duyệt và helper hiện có.

**Verification:** Automated test targeted phải thể hiện đúng failure trước implementation.

**Risks:** Fixture không giống payload thật. Mitigation: dùng `CheckpointManager` và `extra_state` thật của model.

**Complete when:** Test cases đã cố định contract và không gọi two-stage runner.

**Sequential stages:**

1. **Stage 1.1 — Tạo fixture payload:** tạo source checkpoint và loader fixture trong `tmp_path`.
2. **Stage 1.2 — Viết test mismatch:** kiểm tra hai khóa `discrete_assignment.*` được phép và khóa khác bị từ chối.
3. **Stage 1.3 — Chạy test đỏ:** xác nhận test fail vì entry point bridge chưa tồn tại.

Stage 1.2 cần Stage 1.1. Stage 1.3 cần cả Stage 1.1 và 1.2.

### Phase 2: Bridge độc lập tạo checkpoint Stage B

**Result:** Hàm dùng chung và CLI tạo `stage_b_init.pt` với memory banks đã khởi tạo.

**Scope:** Tách helper hiện có, giữ allowlist hai khóa assignment, thêm CLI mỏng.

**Depends on:** Phase 1.

**Verification:** Test strict reload pass; CLI chỉ đọc Stage A, train loader và ghi output bridge.

**Risks:** Nạp nhầm mismatch hoặc bỏ qua memory initialization. Mitigation: reject mọi key ngoài allowlist và assert `memory_initialized`.

**Complete when:** Một payload bridge load strict vào model Stage B direct.

**Sequential stages:**

1. **Stage 2.1 — Tách hàm bridge:** chuyển sáu thao tác bridge vào hàm nhận config và path trực tiếp.
2. **Stage 2.2 — Thêm CLI:** tạo entry point module form gọi hàm bridge, không tạo manifest và không chạy Stage A.
3. **Stage 2.3 — Kiểm tra output:** chạy bridge fixture, kiểm tra memory banks và strict reload.

Stage 2.2 cần Stage 2.1. Stage 2.3 cần Stage 2.2.

### Phase 3: Runner direct dùng bridge

**Result:** Mỗi full/smoke run tạo hoặc tái sử dụng bridge trước khi train Stage B.

**Scope:** Dynamic config path, 18 identity paths, smoke path và standalone YAML nếu cần.

**Depends on:** Phase 2.

**Verification:** Runner tests xác nhận path source/target duy nhất và không có `two_stage`.

**Risks:** Ghi đè artifact Stage A hoặc dùng sai identity. Mitigation: output nằm dưới direct stage output và kiểm tra variant/entity/seed.

**Complete when:** `run_training_experiment()` chỉ nhận output bridge.

**Sequential stages:**

1. **Stage 3.1 — Tách source/target path:** định danh Stage A source và direct target cho từng variant/entity/seed.
2. **Stage 3.2 — Nối full runner:** bridge từng combination trước khi train.
3. **Stage 3.3 — Nối smoke runner:** bridge một combination dưới `benchmark_smoke` trước khi train.
4. **Stage 3.4 — Cập nhật assertions:** khóa path và no-Stage-A contract trong test runner.

Stage 3.2 cần Stage 3.1 và Phase 2. Stage 3.3 chỉ bắt đầu sau khi Stage 3.2 pass. Stage 3.4 cần cả 3.2 và 3.3.

### Phase 4: Local end-to-end verification

**Result:** Bridge và một bước Stage B direct chạy cùng nhau trên fixture local.

**Scope:** Targeted pytest cho bridge, checkpoint roundtrip, direct route và runner config.

**Depends on:** Phase 3.

**Verification:** Các lệnh pytest hiện có pass; kiểm tra thủ công payload metadata.

**Risks:** Test suite toàn repository có failure không liên quan. Mitigation: báo riêng targeted result và không tuyên bố full suite pass.

**Complete when:** Strict load, memory banks, direct routing và no-Stage-A contract đều được chứng minh.

**Sequential stages:**

1. **Stage 4.1 — Chạy targeted suite:** chạy test bridge, direct routing, checkpoint roundtrip và runner.
2. **Stage 4.2 — Kiểm tra payload:** đọc metadata CPU và xác nhận phase/mode/memory state.

Stage 4.2 chỉ bắt đầu sau khi Stage 4.1 pass.

## Dependency summary

| Phase | Requires | Enables |
| --- | --- | --- |
| 1 | Research và code hiện có | Contract implementation |
| 2 | Contract tests | Runner integration |
| 3 | Bridge entry point | End-to-end local check |
| 4 | Runner integration | Sẵn sàng cho remote smoke sau này |

## Decisions confirmed

- Dùng checkpoint bridge `stage_b_init.pt`, không đưa raw Stage A vào Stage B strict loader.
- Khởi tạo memory banks một lần từ train loader không shuffle.
- Không gọi two-stage orchestrator trong direct bridge.
- Giữ module fusion cũ trong state-dict chỉ vì checkpoint compatibility; không gọi và không train chúng.

## Non-blocking uncertainties

- Artifact bridge có thể đã tồn tại trên cloud. Runtime sẽ kiểm tra trước khi tạo lại; việc xác minh cloud thuộc bước vận hành sau, không thuộc local implementation.

## Feedback requested

User đã yêu cầu tạo cả plan, structure và detail trong cùng lượt, nên structure này được đánh dấu `approved` để làm nguồn cho tài liệu detail.
