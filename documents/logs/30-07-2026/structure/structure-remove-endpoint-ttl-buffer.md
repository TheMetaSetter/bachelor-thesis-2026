---
date: 2026-07-30T17:01:56+07:00
topic: "Loại bỏ endpoint TTLBuffer khỏi THESIS online TTA"
status: approved
revision: 1a570cda6eb7976255add3fc5d4f7f385d40dea3
branch: dev
related_documents:
  - documents/logs/30-07-2026/research/research-remove-endpoint-ttl-buffer.md
  - documents/logs/30-07-2026/plan/plan-remove-endpoint-ttl-buffer.md
---

# Implementation Structure: Loại bỏ endpoint `TTLBuffer`

## Summary

Quá trình thực hiện gồm ba pha theo dependency thật của runtime:

1. gỡ atomically endpoint `TTLBuffer`, THESIS telemetry và directly-coupled
   tests;
2. gỡ ghost terminology khỏi shared baseline schema và remaining fixtures;
3. xác minh behavior bằng test local và một thí nghiệm ngắn trên CUDA remote.

Mỗi pha được chia thành các stage nhỏ có kết quả kiểm tra được. Toàn bộ quá
trình phải giữ nguyên `VerificationBuffer`, `ttl_remaining` và quy tắc giảm TTL
sau mỗi verification cycle.

## Request

Dựa trên
[implementation plan](../plan/plan-remove-endpoint-ttl-buffer.md),
tạo dàn ý cho quá trình loại bỏ endpoint `TTLBuffer`. Dàn ý phải nêu rõ các pha,
các stage trong từng pha, dependency, kết quả và điều kiện hoàn tất.

Nhiệm vụ hiện tại chỉ tạo implementation structure. Nhiệm vụ này không sửa
source, test, config, specification hoặc tạo commit.

## Confirmed context

- Runtime hiện tạo endpoint `TTLBuffer` trong online context
  [tại root owner](../../../../src/engine/online_tta/online_engine_run.py#L180-L208)
  và truyền object này qua online sequence.
- Production chỉ gọi `TTLBuffer.add()` để lưu endpoint của window không thuộc
  `strong_anomaly`
  [tại buffer update](../../../../src/engine/online_tta/online_engine_window_metrics.py#L195-L227).
- Runtime chỉ quan sát độ dài của object qua metric
  `online/ttl_buffer_size`
  [tại window output](../../../../src/engine/online_tta/online_engine_window_metrics.py#L230-L278)
  và checkpoint field `ttl_buffer_size`
  [tại finalization](../../../../src/engine/online_tta/online_engine_run.py#L399-L424).
- TTL của admitted verification window thuộc `VerificationBuffer`
  [tại admission và cycle cleanup](../../../../src/engine/online_tta/verification_buffer.py#L30-L76).
  `VerificationCycleController` mới là component kết thúc cycle và kích hoạt
  việc giảm `ttl_remaining`
  [tại cycle orchestration](../../../../src/engine/online_tta/verification_cycle.py#L21-L36).
- Worktree đang có thay đổi riêng về debug timing và chọn đoạn stream. Quá
  trình implementation phải chỉnh trên live worktree và bảo toàn các thay đổi
  đó.

## Scope

### In scope

- Xóa class, runtime ownership, parameter và forwarding của endpoint
  `TTLBuffer`.
- Xóa side effect thêm endpoint.
- Xóa metric `online/ttl_buffer_size` và checkpoint field `ttl_buffer_size`.
- Xóa baseline placeholder cùng tên để THESIS, M2N2 và CANDI không còn ghost
  terminology.
- Cập nhật test và fixture theo runtime/output contract mới.
- Xác minh bằng static checks, test local và một remote CUDA smoke ngắn.

### Out of scope

- Không đổi schema, admission, capacity hoặc TTL của `VerificationBuffer`.
- Không đổi triage, PNN, recurrence, adaptation loss, prediction hoặc
  performance calculation.
- Không đổi config hoặc nội dung lịch sử của `full-spec-v2.md` và
  `full-spec-v3.md`.
- Không tối ưu CUDA hoặc thay đổi debug timing.
- Không chạy full benchmark matrix trong nhiệm vụ loại bỏ này.

## Approved phases and stages

### Phase 1: THESIS runtime và trực tiếp-coupled tests không còn endpoint `TTLBuffer`

**Result:** THESIS online TTA load và xử lý từng window mà không tạo, truyền
hoặc cập nhật endpoint `TTLBuffer`. THESIS telemetry, final checkpoint và các
test import trực tiếp class này cũng được cập nhật trong cùng atomic change.

#### Stage 1.1: Khóa removal boundary trước khi chỉnh

**Outcome:** Implementation có một danh sách exact symbol cần xóa và một danh
sách verification-TTL symbol bắt buộc giữ.

**Main scope:**

- Xác nhận lại exact match của `TTLBuffer`, `ttl_buffer` và
  `ttl_buffer_size` trên live worktree.
- Ghi nhận các phần đang thay đổi sẵn trong ba online engine module để tránh
  ghi đè.
- Tách rõ endpoint `TTLBuffer` khỏi `VerificationBuffer`,
  `ttl_remaining` và `verification_entries`.
- Ghi nhận một pre-change parity snapshot cho đúng O1/A2,
  `machine-1-6`, seed 6 và stream range `[5608,5909)`. Nếu artifact cũ không
  có provenance đầy đủ thì phải chạy snapshot trước khi sửa source.

**Depends on:** Research note và implementation plan đã có.

**Verification:** Search inventory khớp caller graph trong research note; không
có ambiguity về object name.

**Complete when:** Mọi nơi tạo, truyền, ghi và đọc endpoint object đều được
xác định; canonical verification TTL có preservation checklist riêng; Phase 3
có pre-change evidence để so sánh.

#### Stage 1.2: Gỡ đồng bộ runtime, THESIS telemetry và component

**Outcome:** Production source không còn endpoint state, type dependency hoặc
THESIS output field dựa trên state đó.

**Main scope:**

- Gỡ construction trong runtime context.
- Gỡ parameter và forwarding xuyên qua sequence orchestration và window core.
- Gỡ side effect `ttl_buffer.add(...)`.
- Gỡ `online/ttl_buffer_size` và final checkpoint field `ttl_buffer_size`.
- Xóa endpoint component sau khi không còn caller.
- Giữ nguyên thứ tự:
  `prepare_event -> admission/verification -> adaptation_step -> outputs`.

**Depends on:** Stage 1.1.

**Verification:** Online modules import được; signatures của caller và callee
khớp nhau; không còn missing argument hoặc endpoint telemetry reference.

**Complete when:** Entry point chạy đến window processing và finalization mà
không cần `TTLBuffer`; checkpoint vẫn lưu verification state.

#### Stage 1.3: Đồng bộ các test phụ thuộc trực tiếp và khóa Phase 1

**Outcome:** Test collection không import class đã xóa và focused runtime tests
pass với signatures mới.

**Main scope:**

- Xóa unit test chỉ dành cho endpoint `TTLBuffer`.
- Gỡ `TTLBuffer(...)` khỏi các call đến `_run_online_sequence()`.
- Thêm assertion runtime context không còn key `ttl_buffer`.
- Giữ nguyên overlap test của `VerificationBuffer`, max-step assertions và
  causal batch validation.

**Depends on:** Stage 1.2 đã gỡ đồng bộ production contract.

**Verification:** Compile check và directly-coupled focused tests pass; source
và test collection không còn import endpoint module.

**Complete when:** Phase 1 kết thúc ở trạng thái importable và testable, trong
khi gray-zone admission vẫn gọi `VerificationBuffer`.

**Phase risks:**

- Gỡ nhầm verification-entry TTL vì hai object có tên gần nhau.
- Làm mất thay đổi debug timing hoặc stream-range trong dirty worktree.
- Tạm thời làm vỡ import nếu xóa component trước khi gỡ caller.

---

### Phase 2: Shared baseline và remaining fixtures không còn ghost terminology

**Result:** M2N2, CANDI, frozen baselines và remaining fixtures dùng output
contract đã bỏ endpoint TTL terminology.

#### Stage 2.1: Đồng bộ shared baseline schema

**Outcome:** Shared record schema không còn optional argument hoặc output key
liên quan đến endpoint TTL.

**Main scope:**

- Gỡ `ttl_buffer_size` khỏi `build_online_record_schema()`.
- Gỡ `online/ttl_buffer_size` khỏi dictionary mà schema trả về.
- Giữ `online/verification_buffer_size`.

**Depends on:** Phase 1 đã xác lập canonical THESIS output contract.

**Verification:** Shared schema import được và không còn endpoint TTL field.

**Complete when:** Shared baseline contract chỉ chứa runtime concepts thật sự
được baseline hỗ trợ.

#### Stage 2.2: Gỡ baseline placeholders

**Outcome:** M2N2, CANDI và frozen baseline không còn phát placeholder
`online/ttl_buffer_size=0`.

**Main scope:**

- Gỡ constant placeholder trong adaptive và frozen baseline flow.
- Giữ nguyên score, prediction và model update của baseline.

**Depends on:** Stage 2.1 đã cập nhật shared contract.

**Verification:** Baseline contract tests vẫn pass; output metric dictionaries
không còn ghost field.

**Complete when:** Tất cả online methods dùng cùng schema đã bỏ endpoint TTL
telemetry.

#### Stage 2.3: Đồng bộ remaining fixtures và kiểm tra compatibility

**Outcome:** Test suite bảo vệ runtime contract mới và tiếp tục bảo vệ
verification-TTL semantics thật.

**Main scope:**

- Gỡ fake context key và fake metric key cũ.
- Thêm assertion M2N2, CANDI và frozen baseline outputs không còn ghost field.
- Xác nhận old checkpoint có flat `ttl_buffer_size` vẫn được resume path bỏ qua.
- Giữ test của `VerificationBuffer`, verification cycle, runtime state,
  max-step behavior và benchmark wrappers.

**Depends on:** Stage 2.1 và Stage 2.2.

**Verification:** Focused online tests và wrapper contract tests pass.

**Complete when:** Không runtime-output fixture hoặc baseline output nào còn
ghost field. Literal `ttl_buffer_size` chỉ được phép tồn tại trong focused
old-checkpoint compatibility test; không assertion quan trọng về verification
cycle bị xóa hoặc làm yếu đi.

**Phase risks:**

- Artifact mới thay đổi schema dù model behavior không đổi.
- Consumer bên ngoài repository có thể còn đọc field cũ.
- Sửa test quá rộng có thể che regression thay vì phát hiện regression.

---

### Phase 3: Chứng minh main online behavior không đổi

**Result:** Local test evidence và một concrete CUDA run cho thấy refactor chỉ
loại bỏ endpoint state và hai telemetry fields.

#### Stage 3.1: Static và focused local verification

**Outcome:** Source sạch dependency cũ và các contract nhạy cảm vẫn pass.

**Main scope:**

- Search exact endpoint names trong source; source phải zero match.
- Search tests riêng; chỉ old-checkpoint compatibility fixture được phép chứa
  literal `ttl_buffer_size`.
- Compile online TTA modules.
- Chạy focused tests cho verification cycle, runtime state, online sequence và
  baseline/wrapper contracts.
- Kiểm tra diff để phát hiện whitespace error hoặc thay đổi ngoài phạm vi.

**Depends on:** Phase 1 và Phase 2 hoàn tất.

**Verification:** Tất cả static/focused checks pass.

**Complete when:** Không còn dead import, stale parameter, runtime-output
field hoặc lỗi signature; compatibility-test literal được phân loại rõ.

#### Stage 3.2: Full local online regression gate

**Outcome:** Toàn bộ online test suite và THESIS benchmark wrapper test pass.

**Main scope:**

- Chạy toàn bộ `tests/online`.
- Chạy THESIS online benchmark wrapper test.
- Điều tra mọi failure trước khi chuyển sang remote.

**Depends on:** Stage 3.1 pass.

**Verification:** Local regression gate pass hoàn toàn.

**Complete when:** Không còn local failure liên quan đến runtime, serialization
hoặc schema.

#### Stage 3.3: Một remote CUDA parity smoke

**Outcome:** Một thí nghiệm thật xác nhận online flow vẫn hoạt động trên CUDA.

**Main scope:**

- Chạy đúng một O1/A2 THESIS smoke cho `machine-1-6` trên đoạn
  `[5608,5909)`.
- Khóa cùng seed, checkpoint, config và stream range khi so sánh trước/sau.
- So sánh processed windows, triage counts, adaptation count, predictions và
  performance metrics.

**Depends on:** Stage 3.2 pass và remote environment sẵn sàng.

**Verification:** Các giá trị behavior khớp; chỉ
`online/ttl_buffer_size` và `ttl_buffer_size` được phép biến mất.

**Complete when:** Remote run không có import, missing-argument,
serialization hoặc CUDA runtime error; parity condition đạt.

#### Stage 3.4: Final scope audit

**Outcome:** Diff cuối chỉ chứa removal boundary đã phê duyệt.

**Main scope:**

- Xác nhận `VerificationBuffer.default_ttl`, `ttl_remaining` và cycle cleanup
  còn nguyên.
- Xác nhận debug timing và stream-range changes có sẵn không bị mất.
- Ghi rõ artifact schema change và external-consumer uncertainty.

**Depends on:** Stage 3.3 pass.

**Verification:** Manual diff review và final checklist.

**Complete when:** Mọi acceptance criterion của plan đạt và không có thay đổi
ngoài scope.

**Phase risks:**

- Local CPU tests không bao phủ CUDA runtime.
- Hai run khác provenance không thể dùng làm parity evidence.
- Full benchmark chạy quá sớm sẽ làm tăng chi phí chẩn đoán nếu smoke fail.

## Dependency summary

| Phase | Requires | Enables |
| --- | --- | --- |
| Phase 1 — gỡ THESIS endpoint atomically | Removal boundary, dirty-worktree guard và pre-change snapshot | THESIS source/tests ở trạng thái importable không còn endpoint state |
| Phase 2 — gỡ shared ghost contract | Canonical THESIS output contract | Regression tests dùng contract cuối cùng của mọi method |
| Phase 3 — chứng minh parity | Source, outputs và tests đã đồng bộ | Chấp nhận hoặc từ chối refactor dựa trên evidence |

Luồng stage chính:

```text
1.1 boundary
  -> 1.2 runtime + THESIS telemetry + component
  -> 1.3 directly-coupled tests
  -> 2.1 shared baseline schema
  -> 2.2 baseline placeholders
  -> 2.3 remaining fixtures + compatibility
  -> 3.1 focused checks
  -> 3.2 full local online gate
  -> 3.3 remote CUDA smoke
  -> 3.4 final audit
```

## Decisions confirmed

- Canonical object cho admitted verification window là
  `VerificationBuffer`, không phải endpoint `TTLBuffer`.
- Không tạo replacement class hoặc feature flag cho endpoint state.
- Bỏ hai telemetry fields thay vì giữ constant zero.
- Checkpoint cũ không cần migration vì resume path không đọc endpoint entries
  hoặc flat `ttl_buffer_size`.
- Remote verification chỉ chạy sau khi local gates pass.

## Non-blocking uncertainties

- Repository không có consumer đọc hai field cũ. Tool hoặc notebook bên ngoài
  repository vẫn có thể phụ thuộc exact artifact schema.
- Line anchors dựa trên live worktree ngày 30-07-2026. Stage 1.1 phải làm mới
  search inventory nếu source đã thay đổi trước khi implementation bắt đầu.

## Approval note

Ngày 30-07-2026, anh đã phê duyệt việc chuyển THESIS telemetry và các test phụ
thuộc trực tiếp vào cùng atomic removal phase với endpoint `TTLBuffer`. Điều
chỉnh này tránh trạng thái trung gian mà source đã xóa class nhưng caller hoặc
test vẫn import class đó.
