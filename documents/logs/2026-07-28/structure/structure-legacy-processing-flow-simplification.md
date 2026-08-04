---
date: 2026-07-28 18:58:21 +07
researcher: TheMetaSetter
git_commit: 71e798d488ddbd9edcbd100a41dc0384f96558be
branch: dev
repository: bachelor-thesis-2026
topic: "Sequential phase and stage structure for legacy processing-flow simplification"
tags: [structure, legacy-flow, simplification, smd, anomaly-detection]
status: draft
last_updated: 2026-07-28
last_updated_by: TheMetaSetter
---

# Structure: Simplify legacy processing flows

## 1. Purpose

Codebase còn chứa nhiều processing path lịch sử bên cạnh flow chính của `full-spec-v2.md` và `full-spec-v3.md`. Cấu trúc này sắp xếp việc đơn giản hoá theo dependency thực tế, từ kiểm kê và khóa contract đến xoá từng nhóm legacy.

Mục tiêu cuối cùng là giữ lại:

- SMD với cửa sổ `L=20`;
- THESIS two-stage: Stage A → memory initialization → Stage B;
- offline evaluation, threshold calibration và online A0/A1/A2;
- stochastic query của v3;
- native baseline flows thuộc full benchmark của v2;
- checkpoint, metric, provenance, report và demo contracts.

Không xoá baseline chỉ vì baseline có protocol riêng. Không xoá archival hoặc operational tooling cho tới khi phạm vi của chúng được quyết định riêng.

## 2. Global execution order

```text
Phase 0: Establish safety boundary
    ↓
Phase 1: Simplify THESIS runtime lifecycle
    ↓
Phase 2: Simplify configuration lifecycle
    ↓
Phase 3: Remove non-main reconstruction model path
    ↓
Phase 4: Narrow online trainable surface
    ↓
Phase 5: Simplify online stream and batch contract
    ↓
Phase 6: Simplify checkpoint resolution
    ↓
Phase 7: Decide and isolate non-main data/operations paths
```

Mỗi phase phải hoàn tất các stage bên trong theo thứ tự từ trên xuống dưới. Phase sau chỉ bắt đầu khi completion condition của phase trước đã đạt.

## 3. Phase structure

### Phase 0 — Establish safety boundary

**Outcome:** Có baseline evidence và caller map đủ để refactor mà không ảnh hưởng thay đổi đang tồn tại hoặc active v2/v3 flows.

**Stages:**

1. **Stage 0.1 — Capture repository state**  
   Ghi nhận commit, branch, worktree changes và các tài liệu SSOT liên quan.
2. **Stage 0.2 — Build legacy caller map**  
   Xác định caller, config và test của từng marker L01-L13.
3. **Stage 0.3 — Classify active versus historical paths**  
   Phân biệt THESIS active flow, v2 baseline flow, compatibility flow, archival path và operational tooling.
4. **Stage 0.4 — Establish baseline verification**  
   Chạy các test group liên quan và ghi nhận failure có sẵn trước refactor.

**Depends on:** Research note L01-L13.

**Completion check:** Có caller matrix, active config list và baseline test result; source code chưa bị xoá.

### Phase 1 — Simplify THESIS runtime lifecycle

**Outcome:** THESIS model runtime chỉ còn lifecycle Stage A → memory initialization → Stage B.

**Stages:**

1. **Stage 1.1 — Freeze the active two-stage contract**  
   Xác định phase names, state transitions, trainable parameters và freeze boundaries được giữ lại.
2. **Stage 1.2 — Remove legacy objective branches**  
   Loại bỏ các nhánh Stage 1 classification/reconstruction, Stage 2 recovery và multitask-pretraining lịch sử.
3. **Stage 1.3 — Remove legacy trainable-state branches**  
   Loại bỏ Stage 3 memory/fusion substeps và các schedule không thuộc Stage A/B.
4. **Stage 1.4 — Reconnect active model callers**  
   Đảm bảo trainer, evaluator, checkpoint và model facade chỉ gọi lifecycle two-stage.
5. **Stage 1.5 — Verify two-stage vertical slice**  
   Kiểm tra Stage A, memory initialization, Stage B, output contract và freeze semantics.

**Depends on:** Phase 0; scope gate không yêu cầu chạy three-stage runtime.

**Completion check:** Một SMD two-stage smoke flow chạy từ training đến checkpoint với không có legacy phase branch.

### Phase 2 — Simplify configuration lifecycle

**Outcome:** Config loader và validator chỉ có canonical two-stage schema cho THESIS active flow.

**Stages:**

1. **Stage 2.1 — Migrate active configs**  
   Xác nhận mọi v2/v3 active config dùng two-stage schema và Stage B metadata.
2. **Stage 2.2 — Remove three-stage validation path**  
   Gỡ validator và normalization path dành riêng cho three-stage.
3. **Stage 2.3 — Remove historical aliases**  
   Gỡ aliases như `stage3_prototype_warmup`, `three_stage_phase` và các phase metadata aliases không còn cần.
4. **Stage 2.4 — Remove or archive three-stage config files**  
   Xử lý các config three-stage theo checkpoint/migration decision.
5. **Stage 2.5 — Verify config rejection and active loading**  
   Active config phải load được; legacy schema phải fail rõ ràng; generated online config vẫn hợp lệ.

**Depends on:** Phase 1.

**Completion check:** Runtime không còn chấp nhận three-stage config và active benchmark config vẫn load/validate thành công.

### Phase 3 — Remove non-main reconstruction model path

**Outcome:** Registry và config surface không còn standalone `reconstruction_mlp_ae` nếu caller audit xác nhận model này không thuộc scope duy trì.

**Stages:**

1. **Stage 3.1 — Confirm no active experiment dependency**  
   Kiểm tra experiment configs, benchmark launchers, demos và checkpoint fixtures.
2. **Stage 3.2 — Migrate test-only dependencies**  
   Chuyển test generic sang active THESIS model hoặc test primitive phù hợp.
3. **Stage 3.3 — Remove model/config registration**  
   Gỡ model file, model config, registry entry và validation entry.
4. **Stage 3.4 — Update compliance and snapshots**  
   Đồng bộ registry tests, config snapshots và compliance fixtures.
5. **Stage 3.5 — Verify active model registry**  
   Kiểm tra model shape, one train step, checkpoint round-trip và benchmark launcher.

**Depends on:** Phase 0; độc lập logic với Phase 1-2 nhưng thực hiện sau để registry/schema đã ổn định.

**Completion check:** Không còn production caller hoặc launcher sinh `reconstruction_mlp_ae`; active THESIS flow vẫn chạy.

### Phase 4 — Narrow online trainable surface

**Outcome:** Online THESIS chỉ có một target parameter group là `projector_params`.

**Stages:**

1. **Stage 4.1 — Confirm projector-only invariant**  
   Đối chiếu model, engine, config và v2/v3 online contract.
2. **Stage 4.2 — Remove online encoder target branch**  
   Gỡ `online_encoder_params` khỏi model parameter-group surface.
3. **Stage 4.3 — Narrow config validation**  
   Đồng bộ validator và error behavior với projector-only contract.
4. **Stage 4.4 — Verify A0/A1/A2 trainable behavior**  
   Kiểm tra frozen reference/encoder, projector update và state round-trip.

**Depends on:** Phase 0; không phụ thuộc việc xoá stream mode.

**Completion check:** Không có public path cập nhật online encoder; A0/A1/A2 vẫn giữ đúng semantics.

### Phase 5 — Simplify online stream and batch contract

**Outcome:** Online stream chỉ dùng active window policy và không còn legacy two-view batch contract.

**Stages:**

1. **Stage 5.1 — Confirm active stream policies**  
   Xác nhận benchmark cần `sliding_stride_1`, `nonoverlap_tail` hoặc cả hai.
2. **Stage 5.2 — Remove `legacy_stride` mode**  
   Gỡ mode và các caller/config không còn hợp lệ.
3. **Stage 5.3 — Remove legacy two-view validator path**  
   Xử lý `validate_legacy_two_view_batch` và console instrumentation nếu không còn caller production.
4. **Stage 5.4 — Migrate online test fixtures**  
   Chuyển fixture từ `view_a/view_b` sang active online batch và adaptation-step construction.
5. **Stage 5.5 — Verify stream and no-label boundary**  
   Kiểm tra `[B,20,D]`, cursor/tail behavior, label isolation và online engine smoke.

**Depends on:** Phase 4 để target/update contract đã ổn định; Phase 0 caller map.

**Completion check:** Active online batch có một contract thống nhất và không còn `view_a/view_b` residue trong production path.

### Phase 6 — Simplify checkpoint resolution

**Outcome:** Online benchmark dùng một canonical metadata-based Stage B checkpoint resolution path.

**Stages:**

1. **Stage 6.1 — Confirm metadata completeness**  
   Xác nhận offline variant, entity, seed, benchmark mode và stage name luôn có trong active configs.
2. **Stage 6.2 — Migrate callers to metadata resolution**  
   Đảm bảo online entrypoints không cần flat legacy path.
3. **Stage 6.3 — Remove flat checkpoint fallback**  
   Gỡ `resolve_legacy_reference_checkpoint_path` và wrapper call.
4. **Stage 6.4 — Verify failure behavior and provenance**  
   Kiểm tra missing/ambiguous metadata, Stage B identity và SHA256/provenance.

**Depends on:** Phase 2 và Phase 5; config và online input contract phải ổn định trước.

**Completion check:** Checkpoint resolution fail early khi metadata thiếu/sai và không còn path guessing.

### Phase 7 — Decide and isolate non-main paths

**Outcome:** Runtime scope phân biệt rõ SMD/active benchmark với AnomalyArchive archival path và operational tooling.

**Stages:**

1. **Stage 7.1 — Decide dataset scope**  
   Chốt AnomalyArchive là active, archival hoặc out of project scope.
2. **Stage 7.2 — Isolate non-main dataset processing**  
   Nếu không active, tách registry/config/parser/analysis khỏi main runtime trước khi xoá.
3. **Stage 7.3 — Review operational tooling boundary**  
   Phân biệt scientific processing với prune/backfill/re-evaluate operations.
4. **Stage 7.4 — Update documentation and tests**  
   Ghi rõ active scope, archival scope và retention/provenance requirements.
5. **Stage 7.5 — Verify final processing-flow inventory**  
   Kiểm tra active entrypoints, registries, configs, tests và benchmark smoke.

**Depends on:** Các phase simplification trước và quyết định scope dataset/baseline.

**Completion check:** Chỉ còn processing flows thuộc scope đã chốt; baseline v2 và required artifacts vẫn hoạt động.

## 4. Items explicitly retained

Các thành phần sau không nằm trong deletion sequence mặc định:

- THESIS public offline/online entrypoints.
- O0/O1, Stage A/B, memory initialization, threshold calibration và A0/A1/A2.
- RedLamp, traditional baselines và native online baselines thuộc v2.
- `classify_legacy_baseline_window` nếu nó tiếp tục phục vụ native baseline protocol.
- Checkpoint, report, provenance, selected diagnostics và operational evidence cần cho nghiên cứu.

## 5. Final dependency summary

```text
Inventory
  -> two-stage runtime
  -> two-stage config
  -> non-main model removal
  -> projector-only online update
  -> active stream/batch contract
  -> metadata-only checkpoint resolution
  -> dataset and operations scope cleanup
```

Sau tài liệu structure này, bước tiếp theo là viết detail plan cho từng stage bằng `prompts/4_detail_prompt.md`. Chưa nên triển khai code trước khi structure được review và chấp nhận.

