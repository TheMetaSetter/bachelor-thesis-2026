---
date: 2026-08-09T00:00:00+07:00
topic: "Phân rã stage để lập trình anomaly score theo full-spec-v3"
status: proposed
revision: b004e70b26b956809695c1b9d9518adf900ed2e9
related_documents:
  - documents/logs/2026-08-09/research/research-anomaly-score-implementation-gaps.md
  - documents/logs/2026-08-09/plan/plan-anomaly-score-calibration.md
  - documents/spec/full-spec-v3.md
---

# Implementation Structure: Phân rã stage để lập trình anomaly score theo full-spec-v3

## Summary

Structure này phân rã 7 phase trong plan anomaly score thành các stage tuần tự. Mỗi phase tạo một outcome có thể kiểm tra, còn mỗi stage hoàn thành một phần nhỏ trước khi chuyển sang stage kế tiếp.

Luồng phụ thuộc chính là:

```text
calibration/artifact contract
  -> model/scorer xuất q
  -> offline clean-validation calibration
  -> v4 artifact migration
  -> online runtime dùng q trước EWMA
  -> legacy-boundary reconciliation
  -> full verification và recalibration
```

## Request

Đọc `prompts/3_structure_prompt.md` và viết các stage tuần tự trong mỗi phase của `plan-anomaly-score-calibration.md`.

Structure này không triển khai code và không đi xuống mức file-level edit chi tiết. Structure giữ các quyết định của plan:

- official point anomaly score là **shifted-and-scaled logistic sigmoid**;
- `c = median(clean-validation raw point MSE)`;
- `tau = MAD(clean-validation raw point MSE) / 0.6745`;
- `tau <= 0` thì fail-fast;
- offline và online dùng cùng `c`,`tau` theo entity nhưng có threshold riêng;
- EWMA giữ trọng số `0.9/0.1`;
- `window_scores` và training loss vẫn dùng raw MSE.

## Confirmed context

- THESIS model và online adapter hiện xuất raw point MSE ở `point_scores`; raw MC sample MSE nằm trong stochastic output payload.
- Offline benchmark hiện threshold hóa raw clean-validation scores.
- Online calibration hiện đưa raw point scores vào absolute-index EWMA.
- Threshold artifact hiện chưa chứa transform identity, `c` hoặc `tau`.
- EWMA algorithm đã đúng; thay đổi cần thiết là bảo đảm đầu vào của EWMA là transformed score `q`.
- `full-spec-v3.md` là normative design; ontology files là terminology SSOT.

## Scope

### In scope

- Shared calibration contract và artifact schema.
- THESIS model/scorer output semantics.
- Offline calibration và threshold construction.
- V4 artifact recalibration/migration.
- Online runtime, EWMA input, prediction và event records.
- Legacy/provisional boundary, tests, smoke verification và artifact provenance.

### Out of scope

- Thay đổi reconstruction loss, `point_score_loss`, MC sample count, model architecture hoặc EWMA weights.
- Thay đổi `window_scores` thành sigmoid score.
- Sửa RedLamp, traditional baselines hoặc online baseline calibration riêng.
- Chạy toàn bộ benchmark matrix trước khi one-combination verification pass.

## Proposed phases

### Phase 1: Calibration và threshold artifact contract hoạt động thống nhất

**Result:** Project có một định nghĩa duy nhất cho `c`, `tau`, sigmoid transform và các field artifact v4; các giá trị không hợp lệ bị phát hiện trước khi chạy model.

**Stages:**

1. **Stage 1.1 — Khóa calibration semantics và edge policy**
   - Xác nhận input là raw clean-validation point MSE theo entity.
   - Xác nhận `tau = MAD / 0.6745` và `tau <= 0` bị fail-fast.
   - Xác nhận calibration state là dữ liệu bắt buộc của official inference.

2. **Stage 1.2 — Xây shared calibration helper**
   - Tạo calibration object và các thao tác fit/transform dùng chung.
   - Giữ công thức sigmoid ở một owner duy nhất.
   - Trả về các giá trị có thể dùng cho model, NumPy threshold calculation và artifact provenance.

3. **Stage 1.3 — Mở rộng threshold artifact v4**
   - Thêm transform name, `c`, `tau`, estimator và MAD normalizer vào schema.
   - Ràng buộc v4 artifact phải chứa calibration identity hợp lệ.
   - Giữ artifact cũ nguyên vẹn và không cho active v3 runtime dùng artifact thiếu metadata.

4. **Stage 1.4 — Kiểm tra contract nền tảng**
   - Chạy unit tests cho median, MAD, sigmoid, finite checks và artifact round-trip.
   - Kiểm tra artifact mẫu chứa đầy đủ transform metadata.

**Depends on:** Không có phase trước.

**Verification:**

- Automated: calibration unit tests và threshold artifact tests pass.
- Manual: artifact JSON thể hiện đúng transform identity, `c`, `tau` và provenance.

**Risks:** Artifact v3/v4 cũ không có transform metadata. Giữ chúng để audit và fail-closed khi active runtime đọc chúng.

**Complete when:** Shared helper đã được test, schema v4 đã validate đúng, và không còn caller tạo artifact THESIS mà thiếu calibration identity.

### Phase 2: THESIS model/scorer xuất official `point_scores = q`

**Result:** Model tính raw MSE trước, giữ raw diagnostics, và chỉ xuất transformed point anomaly score sau khi calibration state được cài đặt.

**Stages:**

1. **Stage 2.1 — Tách raw score boundary**
   - Xác định rõ raw per-sample MSE, raw MC mean `e`, transformed `q` và raw `window_scores`.
   - Giữ `aux.point_score_samples` ở raw semantics.

2. **Stage 2.2 — Gắn calibration state vào offline model**
   - Cho public THESIS model nhận hoặc giữ calibration state sau khi clean validation fit xong.
   - Khi state có mặt, forward áp dụng shared transform vào top-level `point_scores`.
   - Trong training hoặc trước calibration, không dùng sigmoid như training loss.

3. **Stage 2.3 — Đồng bộ online adapter**
   - Cho source path và projected path dùng cùng calibration state.
   - Bảo đảm `window_scores` vẫn là raw window reconstruction MSE.

4. **Stage 2.4 — Cập nhật output contract**
   - Kiểm tra rank, finite values và semantic range của official `point_scores`.
   - Phân biệt raw diagnostics với transformed anomaly score trong output payload.

5. **Stage 2.5 — Kiểm tra model-level vertical slice**
   - Chạy một forward deterministic/MC đã gắn calibration state.
   - Xác nhận `point_scores = sigmoid((e-c)/tau)`, không transform hai lần, và loss không đổi.

**Depends on:** Phase 1 hoàn tất shared helper và calibration object.

**Verification:**

- Automated: model shape/MC tests và online scorer tests pass.
- Manual: calibrated forward cho score trong `(0,1)`; `window_scores` vẫn raw.

**Risks:** Có thể transform hai lần nếu model và online window seam cùng áp dụng sigmoid. Chỉ shared scorer được sở hữu transform; các downstream seam chỉ consume canonical `q`.

**Complete when:** Offline model và online adapter cùng trả semantic score giống nhau cho cùng `e,c,tau`, và model-level tests khóa được contract này.

### Phase 3: Offline clean-validation calibration và threshold generation hoàn tất

**Result:** Offline benchmark fit calibration từ clean validation trước, sau đó tạo offline q99 và online-EWMA q99 từ transformed timelines.

**Stages:**

1. **Stage 3.1 — Thu raw clean-validation timeline**
   - Chạy non-overlap offline windows để thu raw point MSE.
   - Xác nhận entity grouping trước khi fit.
   - Không dùng synthetic validation hoặc test labels.

2. **Stage 3.2 — Fit và cài calibration state**
   - Fit `c`,`tau` từ raw clean-validation timeline.
   - Gắn state vào offline model và online calibration scorer.
   - Giữ lại raw values cần cho audit.

3. **Stage 3.3 — Tạo transformed offline và online clean timelines**
   - Tạo `q_offline` từ non-overlap clean validation.
   - Tạo stride-1 `q_online`, sau đó áp dụng absolute-index EWMA.
   - Không dùng một timeline để thay thế timeline còn lại.

4. **Stage 3.4 — Tạo threshold artifact**
   - Tính offline threshold `Q0.99(q_offline)`.
   - Tính online threshold `Q0.99(EWMA(q_online))`.
   - Ghi hai threshold riêng và cùng `c`,`tau` vào artifact.

5. **Stage 3.5 — Chạy offline vertical slice**
   - Đánh giá synthetic validation và test bằng transformed `point_scores`.
   - Kiểm tra threshold chỉ lấy từ clean validation.
   - Kiểm tra report và artifact có cùng checkpoint/entity/config identity.

**Depends on:** Phase 1 và Phase 2.

**Verification:**

- Automated: offline calibration/threshold tests và leakage tests pass.
- Manual: artifact thể hiện hai q99 thresholds, cùng `c`,`tau`, nhưng threshold values có thể khác nhau do timeline khác nhau.

**Risks:** Two-pass clean evaluation làm tăng số forward pass. Giới hạn two-pass ở calibration/evaluation boundary; không thêm MC branch vào training loop.

**Complete when:** Một offline evaluation-only combination tạo được artifact hợp lệ và các split downstream dùng đúng transformed score.

### Phase 4: V4 threshold recalibration tạo artifact mới từ raw clean validation

**Result:** Script recalibration không còn copy offline raw threshold cũ; artifact v4 mới được tạo từ calibration và cả hai transformed timelines.

**Stages:**

1. **Stage 4.1 — Khảo sát inventory và identity**
   - Xác định checkpoint, entity, variant, seed, config và artifact v3 tương ứng.
   - Từ chối entry thiếu hoặc mismatch trước khi ghi output.

2. **Stage 4.2 — Thu đủ hai raw calibration timelines**
   - Thu non-overlap offline raw scores.
   - Thu stride-1 online raw scores và raw auxiliary window/latent scores.

3. **Stage 4.3 — Fit và transform lại**
   - Fit `c`,`tau` từ offline clean raw timeline.
   - Transform offline timeline và online timeline; áp dụng EWMA chỉ trên online `q`.

4. **Stage 4.4 — Ghi artifact v4 và audit report**
   - Tính lại offline/online q99 thresholds.
   - Ghi calibration metadata, threshold values và checkpoint hash.
   - Không ghi đè artifact/output đã tồn tại.

5. **Stage 4.5 — Kiểm tra migration output**
   - Validate artifact bằng schema v4.
   - So sánh audit report với input identity và calibration counts.

**Depends on:** Phase 3 đã xác nhận offline threshold flow và Phase 1 artifact contract.

**Verification:**

- Automated: recalibration tests, identity tests và no-overwrite tests pass.
- Manual: audit report chứng minh offline threshold không được copy từ v3.

**Risks:** Kết quả benchmark cũ và mới không còn cùng score semantics. Giữ artifact cũ và provenance đầy đủ; không xóa output cũ.

**Complete when:** Một artifact v4 mới hợp lệ được tạo từ raw clean validation, có cả hai transformed thresholds và audit provenance.

### Phase 5: Online runtime dùng `q` trước EWMA và prediction

**Result:** Online runtime load calibration state từ artifact trước test stream; EWMA, triage, verification và prediction đều dùng transformed score.

**Stages:**

1. **Stage 5.1 — Load và validate artifact identity**
   - Load schema v4 artifact.
   - Validate entity, variant, seed, window size, checkpoint hash, EWMA weights và transform fields.
   - Cài `c`,`tau` vào model/scorer trước khi tạo test stream.

2. **Stage 5.2 — Đồng bộ A0/A1/A2 scoring**
   - Bảo đảm source A0 và projected A1/A2 cùng trả `q`.
   - Bảo đảm fallback scoring không quay về raw MSE hoặc fail-fast rõ ràng nếu thiếu calibration.

3. **Stage 5.3 — Đưa `q` vào absolute-index EWMA**
   - Gọi EWMA với `window_point_scores = q`.
   - Giữ state theo absolute index và trọng số `0.9/0.1`.
   - Không transform thêm ở EWMA hoặc downstream core.

4. **Stage 5.4 — Đồng bộ triage, verification và event record**
   - Prediction dùng `current_window_ewma_point_scores > threshold`.
   - Event record giữ semantic thống nhất giữa window score, EWMA score, threshold và legacy field names.
   - `window_scores` raw tiếp tục phục vụ input-window triage.

5. **Stage 5.5 — Chạy online vertical slice**
   - Chạy một online sequence nhỏ với artifact mới.
   - Kiểm tra score range, EWMA timeline, threshold lookup và strict prediction rule.

**Depends on:** Phase 3 có artifact semantics và Phase 4 có artifact v4 thực tế.

**Verification:**

- Automated: online calibration, EWMA, entrypoint, runtime identity và event record tests pass.
- Manual: một online event cho thấy `q`, EWMA(q), threshold artifact và prediction đúng.

**Risks:** Artifact raw-score cũ có thể làm runtime chạy sai nếu validator không fail-closed. Bắt buộc reject artifact thiếu transform metadata.

**Complete when:** Online test stream không tự calibration, load đúng artifact, và mọi prediction/triage score đều bắt nguồn từ q hoặc EWMA(q).

### Phase 6: Phân định official THESIS path với provisional/legacy paths

**Result:** Raw/legacy threshold logic không bị hiểu nhầm là official v3 anomaly threshold; baseline ngoài THESIS không bị thay đổi ngoài scope.

**Stages:**

1. **Stage 6.1 — Phân loại trainer threshold**
   - Xác định threshold trong training validation là provisional nếu chưa có calibration artifact.
   - Ngăn provisional positive-support threshold trở thành official v3 artifact threshold.

2. **Stage 6.2 — Kiểm tra evaluator fallback**
   - Xác định các caller official đã truyền transformed clean threshold.
   - Giữ legacy fallback chỉ cho path ngoài official v3 hoặc gắn nhãn rõ ràng.

3. **Stage 6.3 — Cập nhật test semantics**
   - Đổi assertion từ raw point score sang transformed anomaly score ở test THESIS.
   - Giữ test baseline và legacy contract độc lập.

4. **Stage 6.4 — Đồng bộ ontology và report fields**
   - Chỉ cập nhật spec/ontology nếu API implementation thực tế dùng tên khác với quyết định đã khóa.
   - Bảo đảm raw point MSE, `point_scores`, `window_scores` và threshold fields không bị dùng lẫn.

**Depends on:** Phase 2, Phase 3 và Phase 5 đã có output semantics thực tế.

**Verification:**

- Automated: regression tests cho model/evaluator/online/baselines pass.
- Manual: review một offline report và một online report để xác nhận score/threshold terminology.

**Risks:** Sửa quá rộng có thể làm baseline hoặc checkpoint monitoring thay đổi ngoài scope. Chỉ thay official THESIS path; giữ adapter/legacy boundary rõ ràng.

**Complete when:** Mọi official threshold đều truy được về clean-validation q timeline; legacy/provisional paths được phân loại và không ghi đè artifact v3.

### Phase 7: Kiểm thử toàn hệ thống và recalibrate artifacts

**Result:** Code mới pass focused tests, regression tests và một end-to-end combination trước khi mở rộng benchmark.

**Stages:**

1. **Stage 7.1 — Focused formula and contract tests**
   - Chạy unit tests cho MAD, sigmoid, q99, strict `>` và artifact schema.
   - Kiểm tra model raw/MC/window output contract.

2. **Stage 7.2 — Calibration/runtime integration tests**
   - Kiểm tra offline two-pass calibration.
   - Kiểm tra online stride-1 calibration, EWMA(q), artifact loading và no test-stream calibration.

3. **Stage 7.3 — Repository regression tests**
   - Chạy nhóm tests liên quan đến core, models, evaluation, online và ops.
   - Kiểm tra `git diff --check`.

4. **Stage 7.4 — One-combination end-to-end smoke**
   - Chạy một combination CPU/dry-run hoặc smoke theo config hiện hành.
   - Theo dõi model scoring, threshold artifact, online load và event output.

5. **Stage 7.5 — Recalibrate và kiểm tra acceptance**
   - Recalibrate artifacts sau khi code pass.
   - Kiểm tra q range, cùng `c`,`tau`, hai q99 thresholds riêng, EWMA weights và strict prediction.
   - Chỉ sau khi stage này pass mới xem xét benchmark matrix rộng.

**Depends on:** Phase 1 đến Phase 6.

**Verification:**

- Automated: focused tests, regression tests và one-combination smoke pass.
- Manual: artifact/report provenance và score timelines được kiểm tra bằng dữ liệu thật.

**Risks:** Threshold semantics mới làm kết quả benchmark thay đổi. Giữ output cũ, checkpoint hashes và artifact provenance để audit/rollback.

**Complete when:** Một end-to-end combination chứng minh raw MSE -> q -> threshold/EWMA -> prediction, và tất cả acceptance checks của full-spec-v3 pass.

## Dependency summary

| Phase | Requires | Enables |
| --- | --- | --- |
| Phase 1 | Spec v3 và research/plan | Calibration state và artifact contract dùng chung |
| Phase 2 | Phase 1 | Model/scorer xuất q đúng semantics |
| Phase 3 | Phase 1-2 | Offline thresholds và artifact mới |
| Phase 4 | Phase 1 và Phase 3 | V4 migration artifacts có provenance |
| Phase 5 | Phase 3-4 | Online runtime dùng q trước EWMA |
| Phase 6 | Phase 2-5 | Official/legacy boundary rõ ràng |
| Phase 7 | Phase 1-6 | Acceptance evidence và artifact recalibration |

## Decisions confirmed

- Raw point MSE được tính trước; anomaly score official là **shifted-and-scaled logistic sigmoid**.
- `c` là median của raw clean-validation point MSE.
- `tau` là MAD-based robust scale với normalizer `0.6745`.
- `tau <= 0` fail-fast vì spec không định nghĩa epsilon.
- Official inference dùng calibration state; training loss không dùng sigmoid.
- Offline và online dùng cùng entity-level `c`,`tau`, nhưng threshold là hai q99 của hai timelines khác nhau.
- EWMA giữ absolute-index semantics và trọng số `0.9 current + 0.1 previous`.
- `window_scores` giữ raw window reconstruction MSE.
- Active online runtime phải reject artifact thiếu transform metadata và không được calibrate từ test stream.

## Non-blocking uncertainties

- Vị trí cuối cùng của shared helper có thể đổi trong detailed plan nếu file-size gate hoặc codebase preference yêu cầu, nhưng không được tạo duplicate calibration implementation.
- Tên field legacy như `raw_point_score` có thể giữ để tương thích nếu ontology ghi rõ giá trị thực tế là transformed anomaly score.
- Cách retention raw MSE timeline có thể được quyết định ở detailed plan dựa trên audit/retention policy; quyết định này không làm đổi thứ tự phase.

## Feedback requested

- Anh xem giúp thứ tự 7 phase và stage như trên đã đúng với delivery path anh muốn chưa.
- Anh muốn giữ Phase 6 như một phase riêng để phân định legacy/provisional, hay gộp nó vào Phase 7 trước khi chuyển sang detailed plan?
- Có stage nào anh muốn tách nhỏ hơn hoặc outcome nào còn thiếu không?
