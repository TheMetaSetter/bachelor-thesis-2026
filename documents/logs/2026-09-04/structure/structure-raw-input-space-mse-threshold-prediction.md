---
date: 2026-09-04T14:04:15+07:00
topic: "Dùng MSE trong raw input space cho thresholding và prediction"
status: proposed
revision: 974af2b3a3d075f5cd4f3368f2cb584a5a8a3720
related_documents:
  - documents/logs/2026-09-04/plan/plan-raw-input-space-mse-threshold-prediction.md
  - documents/logs/2026-09-04/research/research-raw-input-space-mse-score-change-surface.md
  - prompts/3_structure_prompt.md
---

# Implementation Structure: Raw-input-space MSE cho thresholding và prediction

## Summary

Thay score vận hành hiện tại bằng MSE tính trên đơn vị sensor ban đầu. Hệ thống
sẽ dùng cùng một định nghĩa cho point-level score, window-level score,
threshold, prediction, EWMA và triage. Quy trình gồm sáu phase: khóa contract,
tạo scorer, chuyển offline path, cập nhật artifact, chuyển online path, rồi
kiểm chứng trên synthetic validation và histogram.

## Request

### Vấn đề hiện tại

Model nhận input đã standardize nên MSE hiện tại nằm trong normalized input
space. Một số đường đi còn dùng calibrated sigmoid score cho score vận hành.

### Kết quả cần đạt

- Tính raw_input_point_mse theo từng điểm và raw_input_window_mse theo từng
  window.
- Dùng raw-input MSE cho thresholding và prediction ở offline và online.
- Giữ normalized MSE dưới tên diagnostic rõ nghĩa.
- Không dùng calibrated sigmoid score trong raw protocol.
- Phân biệt rõ normal/anomalous point và normal/anomalous window trong artifact,
  metric và histogram.
- Chạy kiểm chứng trước trên một end-to-end combination, sau đó trên
  machine-1-6, machine-3-4 và machine-3-9.

### Loại trừ

- Không đổi reconstruction training loss.
- Không thêm trọng số channel vào simple MSE.
- Không sửa RedLamp hoặc baseline khác.
- Không xóa hoặc ghi đè artifact sigmoid lịch sử.

## Confirmed context

- SequenceStandardScaler đang biến đổi active features trước khi WindowDataset
  tạo batch.
- Checkpoint đã lưu scaler state, nhưng scorer downstream chưa inverse-transform
  input và reconstruction về sensor units.
- Model đã có reconstruction samples và đang lấy trung bình của MSE từng
  sample MC; contract mới phải giữ quy tắc này.
- Offline evaluator hiện threshold một point-score timeline.
- Online calibration, EWMA và triage hiện nhận score normalized/calibrated ở các
  boundary khác nhau.
- Threshold artifact hiện yêu cầu các trường dành riêng cho sigmoid, nên chưa
  biểu diễn đầy đủ raw-input score identity.

## Scope

### In scope

- Contract và terminology cho normalized/raw-input MSE.
- Inverse transform và scorer cho point/window, gồm MC aggregation.
- Offline thresholding, prediction, metric và artifact export.
- Raw-score threshold artifact và kiểm tra score-space mismatch.
- Online calibration, EWMA, triage, prediction và event state.
- Test, synthetic validation, histogram và documentation.

### Out of scope

- Thay đổi dữ liệu huấn luyện hoặc reconstruction objective.
- So sánh trực tiếp threshold raw giữa các machine nếu chưa có quy tắc chuẩn
  hóa mới.
- Dùng threshold normalized cho raw-input score.
- Dùng lại artifact sigmoid cho raw protocol.

## Proposed phases

### Phase 1: Khóa contract raw-input MSE

**Result:** Mọi thành phần dùng cùng định nghĩa score space, công thức MSE và
MC aggregation; raw protocol không còn mơ hồ giữa “raw trước calibration” và
“raw sensor units”.

**Sequential stages:**

1. **Stage 1.1 — Đối chiếu terminology và specification.** So sánh các
   specification liên quan, xác định tên cũ, tên mới, semantic status và runtime
   owner cho từng score.
2. **Stage 1.2 — Khóa công thức score.** Ghi rõ MSE trung bình đều trên feature,
   window score là trung bình theo time, và MC score là trung bình của MSE từng
   reconstruction sample.
3. **Stage 1.3 — Khóa score identity của protocol.** Xác định
   score_space: raw_input và point_score_transform: identity là điều kiện
   của raw protocol; normalized score chỉ là diagnostic.

**Depends on:** Plan raw-input MSE và research change-surface đã xác nhận rằng
MSE hiện tại là normalized-space MSE.

**Affected responsibilities:** Specification, terminology, protocol config và
contract giữa model với evaluator/online engine.

**Constraints:** Giữ specification lịch sử và không đổi training loss.

**Verification:**

- Automated: Config/contract test đọc được raw score identity và không yêu cầu
  sigmoid fields cho raw protocol.
- Manual: Đọc score table và kiểm tra mọi định nghĩa threshold/prediction dùng
  raw-input MSE.

**Risks:** Tên raw cũ có thể tiếp tục gây nhầm với raw sensor units. Structure
xử lý bằng tên có input_space ở mọi score vận hành mới.

**Complete when:** Contract, terminology mapping và protocol identity đã được
thống nhất, không còn blocking decision ảnh hưởng phase order.

### Phase 2: Tạo scaler-aware score primitives

**Result:** Hệ thống tính được raw point/window MSE từ scaled input và
reconstruction mà không thay đổi model training path.

**Sequential stages:**

1. **Stage 2.1 — Xác lập inverse-transform boundary.** Cho scorer sử dụng
   training scaler để đưa input và reconstruction về raw sensor units; active và
   inactive features phải giữ đúng semantics.
2. **Stage 2.2 — Tạo point/window raw scorer.** Tính point MSE theo feature và
   window MSE theo time; giữ normalized MSE như một nhánh diagnostic có tên rõ.
3. **Stage 2.3 — Tích hợp MC aggregation.** Tính MSE riêng cho từng
   reconstruction sample rồi lấy trung bình; không lấy MSE của reconstruction
   trung bình.
4. **Stage 2.4 — Kiểm chứng số học cơ bản.** Dùng tensor nhỏ có giá trị tính tay
   để kiểm tra inverse transform, active/inactive mask, shape và MC result.

**Depends on:** Phase 1 đã khóa công thức và score identity.

**Affected responsibilities:** Scaler, reconstruction scoring boundary và score
contract dùng chung bởi offline/online path.

**Constraints:** Bảo toàn device, dtype, batch/sequence shape và hành vi scaler
đã fit; không truyền raw tensor vào model nếu không cần.

**Verification:**

- Automated: Scaler round-trip, hand-computed MSE, MC aggregation và finite-score
  tests đều pass.
- Manual: Kiểm tra một batch sau injection để xác nhận input được inverse-transform
  sau khi anomaly nhân tạo đã được thêm.

**Risks:** Inverse transform hai lần sẽ làm sai scale score. Structure yêu cầu
scorer nhận diện rõ input space và test đối chiếu normalized/raw trên cùng toy
tensor.

**Complete when:** Một scorer dùng chung trả đúng raw point/window MSE và có test
số học độc lập với evaluator/online engine.

### Phase 3: Chuyển offline evaluation sang raw score

**Result:** Clean-validation threshold, synthetic/test prediction, metric và
score export offline đều dùng raw-input MSE.

**Sequential stages:**

1. **Stage 3.1 — Nối scorer vào evaluator.** Đưa scaler/scorer context vào
   boundary offline và tính raw scores từ batch thực tế cùng reconstruction.
2. **Stage 3.2 — Bảo toàn overlap aggregation.** Gộp raw point timelines và
   normalized diagnostic timelines bằng cùng quy tắc window overlap và coverage.
3. **Stage 3.3 — Chuyển threshold và prediction.** Dùng raw point score cho
   point threshold/prediction và raw window score cho window decision; label
   normal/anomalous phải độc lập với score space.
4. **Stage 3.4 — Xuất artifact song song có provenance.** Ghi raw point/window
   scores, normalized diagnostics, labels, threshold source và score identity để
   downstream reporting không đoán tên field.

**Depends on:** Phase 1 và Phase 2.

**Affected responsibilities:** Offline evaluator, benchmark orchestration,
threshold calibration, metrics và artifact export.

**Constraints:** Synthetic anomaly phải được inject trước khi tính raw input;
không lấy raw sequence trước injection làm input score.

**Verification:**

- Automated: Raw overlap averaging, threshold selection, post-injection score
  change, label alignment và artifact export tests.
- Manual: So sánh một clean window và một synthetic window; xác nhận quyết định
  dùng raw MSE còn normalized field chỉ để chẩn đoán.

**Risks:** Evaluator có thể vẫn đọc top-level calibrated point_scores. Structure
yêu cầu operational selection đi qua raw scorer và test fail nếu chỉ còn
normalized field.

**Complete when:** Một offline run nhỏ tạo được raw point/window threshold và
prediction đúng, đồng thời giữ được diagnostic normalized outputs.

### Phase 4: Persist raw threshold artifact và migration boundary

**Result:** Artifact mới chứng minh được score space/transform của threshold và
từ chối artifact không tương thích; artifact lịch sử vẫn giữ nguyên.

**Sequential stages:**

1. **Stage 4.1 — Định nghĩa raw artifact schema.** Bổ sung score space, identity
   transform, score definitions, threshold source split và provenance cần thiết
   cho raw-input MSE.
2. **Stage 4.2 — Thêm validation và mismatch rejection.** Từ chối artifact
   normalized-space, sigmoid transform, thiếu score identity hoặc sai checkpoint/
   config khi chạy raw protocol.
3. **Stage 4.3 — Giữ compatibility lịch sử.** Cho phép đọc artifact cũ trong
   historical replay, nhưng không tự động dùng chúng cho raw thresholding hoặc
   prediction.
4. **Stage 4.4 — Liên kết artifact với offline output.** Đảm bảo threshold được
   tạo ở Phase 3 và artifact lưu được đúng score definitions, machine, checkpoint
   và config provenance.

**Depends on:** Phase 1 đã khóa schema semantics; Phase 3 đã cung cấp raw
threshold outputs cần persist.

**Affected responsibilities:** Threshold artifact validation, checkpoint/config
provenance và runtime artifact resolution.

**Constraints:** Không sửa hoặc ghi đè artifact sigmoid cũ; raw artifact phải có
schema/version riêng theo convention của repository.

**Verification:**

- Automated: Raw artifact round-trip, legacy-read, mismatch-rejection và
  provenance tests.
- Manual: Mở một artifact mới và xác nhận không có yêu cầu sigmoid-specific cho
  raw protocol.

**Risks:** Config cũ có thể trỏ tới artifact sigmoid. Runtime phải báo lỗi rõ
thay vì âm thầm đổi score hoặc tái sử dụng artifact.

**Complete when:** Raw artifact được tạo/đọc/kiểm tra thành công và historical
artifact vẫn không thay đổi.

### Phase 5: Chuyển online EWMA, triage và prediction sang raw score

**Result:** Mọi quyết định online dùng raw-input MSE và threshold cùng score
space, trong khi event/state vẫn lưu diagnostic normalized score riêng.

**Sequential stages:**

1. **Stage 5.1 — Đưa scaler và score identity vào runtime context.** Đảm bảo
   online scorer có training scaler và runtime biết artifact đang dùng raw
   protocol.
2. **Stage 5.2 — Chuyển online calibration.** Tính raw point/window score từ
   reconstruction samples và dùng chúng cho clean-validation, EWMA và input-window
   threshold calibration.
3. **Stage 5.3 — Chuyển EWMA và triage.** Cập nhật EWMA bằng raw point MSE; dùng
   raw window MSE cho admission, triage và các threshold comparison.
4. **Stage 5.4 — Chuyển prediction và event state.** Ghi raw operational score,
   normalized diagnostic score và score identity vào event/step state; loại bỏ
   fallback âm thầm về calibrated outputs.point_scores.

**Depends on:** Phase 2 scorer, Phase 4 raw artifact và các shape/index contract
hiện có của online engine.

**Affected responsibilities:** Online calibration, stride-1 scoring, EWMA,
triage, prediction, event record và runtime continuation.

**Constraints:** Giữ causal indexing, absolute-index EWMA state và prediction
shape; không để normalized diagnostic điều khiển quyết định.

**Verification:**

- Automated: Raw point/window calibration, EWMA, triage, event-state,
  continuation và score-identity mismatch tests.
- Manual: Trace một online event từ reconstruction tới threshold comparison và
  xác nhận mọi quyết định dùng raw-input units.

**Risks:** Một fallback path có thể bypass scorer. Test phải kiểm tra rằng online
path không chạy được khi chỉ có normalized/calibrated operational field.

**Complete when:** Online threshold, EWMA, triage, prediction và persisted event
state đều dùng raw score với artifact tương thích.

### Phase 6: Kiểm chứng synthetic validation và histogram

**Result:** Ba machine được đánh giá bằng raw protocol; output phân biệt được
normal/anomalous point và normal/anomalous window; histogram hiển thị đúng raw
MSE cùng threshold.

**Sequential stages:**

1. **Stage 6.1 — Chạy focused regression tests.** Chạy score, evaluator,
   artifact và online tests sau khi các phase trước hoàn tất.
2. **Stage 6.2 — Chạy một end-to-end smoke combination.** Xác nhận loader,
   checkpoint/scaler, synthetic injection, scoring, threshold và export đi qua
   trọn flow trước khi mở rộng.
3. **Stage 6.3 — Chạy synthetic validation cho ba machine.** Re-run lần lượt
   machine-1-6, machine-3-4 và machine-3-9; lưu threshold, score summary,
   labels, provenance và checkpoint references theo canonical output hierarchy.
4. **Stage 6.4 — Tạo và kiểm tra histogram.** Tạo histogram riêng cho raw
   point-level và raw window-level score, tách normal/anomalous observations và
   đánh dấu threshold; không dùng sigmoid value làm operational score.
5. **Stage 6.5 — Audit kết quả cuối.** Kiểm tra finite arrays, shape, score-space
   metadata, label alignment, artifact provenance và sự nguyên vẹn của artifact
   lịch sử.

**Depends on:** Phase 3, Phase 4 và Phase 5.

**Affected responsibilities:** Test suite, benchmark execution, synthetic
validation, plotting và experiment reporting.

**Constraints:** Chạy smoke trước matrix ba machine; calibrate threshold theo
từng entity; không so sánh trực tiếp độ lớn raw threshold giữa machine nếu
chưa có quy tắc mới.

**Verification:**

- Automated: Focused tests, end-to-end smoke, finite/shape checks và raw artifact
  metadata checks.
- Manual: Inspect sáu nhóm histogram theo machine và score level; xác nhận bốn
  nhóm normal/anomalous point/window được gắn nhãn và threshold đúng.

**Risks:** Raw sensor units khác nhau giữa machine và có thể làm score magnitude
không so sánh trực tiếp. Báo cáo phải giữ entity provenance và threshold riêng.

**Complete when:** Smoke pass, ba machine pass, histogram dùng raw MSE, và final
audit xác nhận không có raw-protocol execution nào dùng calibrated sigmoid.

## Dependency summary

| Phase | Requires | Enables |
| --- | --- | --- |
| 1. Contract | Research và plan đã xác định change boundary | Scorer và protocol dùng cùng semantics |
| 2. Primitives | Contract raw-input MSE | Offline và online score calculation |
| 3. Offline | Primitives và protocol identity | Raw threshold/prediction outputs |
| 4. Artifact | Raw threshold outputs và score identity | Runtime kiểm tra artifact tương thích |
| 5. Online | Primitives, raw artifact và runtime context | Raw EWMA, triage và prediction |
| 6. Validation | Offline/online path đã chuyển đổi | Kết quả synthetic, histogram và audit cuối |

## Decisions confirmed

- raw_input nghĩa là sensor units trước standardization.
- Simple MSE lấy trung bình đều trên feature.
- Window MSE lấy trung bình theo time từ point MSE.
- MC score lấy trung bình của per-sample MSE.
- Raw-input MSE điều khiển threshold, prediction, EWMA và triage.
- Calibrated sigmoid không thuộc raw operational path.
- Normalized MSE chỉ giữ dưới diagnostic names rõ nghĩa.
- Historical sigmoid artifacts không bị xóa và không được tự động tái sử dụng.

## Non-blocking uncertainties

- Số version cụ thể và tên field cuối cùng của raw artifact phải theo convention
  hiện có khi viết detail plan.
- Có thể giữ model top-level output hiện tại cho compatibility, nhưng evaluator và
  online engine phải chọn raw scorer ở operational boundary.
- Raw MSE có thể bị channel có độ lớn vật lý lớn chi phối; đây là giới hạn của
  equal-feature raw MSE và phải được ghi trong báo cáo.

## Feedback requested

- Anh xác nhận thứ tự sáu phase và các stage tuần tự này trước khi em chuyển sang
  tài liệu detail với file, interface, test case và command cụ thể.
- Anh có muốn tách Phase 6 thành một structure riêng cho benchmark/visualization
  không, hay giữ chung như hiện tại?
