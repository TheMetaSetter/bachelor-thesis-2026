---
date: 2026-08-09T00:00:00+07:00
planner: OpenAI Codex
topic: "Lập trình anomaly score theo full-spec-v3"
status: ready
revision: b004e70b26b956809695c1b9d9518adf900ed2e9
branch: dev
related_research: documents/logs/2026-08-09/research/research-anomaly-score-implementation-gaps.md
---

# Implementation Plan: Lập trình anomaly score theo full-spec-v3

## Summary

Plan này chuyển runtime THESIS từ raw point MSE sang anomaly score `q` theo **shifted-and-scaled logistic sigmoid**:

```text
raw point MSE e
  -> c = median(clean-validation e)
  -> tau = MAD(clean-validation e) / 0.6745
  -> q = sigmoid((e - c) / tau)
  -> offline q99 threshold
     hoặc online EWMA(q) rồi q99 threshold
```

Các phase đi theo thứ tự: khóa contract và calibration API, xây calibration/artifact, tích hợp model scorer, sửa offline calibration, sửa online runtime, xử lý legacy/provisional paths, rồi kiểm thử và recalibrate artifacts.

## Request

Đọc `prompts/2_plan_prompt.md` và viết các phase tổng quan, theo thứ tự, để chỉnh sửa và lập trình theo `documents/logs/2026-08-09/research/research-anomaly-score-implementation-gaps.md`.

Ràng buộc:

- Chỉ lập kế hoạch; chưa sửa source code, config, test hoặc threshold artifact.
- Phạm vi là runtime THESIS v3: offline benchmark, online TTA và threshold artifact.
- Giữ `window_scores` là raw window reconstruction MSE.
- Giữ `point_score_loss` và reconstruction loss trên raw reconstruction error; sigmoid chỉ dùng cho inference, timeline và threshold calibration.
- Giữ EWMA absolute-index với trọng số `0.9 current + 0.1 previous`.

## Current state

- `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:182-227` và `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py:249-270` hiện xuất mean raw point MSE ở top-level `point_scores`.
- `src/models/online_impl/online_adaptation_helpers.py:77-86` cũng xuất raw point MSE cho online A0/A1/A2.
- `src/engine/online_tta/online_calibration.py:61-154` hiện calibration trên raw point scores và đưa raw scores vào EWMA.
- `scripts/benchmarks/run_thesis_offline_benchmark.py:434-452,505-565` hiện tính threshold từ raw clean-validation scores.
- `scripts/ops/recalibrate_thesis_threshold_artifacts_v4.py:226-365` hiện recalibrate online raw timeline và copy offline threshold cũ.
- `src/protocols/threshold_artifact.py:20-374` chưa lưu `c`, `tau` hoặc transform identity.
- `src/engine/online_tta/point_ewma.py:8-34` đã có công thức EWMA đúng với spec; phase implementation chỉ thay đổi dữ liệu đầu vào của EWMA.

Evidence normative nằm ở `documents/spec/full-spec-v3.md:545-604,798-835,898-907`.

## Desired end state

1. Calibration lấy raw clean-validation point MSE theo entity.
2. Calibration lưu:
   - `c = median(raw clean-validation point MSE)`;
   - `tau = MAD(raw clean-validation point MSE) / 0.6745`;
   - transform identity `shifted-and-scaled logistic sigmoid`.
3. Official inference sau khi được gắn calibration state xuất `point_scores = q` trong `(0, 1)`.
4. `aux.point_score_samples` vẫn là raw per-sample point MSE để giữ đúng MC semantics và uncertainty diagnostics.
5. `window_scores` vẫn là raw window reconstruction MSE.
6. Offline threshold là `Q0.99` của transformed non-overlap clean-validation timeline.
7. Online threshold là `Q0.99` của absolute-index `EWMA(q)` trên stride-1 clean-validation timeline.
8. Online runtime đọc `c`, `tau`, transform identity và threshold từ artifact; runtime không calibrate từ test stream.
9. Một điểm chỉ bị đánh dấu anomaly khi score `>` threshold.

## Scope

### In scope

- Shared MAD-based calibration và shifted-and-scaled logistic sigmoid.
- Model/output seam của THESIS offline và online adapter.
- Offline clean-validation two-pass calibration.
- Online stride-1 calibration, EWMA input và runtime artifact loading.
- Schema/validation/provenance của threshold artifact v4.
- Official THESIS evaluator threshold path.
- Focused unit, integration và one-combination smoke verification.
- Recalibration các threshold artifacts sau khi code pass test.

### Out of scope

- Thay đổi reconstruction loss, `point_score_loss`, model architecture hoặc MC sampling protocol.
- Thay đổi `window_scores`, latent-window triage thresholds hoặc EWMA weights.
- Sửa RedLamp, traditional baselines hoặc online baseline calibration riêng.
- Tối ưu performance ngoài phần cần thiết để không transform hai lần.
- Chạy toàn bộ benchmark matrix trước khi one-combination smoke pass.

## Implementation decisions locked for this plan

### Calibration state

Official inference sẽ dùng một calibration state gắn với model/scorer sau khi fit. Model vẫn có thể xuất raw diagnostic score trong training hoặc trước calibration, nhưng official offline/online path chỉ sử dụng top-level `point_scores = q` sau khi calibration state đã được cài đặt.

Lý do: current model forward không nhận `c`,`tau`, trong khi clean validation phải được chạy một lần để ước lượng chúng. Cách này giữ một scorer path chung và tránh viết riêng một sigmoid ở offline evaluator, online calibration và online runtime.

### Shared helper ownership

Đề xuất thêm `src/protocols/point_score_calibration.py` làm helper dùng chung cho calibration object, fit và transform. Đây là file mới được đề xuất; repository hiện chưa có file này. Helper phải hỗ trợ các tensor/array được dùng ở model, calibration và artifact construction, hoặc cung cấp adapter rõ ràng giữa torch và NumPy mà không lặp công thức.

### Non-positive robust scale

Nếu `tau <= 0` sau khi tính MAD, calibration fail-fast bằng lỗi rõ ràng. Spec không định nghĩa epsilon, nên implementation không tự thêm epsilon âm thầm. Test phải khóa hành vi này.

### Entity identity

Mỗi threshold artifact vẫn tương ứng một entity. Nếu một calibration loader chứa nhiều entity, implementation phải fail-fast hoặc tách calibration theo entity trước khi ghi artifact; không dùng `_first_entity_id` để che khuất nhiều entity.

## Phase 1: Khóa shared calibration và artifact contract

### Goal

Có một calibration object và threshold artifact schema v4 chứa đầy đủ transform identity trước khi nối vào model hoặc runtime.

### Changes

#### 1. Shared calibration helper

- **File:** `src/protocols/point_score_calibration.py` — proposed new file.
- **Symbols:** proposed `PointScoreCalibration`, `fit_mad_logistic_calibration`, `transform_point_scores`.
- **Change:** Implement median center, MAD, fixed normalizer `0.6745`, fail-fast khi input không finite hoặc `tau <= 0`, và shifted-and-scaled logistic sigmoid.
- **Reason:** Một công thức duy nhất phải được dùng cho offline, online calibration và runtime.
- **Dependencies:** `src/models/...`, `src/engine/online_tta/online_calibration.py`, threshold scripts và tests.

#### 2. Threshold artifact schema

- **File:** `src/protocols/threshold_artifact.py`.
- **Symbols:** `validate_threshold_artifact`, `build_threshold_artifact`.
- **Change:** Thêm và validate `point_score_transform`, `point_score_c`, `point_score_tau`, `point_score_tau_estimator`, `point_score_mad_normalizer`. Schema v4 bắt buộc các field này; artifact legacy không được dùng trong active THESIS v3 online runtime.
- **Reason:** Runtime phải dùng đúng calibration đã tạo threshold.
- **Dependencies:** mọi caller của `build_threshold_artifact`, threshold fixtures và artifact loaders.

### Verification

#### Automated

- [ ] `tests/engine/test_threshold_artifact.py` — round-trip giữ đúng transform fields và reject artifact thiếu/sai fields.
- [ ] `tests/online/test_threshold_artifact.py` — kiểm tra `tau <= 0`, sai transform name và v4 identity.
- [ ] Unit tests proposed for `src/protocols/point_score_calibration.py` — kiểm tra median, MAD/0.6745, monotonic sigmoid, finite output và fail-fast.

#### Manual

- [ ] Đọc một artifact JSON mẫu — thấy `c`, `tau`, estimator, normalizer và transform name cùng provenance.

### Risks

- Artifact cũ thiếu transform metadata. Mitigation: active v3 loader fail-closed; giữ file cũ nguyên vẹn và tạo artifact mới sau recalibration.

## Phase 2: Tích hợp calibration vào THESIS model/scorer

### Goal

Model tính raw MSE một lần, giữ raw diagnostics, và chỉ xuất transformed `point_scores` khi official calibration state đã được cài đặt.

### Changes

#### 1. Deterministic/MC offline path

- **Files:** `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py`, `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py`, public entrypoint `src/models/thesis_multitask.py`.
- **Symbols:** `_build_monte_carlo_forward_outputs`, `forward`, model setup/state.
- **Change:** Giữ `point_score_samples` và mean raw `e`; áp dụng shared transform vào official top-level `point_scores` khi calibration state tồn tại. `window_scores` tiếp tục lấy raw window reconstruction MSE.
- **Reason:** Đáp ứng output contract mà không biến sigmoid thành training loss.
- **Dependencies:** evaluator, trainer diagnostics, MC shape tests.

#### 2. Online adapter path

- **File:** `src/models/online_impl/online_adaptation_helpers.py`.
- **Symbol:** `ThesisMultitaskEncoderAdapter.score_from_hidden`.
- **Change:** Dùng cùng calibration state/helper khi tính `point_scores`; giữ raw reconstruction calculation để `window_scores` và loss không đổi.
- **Reason:** A0 source path và A1/A2 projected path phải dùng cùng semantics.
- **Dependencies:** `OnlineAdaptationModel.forward_source`, `OnlineAdaptationModel.forward`, online window scoring.

#### 3. Output contract

- **File:** `src/core/contracts.py`.
- **Symbol:** `validate_model_outputs`.
- **Change:** Bổ sung kiểm tra semantic cho calibrated official scores và giữ raw MC sample tensors trong `aux`; không ép training path có calibration state nếu training contract vẫn cần raw diagnostics.
- **Reason:** Phát hiện score range/shape sai tại biên model.

### Verification

#### Automated

- [ ] `tests/models/test_multitask_shapes.py` — raw sample mean chỉ dùng làm `e`; top-level `point_scores` khớp sigmoid khi calibration state được cài đặt.
- [ ] Existing model shape/forward tests — `window_scores` vẫn là raw window MSE và output shapes không đổi.
- [ ] Online model tests — source và projected scorer cùng áp dụng một calibration state.

#### Manual

- [ ] Chạy một forward calibrated — xác nhận `point_scores` nằm trong `(0,1)`, `window_scores` không bị sigmoid, và không có sigmoid lần hai.

### Risks

- Nếu calibration state bị thiếu, model có thể xuất raw score ở official path. Mitigation: official benchmark/runtime phải fail-fast trước scoring; test context phải kiểm tra state đã được cài đặt.

## Phase 3: Rebuild offline clean-validation calibration và thresholds

### Goal

Offline benchmark tạo calibration state trước, sau đó tạo transformed clean-validation timelines và hai threshold artifact values đúng spec.

### Changes

#### 1. Offline benchmark orchestration

- **File:** `scripts/benchmarks/run_thesis_offline_benchmark.py`.
- **Symbols:** `_evaluate_offline_benchmark_splits`, `_build_thresholds`, `collect_offline_artifact_inputs`.
- **Change:** Tổ chức two-pass flow: collect raw clean-validation MSE -> fit `c`,`tau` -> gắn calibration state -> re-evaluate/collect transformed clean scores -> evaluate synthetic/test bằng transformed `point_scores` và threshold clean-only.
- **Reason:** Không thể fit `c`,`tau` từ score đã biến đổi; clean validation phải là nguồn duy nhất.
- **Dependencies:** evaluator, model state, online calibration collector, artifact builder.

#### 2. Threshold values

- **File:** `scripts/benchmarks/run_thesis_offline_benchmark.py`.
- **Symbol:** `_build_thresholds`.
- **Change:** Dùng `Q0.99(q_offline_clean)` cho offline threshold; dùng `Q0.99(EWMA(q_online_clean))` cho online threshold; truyền cùng calibration fields vào `build_threshold_artifact`.
- **Reason:** Hai timeline có cấu trúc khác nhau nên giữ hai threshold riêng.
- **Dependencies:** existing `select_clean_validation_point_threshold`, `select_online_ewma_threshold`, EWMA weights.

#### 3. Evaluator input

- **File:** `src/engine/evaluator.py`.
- **Symbols:** `evaluate`, `pointwise_batch_payloads` flow.
- **Change:** Official THESIS evaluation chỉ nhận calibrated `point_scores`; không để default raw positive-support threshold tự trở thành official threshold.
- **Reason:** Evaluator đang tích lũy score trực tiếp từ model output.
- **Dependencies:** offline benchmark and trainer provisional metrics.

### Verification

#### Automated

- [ ] Offline calibration tests — `c`,`tau` chỉ lấy từ raw clean validation.
- [ ] Threshold tests — offline và online thresholds đều là q99 trên transformed timelines.
- [ ] Regression tests — synthetic/test labels không được đi vào calibration.

#### Manual

- [ ] Kiểm tra artifact sau một evaluation-only run — hai threshold khác nhau nếu hai timelines khác nhau, và cả hai có cùng `c`,`tau`.

### Risks

- Re-evaluation clean validation làm tăng số forward pass. Mitigation: chỉ chạy two-pass ở calibration/evaluation boundary; không thêm MC branch vào trainer.
- Loader chứa nhiều entity có thể làm calibration lẫn entity. Mitigation: validate entity grouping trước khi fit và ghi artifact.

## Phase 4: Rework v4 recalibration and migrate threshold artifacts

### Goal

Script recalibration tạo lại artifact v4 từ raw clean validation thay vì copy offline threshold cũ.

### Changes

#### 1. Collect both offline and online raw timelines

- **File:** `scripts/ops/recalibrate_thesis_threshold_artifacts_v4.py`.
- **Symbols:** `_collect_clean_validation_scores`, `_calibrate_threshold_values`.
- **Change:** Collect non-overlap raw offline scores cùng stride-1 online raw scores; fit calibration từ raw offline clean-validation timeline; transform cả hai timeline; tính lại offline q99 và online EWMA q99.
- **Reason:** Artifact v4 phải phản ánh thiết kế mới cho cả hai threshold.
- **Dependencies:** `collect_nonoverlap_offline_scores`, `collect_stride1_online_scores`, shared helper.

#### 2. Build artifact fields

- **File:** `scripts/ops/recalibrate_thesis_threshold_artifacts_v4.py`.
- **Symbols:** `_v4_artifact_fields`, `build_v4_threshold_artifact`.
- **Change:** Không copy `artifact_v3["offline_point_threshold_nonoverlap"]`; truyền threshold mới và calibration metadata vào builder.
- **Reason:** Threshold cũ được tính trên raw score và không còn tương thích.

### Verification

#### Automated

- [ ] `tests/ops/test_recalibrate_thesis_threshold_artifacts_v4.py` — offline threshold mới, online threshold mới, cùng `c`,`tau`, và valid schema v4.
- [ ] Recalibration dry/preflight tests — script không ghi đè artifact output đã tồn tại.

#### Manual

- [ ] So sánh audit report — có raw calibration count, `c`, `tau`, transformed threshold values và checkpoint identity.

### Risks

- Artifact migration có thể làm kết quả benchmark cũ không so sánh trực tiếp với artifact mới. Mitigation: lưu artifact mới với provenance rõ ràng, không sửa artifact cũ, ghi protocol version trong report.

## Phase 5: Integrate online runtime and EWMA input

### Goal

Online runtime sử dụng đúng calibration artifact trước khi stream test bắt đầu; EWMA, triage, verification và prediction đều nhận transformed score.

### Changes

#### 1. Load and validate calibration identity

- **File:** `src/engine/online_tta/online_engine_run.py`.
- **Symbols:** `_validate_online_artifact_identity`, `_build_runtime_online_context`.
- **Change:** Validate transform name, `c`, `tau`, MAD normalizer và artifact schema cùng entity/checkpoint/EWMA identity; cài calibration state vào model/scorer trước stream.
- **Reason:** Online không được tự calibration từ test stream và không được dùng artifact không khớp.

#### 2. Score extraction and EWMA

- **Files:** `src/engine/online_tta/online_engine_window_metrics.py`, `src/engine/online_tta/online_engine_window_core.py`, `src/engine/online_tta/point_ewma.py`.
- **Symbols:** `_extract_online_window_scores`, `_score_online_window`, `_prepare_online_window_event`.
- **Change:** Đảm bảo `window_point_scores` đã là `q` trước khi gọi `update_window_point_ewma`; không áp dụng transform lần hai trong metrics/core; giữ EWMA state và absolute-index behavior.
- **Reason:** Spec yêu cầu EWMA(q), không phải EWMA(raw MSE).

#### 3. Fallback and event record

- **File:** `src/engine/online_tta/online_engine_step.py`.
- **Symbol:** `_compute_step_scores`.
- **Change:** Fallback phải dùng canonical transformed score hoặc fail-fast nếu scorer chưa calibrated; `raw_point_score` legacy field phải được ghi rõ là transformed point anomaly score nếu contract giữ nguyên tên.
- **Reason:** Tránh một nhánh online dùng raw score còn nhánh chính dùng q.

### Verification

#### Automated

- [ ] `tests/online/test_online_calibration_contract.py` — calibration path dùng q trước EWMA.
- [ ] `tests/online/test_online_ewma_threshold.py` — EWMA toán học không đổi.
- [ ] Online entrypoint/runtime tests — artifact params được load, identity mismatch bị reject, test stream không được dùng để fit.
- [ ] Online record tests — `prediction == ewma_point_score > threshold` và score timeline nhất quán.

#### Manual

- [ ] Kiểm tra một online event — `window_point_scores` thuộc `(0,1)`, EWMA thuộc `(0,1)`, threshold là artifact online q99, prediction dùng strict `>`.

### Risks

- Double transformation làm score sai nhưng vẫn nằm trong `(0,1)`. Mitigation: chỉ cho shared scorer sở hữu transform; add test gọi overlapping window và assert expected q/EWMA values.
- Artifact cũ bị load vào runtime. Mitigation: v4 validator và runtime identity check fail-closed.

## Phase 6: Reconcile provisional, legacy and documentation boundaries

### Goal

Các path không phải official THESIS v3 không vô tình ghi đè hoặc được hiểu là anomaly threshold mới.

### Changes

#### 1. Trainer provisional metrics

- **Files:** `src/engine/trainer.py:482-516`, `src/engine/thresholding.py:15-47`.
- **Change:** Xác định rõ threshold trong training validation là provisional/legacy nếu chưa có clean-validation calibration state; official benchmark không dùng positive-support fallback. Nếu cần, đổi tên metadata để phân biệt provisional threshold với artifact threshold.
- **Reason:** Không để threshold raw trong trainer bị báo cáo như threshold v3.

#### 2. Test and contract naming

- **Files:** `tests/models/test_multitask_shapes.py`, `tests/evaluation/test_point_score_contracts.py`, `tests/online/*`.
- **Change:** Đổi assertions từ “point score bằng raw MSE mean” sang “raw MSE mean được transform thành q”; giữ tên legacy chỉ khi contract còn tương thích và ghi semantic mới.
- **Reason:** Test phải bảo vệ đúng ontology v3.

#### 3. Documentation alignment

- **Files:** `documents/spec/full-spec-v3.md`, `documents/spec/offline_pretraining_terminology_ontology.md`, `documents/spec/online_tta_terminology_ontology.md`.
- **Change:** Chỉ cập nhật nếu implementation chọn tên field/API khác với các quyết định đã ghi; không thay đổi thiết kế đã khóa.
- **Reason:** Giữ documents là SSOT nhưng tránh chỉnh docs không cần thiết.

### Verification

#### Automated

- [ ] Regression tests cho legacy baseline vẫn pass và không nhận artifact THESIS v4 ngoài scope.
- [ ] Contract tests phân biệt raw MSE, transformed point anomaly score và raw window MSE.

#### Manual

- [ ] Review một report offline và một report online — không còn dùng cùng một threshold cho hai timeline; field names khớp ontology.

## Phase 7: Full verification and artifact recalibration

### Goal

Chứng minh code mới đúng ở unit, integration và một end-to-end combination trước khi chạy benchmark rộng.

### Changes

- Không thêm behavior mới. Chạy test, kiểm tra artifact, rồi recalibrate các artifact THESIS theo script đã sửa.
- Giữ output cũ để rollback/audit; không xóa output hoặc threshold artifact hiện có.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest tests/models/test_multitask_shapes.py tests/online/test_online_calibration_contract.py tests/online/test_online_ewma_threshold.py tests/online/test_threshold_artifact.py tests/engine/test_threshold_artifact.py tests/ops/test_recalibrate_thesis_threshold_artifacts_v4.py` — focused score/calibration/artifact tests pass.
- [ ] `.venv/bin/python -m pytest tests/core tests/models tests/evaluation tests/online tests/ops` — regression suite liên quan pass.
- [ ] `git diff --check` — không có whitespace error.
- [ ] One-combination CPU/dry-run flow theo config benchmark hiện hành — model, calibration, artifact, online load và event record hoàn tất.

#### Manual

- [ ] Kiểm tra `q` finite và thuộc `(0,1)` trên clean validation, synthetic validation và test.
- [ ] Kiểm tra `c`,`tau` giống nhau giữa offline artifact và online runtime artifact của cùng entity/checkpoint.
- [ ] Kiểm tra offline threshold là q99 của non-overlap clean timeline; online threshold là q99 của EWMA(q) clean timeline.
- [ ] Kiểm tra prediction chỉ dùng điều kiện strict `score > threshold`.

### Risks

- Full benchmark có thể tạo kết quả khác do threshold semantics đã đổi. Mitigation: không chạy matrix trước one-combination pass; giữ provenance và artifact hashes.
- `tau`/threshold phụ thuộc checkpoint và clean validation. Mitigation: artifact lưu checkpoint hash, config identity, calibration split và transform parameters.

## Testing strategy

Kiểm thử theo tầng:

1. Unit test công thức: median, MAD/0.6745, sigmoid, finite checks, q99 và strict `>`.
2. Contract test model: raw MC samples, q top-level, raw `window_scores`, không transform loss.
3. Calibration integration test: offline non-overlap và online stride-1/EWMA dùng cùng `c`,`tau`.
4. Artifact/runtime test: schema, provenance, identity mismatch, no test-stream calibration.
5. End-to-end one-combination smoke trước khi mở rộng benchmark.

## Migration and rollback

- Không sửa hoặc xóa artifact cũ.
- Tạo lại artifact v4 sau khi code và focused tests pass.
- Active v3 online runtime chỉ nhận artifact có transform metadata đầy đủ.
- Nếu implementation phải rollback, rollback code và trỏ về artifact/protocol cũ trong một runtime legacy riêng; không dùng artifact raw-score cũ với runtime yêu cầu q.
- Không chạy recalibration hàng loạt hoặc ghi đè output khi chưa có manifest/preflight và one-combination verification.

## Documentation

- Giữ `full-spec-v3.md` làm normative design.
- Giữ hai ontology files làm terminology SSOT.
- Sau implementation, cập nhật research log bằng evidence thực tế nếu seam/API cuối cùng khác proposed plan.
- Ghi rõ artifact version, transform identity, `c`, `tau`, quantiles và EWMA weights trong benchmark report/provenance.

## Final verification

- [ ] Raw point MSE được tính trước và không bị thay bằng MSE của mean reconstruction.
- [ ] `point_scores` official là shifted-and-scaled logistic sigmoid score.
- [ ] `window_scores` vẫn là raw window reconstruction MSE.
- [ ] Offline/online dùng cùng entity-level `c`,`tau` nhưng threshold riêng.
- [ ] EWMA chạy trên q với trọng số `0.9/0.1`.
- [ ] Thresholds chỉ lấy từ clean validation và dùng q99.
- [ ] Online runtime không calibrate từ test stream.
- [ ] Artifact/runtime identity và strict prediction rule được kiểm thử.

## Assumptions and non-blocking uncertainties

- Plan chọn fail-fast khi `tau <= 0` vì spec không định nghĩa epsilon. Nếu người phát triển muốn epsilon policy, phải cập nhật spec, artifact contract và tests trước Phase 1.
- Plan đề xuất file mới `src/protocols/point_score_calibration.py`; nếu codebase preference/file-size gate yêu cầu owner khác, chỉ đổi location trước Phase 1, không tạo duplicate helper.
- Training-time provisional thresholds có thể tiếp tục tồn tại cho checkpoint monitoring nếu chúng được gắn nhãn rõ và không được dùng làm official v3 artifact threshold.
