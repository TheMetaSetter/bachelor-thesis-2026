---
date: 2026-08-09T00:00:00+07:00
topic: "Atomic implementation steps cho anomaly score theo full-spec-v3"
status: ready
revision: b004e70b26b956809695c1b9d9518adf900ed2e9
source_structure: documents/logs/2026-08-09/structure/structure-anomaly-score-calibration.md
related_documents:
  - documents/logs/2026-08-09/plan/plan-anomaly-score-calibration.md
  - documents/logs/2026-08-09/research/research-anomaly-score-implementation-gaps.md
  - documents/spec/full-spec-v3.md
---

# Detailed Implementation: Atomic implementation steps cho anomaly score theo full-spec-v3

## Summary

Tài liệu này chuyển 7 phase và 33 stage của structure anomaly score thành các atomic step có thể thực hiện tuần tự.

Em dùng yêu cầu trực tiếp của anh làm xác nhận để mở rộng structure hiện đang mang `status: proposed`; em không thay đổi trạng thái của structure file. Tài liệu này chưa sửa source code, config, test hoặc threshold artifact.

Implementation phải giữ pipeline:

```text
raw point MSE e
  -> c = median(clean-validation e)
  -> tau = MAD(clean-validation e) / 0.6745
  -> q = sigmoid((e - c) / tau)
  -> offline Q0.99(q)
     hoặc online Q0.99(EWMA(q))
```

## Source structure

Source structure là `documents/logs/2026-08-09/structure/structure-anomaly-score-calibration.md`.

Các phase giữ nguyên thứ tự:

1. Calibration và threshold artifact contract.
2. THESIS model/scorer xuất `point_scores = q`.
3. Offline clean-validation calibration và threshold generation.
4. V4 threshold recalibration/migration.
5. Online runtime dùng `q` trước EWMA và prediction.
6. Phân định official THESIS với provisional/legacy paths.
7. Full verification và recalibrate artifacts.

## Current state

- `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:182-227` tính raw per-sample point MSE, lấy MC mean và xuất mean raw MSE ở `point_scores`.
- `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py:249-270` nối deterministic/MC raw scores vào public output.
- `src/models/online_impl/online_adaptation_helpers.py:77-86` xuất raw MSE cho online source/projected scorer.
- `src/engine/online_tta/online_calibration.py:61-154` đọc raw `outputs["point_scores"]` và đưa chúng vào EWMA.
- `scripts/benchmarks/run_thesis_offline_benchmark.py:434-452,505-565` hiện tính threshold từ raw clean-validation scores.
- `scripts/ops/recalibrate_thesis_threshold_artifacts_v4.py:226-365` hiện recalibrate online raw scores và copy offline threshold v3.
- `src/protocols/threshold_artifact.py:20-374` chưa yêu cầu transform identity, `c` hoặc `tau`.
- `src/engine/online_tta/point_ewma.py:8-34` đã có công thức EWMA đúng; không thay đổi công thức này.

## Desired end state

- Raw MSE được giữ cho diagnostics và training loss.
- Official calibrated inference xuất `point_scores = q`.
- `aux.point_score_samples` vẫn là raw per-sample MSE.
- `window_scores` vẫn là raw window reconstruction MSE.
- Offline và online dùng cùng entity-level `c`,`tau`.
- Offline threshold là q99 của transformed non-overlap clean-validation scores.
- Online threshold là q99 của EWMA transformed stride-1 clean-validation scores.
- Online runtime đọc calibration từ artifact và không calibrate từ test stream.
- Prediction dùng strict `score > threshold`.

## Scope

### In scope

- Shared calibration helper và artifact schema v4.
- THESIS offline model, online adapter, evaluator và calibration collectors.
- Offline benchmark threshold construction.
- V4 recalibration script và audit artifact.
- Online runtime load/validate calibration, EWMA input, triage và records.
- Provisional/legacy boundary, focused tests, regression tests và one-combination smoke.

### Out of scope

- Thay đổi reconstruction loss, `point_score_loss`, MC sample count, model architecture hoặc EWMA weights.
- Biến `window_scores` thành sigmoid score.
- Sửa RedLamp, traditional baselines hoặc online baseline calibration riêng.
- Xóa/ghi đè artifact cũ.
- Chạy benchmark matrix rộng trước khi one-combination smoke pass.

## Evidence

- `documents/spec/full-spec-v3.md:545-604` — raw point MSE, median center, MAD-based robust scale và shifted-and-scaled logistic sigmoid.
- `documents/spec/full-spec-v3.md:798-835` — clean-validation-only calibration và hai q99 thresholds riêng.
- `documents/spec/full-spec-v3.md:898-907` — EWMA phải chạy trên transformed point score.
- `documents/spec/full-spec-v3.md:929-944` — `window_scores` vẫn phục vụ triage và online phải đọc artifact thay vì test-stream calibration.
- `src/core/contracts.py:121-204` — current stable model output và stochastic sample validation.
- `src/engine/evaluator.py:446-502` — evaluator lấy `point_scores` và tính threshold/metrics.
- `src/engine/online_tta/online_engine_run.py:75-173` — online artifact load và runtime context.
- `src/engine/online_tta/online_engine_window_metrics.py:104-159` — seam giữa score extraction và EWMA.
- `src/protocols/threshold_artifact.py:20-374` — artifact validation và builder hiện tại.

## Phase 1: Calibration và threshold artifact contract hoạt động thống nhất

### Goal

Tạo một calibration object dùng chung và mở rộng schema v4 để artifact lưu đầy đủ transform identity.

### Dependencies

- `full-spec-v3.md` và các quyết định đã khóa trong plan.
- Không phụ thuộc phase implementation trước.

### Atomic steps

#### Stage 1.1 — Khóa calibration semantics và edge policy

1. **Đọc contract:** Dùng `documents/spec/full-spec-v3.md:561-596` làm nguồn công thức duy nhất.
2. **Chọn input:** Quy định helper nhận raw clean-validation point MSE đã flatten theo entity; không nhận transformed score, synthetic labels hoặc test score.
3. **Tính center:** Định nghĩa `c = median(finite_raw_scores)`.
4. **Tính scale:** Định nghĩa `MAD = median(abs(raw_scores - c))`, sau đó `tau = MAD / 0.6745`.
5. **Khóa lỗi:** Raise `ValueError` nếu input rỗng, có non-finite value hoặc `tau <= 0`; không thêm epsilon âm thầm.
6. **Khóa identity:** Dùng đúng literal `shifted-and-scaled logistic sigmoid`, estimator `mad_based_robust_scale` và normalizer `0.6745` trong artifact metadata.

**File/symbol:** Không sửa file ở stage này; decision owner là `full-spec-v3.md:545-604` và calibration helper được tạo ở Stage 1.2.

**Inputs/outputs:** Input là raw finite clean-validation MSE; output contract là `c`, `tau`, estimator metadata và transform identity.

**Errors:** Empty/non-finite input hoặc `tau <= 0` phải dừng calibration trước khi tạo artifact.

#### Stage 1.2 — Xây shared calibration helper

1. **Tạo file:** Tạo **Proposed new file** `src/protocols/point_score_calibration.py`.
2. **Tạo object:** Định nghĩa proposed immutable `PointScoreCalibration` chứa `center`, `tau`, `transform_name`, `tau_estimator` và `mad_normalizer`.
3. **Tạo fit function:** Định nghĩa proposed `fit_mad_logistic_calibration(raw_point_mse)` để validate input, tính median/MAD/tau và trả object.
4. **Tạo transform function:** Định nghĩa proposed `transform_point_scores(raw_point_mse, calibration)` với công thức `1 / (1 + exp(-(e-c)/tau))`.
5. **Giữ type boundary:** Cho phép helper nhận type mà caller hiện dùng; nếu cần adapter torch/NumPy, đặt adapter trong cùng file và giữ một công thức toán học.
6. **Kiểm tra output:** Bảo đảm output finite, cùng shape với input và tăng đơn điệu theo raw MSE.
7. **Không gắn training loss:** Không import helper vào reconstruction loss hoặc `point_score_loss` ở stage này.

**File/symbol:** **Proposed new file** `src/protocols/point_score_calibration.py`; proposed symbols `PointScoreCalibration`, `fit_mad_logistic_calibration`, `transform_point_scores`.

**Current responsibility:** Repository chưa có owner chung cho MAD-based sigmoid calibration.

**Change:** Thêm owner chung để model, offline calibration và online runtime không tự viết công thức riêng.

**Inputs:** Raw point MSE finite; `tau` phải dương.

**Outputs:** Calibration object và transformed score array/tensor cùng shape.

**Errors:** `ValueError` cho empty/non-finite input hoặc non-positive tau; không trả score im lặng.

**Compatibility:** Không thay đổi `src/protocols/point_scores.py`; file đó tiếp tục xử lý overlap/EWMA timeline.

#### Stage 1.3 — Mở rộng threshold artifact v4

1. **Cập nhật required fields:** Trong `src/protocols/threshold_artifact.py:20-36`, yêu cầu calibration fields cho THESIS schema v4.
2. **Cập nhật validation:** Trong `validate_threshold_artifact`, kiểm tra transform name, `c` finite, `tau > 0`, estimator literal và normalizer `0.6745`.
3. **Giữ baseline compatibility:** Không bắt các baseline schema v3 phải có THESIS transform fields; điều kiện bắt buộc phải dựa trên `method_name`/schema v4.
4. **Cập nhật builder:** Trong `build_threshold_artifact:237-374`, nhận các calibration fields và ghi chúng vào top-level artifact cùng `provenance`.
5. **Giữ threshold semantics:** Không đổi `offline_point_threshold_nonoverlap`, `online_point_threshold_ewma`, EWMA weights hoặc triage threshold fields.
6. **Fail closed:** `load_threshold_artifact` phải reject THESIS v4 artifact thiếu calibration metadata trước khi runtime dùng artifact.

**File/symbol:** `src/protocols/threshold_artifact.py:validate_threshold_artifact`, `build_threshold_artifact`, `write_threshold_artifact`, `load_threshold_artifact`.

**Inputs:** Existing artifact identity plus calibration metadata.

**Outputs:** Schema v4 artifact có transform identity và thresholds.

**Errors:** Missing field, wrong literal, non-finite `c`, `tau <= 0`, normalizer mismatch hoặc identity mismatch phải raise.

**Compatibility:** Preserve non-THESIS v3 builder callers in `scripts/benchmarks/run_offline_benchmark.py`, `scripts/benchmarks/run_online_streaming_benchmark.py`, `src/baselines/online/frozen.py` và `src/baselines/online/adaptive.py`.

#### Stage 1.4 — Kiểm tra contract nền tảng

1. **Mở rộng unit tests:** Thêm calibration formula tests vào `tests/evaluation/test_point_score_contracts.py`.
2. **Kiểm tra center/scale:** Dùng một raw score vector có median/MAD tính được bằng tay; assert `c` và `tau`.
3. **Kiểm tra transform:** Assert score tăng theo raw MSE, nằm trong `(0,1)` với finite `tau`, và giữ shape.
4. **Kiểm tra lỗi:** Assert empty, non-finite và `MAD=0` bị reject.
5. **Cập nhật artifact fixtures:** Bổ sung fields cho THESIS fixtures trong `tests/engine/test_threshold_artifact.py`, `tests/online/test_threshold_artifact.py` và `tests/ops/test_recalibrate_thesis_threshold_artifacts_v4.py`.
6. **Chạy focused tests:** Chỉ chuyển sang Phase 2 sau khi helper và artifact tests pass.

**Tests:**

- **Location:** `tests/evaluation/test_point_score_contracts.py`.
- **Level:** Unit/contract.
- **Setup:** Raw finite array, zero-MAD array, NaN-containing array và calibration artifact.
- **Action:** Fit, transform, serialize-related validation.
- **Expected result:** Công thức đúng, lỗi bị reject, output finite.
- **Edge cases:** Empty input, all-equal input, non-finite input, `tau <= 0`.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest tests/evaluation/test_point_score_contracts.py tests/engine/test_threshold_artifact.py tests/online/test_threshold_artifact.py tests/ops/test_recalibrate_thesis_threshold_artifacts_v4.py` — helper và artifact contract pass.
- [ ] `git diff --check` — không có whitespace error.

#### Manual

- [ ] Đọc artifact fixture — thấy transform name, `c`, `tau`, estimator, normalizer và provenance.

### Risks and recovery

- **Risk:** Thêm required args làm hỏng baseline artifact callers.
- **Mitigation:** Chỉ bắt buộc calibration fields cho THESIS schema v4; giữ schema v3 compatibility cho baseline.
- **Verification:** Chạy artifact tests và `rg` toàn bộ `build_threshold_artifact` callers.
- **Recovery:** Revert builder field requirement, giữ artifact cũ nguyên vẹn và không chạy active v3 online với artifact mới thiếu metadata.

### Complete when

- Shared helper và artifact schema v4 được test.
- THESIS v4 artifact thiếu transform metadata bị reject.
- Baseline artifact tests không bị thay đổi semantics.

## Phase 2: THESIS model/scorer xuất official `point_scores = q`

### Goal

Model và online adapter cùng dùng calibration state để biến raw MC mean point MSE thành `q`, nhưng không biến sigmoid thành training loss.

### Dependencies

- Phase 1 có `PointScoreCalibration` và artifact fields.

### Atomic steps

#### Stage 2.1 — Tách raw score boundary

1. **Xác nhận MC raw score:** Giữ `point_score_samples` tại `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:182-218` là per-sample raw MSE.
2. **Tính raw mean:** Giữ `point_score_mean = point_score_samples.mean(dim=1)` làm `e`.
3. **Giữ raw window score:** Giữ `window_score_mean = window_score_samples.mean(dim=1)` làm raw window reconstruction MSE.
4. **Đặt calibration seam:** Thêm một proposed model-level helper/method để chuyển raw mean `e` thành q khi model có calibration state.
5. **Không transform samples:** Không sigmoid từng `point_score_samples`; chỉ transform MC mean `e` theo spec.

**Files/symbols:** `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:_build_monte_carlo_forward_outputs`; `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py:forward`.

**Inputs:** Raw per-sample MSE, optional calibration state.

**Outputs:** Raw `e` for diagnostics, q for calibrated top-level `point_scores`, raw window score.

**Errors:** Calibrated official path thiếu state phải fail ở orchestration trước forward hoặc emit explicit uncalibrated state; không silently dùng raw score cho official thresholding.

**Compatibility:** Training forward và raw diagnostics remain available.

#### Stage 2.2 — Gắn calibration state vào offline model

1. **Chọn owner state:** Dùng `ThesisMultitaskStateMixin` (`src/models/thesis_multitask_impl/thesis_multitask_state_mixin.py`) hoặc public `ThesisMultitaskModel` (`src/models/thesis_multitask.py`) làm owner cho runtime-only calibration state.
2. **Thêm setter:** Tạo proposed `set_point_score_calibration(calibration)` và `clear_point_score_calibration()` ở owner được chọn.
3. **Không persist checkpoint state:** Calibration state thuộc threshold artifact; không đưa `c`,`tau` vào model checkpoint serialization nếu artifact vẫn là owner identity.
4. **Áp dụng trong forward:** Trong routing forward, sau khi có raw `e`, gọi helper transform nếu calibration state tồn tại.
5. **Giữ training behavior:** Khi `self.training` hoặc calibration state chưa có, giữ raw output cho training diagnostics; official evaluator phải set state trước thresholded evaluation.
6. **Expose status:** Ghi một explicit aux marker hoặc runtime assertion để evaluator biết score đã calibrated; không suy luận từ score range vì raw score cũng có thể nhỏ hơn 1.

**Files/symbols:** `src/models/thesis_multitask.py:ThesisMultitaskModel`; `src/models/thesis_multitask_impl/thesis_multitask_state_mixin.py:ThesisMultitaskStateMixin`; `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py:forward`.

**Inputs:** `PointScoreCalibration` object fitted from clean validation.

**Outputs:** Model runtime state and calibrated top-level `point_scores`.

**Errors:** Reject invalid calibration object; official scoring must not proceed without calibration state.

**Compatibility:** Do not change public top-level keys or training loss inputs.

#### Stage 2.3 — Đồng bộ online adapter

1. **Add adapter state:** Cho `ThesisMultitaskEncoderAdapter` giữ cùng calibration object hoặc delegate setter tới copied offline model.
2. **Apply source/projected:** Trong `score_from_hidden`, transform raw point MSE mean after reconstruction error calculation for both `score_source` and `score_projected`.
3. **Preserve window score:** Keep `window_scores = raw point_scores.mean(dim=1)` based on raw MSE, not q.
4. **Propagate through online model:** Ensure `forward_source` and `forward` expose q at top-level and preserve raw diagnostics under aux.
5. **Avoid double transform:** Do not add another sigmoid in `online_engine_window_metrics` after adapter returns q.

**File/symbol:** `src/models/online_impl/online_adaptation_helpers.py:ThesisMultitaskEncoderAdapter.score_from_hidden`; callers `src/models/online_impl/online_adaptation.py:forward_source, forward`.

**Inputs:** Hidden representation, input window and calibration state.

**Outputs:** q point scores, raw window score, reconstruction and existing latent score.

**Errors:** Missing calibration in official online context must be rejected before stream.

**Compatibility:** A0 source and A1/A2 projected paths share one score semantic.

#### Stage 2.4 — Cập nhật output contract

1. **Extend validation:** In `src/core/contracts.py:validate_model_outputs`, validate calibrated q only when explicit calibrated marker/state says it is official.
2. **Check range:** For calibrated q, assert finite and `0 < q < 1` unless a documented numerical boundary policy allows exact float saturation.
3. **Keep sample ranks:** Preserve `validate_stochastic_query_aux` rank checks for raw `point_score_samples` and `window_score_samples`.
4. **Keep raw uncertainty meaning:** Do not silently replace raw point-score variance with per-sample sigmoid variance.
5. **Update trace naming only if needed:** Keep existing trace payload shape unless a new raw diagnostic field is required by the acceptance tests.

**File/symbol:** `src/core/contracts.py:validate_model_outputs`, `validate_stochastic_query_aux`.

**Inputs:** Model output dict and calibration marker.

**Outputs:** Validated output contract.

**Errors:** Wrong rank, non-finite q, invalid calibrated range or inconsistent marker must raise.

**Compatibility:** Uncalibrated training outputs remain valid only outside official thresholding path.

#### Stage 2.5 — Kiểm tra model-level vertical slice

1. **Create calibration fixture:** Construct `PointScoreCalibration` with known `c` and positive `tau`.
2. **Run deterministic forward:** Assert raw mean `e` is unchanged and top-level q matches sigmoid formula.
3. **Run stochastic forward:** Assert q uses mean raw sample MSE, not mean of independently transformed samples.
4. **Run online source/projected forward:** Assert both paths use same calibration object.
5. **Check window/loss:** Assert `window_scores` and reconstruction/point score loss remain raw-error based.
6. **Check double transform:** Apply repeated downstream extraction and assert value remains q, not sigmoid(q).

**Tests:**

- **Location:** `tests/models/test_multitask_shapes.py`, `tests/models/test_thesis_multitask_point_score_loss.py`, `tests/online/test_online_calibration_contract.py`.
- **Level:** Unit and integration.
- **Setup:** Existing model fixtures plus known calibration object.
- **Action:** Forward, source/projected score and loss calculation.
- **Expected result:** q semantics correct; raw sample/window/loss semantics preserved.
- **Edge cases:** One MC sample, deterministic inference, zero/invalid calibration state.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest tests/models/test_multitask_shapes.py tests/models/test_thesis_multitask_point_score_loss.py tests/online/test_online_calibration_contract.py` — model/scorer contract pass.

#### Manual

- [ ] Inspect one calibrated forward payload — `point_scores` is q, `point_score_samples` is raw, `window_scores` is raw.

### Risks and recovery

- **Risk:** Model output semantics change before calibration state is available.
- **Mitigation:** Official evaluator/runtime asserts calibrated state; training path remains explicitly uncalibrated.
- **Verification:** Model contract tests cover both state-present and state-absent paths.
- **Recovery:** Clear calibration state and run legacy/raw diagnostics only; do not create official artifact from uncalibrated output.

### Complete when

- Offline and online THESIS scorers share one transform owner.
- q uses MC mean raw MSE.
- `window_scores`, raw samples and loss behavior remain unchanged.
- No downstream path applies sigmoid a second time.

## Phase 3: Offline clean-validation calibration và threshold generation hoàn tất

### Goal

Offline benchmark fit calibration from raw clean validation, then evaluates all official splits with q and builds separate offline/online thresholds.

### Dependencies

- Phase 1 artifact/helper contract.
- Phase 2 calibrated model/scorer.

### Atomic steps

#### Stage 3.1 — Thu raw clean-validation timeline

1. **Locate orchestration:** Use `scripts/benchmarks/run_thesis_offline_benchmark.py:collect_offline_artifact_inputs` and `_evaluate_offline_benchmark_splits` as owners.
2. **Run first clean pass:** Evaluate clean validation with calibration state absent only for raw collection; do not use its provisional threshold as official output.
3. **Extract covered scores:** Use existing `_evaluation_outputs_to_score_payload` and covered masks to collect raw point MSE values.
4. **Validate entity grouping:** Check records do not mix entities for one artifact; reject multi-entity ambiguity instead of using `_first_entity_id` silently.
5. **Exclude labels:** Ensure `val_synth` and `test` are not read during fit.

**Files/symbols:** `scripts/benchmarks/run_thesis_offline_benchmark.py:collect_offline_artifact_inputs`, `_evaluate_offline_benchmark_splits`, `_evaluation_outputs_to_score_payload`; `src/engine/evaluator.py:Evaluator.evaluate`.

**Inputs:** Clean validation loader and Stage B checkpoint.

**Outputs:** Raw clean-validation point MSE timeline grouped by entity.

**Errors:** Empty covered timeline, mixed entities or non-finite score must stop calibration.

**Compatibility:** Preserve existing evaluation records for audit; do not treat provisional raw threshold as official.

#### Stage 3.2 — Fit và cài calibration state

1. **Fit from raw offline timeline:** Call `fit_mad_logistic_calibration` on the entity raw clean timeline.
2. **Validate object:** Confirm `c`, `tau`, estimator and normalizer match Phase 1 contract.
3. **Install on offline model:** Call proposed model setter before re-evaluating clean/synthetic/test.
4. **Install on online calibration scorer:** Ensure the same model/adapter instance used for stride-1 calibration receives the same object.
5. **Persist in in-memory artifact inputs:** Carry calibration object through `_build_thresholds` without recomputing it from transformed scores.

**Files/symbols:** `scripts/benchmarks/run_thesis_offline_benchmark.py:_evaluate_offline_benchmark_splits`, `collect_offline_artifact_inputs`, `_build_thresholds`; proposed helper and model setter.

**Inputs:** Raw clean-validation timeline.

**Outputs:** Calibrated model/scorer and entity calibration object.

**Errors:** Fit errors propagate before synthetic/test evaluation or artifact write.

**Compatibility:** Same c/tau will be reused for offline and online thresholds.

#### Stage 3.3 — Tạo transformed offline và online clean timelines

1. **Re-run clean offline evaluation:** Evaluate clean validation after installing calibration state; collect q timeline using non-overlap windows.
2. **Collect online q:** Call `collect_stride1_online_scores` with calibrated model and existing stride-1 settings.
3. **Apply EWMA to q:** Keep `update_window_point_ewma` and weights unchanged; ensure collector receives q from model.
4. **Separate timelines:** Store offline q timeline and online EWMA(q) timeline under distinct keys.
5. **Preserve triage data:** Keep raw input-window and latent-window scores for their existing thresholds.

**Files/symbols:** `src/engine/online_tta/online_calibration.py:_collect_batch_scores`, `run_stride1_sequence_scores`, `_collect_offline_scores`, `collect_stride1_online_scores`; `scripts/benchmarks/run_thesis_offline_benchmark.py` orchestration.

**Inputs:** Calibrated model, clean validation sequences, window/stride/EWMA protocol.

**Outputs:** `offline_point_q`, `online_point_q`, `online_ewma_q`, raw triage timelines.

**Errors:** Batch size, shape, finite values and entity mismatches must fail clearly.

**Compatibility:** Keep existing `input_window` and `latent_window` collection semantics.

#### Stage 3.4 — Tạo threshold artifact

1. **Compute offline threshold:** Pass transformed non-overlap clean q to `select_clean_validation_point_threshold`.
2. **Compute online threshold:** Pass transformed EWMA clean timeline to `select_online_ewma_threshold`.
3. **Compute triage thresholds:** Keep input/latent quantile calculations raw and unchanged.
4. **Build artifact:** Pass both q99 threshold values, same c/tau, checkpoint hash, config identity and EWMA weights to `build_threshold_artifact`.
5. **Write artifact:** Use existing `write_threshold_artifact`; validate before write.
6. **Record provenance:** Include calibration split, transform identity, values and source timelines in audit/report metadata.

**File/symbol:** `scripts/benchmarks/run_thesis_offline_benchmark.py:_build_thresholds`, `_export_offline_artifacts`.

**Inputs:** Transformed clean timelines and existing triage calibration timelines.

**Outputs:** `thresholds/thresholds.json` with separate offline/online q99 thresholds.

**Errors:** Non-finite/empty timeline or artifact identity mismatch must prevent write.

**Compatibility:** Preserve output path and existing threshold field names.

#### Stage 3.5 — Chạy offline vertical slice

1. **Run evaluation-only combination:** Use existing evaluation-only wrapper/config path with one known Stage B checkpoint.
2. **Check clean:** Confirm clean artifact calibration uses raw MSE to fit and q to threshold.
3. **Check synthetic/test:** Confirm synthetic/test receive the clean-derived threshold and do not affect calibration.
4. **Check report:** Confirm score arrays, thresholds and provenance identify the same checkpoint/entity/config.
5. **Stop on mismatch:** Do not proceed to Phase 4 if artifact or report contains raw threshold semantics.

**Tests:**

- **Location:** `tests/benchmarks/test_thesis_offline_artifact_exports.py`, `tests/benchmarks/test_full_spec_runtime_readiness.py`, existing evaluator tests.
- **Level:** Integration/contract.
- **Setup:** Existing fake model/checkpoint/data fixtures.
- **Action:** Run offline artifact collection/build flow.
- **Expected result:** q-based thresholds and shared c/tau are persisted.
- **Edge cases:** Empty clean timeline, mixed entities, missing checkpoint hash, synthetic-only data.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest tests/benchmarks/test_thesis_offline_artifact_exports.py tests/benchmarks/test_full_spec_runtime_readiness.py tests/online/test_online_calibration_contract.py` — offline calibration/collector contract pass.

#### Manual

- [ ] Inspect one generated artifact and report — offline threshold is q99 of non-overlap q; online threshold is q99 of EWMA(q); c/tau match.

### Risks and recovery

- **Risk:** First raw pass accidentally becomes official metric output.
- **Mitigation:** Mark it calibration-only and re-run official evaluation after setting state.
- **Verification:** Assert official report score range and threshold source metadata.
- **Recovery:** Discard newly generated artifact only; keep checkpoint and prior outputs.

### Complete when

- One offline evaluation-only combination creates a valid q-based artifact.
- Clean-only calibration and separate offline/online thresholds are proven by tests and report inspection.

## Phase 4: V4 threshold recalibration tạo artifact mới từ raw clean validation

### Goal

Recalibration script rebuilds both thresholds from clean validation and records calibration provenance without overwriting existing output.

### Dependencies

- Phase 1 schema/helper.
- Phase 3 verified threshold semantics.

### Atomic steps

#### Stage 4.1 — Khảo sát inventory và identity

1. **Read inventory:** Use `discover_stage_b_inventory` and existing preflight checks in `scripts/ops/recalibrate_thesis_threshold_artifacts_v4.py`.
2. **Validate v3 identity:** Keep `_validate_v3_identity` for entity/variant/seed/config matching.
3. **Validate checkpoint:** Keep `_validate_checkpoint_identity` and checkpoint hash verification.
4. **Reject existing output:** Preserve refusal to overwrite `threshold_artifact_v4_path`.
5. **Freeze inventory:** Do not start score collection for entries that fail preflight.

**File/symbol:** `scripts/ops/recalibrate_thesis_threshold_artifacts_v4.py:discover_stage_b_inventory`, `preflight_inventory`, `_validate_v3_identity`, `_validate_checkpoint_identity`.

**Inputs:** Existing v3 artifacts, configs and Stage B checkpoints.

**Outputs:** Validated inventory entries.

**Errors:** Missing/mismatch/existing output stops that entry before writes.

**Compatibility:** Preserve current no-overwrite behavior.

#### Stage 4.2 — Thu đủ hai raw calibration timelines

1. **Build A0 scorer:** Reuse `_collect_clean_validation_scores` model/config loading and scaler restoration.
2. **Collect offline raw:** Call `collect_nonoverlap_offline_scores` for clean validation.
3. **Collect online raw:** Call `collect_stride1_online_scores` before calibration state is installed, or add a raw collection mode that bypasses q only for fit input.
4. **Retain triage raw scores:** Keep input-window and latent-window arrays for existing triage thresholds.
5. **Return explicit keys:** Return separate `offline_raw_point`, `online_raw_point`, `input_window`, and `latent_window` keys; do not overload `point`/`ewma` names.

**Files/symbols:** `scripts/ops/recalibrate_thesis_threshold_artifacts_v4.py:_collect_clean_validation_scores`; `src/engine/online_tta/online_calibration.py:collect_nonoverlap_offline_scores`, `collect_stride1_online_scores`.

**Inputs:** Stage B checkpoint, clean validation sequences, protocol config.

**Outputs:** Both raw point timelines and raw triage timelines.

**Errors:** Missing scaler/checkpoint, non-finite score or empty timeline stops entry.

**Compatibility:** Preserve A0/source scoring identity and view-noise settings.

#### Stage 4.3 — Fit và transform lại

1. **Fit calibration:** Fit c/tau from `offline_raw_point` only.
2. **Transform offline:** Transform `offline_raw_point` into q.
3. **Transform online:** Transform `online_raw_point` into q using same object.
4. **Build online EWMA:** Apply existing vector EWMA to transformed online q.
5. **Leave triage raw:** Do not sigmoid input-window or latent-window scores.
6. **Compute values:** Produce offline q99, online EWMA q99 and existing triage quantiles.

**File/symbol:** `scripts/ops/recalibrate_thesis_threshold_artifacts_v4.py:_calibrate_threshold_values`, plus shared calibration helper.

**Inputs:** Raw timelines and quantile protocol fields.

**Outputs:** Threshold values and calibration object.

**Errors:** Raw fit/transform/quantile errors stop artifact construction.

**Compatibility:** Preserve q99 and triage quantile config names.

#### Stage 4.4 — Ghi artifact v4 và audit report

1. **Update field builder:** Change `_v4_artifact_fields` so it no longer copies v3 offline threshold.
2. **Pass new values:** Pass recalculated offline/online thresholds and c/tau into `build_threshold_artifact`.
3. **Write v4:** Use `write_threshold_artifact` and preserve existing output path/refusal behavior.
4. **Write audit:** Add raw calibration counts, c/tau, transform identity, threshold values and checkpoint hash to audit JSON.
5. **Keep provenance:** Preserve v3 artifact path and variant-resolution metadata.

**Files/symbols:** `scripts/ops/recalibrate_thesis_threshold_artifacts_v4.py:_v4_artifact_fields`, `build_v4_threshold_artifact`, `recalibrate_entry`.

**Inputs:** Validated inventory, threshold values and calibration metadata.

**Outputs:** New v4 artifact and audit report.

**Errors:** Artifact validator failure must prevent successful status.

**Compatibility:** Do not mutate v3 artifact or checkpoint.

#### Stage 4.5 — Kiểm tra migration output

1. **Validate artifact:** Call `validate_threshold_artifact` on every new artifact.
2. **Compare identities:** Assert artifact identity equals inventory/checkpoint/config identity.
3. **Compare threshold provenance:** Assert offline threshold differs from copied v3 path when raw calibration differs; at minimum assert it was recomputed and recorded.
4. **Check no overwrite:** Run existing preflight test with an existing v4 output.
5. **Classify status:** Mark entry created only after artifact and audit write complete.

**Tests:**

- **Location:** `tests/ops/test_recalibrate_thesis_threshold_artifacts_v4.py`.
- **Level:** Integration/contract.
- **Setup:** Existing fake v3 artifact, checkpoint and calibration arrays.
- **Action:** Build/recalibrate v4 artifact.
- **Expected result:** Both thresholds and c/tau are new, schema valid, output is not overwritten.
- **Edge cases:** Variant recovery, identity mismatch, existing v4 file, non-finite scores.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest tests/ops/test_recalibrate_thesis_threshold_artifacts_v4.py` — recalibration and no-overwrite tests pass.

#### Manual

- [ ] Inspect audit JSON — offline threshold is recomputed from transformed clean timeline, not copied from v3.

### Risks and recovery

- **Risk:** New artifacts change benchmark comparability.
- **Mitigation:** Preserve old artifacts, checkpoint hashes and audit reports; use new provenance/version.
- **Verification:** Compare identity and calibration fields before runtime load.
- **Recovery:** Do not delete v3/v4 outputs; stop using new artifact if validation fails.

### Complete when

- A representative inventory entry produces a valid v4 artifact with both transformed thresholds and complete audit provenance.

## Phase 5: Online runtime dùng `q` trước EWMA và prediction

### Goal

Online startup loads calibration from artifact, then all point prediction paths use q or EWMA(q).

### Dependencies

- Phase 3 threshold semantics.
- Phase 4 valid v4 artifact.

### Atomic steps

#### Stage 5.1 — Load và validate artifact identity

1. **Load artifact:** Keep `resolve_threshold_artifact` and `load_threshold_artifact` in `src/engine/online_tta/online_engine_run.py`.
2. **Validate identity:** Extend `_validate_online_artifact_identity` with transform name, c, tau, estimator and normalizer checks.
3. **Validate schema:** Reject any active THESIS artifact that lacks v4 calibration fields.
4. **Build model:** Create model as current flow does, then install calibration object before stream creation.
5. **Store runtime identity:** Keep artifact and calibration in context/runtime state for resume and audit.
6. **Do not calibrate test:** Remove/avoid any startup path that computes thresholds from test stream.

**File/symbol:** `src/engine/online_tta/online_engine_run.py:_validate_online_artifact_identity`, `_build_runtime_online_context`.

**Inputs:** Experiment config, protocol config, threshold artifact and reference checkpoint hash.

**Outputs:** Runtime context with calibrated model and online threshold.

**Errors:** Missing/mismatched transform or identity raises before stream.

**Compatibility:** Preserve entity/checkpoint/EWMA identity checks and runtime state fields.

#### Stage 5.2 — Đồng bộ A0/A1/A2 scoring

1. **A0 path:** Ensure `forward_source` uses calibrated adapter/model.
2. **A1/A2 path:** Ensure `forward` projected scoring uses same calibration object.
3. **Fallback path:** In `online_engine_step:_compute_step_scores`, use canonical q if forward fallback is needed; otherwise raise for missing calibration.
4. **Preserve raw auxiliary values:** Keep reconstruction and raw window scores available for losses/triage where required.
5. **Test all variants:** Use existing A0/A1/A2 fixtures and assert identical transform semantics.

**Files/symbols:** `src/models/online_impl/online_adaptation.py:forward_source, forward`; `src/models/online_impl/online_adaptation_helpers.py:score_from_hidden`; `src/engine/online_tta/online_engine_step.py:_compute_step_scores`.

**Inputs:** Online window, model variant and calibration state.

**Outputs:** q point score and raw triage inputs.

**Errors:** Missing calibration in any official variant fails before prediction.

**Compatibility:** Preserve projector-only update behavior.

#### Stage 5.3 — Đưa `q` vào absolute-index EWMA

1. **Extract canonical q:** In `_extract_online_window_scores`, read q returned by calibrated model/scorer.
2. **Avoid second transform:** Do not call sigmoid in `_extract_online_window_scores` if model/scorer already transformed.
3. **Call EWMA:** Pass q to `update_window_point_ewma` from `_score_online_window`.
4. **Preserve state:** Keep `previous_ewma_point_scores`, absolute indices and active score map unchanged.
5. **Preserve weights:** Continue reading `online_ewma_current_weight` and `online_ewma_previous_weight` from protocol/artifact identity.

**Files/symbols:** `src/engine/online_tta/online_engine_window_metrics.py:_score_online_window`, `_extract_online_window_scores`; `src/engine/online_tta/point_ewma.py:update_window_point_ewma`.

**Inputs:** q vector, absolute indices, previous EWMA map and weights.

**Outputs:** Current EWMA(q) vector and active map.

**Errors:** Shape/index mismatch follows existing EWMA validation; uncalibrated q must be rejected upstream.

**Compatibility:** No change to EWMA formula or state serialization.

#### Stage 5.4 — Đồng bộ triage, verification và event record

1. **Prediction:** Keep strict `current_window_ewma_point_scores > threshold_value` in `_build_online_window_outputs`.
2. **Triage:** Keep raw `input_window_score` and latent score for triage thresholds; use q/EWMA(q) for point prediction only.
3. **Verification/update:** Pass transformed point score to verification record and update decision fields that are point-score based.
4. **Legacy field semantics:** If `raw_point_score` remains in `online_engine_step`, document/store its actual transformed-score meaning or add a separate raw diagnostic field without changing threshold comparison.
5. **Metric names:** Keep `online/ewma_point_score` as EWMA(q); rename only if ontology/runtime consumers can be migrated together.

**Files/symbols:** `src/engine/online_tta/online_engine_window_core.py:_run_current_window_action`, `_admit_and_verify_gray_zone`, `_build_event_outputs`; `src/engine/online_tta/online_engine_window_metrics.py:_build_online_window_outputs`; `src/engine/online_tta/online_engine_step.py:_build_step_record`.

**Inputs:** q vector, EWMA(q), raw window/latent triage scores and artifact threshold.

**Outputs:** Prediction, event record, metrics and adaptation inputs.

**Errors:** Inconsistent score semantic or threshold mismatch must fail before update/admission.

**Compatibility:** Preserve event ordering, verification buffer behavior and projector-only update.

#### Stage 5.5 — Chạy online vertical slice

1. **Prepare artifact:** Use representative valid v4 artifact from Phase 4.
2. **Start runtime:** Run one small online sequence through existing benchmark entrypoint/config.
3. **Inspect startup:** Confirm artifact identity and calibration state loaded before stream.
4. **Inspect events:** Confirm q range, EWMA(q), threshold and prediction relation.
5. **Inspect no-calibration guard:** Use a fixture missing transform fields and confirm startup rejects it.

**Tests:**

- **Location:** `tests/online/test_online_entrypoint.py`, `tests/online/test_entity_threshold_runtime.py`, `tests/online/test_online_ewma_threshold.py`.
- **Level:** Integration/contract.
- **Setup:** Existing fake online model/artifact/context fixtures.
- **Action:** Build context and process overlapping windows.
- **Expected result:** q reaches EWMA, identity mismatch rejects, prediction is strict `>`.
- **Edge cases:** Missing artifact field, old schema, overlapping absolute index, A0/A1/A2.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest tests/online/test_online_entrypoint.py tests/online/test_entity_threshold_runtime.py tests/online/test_online_ewma_threshold.py tests/online/test_online_calibration_contract.py` — online runtime and EWMA(q) pass.

#### Manual

- [ ] Inspect one online event and confirm `window_point_scores`/EWMA values are transformed scores, while input-window triage remains raw.

### Risks and recovery

- **Risk:** Double transformation produces plausible but incorrect scores.
- **Mitigation:** One shared scorer owns transform; downstream seam only validates/consumes q.
- **Verification:** Overlap fixture compares exact expected q and EWMA(q).
- **Recovery:** Stop runtime on semantic mismatch; keep old artifact and runtime outputs unchanged.

### Complete when

- Online startup loads valid calibration, test stream remains calibration-free, EWMA receives q, and event prediction uses strict threshold comparison.

## Phase 6: Phân định official THESIS path với provisional/legacy paths

### Goal

Raw/legacy threshold logic no longer masquerades as official THESIS v3 thresholding, while non-THESIS baselines remain behavior-compatible.

### Dependencies

- Phase 2, Phase 3 and Phase 5 have stable output semantics.

### Atomic steps

#### Stage 6.1 — Phân loại trainer threshold

1. **Trace trainer path:** Inspect `src/engine/trainer.py:_aggregate_reconstructed_pointwise_metrics` and its callers.
2. **Classify threshold:** Mark positive-support threshold as provisional when no clean-validation calibration state/artifact exists.
3. **Protect official path:** Ensure official THESIS artifact creation never reads trainer provisional threshold.
4. **Preserve monitoring:** Keep checkpoint monitoring behavior unless it directly feeds official artifact; add explicit source metadata if needed.
5. **Test separation:** Add assertion that provisional threshold cannot be used as artifact threshold.

**Files/symbols:** `src/engine/trainer.py:_aggregate_reconstructed_pointwise_metrics`; `src/engine/thresholding.py:select_point_score_threshold`, `resolve_evaluation_threshold`.

**Inputs:** Training/validation payloads and optional threshold source.

**Outputs:** Provisional metrics or official clean-derived threshold, clearly separated.

**Errors:** Official path receiving raw positive-support threshold must fail or be explicitly rejected.

**Compatibility:** Preserve non-official checkpoint monitoring unless migration is required.

#### Stage 6.2 — Kiểm tra evaluator fallback

1. **Trace default:** Inspect `src/engine/evaluator.py:488-493` and all callers.
2. **Require official threshold:** Ensure THESIS benchmark passes clean-derived transformed threshold explicitly.
3. **Classify fallback:** Keep raw positive-support fallback only for legacy/non-THESIS path or mark it non-official.
4. **Validate source metadata:** Make report `threshold_source` distinguish `clean_validation_quantile` from legacy fallback.
5. **Test caller coverage:** Add tests for official caller and fallback rejection/labeling.

**File/symbol:** `src/engine/evaluator.py:Evaluator.evaluate`, `resolve_evaluation_threshold`.

**Inputs:** Point scores, explicit threshold and source metadata.

**Outputs:** Metrics with correct threshold provenance.

**Errors:** Missing official threshold/calibration in THESIS path must fail before official metrics.

**Compatibility:** Do not break legacy evaluators that intentionally use fallback, but keep their source distinct.

#### Stage 6.3 — Cập nhật test semantics

1. **Update MC assertion:** Change `tests/models/test_multitask_shapes.py:289-305` from raw equality to q formula.
2. **Update calibration fixture:** Change raw expected values in `tests/online/test_online_calibration_contract.py` to expected q/EWMA(q) when calibrated.
3. **Update artifact fixtures:** Add calibration metadata to THESIS artifacts while preserving non-THESIS fixtures.
4. **Add strict threshold test:** Assert equality does not produce anomaly; only `score > threshold` does.
5. **Keep baseline tests:** Run RedLamp/traditional tests without applying THESIS transform.

**Files:** `tests/models/test_multitask_shapes.py`, `tests/online/test_online_calibration_contract.py`, `tests/online/test_threshold_artifact.py`, `tests/engine/test_threshold_artifact.py`, `tests/evaluation/test_point_score_contracts.py`.

**Inputs:** Existing fixtures updated with explicit score semantics.

**Outputs:** Tests that distinguish raw MSE, q and raw window MSE.

**Errors:** A test relying on ambiguous `point_scores` semantics must be rewritten rather than patched with a different constant.

#### Stage 6.4 — Đồng bộ ontology và report fields

1. **Compare implementation names:** Compare actual fields with `full-spec-v3.md` and both ontology files.
2. **Update only drift:** Change documentation only where final implementation names differ from locked terminology.
3. **Preserve mapping:** Keep raw point MSE, `point_scores`, `window_scores`, `window_point_scores`, EWMA score and threshold names distinct.
4. **Check report fields:** Ensure offline/online report exposes transform identity and threshold provenance.
5. **Do not change design:** Do not reopen sigmoid/MAD/EWMA decisions in documentation.

**Files:** `documents/spec/full-spec-v3.md`, `documents/spec/offline_pretraining_terminology_ontology.md`, `documents/spec/online_tta_terminology_ontology.md`; report writers touched by actual field drift.

**Inputs:** Final code field names and artifact JSON.

**Outputs:** SSOT/ontology aligned with executable behavior.

**Errors:** Terminology mismatch must be recorded and fixed before final acceptance.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest tests/models tests/evaluation tests/online` — official and legacy score contracts pass.
- [ ] `rg -n "positive_support_quantile|point_scores" src scripts tests` — every official THESIS usage is classified.

#### Manual

- [ ] Review one offline and one online report; confirm no raw/provisional threshold is labeled as official v3.

### Risks and recovery

- **Risk:** Broad test/doc edits change baseline behavior.
- **Mitigation:** Restrict semantic changes to THESIS v3 and preserve explicit legacy source labels.
- **Verification:** Run baseline tests and inspect report provenance.
- **Recovery:** Revert only boundary changes; do not revert shared artifact schema without restoring caller compatibility.

### Complete when

- Official THESIS thresholds trace to clean-validation q timelines.
- Provisional/legacy fallback is labeled or rejected.
- Ontology and report fields match executable semantics.

## Phase 7: Kiểm thử toàn hệ thống và recalibrate artifacts

### Goal

Prove the complete raw MSE -> q -> threshold/EWMA -> prediction path before expanding experiments.

### Dependencies

- Phase 1 through Phase 6.

### Atomic steps

#### Stage 7.1 — Focused formula and contract tests

1. Run the calibration formula tests.
2. Run model raw/MC/window output contract tests.
3. Run artifact schema/provenance tests.
4. Run strict threshold comparison tests.
5. Stop on any failure before integration tests.

**Files/tests:** `tests/evaluation/test_point_score_contracts.py`, `tests/models/test_multitask_shapes.py`, `tests/engine/test_threshold_artifact.py`, `tests/online/test_threshold_artifact.py`.

**Expected result:** Local formulas and contracts pass without running a benchmark.

#### Stage 7.2 — Calibration/runtime integration tests

1. Run offline two-pass calibration tests.
2. Run online stride-1 calibration tests.
3. Run EWMA overlap/absolute-index tests.
4. Run online artifact identity/no-test-calibration tests.
5. Run recalibration script tests.

**Files/tests:** `tests/online/test_online_calibration_contract.py`, `tests/online/test_online_ewma_threshold.py`, `tests/online/test_online_entrypoint.py`, `tests/ops/test_recalibrate_thesis_threshold_artifacts_v4.py`.

**Expected result:** Offline and online use same c/tau; online EWMA receives q.

#### Stage 7.3 — Repository regression tests

1. Run existing focused group command for core/models/evaluation/online/ops.
2. Inspect failures for raw-vs-q expectation drift.
3. Run `git diff --check`.
4. Confirm only intended documentation/detail changes are present before implementation begins.

**Command:** `.venv/bin/python -m pytest tests/core tests/models tests/evaluation tests/online tests/ops`.

**Expected result:** Related regression tests pass and diff has no whitespace errors.

#### Stage 7.4 — One-combination end-to-end smoke

1. Use the existing debug CPU experiment config `scripts/configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O0__machine_1_6__w20__debug_cpu.yaml` or the repository's existing launch entry for that configuration.
2. Run the offline wrapper in its existing evaluation-only/dry-run mode as appropriate for the selected checkpoint.
3. Inspect generated clean-validation scores, thresholds, artifact provenance and report.
4. Run the corresponding one-combination online benchmark with the generated v4 artifact.
5. Inspect startup identity, q range, EWMA timeline, prediction and event record.
6. Do not expand to the benchmark matrix if this path fails.

**Existing entrypoints:** `scripts/run_thesis_offline_benchmark.py` and `scripts/run_thesis_online_benchmark.py`; exact arguments must be read from their parsers/config before execution.

**Expected result:** One concrete combination completes end-to-end with no raw-score threshold leakage.

#### Stage 7.5 — Recalibrate và kiểm tra acceptance

1. Recalibrate artifacts only after code/tests/smoke pass.
2. Validate every new artifact with `load_threshold_artifact`.
3. Check q finite/range on clean, synthetic and test outputs.
4. Check offline and online thresholds are separate q99 values.
5. Check c/tau equal across offline artifact and online runtime artifact for same identity.
6. Check prediction equality uses strict `>`.
7. Preserve old artifacts, checkpoint hashes and audit reports.
8. Approve broader benchmark only after all checks pass.

**Files/tests:** `scripts/ops/recalibrate_thesis_threshold_artifacts_v4.py`, artifact/report outputs, acceptance tests from previous phases.

**Inputs:** Tested code, representative checkpoint, clean validation and existing artifact inventory.

**Outputs:** Recalibrated v4 artifacts and acceptance evidence.

**Errors:** Any artifact mismatch, q range failure, threshold leakage or event inconsistency blocks broader runs.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest tests/models/test_multitask_shapes.py tests/online/test_online_calibration_contract.py tests/online/test_online_ewma_threshold.py tests/online/test_threshold_artifact.py tests/engine/test_threshold_artifact.py tests/ops/test_recalibrate_thesis_threshold_artifacts_v4.py` — focused suite pass.
- [ ] `.venv/bin/python -m pytest tests/core tests/models tests/evaluation tests/online tests/ops` — related regression suite pass.
- [ ] `git diff --check` — no whitespace error.

#### Manual

- [ ] Inspect one artifact/report pair and one online event stream for raw MSE -> q -> threshold/EWMA -> prediction.
- [ ] Confirm no existing artifact/output was deleted or overwritten.

### Risks and recovery

- **Risk:** Changed threshold semantics alter benchmark metrics.
- **Mitigation:** Retain old artifacts, checkpoint hashes, protocol and audit metadata; run one combination first.
- **Verification:** Compare provenance and score timelines before broad run.
- **Recovery:** Stop at artifact selection boundary; keep old outputs and do not delete new failed outputs until classified.

### Complete when

- Focused and regression tests pass.
- One end-to-end combination passes.
- Recalibrated artifact satisfies schema, identity, q99, c/tau, EWMA and strict prediction checks.

## Interface and data changes

### Proposed calibration interface

**Proposed new file:** `src/protocols/point_score_calibration.py`.

```text
fit_mad_logistic_calibration(raw_point_mse)
    -> PointScoreCalibration(center=c, tau=tau, metadata=...)

transform_point_scores(raw_point_mse, calibration)
    -> transformed q with the same shape/type boundary
```

Required rules:

- Fit only from finite raw clean-validation point MSE.
- Fit one calibration object per entity artifact.
- Reject empty/non-finite input and `tau <= 0`.
- Transform the MC mean raw point MSE, not each MC sample independently.

### Artifact schema

For THESIS v4, add:

```text
point_score_transform: "shifted-and-scaled logistic sigmoid"
point_score_c: float
point_score_tau: float
point_score_tau_estimator: "mad_based_robust_scale"
point_score_mad_normalizer: 0.6745
```

The existing threshold names remain stable:

- `offline_point_threshold_nonoverlap` — q99 of transformed offline clean timeline.
- `online_point_threshold_ewma` — q99 of EWMA transformed online clean timeline.

### Model output

- Top-level `point_scores`: q in official calibrated inference.
- `aux.point_score_samples`/existing stochastic query payload: raw per-sample MSE.
- `window_scores`: raw window reconstruction MSE.
- Training loss: raw reconstruction error.

### Compatibility

- THESIS active online runtime requires v4 calibration metadata.
- Non-THESIS baseline artifact callers retain their existing schema/semantics.
- Existing v3 artifacts are preserved for audit/legacy use and are not silently upgraded in place.

## Deployment and rollout

1. Implement and test the helper/artifact contract first.
2. Integrate model/scorer and pass model-level tests.
3. Generate one offline artifact and validate its provenance.
4. Recalibrate one representative v4 artifact without overwriting old output.
5. Run one online combination using that artifact.
6. Only then migrate the remaining THESIS artifacts sequentially.
7. Keep old artifacts and outputs until new artifacts pass identity/metric/provenance checks.
8. Do not run the full benchmark matrix before one-combination smoke succeeds.

## Documentation changes

- Keep `documents/spec/full-spec-v3.md` normative.
- Keep `documents/spec/offline_pretraining_terminology_ontology.md` and `documents/spec/online_tta_terminology_ontology.md` as terminology SSOT.
- Update those documents only if final implementation field names differ from the locked design.
- Add implementation evidence to the research log if the selected model/scorer seam differs from this detail plan.
- Record transform identity, c, tau, quantiles, EWMA weights, checkpoint hash and calibration split in reports/provenance.

## Final verification

- [ ] Raw point MSE is computed before score transformation.
- [ ] MC mean raw point MSE, not MSE of mean reconstruction, is transformed.
- [ ] Official `point_scores` is the shifted-and-scaled logistic sigmoid score.
- [ ] `aux.point_score_samples` remains raw.
- [ ] `window_scores` remains raw window reconstruction MSE.
- [ ] Offline and online use the same entity-level c/tau.
- [ ] Offline and online thresholds are separate q99 values.
- [ ] EWMA receives q with weights `0.9/0.1`.
- [ ] Online runtime does not calibrate from the test stream.
- [ ] Artifact identity, q range and strict `score > threshold` are verified.
- [ ] Existing artifacts remain recoverable.

## Assumptions and non-blocking uncertainties

- The direct user request is treated as approval to expand the proposed structure; the source structure file remains unchanged at `status: proposed`.
- `src/protocols/point_score_calibration.py` is a proposed new file. If an implementation-time file-size gate requires another owner, move the helper before coding and keep one implementation only.
- Runtime-only calibration state is proposed to remain artifact-owned rather than checkpoint-persisted. If resume requires model serialization, add an explicit identity check against the artifact before allowing continuation.
- Exact one-combination CLI arguments must be read from the current parser/config at implementation time; this detail plan does not invent unverified overrides.
