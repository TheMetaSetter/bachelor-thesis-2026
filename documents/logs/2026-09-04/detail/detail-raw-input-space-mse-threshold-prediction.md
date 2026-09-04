---
date: 2026-09-04T15:42:04+07:00
topic: "Chi tiết triển khai MSE trong raw input space cho thresholding và prediction"
status: implemented_with_validation_blocker
revision: 974af2b3a3d075f5cd4f3368f2cb584a5a8a3720
source_structure: documents/logs/2026-09-04/structure/structure-raw-input-space-mse-threshold-prediction.md
related_documents:
  - documents/logs/2026-09-04/plan/plan-raw-input-space-mse-threshold-prediction.md
  - documents/logs/2026-09-04/research/research-raw-input-space-mse-score-change-surface.md
  - prompts/4_detail_prompt.md
---

# Detailed Implementation: Raw-input-space MSE cho thresholding và prediction

## Summary

Thay score vận hành normalized/calibrated bằng simple MSE trên sensor units ban
đầu. Scorer dùng scaler của checkpoint để inverse-transform input và
reconstruction, sau đó trả về point-level và window-level MSE. Offline và online
dùng raw MSE cho threshold, prediction, EWMA và triage. Normalized MSE chỉ còn là
diagnostic có tên rõ. Synthetic validation phải giữ riêng normal/anomalous point
và normal/anomalous window.

## Source structure

Structure đã khóa sáu phase theo thứ tự:

1. Khóa contract raw-input MSE.
2. Tạo scaler-aware score primitives.
3. Chuyển offline evaluation.
4. Tạo raw threshold artifact và migration boundary.
5. Chuyển online EWMA, triage và prediction.
6. Kiểm chứng synthetic validation và histogram.

Tài liệu này chỉ mở rộng các phase và stage đó. Tài liệu không đổi training loss,
RedLamp, baseline khác hoặc artifact sigmoid lịch sử.

## Current state

- `SequenceStandardScaler.transform_sequence` chuẩn hóa active features trong
  `src/data/scalers.py:36-58`; class chưa có inverse-transform tensor.
- Dataset bundle giữ cả `raw_sequences` và `scaled_sequences` trong
  `src/data/loaders.py:150-193`. `WindowDataset.__getitem__` chỉ đưa scaled `x`
  vào window ở `src/data/loaders.py:231-298`.
- Checkpoint lưu `scaler_state_dict` trong `src/engine/checkpoint.py:185-215`.
  `rebuild_dataset_bundle_with_scaler_state` có thể nạp lại scaler ở
  `src/data/loaders.py:196-228`.
- Model tính MSE trên scaled `batch["x"]`. MC path tính per-sample MSE rồi lấy
  trung bình ở `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:186-196`.
  Model sau đó đưa point score qua sigmoid ở
  `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py:256-299`.
- Offline evaluator đọc `outputs["point_scores"]` ở
  `src/engine/evaluator.py:415-502`, rồi gộp các window chồng lấp thành một
  point timeline.
- Online calibration tính `input_window_score` bằng scaled `outputs["recon"]`
  và scaled batch ở `src/engine/online_tta/online_calibration.py:61-133`.
  Online window metrics lặp lại phép tính đó ở
  `src/engine/online_tta/online_engine_window_metrics.py:150-171`.
- Online runtime luôn nạp `PointScoreCalibration` ở
  `src/engine/online_tta/online_engine_run.py:103-131`; threshold artifact hiện
  là schema 4 và bắt buộc các field sigmoid ở
  `src/protocols/threshold_artifact.py:11-93,282-450`.
- Window ground-truth label có thể lấy bằng quy tắc hiện có
  `point_labels_to_window_labels` trong `src/data/api.py:226-231`: window
  anomalous nếu có ít nhất một anomalous point.

## Desired end state

Với `x_scaled [B,L,D]` và reconstruction samples
`reconstruction_scaled [B,M,L,D]`, scorer phải tính:

```text
x_raw = inverse_transform(x_scaled)
reconstruction_raw[m] = inverse_transform(reconstruction_scaled[m])
point_mse[m,t] = mean_d((x_raw[t,d] - reconstruction_raw[m,t,d])²)
raw_input_point_mse[t] = mean_m(point_mse[m,t])
raw_input_window_mse = mean_t(raw_input_point_mse[t])
```

`normalized_input_point_mse` và `normalized_input_window_mse` dùng cùng thứ tự
MC nhưng tính trên scaled tensors. `point_scores` ở raw protocol phải là
`raw_input_point_mse`; không có sigmoid transform. Window decision dùng
`raw_input_window_mse`. Threshold luôn được fit trên cùng score space mà runtime
sẽ dùng.

## Scope

### In scope

- Score contract, terminology và protocol config.
- Scaler inverse transform và scorer dùng chung.
- Offline threshold, prediction, metric và artifact export.
- Raw threshold artifact, provenance và mismatch rejection.
- Online calibration, EWMA, triage, prediction và runtime state.
- Focused tests, smoke run, ba synthetic validation machine và histogram.

### Out of scope

- Thay reconstruction training loss hoặc input representation khi train.
- Thêm channel weighting.
- Sửa RedLamp hoặc baseline khác.
- Xóa, ghi đè hoặc recalibrate artifact sigmoid lịch sử.
- So sánh độ lớn raw threshold giữa các machine khi chưa có quy tắc chuẩn hóa.

## Evidence

- `src/data/scalers.py:10-58` — scaler fit/transform và active-feature mask.
- `src/data/loaders.py:169-193,196-228` — raw/scaled sequence bundle và scaler checkpoint reload.
- `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:172-196` — reconstruction samples và MC normalized MSE.
- `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py:256-299` — sigmoid output và normalized `aux.raw_point_scores`.
- `src/engine/evaluator.py:122-264,415-554` — overlap aggregation, threshold và metrics.
- `src/engine/online_tta/online_calibration.py:61-248` — online/offline clean-validation score collection.
- `src/engine/online_tta/online_engine_window_metrics.py:81-277` — online score extraction, EWMA input và event output.
- `src/engine/online_tta/online_engine_window_core.py:22-251` — triage, update, buffer và event orchestration.
- `src/protocols/threshold_artifact.py:11-93,282-465` — schema 4 and sigmoid-specific artifact contract.
- `src/protocols/smd_benchmark_protocol.py:27-57` — current protocol validation rules.
- `scripts/benchmarks/run_thesis_offline_benchmark.py:310-714` — offline collection, threshold construction and export.
- `scripts/benchmarks/run_thesis_online_benchmark.py:376-395` — online CLI arguments.
- `src/data/api.py:226-231` — existing point-to-window label rule.

## Phase 1: Khóa contract raw-input MSE

### Goal

Tạo một định nghĩa duy nhất cho raw input space, normalized input space, point
score, window score, label và transform. Contract mới không dùng sigmoid cho raw
protocol và không làm thay đổi v3.

### Dependencies

- Structure raw-input MSE.
- Plan raw-input MSE.
- `documents/spec/full-spec-v1.md`, `full-spec-v2.md`, `full-spec-v3.md`.

### Stage 1.1 — Đối chiếu terminology và specification

**Files and symbols**

- Existing: `documents/spec/full-spec-v1.md`, `full-spec-v2.md`,
  `full-spec-v3.md`.
- Existing: `documents/spec/online_tta_terminology_ontology.md` and
  `documents/spec/offline_pretraining_terminology_ontology.md`.
- Proposed new file: `documents/spec/full-spec-v4.md`.
- Sections: v3 output contract, Section 8.2 score definition, Section 10
  terminology table, Section 12 prediction/triage, Section 14 compatibility.

**Current responsibility**

`full-spec-v3.md` gọi `point_scores` là sigmoid score, gọi `raw_point_scores`
là intermediate normalized reconstruction MSE, và dùng `window_scores` cho
window-level reconstruction score. Hai nghĩa của “raw” chưa chỉ rõ sensor units.

**Change**

Tạo v4 như một specification mới. Giữ nguyên v1-v3. Trong v4, dùng các tên
`raw_input_point_mse`, `raw_input_window_mse`,
`normalized_input_point_mse`, và `normalized_input_window_mse`. Đánh dấu
`point_scores` là operational alias chỉ trong raw protocol; tên alias không được
dùng để suy ra input space khi thiếu metadata.

**Inputs**

- Các tên và định nghĩa từ v1-v3.
- Current runtime keys từ source evidence ở trên.

**Outputs**

- Một terminology mapping table có cột `old_name`, `new_name`, `status`,
  `semantic_equivalence`, `owner`, và `migration_boundary`.
- Một v4 score table có input space, formula, granularity, threshold role và
  diagnostic/operational status.

**Errors**

- Dừng trước khi viết v4 nếu một tên runtime v3 có nhiều semantic khác nhau mà
  mapping table không thể phân biệt bằng owner, shape và lifecycle.

**Compatibility**

- Không sửa nội dung v1-v3.
- Không đổi tên field lịch sử trong artifact lịch sử.

**Atomic steps**

- [ ] Đọc các bảng score và prediction trong `full-spec-v1.md`, `full-spec-v2.md` và `full-spec-v3.md`.
- [ ] Ghi mỗi score runtime hiện có vào mapping table với đúng file owner và shape.
- [ ] Đánh dấu `raw_point_scores` cũ là normalized-input, uncalibrated intermediate score.
- [ ] Đánh dấu `point_scores` v3 là sigmoid-transformed score chỉ trong protocol cũ.
- [ ] Định nghĩa `raw_input_point_mse` là MSE trong original sensor units.
- [ ] Định nghĩa `normalized_input_point_mse` là MSE sau standardization.
- [ ] Ghi migration boundary giữa artifact v3/v4 lịch sử và raw protocol mới.
- [ ] Viết `documents/spec/full-spec-v4.md` mà không ghi đè `full-spec-v3.md`.
- [ ] Cập nhật hai ontology document để “raw input” luôn có hậu tố `input`.

### Stage 1.2 — Khóa công thức score

**Files and symbols**

- `documents/spec/full-spec-v4.md`, Section 8 score formulas.
- `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py`,
  `_build_monte_carlo_forward_outputs`.
- `src/data/api.py`, `point_labels_to_window_labels`.

**Current responsibility**

Model đã có `point_score_samples [B,M,L]` và `window_score_samples [B,M]` trên
scaled tensors. Model lấy `mean(dim=1)` cho MC score. Ground-truth point label
đã có trong batch, còn window label chưa được đưa vào evaluator record.

**Change**

Khóa công thức equal-feature MSE, mean-over-time window MSE và
mean-of-per-sample-MSE MC aggregation. Khóa window label là `max`/any trên
point labels: có ít nhất một anomalous point thì window anomalous. Label là
ground truth, không phải prediction.

**Inputs**

- `x_raw` or `x_scaled` with shape `[B,L,D]`.
- Reconstruction samples with shape `[B,M,L,D]`.
- Binary `point_labels [B,L]`, where `0` is normal and `1` is anomalous.

**Outputs**

- Point score shape `[B,L]`.
- Window score shape `[B]`.
- Window labels shape `[B]`.
- Prediction masks computed separately by `score > threshold`.

**Errors**

- Reject mismatched `[B,L,D]` and `[B,M,L,D]` dimensions.
- Reject non-finite score values.
- Reject any window label derived from scores instead of point labels.

**Atomic steps**

- [ ] Ghi công thức feature mean `mean_d((x - reconstruction)²)` vào v4.
- [ ] Ghi công thức window mean `mean_t(point_mse[t])` vào v4.
- [ ] Ghi công thức MC mean `mean_m(point_mse[m,t])` vào v4.
- [ ] Ghi rõ không được tính MSE từ reconstruction trung bình.
- [ ] Ghi `window_label = (point_labels.sum(dim=1) > 0).long()` vào v4.
- [ ] Ghi rõ `normal point`, `anomalous point`, `normal window`, và `anomalous window` là label categories.
- [ ] Ghi rõ `prediction` là kết quả threshold comparison và không thay thế ground-truth label.
- [ ] Ghi rõ sigmoid calibration không nằm trong raw protocol.

### Stage 1.3 — Khóa score identity của protocol

**Files and symbols**

- Existing: `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml`.
- Existing: `src/protocols/smd_benchmark_protocol.py:validate_protocol_config`.
- Existing tests: `tests/benchmarks/test_benchmark_protocol_config.py`.

**Current responsibility**

Protocol hiện khóa window, quantile, stride và EWMA nhưng không có score-space
selector. `validate_protocol_config` chưa reject calibrated transform.

**Change**

Thêm `score_space: raw_input` và `point_score_transform: identity`. Validator
phải yêu cầu đúng hai giá trị này cho raw protocol, đồng thời giữ nguyên
`window_size: 20`, clean-validation splits, stride và EWMA weights.

**Atomic steps**

- [ ] Thêm `score_space: raw_input` vào YAML protocol hiện có.
- [ ] Thêm `point_score_transform: identity` vào YAML protocol hiện có.
- [ ] Đọc hai field mới trong `validate_protocol_config`.
- [ ] Reject protocol thiếu `score_space`.
- [ ] Reject protocol có `score_space` khác `raw_input`.
- [ ] Reject protocol có `point_score_transform` khác `identity`.
- [ ] Giữ nguyên mọi threshold split, quantile, stride và EWMA value hiện tại.
- [ ] Thêm assertion protocol test cho cặp `raw_input` và `identity`.

### Tests and verification

- **Automated:**
  `.venv/bin/python -m pytest -q tests/benchmarks/test_benchmark_protocol_config.py`
  phải pass và phải có assertion cho raw protocol identity.
- **Manual:** Đọc score table v4 và xác nhận không có threshold/prediction nào
  trỏ tới sigmoid output.
- **Risk:** Nếu v4 ghi đè v3, historical replay mất semantic. Mitigation là
  tạo file v4 riêng và kiểm tra v3 vẫn tồn tại nguyên trạng.

### Complete when

V4, ontology và protocol config định nghĩa cùng một raw identity; test reject
được missing/mismatched identity; v1-v3 và artifact lịch sử chưa bị sửa.

## Phase 2: Tạo scaler-aware score primitives

### Goal

Tính raw point/window MSE từ scaled model input và reconstruction samples bằng
scaler đã fit, không thay đổi model training objective.

### Dependencies

- Phase 1 đã khóa formula và `score_space`.
- Checkpoint có `scaler_state_dict`.

### Stage 2.1 — Xác lập inverse-transform boundary

**Files and symbols**

- `src/data/scalers.py`, `SequenceStandardScaler`.
- `src/data/loaders.py`, `rebuild_dataset_bundle_with_scaler_state`.
- Proposed tests: `tests/evaluation/test_raw_input_mse_scores.py`.

**Current responsibility**

`transform_sequence` dùng active mask và `clamp(feature_std, min=epsilon)` khi
scale. Inactive features được giữ nguyên. Scaler chỉ kiểm tra fit state khi
transform.

**Change**

Thêm operation `inverse_transform_tensor(values: torch.Tensor) -> torch.Tensor`
cho tensor có feature dimension ở vị trí cuối. Operation dùng cùng active mask
và clamped std: `raw = scaled * std + mean` trên active features; inactive
features giữ nguyên.

**Inputs**

- Floating tensor `[L,D]`, `[B,L,D]` hoặc `[B,M,L,D]`.
- Fitted `feature_mean`, `feature_std`, `feature_active_mask`.

**Outputs**

- Tensor cùng shape, device và dtype logic với input.

**Errors**

- Raise `RuntimeError` nếu scaler chưa fit hoặc state thiếu mean/std/mask.
- Raise `TypeError` nếu input không phải tensor.
- Raise `ValueError` nếu last dimension khác số feature trong scaler.

**Atomic steps**

- [ ] Thêm `inverse_transform_tensor` sau `_resolve_active_feature_std`.
- [ ] Chuyển mean/std/mask sang device của input trước phép tính.
- [ ] Áp dụng inverse transform bằng indexing feature dimension cuối.
- [ ] Giữ nguyên inactive feature values.
- [ ] Giữ nguyên input tensor không bị mutate tại chỗ.
- [ ] Dùng float arithmetic và trả về shape như input.
- [ ] Thêm test inverse-transform cho active feature có mean/std khác 0.
- [ ] Thêm test inactive feature không đổi sau inverse transform.
- [ ] Thêm test round-trip `inverse_transform_tensor(transform_sequence(x))`.
- [ ] Thêm test reject scaler chưa fit và feature dimension sai.

### Stage 2.2 — Tạo shared point/window raw scorer

**Files and symbols**

- Proposed new file: `src/protocols/reconstruction_scores.py`.
- Proposed symbol:
  `score_reconstruction(input_scaled, reconstruction, scaler) -> dict[str, torch.Tensor]`.
- Existing consumer boundary: `src/engine/evaluator.py` and
  `src/engine/online_tta/online_engine_window_metrics.py`.

**Current responsibility**

Model và online code đang lặp normalized reconstruction MSE ở nhiều boundary.
`src/protocols/point_scores.py` chỉ xử lý timeline/overlap, không sở hữu
reconstruction arithmetic.

**Change**

Tạo một scorer nhỏ, không đưa scaler vào model. Scorer nhận scaled input và
deterministic hoặc MC reconstruction. Scorer trả bốn field:

```text
raw_input_point_mse [B,L]
raw_input_window_mse [B]
normalized_input_point_mse [B,L]
normalized_input_window_mse [B]
```

**Inputs**

- `input_scaled: Tensor[B,L,D]`.
- `reconstruction: Tensor[B,L,D]` hoặc `Tensor[B,M,L,D]`.
- Fitted `SequenceStandardScaler`.

**Outputs**

- Deterministic input được xem như `M=1`.
- MC path tính per-sample point/window MSE rồi mean theo `M`.
- Mọi output finite và có dtype/device phù hợp với score arithmetic.

**Errors**

- Reject input rank khác 3.
- Reject reconstruction rank khác 3 hoặc 4.
- Reject reconstruction batch/window/feature shape không khớp input.
- Reject non-finite input, reconstruction hoặc score.

**Atomic steps**

- [ ] Tạo module `src/protocols/reconstruction_scores.py`.
- [ ] Viết validator shape cho input `[B,L,D]`.
- [ ] Viết nhánh deterministic bằng cách thêm singleton MC dimension.
- [ ] Inverse-transform input một lần bằng scaler.
- [ ] Inverse-transform từng MC reconstruction sample.
- [ ] Tính raw per-sample point MSE bằng feature mean.
- [ ] Tính raw per-sample window MSE bằng time mean.
- [ ] Tính normalized per-sample point MSE trên scaled tensors.
- [ ] Tính normalized per-sample window MSE trên scaled tensors.
- [ ] Lấy mean theo MC dimension cho cả bốn output.
- [ ] Trả đúng bốn field với shape `[B,L]` và `[B]`.
- [ ] Không mutate input hoặc reconstruction tensors.
- [ ] Không gọi `transform_official_point_scores` trong scorer.

### Stage 2.3 — Tích hợp MC aggregation tại score boundary

**Files and symbols**

- `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py`,
  `_build_monte_carlo_forward_outputs`.
- `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py`,
  `forward`.
- `src/protocols/reconstruction_scores.py`.

**Current responsibility**

Model giữ `reconstruction_samples` trong `aux.stochastic_query` và giữ
normalized `point_score_samples`. Top-level `point_scores` hiện sigmoid.

**Change**

Giữ model output cũ để compatibility, nhưng downstream raw protocol phải lấy
`reconstruction_samples` nếu có và dùng shared scorer. Khi samples không được
giữ, downstream dùng `outputs["recon"]` như deterministic input. Không đổi
training loss hoặc model input.

**Atomic steps**

- [ ] Xác nhận `aux.stochastic_query.reconstruction_samples` có shape `[B,M,L,D]` trong eval path.
- [ ] Xác nhận `outputs["recon"]` có shape `[B,L,D]` cho fallback deterministic.
- [ ] Ghi rõ consumer precedence: MC samples trước, deterministic recon sau.
- [ ] Giữ `outputs["point_scores"]` và `outputs["window_scores"]` để replay v3.
- [ ] Không dùng top-level sigmoid field trong raw evaluator.
- [ ] Truyền post-injection `batch["x"]` vào scorer.
- [ ] Thêm test phân biệt `mean(MSE(sample_m))` với `MSE(mean(sample_m))`.
- [ ] Thêm test xác nhận raw scorer không gọi calibration object.

### Stage 2.4 — Kiểm chứng số học cơ bản

**Files and symbols**

- Proposed: `tests/evaluation/test_raw_input_mse_scores.py`.
- `SequenceStandardScaler` và `score_reconstruction`.

**Atomic steps**

- [ ] Tạo toy input có hai feature với mean/std đã biết.
- [ ] Tạo toy reconstruction có sai số khác nhau theo feature.
- [ ] Tính expected raw point MSE bằng tay trong test.
- [ ] Tính expected raw window MSE bằng tay trong test.
- [ ] Tính expected normalized point/window MSE bằng tay trong test.
- [ ] Assert raw và normalized output khác nhau khi std khác 1.
- [ ] Assert tất cả score finite.
- [ ] Assert output shape đúng `[B,L]` và `[B]`.
- [ ] Chạy `.venv/bin/python -m pytest -q tests/evaluation/test_raw_input_mse_scores.py`.

### Tests and verification

- **Automated:** Scaler round-trip, hand-computed MSE, MC aggregation, shape và
  finite checks phải pass.
- **Manual:** In một toy batch trước và sau inverse transform; xác nhận subtraction
  diễn ra trong sensor units.
- **Risk:** Inverse-transform hai lần làm score sai. Test phải dùng cùng toy
  tensor để assert normalized/raw expected values.

### Complete when

Một scorer dùng chung trả đúng bốn tensor score và có test số học độc lập với
evaluator/online runtime.

## Phase 3: Chuyển offline evaluation sang raw score

### Goal

Offline clean-validation threshold, synthetic/test prediction, metrics và score
artifacts đều dùng raw-input MSE.

### Dependencies

- Phase 1 protocol identity.
- Phase 2 scorer.
- `Evaluator` vẫn giữ overlap averaging và coverage mask hiện có.

### Stage 3.1 — Nối scorer vào evaluator

**Files and symbols**

- `src/engine/evaluator.py`, `Evaluator.evaluate`,
  `_validate_window_payload`, `accumulate_pointwise_window_payload`,
  `_build_reconstructed_evaluation_record`.
- Proposed test: `tests/evaluation/test_raw_input_mse_scores.py`.

**Current responsibility**

Evaluator lấy `outputs["point_scores"]`, đưa chúng vào batch payload, overlap
average và threshold. Evaluator chưa có scaler/scoring-space argument.

**Change**

Thêm raw scoring context vào evaluator boundary. Raw protocol yêu cầu fitted
scaler. Với mỗi batch, chọn MC reconstruction samples hoặc deterministic recon,
gọi `score_reconstruction`, rồi lưu operational raw point/window scores cùng
normalized diagnostics.

**Proposed interface**

```text
Evaluator.evaluate(
    model,
    data_loader,
    point_score_threshold=None,
    threshold_source=None,
    *,
    score_space="model_output",
    scaler=None,
)
```

Raw protocol phải dùng `score_space="raw_input"` và `scaler` không được `None`.

**Atomic steps**

- [ ] Thêm `score_space` và `scaler` vào `Evaluator.evaluate` theo keyword-only boundary.
- [ ] Reject `score_space="raw_input"` khi scaler không tồn tại.
- [ ] Trích MC reconstruction samples từ `outputs["aux"]["stochastic_query"]` khi samples có mặt.
- [ ] Dùng `outputs["recon"]` khi MC samples không có mặt.
- [ ] Gọi `score_reconstruction` với batch sau mọi synthetic preparation.
- [ ] Đặt raw point score vào operational payload field của raw protocol.
- [ ] Đặt normalized point score vào diagnostic payload field.
- [ ] Đặt raw và normalized window score vào từng window payload.
- [ ] Tạo window label bằng `point_labels_to_window_labels`.
- [ ] Giữ `covered_point_mask` và overlap sum/count cho raw timeline.
- [ ] Gộp normalized timeline bằng cùng start/end index và coverage rule.
- [ ] Dùng raw point timeline cho `resolve_evaluation_threshold` khi raw protocol.
- [ ] Dùng raw point timeline cho `compute_pointwise_metrics` khi raw protocol.
- [ ] Trả window records riêng để window metrics không suy ra từ point threshold.

### Stage 3.2 — Bảo toàn overlap aggregation

**Files and symbols**

- `src/engine/evaluator.py`, `accumulate_pointwise_window_payload` và
  `reconstruct_pointwise_records_from_window_payload`.
- `src/protocols/point_scores.py`, timeline helpers.

**Change**

Mỗi score space có sum/count độc lập nhưng dùng cùng window metadata. Không
được để raw và normalized timeline có coverage khác nhau.

**Atomic steps**

- [ ] Mở rộng batch payload validator để kiểm tra raw/normalized point shape `[B,L]`.
- [ ] Mở rộng accumulator để cộng raw score theo `[start_index:end_index]`.
- [ ] Mở rộng accumulator để cộng normalized score theo cùng slice.
- [ ] Tăng score count một lần cho cả hai score space.
- [ ] Giữ point label aggregation bằng elementwise maximum.
- [ ] Tạo `covered_point_mask` một lần và dùng cho hai timeline.
- [ ] Kiểm tra entity metadata trước khi ghi vào accumulator.
- [ ] Tạo evaluation record với raw operational field và normalized diagnostic field.
- [ ] Assert raw/normalized record lengths bằng nhau.
- [ ] Thêm test hai window chồng lấp có raw values khác nhau.
- [ ] Assert point ở vùng overlap bằng trung bình raw của các window phủ điểm đó.
- [ ] Assert normalized overlap dùng cùng count nhưng không thay raw value.

### Stage 3.3 — Chuyển threshold và prediction

**Files and symbols**

- `src/engine/evaluator.py`, `resolve_evaluation_threshold` call và metrics call.
- `src/engine/thresholding.py`, existing threshold helpers.
- `scripts/benchmarks/run_thesis_offline_benchmark.py`,
  `_evaluate_offline_benchmark_splits` và `_evaluate_named_split`.

**Change**

Clean validation fit offline point threshold từ raw point MSE. Synthetic/test
prediction so sánh raw point MSE với threshold. Window prediction so sánh raw
window MSE với window threshold. Normalized fields chỉ phục vụ diagnostic.

**Atomic steps**

- [ ] Truyền `score_space="raw_input"` và data bundle scaler vào mọi THESIS offline evaluation.
- [ ] Bỏ bước `fit_mad_logistic_calibration` khỏi raw operational path.
- [ ] Không gọi `model.set_point_score_calibration` cho raw protocol.
- [ ] Fit offline point threshold từ clean-validation raw point scores.
- [ ] Truyền raw threshold vào synthetic validation evaluator.
- [ ] Truyền raw threshold vào test evaluator.
- [ ] Tính window prediction bằng raw window score và raw window threshold.
- [ ] Giữ threshold source là clean validation.
- [ ] Giữ test labels ở metrics-only role.
- [ ] Thêm test synthetic perturbation sau injection làm raw score tăng.
- [ ] Thêm test normalized score không làm thay đổi raw prediction.

### Stage 3.4 — Xuất artifact song song có provenance

**Files and symbols**

- `scripts/benchmarks/run_thesis_offline_benchmark.py`,
  `_evaluation_outputs_to_score_payload`, `_write_score_npz`,
  `_build_uq_summary_inputs`, `_export_offline_artifacts`.
- Existing output directories under `outputs/benchmark` or
  `outputs/benchmark_smoke`.

**Change**

Mở rộng score payload và `.npz` export để ghi raw point/window score, labels,
prediction và normalized diagnostics. Ghi score-space metadata trong protocol,
threshold artifact và report. Không ghi đè historical sigmoid output directory.

**Proposed `.npz` fields**

```text
raw_input_point_mse
normalized_input_point_mse
raw_input_window_mse
normalized_input_window_mse
point_labels
window_labels
point_predictions
window_predictions
covered_point_mask
```

**Atomic steps**

- [ ] Đổi payload converter để giữ raw point/window arrays thay vì chỉ `point_scores`.
- [ ] Giữ label arrays riêng cho point và window.
- [ ] Ghi point prediction bằng raw point threshold.
- [ ] Ghi window prediction bằng raw window threshold.
- [ ] Mở rộng `_write_score_npz` với các field raw/normalized/label/prediction.
- [ ] Ghi `score_space` và `point_score_transform` vào resolved protocol JSON.
- [ ] Đưa raw score fields vào UQ summary input mà không đổi split names.
- [ ] Ghi threshold source và checkpoint/config provenance vào report.
- [ ] Giữ normalized diagnostics dưới tên đầy đủ, không dùng `raw_point_scores` mơ hồ.
- [ ] Giữ output hierarchy `outputs/<experiment_type>/<dataset>/<entity>/<seed>/<method>/<phase>/<stage>`.
- [ ] Cập nhật export tests để đọc đúng từng field.

### Tests and verification

- **Automated:**
  `.venv/bin/python -m pytest -q tests/evaluation/test_evaluator_thresholding.py tests/benchmarks/test_thesis_offline_artifact_exports.py tests/benchmarks/test_run_offline_benchmark.py`.
- **Manual:** Mở một clean window và một synthetic window; kiểm tra raw score,
  point/window label, threshold và prediction là các field riêng.
- **Risk:** Scorer đọc sequence trước injection. Test phải inject perturbation
  trước gọi evaluator và assert raw score phản ứng với perturbation.

### Complete when

Một offline smoke evaluator tạo raw point/window threshold và prediction; output
giữ normalized diagnostics; window labels và point labels không bị trộn.

## Phase 4: Persist raw threshold artifact và migration boundary

### Goal

Artifact raw chứng minh được score space, transform, formula, checkpoint và
config provenance. Artifact normalized/sigmoid bị reject trong raw runtime;
artifact lịch sử vẫn đọc được khi replay đúng protocol cũ.

### Dependencies

- Phase 1 identity.
- Phase 3 raw thresholds.

### Stage 4.1 — Định nghĩa raw artifact schema

**Files and symbols**

- `src/protocols/threshold_artifact.py`, constants,
  `validate_threshold_artifact`, `build_threshold_artifact`,
  `load_threshold_artifact`.
- Existing schema versions 3 and 4.
- Proposed new schema version: 5, subject to confirming no later schema exists
  before implementation.

**Change**

Thêm schema 5 cho raw-input MSE. Schema 5 bắt buộc:

```text
score_space: raw_input
point_score_transform: identity
point_score_definition: raw_input_point_mse
window_score_definition: raw_input_window_mse
```

Schema 5 không bắt buộc `point_score_c`, `point_score_tau`,
`point_score_tau_estimator` hoặc `point_score_mad_normalizer`.

**Atomic steps**

- [ ] Kiểm tra constant schema hiện tại là 4 trước khi thêm version kế tiếp.
- [ ] Đặt constant raw schema version là 5 nếu không có version cao hơn.
- [ ] Thêm raw identity fields vào required key set của schema 5.
- [ ] Thêm raw point/window definition fields vào schema 5.
- [ ] Giữ validation branch riêng cho schema 3.
- [ ] Giữ validation branch sigmoid riêng cho schema 4 THESIS.
- [ ] Bổ sung raw fields vào `provenance` và kiểm tra equality với top-level fields.
- [ ] Đổi score rules raw thành tên có input-space rõ ràng.
- [ ] Giữ latent window score rule riêng vì latent distance không phải input MSE.
- [ ] Cập nhật builder để nhận raw identity và không nhận sigmoid parameters bắt buộc.
- [ ] Ghi `calibration_split: clean_validation` cho mọi raw threshold record.

### Stage 4.2 — Validation và mismatch rejection

**Files and symbols**

- `src/protocols/threshold_artifact.py`, `validate_threshold_artifact`.
- `src/engine/online_tta/online_engine_run.py`,
  `_validate_online_artifact_identity`.
- `src/engine/online_tta/checkpoint_resolution.py`, artifact resolution.

**Atomic steps**

- [ ] Reject schema 5 khi `score_space` không phải `raw_input`.
- [ ] Reject schema 5 khi transform không phải `identity`.
- [ ] Reject schema 5 khi point/window definition thiếu hoặc sai.
- [ ] Reject raw artifact có sigmoid-specific fields với giá trị operational không tương thích.
- [ ] Reject raw protocol khi artifact schema là 4 sigmoid.
- [ ] Reject artifact có checkpoint hash khác checkpoint runtime.
- [ ] Reject artifact có entity, variant, seed hoặc window size khác config.
- [ ] Reject artifact có EWMA weights khác protocol config.
- [ ] Dùng error message nêu rõ field mismatch và expected/actual values.
- [ ] Thêm unit test cho normalized-space mismatch.
- [ ] Thêm unit test cho sigmoid-transform mismatch.
- [ ] Thêm unit test cho missing score identity.
- [ ] Thêm unit test cho checkpoint/config mismatch.

### Stage 4.3 — Giữ compatibility lịch sử

**Files and symbols**

- `src/protocols/threshold_artifact.py`, schema 3/4 branches.
- `tests/engine/test_threshold_artifact.py`.
- `tests/online/test_threshold_artifact.py`.

**Change**

`load_threshold_artifact` tiếp tục đọc artifact v3/v4 hợp lệ. Raw runtime chỉ
chấp nhận schema 5. Không tự động nâng cấp hoặc rewrite JSON cũ.

**Atomic steps**

- [ ] Giữ test schema 4 round-trip với sigmoid fields.
- [ ] Thêm fixture schema 3 cho historical read.
- [ ] Assert load historical artifact không ghi file.
- [ ] Assert raw runtime không chọn historical artifact khi config trỏ nhầm.
- [ ] Assert error message hướng người chạy tới raw artifact mới.
- [ ] Không sửa path dưới các thư mục historical sigmoid.
- [ ] Thêm test raw artifact round-trip riêng với schema 5.

### Stage 4.4 — Liên kết artifact với offline output

**Files and symbols**

- `scripts/benchmarks/run_thesis_offline_benchmark.py`, `_build_thresholds` và
  `_export_offline_artifacts`.
- `src/protocols/threshold_artifact.py`, `build_threshold_artifact`.

**Atomic steps**

- [ ] Lấy raw clean-validation point MSE từ payload đã export nội bộ.
- [ ] Lấy raw online EWMA point MSE từ Phase 5 collector.
- [ ] Lấy raw input-window MSE cho `input_window` threshold.
- [ ] Giữ latent thresholds từ latent score collector.
- [ ] Gọi raw artifact builder với `score_space="raw_input"`.
- [ ] Gọi raw artifact builder với `point_score_transform="identity"`.
- [ ] Ghi checkpoint SHA256 vào artifact và provenance.
- [ ] Ghi resolved config SHA256 vào artifact và provenance.
- [ ] Ghi machine/entity, seed, split, stride và window size vào artifact.
- [ ] Ghi artifact path vào benchmark report.
- [ ] Assert `write_threshold_artifact` validate trước khi ghi.

### Tests and verification

- **Automated:**
  `.venv/bin/python -m pytest -q tests/engine/test_threshold_artifact.py tests/online/test_threshold_artifact.py`.
- **Manual:** Mở raw JSON mới; xác nhận có raw identity/definition và không có
  yêu cầu sigmoid calibration fields.
- **Risk:** Config cũ trỏ tới schema 4. Raw runtime phải fail rõ ràng, không
  fallback sang score khác.

### Complete when

Schema 5 round-trip pass, mismatch cases bị reject, v3/v4 historical read vẫn
pass, và artifact mới liên kết đúng threshold với checkpoint/config.

## Phase 5: Chuyển online EWMA, triage và prediction sang raw score

### Goal

Mọi operational decision online dùng raw-input MSE cùng raw threshold. Event và
runtime state lưu raw operational score, normalized diagnostic score và identity.

### Dependencies

- Phase 2 scorer.
- Phase 4 raw artifact.
- Current causal stride-1, absolute-index và event state contracts.

### Stage 5.1 — Đưa scaler và score identity vào runtime context

**Files and symbols**

- `src/engine/online_tta/online_engine_run.py`,
  `_build_runtime_online_context`, `_run_online_sequence`.
- `src/engine/online_tta/online_engine_shared.py` nếu cần truyền context.
- `src/data/loaders.py`, `rebuild_dataset_bundle_with_scaler_state`.
- `src/engine/online_tta/runtime_state.py`, `OnlineRuntimeState`.

**Change**

Runtime nạp scaler khớp checkpoint và truyền scaler vào `_score_online_window`.
Runtime đọc raw identity từ artifact trước khi model forward. Runtime state giữ
identity cùng threshold artifact để continuation không đổi score space.

**Atomic steps**

- [ ] Load threshold artifact trước khi khởi tạo operational online model state.
- [ ] Validate raw identity trước khi bắt đầu stream.
- [ ] Lấy scaler từ data bundle hoặc rebuild bundle bằng checkpoint scaler state.
- [ ] Truyền scaler từ runtime context tới `_run_online_sequence`.
- [ ] Truyền scaler tiếp tới `_process_online_window` và `_score_online_window`.
- [ ] Giữ stream input ở scaled representation.
- [ ] Không thêm raw tensors vào model batch chỉ để scoring.
- [ ] Ghi `score_space` và transform vào runtime state hoặc artifact reference.
- [ ] Reject resume state nếu score identity khác artifact hiện tại.
- [ ] Giữ causal indexing và batch size 1.

### Stage 5.2 — Chuyển online calibration

**Files and symbols**

- `src/engine/online_tta/online_calibration.py`, `_collect_batch_scores`,
  `run_stride1_sequence_scores`, `_collect_offline_scores`,
  `collect_clean_validation_scores`.
- `src/engine/thresholding.py`, `select_clean_validation_point_threshold`,
  `select_online_ewma_threshold`.

**Change**

Collector trả raw point MSE, raw input-window MSE và normalized diagnostics.
Offline point threshold, online EWMA threshold và input-window threshold đều
fit từ raw clean validation.

**Atomic steps**

- [ ] Thêm scaler argument vào `_collect_batch_scores`.
- [ ] Chọn reconstruction samples hoặc deterministic recon trong collector.
- [ ] Gọi shared scorer với scaled online batch.
- [ ] Trả raw point vector thay cho `outputs["point_scores"]` operationally.
- [ ] Trả raw input-window scalar thay cho scaled `input_scores`.
- [ ] Giữ latent window score unchanged.
- [ ] Đưa raw point vector vào `update_window_point_ewma`.
- [ ] Giữ absolute indices trong stride-1 EWMA update.
- [ ] Thu raw EWMA values vào `collect_stride1_online_scores`.
- [ ] Fit `online_ewma_point_threshold` từ raw EWMA values.
- [ ] Fit `input_window_threshold` từ raw window values.
- [ ] Bỏ calibration sigmoid khỏi raw collector.
- [ ] Giữ test clean-validation không có test labels.

### Stage 5.3 — Chuyển EWMA và triage

**Files and symbols**

- `src/engine/online_tta/online_engine_window_metrics.py`,
  `_extract_online_window_scores`, `_score_online_window`,
  `_update_online_window_buffers`.
- `src/engine/online_tta/online_engine_window_core.py`,
  `_prepare_online_window_event`, `_admit_and_verify_gray_zone`.
- `src/engine/online_tta/triage.py`, `classify_online_window`.

**Change**

`update_window_point_ewma` nhận raw point MSE. `classify_online_window` nhận
raw input-window MSE cho B threshold; latent score vẫn giữ vai trò latent
geometry. Buffer admission lưu raw score có tên rõ.

**Atomic steps**

- [ ] Đổi `_extract_online_window_scores` để gọi shared scorer.
- [ ] Trả operational raw point vector từ scorer.
- [ ] Trả normalized point vector dưới diagnostic field.
- [ ] Trả raw input-window scalar từ scorer.
- [ ] Trả normalized input-window scalar dưới diagnostic field.
- [ ] Truyền operational raw point vector vào `update_window_point_ewma`.
- [ ] Truyền raw input-window scalar vào `classify_online_window`.
- [ ] Giữ latent low/high threshold và latent score không đổi.
- [ ] Dùng raw point score cho hard-old/Pnn event payload.
- [ ] Dùng raw input-window score cho verification buffer entry.
- [ ] Reject fallback nếu chỉ có normalized/calibrated point field.
- [ ] Thêm test raw values điều khiển triage boundary.

### Stage 5.4 — Chuyển prediction và event state

**Files and symbols**

- `src/engine/online_tta/online_engine_step.py`, `_compute_step_scores`,
  `_build_step_record`, `execute_online_tta_step`.
- `src/engine/online_tta/online_engine_window_metrics.py`,
  `_build_online_window_outputs`.
- `src/engine/online_tta/runtime_state.py`, `OnlineRuntimeState.to_dict` và
  `validate_resume_state`.

**Change**

Record mới phải phân biệt operational raw score và normalized diagnostic. `prediction`
được tính từ raw EWMA point score với raw threshold. Window prediction dùng raw
window score với raw input-window threshold. Event state phải giữ identity.

**Proposed record fields**

```text
raw_input_point_mse
normalized_input_point_mse
raw_input_window_mse
normalized_input_window_mse
ewma_raw_input_point_mse
point_prediction
window_prediction
score_space
point_score_transform
```

**Atomic steps**

- [ ] Đổi `_compute_step_scores` fallback để không đọc `outputs["point_scores"]` cho raw protocol.
- [ ] Truyền raw point score đã tính ở window boundary vào step record.
- [ ] Đặt `prediction` từ raw EWMA score và raw point threshold.
- [ ] Đặt `point_prediction` alias rõ nghĩa nếu record contract cần backward compatibility.
- [ ] Ghi raw window score và window prediction vào event record.
- [ ] Ghi normalized point/window diagnostics riêng.
- [ ] Ghi raw identity vào record và metric payload.
- [ ] Giữ `absolute_indices`, `point_index` và window start/end hiện có.
- [ ] Giữ `active_ewma_point_scores` theo absolute index.
- [ ] Ghi raw score fields vào `OnlineRuntimeState.verification_history`.
- [ ] Validate identity khi serialize/restore runtime state.
- [ ] Thêm test continuation với matching raw artifact.
- [ ] Thêm test resume reject với mismatched raw artifact.

### Tests and verification

- **Automated:**
  `.venv/bin/python -m pytest -q tests/online/test_online_calibration_contract.py tests/online/test_online_ewma_threshold.py tests/online/test_online_tta_triage.py tests/online/test_online_runtime_state.py tests/online/test_online_state_roundtrip.py`.
- **Manual:** Trace một event từ reconstruction đến EWMA, triage, threshold và
  record; mọi comparison phải dùng raw-input value.
- **Risks:** Fallback có thể bypass scorer; test phải dùng fake output chỉ có
  normalized field và xác nhận raw runtime fail.

### Complete when

Online calibration, EWMA, triage, prediction, event và continuation đều dùng
raw-input score với artifact schema 5 tương thích.

## Phase 6: Synthetic validation và histogram

### Goal

Chạy smoke trước rồi chạy `machine-1-6`, `machine-3-4`, `machine-3-9`. Output
phân biệt normal/anomalous point và window. Histogram tách hai score level, hai
label category và đánh dấu threshold raw.

### Dependencies

- Phase 3 offline path.
- Phase 4 raw artifact.
- Phase 5 online path.

### Stage 6.1 — Focused regression tests

**Files and symbols**

- Existing tests: `tests/evaluation/test_evaluator_thresholding.py`,
  `tests/evaluation/test_point_score_contracts.py`,
  `tests/benchmarks/test_thesis_offline_artifact_exports.py`,
  `tests/online/test_online_calibration_contract.py`,
  `tests/online/test_online_ewma_threshold.py`,
  `tests/online/test_online_tta_triage.py`.
- Proposed: `tests/evaluation/test_raw_input_mse_scores.py`.

**Atomic steps**

- [ ] Chạy scorer unit tests.
- [ ] Chạy scaler round-trip tests.
- [ ] Chạy evaluator overlap/threshold tests.
- [ ] Chạy threshold artifact schema tests.
- [ ] Chạy online calibration/EWMA/triage tests.
- [ ] Chạy online runtime-state continuation tests.
- [ ] Chạy benchmark export tests.
- [ ] Ghi lại command và số test pass trong experiment log.

### Stage 6.2 — End-to-end smoke combination

**Files and symbols**

- `scripts/benchmarks/run_thesis_offline_benchmark.py`, CLI parser at
  `:834-857`.
- `scripts/benchmarks/run_thesis_online_benchmark.py`, CLI parser at
  `:376-395`.
- One existing THESIS smoke experiment config for a requested machine.

**Atomic steps**

- [ ] Chọn một config THESIS smoke có checkpoint hợp lệ và ghi path đầy đủ vào log.
- [ ] Chọn raw protocol YAML đã có `score_space: raw_input` và `point_score_transform: identity`.
- [ ] Chạy offline wrapper với `--experiment-config`, `--protocol-config` và smoke config đã chọn.
- [ ] Xác nhận offline smoke tạo raw threshold artifact.
- [ ] Xác nhận offline smoke tạo raw point/window score arrays.
- [ ] Xác nhận offline smoke tạo label arrays và prediction arrays.
- [ ] Chạy online wrapper với `--experiment-config`, `--protocol-config` và `--online-variant A0`.
- [ ] Xác nhận online smoke load được raw artifact.
- [ ] Xác nhận online smoke ghi raw score identity trong records.
- [ ] Dừng mở rộng run nếu smoke có score-space, shape hoặc provenance failure.

### Stage 6.3 — Synthetic validation cho ba machine

**Data and configs**

- SMD entities: `machine-1-6`, `machine-3-4`, `machine-3-9`.
- Ground-truth files: `data/ServerMachineDataset/test_label/<entity>.txt`.
- Existing THESIS experiment config families under
  `scripts/configs/experiment/offline_benchmark/thesis/` and
  `scripts/configs/experiment/online_benchmark/thesis/`.
- Canonical outputs under `outputs/benchmark` or `outputs/benchmark_smoke`.

**Atomic steps**

- [ ] Chạy synthetic validation cho `machine-1-6` bằng raw protocol sau smoke pass.
- [ ] Chạy synthetic validation cho `machine-3-4` bằng raw protocol sau machine-1-6 pass.
- [ ] Chạy synthetic validation cho `machine-3-9` bằng raw protocol sau machine-3-4 pass.
- [ ] Ghi entity, seed, variant, checkpoint path và config path cho từng run.
- [ ] Ghi synthetic injector seed và anomaly metadata cho từng run.
- [ ] Ghi point-level raw scores và point labels cho từng run.
- [ ] Ghi window-level raw scores và window labels cho từng run.
- [ ] Ghi threshold và prediction arrays cho từng run.
- [ ] Ghi normalized diagnostics cùng run nhưng không dùng chúng để threshold.
- [ ] Ghi `score_space` và transform vào mỗi run report.
- [ ] Lưu report-ready summaries, provenance và plots thay vì mọi forward tensor.
- [ ] Calibrate threshold riêng cho từng entity.
- [ ] Không so sánh trực tiếp numeric raw thresholds giữa ba entity.

### Stage 6.4 — Tạo và kiểm tra histogram

**Proposed new file:** `scripts/visualization/plot_raw_input_mse_histograms.py`.

**Current responsibility**

`scripts/plot_machine_3_4_anomaly_scores.py` đang đọc một record cũ và vẽ score
distribution không phân biệt label categories. `scripts/visualization/visualize_evaluation_results.py`
đang vẽ timeline, không sở hữu histogram raw point/window.

**Change**

Tạo plotting entrypoint đọc raw `.npz` và threshold artifact cùng run. Với mỗi
machine, vẽ hai panel: point-level raw MSE và window-level raw MSE. Mỗi panel
tách `normal` và `anomalous`, vẽ threshold tương ứng, và ghi số lượng mẫu.

**Inputs**

- Raw `.npz` fields từ Stage 3.4.
- Raw threshold artifact schema 5.
- `point_labels`, `window_labels`, `point_predictions`, `window_predictions`.

**Outputs**

- One PNG per entity/run.
- One JSON summary per entity/run with finite counts, min, median, q95, q99,
  max, normal count, anomalous count, threshold and above-threshold counts.

**Atomic steps**

- [ ] Tạo script `scripts/visualization/plot_raw_input_mse_histograms.py`.
- [ ] Thêm CLI argument cho score artifact path.
- [ ] Thêm CLI argument cho threshold artifact path.
- [ ] Thêm CLI argument cho output directory.
- [ ] Đọc và validate raw artifact trước khi plot.
- [ ] Đọc raw point/window scores từ `.npz`.
- [ ] Đọc point/window labels từ `.npz`.
- [ ] Tạo normal mask `label == 0` cho point và window.
- [ ] Tạo anomalous mask `label == 1` cho point và window.
- [ ] Reject array shape mismatch trước khi gọi Matplotlib.
- [ ] Reject non-finite raw values trước khi plot.
- [ ] Vẽ point-level normal và anomalous histograms bằng shared bins.
- [ ] Vẽ window-level normal và anomalous histograms bằng shared bins.
- [ ] Vẽ point threshold trên point panel.
- [ ] Vẽ window threshold trên window panel.
- [ ] Ghi rõ đơn vị `raw input MSE` trên axes.
- [ ] Ghi entity, score space và transform trong figure metadata/title.
- [ ] Ghi category counts vào JSON summary.
- [ ] Không đọc hoặc transform calibrated sigmoid values.
- [ ] Thêm test histogram loader với toy NPZ.

### Stage 6.5 — Final audit

**Atomic steps**

- [ ] Kiểm tra mọi raw threshold artifact có `score_space: raw_input`.
- [ ] Kiểm tra mọi raw threshold artifact có `point_score_transform: identity`.
- [ ] Kiểm tra point/window arrays đều finite.
- [ ] Kiểm tra point score shape khớp point label shape.
- [ ] Kiểm tra window score shape khớp window label shape.
- [ ] Kiểm tra normal/anomalous masks không overlap và phủ đủ labeled observations.
- [ ] Kiểm tra threshold source chỉ là `clean_validation`.
- [ ] Kiểm tra synthetic injection xảy ra trước scoring.
- [ ] Kiểm tra smoke đã pass trước ba machine run.
- [ ] Kiểm tra histogram có bốn category groups: normal point, anomalous point, normal window, anomalous window.
- [ ] Kiểm tra report ghi riêng threshold point và threshold window.
- [ ] Kiểm tra không có raw protocol run nào load sigmoid parameters.
- [ ] Kiểm tra historical sigmoid files không đổi bằng `git status` và file hashes.
- [ ] Ghi kết luận limitation rằng raw MSE giữ equal feature weights và có thể bị channel magnitude chi phối.

### Tests and verification

- **Automated:** Chạy focused suite rồi chạy full test command hiện có:
  `.venv/bin/python -m pytest -q`. Kết quả phải ghi rõ exit code và số failure.
- **Automated:** Dùng `.venv/bin/python -m py_compile` cho scorer, evaluator,
  benchmark scripts và histogram script.
- **Manual:** Inspect histogram của cả ba machine và xác nhận raw MSE/threshold
  annotation, không phải sigmoid value.
- **Manual:** Đọc final report và xác nhận bốn category label được phân biệt rõ.
- **Risk:** Raw sensor units khác nhau giữa machine. Mitigation là threshold
  per-entity và provenance đầy đủ; không lập ranking raw threshold cross-entity.

### Complete when

Smoke pass, ba machine synthetic validation pass, raw point/window histogram
được tạo, bốn label categories được tách, raw artifact identity được audit và
không có raw execution nào dùng calibrated sigmoid. Hiện còn thiếu Stage B
THESIS checkpoint cho `machine-3-4` và `machine-3-9` trong checkout này.

## Interface and data changes

### Scoring interface

The proposed `score_reconstruction` interface accepts scaled tensors and one
fitted `SequenceStandardScaler`. It returns raw and normalized point/window MSE.
It does not mutate the model, scaler or input tensors.

### Evaluator interface

Raw protocol calls `Evaluator.evaluate` with `score_space="raw_input"` and a
fitted scaler. A missing scaler is an error. Existing callers that do not select
raw protocol keep their current model-output compatibility path until migrated
explicitly.

### Artifact interface

Schema 5 records raw identity and raw point/window definitions. Schema 3 and 4
remain readable for historical replay. Raw runtime never treats a schema 3/4
artifact as a raw artifact.

### Score arrays

Point and window score arrays carry separate labels and predictions. A window
label is derived from point labels by any-anomaly rule. A prediction is derived
from threshold comparison. The two concepts must not share one field.

## Deployment and rollout

1. Land contract and tests before changing operational selection.
2. Land scorer and verify toy arithmetic.
3. Land offline raw path and run one smoke combination.
4. Land schema 5 and generate new raw threshold artifacts in new output paths.
5. Land online raw path and run online smoke with schema 5 artifact.
6. Run the three requested machines sequentially.
7. Keep old sigmoid config/artifact paths available for historical replay.

If raw validation fails, select the previous protocol/artifact explicitly for
historical work. Do not overwrite old score or threshold files.

## Documentation changes

- Create `documents/spec/full-spec-v4.md`.
- Update `documents/spec/online_tta_terminology_ontology.md`.
- Update `documents/spec/offline_pretraining_terminology_ontology.md`.
- Record raw score identity, formulas, threshold split, checkpoint scaler
  provenance and four label categories in the dated experiment log.
- Record all histogram paths and entity-specific threshold values in the final
  report.

## Final verification

- [x] New threshold artifact declares `score_space: raw_input`.
- [x] New threshold artifact declares `point_score_transform: identity`.
- [x] Offline point/window thresholding uses raw-input MSE.
- [x] Online EWMA, triage and prediction use raw-input MSE.
- [x] Normalized MSE remains diagnostic only.
- [x] No raw runtime loads calibrated sigmoid parameters.
- [ ] Synthetic validation covers `machine-1-6`, `machine-3-4`, and `machine-3-9`.
- [x] Point/window arrays and labels are stored separately.
- [x] Histograms show normal/anomalous point and window groups with thresholds.
- [x] Historical sigmoid artifacts remain unchanged and are not silently reused.

## Assumptions and non-blocking uncertainties

- “Raw input space” means original sensor-value units before standardization.
- Equal-feature raw MSE is intentional; no channel weighting is introduced.
- Window anomalous label means at least one anomalous point in the window, matching
  `point_labels_to_window_labels`.
- Schema 5 is the next version only if a read-only check confirms no later schema
  constant exists before implementation.
- The current model output remains available for compatibility; only raw protocol
  boundaries change operational selection.
- The existing `scripts/benchmarks/_internal/run_thesis_offline_benchmark_helpers.py`
  is not an operational owner for score selection; update it only if an import or
  test proves that it is used by the selected runner.

## Execution record — 2026-09-04

- Implemented raw-input MSE scorer, scaler inverse transform, offline evaluator,
  schema 5 threshold artifact, online EWMA/triage/prediction path, runtime scaler
  reload, and histogram export.
- Focused regression: `73 passed`.
- Full regression: `528 passed, 3 failed, 1 skipped`. The three failures are
  pre-existing model snapshot/API and test-fixture failures; no raw-MSE test
  failed.
- Offline smoke passed for `machine-1-6` with the debug CPU checkpoint. It wrote
  raw threshold, score arrays, labels, predictions and histogram files.
- Online A0 smoke passed for `machine-1-6` over 2 causal windows. Records contain
  `score_space=raw_input` and `point_score_transform=identity`.
- The smoke synthetic subset contains only normal labels (`80` points, `4`
  windows), so it does not demonstrate anomalous histogram groups.
- Requested validation for `machine-3-4` and `machine-3-9` is blocked because
  this checkout has no valid THESIS Stage B checkpoint for either entity.
