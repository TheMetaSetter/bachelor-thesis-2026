---
date: 2026-08-09T00:00:00+07:00
researcher: OpenAI Codex
topic: "Xác định code cần sửa để thực hiện thiết kế anomaly score trong full-spec-v3"
status: complete
revision: b004e70b26b956809695c1b9d9518adf900ed2e9
branch: dev
---

# Research: Xác định code cần sửa để thực hiện thiết kế anomaly score trong full-spec-v3

## Summary

Runtime THESIS hiện vẫn dùng raw point MSE làm `point_scores`. Chưa có code nào thực hiện **shifted-and-scaled logistic sigmoid**, chưa ước lượng `c` và `tau` từ clean validation, và threshold artifact chưa lưu hai tham số này.

EWMA đã tồn tại và đang dùng đúng luồng theo absolute index. Vì vậy, code cần sửa chủ yếu nằm ở hai điểm: biến raw point MSE thành anomaly score trước khi tạo timeline; và truyền cùng `c`, `tau` từ calibration sang offline evaluator, online runtime và threshold artifact. Hàm EWMA không cần đổi công thức.

## Research question

Sử dụng `prompts/1_research_prompt.md` để xác định những đoạn code cần sửa để đáp ứng thiết kế anomaly score mới trong `documents/spec/full-spec-v3.md`.

Phạm vi là runtime THESIS v3, gồm offline benchmark, online TTA và threshold artifact. Các baseline khác không được xem là một phần của thiết kế THESIS v3 trong spec.

## System context

Spec v3 định nghĩa pipeline:

```text
raw point MSE e
  -> c = median(clean-validation e)
  -> tau = MAD(clean-validation e) / 0.6745
  -> shifted-and-scaled logistic sigmoid q
  -> offline q99 threshold hoặc online absolute-index EWMA(q) rồi q99 threshold
```

`full-spec-v3.md:545-604` định nghĩa raw point MSE, `c`, `tau` và `q`. `full-spec-v3.md:798-835` yêu cầu hai timeline threshold riêng biệt. `full-spec-v3.md:898-907` yêu cầu EWMA nhận `q`, không nhận raw MSE.

## Confirmed current execution path

### Offline model output

1. `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:182-227` tính MSE theo từng Monte Carlo sample, lấy mean thành `point_score_mean`, rồi xuất mean raw MSE ở top-level `point_scores`.
2. `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py:249-270` dùng raw MSE cho deterministic path và dùng output từ MC path cho top-level `point_scores`/`window_scores`.
3. `src/engine/evaluator.py:446-467` lấy `outputs["point_scores"]` và đưa trực tiếp vào overlap aggregation. `src/engine/evaluator.py:477-502` sau đó tính metric và threshold trên timeline này.
4. `scripts/benchmarks/run_thesis_offline_benchmark.py:434-452` tính clean threshold bằng `select_clean_validation_point_threshold`, rồi dùng threshold đó cho synthetic validation và test.
5. `scripts/benchmarks/run_thesis_offline_benchmark.py:505-565` tạo threshold artifact. Cả offline threshold và online EWMA threshold hiện đều nhận score chưa qua sigmoid.

### Online calibration và online scoring

1. `src/models/online_impl/online_adaptation_helpers.py:77-86` tính raw point MSE và xuất nó ở `point_scores`.
2. `src/engine/online_tta/online_calibration.py:61-76` đọc `outputs["point_scores"]`; `:82-128` đưa chúng qua `update_window_point_ewma`; `:132-154` thu raw offline point scores. Vì vậy calibration hiện tính EWMA trên raw MSE.
3. `src/engine/online_tta/online_engine_run.py:102-173` load threshold artifact nhưng chỉ đưa `online_ewma_point` vào runtime context. `c`, `tau` và tên score transformation chưa được đọc hoặc kiểm tra.
4. `src/engine/online_tta/online_engine_window_metrics.py:104-123` trích score rồi mới gọi EWMA. Đây là seam phù hợp để bảo đảm EWMA nhận `q`.
5. `src/engine/online_tta/online_engine_window_metrics.py:148-159` hiện đọc `outputs["point_scores"]` trực tiếp. `src/engine/online_tta/online_engine_window_core.py:100-180` truyền các score đó vào triage, verification và event record.
6. `src/engine/online_tta/online_engine_step.py:36-74` có fallback tự forward model và lấy `outputs["point_scores"][0, -1]`; fallback này cũng phải dùng cùng transform nếu vẫn được giữ.

### Threshold artifact

`src/protocols/threshold_artifact.py:20-235` validate schema nhưng không yêu cầu score transformation, `c`, `tau`, estimator hoặc MAD normalizer. `:237-374` cũng không nhận và không ghi các field này. Vì online runtime dùng artifact đã load, đây là điểm bắt buộc phải mở rộng để offline và online dùng cùng calibration identity.

### EWMA

`src/engine/online_tta/point_ewma.py:8-34` chỉ kết hợp score hiện tại và score trước đó theo absolute index. Công thức này phù hợp spec và không cần đổi. Chỉ cần bảo đảm đầu vào của nó là `q`.

## Code segments that must change

| Mức | File và symbol | Vì sao phải sửa | Kết quả cần đạt |
| --- | --- | --- | --- |
| P0 | `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:182-227` và `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py:249-270` | Top-level `point_scores` hiện là raw MSE; MC sample tensor cũng đang được dùng trực tiếp như score | Giữ raw per-sample MSE để tính `e`; áp dụng transform sau khi có `e`; top-level `point_scores` phải là `q`; `aux.point_score_samples` vẫn là raw per-sample MSE theo spec |
| P0 | `src/models/online_impl/online_adaptation_helpers.py:77-86` | Online A0/A1/A2 cũng xuất raw MSE ở `point_scores` | Online scorer phải trả hoặc chuyển raw MSE thành cùng `q` trước khi EWMA; `window_scores` vẫn là raw window reconstruction MSE |
| P0 | Thêm helper dùng chung trong seam calibration/threshold, tương ứng owner `src/engine/thresholds/stochastic_calibration.py` được nêu ở `full-spec-v3.md:106-145` | Repository chưa có implementation của `c`, MAD-based robust scale và logistic transform | Có một implementation duy nhất cho `fit`: median, MAD/0.6745; `transform`: `1/(1+exp(-(e-c)/tau))`; và kiểm tra finite/scale hợp lệ |
| P0 | `src/engine/online_tta/online_calibration.py:61-154` | Calibration hiện thu raw score rồi EWMA raw score; offline collector cũng lấy raw `point_scores` | Thu raw clean-validation MSE trước; fit `c`, `tau` một lần theo entity; transform offline scores; transform point scores trước `update_window_point_ewma`; trả cả raw timeline cần audit và transformed timeline dùng threshold |
| P0 | `scripts/benchmarks/run_thesis_offline_benchmark.py:434-452` | Benchmark đang lấy threshold từ `clean_payload["point_scores"]` trước khi có score transformation | Đổi thứ tự thành: collect raw clean validation -> fit `c`,`tau` -> transform clean/synthetic/test score path -> lấy offline q99 từ transformed clean scores; không dùng synthetic/test để fit |
| P0 | `scripts/benchmarks/run_thesis_offline_benchmark.py:505-565` | `_build_thresholds` đang lấy offline raw scores và online EWMA raw scores | Tính `offline_point_threshold_nonoverlap = Q0.99(q_offline_clean)`; tính `online_point_threshold_ewma = Q0.99(EWMA(q_online_clean))`; ghi cùng `c`,`tau` vào artifact |
| P0 | `scripts/ops/recalibrate_thesis_threshold_artifacts_v4.py:226-365` | V4 recalibration hiện chỉ collect stride-1 online scores, copy offline threshold v3 ở `:319-322`, rồi calibrate online raw scores | Collect lại non-overlap raw clean scores; fit `c`,`tau`; recalibrate cả offline và online threshold sau transform; không copy offline threshold cũ; ghi transform metadata |
| P0 | `src/protocols/threshold_artifact.py:20-235,237-374` | Artifact chưa có calibration parameters và transform identity | Thêm required fields tương ứng `point_score_transform`, `point_score_c`, `point_score_tau`, `point_score_tau_estimator`, `point_score_mad_normalizer`; validate đúng tên, finite values và `tau > 0`; builder phải ghi các field vào artifact và provenance |
| P0 | `src/engine/online_tta/online_engine_run.py:75-119,150-173` | Runtime chỉ validate identity của entity/checkpoint/EWMA; không validate transform identity và không đưa `c`,`tau` vào scorer | Validate transform fields của artifact; nạp chúng vào online scoring context; reject artifact thiếu hoặc không khớp transform |
| P0 | `src/engine/online_tta/online_engine_window_metrics.py:104-159` | Đây là điểm ngay trước EWMA nhưng hiện đọc score raw | Apply cùng artifact `c`,`tau` trước `update_window_point_ewma`; record `window_point_scores` và `current_window_ewma_point_scores` phải là transformed anomaly scores |
| P0 | `src/engine/online_tta/online_engine_step.py:36-74` và `src/engine/online_tta/online_engine_window_core.py:100-180` | Fallback scoring, triage, verification và event record dùng trực tiếp `point_scores` | Bảo đảm mọi score dùng cho prediction, EWMA, triage/admission và record đều nhất quán là `q` hoặc EWMA(`q`); không transform lần hai |
| P1 | `src/engine/evaluator.py:446-502` | Evaluator tích lũy `point_scores` mà không biết raw hay transformed; default fallback còn tự chọn positive-support quantile | Official THESIS evaluation phải nhận transformed scores và threshold đã fit từ clean validation; fallback raw positive-support không được dùng cho official v3 path |
| P1 | `src/engine/thresholding.py:15-47` và `src/engine/trainer.py:482-516` | `select_point_score_threshold` dùng positive-support rule; trainer còn tạo threshold trong validation loop trước artifact calibration | Giữ helper cho legacy path nếu cần, nhưng loại khỏi official THESIS v3; nếu trainer vẫn báo provisional metrics thì phải ghi rõ đó không phải official anomaly threshold |
| P1 | `src/core/contracts.py:121-204` | Contract hiện chỉ kiểm tra rank/type, chưa mô tả hoặc kiểm tra semantic range của `point_scores` | Bổ sung contract cho `point_scores` là transformed anomaly score nếu áp dụng ở model output; giữ raw MSE và raw sample tensors dưới `aux`/diagnostic contract |

## Code that should not change for this decision

- `src/engine/online_tta/point_ewma.py:8-34`: giữ công thức EWMA và absolute-index state.
- `src/models/thesis_multitask_impl/thesis_multitask_loss_core_mixin.py:30-50,270-330`: reconstruction loss và `point_score_loss` đang tính trực tiếp từ reconstruction error; sigmoid là inference/calibration transform, không phải training loss.
- `window_scores`: vẫn là raw window reconstruction MSE để dùng cho input-window triage theo `full-spec-v3.md:929-936`.
- `src/models/baseline_impl/redlamp_baseline.py` và `src/baselines/`: không tự động sửa. Spec v3 chỉ khóa public THESIS model và THESIS online path; các baseline đang có calibration contract riêng.

## Tests that must be added or updated

| File | Evidence hiện tại | Test cần có |
| --- | --- | --- |
| `tests/models/test_multitask_shapes.py:289-305` | Đang assert `point_scores == mean(point_score_samples)` | Assert `point_score_samples` là raw, `point_scores == sigmoid((mean(raw)-c)/tau)` khi calibration được cấp, và score nằm trong `(0,1)` |
| `tests/online/test_online_calibration_contract.py:56-128` | Expected values là raw score và raw EWMA | Test fit/apply `c`,`tau`, transformed offline timeline, và EWMA trên transformed score |
| `tests/online/test_online_ewma_threshold.py:9-18` | Chỉ kiểm tra toán EWMA trên số đầu vào | Giữ test toán EWMA; thêm test caller truyền `q`, không truyền raw MSE |
| `tests/online/test_threshold_artifact.py` và `tests/engine/test_threshold_artifact.py` | Artifact round-trip chưa có transform fields | Test round-trip, missing field, `tau <= 0`, sai transform name và mismatch identity |
| `tests/ops/test_recalibrate_thesis_threshold_artifacts_v4.py:57-97` | Chỉ kiểm tra online q và triage thresholds | Kiểm tra offline threshold được recalibrate từ transformed clean scores, không copy threshold v3; kiểm tra artifact lưu `c`,`tau` |
| `tests/evaluation/test_point_score_contracts.py` | Chỉ kiểm tra overlap aggregation | Thêm unit test MAD-based fit, sigmoid transform, q99 và strict `score > threshold` |
| `tests/online/test_online_entrypoint.py` và các test runtime online | Mock artifact hiện chỉ có threshold value | Cập nhật fixture để có transform metadata và kiểm tra runtime truyền cùng calibration params vào scoring path |

## Configuration observed

| Setting | Active value | Evidence | Scope |
| --- | --- | --- | --- |
| Offline threshold quantile | `0.99` | `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml` | Offline THESIS calibration |
| Online threshold quantile | `0.99` | `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml` | Online EWMA calibration |
| EWMA current weight | `0.9` | `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml` | Online point timeline |
| EWMA previous weight | `0.1` | `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml` | Online point timeline |
| Transform center/scale | Chưa có config field; phải derive từ clean validation và persist vào artifact | `src/protocols/threshold_artifact.py:20-374` không có field | Entity-scoped calibration |

## Conflicts and uncertainties

1. Spec yêu cầu top-level `point_scores` là `q`, nhưng current model forward không nhận `c`,`tau`. Cần chọn một seam implementation rõ ràng: truyền calibration params vào model/scorer, hoặc postprocess ngay sau model forward trong evaluator/online scorer. Evidence hiện tại chưa khóa API cụ thể này.
2. Spec không nêu cách xử lý `MAD = 0`, nhưng sigmoid cần `tau` dương. Code implementation phải có fail-fast hoặc epsilon policy; đây là open implementation value, chưa được tự suy đoán trong research này.
3. `scripts/benchmarks/_internal/run_thesis_offline_benchmark_helpers.py:357-404` có `_build_thresholds` khác với `_build_thresholds` active ở `scripts/benchmarks/run_thesis_offline_benchmark.py:505-565`. Top-level script chỉ import `_export_offline_retention_bundle` từ internal helper ở `:662`; do đó internal `_build_thresholds` không được xác nhận là active owner của benchmark path này.
4. `scripts/benchmarks/run_thesis_offline_benchmark.py:501` chọn entity bằng `_first_entity_id`. Các config hiện hành có vẻ chạy một entity mỗi artifact, nhưng code hiện tại chưa tự chứng minh rằng một loader không thể chứa nhiều entity. Nếu loader chứa nhiều entity, `c`,`tau` và artifact phải được tách theo entity đúng spec.
5. `src/engine/trainer.py:504` và evaluator fallback vẫn có threshold logic raw/legacy. Available evidence chưa cho thấy chúng được dùng để tạo official THESIS v3 threshold artifact, nhưng chúng có thể làm metric trong quá trình training khác với metric sau calibration.

## Open questions

- Khi `MAD(clean_validation) = 0`, project sẽ dùng epsilon cố định hay fail-fast?
- `point_scores` sẽ được transform trong model output contract hay ở một shared scoring adapter sau `forward`? Đây là quyết định API, không thể xác nhận chỉ từ code hiện tại.
- Có cần lưu raw point MSE timeline riêng trong score artifact hay chỉ lưu `c`,`tau` và transformed score? Spec yêu cầu raw values cho một số diagnostics nhưng retention policy hiện có thể loại bỏ sample tensors.

## Evidence

- `documents/spec/full-spec-v3.md:545-604` — định nghĩa raw point MSE, median center, MAD-based robust scale và shifted-and-scaled logistic sigmoid.
- `documents/spec/full-spec-v3.md:798-835` — quy định clean-validation calibration, cùng `c`,`tau` cho offline/online, và hai q99 thresholds riêng.
- `documents/spec/full-spec-v3.md:898-907` — quy định EWMA phải chạy trên transformed point anomaly score.
- `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:182-227` — hiện tính MC raw MSE và xuất mean raw MSE ở `point_scores`.
- `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py:249-270` — hiện nối deterministic/MC raw scores vào public model output.
- `src/models/online_impl/online_adaptation_helpers.py:77-86` — online adapter hiện xuất raw MSE.
- `src/engine/online_tta/online_calibration.py:61-154` — online calibration hiện thu raw scores và EWMA raw scores.
- `scripts/benchmarks/run_thesis_offline_benchmark.py:434-452,505-565` — active offline benchmark hiện threshold hóa raw scores và tạo artifact không có transform metadata.
- `scripts/ops/recalibrate_thesis_threshold_artifacts_v4.py:226-365` — v4 recalibration hiện copy offline threshold cũ và chỉ recalibrate online raw timeline.
- `src/protocols/threshold_artifact.py:20-374` — artifact schema hiện chưa yêu cầu `c`, `tau` hoặc transform identity.
- `src/engine/online_tta/online_engine_window_metrics.py:104-159` — seam hiện tại giữa model score extraction và EWMA.
- `src/engine/online_tta/online_engine_run.py:75-173` — online runtime hiện load artifact nhưng chưa nạp score-transform parameters.

## Validation performed

- Đã đọc `prompts/1_research_prompt.md` trước khi tìm kiếm rộng.
- Đã kiểm tra branch/revision: `dev` tại `b004e70b26b956809695c1b9d9518adf900ed2e9`.
- Chỉ ghi research log này; chưa sửa source code, config, test hoặc threshold artifact.
- Chưa chạy test vì yêu cầu hiện tại là xác định phạm vi code cần sửa, không phải implementation.
