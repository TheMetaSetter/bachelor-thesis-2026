---
date: 2026-08-04 18:15:00 +07:00
topic: "Atomic implementation steps for the desired online runtime flow"
status: ready
revision: 4be64456d6aa652457a0702154bae0d9b742a803
source_structure: documents/logs/2026-08-04/structure/structure-online-runtime-desired-flow.md
related_documents:
  - documents/logs/2026-08-04/plan/plan-online-runtime-desired-flow.md
  - documents/logs/2026-08-03/research/research-online-runtime-desired-flow-change-surface.md
  - documents/spec/online_tta_terminology_ontology.md
  - prompts/4_detail_prompt.md
---

# Detailed Implementation: Desired online runtime flow

## Summary

Tài liệu này chia structure thành các atomic step. Mỗi step nói rõ người thực
hiện phải kiểm tra hoặc thay đổi gì, ở file và symbol nào, và phải kiểm tra kết
quả ra sao. Tài liệu chưa sửa source code.

Luồng đích là:

```text
load stage_b_best_checkpoint and threshold_artifact
  -> receive causal_window with absolute_indices
  -> create source_hidden or projected_hidden
  -> create window_point_scores
  -> update current_window_ewma_point_scores
  -> create window_point_predictions
  -> classify triage_region
  -> run the current online action
  -> run verification_cycle for buffered verification_entry objects
  -> emit online_event_record and the live UI event
  -> save online_runtime_state
```

Các tên trong backtick là tên canonical hoặc tên source hiện có. Scalar endpoint
hiện tại như `raw_point_score`, `previous_ewma_score`, `ewma_point_score` và
`prediction` không được gọi là các vector canonical tương ứng.

## Source structure

Tài liệu dùng trực tiếp structure do anh chỉ định:
`documents/logs/2026-08-04/structure/structure-online-runtime-desired-flow.md`.
Structure có bảy phase và các phase giữ nguyên thứ tự. Mỗi phase trong tài liệu
này có các stage tương ứng, sau đó được chia thành atomic step. Plan nguồn là
`documents/logs/2026-08-04/plan/plan-online-runtime-desired-flow.md`.

Structure có `status: proposed`, nhưng yêu cầu trực tiếp của anh đã chọn file
này làm đầu vào cho detail. Vì vậy tài liệu này không tự đổi outcome hoặc thứ tự
phase của structure.

## Current state

Entry point thực tế đi theo chuỗi:

```text
python -m scripts.run_thesis_online_benchmark
  -> scripts.benchmarks.run_thesis_online_benchmark.main
  -> run_thesis_online_benchmark
  -> run_thesis_online_tta_experiment
  -> _run_online_sequence
  -> _process_online_window
```

`_build_runtime_online_context` hiện gọi
`calibrate_entity_threshold_artifacts` khi online bắt đầu. `_run_online_sequence`
giữ một `previous_ewma_score` scalar. `_process_online_window` hiện gọi theo
thứ tự `prepare_event -> buffer_and_verification -> adaptation_step ->
build_outputs`. `_prepare_online_window_event` tạo preliminary `pnn_mask` trước
khi phân loại `triage_region`. Các facts này được ghi trong research report và
được kiểm tra lại ở source tại:

- `src/engine/online_tta/online_engine_run.py:129-315`;
- `src/engine/online_tta/online_engine_window_core.py:53-324`;
- `src/engine/online_tta/online_engine_window_metrics.py:33-273`;
- `src/engine/online_tta/runtime_state.py:13-270`.

## Desired end state

Sau khi hoàn thành, runtime phải có các tính chất sau:

1. Online đọc `threshold_artifact` và `stage_b_best_checkpoint` của cùng offline
   run, rồi dừng nếu identity hoặc schema không khớp.
2. `causal_window` mang `absolute_indices`. `window_point_scores`,
   `current_window_ewma_point_scores` và `window_point_predictions` đều giữ đủ
   các point trong window.
3. Point mới dùng score hiện tại làm EWMA. Point còn xuất hiện trong sliding
   window được cập nhật lại. Không thêm prediction-finalisation state.
4. A0 dùng `source_hidden`, không tạo `online_mlp_projector`, không tạo
   optimizer và không đi vào adaptation hoặc verification.
5. A1/A2 phân loại `triage_region` trước khi tạo signature hoặc `pnn_mask`.
   Accepted `hard_old_normality` của A2 chạy trước `verification_cycle`.
6. `hard_old_interval_guard` chỉ quyết định có được update hay không. Guard bị
   từ chối không được đổi `triage_region` thành `gray_zone`.
7. `recurrent_signature_set` chỉ sống trong một `verification_cycle`.
8. `VerificationResult` và `was_adapted` vẫn giữ nguyên vai trò hiện có; không
   thêm `verification_outcome` chung.
9. Record, retention, checkpoint và UI đều dùng vector contract. UI nhận event
   trực tiếp bằng callback read-only.
10. Chỉ đặt `runtime_protocol_status` thành `full_spec_v3` sau khi focused tests
    và một smoke path chạy thành công.

## Scope

### In scope

- Offline-to-online `threshold_artifact` và checkpoint handoff.
- A0/A1/A2 model construction, optimizer và loss boundary.
- `absolute_indices`, vector score, EWMA và prediction state.
- Triage, hard-old guard, current action, `verification_cycle` và cycle-local
  `recurrent_signature_set`.
- Runtime state, checkpoint resume, event records, metrics, retention, demo và
  direct UI callback.
- Focused tests và một online smoke path.

### Out of scope

- Đổi công thức trong hard-old hoặc masked PNN loss.
- Đổi quy tắc bốn vùng của `classify_online_window`.
- Xóa `VerificationResult`, `was_adapted` hoặc `verification_buffer`.
- Chạy full benchmark matrix trước khi một smoke path cụ thể pass.
- Tự tạo tên config canonical cho đường dẫn `threshold_artifact` khi ontology
  chưa chốt tên đó.

## Evidence

- `documents/logs/2026-08-03/research/research-online-runtime-desired-flow-change-surface.md:16-30` — bốn nhóm thay đổi và hai quy tắc đã chốt.
- `documents/logs/2026-08-03/research/research-online-runtime-desired-flow-change-surface.md:70-88` — thứ tự runtime mong muốn.
- `documents/logs/2026-08-03/research/research-online-runtime-desired-flow-change-surface.md:131-149` — artifact handoff và lỗi thứ tự ghi checkpoint hash.
- `documents/logs/2026-08-03/research/research-online-runtime-desired-flow-change-surface.md:151-170` — A0 hiện vẫn tạo projector và optimizer.
- `documents/logs/2026-08-03/research/research-online-runtime-desired-flow-change-surface.md:172-199` — absolute index và vector score contract.
- `documents/logs/2026-08-03/research/research-online-runtime-desired-flow-change-surface.md:200-260` — prediction, triage, guard và verification order.
- `documents/logs/2026-08-03/research/research-online-runtime-desired-flow-change-surface.md:451-489` — sliding-window update, cycle-local signature và direct UI decision.
- `documents/spec/online_tta_terminology_ontology.md:18-35` — offline-first ownership và `threshold_artifact`.
- `documents/spec/online_tta_terminology_ontology.md:76-85` — meaning của A0, A1, A2.
- `documents/spec/online_tta_terminology_ontology.md:116-181` — artifact, causal-window, vector và endpoint contracts.
- `src/engine/online_tta/online_engine_run.py:129-220` — startup calibration hiện tại.
- `src/engine/online_tta/online_engine_window_core.py:53-118` — per-window call order hiện tại.
- `src/engine/online_tta/online_engine_window_metrics.py:82-147` — scalar endpoint extraction và scalar EWMA hiện tại.
- `src/engine/online_tta/online_engine_window_metrics.py:150-194` — preliminary signature history và PNN mask hiện tại.
- `src/engine/online_tta/runtime_state.py:13-112` — scalar runtime schema hiện tại.
- `scripts/benchmarks/run_thesis_offline_benchmark.py:499-556` — offline threshold artifact creation hiện tại.
- `scripts/benchmarks/run_thesis_online_benchmark.py:226-260` — checkpoint resolution và record normalization hiện tại.

## Phase 1: Establish the offline calibration handoff

### Stage 1.1: Canonical artifact contract

**Mục tiêu:** xác nhận những field online phải đọc từ `threshold_artifact` và
định danh nào phải khớp với `stage_b_best_checkpoint`.

#### Atomic steps

1. Mở `documents/spec/offline_pretraining_terminology_ontology.md` và
   `documents/spec/online_tta_terminology_ontology.md`. Ghi lại owner của
   `threshold_artifact`, `stage_b_best_checkpoint`, `offline_variant`,
   `entity_id`, `seed`, `window_size` và EWMA weights.
2. Đối chiếu các field canonical với schema hiện tại trong
   `src/protocols/threshold_artifact.py`, đặc biệt
   `validate_threshold_artifact`, `build_threshold_artifact` và
   `load_threshold_artifact`.
3. Xác nhận bốn giá trị phải có trong artifact: `online_point_ewma_threshold`,
   `input_window_threshold`, `latent_window_low_threshold` và
   `latent_window_high_threshold`. Không map `offline_point_threshold` thành
   `online_point_ewma_threshold`.
4. Xác nhận artifact lưu `checkpoint_sha256`, entity identity, `window_size`,
   `ewma_current_weight`, `ewma_previous_weight` và schema version.
5. Kiểm tra config contract hiện tại trong
   `src/core/config_model_validation.py` và
   `src/engine/online_tta/checkpoint_resolution.py` để xác định cách resolve
   artifact từ cùng offline run với checkpoint.
6. Nếu ontology đã có field path canonical, dùng đúng field đó. Nếu chưa có,
   dừng trước khi viết code resolver và yêu cầu cập nhật ontology; không tự đặt
   `threshold_artifact_path` hoặc tên tương tự thành config canonical.
7. Ghi kết quả mapping vào test expectation hoặc implementation note của Phase
   1. Không đổi tên runtime field chỉ vì tên serialization nested khác nhau.

#### Contract của stage

- **Input:** offline ontology, online ontology, current artifact schema và task
  metadata.
- **Output:** danh sách field canonical, identity checks và một quyết định rõ
  về đường dẫn artifact.
- **Lỗi:** thiếu field, tên không được ontology chấp thuận, hoặc artifact field
  có nghĩa mơ hồ phải làm run fail trước khi xử lý stream.
- **Compatibility:** giữ đọc artifact cũ chỉ khi schema hiện tại đã nói rõ cách
  tương thích; không tự suy diễn scalar thành vector.

#### Verification

- Unit/contract: mở rộng `tests/online/test_threshold_artifact.py` và
  `tests/engine/test_threshold_artifact.py` để kiểm tra bốn threshold và
  checkpoint identity.
- Command: `.venv/bin/python -m pytest -q tests/online/test_threshold_artifact.py tests/engine/test_threshold_artifact.py`.
- Kết quả mong đợi: artifact hợp lệ được load; artifact thiếu field hoặc hash
  sai bị reject.

### Stage 1.2: Offline artifact production

**Mục tiêu:** offline tạo artifact bằng cùng score/EWMA semantics mà online sẽ
dùng.

#### Atomic steps

1. Trong `scripts/benchmarks/run_thesis_offline_benchmark.py`, đọc
   `_build_thresholds` và `_export_offline_artifacts`. Ghi nhận score input mà
   mỗi threshold đang dùng.
2. Tách score path cho online calibration khỏi đường scalar endpoint cũ. Score
   path phải giữ `window_point_scores` của từng `causal_window` và ghép theo
   `absolute_indices`.
3. Gọi helper EWMA dùng chung cho offline và online. Helper phải áp dụng quy
   tắc: point chưa có state dùng score hiện tại; point đã có state dùng
   `ewma_previous_weight * previous + ewma_current_weight * current`.
4. Tạo `online_point_ewma_threshold` từ vector EWMA sau khi áp dụng đúng stride
   và overlap của online causal stream.
5. Tính và truyền riêng `input_window_threshold`,
   `latent_window_low_threshold` và `latent_window_high_threshold` vào
   `build_threshold_artifact`. Không dùng lại một threshold scalar cho bốn vai
   trò.
6. Tính SHA-256 của `stage_b_best_checkpoint` trước khi gọi
   `write_threshold_artifact`. Đưa hash vào artifact trước khi serialize.
7. Giữ `variant_name` là identity của offline run. Không dùng field này để
   chứa `online_variant` A0/A1/A2.
8. Giữ đường dẫn artifact trong report và retention manifest của offline run để
   online resolver có thể kiểm tra provenance.

#### Contract của stage

- **Input:** clean validation score artifacts, offline protocol config, Stage B
  checkpoint path và window metadata.
- **Output:** một `threshold_artifact` có bốn threshold canonical và checkpoint
  provenance.
- **Lỗi:** score thiếu absolute index, EWMA weights không hợp lệ, checkpoint
  không tồn tại hoặc threshold low lớn hơn threshold high.
- **Compatibility:** artifact cũ không bị overwrite; run cũ chỉ được dùng qua
  legacy path được ghi rõ.

#### Verification

- Unit: thêm case cho first-seen point và overlapping point vào test threshold
  artifact hoặc test point-score helper.
- Integration: mở rộng `tests/benchmarks/test_thesis_offline_artifact_exports.py`
  để kiểm tra artifact được ghi sau khi checkpoint hash đã có.
- Command: `.venv/bin/python -m pytest -q tests/benchmarks/test_thesis_offline_artifact_exports.py tests/online/test_online_ewma_threshold.py`.
- Kết quả mong đợi: artifact chứa bốn threshold, EWMA weights và hash đúng.

### Stage 1.3: Online resolution gate

**Mục tiêu:** online resolve artifact và checkpoint từ cùng offline identity rồi
reject mismatch trước khi runtime state đổi.

#### Atomic steps

1. Trong `scripts/benchmarks/run_thesis_online_benchmark.py`, giữ metadata
   `offline_variant`, `entity_id`, `seed`, `benchmark_mode` và `stage_name` làm
   khóa resolve.
2. Trong `src/engine/online_tta/checkpoint_resolution.py`, dùng
   `resolve_stage_b_checkpoint` để lấy `stage_b_best_checkpoint`; giữ nguyên
   lỗi khi có zero hoặc nhiều candidate.
3. Thêm bước resolve `threshold_artifact` từ cùng offline run. Bước này chỉ
   dùng field path đã được Stage 1.1 chấp thuận hoặc quy tắc directory đã được
   ontology ghi rõ.
4. Gọi `load_threshold_artifact` và `validate_threshold_artifact` trước khi
   build `OnlineRuntimeState`, `VerificationBuffer` hoặc mutable model state.
5. So sánh artifact với checkpoint và request: entity, offline variant, seed,
   window size, EWMA weights, schema version và checkpoint SHA-256.
6. Nếu bất kỳ identity nào lệch, raise một lỗi chỉ rõ field lệch. Không fallback
   sang calibration từ clean validation.
7. Trả cả artifact path và artifact object trong startup context để các phase sau
   không cần tự đọc lại file.

#### Contract của stage

- **Input:** online experiment config, protocol config, offline metadata.
- **Output:** verified `stage_b_best_checkpoint` và `threshold_artifact`.
- **Lỗi:** file thiếu, candidate mơ hồ, schema không hợp lệ hoặc identity mismatch.
- **Compatibility:** legacy artifact chỉ vào nhánh tương thích được gắn nhãn;
  không biến scalar artifact thành vector artifact.

#### Verification

- Extend `tests/online/test_online_entrypoint.py` với cases artifact hợp lệ,
  artifact thiếu và hash mismatch.
- Command: `.venv/bin/python -m pytest -q tests/online/test_online_entrypoint.py tests/online/test_online_reference_checkpoint.py`.
- Kết quả mong đợi: lỗi xảy ra trước lần gọi `_run_online_sequence` đầu tiên.

### Stage 1.4: Startup handoff

**Mục tiêu:** THESIS online chỉ đọc artifact, không recalibrate threshold.

#### Atomic steps

1. Trong `src/engine/online_tta/online_engine_run.py`, sửa
   `_build_runtime_online_context` để nhận artifact đã resolve.
2. Đưa `load_threshold_artifact` vào trước `build_online_runtime_state` và
   trước khi tạo mutable verification state.
3. Xóa `calibrate_entity_threshold_artifacts` khỏi normal THESIS startup path.
   Giữ helper calibration chỉ cho offline hoặc legacy path có owner rõ ràng.
4. Xóa `_persist_threshold_artifacts` khỏi normal online startup; online không
   được ghi artifact mới từ test stream.
5. Lưu artifact identity và path trong context, completion report và runtime
   state.
6. Bảo đảm `dry_run` vẫn không đọc hoặc ghi artifact thật nếu context hiện tại
   chỉ dùng để kiểm tra config.

#### Verification

- Extend `tests/online/test_online_engine_max_steps.py` để spy vào
  `calibrate_entity_threshold_artifacts` và xác nhận nó không được gọi trong
  normal path.
- Command: `.venv/bin/python -m pytest -q tests/online/test_online_entrypoint.py tests/online/test_online_engine_max_steps.py`.
- Manual: một A2 smoke report phải ghi cùng entity, window size, EWMA weights và
  checkpoint hash với offline artifact.

#### Complete when

- Startup chỉ tạo stream state sau khi artifact và checkpoint đã pass identity
  checks.
- Mismatch dừng run trước window đầu tiên.

## Phase 2: Separate inference-only and adaptation variants

### Stage 2.1: Variant construction boundary

**Mục tiêu:** model biết `online_variant` trước khi tạo module mutable.

#### Atomic steps

1. Trong `src/models/online_impl/online_adaptation.py`, xác định nơi
   `OnlineAdaptationModel.__init__` nhận config và luôn tạo
   `online_mlp_projector`.
2. Truyền `online_variant` hoặc một construction flag có nghĩa tương đương từ
   `_build_runtime_online_context`; không đọc variant ngầm từ tên experiment.
3. Validate variant chỉ là A0, A1 hoặc A2 ở một boundary duy nhất.
4. Với A0, bỏ hoàn toàn module projector, alias `projector`, anchor state và
   projector parameter group.
5. Với A1/A2, giữ projector path hiện tại và không đổi formula trong model.
6. Cập nhật các accessor như `parameters_for_online_update` để A0 trả về
   empty/none theo contract đã chọn, thay vì truy cập thuộc tính không tồn tại.

#### Contract của stage

- **Input:** `online_variant`, offline checkpoint và model config.
- **Output:** A0 chỉ có `frozen_source_model`/`source_hidden`; A1/A2 có
  `online_mlp_projector`.
- **Lỗi:** variant không hợp lệ hoặc caller yêu cầu projector trên A0 phải fail
  rõ ràng.
- **Compatibility:** không đổi inherited offline model parameters.

#### Verification

- Extend `tests/online/test_online_tta_trainable_surface.py` và
  `tests/online/test_online_tta_variants.py`.
- Command: `.venv/bin/python -m pytest -q tests/online/test_online_tta_trainable_surface.py tests/online/test_online_tta_variants.py`.
- Kết quả mong đợi: `hasattr(model_a0, "online_mlp_projector")` là false và
  A1/A2 vẫn có projector.

### Stage 2.2: A0 inference path

**Mục tiêu:** A0 kết thúc sau prediction state, không chạy adaptation.

#### Atomic steps

1. Trong `_build_runtime_online_context`, chỉ build optimizer nếu variant là
   A1 hoặc A2.
2. Trong `_forward_online_window`, giữ `forward_source` cho A0 để lấy
   `source_hidden` mà không gọi projector.
3. Trong `_process_online_window`, chờ đến khi có
   `window_point_predictions` và đã cập nhật point state rồi return event A0.
4. Không gọi `_classify_event_window`, `_update_online_window_buffers`,
   `verification_cycle` hoặc `execute_online_tta_step` cho A0.
5. Record A0 phải có `did_update=False` rõ ràng và không tạo
   `verification_entry`.

#### Verification

- Add spy test vào proposed `tests/online/test_online_window_flow.py` để đếm
  các lời gọi triage, optimizer, verification và update của A0.
- Command: `.venv/bin/python -m pytest -q tests/online/test_online_tta_variants.py tests/online/test_online_tta_trainable_surface.py`.
- Kết quả mong đợi: A0 vẫn tạo score/prediction vector nhưng không mutate model.

### Stage 2.3: A1/A2 adaptation path

**Mục tiêu:** A1/A2 chỉ update trong các branch được ontology cho phép.

#### Atomic steps

1. Giữ `online_mlp_projector` là mutable module duy nhất cho A1/A2.
2. Tạo optimizer một lần ở runtime context hoặc theo owner hiện tại đã được
   kiểm tra; không tạo optimizer cho A0 hoặc action bị skip.
3. Để Phase 4 quyết định khi nào A2 hard-old hoặc PNN update được gọi.
4. Đảm bảo verification update dùng non-empty `pnn_mask` và không dùng
   `triage_region="pnn_verified"` như một canonical triage region.
5. Giữ `VerificationResult` per-entry và `was_adapted` bookkeeping hiện có.

#### Verification

- Test A1 với empty mask phải không update.
- Test A2 với accepted hard-old và verified PNN phải update đúng branch.
- Command: `.venv/bin/python -m pytest -q tests/online/test_online_tta_variants.py tests/online/test_online_verification_buffer.py`.

### Stage 2.4: Loss configuration path

**Mục tiêu:** tách threshold và đọc `lambda_online_contrastive` từ model config.

#### Atomic steps

1. Thêm `lambda_online_contrastive` vào
   `configs/model/online_adaptation.yaml` và các generated online experiment
   configs qua `scripts/benchmarks/generate_online_benchmark_configs.py`.
2. Thêm field vào allow-list và numeric validation trong
   `src/core/config_model_validation.py`; reject giá trị âm.
3. Truyền field qua `OnlineAdaptationModel.__init__` hoặc model config owner.
4. Trong `src/engine/online_tta/online_engine_step.py`, nhận
   `input_window_threshold` riêng với `online_point_ewma_threshold`.
5. Dùng `input_window_threshold` cho `hard_old_reconstruction_loss`.
6. Bỏ literal `0.1` trong A2 contrastive branch và dùng configured weight.
7. Không đổi formula của hard-old hoặc masked PNN loss.

#### Verification

- Extend `tests/online/test_full_spec_online_losses.py` với hai threshold khác
  nhau và lambda khác `0.1`.
- Command: `.venv/bin/python -m pytest -q tests/online/test_full_spec_online_losses.py tests/online/test_online_tta_variants.py tests/online/test_online_benchmark_config_generation.py`.
- Kết quả mong đợi: mỗi branch dùng đúng parameter owner.

#### Complete when

- A0 không có projector/optimizer/update.
- A1/A2 có projector đúng lúc và không update khi không có điều kiện.
- Threshold và loss weight không còn bị trộn hoặc hard-code.

## Phase 3: Make point identity and vector state explicit

### Stage 3.1: Absolute point identity

**Mục tiêu:** mỗi point có identity ổn định qua các window chồng lấn.

#### Atomic steps

1. Trong `src/data/stream.py`, sửa `SMDOnlineStream._build_window` để tạo
   `absolute_indices: LongTensor[L]` từ `start_index` đến `end_index - 1`.
2. Giữ `meta.start_index`, `meta.end_index` và `meta.stream_step` để tương thích
   và debug.
3. Trong `src/data/collate.py`, sửa `collate_windows` để ghép thành
   `absolute_indices [B,L]`.
4. Trong `src/core/contracts.py`, cập nhật `validate_window`, `validate_batch`
   và `validate_online_batch` để kiểm tra dtype, rank, length và strictly
   increasing order.
5. Trong `src/engine/online_tta/verification_adapter.py`, sửa
   `build_entry_batch` để dựng lại cùng absolute interval từ
   `verification_entry`.
6. Kiểm tra entry batch bằng contract trước khi gọi verification model.

#### Contract của stage

- **Input:** causal stream window và buffered interval.
- **Output:** `causal_window.absolute_indices` có shape `[B,L]`.
- **Lỗi:** index thiếu, sai dtype, sai length hoặc không tăng phải reject.
- **Compatibility:** metadata interval vẫn được giữ; state mới dùng tensor
  absolute index làm identity chính.

#### Verification

- Extend `tests/online/test_online_stream.py` với expected indices cho window
  đầu và window overlap.
- Add contract cases cho rank/dtype/order vào `tests/core/test_contracts.py`.
- Command: `.venv/bin/python -m pytest -q tests/online/test_online_stream.py tests/core/test_contracts.py tests/online/test_online_verification_buffer.py`.

### Stage 3.2: Shared score representation

**Mục tiêu:** giữ toàn bộ `window_point_scores [L]`, không chỉ endpoint.

#### Atomic steps

1. Trong `src/engine/online_tta/online_engine_window_metrics.py`, sửa
   `_extract_online_window_scores` để lấy toàn bộ `outputs["point_scores"]`
   theo chiều point.
2. Xác nhận shape sau batch squeeze là `[L]`, không flatten nhầm feature hoặc
   batch dimension.
3. Giữ `input_window_score` và `latent_window_score` là scalar window metrics;
   không dùng chúng thay cho `window_point_scores`.
4. Trong `_score_online_window`, trả vector score và scoring outputs cùng với
   batch trên device.
5. Cập nhật callers để không còn gán endpoint scalar vào tên vector canonical.

#### Verification

- Unit test output có L khác 1 phải trả đúng L score.
- Command: `.venv/bin/python -m pytest -q tests/online/test_full_spec_online_contract.py tests/online/test_online_ewma_threshold.py`.
- Kết quả mong đợi: endpoint scalar chỉ còn ở compatibility record nếu cần.

### Stage 3.3: Per-point EWMA update

**Mục tiêu:** merge score theo absolute index và áp dụng first-seen rule.

#### Atomic steps

1. Mở `src/protocols/point_scores.py` và giữ helper scalar hiện tại cho caller
   cũ nếu test contract vẫn yêu cầu.
2. Thêm một helper mapping-based nhỏ cho
   `window_point_scores`, `absolute_indices`, previous per-point state và EWMA
   weights. Đây là proposed helper symbol; chọn tên rõ nghĩa trước khi code,
   không tạo abstraction lớn hơn cần thiết.
3. Với từng absolute index trong current window, nếu state chưa có thì đặt
   `current_window_ewma_point_scores[t] = window_point_scores[t]`.
4. Nếu state đã có, tính
   `ewma_previous_weight * previous + ewma_current_weight * current`.
5. Trả current window vector và updated mapping. Không khởi tạo previous value
   bằng zero.
6. Chỉ giữ state cần cho point còn active và latest output cần cho record/UI;
   không tạo finalization table.
7. Gọi cùng helper từ offline calibration path để threshold và online inference
   dùng cùng semantics.

#### Contract của stage

- **Input:** absolute indices `[L]`, scores `[L]`, previous mapping và weights.
- **Output:** `current_window_ewma_point_scores [L]` và mapping mới.
- **Lỗi:** shape mismatch, duplicate index trong một window hoặc weight không
  hợp lệ phải fail.
- **Compatibility:** không map `previous_ewma_score` scalar thành state của mọi
  point.

#### Verification

- Unit cases: first window, overlap, gap và point xuất hiện lại.
- Command: `.venv/bin/python -m pytest -q tests/online/test_online_ewma_threshold.py`.
- Kết quả mong đợi: first-seen score giữ nguyên; overlap dùng cả previous và
  current weight.

### Stage 3.4: Prediction state

**Mục tiêu:** tạo `window_point_predictions [L]` trước adaptation.

#### Atomic steps

1. Chọn threshold `online_point_ewma_threshold` từ verified artifact.
2. Với từng phần tử của `current_window_ewma_point_scores`, áp dụng cùng binary
   threshold rule hiện có.
3. Đặt kết quả vào `window_point_predictions [L]`.
4. Cập nhật latest prediction map theo absolute index khi point xuất hiện trong
   current window.
5. Nếu point biến mất khỏi các window sau, giữ latest prediction map và không
   cập nhật lại nó.
6. Đưa vectors vào event trước khi model update, verification hoặc UI callback.

#### Verification

- Integration test với hai window overlap kiểm tra shared point đổi prediction
  khi EWMA đổi.
- Test point không còn trong window sau kiểm tra latest value vẫn đọc được.
- Command: `.venv/bin/python -m pytest -q tests/online/test_online_ewma_threshold.py tests/online/test_full_spec_online_contract.py`.

### Stage 3.5: Minimal end-to-end vector path

**Mục tiêu:** đưa vectors qua event boundary khi chưa có adaptation.

#### Atomic steps

1. Trong `_prepare_online_window_event`, gọi score extraction, EWMA helper và
   prediction helper theo thứ tự.
2. Đưa `window_point_scores`, `current_window_ewma_point_scores`,
   `window_point_predictions` và `absolute_indices` vào event.
3. Tạm thời chạy bằng A0 fixture để tách vector contract khỏi model mutation.
4. Kiểm tra `_finalize_window_result` trả vector state mà không làm mất metric
   scalar cần cho legacy report.
5. Chỉ sau khi vector event pass mới nối vào Phase 4 triage và update.

#### Verification

- Proposed integration test: `tests/online/test_online_window_flow.py`.
- Kết quả mong đợi: A0 stream tạo vector event cho first, overlap và departed
  point mà không gọi update.

#### Complete when

- Absolute index và ba vector canonical đã đi xuyên qua một window event.
- Không cần explicit prediction finalization.

## Phase 4: Enforce the desired per-window order

### Stage 4.1: Triage stage

**Mục tiêu:** phân loại `triage_region` trước signature, PNN hoặc verification.

#### Atomic steps

1. Trong `src/engine/online_tta/online_engine_window_core.py`, tách scoring/EWMA
   khỏi triage classification trong `_process_online_window`.
2. Gọi `classify_online_window` trong `_classify_event_window` bằng ba threshold
   canonical: input, latent low và latent high.
3. Không truyền `hard_old_interval_guard` vào classifier.
4. Giữ nguyên đúng bốn region: `normal`, `gray_zone`, `hard_old_normality` và
   `strong_anomaly` theo contract hiện có.
5. Lưu `triage_region` vào event trước khi tạo signature hoặc `pnn_mask`.

#### Verification

- Extend `tests/online/test_online_tta_triage.py` với guard history không làm
  đổi region.
- Kết quả mong đợi: cùng scores luôn cho cùng `triage_region`, dù guard có
  interval overlap.

### Stage 4.2: Current action stage

**Mục tiêu:** xử lý action của current `causal_window` trước verification cycle.

#### Atomic steps

1. Với A2 và `triage_region=hard_old_normality`, chạy guarded hard-old update.
2. Với `triage_region=gray_zone`, tạo `verification_entry` vào
   `verification_buffer`; chưa chạy verification ngay trong bước admission.
3. Với `normal` hoặc `strong_anomaly`, không update và không tạo entry.
4. Với A1, không update current event nếu chưa có verified non-empty `pnn_mask`.
5. Với A0, return sau prediction như Stage 2.2.
6. Tạo `did_update` từ kết quả action thật, không suy diễn từ variant hoặc field
   bị thiếu.

#### Verification

- Spy order test phải thấy current hard-old action xảy ra trước callback
  verification.
- Buffer tests phải thấy chỉ gray-zone current windows được admit.

### Stage 4.3: Guard stage

**Mục tiêu:** `hard_old_interval_guard` chỉ chặn update, không đổi region.

#### Atomic steps

1. Trong `_execute_window_event_step`, chỉ gọi
   `NonOverlapGuard.accept` khi current region là `hard_old_normality` và
   variant/action cho phép hard-old update.
2. Nếu guard reject, giữ `event["triage_region"]` là
   `hard_old_normality`.
3. Set `did_update=False`, không gọi optimizer và không add interval mới.
4. Nếu update pass, gọi `NonOverlapGuard.add` sau optimizer step thành công.
5. Không gọi `_classify_event_window` lần hai sau guard.
6. Giữ interval semantics trong `src/engine/online_tta/non_overlap_guard.py`;
   chỉ đổi owner của quyết định region/update.

#### Verification

- Extend `tests/online/test_online_tta_triage.py` với rejected overlap.
- Kết quả mong đợi: record có `triage_region=hard_old_normality` và
  `did_update=False`.

### Stage 4.4: Verification stage

**Mục tiêu:** chỉ `verification_cycle` xử lý buffered entries và tạo PNN data.

#### Atomic steps

1. Tách `_admit_and_verify_online_window` thành admission step và cycle-run
   step; mỗi hàm chỉ có một trách nhiệm.
2. Chỉ admit current `gray_zone` thành `verification_entry` bằng
   `_update_online_window_buffers`.
3. Khi `VerificationCycleController.maybe_run` đủ điều kiện, lấy entries từ
   `verification_buffer`.
4. Trong cycle, tạo signatures, `recurrent_signature_set` và `pnn_mask` từ các
   entries đang kiểm tra; không lấy current normal/hard-old window vào set.
5. Gọi `verify_buffer_entries` và giữ một `VerificationResult` cho mỗi
   `entry_id`.
6. Chỉ gọi `execute_online_tta_step` khi candidate adapted và `pnn_mask` không
   rỗng; giữ `was_adapted` cho buffer bookkeeping.
7. Không thêm `verification_outcome` vào event hoặc record. Nếu cần log, ghi kết
   quả theo `entry_id` trong verification history hiện có.

#### Verification

- Spy test kiểm tra triage -> current action -> cycle.
- Command: `.venv/bin/python -m pytest -q tests/online/test_online_verification_buffer.py tests/online/test_verification_cycle.py`.
- Kết quả mong đợi: mỗi entry có `VerificationResult`; không có field
  `verification_outcome` canonical.

### Stage 4.5: Cycle-local signature stage

**Mục tiêu:** `recurrent_signature_set` được tạo và bỏ trong một cycle.

#### Atomic steps

1. Trong `src/engine/online_tta/online_engine_window_metrics.py`, bỏ việc
   append mọi current window vào global `signature_history` trước triage.
2. Tạo local collection khi cycle bắt đầu từ buffered entries đang được verify.
3. Tính `recurrent_signature_set` từ local collection và dùng nó để tạo
   `pnn_mask`.
4. Kết thúc cycle thì giải phóng local collection; không đưa nó vào
   `online_runtime_state`.
5. Xóa parameter plumbing của `signature_history` trong
   `online_engine_run.py`, `online_engine_window_core.py` và
   `online_engine_shared.py` nếu không còn caller hợp lệ.
6. Không đổi tên `signature_history` thành `recurrent_signature_set` nếu schema,
   lifecycle và owner vẫn khác nhau.

#### Verification

- Extend `tests/online/test_verification_cycle.py` với hai cycle độc lập.
- Kết quả mong đợi: cycle sau không nhìn thấy signature chỉ có ở cycle trước.

#### Complete when

Per-window order là:

```text
score -> EWMA/prediction -> triage_region -> current action
  -> verification_cycle -> online_event_record
```

Guard reject không đổi region, A0 không vào adaptation/verification, và PNN
không được tạo trước triage.

## Phase 5: Persist and restore the vector runtime state

### Stage 5.1: Vector runtime schema

**Mục tiêu:** `online_runtime_state` lưu đủ point state để xử lý window kế tiếp.

#### Atomic steps

1. Trong `src/engine/online_tta/runtime_state.py`, sửa `OnlineRuntimeState` để
   thay `previous_ewma_score` bằng serializable absolute-index EWMA state và
   latest prediction state.
2. Lưu current window indices/cursor cần để resume đúng stream position.
3. Giữ `verification_entries`, `verification_history` và
   `hard_old_intervals`.
4. Bỏ `signature_history` và `recurrent_signatures` khỏi persistent canonical
   state; không thêm finalized-point list.
5. Cập nhật `to_dict`, `from_dict`, `build_online_runtime_state` và field
   validation.
6. Tăng `runtime_schema_version` và validate version trước khi khôi phục mutable
   objects.

#### Contract của stage

- **Input:** mapping absolute index -> EWMA/prediction, cursor, buffer state,
  artifact identity.
- **Output:** JSON-serializable state đủ cho next-window result.
- **Lỗi:** scalar-only payload hoặc sai version bị reject trước mutation.
- **Compatibility:** giữ old checkpoints để rollback; không backfill vector bằng
  zero.

#### Verification

- Extend `tests/online/test_online_runtime_state.py` với round-trip vector map.
- Command: `.venv/bin/python -m pytest -q tests/online/test_online_runtime_state.py`.

### Stage 5.2: Mutable verification state

**Mục tiêu:** checkpoint lưu buffer và guard state, nhưng không lưu signature set
dài hạn.

#### Atomic steps

1. Trong `src/engine/online_tta/online_engine_shared.py`, sửa
   `_sync_online_runtime_state` để ghi vector state, cursor và point latest
   predictions.
2. Serialize entries theo `entry_id`, interval, score và fields cần để
   `build_entry_batch` dựng lại cùng `absolute_indices`.
3. Serialize `verification_history` và hard-old intervals sau khi action/cycle
   đã hoàn tất.
4. Không serialize local `recurrent_signature_set` của cycle đã kết thúc.
5. Validate that restored artifact identity equals current startup artifact.

#### Verification

- Test restore buffer/guard identity trong `tests/online/test_online_runtime_state.py`.
- Kết quả mong đợi: buffer entries và intervals giống trước checkpoint; signature
  local không tồn tại sau restore.

### Stage 5.3: Checkpoint version gate

**Mục tiêu:** reject incompatible scalar state trước khi runtime mutation.

#### Atomic steps

1. Trong `validate_resume_state` và `restore_online_runtime_state`, kiểm tra
   `runtime_schema_version` trước khi gán state vào model/buffer.
2. Nếu payload chỉ có `previous_ewma_score`, raise lỗi schema rõ ràng cho vector
   runtime.
3. Nếu entity, `online_variant`, threshold artifact hoặc checkpoint identity
   lệch, raise trước restore.
4. Giữ loader legacy ở một branch được gắn nhãn; branch đó không được tự tạo
   vector giả.
5. Cập nhật checkpoint export ở
   `scripts/benchmarks/run_thesis_online_benchmark.py` để có cả structured
   runtime state và legacy wrapper nếu compatibility cần.

#### Verification

- Test malformed version và scalar payload trong
  `tests/online/test_online_state_roundtrip.py`.
- Command: `.venv/bin/python -m pytest -q tests/online/test_online_state_roundtrip.py tests/online/test_online_runtime_state.py`.
- Kết quả mong đợi: state cũ bị reject trước khi object mutable đổi.

### Stage 5.4: Resume equivalence

**Mục tiêu:** resumed stream cho cùng next event như uninterrupted stream.

#### Atomic steps

1. Chọn một deterministic A2 fixture có ít nhất hai overlapping windows và một
   gray-zone entry.
2. Chạy uninterrupted đến cursor `k`, lưu next `online_event_record`.
3. Chạy lại đến cursor `k`, serialize state, restore vào process mới.
4. Xử lý window kế tiếp trong process restored.
5. So sánh absolute indices, EWMA vector, prediction vector, triage region,
   did_update, buffer state và loss summary.
6. Nếu khác, trace divergence từ state restore trước khi sửa record consumer.

#### Verification

- Command: `.venv/bin/python -m pytest -q tests/online/test_online_state_roundtrip.py`.
- Manual: một smoke run stop/resume ở cursor cố định phải có next event giống
  uninterrupted run.

#### Complete when

- Vector state và mutable verification state round-trip được.
- Incompatible old state fail trước mutation.
- Next event của resumed run giống uninterrupted run.

## Phase 6: Publish vector results to records and the live demo

### Stage 6.1: Event record boundary

**Mục tiêu:** một `online_event_record` chứa cùng vector contract mà runtime đã
tính.

#### Atomic steps

1. Trong `src/engine/online_tta/online_engine_step.py`, tách record construction
   khỏi model update trong `execute_online_tta_step`.
2. Trong `_build_step_record`, giữ `entity_id`, interval, `online_variant`,
   `triage_region`, `did_update`, và loss summary.
3. Trong `_build_event_window_outputs` và
   `_build_online_window_outputs`, thêm `window_point_scores`,
   `current_window_ewma_point_scores`, `window_point_predictions` và
   `online_point_ewma_threshold`.
4. Giữ `absolute_indices` dưới `causal_window` trừ khi ontology có quyết định
   mới cho record field này.
5. Không tạo một vector bằng cách lặp lại endpoint scalar.
6. Nếu legacy scalar record vẫn cần đọc, ghi rõ đó là compatibility boundary.

#### Verification

- Extend `tests/online/test_full_spec_online_contract.py` để kiểm tra field names,
  shapes và absence of fabricated copies.
- Kết quả mong đợi: event vector có cùng L với causal window.

### Stage 6.2: Metrics and retention

**Mục tiêu:** metrics và retention giữ vector cùng explicit update outcome.

#### Atomic steps

1. Trong `scripts/benchmarks/run_thesis_online_benchmark.py`, sửa
   `_normalize_online_records` để giữ `did_update=False` nếu field đã có.
2. Không dùng `setdefault("did_update", True)` cho non-A0 record mới.
3. Trong `_export_online_retention_bundle`, ghi vector records, metrics,
   threshold artifact và new runtime state.
4. Tăng bundle schema version nếu schema record/state thay đổi.
5. Giữ scalar baseline adapter ở branch riêng và gắn nhãn baseline.
6. Kiểm tra manifest có checkpoint hash, artifact hash, config hash và vector
   record count.

#### Verification

- Command: `.venv/bin/python -m pytest -q tests/benchmarks/test_thesis_online_benchmark_wrapper.py tests/online/test_full_spec_online_contract.py`.
- Kết quả mong đợi: explicit false không bị đổi thành true; retention đọc lại
  được vectors.

### Stage 6.3: Live event delivery

**Mục tiêu:** UI nhận event trực tiếp sau khi prediction vector đã sẵn sàng.

#### Atomic steps

1. Xác định demo entrypoint hiện có trong `demo/online_replay.py`,
   `demo/demo_state.py` và `demo/plotting.py`.
2. Thêm optional read-only `online_event_callback` ở owner của online loop
   (`src/engine/online_tta/online_engine_run.py` hoặc boundary được xác nhận
   trong source).
3. Gọi callback sau khi `window_point_predictions` và record fields đã hoàn tất.
4. Pass một copy sâu hoặc immutable view; callback không được nhận reference có
   thể sửa model, buffer hoặc runtime state.
5. Nếu callback raise, quyết định rõ theo runtime policy: UI lỗi không được làm
   mất record hoặc làm hỏng runtime state; ghi diagnostic và tiếp tục hoặc
   dừng theo policy đã có.
6. Giữ file replay như compatibility reader, không gọi nó là live callback.

#### Verification

- Add callback isolation test trong `tests/demo/test_live_online_replay.py` hoặc
  proposed online-flow test.
- Kết quả mong đợi: callback thấy vector event; mutation thử nghiệm trong
  callback không đổi record/runtime state.

### Stage 6.4: Demo compatibility

**Mục tiêu:** THESIS demo vẽ vector theo absolute index; baseline scalar vẫn đọc
được qua adapter riêng.

#### Atomic steps

1. Trong `demo/online_replay.py`, nhận `window_point_scores`,
   `current_window_ewma_point_scores`, `window_point_predictions` và
   `absolute_indices` từ THESIS record.
2. Trong `demo/demo_state.py`, giữ state theo absolute index thay vì chỉ theo
   record endpoint.
3. Trong `demo/plotting.py`, tạo time axis từ absolute indices.
4. Khi event sau chứa point đã có, cập nhật đúng point đó trên UI.
5. Không lặp endpoint value cho mọi point trong window.
6. Giữ scalar-to-array adapter chỉ cho baseline records và đặt tên adapter rõ
   là compatibility path.

#### Verification

- Extend `tests/demo/test_demo_state.py` và `tests/demo/test_live_online_replay.py`.
- Command: `.venv/bin/python -m pytest -q tests/demo/test_demo_state.py tests/demo/test_live_online_replay.py`.
- Manual: hai overlapping events làm shared point trên UI đổi giá trị.

#### Complete when

- Một event vector điều khiển được report, retention, plot và callback.
- Không có consumer THESIS nào tạo vector giả từ endpoint scalar.

## Phase 7: Validate the complete flow and align specifications

### Stage 7.1: Focused validation

**Mục tiêu:** test từng boundary theo dependency order.

#### Atomic steps

1. Chạy artifact schema và checkpoint identity tests.
2. Chạy absolute-index, batch-contract và point-EWMA tests.
3. Chạy A0/A1/A2, loss, triage, guard và verification-order tests.
4. Chạy runtime-state round-trip và resume-equivalence tests.
5. Chạy record, retention, demo và callback-isolation tests.
6. Nếu nhóm nào fail, sửa nhóm đó trước khi chạy smoke; không chuyển sang
   full matrix.

#### Automated verification

```text
.venv/bin/python -m pytest -q \
  tests/online/test_online_ewma_threshold.py \
  tests/online/test_online_tta_triage.py \
  tests/online/test_online_tta_variants.py \
  tests/online/test_online_verification_buffer.py \
  tests/online/test_verification_cycle.py \
  tests/online/test_online_runtime_state.py \
  tests/online/test_online_stream.py \
  tests/online/test_threshold_artifact.py
```

Expected result: all listed tests pass without modifying retained artifacts.

### Stage 7.2: One concrete smoke path

**Mục tiêu:** chứng minh full flow trên một combination trước khi mở rộng.

#### Atomic steps

1. Chọn một existing THESIS online config có `online_variant=A2`, một
   `experiment_config`, một protocol config và một offline Stage B checkpoint.
2. Kiểm tra read-only artifact/checkpoint identity trước khi chạy.
3. Chạy command entrypoint hiện có với đúng config và một explicit online
   variant; không chạy matrix.
4. Inspect startup report: artifact path, checkpoint hash, entity, window size,
   EWMA weights và schema.
5. Inspect one overlapping pair: absolute indices, three vectors, triage region,
   hard-old guard outcome, verification history và did_update.
6. Nếu UI callback bật, inspect event copy và callback isolation.
7. Giữ smoke artifacts theo retention policy; không xóa broad output tree.

#### Verification

- Expected: one run completes, no online recalibration occurs, and vector event
  matches the focused integration test.
- Manual comparison: resumed next event equals uninterrupted next event.

### Stage 7.3: Protocol status

**Mục tiêu:** chỉ công bố `full_spec_v3` sau evidence end-to-end.

#### Atomic steps

1. Trong `src/engine/online_tta/online_engine_run.py`, tìm completion report
   field `runtime_protocol_status`.
2. Giữ status cũ trong lúc focused tests hoặc smoke chưa pass.
3. Sau khi Stage 7.1 và 7.2 pass, đổi status thành `full_spec_v3`.
4. Ghi artifact/checkpoint/report paths và test evidence cùng completion report.
5. Không gắn status mới cho dry-run hoặc legacy scalar path.

#### Verification

- Extend `tests/online/test_full_spec_online_contract.py` để status chỉ pass khi
  vector contract fields có mặt.
- Kết quả mong đợi: report của verified smoke run có đúng
  `runtime_protocol_status=full_spec_v3`.

### Stage 7.4: Specification alignment

**Mục tiêu:** source, ontology, full spec và pseudocode nói cùng một behavior.

#### Atomic steps

1. Cập nhật `documents/spec/online_tta_terminology_ontology.md` với field path
   artifact đã được chốt, vector state, A0 boundary, cycle-local signature và
   callback owner.
2. Cập nhật `documents/spec/full-spec-v3.md` với schema record/state đã được
   test; không ghi behavior chưa có evidence.
3. Cập nhật `documents/notes/online_runtime_flow_debug.md` để pseudocode dùng
   đúng các tên `causal_window`, `window_point_scores`,
   `current_window_ewma_point_scores`, `window_point_predictions`,
   `triage_region`, `verification_entry`, `verification_cycle`,
   `recurrent_signature_set`, `pnn_mask` và `online_event_record`.
4. Ghi rõ `pnn_verified` chỉ là internal compatibility control nếu vẫn còn
   trong step API; không dùng nó làm `triage_region`.
5. Ghi rõ `VerificationResult` và `was_adapted` vẫn là hai object có owner khác
   nhau; không thêm `verification_outcome`.
6. Chạy terminology audit bằng `rg` để tìm tên obsolete và kiểm tra từng match
   là canonical, source alias hoặc legacy compatibility.

#### Verification

- Command: `rg -n "verification_outcome|signature_history|previous_ewma_score" src scripts tests documents/spec documents/notes`.
- Command: `.venv/bin/python -m pytest -q tests/online/test_full_spec_online_contract.py tests/online/test_full_spec_online_losses.py`.
- Manual: đọc pseudocode một lượt từ đầu đến cuối và map mỗi object vào đúng
  một ontology entry.

#### Complete when

- Runtime, tests, reports, ontology và pseudocode cùng mô tả `full_spec_v3`.
- Không còn tài liệu nào mô tả endpoint-only scoring hoặc pre-triage
  verification là desired behavior.

## Interface and data changes

### `causal_window`

- Target: thêm `absolute_indices: LongTensor[B,L]`.
- Owner: `SMDOnlineStream._build_window`, `collate_windows` và contracts.
- Validation: rank, dtype, length và strictly increasing order.
- Compatibility: giữ interval metadata.

### `online_event_record`

- Target fields: `window_point_scores`,
  `current_window_ewma_point_scores`, `window_point_predictions`,
  `online_point_ewma_threshold`, `triage_region`, `did_update`, loss summary và
  identity fields.
- `absolute_indices` vẫn thuộc `causal_window` nếu ontology chưa cho phép đưa
  nó lên record root.
- Không thêm `verification_outcome`.

### `online_runtime_state`

- Target: absolute-index point EWMA/latest prediction maps, cursor, buffer
  entries/history, hard-old intervals, artifact/checkpoint identity.
- Removed from canonical persistence: global `signature_history`, persistent
  `recurrent_signatures`, finalized-point list và scalar-only
  `previous_ewma_score`.
- Migration: tăng `runtime_schema_version`; reject incompatible state trước
  mutation; giữ old files để rollback.

### `threshold_artifact`

- Target: four online thresholds, EWMA weights, window/entity identity, schema
  version và checkpoint SHA-256.
- Owner: offline benchmark creates; online runtime reads.
- The canonical config/path field remains an explicit Phase 1 gate until the
  ontology accepts it.

## Deployment and rollout

1. Merge and test artifact/schema changes before enabling vector runtime.
2. Run one concrete smoke combination with old scalar artifacts untouched.
3. Enable A0 first to validate vector scoring without model mutation.
4. Enable A1/A2 only after triage/order and resume tests pass.
5. Enable direct UI callback after callback-isolation tests pass. File replay
   remains the fallback reader.
6. Mark `full_spec_v3` only after the smoke report and focused test evidence.
7. Roll back by selecting the previous code and matching scalar artifact/checkpoint
   pair. Do not overwrite old retained bundles.

## Documentation changes

- `documents/spec/online_tta_terminology_ontology.md`: final object ownership,
  artifact field, vector state and lifecycle.
- `documents/spec/full-spec-v3.md`: verified record/state contract.
- `documents/notes/online_runtime_flow_debug.md`: complete pseudocode in
  canonical names and plain language.
- This detail file: atomic implementation instructions and verification gates.

## Final verification

- [ ] Online loads the offline `threshold_artifact` and matching
  `stage_b_best_checkpoint` without recalibration.
- [ ] A0 creates no `online_mlp_projector`, optimizer, verification entry or
  verification cycle.
- [ ] First-seen and overlapping points produce correct EWMA/prediction vectors.
- [ ] A hard-old guard rejection keeps `triage_region=hard_old_normality` and
  sets `did_update=False`.
- [ ] Triage precedes PNN creation; accepted A2 hard-old action precedes
  `verification_cycle`.
- [ ] `recurrent_signature_set` is cycle-local and not persisted.
- [ ] `VerificationResult` and `was_adapted` remain; no common
  `verification_outcome` is added.
- [ ] Resumed next event equals uninterrupted next event.
- [ ] Retention and direct UI callback expose vectors without endpoint copying.
- [ ] Focused tests and one smoke path pass before `full_spec_v3`.

## Assumptions and non-blocking uncertainties

- The canonical configuration field or approved resolver rule for locating the
  offline `threshold_artifact` must be confirmed before Phase 1 source edits.
  The detail does not invent a field name.
- `absolute_indices` remains owned by `causal_window`; moving it to the record
  root needs an ontology decision.
- The existing demo remains the UI owner. If its API cannot accept a callback,
  keep file replay as a compatibility fallback and record the limitation.
- The plan mentioned a later “Phase 8” in one sentence, but the approved
  structure has seven phases. This detail follows the seven-phase structure.
