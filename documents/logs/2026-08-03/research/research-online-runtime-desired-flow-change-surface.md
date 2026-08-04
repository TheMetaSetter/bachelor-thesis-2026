---
date: 2026-08-04 16:28:27 +07:00
researcher: OpenAI Codex
topic: "Xác định code liên quan và loại bỏ trường kết quả verification dư thừa trong luồng online"
status: complete
revision: 4be64456d6aa652457a0702154bae0d9b742a803
branch: dev
terminology_ontology: documents/spec/online_tta_terminology_ontology.md
terminology_audit_status: aligned
---

# Nghiên cứu: Những phần code cần sửa để chạy đúng luồng online mong muốn

## Tóm tắt

Không thể làm cho pha online chạy đúng “Flow người dùng mong muốn” bằng cách chỉ sửa một hàm. Cần thay đổi bốn nhóm code:

1. `online_tta_phase` phải nạp `stage_b_best_checkpoint` và đọc `threshold_artifact` do `offline_pretraining_phase` tạo. Pha online không được tự hiệu chỉnh lại các ngưỡng khi khởi động.
2. Code tính điểm phải giữ toàn bộ `window_point_scores [L]`, ghép các điểm số theo `causal_window.absolute_indices`, rồi tạo `window_point_predictions [L]`. Code hiện tại chỉ giữ `endpoint_point_score`.
3. Code điều phối phải tạo `triage_region` trước khi chạy `verification_cycle` và tạo `pnn_mask`. Với A2, `online_update_event` cho `hard_old_normality` phải chạy trước `verification_cycle`. Không cần thêm một trường kết quả verification riêng; `pnn_mask` và `VerificationResult` đã cung cấp dữ liệu cần thiết.
4. `online_runtime_state`, checkpoint, `online_event_record`, các chỉ số và demo phải chuyển từ những đối tượng chỉ chứa một giá trị ở endpoint sang những vector chứa kết quả của cả window.

Một số phần hiện có vẫn dùng lại được. Chúng gồm hàm phân loại bốn vùng để tạo `triage_region`, `verification_buffer`, `verification_cycle` dùng `frozen_source_model`, `recurrent_signature_set`, `pnn_mask`, `pnn_reconstruction_loss` và bộ tối ưu chỉ cập nhật các tham số của `online_mlp_projector`. Phần lớn thay đổi nằm trong `src/engine/online_tta/online_engine_run.py`, `src/engine/online_tta/online_engine_window_core.py`, `src/engine/online_tta/online_engine_window_metrics.py`, `src/engine/online_tta/online_engine_step.py` và `src/engine/online_tta/runtime_state.py`.

Hai quy tắc đã được chốt:

- Nếu một point tại vị trí tuyệt đối chưa có EWMA trước đó, code dùng trực tiếp điểm số tương ứng trong `window_point_scores`.
- A0 không khởi tạo `online_mlp_projector`.

Prediction của point được cập nhật mỗi khi point còn xuất hiện trong sliding window. Khi các window sau không còn chứa point, prediction gần nhất được giữ nguyên. Không cần state hoặc bước finalization riêng.

Trong tài liệu này, *point* là một thời điểm trên toàn bộ stream. *Window* là một đoạn gồm nhiều point liên tiếp. *Absolute index* là vị trí của point trên toàn bộ stream, không phải vị trí cục bộ bên trong window. *Endpoint* là point cuối của một window.

*Scalar* là một giá trị đơn. *Vector* là một dãy nhiều giá trị. *Loss* là giá trị mà bước train dùng để cập nhật model. `threshold_artifact` là tệp lưu các ngưỡng và thông tin dùng để kiểm tra nguồn gốc của chúng. `stage_b_best_checkpoint` là checkpoint Stage B được quy tắc theo dõi chọn là tốt nhất. *Schema* quy định một tệp hoặc đối tượng có những trường nào và mỗi trường chứa kiểu dữ liệu gì.

Các tên nằm trong dấu backtick là tên chính xác trong ontology hoặc trong mã nguồn.

## Kiểm tra thuật ngữ

`documents/spec/online_tta_terminology_ontology.md` là tài liệu quyết định tên đối tượng trong nghiên cứu này. Các quy tắc sau được áp dụng:

- Dùng tên canonical khi nói về một đối tượng trong luồng mong muốn.
- Giữ nguyên tên lớp, hàm, tham số và trường dữ liệu khi trích dẫn code hiện tại. Mỗi tên như vậy phải được ghi rõ là tên hoặc alias trong code lúc chạy.
- Không dùng tên của vector trong luồng mong muốn để gọi một giá trị scalar của endpoint trong code hiện tại.
- Không tạo thêm object canonical nếu code đã có dữ liệu trực tiếp để quyết định và ghi nhận kết quả.
- “hard-old” chỉ là cách viết ngắn trong câu văn cho `hard_old_normality`. Nó không phải một tên định danh mới.

| Tên trong code hiện tại | Tên canonical | Quan hệ theo ontology | Cách dùng trong nghiên cứu này |
| --- | --- | --- | --- |
| trường `raw_point_score` | `endpoint_point_score` | Đổi tên để nói rõ đây là một scalar | Chỉ dùng khi mô tả code hiện tại hoặc việc chuyển đổi |
| trường `previous_ewma_score` | `previous_endpoint_ewma_point_score` | Đổi tên để nói rõ đây là một scalar | Chỉ dùng khi mô tả code hiện tại hoặc việc chuyển đổi |
| trường `ewma_point_score` | `current_endpoint_ewma_point_score` | Đổi tên để nói rõ đây là một scalar | Chỉ dùng khi mô tả code hiện tại hoặc việc chuyển đổi |
| trường `prediction` | `endpoint_point_prediction` | Tên hiện đang dùng lúc chạy | Chỉ dùng khi mô tả code hiện tại hoặc việc chuyển đổi |
| tham số `triage_decision` | `triage_region` khi giá trị thuộc bốn vùng | Exact alias có điều kiện | Không dùng tham số này để chứa `pnn_verified` trong luồng mới |
| biến cục bộ `threshold_value` | `online_point_ewma_threshold` trong chuỗi lời gọi hiện tại | Contextual alias | Phải tách khỏi `input_window_threshold` |
| lớp `NonOverlapGuard` | `hard_old_interval_guard` | Tên lớp trong code của đối tượng canonical | Chỉ dùng tên lớp khi trích dẫn code |
| lớp `VerificationBuffer` | `verification_buffer` | Tên lớp và tên instance | Không đồng nhất với `verification_entry` |
| trường `signature_history` | Chưa xác định | Ontology chưa xác nhận đây là alias của `recurrent_signature_set` | Chỉ mô tả cách code hiện tại lưu dữ liệu; chưa được dùng thay cho `recurrent_signature_set` |

Nghiên cứu này không định nghĩa thêm object canonical cho kết quả verification. `pnn_mask` không rỗng đã là điều kiện cho PNN update path. `VerificationResult` đã lưu `adapted`, `reason` và `pnn_mask` cho từng entry.

## Câu hỏi nghiên cứu

Sử dụng `prompts/1_research_prompt.md` để xác định những đoạn code cần sửa để pha online chạy theo “Flow người dùng mong muốn” trong `documents/notes/online_runtime_flow_debug.md`.

## Ngữ cảnh hệ thống

### Luồng cần đạt

“Flow người dùng mong muốn” yêu cầu các bước chạy theo thứ tự sau:

```text
nạp stage_b_best_checkpoint và khôi phục frozen_source_model
  -> đọc threshold_artifact
  -> nhận causal_window
  -> frozen_source_model tạo source_hidden
  -> A0 dùng source_hidden; A1/A2 dùng online_mlp_projector để tạo projected_hidden
  -> tạo window_point_scores
  -> tạo current_window_ewma_point_scores
  -> tạo window_point_predictions
  -> A0 kết thúc bước hiện tại; A1/A2 tạo triage_region
  -> chạy online_update_event được phép cho hard_old_normality hoặc tạo verification_entry cho gray_zone
  -> chạy verification_cycle khi đủ điều kiện và dùng pnn_mask để quyết định PNN update
  -> tạo online_event_record
  -> lưu previous_window_ewma_point_scores và trạng thái cần để chạy tiếp vào online_runtime_state
```

Luồng đầy đủ nằm tại `documents/notes/online_runtime_flow_debug.md:26-268`. Theo ontology, `window_point_scores`, `current_window_ewma_point_scores` và `window_point_predictions` là các vector của luồng mong muốn. Chúng không phải alias của ba giá trị scalar đang dùng lúc chạy: `endpoint_point_score`, `current_endpoint_ewma_point_score` và `endpoint_point_prediction` (`documents/spec/online_tta_terminology_ontology.md:153-182`).

### Điểm bắt đầu thực thi

Lệnh thực tế đi qua chuỗi hàm sau:

```text
python -m scripts.run_thesis_online_benchmark
  -> scripts.benchmarks.run_thesis_online_benchmark.main
  -> run_thesis_online_benchmark
  -> run_thesis_online_tta_experiment
  -> _run_online_sequence
  -> _process_online_window
```

Bằng chứng: `scripts/run_thesis_online_benchmark.py:1-9`, `scripts/benchmarks/run_thesis_online_benchmark.py:226-307`, `src/engine/online_tta/online_engine_run.py:518-589` và `src/engine/online_tta/online_engine_run.py:223-315`.

## Luồng code hiện tại

Mỗi batch đi qua bốn hàm trong `_process_online_window`:

```text
_prepare_online_window_event
  -> _admit_and_verify_online_window
  -> _execute_window_event_step
  -> _build_event_window_outputs
```

Thứ tự gọi nằm tại `src/engine/online_tta/online_engine_window_core.py:53-118`.

Trong `_prepare_online_window_event`, code chạy bốn việc theo thứ tự này:

1. Chạy model và lấy trường `raw_point_score`. Trường này tương ứng với `endpoint_point_score`.
2. Tính trường `ewma_point_score`. Trường này tương ứng với `current_endpoint_ewma_point_score`.
3. Tạo `pnn_mask` sơ bộ và thêm mọi A1/A2 `causal_window` vào trường toàn cục `signature_history`.
4. Sau cùng mới tạo `triage_region`.

Bằng chứng: `src/engine/online_tta/online_engine_window_core.py:151-220` và `src/engine/online_tta/online_engine_window_metrics.py:82-194`.

Sau đó, code có thể tạo `verification_entry` từ một `gray_zone`, chạy `verification_cycle` và cập nhật model bằng các `verification_entry` đã lưu. Code chỉ xét `online_update_event` của `causal_window` hiện tại sau các bước này. Bằng chứng: `src/engine/online_tta/online_engine_window_core.py:223-278`.

## Các nhóm code cần sửa

### 1. Chuyển `threshold_artifact` từ pha offline sang pha online

| Hành vi cần đạt | Code hiện tại | Vị trí cần sửa |
| --- | --- | --- |
| `online_tta_phase` đọc `threshold_artifact` do `offline_pretraining_phase` tạo | Mỗi lần chạy online, `_build_runtime_online_context` gọi `calibrate_entity_threshold_artifacts` trên clean validation | `src/engine/online_tta/online_engine_run.py:129-220` |
| `threshold_artifact` và `stage_b_best_checkpoint` thuộc cùng một lần chạy offline | Code bao ngoài chỉ tìm `stage_b_best_checkpoint`. Nó chưa tìm và nạp `threshold_artifact` của pha offline | `scripts/benchmarks/run_thesis_online_benchmark.py:226-256`; `src/engine/online_tta/checkpoint_resolution.py:35-113` hoặc một hàm tìm tệp riêng |
| `threshold_artifact` chứa đủ bốn ngưỡng online canonical | Hàm offline `_build_thresholds` truyền `offline_point_threshold` và `online_point_ewma_threshold`. `offline_point_threshold` không phải alias của `online_point_ewma_threshold`. Hàm này chưa truyền `input_window_threshold`, `latent_window_low_threshold` và `latent_window_high_threshold` | `scripts/benchmarks/run_thesis_offline_benchmark.py:499-533` |
| `online_point_ewma_threshold` dùng EWMA của các window chồng lấn theo absolute index | Code offline tính EWMA trực tiếp trên trục thời gian gồm các point không chồng lấn. Code hiệu chỉnh online chỉ lấy `endpoint_point_score` | `scripts/benchmarks/run_thesis_offline_benchmark.py:504-527`; `src/engine/online_tta/online_calibration.py:49-110`; `src/engine/online_tta/online_calibration.py:153-188` |
| `threshold_artifact` lưu hash của checkpoint trước khi pha online đọc | Code offline ghi `threshold_artifact` trước khi tính hash của `stage_b_best_checkpoint` | `scripts/benchmarks/run_thesis_offline_benchmark.py:536-556` |

Cần thực hiện các thay đổi sau:

- Tìm đúng `threshold_artifact` bằng cùng `offline_variant`, `entity_id`, `seed` và `benchmark_mode` đang dùng để tìm `stage_b_best_checkpoint`. Phải thêm tên trường cấu hình chứa đường dẫn này vào ontology trước khi triển khai; không được tự đặt một tên mới rồi xem đó là canonical.
- Gọi `load_threshold_artifact` trước khi tạo `online_runtime_state`. Sau khi đọc, phải kiểm tra đúng entity, `window_size`, SHA-256 của `stage_b_best_checkpoint`, các EWMA weights và phiên bản schema.
- Bỏ bước hiệu chỉnh lại ngưỡng online và bỏ `_persist_threshold_artifacts` khỏi quá trình khởi động THESIS online.
- Sửa code tạo `threshold_artifact` offline để tính `online_point_ewma_threshold`, `input_window_threshold`, `latent_window_low_threshold` và `latent_window_high_threshold` bằng đúng cách tính điểm mà `window_point_predictions` sẽ dùng.
- Trường nguồn gốc `variant_name` hiện chứa `online_variant` trong quá trình hiệu chỉnh online. Cần đổi trường này thành định danh của pha offline hoặc thêm một trường có nghĩa rõ ràng. Không dùng cùng một trường mơ hồ cho cả O0/O1 và A0/A1/A2.

Hai nhóm test đang xác nhận hành vi cũ nằm tại `tests/online/test_online_entrypoint.py:162-300` và `tests/online/test_online_entrypoint.py:303-439`. Sau khi sửa code, test phải kiểm tra việc nạp và xác thực `threshold_artifact` của pha offline. Test cũng phải xác nhận pha online không tự hiệu chỉnh lại ngưỡng.

### 2. Khởi tạo A0 và kết thúc sớm

Code hiện tại luôn làm các việc sau:

- Tạo `OnlineAdaptationModel`; hàm khởi tạo của lớp này luôn tạo `online_mlp_projector`.
- Tạo optimizer trước khi kiểm tra biến thể có phải A0 hay không.
- Kiểm tra rằng các tham số của `online_mlp_projector` có thể được cập nhật khi train.
- Vẫn tính `input_window_score`, `latent_window_score` và `triage_region`, rồi chạy `verification_buffer` và `verification_cycle` cho A0.

Bằng chứng: `src/models/online_impl/online_adaptation.py:90-106`, `src/engine/online_tta/online_engine_run.py:135-142`, `src/engine/online_tta/online_engine_window_core.py:73-110` và `src/engine/online_tta/online_engine_window_metrics.py:33-53`.

Cần sửa các vị trí sau:

- `_build_runtime_online_context` trong `src/engine/online_tta/online_engine_run.py` chỉ tạo optimizer cho A1/A2.
- `OnlineAdaptationModel.__init__` trong `src/models/online_impl/online_adaptation.py` không khởi tạo `online_mlp_projector` khi `online_variant = A0`.
- `_process_online_window` trong `src/engine/online_tta/online_engine_window_core.py` kết thúc bước A0 ngay sau khi tạo `window_point_predictions` và cập nhật trạng thái point trong `online_runtime_state`. A0 phải dừng trước khi tạo `triage_region`, thêm `verification_entry` hoặc chạy `verification_cycle`.
- `_forward_online_window` trong `src/engine/online_tta/online_engine_window_metrics.py` có thể tiếp tục gọi `forward_source`, vì đường chạy này không gọi `online_mlp_projector`.
- A0 không gọi `execute_online_tta_step` trong `src/engine/online_tta/online_engine_step.py`.

Quy tắc đã chốt là A0 không tạo `online_mlp_projector`. Vì vậy, code phải biết `online_variant` trước khi tạo model. Có thể truyền `online_variant` vào `OnlineAdaptationModel` hoặc tạo một đường khởi tạo riêng cho A0. Dù chọn cách nào, model A0 không được chứa `online_mlp_projector`.

### 3. Absolute index của `causal_window`

Ontology định nghĩa `causal_window.absolute_indices`. Stream hiện tại chỉ lưu `absolute_start_index` và `absolute_end_index` trong `meta`. Ontology chưa xác nhận hai trường này là trường canonical, và batch hiện chưa chứa tensor `absolute_indices`.

Bằng chứng: `src/data/stream.py:113-160`, `src/data/collate.py:11-35` và `src/core/contracts.py:66-110`.

Cần sửa bốn vị trí:

- `SMDOnlineStream._build_window` trong `src/data/stream.py` tạo `absolute_indices: LongTensor[L]`.
- `collate_windows` trong `src/data/collate.py` ghép các tensor thành kích thước `[B,L]`.
- `validate_window`, `validate_batch` và `validate_online_batch` trong `src/core/contracts.py` kiểm tra kiểu dữ liệu, số chiều, chiều dài và thứ tự tăng của các indices.
- `build_entry_batch` trong `src/engine/online_tta/verification_adapter.py` dựng lại `absolute_indices` khi đọc lại một `verification_entry` đã lưu.

### 4. `window_point_scores`, EWMA vector và `window_point_predictions`

`_extract_online_window_scores` hiện chỉ lấy `outputs["point_scores"][0, -1]`. Giá trị đơn này là `endpoint_point_score`. `_run_online_sequence` cũng chỉ giữ trường `previous_ewma_score: float | None`, tương ứng với `previous_endpoint_ewma_point_score`.

Bằng chứng: `src/engine/online_tta/online_engine_window_metrics.py:136-147` và `src/engine/online_tta/online_engine_run.py:265-300`.

Các vị trí cần sửa:

| Việc cần làm | Hàm hoặc file hiện tại |
| --- | --- |
| Giữ `window_point_scores [L]` thay vì chỉ giữ một scalar ở endpoint | `_extract_online_window_scores` và `_score_online_window` trong `src/engine/online_tta/online_engine_window_metrics.py:82-147` |
| Ghép điểm số theo `causal_window.absolute_indices` và tạo `current_window_ewma_point_scores` | Thêm hàm và trạng thái trong `src/protocols/point_scores.py`; hàm hiện tại tại `src/protocols/point_scores.py:71-92` chỉ tính EWMA theo thời gian cho một scalar |
| Truyền `previous_window_ewma_point_scores` từ window này sang window kế tiếp | `_run_online_sequence` trong `src/engine/online_tta/online_engine_run.py:223-315` |
| Truyền các vector của luồng mới qua `online_event` | `_prepare_online_window_event` và `_finalize_window_result` trong `src/engine/online_tta/online_engine_window_core.py:121-220` |
| Tạo `window_point_predictions [L]` trước khi cập nhật model | Tách code tạo `online_event_record` khỏi `execute_online_tta_step`; code hiện tại nằm tại `src/engine/online_tta/online_engine_step.py:76-101` |
| Ghi các vector của luồng mới vào bản ghi và các chỉ số | `_build_online_window_outputs` trong `src/engine/online_tta/online_engine_window_metrics.py:226-273` |

`execute_online_tta_step` hiện làm hai việc: tính loss và cập nhật model, sau đó tạo `endpoint_point_prediction` trong `online_event_record`. Luồng mới phải tách hai việc này. Code phải tạo `window_point_predictions` từ `online_model_outputs` đã có trước khi model được cập nhật. Bước cập nhật model chỉ trả kết quả của `online_update_event` và `online_total_loss`.

Khi tách code, cần sửa thêm một lỗi dữ liệu. `_execute_window_event_step` đang truyền `event["input_window_score"]` vào tham số `raw_point_score`. Vì vậy, `record["raw_point_score"]` có thể chứa `input_window_score`, dù trường này phải chứa `endpoint_point_score`. Trong khi đó, chỉ số `online/raw_point_score` lại lấy đúng `endpoint_point_score`. Bằng chứng: `src/engine/online_tta/online_engine_window_core.py:265-275`, `src/engine/online_tta/online_engine_step.py:87-99` và `src/engine/online_tta/online_engine_window_metrics.py:236-249`.

### 5. `triage_region` phải có trước `verification_cycle` và `pnn_mask`

Code hiện tạo `pnn_mask` trước khi có `triage_region`. Đường chạy này bắt đầu tại `src/engine/online_tta/online_engine_window_core.py:183-209` và làm bốn việc:

- Tính `known_anomaly_mask`.
- Tính `continuous_signature_ids`.
- Thêm `causal_window` hiện tại vào trường toàn cục `signature_history`.
- Tạo `pnn_mask` trước khi phân loại window.

Bằng chứng chi tiết: `src/engine/online_tta/online_engine_window_metrics.py:150-194`.

Để chạy đúng luồng mong muốn, cần thay đổi như sau:

- Bỏ lời gọi `_attach_event_pnn_mask` khỏi `_prepare_online_window_event`.
- Không gắn `pnn_mask` vào batch hiện tại trước khi tạo `triage_region`.
- Không thêm các `causal_window` thuộc `normal`, `hard_old_normality` hoặc `strong_anomaly` vào `signature_history`.
- Chỉ tính các tensor verification từ những `verification_entry` đang nằm trong `verification_buffer`, và chỉ tính khi `verification_cycle` đủ điều kiện chạy.

`verification_adapter.verify_buffer_entries` đã chỉ dùng `verification_entry` và `frozen_source_model` tại `src/engine/online_tta/verification_adapter.py:53-113`. Đây nên là đường chạy duy nhất tạo `pnn_mask`.

Phải bỏ trường toàn cục `signature_history` khỏi luồng đang chạy. Trường này xuất hiện tại `src/engine/online_tta/online_engine_run.py:219`, `src/engine/online_tta/online_engine_run.py:239-246` và `src/engine/online_tta/runtime_state.py:27-28`. Nếu cần giữ signatures qua nhiều `verification_cycle`, đặc tả và ontology phải xác định tên canonical, nơi sở hữu và vòng đời của trạng thái đó trước khi triển khai. Không được tự xem `signature_history` là `recurrent_signature_set`. Cũng không được tiếp tục lưu signatures của mọi A1/A2 `causal_window`.

### 6. `hard_old_interval_guard` không được đổi `triage_region`

`_classify_event_window` hiện đổi một `causal_window` chồng lấn có `triage_region = hard_old_normality` thành `triage_region = gray_zone`. Vì bị đổi vùng, window này có thể bị thêm vào `verification_buffer` dưới dạng một `verification_entry`.

Bằng chứng: `src/engine/online_tta/online_engine_window_core.py:307-324` và code thêm entry tại `src/engine/online_tta/online_engine_window_metrics.py:197-223`.

Luồng mong muốn yêu cầu:

```text
giữ triage_region = hard_old_normality
không chạy verification_cycle cho causal_window hiện tại
đặt did_update = false khi hard_old_interval_guard từ chối
không thêm causal_window vào verification_buffer
```

Pseudocode cũ từng gán một giá trị `NOT_RUN` cho kết quả verification. Quy tắc đó đã được loại bỏ cùng với trường kết quả riêng; nhánh này chỉ cần giữ `did_update = false` khi guard từ chối.

Cần sửa code như sau:

- `_classify_event_window` chỉ gọi `classify_online_window`. Bỏ tham số `hard_old_guard`, hiện trỏ tới `hard_old_interval_guard`, khỏi chữ ký của hàm phân loại.
- Chuyển bước kiểm tra `hard_old_interval_guard` vào nhánh A2 xử lý `online_update_event` cho `hard_old_normality`.
- Chỉ gọi `hard_old_interval_guard.add(interval)` sau khi `online_update_event` chạy thành công.

`NonOverlapGuard.accept/add` tại `src/engine/online_tta/non_overlap_guard.py:6-26` có thể giữ nguyên.

### 7. Chạy A2 `online_update_event` trước `verification_cycle`

Luồng mong muốn xử lý hành động của `triage_region` hiện tại trước, rồi mới chạy `verification_cycle`. Code hiện tại làm ngược lại: nó gọi `_admit_and_verify_online_window` trước `_execute_window_event_step`.

Bằng chứng: `src/engine/online_tta/online_engine_window_core.py:89-110`.

Thứ tự mới trong `src/engine/online_tta/online_engine_window_core.py` phải là:

```text
window_point_scores và window_point_predictions
  -> A0 kết thúc bước hiện tại
  -> tạo triage_region
  -> chạy online_update_event cho hard_old_normality, hoặc thêm gray_zone vào buffer, hoặc không làm gì
  -> chạy verification_cycle
  -> ghép online_event, online_update_event và kết quả verification
```

Nên tách `_admit_and_verify_online_window` tại `src/engine/online_tta/online_engine_window_core.py:223-250` thành hai hàm ngắn. Một hàm thêm `verification_entry` vào buffer. Hàm còn lại chạy `verification_cycle`. Cách tách này cũng giúp mỗi hàm không vượt quá 50 dòng theo quy tắc của repository.

### 8. Không thêm trường kết quả verification riêng

Đường chạy verification hiện gọi:

```python
execute_online_tta_step(..., triage_decision="pnn_verified")
```

Bằng chứng: `src/engine/online_tta/online_engine_window_metrics.py:54-78`.

`pnn_verified` không phải một trong bốn giá trị của `triage_region`. Trong code hiện tại, đây chỉ là chuỗi điều khiển nội bộ được truyền qua tham số `triage_decision` khi verification tìm thấy `pnn_mask` không rỗng.

Code hiện tại đã có đủ dữ liệu mà một trường kết quả verification riêng có thể cung cấp:

- `verify_buffer_entries` tạo một `VerificationResult` cho từng `verification_entry` tại `src/engine/online_tta/verification_adapter.py:82-114`.
- `VerificationResult` chứa `adapted`, `reason` và `pnn_mask` tại `src/engine/online_tta/verification_adapter.py:20-29`.
- `VerificationCycleController` dùng `adapted` để cập nhật trạng thái entry và xử lý TTL tại `src/engine/online_tta/verification_cycle.py:21-36`.
- `_verify_and_adapt_entries` chỉ gọi PNN update khi `candidate.adapted` là `True` tại `src/engine/online_tta/online_engine_window_metrics.py:41-78`.

Kết luận: không thêm và xóa trường kết quả verification riêng khỏi desired contract. Giữ `VerificationResult` vì đây là dữ liệu runtime đang được dùng. Giữ chuỗi `pnn_verified` trong compatibility path cho đến khi API update được đổi sang nhận điều kiện từ `pnn_mask` trực tiếp.

### 9. Tham số loss và cấu hình

Code hiện tại có hai điểm không đúng với luồng mong muốn:

1. `hard_old_reconstruction_loss` đang nhận biến `threshold_value`. Trong chuỗi lời gọi hiện tại, biến này trỏ tới `online_point_ewma_threshold`. Loss này phải nhận `input_window_threshold`.
2. A2 đang dùng trực tiếp hệ số contrastive `0.1` trong code. Luồng mong muốn phải lấy hệ số từ `lambda_online_contrastive`.

Bằng chứng: `src/engine/online_tta/online_engine_run.py:202-204`, `src/engine/online_tta/online_engine_window_core.py:253-275` và `src/engine/online_tta/online_engine_step.py:142-168`.

Điều chỉnh cần có:

- Truyền `online_point_ewma_threshold` và `input_window_threshold` bằng hai tham số riêng. Không dùng `threshold_value` cho hai đối tượng có nghĩa khác nhau.
- Thêm `lambda_online_contrastive` vào file cấu hình được chọn làm nơi sở hữu tham số này. Code phải kiểm tra giá trị không âm rồi truyền nó vào `online_update_event`.
- Bỏ nhánh dự phòng A1 tại `src/engine/online_tta/online_engine_step.py:121-131`. Nhánh này cho phép chuỗi điều khiển `pnn_verified` khi không có `pnn_mask`. A1 chỉ được tạo `online_update_event` khi `pnn_mask` có ít nhất một vị trí.
- Giữ `compute_hard_old_hinge_loss` và `compute_masked_pnn_reconstruction_loss` tại `src/engine/online_tta/online_losses.py:42-66`, vì công thức của hai hàm đã đúng.

Các file cấu hình liên quan gồm:

- `configs/task/online_adaptation.yaml` nếu `lambda_online_contrastive` thuộc chính sách online, hoặc `configs/model/online_adaptation.yaml` nếu nó thuộc mục tiêu của model.
- Danh sách trường được phép và code kiểm tra cấu hình tại `src/core/config_model_validation.py:166-178`, `src/core/config_model_validation.py:214-235` và `src/core/config_model_validation.py:774-800`.
- Code tạo cấu hình tại `scripts/benchmarks/generate_online_benchmark_configs.py:80-118`.
- Các file cấu hình thí nghiệm được tạo dưới `configs/experiment/online_benchmark/thesis/`.

SSOT chưa xác định file nào sở hữu `lambda_online_contrastive`. Không được tự xem nó là `lambda_align`, `lambda_proto` hoặc một loss weight của pha offline.

### 10. `online_runtime_state`, checkpoint và khôi phục lần chạy

`online_runtime_state` hiện lưu trường `previous_ewma_score: float | None`, tương ứng với `previous_endpoint_ewma_point_score`. Nó cũng lưu trường toàn cục `signature_history`. Luồng mới cần lưu các vector mong muốn hoặc một cấu trúc tương đương theo absolute index, cùng trạng thái của `verification_buffer` và `hard_old_interval_guard`. Không cần lưu danh sách point đã chốt vì sliding window tự quyết định point nào còn được cập nhật.

Bằng chứng: `src/engine/online_tta/runtime_state.py:13-89` và `documents/spec/full-spec-v3.md:1078-1096`.

Code cần điều chỉnh:

- `OnlineRuntimeState` thay trường EWMA scalar của endpoint bằng schema mới. Schema phải chứa `causal_window.absolute_indices` và đủ dữ liệu để khôi phục `previous_window_ewma_point_scores`.
- Tăng `runtime_schema_version`. Không được đọc trạng thái vector mới như trường scalar của phiên bản cũ.
- `_sync_online_runtime_state` tại `src/engine/online_tta/online_engine_shared.py:64-84` ghi đúng trạng thái mới và không ghi signatures của `causal_window` chưa được thêm vào `verification_buffer`.
- `_run_online_sequence` khôi phục trạng thái từ dữ liệu đã lưu. Nó không được luôn đặt `previous_ewma_score = None` như tại `src/engine/online_tta/online_engine_run.py:267`.
- `_finalize_online_execution` và nhánh hỗ trợ checkpoint cũ trong code bao ngoài benchmark phải cập nhật các key của trạng thái bổ sung tại `src/engine/online_tta/online_engine_run.py:412-436` và `scripts/benchmarks/run_thesis_online_benchmark.py:118-139`.
- Test resume phải so sánh `window_point_predictions` của một lần chạy liên tục với kết quả của lần chạy được lưu rồi khôi phục. Không chỉ kiểm tra việc lưu và đọc lại scalar endpoint.

### 11. `online_event_record`, các chỉ số, tệp kết quả và demo

Mỗi `online_event_record` hiện là một bản ghi chỉ chứa scalar của endpoint. Các trường `raw_point_score`, `ewma_point_score` và `prediction` lần lượt tương ứng với `endpoint_point_score`, `current_endpoint_ewma_point_score` và `endpoint_point_prediction`. Trường `triage_decision` có thể nhận cả vùng triage và chuỗi điều khiển nội bộ `pnn_verified`. Bằng chứng: `src/engine/online_tta/online_engine_step.py:76-101` và `src/engine/online_tta/online_engine_window_metrics.py:54-78`.

Bản ghi mới cần ít nhất các trường sau:

```text
entity_id
point_index
start_index
end_index
window_point_scores
current_window_ewma_point_scores
window_point_predictions
online_point_ewma_threshold
online_variant
triage_region
did_update
online_total_loss
```

`absolute_indices` vẫn thuộc `causal_window`. Nếu cần lưu tensor này trong bản ghi ngoài `start_index` và `end_index`, ontology phải thêm trường đó vào `online_event_record` trước khi triển khai.

Các phần đọc bản ghi cần được sửa:

- `_build_online_window_outputs` và `_finalize_online_execution`;
- `_normalize_online_records` tại `scripts/benchmarks/run_thesis_online_benchmark.py:72-90`. Hàm này không được tự đặt `did_update=True` khi bản ghi thiếu trường.
- Code lưu bản ghi và checkpoint tại `scripts/benchmarks/run_thesis_online_benchmark.py:142-223`.
- `demo/online_replay.py:99-160`, vì demo hiện chuyển các scalar thành một mảng có độ dài bằng số window.
- `demo/demo_state.py:40-57` và `demo/plotting.py`. Hai phần này phải phân biệt kết quả vector theo absolute index của THESIS với schema scalar của baseline.

Pseudocode có `demo_ui_is_enabled`, nhưng online engine hiện không có callback hoặc trường cấu hình cho UI. Demo đang đọc tệp kết quả sau khi lần chạy kết thúc. Nếu cần hiển thị ngay trong vòng lặp online, code phải thêm một callback chỉ đọc event. Nếu chỉ cần phát lại sau khi chạy xong, demo chỉ cần chuyển sang schema bản ghi mới.

### 12. Định danh của protocol và báo cáo

Sau khi code chạy đúng luồng vector mới, báo cáo hoàn tất không nên luôn ghi `runtime_protocol_status = "full_spec_v2"` tại `src/engine/online_tta/online_engine_run.py:487-515`. Giá trị đúng có thể là v3 hoặc một tên mới, nhưng người dùng chưa chốt. Nghiên cứu này không tự chọn giá trị đó.

## Phần code có thể giữ lại

| Phần code | Lý do có thể giữ |
| --- | --- |
| `triage.classify_online_window` | Quy tắc phân loại bốn vùng đã đúng. Chỉ cần bỏ guard khỏi hàm bao ngoài. Bằng chứng: `src/engine/online_tta/triage.py:17-41`. |
| lớp `VerificationBuffer` | `verification_buffer` đã kiểm tra không chồng lấn, giới hạn số `verification_entry`, cờ bắt đầu `verification_cycle` mới và TTL sau mỗi cycle. Bằng chứng: `src/engine/online_tta/verification_buffer.py:7-85`. |
| `verification_adapter.verify_buffer_entries` | Hàm chỉ đọc các `verification_entry` đã lưu và dùng `frozen_source_model`. Hàm đã tạo `known_anomaly_mask`, `continuous_signature_ids`, `recurrent_signature_set` và `pnn_mask`. Bằng chứng: `src/engine/online_tta/verification_adapter.py:53-113`. |
| module `signature_verification` | Code chỉ thêm một signature vào `recurrent_signature_set` khi signature đó xuất hiện trong hơn một `verification_entry` không chồng lấn. Code tạo `pnn_mask` sau khi loại các vị trí thuộc `known_anomaly_mask`. Bằng chứng: `src/engine/online_tta/signature_verification.py:230-277`. |
| `online_optimizer.build_online_optimizer` | Hàm tạo một AdamW mới chỉ quản lý các tham số của `online_mlp_projector`. Bằng chứng: `src/engine/online_tta/online_optimizer.py:22-33`. |
| `compute_hard_old_hinge_loss` | Công thức của `hard_old_reconstruction_loss` đã đúng. Lỗi nằm ở nơi gọi hàm vì nơi đó truyền sai threshold. Bằng chứng: `src/engine/online_tta/online_losses.py:42-44`. |
| `compute_masked_pnn_reconstruction_loss` | Hàm chỉ tính `pnn_reconstruction_loss` tại các vị trí có `pnn_mask = TRUE`. Bằng chứng: `src/engine/online_tta/online_losses.py:57-66`. |
| `OnlineAdaptationModel.forward_source/forward` | Đường chạy A0 đã dùng `source_hidden`. Đường chạy A1/A2 tạo `source_hidden` một lần rồi đưa nó qua `online_mlp_projector` để tạo `projected_hidden`. Bằng chứng: `src/models/online_impl/online_adaptation.py:291-373`. |

## Các test cần sửa

Các test tập trung đã chạy thành công. Tuy nhiên, chúng đang kiểm tra hành vi scalar hiện tại và từng phần riêng lẻ. Chúng chưa kiểm tra toàn bộ thứ tự của luồng mới từ đầu đến cuối.

Lệnh đã chạy:

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

Kết quả: `24 passed in 1.98s`.

Khi bắt đầu triển khai, cần thêm hoặc sửa các test sau:

| Nội dung test | Điều kiện phải kiểm tra | File liên quan |
| --- | --- | --- |
| Absolute indices | Stream và batch tạo đúng các indices trong đoạn `[start,end)` | `test_online_stream.py`, `test_full_spec_online_contract.py` |
| EWMA của window chồng lấn | Hai window chồng lấn cập nhật đúng cùng một point theo absolute index. Point xuất hiện lần đầu dùng trực tiếp score hiện tại | `test_online_ewma_threshold.py` hoặc test mới |
| Prediction theo sliding window | Window sau cập nhật prediction của các point nó vẫn chứa; point không còn xuất hiện giữ prediction gần nhất | Test tích hợp mới cho `_process_online_window` |
| A0 | Không khởi tạo hoặc gọi `online_mlp_projector`; không tạo optimizer, `triage_region` hoặc `verification_entry`; không chạy `verification_cycle` | `test_online_entrypoint.py`, `test_online_tta_variants.py` |
| Thứ tự xử lý | Dùng spy để xác nhận `triage_region` có trước `pnn_mask`, và A2 `online_update_event` cho `hard_old_normality` có trước `verification_cycle` | Test tích hợp mới cho `online_engine_window_core.py` |
| Window hard-old chồng lấn | Khi `hard_old_interval_guard` từ chối, code vẫn giữ `triage_region=hard_old_normality` và không tạo `verification_entry` | `test_online_verification_buffer.py` hoặc test mới cho core |
| `threshold_artifact` và checkpoint | Pha online nạp `threshold_artifact` của pha offline. Nếu hash checkpoint, `window_size` hoặc entity không khớp, code phải dừng trước khi thay đổi trạng thái stream | `test_online_entrypoint.py`, `test_threshold_artifact.py` |
| Tham số loss | `hard_old_reconstruction_loss` nhận `input_window_threshold`; `lambda_online_contrastive` lấy từ cấu hình; A1 không tạo `online_update_event` khi thiếu `pnn_mask` hoặc mask rỗng | `test_full_spec_online_losses.py`, `test_online_tta_variants.py` |
| Kết quả verification | `verification_cycle` trả `VerificationResult` riêng cho từng `verification_entry` | `test_verification_cycle.py` và test tích hợp cho core |
| Lưu và khôi phục lần chạy | Trạng thái EWMA vector theo absolute index, `verification_buffer` và `hard_old_interval_guard` không đổi sau khi lưu rồi khôi phục | `test_online_runtime_state.py` |
| Demo | Demo dựng lại đúng trục thời gian theo absolute index từ bản ghi vector của THESIS, đồng thời vẫn đọc được bản ghi scalar của baseline | `tests/demo/test_demo_state.py` |

## Cấu hình hiện đang dùng

| Cấu hình | Giá trị hiện tại | Bằng chứng | Phạm vi tác động |
| --- | --- | --- | --- |
| `window_size` | `20` | `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml:2` | Stream online và hiệu chỉnh ngưỡng |
| `online_window_stride` | `1` | `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml:7` | Mức chồng lấn của `causal_window` |
| `online_ewma_current_weight` | `0.9` | `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml:10` | Cách code hiện tại tính `current_endpoint_ewma_point_score` |
| `online_ewma_previous_weight` | `0.1` | `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml:11` | Cách code hiện tại tính `current_endpoint_ewma_point_score` |
| `batch_size` | `1` | File cấu hình được tạo tại `scripts/configs/experiment/online_benchmark/thesis/smd__thesis__online__O1_A2__machine_1_6__w20__seed6__main.yaml:17` | Mỗi bước xử lý một `causal_window` |
| Optimizer của `online_update_event` | AdamW, `lr=0.001`, `weight_decay=0.0` | File cấu hình được tạo tại `scripts/configs/experiment/online_benchmark/thesis/smd__thesis__online__O1_A2__machine_1_6__w20__seed6__main.yaml:44-47` | Giá trị trong cấu hình benchmark khác giá trị mặc định khi tạo optimizer mới tại `src/engine/online_tta/online_optimizer.py:22-33` |
| Sức chứa của `verification_buffer` | `8` | `src/engine/online_tta/verification_buffer.py:14`; `src/engine/online_tta/verification_cycle.py:15` | Điều kiện chạy `verification_cycle` |
| Lịch sử của `hard_old_interval_guard` | `max_size=1` | `src/engine/online_tta/online_engine_run.py:218` | Chỉ giữ A2 `online_update_event` gần nhất đã được chấp nhận cho `hard_old_normality` |
| `lambda_online_contrastive` | Code đang ghi trực tiếp giá trị `0.1` | `src/engine/online_tta/online_engine_step.py:168` | A2 `online_update_event` |

## Mâu thuẫn và điểm chưa rõ

### 1. EWMA của point xuất hiện lần đầu

**Đã chốt:** nếu một point theo absolute index chưa có EWMA trước đó, đặt `current_window_ewma_point_scores[t]` bằng đúng `window_point_scores[t]` tại bước thời gian hiện tại. Không nhân điểm số đầu tiên với `ewma_current_weight`. Không xem `previous_window_ewma_point_scores[t]` là `0`.

```text
IF previous_window_ewma_point_scores[t] DOES NOT EXIST THEN
    current_window_ewma_point_scores[t] <- window_point_scores[t]
ELSE
    current_window_ewma_point_scores[t]
        <- ewma_previous_weight * previous_window_ewma_point_scores[t]
           + ewma_current_weight * window_point_scores[t]
ENDIF
```

Quyết định này bổ sung quy tắc khởi tạo còn thiếu vào công thức EWMA theo point trong `full-spec-v2` (`documents/spec/full-spec-v2.md:787-803`). Pseudocode tham chiếu cũ cũng ghi rõ: nếu một point theo absolute index chưa có trạng thái, dùng trực tiếp điểm số hiện tại (`documents/spec/online-tta-thesis-spec.md:898-910`). Vì vậy, pseudocode mong muốn phải bỏ bước tạo `previous_window_ewma_point_scores` toàn số `0`. Code phải kiểm tra trạng thái riêng cho từng point.

### 2. Cập nhật `window_point_predictions` theo sliding window

Một point theo absolute index có thể xuất hiện trong nhiều `causal_window` chồng lấn nhau. Mỗi lần point xuất hiện lại, điểm số mới có thể đổi EWMA và prediction của point đó.

Mỗi khi một point xuất hiện trong `causal_window` hiện tại, hệ thống cập nhật EWMA score, prediction và thông tin hiển thị của point đó theo absolute index. Nếu point xuất hiện lại trong window đến sau, hệ thống tiếp tục cập nhật. Khi point không còn xuất hiện trong sliding window, code không chạm vào point đó nữa và giữ prediction gần nhất.

### 3. A0 không tạo `online_mlp_projector`

**Đã chốt:** A0 không khởi tạo `online_mlp_projector`. Vì module này không tồn tại trong A0, A0 không có các tham số của projector, không tạo optimizer cho projector và không gọi projector khi chạy model. Code tạo model phải kiểm tra `online_variant` và chỉ tạo `online_mlp_projector` cho A1/A2.

### 4. Chỉ giữ `recurrent_signature_set` trong một cycle hay giữ qua nhiều cycle

Chưa rõ code phải giữ `recurrent_signature_set` trong bao lâu. Có hai cách:

- **Chỉ dùng trong một cycle:** Mỗi `verification_cycle` tạo lại set từ các `verification_entry` đang được kiểm tra. Khi cycle kết thúc, code bỏ set này. Cycle sau không dùng kết quả của cycle trước.
- **Giữ qua nhiều cycle:** Code lưu set vào `online_runtime_state` rồi dùng lại trong các `verification_cycle` sau. Nếu chọn cách này, đặc tả phải nói rõ khi nào thêm signature, khi nào xóa signature và xử lý ra sao khi `verification_entry` gốc đã cập nhật model hoặc đã hết TTL.

Luồng mong muốn đang tạo lại `recurrent_signature_set` từ các `verification_entry` hiện có. Tuy nhiên, phần khởi tạo trạng thái có thể khiến người đọc hiểu rằng set được lưu lâu dài. Code hiện tại có trường `signature_history`, nhưng ontology chưa xác nhận đây là alias của `recurrent_signature_set`. `full-spec-v3` cũng chưa nói có giữ signature của `verification_entry` đã có kết luận hoặc đã hết TTL hay không (`documents/spec/full-spec-v3.md:1094-1096`).

**Đã chốt**: Chọn cách đơn giản hơn là chỉ giữ `recurrent_signature_set` trong một cycle.

### 5. Kết quả verification của lần xử lý hiện tại

Một `verification_cycle` có thể kiểm tra nhiều `verification_entry` đã lưu từ các bước thời gian trước. Những entry này có thể không liên quan đến `causal_window` hiện tại.

Không cần thêm một trường kết quả chung vào `online_event_record`. Một `verification_cycle` có thể kiểm tra nhiều `verification_entry` cũ, nên một scalar chung cũng không đại diện đúng cho từng entry.

Code đã trả `VerificationResult` riêng cho từng entry. `VerificationResult` chứa `adapted`, `reason` và `pnn_mask`; các trường này đủ cho update, trạng thái buffer và debug. Nếu cần ghi log, lưu kết quả theo `entry_id` trong dữ liệu của cycle, không tạo thêm object canonical.

### 6. Demo trực tiếp hay phát lại tệp kết quả

Có hai cách hiển thị:

- **Demo trực tiếp:** Ngay khi vòng lặp online tạo `window_point_predictions`, nó gửi kết quả sang UI. Người dùng thấy kết quả mới trong lúc stream đang chạy.
- **Phát lại tệp kết quả:** Pha online chạy xong và lưu báo cáo. Sau đó UI đọc báo cáo rồi phát lại kết quả. UI không nhận prediction trực tiếp từ vòng lặp online.

Luồng mong muốn đặt `SHOW window_point_predictions` bên trong vòng lặp, nên đang mô tả demo trực tiếp. Demo hiện tại phát lại tệp kết quả. Specification chưa chốt phải thêm callback trực tiếp hay tiếp tục phát lại tệp kết quả.

**Đã chốt**: UI nhận kết quả trực tiếp từ vòng lặp online.

## Câu hỏi còn mở

1. `recurrent_signature_set` chỉ tồn tại trong một `verification_cycle` hay được lưu để dùng lại trong các cycle sau? Trả lời: `recurrent_signature_set` chỉ tồn tại trong một `verification_cycle`
2. `lambda_online_contrastive` thuộc cấu hình task/protocol hay cấu hình model? Trả lời: `lambda_online_contrastive` thuộc cấu hình model.
3. Sau khi chuyển sang luồng mới, `runtime_protocol_status` phải là `full_spec_v3` hay một tên định danh mới? Cứ để là `full_spec_v3`. Nếu spec không đồng bộ với pseudocode runtime flow thì cập nhật spec sau.
4. `demo_ui_is_enabled` yêu cầu callback trực tiếp hay cách phát lại tệp kết quả hiện tại đã đủ? Callback trực tiếp nhé.

## Danh mục bằng chứng

- `documents/notes/online_runtime_flow_debug.md:26-268` — pseudocode đầy đủ của luồng cần đạt.
- `documents/spec/online_tta_terminology_ontology.md:153-182` — phân biệt các vector của luồng mong muốn với các scalar endpoint trong code hiện tại.
- `documents/spec/full-spec-v3.md:781-940` — quy tắc dùng lại `source_hidden` và các contract của `window_point_predictions`, `triage_region`, `verification_cycle`, `pnn_mask`, `online_update_event`.
- `documents/spec/full-spec-v3.md:1078-1096` — các trường cần lưu trong `online_runtime_state` để khôi phục lần chạy.
- `src/engine/online_tta/online_engine_run.py:129-315` — cách code hiện khởi động, hiệu chỉnh ngưỡng và giữ trạng thái giữa các window.
- `src/engine/online_tta/online_engine_window_core.py:53-324` — thứ tự xử lý mỗi window và việc đổi `hard_old_normality` thành `gray_zone`.
- `src/engine/online_tta/online_engine_window_metrics.py:33-273` — cách code hiện tính scalar endpoint, tạo `pnn_mask` trước triage, tạo `verification_entry` và trả kết quả.
- `src/engine/online_tta/online_engine_step.py:76-235` — code đang gộp việc tạo bản ghi với cập nhật model, các nhánh A1/A2 và hệ số lambda ghi trực tiếp.
- `src/engine/online_tta/runtime_state.py:13-270` — `previous_endpoint_ewma_point_score` và schema checkpoint của `online_runtime_state`.
- `src/engine/online_tta/verification_adapter.py:53-113` — verification chỉ dùng các entry trong buffer và `frozen_source_model`.
- `scripts/benchmarks/run_thesis_offline_benchmark.py:499-556` — code hiện tạo ngưỡng offline, các ngưỡng triage còn thiếu và thứ tự ghi định danh checkpoint chưa đúng.

## Phạm vi nghiên cứu

Nghiên cứu này chỉ xác định những phần code cần sửa và những quy tắc còn mâu thuẫn. Nó không sửa mã nguồn, file cấu hình, test, tệp kết quả lúc chạy hoặc pseudocode của luồng mong muốn.
