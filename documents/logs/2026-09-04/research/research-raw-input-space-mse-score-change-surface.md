---
date: 2026-09-04T12:51:49+07:00
researcher: OpenAI Codex
topic: raw-input-space MSE anomaly score
status: complete
revision: 974af2b3a3d075f5cd4f3368f2cb584a5a8a3720
branch: dev
---

# Summary

MSE hiện tại nằm trong **normalized input space**. Loader đưa `scaled_sequences` vào `WindowDataset`; model tính sai số giữa `recon` và `batch["x"]`, mà `batch["x"]` đã được standardize.

MSE trong **raw input space** chưa được triển khai. Để thêm nó như một score đơn giản thứ hai cho cả point-level và window-level, cần giữ score normalized hiện tại, inverse-transform input và reconstruction bằng scaler, rồi tính MSE riêng. Không nên dùng lại tên `raw_point_scores`, vì tên này hiện mang nghĩa raw/uncalibrated score nhưng vẫn được tính trên normalized input.

Khuyến nghị tối thiểu: xuất song song hai nhóm score:

- `normalized_input_point_mse`, `normalized_input_window_mse`: score hiện tại.
- `raw_input_point_mse`, `raw_input_window_mse`: score mới, tính trên đơn vị sensor ban đầu.

Nếu score mới được dùng để threshold, prediction, EWMA hoặc triage, phải thêm identity của score space vào threshold artifact và recalibrate trong đúng score space. Không được dùng threshold normalized cho raw-space MSE.

# Research question

Đọc `prompts/1_research_prompt.md` và xác định các phần code cần sửa để thêm một phiên bản MSE trong raw input space cho point-level và window-level anomaly score.

# System context

## Data and normalization

`SequenceStandardScaler` fit mean/std trên training sequence và biến đổi active feature theo `(raw - mean) / std` (`src/data/scalers.py:17-58`). Scaler chưa có inverse-transform method.

`_build_dataset_bundle_from_sequences` giữ cả `raw_sequences` và `scaled_sequences`, nhưng dataset/window loader chỉ lấy `scaled_sequences` (`src/data/loaders.py:150-193`, `src/data/loaders.py:231-298`). Checkpoint đã lưu `scaler_state_dict` (`src/engine/checkpoint.py:185-215`), nên không cần thay checkpoint serialization; cần truyền scaler tới nơi tính score.

## Current model score

Model tính:

```text
point_mse = mean((reconstruction - batch["x"])**2, dim=feature)
window_mse = mean(point_mse, dim=time)
```

Các phép tính nằm ở `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:186-196` và deterministic path ở `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py:241-260`. Vì `batch["x"]` là scaled input, đây là normalized-space MSE.

Với MC reconstruction, code hiện tại tính mean của per-sample MSE (`point_score_samples.mean(dim=1)`), phù hợp contract tại `documents/spec/full-spec-v3.md:517-554`. Raw-space implementation cũng phải giữ quy tắc này, không đổi thành MSE của mean reconstruction.

# Execution path

```text
raw sequence
  -> SequenceStandardScaler
  -> scaled sequence / WindowDataset
  -> model reconstruction
  -> normalized point/window MSE
  -> evaluator or online engine
  -> threshold, EWMA, prediction, artifacts
```

Raw-space branch cần thêm tại sau model reconstruction:

```text
scaled input + reconstruction + scaler
  -> inverse-transform both tensors
  -> raw point MSE
  -> mean over time for raw window MSE
```

Với synthetic validation, phải inverse-transform batch sau khi synthetic anomaly đã được inject vào scaled `batch["x"]`. Không lấy raw sequence trước injection để tính score, vì như vậy sẽ bỏ qua anomaly nhân tạo.

# Required change surface

| Mức | File | Thay đổi cần thiết |
|---|---|---|
| P0 | `src/data/scalers.py:36-58` | Thêm inverse-transform tensor helper, giữ đúng active/inactive feature mask và `epsilon`. Thêm test round-trip. |
| P0 | Model scoring seam: `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:186-250`, `.../thesis_multitask_routing_forward_helpers.py:241-279` | Tách rõ normalized MSE và raw-input MSE. Với MC, inverse-transform từng `reconstruction_samples` rồi lấy mean per-sample MSE. Có thể giữ model output hiện tại và thêm score vào `aux` nếu chỉ cần diagnostic song song. |
| P0 | `src/engine/evaluator.py:92-264`, `:415-554` | Mở rộng payload, overlap aggregation, point records và metric extraction để giữ hai point-score timelines; evaluator cần nhận scaler hoặc nhận raw scores đã được tính. |
| P0 | `scripts/benchmarks/run_thesis_offline_benchmark.py:310-362`, `:438-515`, `:624-715` | Giữ scaler trong artifact inputs; xuất riêng raw point/window score và provenance score space cho clean validation, synthetic validation và test. |
| P0 | `src/data/stream.py:113-163` | Online window hiện chỉ chứa scaled `x`. Cần để scorer inverse-transform scaled window/reconstruction bằng scaler; không nhất thiết đưa raw `x` vào model batch. |
| P0 | `src/engine/online_tta/online_calibration.py:61-214` | Thêm raw point/window MSE vào calibration/collection path. `input_scores` hiện tại cũng là normalized-space MSE vì dùng `batch["x"]`. |
| P0 | `src/engine/online_tta/online_engine_window_metrics.py:81-204`, `:207-277` | Xuất raw-space point/window fields riêng; xác định score nào được dùng cho EWMA, admission, threshold và prediction. Tránh tiếp tục dùng tên mơ hồ `raw_window_point_scores`. |
| P1 | `src/engine/online_tta/online_engine_run.py:103-194`, `:318-371`, `:529-537`; `online_engine_step.py:36-100` | Truyền scaler/score-space identity qua runtime context và chọn đúng score cho event/step record nếu raw score là score vận hành. |
| P1 | `src/protocols/threshold_artifact.py:21-93`, `:282-430` | Thêm `score_space` và definition/version cho score. Nếu raw MSE điều khiển prediction, tạo threshold raw-space riêng và reject mismatch. Schema hiện tại còn yêu cầu trường sigmoid-specific. |
| P1 | Config protocol, hiện là `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml` | Thêm lựa chọn score space chỉ khi cần dùng raw score làm score vận hành. Nếu chỉ xuất diagnostic song song, không cần thêm flag; artifact vẫn phải ghi score identity. |
| P1 | Tests | Thêm test inverse-transform, hand-computed raw point/window MSE, MC averaging, synthetic-after-injection, evaluator overlap, online EWMA selection và artifact score-space mismatch. Cập nhật shape/contract/export tests hiện có. |

# Terminology mapping

| Tên hiện tại | Ý nghĩa hiện tại | Tên nên dùng cho change này | Trạng thái |
|---|---|---|---|
| `raw_point_scores` | Raw/uncalibrated score; input vẫn normalized | `normalized_input_point_mse` | Giữ semantic, nên làm rõ tên ở artifact/report |
| `window_scores` | Window reconstruction MSE trên `batch["x"]` normalized | `normalized_input_window_mse` | Giữ semantic nếu chạy song song |
| chưa có | MSE trên đơn vị sensor ban đầu | `raw_input_point_mse` | Object mới |
| chưa có | Mean theo time của raw point MSE | `raw_input_window_mse` | Object mới |
| `input_window_score` | Full-window MSE hiện tính normalized | `normalized_input_window_mse` hoặc giữ tên cũ kèm `score_space` | Cần làm rõ, không được gán mặc định là raw |
| `point_scores` / `window_scores` | Stable top-level model contract; point score có thể qua calibration theo spec | Giữ nếu raw score chỉ diagnostic; nếu thay semantic thì là breaking protocol change | Cần quyết định |

# Implemented, configured, tested, documented, inferred

- **Implemented:** scaler transform; raw/scaled sequence bundle; checkpoint scaler state; normalized point/window reconstruction MSE; MC mean-of-per-sample-MSE; offline and online consumers của một score timeline.
- **Configured:** clean-validation threshold protocol và EWMA; chưa thấy score-space selector riêng trong protocol config.
- **Tested:** hiện có shape, finite-score, MC aggregation, overlap, scaler-state và online EWMA tests; chưa có test cho raw-input-space MSE.
- **Documented:** spec mô tả point/window MSE và hiện gọi `point_scores` là official transformed score (`documents/spec/full-spec-v3.md:556-620`, `:899-909`). Spec chưa định nghĩa rõ “raw input space” là units sensor ban đầu hay chỉ raw trước calibration.
- **Inferred:** inverse-transform reconstruction và input ngay trước MSE là cách phù hợp để giữ synthetic anomaly sau injection và không thêm raw tensor vào model contract. Cần kiểm chứng bằng test.

# Conflicts and uncertainties

1. `documents/spec/full-spec-v3.md` vẫn khóa sigmoid calibration cho official THESIS point score (`:1500+`), trong khi yêu cầu hiện tại muốn simple MSE. Nếu raw-space MSE chỉ là score bổ sung, có thể giữ official contract và xuất thêm field. Nếu nó thay thế score vận hành, phải cập nhật spec, config, threshold artifact và prediction path cùng lúc.

2. “Raw” đang bị dùng cho hai nghĩa: uncalibrated và unscaled sensor units. Change này cần dùng tên có `input_space` để tránh nhầm.

3. MSE theo raw sensor units có thể bị chi phối bởi channel có độ lớn vật lý lớn. Đây là hệ quả toán học của raw MSE, không phải lỗi triển khai; nếu muốn weighting thì đó là một score definition khác và cần đặc tả riêng.

# Open questions

- Raw-space MSE chỉ để diagnostic/ablation, hay sẽ dùng cho threshold, prediction, EWMA và triage?
- `point_scores`/`window_scores` có giữ normalized semantic hiện tại và thêm field mới, hay đổi sang raw-space semantic?
- Raw MSE có dùng toàn bộ channel với trọng số bằng nhau không?
- Khi `reconstruction_samples` không được export, scorer có cần giữ raw MC score trước khi loại samples không?

# Evidence

- Prompt and workflow: `prompts/1_research_prompt.md`.
- Data/scaler: `src/data/scalers.py:17-87`; `src/data/loaders.py:150-193`, `:231-298`; `src/data/api.py:65-74`.
- Model score: `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:186-250`; `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py:241-279`.
- Contracts: `src/core/contracts.py:66-105`, `:121-223`.
- Offline evaluation/export: `src/engine/evaluator.py:122-264`, `:415-554`; `scripts/benchmarks/run_thesis_offline_benchmark.py:310-362`, `:500-715`.
- Online evaluation: `src/engine/online_tta/online_calibration.py:61-214`; `src/engine/online_tta/online_engine_window_metrics.py:81-277`; `src/engine/online_tta/online_engine_run.py:318-371`.
- Score/threshold protocol: `src/protocols/point_scores.py:32-92`; `src/protocols/threshold_artifact.py:21-93`, `:282-430`; `documents/spec/full-spec-v3.md:517-620`, `:819-909`.

# Validation performed

Đã đọc research prompt trước, kiểm tra revision/branch/worktree và truy vết data flow offline/online. Chưa sửa production code và chưa chạy test vì yêu cầu hiện tại là reconnaissance/change-surface research.
