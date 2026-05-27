# Detailed Implementation Plan: RedLamp MLP Baseline Gradient Conflict Profiling

## Phase 0. Scope Lock And Contract Freeze

### Phase summary
Phase này khóa phạm vi theo mục tiêu thesis hiện tại: kiểm chứng gradient conflict trên `RedLampMLPBaseline` bằng cơ chế profiling không làm đổi trajectory tối ưu hóa mặc định. Mục tiêu là tạo đường đo đáng tin cậy trước khi xem xét can thiệp optimizer.

### File-level edits
- Không chỉnh sửa mã nguồn ở phase này.
- Tài liệu tham chiếu chính:
  - `documents/logs/05-27-2026/plan/plan-redlamp-mlp-baseline-gradient-conflict-profiling.md`
  - `src/models/redlamp_mlp_baseline.py`
  - `src/engine/trainer.py`

### Interface and contract definitions
- Batch contract giữ nguyên qua `validate_batch(batch)`:
  - `batch["x"]: Tensor[B, L, D]`
  - `classification_labels`, `synthetic_anomaly_mask`, `augmentation_metadata` tiếp tục do injector chuẩn bị.
- Encoder contract (baseline hiện tại):
  - `hidden = encoder(x)` với shape `[B, L, latent_dim]`.
- Model output contract giữ nguyên qua `validate_model_outputs(outputs)`:
  - `hidden`, `pooled`, `recon`, `logits`, `point_scores`, `window_scores`, `aux`.
- Engine contract:
  - `training_step(...)` trả về dict có `loss`, `log`, `outputs`, `loss_terms`, `batch`.
  - Profiling metrics sẽ được nối vào `log` mà không thay đổi key cũ.

### Design pattern application
- Composition over inheritance: thêm helper profiling nội bộ trong `RedLampMLPBaseline`, không tạo cây class mới.
- Adapter pattern: không thay đổi adapter datasets/encoders hiện tại; profiling bám trực tiếp encoder parameter tensors.
- Strategy pattern: chế độ profiling bật/tắt qua config flag như một strategy quan sát.
- Registry/factory: giữ nguyên registry model hiện có, chỉ mở rộng tham số config model.

### Risk mitigation mapping
- Prototype redundancy: không áp dụng ở baseline MLP (không có prototype branches); ghi rõ out-of-scope để tránh nhiễu diễn giải.
- Fusion collapse: không áp dụng trực tiếp (baseline không dùng fusion gate); thay bằng theo dõi dominance gradient giữa CE và MSE.
- Adaptation contamination: không áp dụng (không có online adaptation); giữ synthetic pipeline như cũ để nhất quán dữ liệu.
- Projector drift: không áp dụng (không có projector).
- Metric inflation: giữ nguyên evaluator thresholding `q=0.95`, không đổi đường tính anomaly score.

### Acceptance criteria
- Tài liệu detail thống nhất với plan baseline đã chốt.
- Không có thay đổi contract công khai của batch/model outputs/trainer.

---

## Phase 1. Add Profiling Configuration Surface

### Phase summary
Tạo bề mặt cấu hình tối thiểu, rõ ràng, để bật/tắt gradient profiling và cấu hình smoothing/logging mà không phát sinh nhiều codepath.

### File-level edits
- `src/models/redlamp_mlp_baseline.py`
  - Mở rộng `__init__` với nhóm tham số profiling:
    - `enable_gradient_conflict_profiling: bool = False`
    - `gradient_profiling_scope: str = "encoder_all"`
    - `gradient_focus_layer_name: str = "encoder_last_linear"`
    - `gradient_log_every_n_steps: int = 1`
    - `gradient_ema_alpha: float = 0.1`
    - `gradient_sma_window: int = 50`
    - `gradient_profile_include_bias: bool = False`
- File config baseline tương ứng trong `configs/model/` hoặc `configs/experiment/`
  - Thêm các trường trên với default bảo toàn backward compatibility.

### Explicit edit content
- Validate tham số trong `__init__`:
  - `0 < gradient_ema_alpha <= 1`
  - `gradient_sma_window >= 1`
  - `gradient_log_every_n_steps >= 1`
  - `gradient_profiling_scope in {"encoder_all"}` cho vòng đầu.
- Khởi tạo state buffer trong model:
  - step counter profiling.
  - dictionary lưu lịch sử raw metric theo layer-key.
  - dictionary EMA state theo metric-key.
  - deque/circular buffer phục vụ SMA.

### Interface and contract definitions
- Không thay đổi output contract bắt buộc.
- Thêm contract nội bộ cho metric keys, ví dụ:
  - `train_gradconf_raw/<layer>/cosine_sim`
  - `train_gradconf_ema/<layer>/cosine_sim`
  - `train_gradconf_sma/<layer>/cosine_sim`
  - tương tự cho `r_ratio`, `norm_ce`, `norm_mse`, `norm_total`.

### Design pattern application
- Single responsibility: config parsing + state initialization tách khỏi tính toán gradient.
- Least codepaths: một cờ global `enable_gradient_conflict_profiling` bao toàn bộ path đo.

### Risk mitigation mapping
- Metric inflation risk: đặt tên metric tách namespace `gradconf/*` để không trùng evaluator metric chính.

### Acceptance criteria
- Khi flag profiling tắt: hành vi mô hình và log keys cũ không đổi.
- Khi profiling bật: model khởi tạo được với cấu hình mặc định mới.

---

## Phase 2. Implement Layer Enumeration And Gradient Vector Utilities

### Phase summary
Cài các helper low-level để trích đúng tensor gradient theo layer encoder, làm nền cho cosine similarity và R-ratio.

### File-level edits
- `src/models/redlamp_mlp_baseline.py`
  - Thêm helper methods:
    - `_get_encoder_profiled_parameters(...)`
    - `_extract_layerwise_gradients(...)`
    - `_flatten_tensor_for_metrics(...)`
    - `_compute_cosine_similarity(...)`
    - `_compute_preservation_ratio(...)`

### Explicit edit content
- Layer enumeration rule (vòng đầu):
  - Chỉ lấy tham số `weight` của các `nn.Linear` thuộc `self.encoder`.
  - Bỏ `bias` nếu `gradient_profile_include_bias=False`.
- Focus layer rule:
  - map `encoder_last_linear` -> layer linear cuối cùng của encoder MLP.
- Numerical stability:
  - dùng `epsilon=1e-12` khi chuẩn hóa norm để tránh chia 0.
- Metric formula implementation:
  - `cosine = dot(g_ce, g_mse) / (||g_ce|| * ||g_mse|| + eps)`
  - `r_ratio = ||g_total|| / (||g_ce|| + ||g_mse|| + eps)`

### Interface and contract definitions
- Contract internal cho layer map:
  - `OrderedDict[str, nn.Parameter]` với key ổn định qua các step.
- Contract tensor shape:
  - gradient vector mỗi layer phải flatten thành 1D trước khi tính dot/norm.

### Design pattern application
- Composition: helper functions thuần toán học, không dính logic train loop.
- Stable interfaces: keys layer cố định, giúp logging và test deterministic.

### Risk mitigation mapping
- Fusion collapse proxy: theo dõi `norm_ce` vs `norm_mse` ratio để phát hiện dominance dài hạn.

### Acceptance criteria
- Với input gradient giả lập, helper trả cosine và R-ratio finite, đúng range kỳ vọng (`cosine` trong `[-1,1]`, `r_ratio` trong `[0,1]` với vector hữu hạn).

---

## Phase 3. Integrate Diagnostic Gradient Profiling In Training Path

### Phase summary
Nhúng logic đo gradient conflict vào training path của baseline theo hướng “observe-only”: đo riêng gradient từng objective nhưng không đổi update trajectory production.

### File-level edits
- `src/models/redlamp_mlp_baseline.py`
  - Thêm method nội bộ, ví dụ:
    - `_profile_encoder_gradient_conflict(...)`
  - Mở rộng `training_step(...)`/`_shared_step(...)` để ghi metric khi stage=`train` và flag bật.
- `src/engine/trainer.py` (chỉ nếu cần)
  - đảm bảo logger nhận toàn bộ key động từ `step_result["log"]`.

### Explicit edit content
- Quy trình đo cho mỗi iteration được profile:
  1. Forward một lần để lấy `reconstruction_loss`, `classification_loss`.
  2. Tạo weighted losses:
     - `loss_ce_weighted = lambda_cls * classification_loss`
     - `loss_mse_weighted = reconstruction_loss`
  3. Dùng `torch.autograd.grad` để lấy `g_ce` và `g_mse` trên cùng parameter list encoder (không đụng `.grad` global nếu chưa cần).
  4. Tính `g_total = g_ce + g_mse` theo layer.
  5. Tính raw metrics layer-wise + focus-layer metrics.
  6. Cập nhật EMA/SMA state và append vào `log`.
- Optimization trajectory preservation:
  - `training_step` vẫn trả `loss = reconstruction_loss + lambda_cls * classification_loss` y như hiện trạng.
  - Optimizer backward/step chính vẫn đi qua loss chuẩn của trainer.

### Interface and contract definitions
- `training_step` output contract không đổi, chỉ bổ sung key log.
- Không thay đổi `loss_terms` keys hiện hữu.

### Design pattern application
- Strategy pattern: profiling path là strategy chẩn đoán tùy chọn.
- Separation of concerns: train objective và diagnostic objective tách rõ.

### Risk mitigation mapping
- Adaptation contamination/projector drift: không liên quan; ghi rõ non-applicable.
- Metric inflation: diagnostic metrics không tham gia early stopping/checkpoint monitor mặc định.

### Acceptance criteria
- Profiling bật không làm crash backward do graph reuse.
- Loss train chính không đổi công thức so với baseline ban đầu.
- Metrics layer-wise + focus-layer xuất hiện đều đặn theo `gradient_log_every_n_steps`.

---

## Phase 4. Smoothing, Logging Schema, And Artifact Traceability

### Phase summary
Chuẩn hóa hệ thống lưu metric raw + EMA + SMA để đọc xu hướng dài hạn và kiểm tra độ ổn định ngắn hạn.

### File-level edits
- `src/models/redlamp_mlp_baseline.py`
  - Thêm helpers:
    - `_update_ema(metric_key, value)`
    - `_update_sma(metric_key, value)`
    - `_build_gradient_conflict_log_dict(...)`
- (Tùy chọn) `src/engine/logger.py` nếu cần normalize naming.

### Explicit edit content
- EMA update:
  - `ema_t = alpha * x_t + (1 - alpha) * ema_{t-1}`
- SMA update:
  - rolling mean cửa sổ `window=50`.
- Log cả 2 mức:
  - per-layer metrics.
  - focus-layer metrics (đặt prefix riêng `focus/`).

### Interface and contract definitions
- Metric naming contract cố định để dashboard dễ lọc.
- Giá trị metric log phải là Python float.

### Design pattern application
- Single responsibility: smoothing state nằm trong module profiling helper, không rải trong `_shared_step`.

### Risk mitigation mapping
- Metric inflation: tách raw vs smoothed rõ ràng để tránh diễn giải nhầm.

### Acceptance criteria
- Mỗi metric có đủ 3 biến thể: raw, ema, sma.
- EMA dùng `alpha=0.1` và SMA dùng `window=50` đúng theo quyết định đã chốt.

---

## Phase 5. Tests And Validation

### Phase summary
Bổ sung test tối thiểu nhưng đủ chặt để xác nhận correctness toán học, compatibility contract và tính ổn định runtime.

### File-level edits
- `tests/test_redlamp_gradient_conflict_metrics.py`
  - unit tests cho cosine/R-ratio helper.
  - unit tests EMA/SMA update logic.
- `tests/test_redlamp_baseline_with_gradient_profiling_step.py`
  - integration smoke test 1 train step với profiling bật.
- Cập nhật test config fixtures nếu cần.

### Explicit edit content
- Unit test 1:
  - tạo vector gradient thủ công, so giá trị cosine/R-ratio với tính tay.
- Unit test 2:
  - kiểm tra EMA recursion và SMA rolling window đúng trên chuỗi số nhỏ.
- Integration test:
  - tạo batch synthetic nhỏ.
  - gọi `training_step`, chạy backward/step 1 lượt.
  - assert có key metric expected trong `log`.
  - assert tất cả metric finite (`torch.isfinite` / `math.isfinite`).

### Interface and contract definitions
- Test phải xác nhận output contract cũ còn nguyên keys bắt buộc.

### Design pattern application
- Minimal vertical slice testing: unit + 1 integration step theo nguyên tắc codebase.

### Risk mitigation mapping
- Evaluation metric inflation: thêm assert rằng evaluator threshold config không bị chỉnh sửa bởi profiling changes.

### Acceptance criteria
- Toàn bộ test mới pass bằng `pytest`.
- Không phát sinh regression ở test baseline hiện hữu.

---

## Phase 6. Experiment Readiness Checklist

### Phase summary
Chốt checklist trước khi chạy experiment dài để bảo đảm tái lập và diễn giải đúng.

### File-level edits
- `documents/logs/05-27-2026/detail/detail-redlamp-mlp-baseline-gradient-conflict-profiling.md` (mục checklist này).
- (Tùy chọn) thêm run note trong `documents/logs/05-27-2026/research/` sau khi chạy thật.

### Validation steps
- Chạy preflight config và in resolved config để xác nhận:
  - profiling flag bật.
  - `ema_alpha=0.1`, `sma_window=50`, `q=0.95` giữ nguyên.
- Chạy train ngắn (ví dụ vài trăm steps) để kiểm tra logging throughput.
- Kiểm tra dashboard/log:
  - focus-layer cosine và R-ratio có trajectory đọc được.
  - raw nhiễu nhưng EMA mượt hơn SMA/hoặc tương thích xu hướng.

### Measurable acceptance criteria
- `train_gradconf_raw/focus/cosine_sim` được log ở >= 95% số step đáng lẽ phải log.
- `train_gradconf_ema/focus/cosine_sim` và `train_gradconf_sma/focus/cosine_sim` tồn tại đồng thời.
- Không thay đổi công thức `total_loss` baseline và không đổi evaluator threshold quantile (`q=0.95`).
- Không có lỗi OOM hoặc autograd graph reuse trong smoke run cấu hình chuẩn.

---

## Programming Order (Execution Blueprint)
1. Chỉnh `src/models/redlamp_mlp_baseline.py` để thêm config + state + helper toán học (Phase 1-2).
2. Tích hợp profiling observe-only vào training path (Phase 3).
3. Thêm smoothing + logging schema hoàn chỉnh (Phase 4).
4. Viết unit/integration tests (Phase 5).
5. Chạy pytest và smoke validation, sau đó ghi nhận kết quả (Phase 6).

## Notes on Out-of-Scope Items
- Manual `.grad` injection cho toàn encoder scope chưa triển khai trong bản này; đó là bước kế tiếp chỉ khi evidence conflict đủ mạnh.
- Kiến trúc CNN encoder chuẩn REDLAMP không thuộc baseline MLP phase này; sẽ xử lý ở luồng mô hình khác.
