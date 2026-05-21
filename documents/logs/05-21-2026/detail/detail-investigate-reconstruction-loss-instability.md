---
date: 2026-05-21 18:25:00 +07
author: Codex
git_commit: 84df6bee3dc9314ed462f983b4efa3bae4590d72
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed plan for implementation and experiment execution: reconstruction loss instability diagnosis"
tags: [detail, reconstruction-loss, diagnostics, cka, thesis-multitask]
status: proposed
last_updated: 2026-05-21
last_updated_by: Codex
source_plan: documents/logs/05-21-2026/plan/plan-investigate-reconstruction-loss-instability.md
source_research: documents/logs/05-21-2026/research/research-current-codebase-status-before-planning-reconstruction-loss-and-cka.md
---

# Detailed Plan: Reconstruction Loss Instability Diagnosis

## 1. Thesis-Aligned Objective
Mục tiêu triển khai là xác định nguyên nhân chính gây dao động mạnh của `reconstruction_loss` trong pipeline `thesis_multitask` mà không phá vỡ các contract hiện có của codebase. Kế hoạch ưu tiên một vertical slice quan sát được, tái lập được, và ablation-friendly, phù hợp với định hướng trong `documents/design/idea.md`, `documents/design/design_starter.md`, và `codebase_preferences.md`.

## 2. Contract Definitions to Preserve

### 2.1 Batch Contract (Dataset -> Model)
`batch` phải tiếp tục hỗ trợ các trường sau:
- `x: Tensor[B, L, D]`
- `point_labels: Tensor[B, L]` (hoặc tương đương hiện hành)
- `mask`, `timestamps`, `meta`
- synthetic fields khi bật augmentation: `synthetic_anomaly_mask`, `classification_labels`, `augmentation_metadata`

### 2.2 Encoder Contract
Model phải tiếp tục xuất hidden representation theo thesis-facing shape `[B, L, H]` trong forward output (không đổi semantic contract).

### 2.3 Model Output Contract (Model -> Engine)
Giữ nguyên các keys đang được trainer/evaluator sử dụng (`recon`, `logits`, `point_scores`, `aux`, loss dict/stage log dict hiện hành). Diagnostics mới chỉ được thêm dưới dạng metric bổ sung.

### 2.4 Engine Contract
Trainer/logger/checkpoint monitor không đổi điều kiện hoạt động hiện tại, đặc biệt monitor `val_synth_vus_pr` và checkpoint save logic.

## 3. Design Pattern Application in This Task
- **Composition over inheritance**: bổ sung diagnostics bằng helper functions nội bộ trong `thesis_multitask.py` và aggregation helper trong `trainer.py`, không mở rộng class hierarchy mới.
- **Adapter pattern for encoders**: không thay đổi adapter/encoder interface; diagnostics chỉ đọc output contract từ model.
- **Strategy pattern for tasks**: giữ phân tách stage strategy hiện hành (`train`, `val`, `val_synth`, `test`), chỉ thêm metrics theo stage.
- **Registry/factory for datasets and models**: không thay đổi registry contract; mọi run vẫn khởi tạo qua config + registry path hiện hành.

---

## 4. Phase-by-Phase Implementation

## Phase 0: Reproducibility Gate and Baseline Snapshot
### Phase Summary
Thiết lập baseline có thể tái lập trước khi chèn instrumentation, để mọi kết luận hậu kiểm đều có đối chứng rõ ràng.

### File-Level Edits
- `documents/logs/05-21-2026/detail/detail-investigate-reconstruction-loss-instability.md` (tài liệu này): bổ sung command và tiêu chí baseline.
- Không sửa mã nguồn trong phase này.

### Explicit Execution Content
1. Chọn một experiment config baseline duy nhất (ưu tiên config đã dùng ở research 2026-05-21).
2. Chạy baseline ngắn 5-10 epoch.
3. Lưu lại:
   - resolved config path,
   - output directory,
   - metrics JSONL path,
   - checkpoint path (nếu có).

### Acceptance Criteria
- Có một baseline run ID duy nhất với metadata đầy đủ.
- Có thể tái chạy baseline bằng cùng command và ra cùng cấu trúc artifact.

---

## Phase 1: Diagnostics Configuration Contract
### Phase Summary
Thêm configuration contract để bật/tắt diagnostics mà không tăng số codepath ẩn.

### File-Level Edits
- `src/core/config.py`
- `configs/experiment/smd_thesis_multitask_redlamp_multiclass_window20_exp2_small_100ep.yaml` (hoặc một config diagnosis riêng trong cùng thư mục)

### Explicit Edit Content
1. Thêm nhóm config diagnostics (ví dụ trong `logging` hoặc `task` block, theo kiến trúc hiện tại):
   - `enable_reconstruction_diagnostics: bool = false`
   - `diagnostics_log_interval_steps: int = 1`
   - `diagnostics_include_grad_norm: bool = false`
2. Trong `src/core/config.py`:
   - inject defaults nếu keys vắng mặt,
   - validate kiểu và miền giá trị (`interval >= 1`).
3. Cập nhật config experiment diagnosis để bật diagnostics có chủ đích.

### Interface Notes
- Không thay đổi key cũ.
- Key mới phải backward-compatible với mọi config cũ.

### Acceptance Criteria
- Config cũ chạy bình thường khi không có keys mới.
- Config sai kiểu hoặc sai miền bị fail sớm với thông báo rõ ràng.

---

## Phase 2: In-Model Reconstruction Diagnostics
### Phase Summary
Gắn instrumentation vào reconstruction path để đo trực tiếp nguồn dao động ở cấp batch.

### File-Level Edits
- `src/models/thesis_multitask.py`

### Explicit Edit Content
1. Trong `_compute_reconstruction_loss`:
   - tính và trả thêm diagnostics payload:
     - `recon_mse_mean`
     - `recon_mse_std`
     - `active_normal_cells`
     - `normal_cell_ratio`
     - `synthetic_cell_ratio`
     - `fallback_to_full_mse_flag`
2. Trong `_shared_step`:
   - merge payload vào stage log dict với prefix chuẩn, ví dụ `diag/recon/...`.
3. Nếu bật `diagnostics_include_grad_norm`:
   - thêm hook/helper đo grad norm dạng read-only (`detach`), không sửa gradient.
4. Viết comment ngắn ở các block mới để giải thích mục đích chẩn đoán.

### Interface Notes
- Giá trị loss trả về không đổi công thức.
- Output contract cho evaluator giữ nguyên shape và key bắt buộc.

### Acceptance Criteria
- Khi diagnostics bật: metrics mới xuất hiện trong stage logs.
- Khi diagnostics tắt: metrics mới không xuất hiện.
- Numerical path của `L_total` giữ nguyên so với trước ở cùng input/seed.

---

## Phase 3: Trainer Aggregation for Instability Indicators
### Phase Summary
Chuẩn hóa thống kê cấp epoch để chuyển từ quan sát batch rời rạc sang bằng chứng định lượng ổn định.

### File-Level Edits
- `src/engine/trainer.py`

### Explicit Edit Content
1. Mở rộng aggregation logic để tính cho `diag/recon/*`:
   - mean, std, min, max, p90, p95.
2. Tạo indicators cấp epoch:
   - `diag/recon/recon_loss_cv`
   - `diag/recon/recon_loss_p95_to_p50`
3. Tách logs theo stage rõ ràng (`train`, `val`, `val_synth`) để tránh nhiễu diễn giải.

### Interface Notes
- Không thay đổi luồng scheduler/monitor/checkpoint.
- Không can thiệp vào evaluator output path.

### Acceptance Criteria
- JSON metrics cấp epoch chứa đủ indicators mới.
- Checkpoint selection theo `val_synth_vus_pr` không thay đổi hành vi.

---

## Phase 4: Logger Routing and Naming Policy
### Phase Summary
Đảm bảo metrics diagnostics đi qua toàn bộ logging sink một cách nhất quán và dễ truy vấn.

### File-Level Edits
- `src/engine/logger.py`

### Explicit Edit Content
1. Bảo đảm metrics mới được ghi đầy đủ vào JSONL.
2. Nếu `use_wandb=true`, metrics mới phải được mirror lên W&B cùng cadence.
3. Chuẩn hóa namespace metrics:
   - `diag/recon/*`
   - `diag/cka/*`
   - `diag/grad/*`
4. Ghi run metadata tĩnh ở đầu run để hỗ trợ so sánh công bằng.

### Acceptance Criteria
- JSONL và W&B có cùng tập metric diagnostics cho một epoch.
- Dashboard filter theo prefix hoạt động nhất quán.

---

## Phase 5: Experiment Matrix Execution
### Phase Summary
Thực thi ma trận ablation tối thiểu để cô lập từng nhóm giả thuyết nguyên nhân.

### File-Level Edits
- Các file config experiment tương ứng trong `configs/experiment/`.
- Có thể thêm config variants mới nếu cần, đặt tên rõ và nhất quán.

### Explicit Edit Content
Tối thiểu 4 runs với seed và data config cố định:
1. `reconstruction_normal_only=false`, synthetic on.
2. `reconstruction_normal_only=true`, synthetic on.
3. `reconstruction_normal_only=true`, synthetic off.
4. `reconstruction_normal_only=true`, synthetic on, CKA-gated fusion off.

Run policy:
- Smoke diagnostics run ngắn trước.
- Chỉ kéo dài run có tín hiệu phân biệt rõ.

### Acceptance Criteria
- Có đủ artifact cho cả 4 runs.
- Mỗi run có bảng metrics instability tương ứng và so sánh được trực tiếp.

---

## Phase 6: Analysis, Conclusion, and Next-Action Gate
### Phase Summary
Kết luận nguyên nhân theo bằng chứng đo được và quyết định can thiệp thuật toán chỉ khi đủ điều kiện.

### File-Level Edits
- Tạo research log mới trong `documents/logs/05-21-2026/research/` để tổng hợp kết quả.

### Explicit Edit Content
1. So sánh `recon_loss_cv`, `p95/p50`, `spike frequency` theo 4 runs.
2. Ánh xạ kết quả vào 4 nhóm nguyên nhân:
   - mask sparsity/fallback,
   - synthetic intensity,
   - fusion/CKA dynamics,
   - optimizer/gradient dynamics.
3. Chỉ đề xuất sửa thuật toán khi có nguyên nhân trội được xác nhận.

### Acceptance Criteria
- Có kết luận nguyên nhân ưu tiên xếp hạng theo mức độ bằng chứng.
- Có danh sách next actions gắn trực tiếp với nguyên nhân đã xác nhận.

---

## 5. Risk Mitigation Matrix (Required)

1. **Prototype redundancy risk** (continuous/discrete trùng vai trò):
- Theo dõi thêm tương quan nhánh và usage metrics theo epoch.
- Nếu redundancy cao, chuẩn bị ablation tắt từng nhánh trong run tiếp theo.

2. **Fusion collapse risk** (alpha/beta nghiêng cực đoan):
- Giám sát `alpha`, `beta`, `alpha_std`, `beta_std` đồng thời với `diag/recon/*`.
- Đặt cảnh báo khi phân phối trọng số sát biên trong nhiều epoch liên tiếp.

3. **Adaptation contamination risk**:
- Dù scope hiện tại là offline diagnosis, vẫn giữ tách clean/synthetic rõ theo stage để tránh contamination logic vào kết luận.

4. **Projector drift risk**:
- Không bật cơ chế adaptation/projector tuning trong ma trận diagnosis hiện tại.
- Nếu cần mở rộng sang online phase, tách thành plan độc lập.

5. **Evaluation metric inflation risk**:
- Giữ monitor metric hiện tại, không đổi thresholding/evaluation protocol giữa các runs.
- Báo cáo đồng thời instability metrics và task metrics để tránh kết luận thiên lệch.

---

## 6. Test and Validation Plan

## Unit Tests
- `tests/test_config_loading.py`:
  - default injection cho diagnostics keys,
  - validation lỗi kiểu/miền.
- `tests/test_one_multitask_train_step.py`:
  - diagnostics keys bật/tắt đúng theo config.
- `tests/test_model_shapes.py` (hoặc test model chuyên biệt tương đương):
  - output contract không đổi sau instrumentation.

## Integration Tests
- Smoke run 1-2 epoch với diagnostics on/off:
  - train loop hoàn tất,
  - logger ghi đủ,
  - checkpoint flow không đổi.

## Suggested Verification Commands
- `pytest -q tests/test_config_loading.py tests/test_one_multitask_train_step.py`
- `pytest -q tests/test_model_shapes.py tests/test_checkpoint_roundtrip.py`
- Một lệnh train smoke với config diagnosis đã chọn.

## Acceptance Criteria
- Tất cả test liên quan pass.
- Không có regression ở save/load checkpoint.
- Không có thay đổi ngoài ý muốn trong engine contract.

---

## 7. Commit-Level Execution Checklist
1. Commit 1: config contract + test config.
2. Commit 2: model diagnostics instrumentation + model-step tests.
3. Commit 3: trainer aggregation + logger routing.
4. Commit 4: experiment configs for 4-run matrix.
5. Commit 5: result synthesis research log.

Mỗi commit phải giữ nguyên nguyên tắc readability-first, least codepaths, và khả năng rollback độc lập.

## 8. Completion Definition
Kế hoạch được xem là hoàn tất khi thỏa đồng thời:
1. Diagnostics pipeline chạy được từ config đến logger.
2. Có 4 run matrix với artifact đầy đủ và so sánh công bằng.
3. Có kết luận nguyên nhân dao động reconstruction loss dựa trên chỉ báo định lượng.
4. Không phá vỡ batch/model/engine contracts hiện hành.
