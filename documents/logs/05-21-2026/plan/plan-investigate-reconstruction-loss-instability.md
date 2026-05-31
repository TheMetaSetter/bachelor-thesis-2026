---
date: 2026-05-21 18:05:00 +07
planner: Codex
git_commit: 84df6bee3dc9314ed462f983b4efa3bae4590d72
branch: dev
repository: bachelor-thesis-2026
topic: "Kế hoạch lập trình chi tiết cho thí nghiệm điều tra nguyên nhân reconstruction loss dao động mạnh"
tags: [plan, reconstruction-loss, diagnostics, cka, thesis-multitask]
status: proposed
last_updated: 2026-05-21
last_updated_by: Codex
source_research: documents/logs/05-21-2026/research/research-current-codebase-status-before-planning-reconstruction-loss-and-cka.md
---

# Detailed Implementation Plan: Điều tra nguyên nhân reconstruction loss dao động mạnh

## Scope and Objective
Mục tiêu của kế hoạch này là thiết kế và triển khai một vertical slice chẩn đoán có khả năng tái lập để xác định các nguồn gây dao động mạnh của `reconstruction_loss` trong pipeline `thesis_multitask` hiện tại (offline multitask trên SMD, `window_size=20`), đồng thời giữ nguyên các contract đã ổn định của codebase.

Kế hoạch chỉ tập trung vào **instrumentation, logging, và protocol thí nghiệm**; không thay đổi bản chất kiến trúc model hoặc objective mặc định ngoài các can thiệp cần thiết để cô lập nguyên nhân.

## Current State
- Pipeline hiện tại đã hoàn thiện đường đi dữ liệu và training loop: SMD parse/clean/scale/window -> collate contract -> `thesis_multitask` -> trainer/evaluator/logger/checkpoint.
- `reconstruction_loss` đã có hai chế độ: full MSE và normal-only MSE (`reconstruction_normal_only` + `synthetic_anomaly_mask`), có fallback khi không còn normal cells.
- Stage logging đã có CKA (`cka_reconstruction_*`, `cka_classification_*`) và fusion stats (`alpha`, `beta`, std), nhưng chưa có các chỉ báo định lượng trực tiếp về độ nhiễu của reconstruction path theo batch.
- Validation monitor đang hỗ trợ `val_synth_vus_pr`, phù hợp với policy hiện tại của repo.

## Established Contracts That Must Be Preserved
- **Batch contract** (data/collate/model input): `x`, `point_labels`, `mask`, `timestamps`, `meta` và các field synthetic mở rộng (`synthetic_anomaly_mask`, `classification_labels`, metadata augmentation).
- **Encoder contract**: hidden representation theo thesis-facing shape `[B, L, H]`.
- **Model output contract**: giữ `recon`, `logits`, `point_scores`, `aux` và không phá vỡ interface trainer/evaluator đang dùng.
- **Codebase constraints**: `1 model - 1 file`, readability-first, minimal codepath branching, ablation-friendly qua YAML.

## Design Options

### Option A (Recommended): In-Model Diagnostics + Trainer Aggregation (Minimal-Intrusion)
Thêm instrumentation trực tiếp trong `src/models/thesis_multitask.py` để xuất các signal theo batch vào `aux/log_dict`, trainer chỉ aggregate và logger giữ nguyên cơ chế.

Ưu điểm:
- Bám sát `1 model - 1 file`.
- Không tạo hệ thống phụ mới.
- Dễ kiểm soát độ lệch runtime và dễ rollback.

Hạn chế:
- Cần kỷ luật để tránh phình `thesis_multitask.py`.

### Option B: Tách diagnostics sang module mới trong `src/engine/`
Tạo diagnostics collector độc lập, model chỉ gửi raw tensors.

Ưu điểm:
- Phân tách kỹ thuật rõ.

Hạn chế:
- Dễ vi phạm tinh thần self-contained model logic của repo.
- Tăng coupling giữa model-engine cho một nhu cầu chẩn đoán ngắn hạn.

### Option C: Hậu kiểm hoàn toàn bằng script external từ logs/checkpoints
Không sửa training loop, chỉ parse log hiện có.

Ưu điểm:
- Ít thay đổi code runtime.

Hạn chế:
- Không đủ observability để truy nguyên dao động vì thiếu batch-level root signals.

## Selected Approach
Chọn **Option A** vì đáp ứng tốt nhất đồng thời 4 tiêu chí: (i) truy nguyên được nguyên nhân, (ii) giữ stable interfaces, (iii) tuân thủ codebase preferences, (iv) triển khai nhanh cho chuỗi ablation.

## Risk and Mitigation
- Risk: Logging quá nhiều gây overhead hoặc nhiễu đọc log.
  - Mitigation: cờ config bật/tắt diagnostics và giới hạn metric cốt lõi.
- Risk: Đồng nhất sai giữa train clean/synthetic path.
  - Mitigation: log rõ theo stage (`train`, `val`, `val_synth`) và tách metric clean vs synthetic.
- Risk: Kết luận sai do metric aggregate che mất outlier batch.
  - Mitigation: bổ sung thống kê phân tán theo epoch (mean/std/p90/p95/max).
- Risk: Can thiệp diagnostics làm thay đổi hành vi optimizer.
  - Mitigation: chỉ đọc tensor/grad với `detach`, không chỉnh graph; bật từng bước bằng ablation.

## Open Questions (Must Resolve Before Full Experiment Batch)
- Có cần thu thập gradient norm theo **toàn model** hay theo **nhóm tham số** (`encoder`, `decoder`, `fusion/prototype`) để định vị nguồn dao động chính xác hơn?
- Mức chi tiết cần lưu cho batch distribution là theo toàn epoch hay thêm rolling window theo step?
- Ngưỡng “dao động mạnh” dùng để ra quyết định là CV, p95/p50 ratio, hay max/median ratio?

---

## Programming Plan (Detailed)

## Phase 0: Baseline Freeze and Reproducibility Gate
1. Xác nhận một config baseline duy nhất cho diagnosis run (ưu tiên config experiment đang active trong research).
2. Chạy baseline ngắn (ví dụ 5-10 epoch) để lấy mốc dao động hiện tại, lưu artifact logs làm đối chứng.
3. Ghi rõ command chuẩn trong tài liệu plan để các lần chạy sau không lệch đường dẫn config.

Deliverable:
- Baseline run metadata + log path + checkpoint path (nếu có improvement monitor).

## Phase 1: Add Diagnostics Config Contract
### Files
- `src/core/config.py`
- `configs/experiment/thesis/exp2/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20-exp2-small-100ep__w20__seed11__default.yaml` (hoặc config diagnosis mới nếu tách riêng)

### Changes
1. Mở rộng schema `logging` hoặc `task/model` bằng một nhóm cờ diagnostics rõ ràng, ví dụ:
   - `enable_reconstruction_diagnostics: bool`
   - `diagnostics_log_interval_steps: int`
   - `diagnostics_include_grad_norm: bool`
2. Thiết lập default an toàn: tắt diagnostics nếu không khai báo.
3. Validate giá trị kiểu số và miền hợp lệ để tránh config mơ hồ.

Contract enforcement:
- Config loader phải đảm bảo absent -> default và invalid -> fail sớm.

## Phase 2: Instrument Reconstruction Path in Model
### File
- `src/models/thesis_multitask.py`

### Changes
1. Trong `_compute_reconstruction_loss`, tính thêm và trả về các signal batch-level:
   - `recon_mse_mean`
   - `recon_mse_std`
   - `active_normal_cells`
   - `normal_cell_ratio`
   - `synthetic_cell_ratio`
   - `fallback_to_full_mse_flag`
2. Trong `_shared_step`, gắn các signal này vào stage log dictionary với prefix rõ ràng, ví dụ:
   - `train_recon_mse_std_batch`
   - `train_normal_cell_ratio_batch`
3. Bảo toàn hành vi loss hiện tại: diagnostics chỉ quan sát, không thay đổi `L_total`.
4. Nếu bật diagnostics grad norm: đo norm sau backward ở trainer hook hoặc model helper (detach-only), không can thiệp optimizer step.

Contract enforcement:
- Không thay đổi keys bắt buộc mà trainer/evaluator hiện tại dùng.
- Không thay đổi shape/type của outputs chuẩn.

## Phase 3: Trainer Aggregation for Instability Indicators
### File
- `src/engine/trainer.py`

### Changes
1. Mở rộng hàm aggregate epoch metrics để giữ thêm thống kê phân tán cho reconstruction diagnostics:
   - mean, std, min, max, p90/p95 cho các batch metrics chính.
2. Tạo nhóm chỉ báo instability cấp epoch:
   - `recon_loss_cv = std(recon_loss_batch) / (mean(recon_loss_batch)+eps)`
   - `recon_loss_p95_to_p50`
3. Log riêng cho từng stage để tránh trộn clean path và synthetic path.

Contract enforcement:
- Scheduler/checkpoint monitor logic giữ nguyên metric monitor hiện tại.
- Không làm thay đổi criteria lưu checkpoint.

## Phase 4: Logger and Artifact Readability
### File
- `src/engine/logger.py`

### Changes
1. Đảm bảo mọi metric diagnostics mới đều đi qua JSONL và W&B nhất quán với metric hiện hữu.
2. Chuẩn hóa prefix naming để tiện filter dashboard:
   - `diag/recon/...`
   - `diag/cka/...`
   - `diag/grad/...`
3. Ghi metadata tĩnh của run (window_size, stride, synthetic injection enable, reconstruction mode) ngay đầu run.

## Phase 5: Focused Experiment Matrix (Ablation-Oriented)
### Experiment policy
Thiết kế tối thiểu 4 run để cô lập nguyên nhân, giữ seed/dataset/config base đồng nhất:
1. `reconstruction_normal_only = false`, synthetic on.
2. `reconstruction_normal_only = true`, synthetic on.
3. `reconstruction_normal_only = true`, synthetic off (clean control).
4. `reconstruction_normal_only = true`, synthetic on + CKA-gated fusion off (để tách ảnh hưởng fusion dynamics).

Mỗi run dùng cùng training length ngắn trước (smoke-diagnostics), sau đó chỉ kéo dài run có tín hiệu rõ.

## Phase 6: Post-Run Analysis and Decision Criteria
1. So sánh các chỉ báo dao động (`cv`, `p95/p50`, max spike frequency) giữa 4 run.
2. Tách kết luận theo nhóm nguyên nhân:
   - mask sparsity / fallback behavior,
   - synthetic injection intensity,
   - fusion/CKA dynamics,
   - optimizer/gradient instability.
3. Chỉ khi đã xác định nhóm nguyên nhân chính mới đề xuất thay đổi thuật toán (không nhảy ngay sang tuning ngẫu nhiên).

---

## Test Plan

## Unit Tests
- `tests/test_config_loading.py`
  - Verify default + validation cho diagnostics config keys.
- `tests/test_one_multitask_train_step.py`
  - Verify diagnostics keys xuất hiện đúng khi bật cờ và không xuất hiện khi tắt.
- `tests/test_model_shapes.py` hoặc test model-specific tương đương
  - Verify output contract không đổi sau instrumentation.

## Integration/Smoke Tests
- Run training smoke 1-2 epoch với diagnostics on/off để kiểm tra:
  - không crash,
  - metric logger ghi đủ keys,
  - checkpoint flow không đổi.

## Non-regression Assertions
- Loss scalar và optimizer step vẫn hợp lệ.
- `val_synth_vus_pr` monitor/save behavior không thay đổi.

---

## Validation Procedure
1. Validate config load contract.
2. Validate runtime contract bằng smoke run.
3. Validate metric completeness trong JSONL/W&B.
4. Validate comparability giữa các run bằng fixed seed + fixed data config.

## Deliverables
- Code instrumentation + tests theo các phase trên.
- Tài liệu kết quả diagnosis (research log mới) tổng hợp bảng chỉ báo dao động và kết luận nguyên nhân ưu tiên.
- Danh sách hành động tiếp theo theo thứ tự ưu tiên (chỉ sau khi có bằng chứng từ diagnostics).

## Minimal Vertical Slice First
Vertical slice bắt buộc trước mở rộng:
1. Bật diagnostics config.
2. Log 3 chỉ báo cốt lõi (`recon_loss_batch`, `normal_cell_ratio`, `fallback_flag`).
3. Aggregate epoch CV.
4. Chạy 1 smoke run xác nhận pipeline.

Chỉ sau khi vertical slice ổn định mới mở rộng sang grad norms, quantiles nâng cao, và ma trận ablation đầy đủ.

## Suggested Implementation Order
1. `src/core/config.py` + config YAML cập nhật.
2. `src/models/thesis_multitask.py` instrumentation đọc-only.
3. `src/engine/trainer.py` aggregation indicators.
4. `src/engine/logger.py` naming + routing consistency.
5. Tests + smoke runs.
6. Chạy experiment matrix chẩn đoán.

## Final Note
Kế hoạch này giữ nguyên thesis intent trong `documents/design/idea.md` và nguyên tắc kỹ thuật trong `documents/design/design_starter.md`: contracts ổn định, composition, ablation-friendly config, và triển khai theo minimal vertical slice trước khi mở rộng can thiệp.
