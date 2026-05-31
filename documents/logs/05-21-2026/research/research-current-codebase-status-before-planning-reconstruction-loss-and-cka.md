---
date: 2026-05-21 17:09:02 +07
researcher: TheMetaSetter
git_commit: 84df6bee3dc9314ed462f983b4efa3bae4590d72
branch: dev
repository: bachelor-thesis-2026
topic: "Nghiên cứu tình hình codebase hiện tại trước khi lên kế hoạch lập trình (trọng tâm reconstruction loss, CKA fusion, và train/eval logging)"
tags: [research, time-series, anomaly-detection, multi-class]
status: complete
last_updated: 2026-05-21
last_updated_by: TheMetaSetter
---

# Research: Nghiên cứu tình hình codebase hiện tại trước khi lên kế hoạch lập trình

**Date**: 2026-05-21 17:09:02 +07  
**Researcher**: TheMetaSetter  
**Git Commit**: 84df6bee3dc9314ed462f983b4efa3bae4590d72  
**Branch**: dev

## Research Question
Được. Hãy nghiên cứu tình hình codebase hiện tại để ghi chú trước khi lên bất kì kế hoạch lập trình nào. Sử dụng prompt `prompts/1_research_prompt.md`.

## Summary
Codebase hiện tại đã triển khai đầy đủ pipeline offline multitask cho SMD với cửa sổ độ dài 20, gồm: parser + scaler + windowing, synthetic anomaly injection theo taxonomy RedLamp, mô hình `thesis_multitask` có hai nhánh prototype (continuous và discrete), fusion theo `alpha` và `beta` (có thể bật CKA-gated fusion), objective đa thành phần với reconstruction/classification/optional/contrastive, và trainer epoch-based với synthetic validation + checkpoint monitor theo `val_synth_vus_pr`.

Theo trạng thái config hiện tại cho thí nghiệm nhỏ vừa tạo, optimizer là AdamW với learning rate 0.001, scheduler là ReduceLROnPlateau, không bootstrap (`bootstrap_encoder_epochs: 0`), và logging W&B được bật ở chế độ online.

## Detailed Findings

### Data Preparation
- Dữ liệu SMD được parse theo entity/split, làm sạch qua `SequenceCleaningPipeline`, fit scaler trên train split, sau đó transform cho train/val/test (`src/data/loaders.py`).
- Windowing được thực hiện bằng `WindowDataset` với chỉ số `(sequence_index, start, end)`, materialize cửa sổ khi `__getitem__`, giữ contract `x`, `point_labels`, `mask`, `timestamps`, `meta` (`src/data/loaders.py`).
- Config data cho experiment hiện tại dùng `window_size: 20`, `stride: 20`, `entity_ids: [machine-2-1]` (`configs/data/smd_rtx3090_machine_2_1_20.yaml`).
- `collate_windows` stack các tensor và validate batch contract trước khi trả loader batch (`src/data/collate.py`).

### Augmentation and Labeling
- Synthetic anomaly injector định nghĩa 11 anomaly families RedLamp và class names multiclass tương ứng (`src/data/augment.py`).
- `augment_batch` tạo quyết định inject theo batch, ghi `synthetic_anomaly_mask`, `classification_labels`, `augmentation_metadata`, đồng thời cập nhật `point_labels` bằng hợp của nhãn gốc và mask synthetic (`src/data/augment.py`).
- Segment bounds và affected channels được lấy ngẫu nhiên thông qua generator nội bộ; deterministic seed chỉ áp cho injector validation khi cấu hình (`src/data/augment.py`).

### Modeling and Training
- `thesis_multitask` chuẩn bị batch theo stage:
  - train: có thể inject synthetic nếu bật `use_synthetic_augmentation`.
  - val_synth: inject bằng validation injector (deterministic seed).
  - val/test: batch clean path.
- Forward path tạo `hidden`, tra cứu nhánh continuous/discrete, fusion thành `hidden_reconstruction` và `hidden_classification`, rồi xuất `recon`, `logits`, `point_scores` (`src/models/thesis_multitask.py`).
- Reconstruction loss:
  - Nếu `reconstruction_normal_only=False` hoặc không có mask: mean MSE toàn bộ.
  - Nếu `reconstruction_normal_only=True`: chỉ tính trên normal cells bằng mask `1 - synthetic_anomaly_mask`, fallback về mean toàn bộ nếu không còn normal cells (`src/models/thesis_multitask.py`).
- Total loss được lắp trong `_shared_step`:
  - `L_total = L_recon + lambda_cls * L_cls + optional_losses + lambda_contrastive * L_contrastive`.
  - Optional losses gồm diversity/variance/covariance/usage/gate theo config enable/weight (`src/models/thesis_multitask.py`).
- CKA-related metrics:
  - Fusion aux chứa `cka_reconstruction_mean/std` và `cka_classification_mean/std`.
  - Stage log ghi 4 metric CKA cho `train` và `val_synth`; đồng thời log `alpha`, `beta`, `alpha_std`, `beta_std` (`src/models/thesis_multitask.py`).

### Training Engine, Checkpoint, and Logging
- Trainer chạy epoch loop, gọi `training_step`, aggregate train/val/val_synth logs theo mean batch, rồi log `epoch_metrics` (`src/engine/trainer.py`).
- Synthetic validation được chạy qua `synthetic_validation_step`; pointwise metrics reconstructed theo timeline entity và tính `vus_pr`, threshold (`src/engine/trainer.py`, `src/engine/evaluator.py`).
- Checkpoint monitor hỗ trợ `val_synth_vus_pr` và các metric khác; chế độ max cho VUS/ROC/PR (`src/engine/trainer.py`).
- `ExperimentLogger` luôn ghi JSONL; nếu `logging.use_wandb=true` thì gọi `wandb.init(...)` và `wandb.log(metrics)` cho mỗi epoch (`src/engine/logger.py`).

### Evaluation
- Evaluator dùng `model.test_step`, gom `point_scores` từ các window về timeline, lấy threshold theo quantile 0.95, tính pointwise metrics và curve payload (`src/engine/evaluator.py`, `src/metrics/pointwise.py`).
- Trong training runtime, ngoài aggregate stage logs, trainer còn gọi `_aggregate_reconstructed_pointwise_metrics` cho `val_synth` từ payload window-level (`src/engine/trainer.py`).

## Code References
- `src/data/loaders.py:64` - `WindowDataset` định nghĩa window contract và metadata.
- `src/data/loaders.py:118` - `SMDDatasetBuilder` parse/clean/scale/window/build DataLoader.
- `src/data/collate.py:11` - `collate_windows` tạo batch tensor và validate contract.
- `src/data/augment.py:728` - `augment_batch` ghi `synthetic_anomaly_mask`, `classification_labels`, `augmentation_metadata`.
- `src/models/thesis_multitask.py:1484` - stage-dependent batch preparation.
- `src/models/thesis_multitask.py:1570` - forward contract (`hidden`, `recon`, `logits`, `point_scores`).
- `src/models/thesis_multitask.py:1755` - `_compute_reconstruction_loss` với normal-only masking.
- `src/models/thesis_multitask.py:1966` - `_compute_total_loss` weighted sum objective.
- `src/models/thesis_multitask.py:1985` - `_build_stage_log` (bao gồm alpha/beta và CKA).
- `src/models/thesis_multitask.py:2205` - `training_step`/`validation_step`/`synthetic_validation_step`.
- `src/engine/trainer.py:390` - vòng lặp train epoch, aggregate logs, scheduler step, checkpoint.
- `src/engine/logger.py:72` - W&B init gate theo `use_wandb`.
- `src/engine/logger.py:119` - `log_metrics` ghi JSONL và mirror lên W&B.
- `src/engine/evaluator.py:187` - evaluate loop và pointwise metric computation.
- `src/core/config.py:55` - inject/validate `model.window_size` theo `data.window_size`.
- `src/core/config.py:256` - validate `checkpoint_monitor_metric` whitelist.
- `configs/experiment/thesis/exp2/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20-exp2-small-100ep__w20__seed11__default.yaml:1` - experiment mới cho thí nghiệm nhỏ.

## Pipeline Documentation
1. Load experiment YAML -> resolve `data/model/task` references -> apply overrides -> validate config contracts (`src/core/config.py`).
2. Build SMD dataset bundle: parse -> clean -> scale -> window dataset -> collate loaders (`src/data/loaders.py`, `src/data/collate.py`).
3. For each train batch, model injects synthetic anomalies (if enabled), computes multitask forward and losses (`src/models/thesis_multitask.py`).
4. Trainer performs optimizer step, aggregates epoch metrics for train/val/val_synth, computes reconstructed pointwise metrics for val_synth, and updates LR scheduler (`src/engine/trainer.py`).
5. Logger persists metrics to JSONL and optionally to W&B (`src/engine/logger.py`).
6. Best checkpoint is saved when configured monitor metric improves (`src/engine/trainer.py`).

## Historical Context (from documents/)
- `documents/design/idea.md` và `documents/design/design_starter.md` nêu contract hiện tại cho thesis model: hidden representation `[B, L, H]`, hai nhánh prototype continuous/discrete, fusion task-specific, objective modular, và active window length là `L=20`.
- Hai tài liệu design cũng chỉ rõ hướng triển khai offline pre-training với two-view contrastive và CKA-gated fusion là implementation contract đang hoạt động.

## Open Questions
- Chưa có instrumentation định lượng trực tiếp trong log hiện tại để tách đóng góp phương sai của từng nguồn rung lắc `reconstruction_loss` (ví dụ: `active_normal_cells`, mask ratio per batch, hoặc gradient norm theo tham số nhóm).
- Trong runtime hiện tại, CKA metrics được log cho `train` và `val_synth`; nếu cần đối chiếu với `val` clean path thì chưa có metric CKA tương ứng ở stage `val`.
