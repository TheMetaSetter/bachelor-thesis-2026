---
date: 2026-05-27 12:41:26 +0700 (+07)
researcher: Artificial Intelligence Agent
git_commit: 57aeba72e81071194e6e271faab39fbc1e955c89
branch: dev
repository: bachelor-thesis-2026
topic: "Nghiên cứu codebase hiện tại để nắm rõ tình hình các luồng tính toán chi tiết hiện tại, phục vụ việc lên kế hoạch lập trình thí nghiệm dựa trên thesis_multitask.py sau này"
tags: [research, time-series, anomaly-detection, multi-class]
status: complete
last_updated: 2026-05-27
last_updated_by: Artificial Intelligence Agent
---

# Research: Nghiên cứu codebase hiện tại để nắm rõ tình hình các luồng tính toán chi tiết hiện tại, phục vụ việc lên kế hoạch lập trình thí nghiệm dựa trên thesis_multitask.py sau này

**Date**: 2026-05-27 12:41:26 +0700 (+07)
**Researcher**: Artificial Intelligence Agent
**Git Commit**: 57aeba72e81071194e6e271faab39fbc1e955c89
**Branch**: dev

## Research Question
Nghiên cứu codebase hiện tại để nắm rõ tình hình các luồng tính toán chi tiết hiện tại, phục vụ việc lên kế hoạch lập trình thí nghiệm dựa trên `src/models/thesis_multitask.py` sau này.

## Summary
Pipeline hiện tại là config-driven, đi theo trục `scripts/train.py` -> `Trainer` -> `ThesisMultitaskModel._shared_step(...)` cho train/validation và `scripts/evaluate.py` -> `Evaluator.evaluate(...)` cho offline checkpoint evaluation. Dữ liệu SMD được parse theo entity, tách train/val từ train split gốc, chuẩn hóa theo train split, rồi window hóa thành batch contract `[B, L, D]` qua `collate_windows`.

`thesis_multitask.py` hiện là MLP-based multitask model (không phải CNN encoder). Forward path gồm encoder -> continuous prototype lookup -> discrete prototype lookup -> fusion theo task -> reconstruction/classification heads. Loss tổng được assemble tại `_shared_step`: reconstruction + classification (có weight theo stage) + optional losses + optional contrastive term. Validation có hai nhánh: `val` (clean) và `val_synth` (inject synthetic anomalies), trong đó `val_synth` còn được aggregate thêm pointwise metrics (bao gồm `val_synth_vus_pr`) để theo dõi checkpoint monitor.

## Detailed Findings

### Data Preparation
Dữ liệu SMD được đọc bởi `SMDDatasetParser.parse()` từ `train/*.txt`, `test/*.txt`, `test_label/*.txt`, validate số file và entity filtering, rồi build raw sequence contract với `meta` chuẩn (`dataset_name`, `entity_id`, `split`, `num_channels`, `sequence_length`) tại `src/data/datasets/smd.py:52` và `src/data/datasets/smd.py:61`.

Train/val được cắt trực tiếp từ train sequence theo `validation_split_ratio` tại `src/data/datasets/smd.py:130`; point labels của train/val đặt về zeros, còn test dùng nhãn thật từ `test_label` tại `src/data/datasets/smd.py:152`.

`SMDDatasetBuilder.build(...)` chạy cleaning pipeline, fit scaler trên train split, transform toàn bộ split và dựng `WindowDataset` + `DataLoader` tại `src/data/loaders.py:116` và `src/data/loaders.py:164`. Window metadata giữ `start_index/end_index/window_size` để evaluator reconstruct timeline (`src/data/loaders.py:105`).

`collate_windows(...)` stack thành batch contract cố định gồm `x`, `point_labels`, `mask`, `timestamps`, `meta`, và validate bằng `validate_batch(...)` tại `src/data/collate.py:10`. Contract checker nằm ở `src/core/contracts.py:91`.

Synthetic augmentation cho multitask được đóng gói tại `SyntheticAnomalyInjector.augment_batch(...)` trong `src/data/augment.py:724`. Thành phần thêm vào batch gồm `classification_labels`, `classification_class_names`, `synthetic_anomaly_mask`, `augmentation_metadata` (`src/data/augment.py:776` đến `src/data/augment.py:781`). Taxonomy active hiện tại là 11 RedLamp families (`src/data/augment.py:20`).

### Modeling and Training
Model được build từ merge `model` + `task` config tại `scripts/train.py:46` và `scripts/train.py:62`. Với thí nghiệm mục tiêu, config sử dụng `thesis_multitask` + `multitask_tsad` + `classification_label_mode: redlamp_multiclass`, `num_classes: 12`, `reconstruction_normal_only: true`, monitor `val_synth_vus_pr` (xem file config experiment/model/task tương ứng).

Encoder hiện tại là `MultitaskWindowEncoder` dùng MLP (`build_multilayer_perceptron`) tại `src/models/thesis_multitask.py:95`; output encoder contract là `hidden` và `pooled` ở `src/models/thesis_multitask.py:118`.

Runtime step của model tập trung ở `_shared_step(...)` (`src/models/thesis_multitask.py:2206`):
1. Chuẩn bị batch qua `_prepare_batch(...)` hoặc cặp clean/augmented nếu bật two-view contrastive (`src/models/thesis_multitask.py:1565`, `src/models/thesis_multitask.py:1570`).
2. Gọi `forward(...)` (`src/models/thesis_multitask.py:1585`):
   - `validate_batch` trước forward (`line 1597`).
   - Encode hidden (`line 1606`).
   - Lookup hai nhánh prototype (`lines 1651-1664`).
   - Fusion ra hidden_reconstruction/hidden_classification (`lines 1667-1677`).
   - Reconstruction head + classification head (flatten window) (`lines 1680-1698`).
   - Point score = MSE theo timestep (`line 1704`), window score = mean point score (`line 1714`).
3. Tính loss thành phần:
   - Reconstruction loss (`src/models/thesis_multitask.py:1774`): nếu `reconstruction_normal_only=true` và có `synthetic_anomaly_mask`, chỉ average trên normal cells, fallback full MSE nếu mask rỗng (`lines 1780-1796`).
   - Classification loss (`src/models/thesis_multitask.py:1838`): CE hoặc refurbished soft-target CE.
   - Optional losses qua switch + weight config (`src/models/thesis_multitask.py:2013`, `src/models/thesis_multitask.py:2027`).
4. Assemble `total_loss` và stage log (`src/models/thesis_multitask.py:2244` và `src/models/thesis_multitask.py:2287`).

Trainer loop (`src/engine/trainer.py:535`) thực hiện `loss.backward()` và `optimizer.step()` cho train batch (`lines 609-629`). Validation chạy `validation_step` và `synthetic_validation_step` nếu model hỗ trợ (`lines 659-686`).

Epoch metrics gồm:
- mean các stage logs,
- reconstruction diagnostics aggregate,
- classification metrics aggregate từ logits/labels,
- pointwise reconstruction metrics của `val_synth` bằng reconstruct timeline (`_aggregate_reconstructed_pointwise_metrics`, `src/engine/trainer.py:482`),
- optional full evaluator metrics trên val clean (`line 755`).

Checkpoint best được chọn theo `checkpoint_monitor_metric` đã validate trong config (`src/core/config.py:256`), rồi save `best.pt` qua `CheckpointManager` (`src/engine/trainer.py:822`).

### Evaluation
Entrypoint evaluation là `scripts/evaluate.py`. Script rebuild model từ config, load checkpoint, load scaler state, rồi gọi `Evaluator.evaluate(model, test_loader)` (`scripts/evaluate.py:133`).

`Evaluator` chạy `model.test_step(...)` theo batch (`src/engine/evaluator.py:165`), thu `point_scores` và `point_labels`, rồi reconstruct về timeline per entity bằng average các vùng overlap (`src/engine/evaluator.py:99` và `src/engine/evaluator.py:111`).

Threshold mặc định chọn theo quantile 0.95 của concatenated point scores (`src/engine/evaluator.py:24` và `src/engine/evaluator.py:264`). Sau đó compute pointwise metrics + curves qua `src/metrics/pointwise.py` (`src/engine/evaluator.py:276` và `src/engine/evaluator.py:286`).

Kết quả xuất ra `evaluation_records.json`, `evaluation_metrics.json`, `evaluation_curves.json` trong `output_dir` (`scripts/evaluate.py:147` đến `scripts/evaluate.py:173`).

## Code References
- `scripts/train.py:38` - đăng ký runtime components cho train.
- `scripts/train.py:343` - gọi `trainer.train(...)`.
- `src/engine/trainer.py:535` - epoch training loop trung tâm.
- `src/engine/trainer.py:609` - train batch step: forward/loss/backward/step.
- `src/engine/trainer.py:683` - synthetic validation payload dùng `synthetic_anomaly_mask`.
- `src/engine/trainer.py:492` - reconstruct pointwise metrics cho `val_synth`.
- `src/models/thesis_multitask.py:1585` - forward chính của thesis multitask.
- `src/models/thesis_multitask.py:1774` - masked reconstruction loss logic.
- `src/models/thesis_multitask.py:2206` - `_shared_step` assemble objective và logging.
- `src/data/augment.py:724` - augment_batch sinh nhãn giả anomaly multiclass và mask.
- `src/data/loaders.py:116` - build SMD dataset bundle end-to-end.
- `src/core/contracts.py:91` - batch contract validator.
- `src/engine/evaluator.py:199` - evaluate loop trên test loader.

## Pipeline Documentation
Luồng offline hiện hành cho `thesis_multitask`:
1. Load experiment config và resolve 3 config con (`data/model/task`) tại `src/core/config.py:776`.
2. Build SMD loaders theo window size/stride trong data config (`src/data/loaders.py:164`).
3. Build model từ model+task kwargs (`scripts/train.py:46`).
4. Train theo epoch ở `Trainer.train` với train, val, và optional val_synth (`src/engine/trainer.py:535`).
5. Mỗi batch train gọi `ThesisMultitaskModel.training_step` -> `_shared_step` -> `forward` -> loss assembly (`src/models/thesis_multitask.py:2299`).
6. Aggregation theo epoch + scheduler + checkpoint best (`src/engine/trainer.py:736` đến `src/engine/trainer.py:842`).
7. Evaluate checkpoint bằng evaluator để ra metrics/curves/records (`scripts/evaluate.py:133`).

Window length hiện tại trong thí nghiệm mục tiêu là `window_size = 20` (data config machine_2_1_20 + model window size resolve/validate trong `src/core/config.py:55`).

Label taxonomy cho classification pseudo-label ở thí nghiệm mục tiêu là `redlamp_multiclass` gồm 12 lớp (`normal` + 11 anomaly families) (`src/data/augment.py:33`, `configs/task/multitask_tsad_redlamp_multiclass_window20.yaml:5`).

## Historical Context (from documents/)
`documents/design/idea.md` và `documents/design/design_starter.md` mô tả kiến trúc thesis theo hướng self-contained model file, objective modularity, và contract-based runtime. Trạng thái code hiện tại bám hướng đó: `thesis_multitask.py` gom kiến trúc + training step logic + logging trong một file; engine giữ vai trò điều phối loop, không định nghĩa loss theo model.

## Open Questions
1. Trong pipeline hiện tại, threshold của evaluator đang lấy trực tiếp từ score của chính tập đang evaluate (quantile 0.95). Cần xác nhận protocol này có được giữ nguyên cho các thí nghiệm gradient-conflict sắp tới hay chuyển sang calibration split cố định.
2. `thesis_multitask.py` hiện là MLP encoder; nếu thí nghiệm tiếp theo thay encoder sang REDLAMP CNN, cần chốt trước strategy giữ nguyên hay thay đổi output/shape contract của `pooled` (hiện là flattened classification hidden).
3. Cần chốt phạm vi logging gradient-conflict: chỉ bottleneck/projection layer hay toàn bộ encoder parameters theo từng block/layer group.
