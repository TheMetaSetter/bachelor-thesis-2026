---
date: 2026-07-04 21:45:02 +07 +0700
researcher: TheMetaSetter
git_commit: 3f2d0b9a7c2d4a453da2cc918939bdef719b697e
branch: dev
repository: bachelor-thesis-2026
topic: "Current codebase state relevant to offline_pretraining_two_stage_kmeans_memory_design.md"
tags: [research, time-series, anomaly-detection, multi-class]
status: complete
last_updated: 2026-07-04
last_updated_by: TheMetaSetter
---

# Research: Current codebase state relevant to offline_pretraining_two_stage_kmeans_memory_design.md

**Date**: 2026-07-04 21:45:02 +07 +0700  
**Researcher**: TheMetaSetter  
**Git Commit**: `3f2d0b9a7c2d4a453da2cc918939bdef719b697e`  
**Branch**: `dev`

## Research Question

Use `prompts/1_research_prompt.md` to inspect the repository state and identify the important code paths that matter before planning implementation for `documents/design/offline_pretraining_two_stage_kmeans_memory_design.md`.

## Summary

Ý tưởng chính ở đây là design mới đã khóa một hợp đồng huấn luyện `two-stage`, nhưng code runtime hiện tại vẫn lấy `three-stage` làm trục chính ở config validation, orchestration script, active experiment YAML, và semantic phase naming. File trung tâm vẫn là `src/models/thesis_multitask.py`, và file này đã có đủ các bề mặt mà design mới quan tâm: shared encoder, continuous memory bank, discrete codebook, task-specific fusion, synthetic anomaly injection, two-view contrastive loss, và memory initialization hook.

Tuy nhiên, trạng thái code hôm nay chưa hiện thực đúng hợp đồng của design mới. Cụ thể, memory initialization hiện không dùng k-means; nó dùng một thủ tục `covering selection` trên hidden vectors đã được chuẩn hóa, rồi ghi trực tiếp vào `continuous_prototype_bank` và `discrete_codebook` (`src/models/thesis_multitask.py:1468-1515`, `src/models/thesis_multitask.py:1609-1669`). Ngoài ra, hook khởi tạo memory hiện được gọi ở đầu mỗi epoch trong trainer (`src/engine/trainer.py:597-615`). Với config model active cho three-stage benchmark path, `bootstrap_encoder_epochs: 0` (`configs/model/thesis_multitask_three_stage_window20.yaml:45`), nên nếu code path đó được dùng trực tiếp trong một phase có prototype path hoạt động, memory có thể được khởi tạo ngay khi vào epoch đầu tiên, không phải “cuối Stage A sau 80 epoch” như design mới.

## Detailed Findings

### Data Preparation

Data loader active cho bài toán này vẫn đi theo SMD full-sequence -> scale -> window -> batch. `src/data/loaders.py` là file nên đọc đầu tiên cho data path. Nó resolve root, resolve stride theo split, fit `SequenceStandardScaler` trên train split, transform toàn bộ split, rồi mới window hóa (`src/data/loaders.py:54-102`, `src/data/loaders.py:150-193`). Điều này nghĩa là normalization hiện là `train_only_before_windowing`, đúng với bề mặt mà evaluation audit cũng ghi nhận sau này.

Config dữ liệu mà exp4 active đang dùng là `configs/data/smd_rtx3090_machine_3_4_20_stride1.yaml`, với `entity_ids: [machine-3-4]`, `window_size: 20`, `stride: 1`, `batch_size: 256` (`configs/data/smd_rtx3090_machine_3_4_20_stride1.yaml:1-10`). Nói ngắn gọn, window length hiện hành cho line benchmark này là 20, không phải 100.

Synthetic anomaly injection cho multitask offline path nằm hoàn toàn trong `src/data/augment.py`. Taxonomy active là 11 anomaly families của RedLamp cộng thêm `normal` thành 12 class names (`src/data/augment.py:21-35`). Task config `configs/task/multitask_tsad_redlamp_multiclass_window20.yaml` bật cả synthetic augmentation và synthetic validation, dùng `classification_label_mode: redlamp_multiclass`, `train_balance_classes: true`, `anomaly_probability: 0.5`, `min_segment_fraction: 0.2`, `max_segment_fraction: 0.3` (`configs/task/multitask_tsad_redlamp_multiclass_window20.yaml:1-29`).

Trong runtime batch augmentation, injector clone batch, chọn class label mục tiêu cho từng window, inject anomaly theo family tương ứng, rồi thêm các field `classification_labels`, `classification_class_names`, `synthetic_anomaly_mask`, và `augmentation_metadata` mà không phá vỡ batch contract gốc (`src/data/augment.py:866-931`). Hiểu nôm na thì loader không biết gì về synthetic anomaly; model tự quyết định lúc nào batch được augment.

### Modeling and Training

Contract batch/output hiện được cố định trong `src/core/contracts.py`. Batch offline luôn phải có `x`, `point_labels`, `mask`, `timestamps`, `meta`, trong đó `x` có shape `[B, L, D]` (`src/core/contracts.py:95-109`). Output model luôn phải có `hidden`, `pooled`, `recon`, `logits`, `point_scores`, `window_scores`, `aux`, với `hidden` có shape `[B, L, H]` (`src/core/contracts.py:127-143`). Đây là contract thật của codebase hôm nay.

Training entrypoint offline là `scripts/train.py`. Script này load experiment config, register dataset/model, merge `model` với `task` config để build model, rồi giao cho `Trainer` chạy (`scripts/train.py:44-82`, `scripts/train.py:224-260`). Nói cách khác, bề mặt cấu hình thật để plan implementation phải đi qua `experiment` YAML, rồi chảy vào `model` + `task`.

File model trung tâm vẫn là `src/models/thesis_multitask.py`. Ở mức xây khối, model hiện tạo:

- shared encoder qua `MultitaskWindowEncoder` (`src/models/thesis_multitask.py:921-923`);
- `continuous_prototype_bank` là buffer kích thước `continuous_num_prototypes x hidden_dim` (`src/models/thesis_multitask.py:926-935`);
- discrete side gồm `discrete_assignment`, `discrete_codebook`, `discrete_ema_counts`, `discrete_ema_sums` (`src/models/thesis_multitask.py:937-966`);
- reconstruction head và classification head đều lấy supervision từ fused hidden, không phải branch-local head (`src/models/thesis_multitask.py:984-1010`).

Điểm rất quan trọng là code hiện vẫn giữ cả hai semantic surface cho discrete branch. Khi build model, `discrete_assignment` vẫn luôn được tạo nếu discrete memory bật (`src/models/thesis_multitask.py:939-943`). Nhưng lúc lookup, nếu `discrete_query_mode == "cosine_topk"` thì code dùng dot-product với codebook, lấy `topk`, softmax trên top-k logits, rồi aggregate codewords (`src/models/thesis_multitask.py:1932-1958`). Chỉ khi không ở `cosine_topk` nó mới rơi về `discrete_assignment + gumbel_softmax` (`src/models/thesis_multitask.py:1959-1968`). Vậy nên design mới nói “tránh giữ Gumbel-only machinery” là chưa đúng với runtime hiện tại; runtime hiện vẫn mang cả surface đó.

Hai-view contrastive loss đã có mặt trong model. Nó chỉ lấy các token normal theo `synthetic_anomaly_mask == 0`, normalize anchor/positive tokens, rồi tính cross-entropy trên similarity matrix (`src/models/thesis_multitask.py:2160-2183`). Pair clean/augmented batch cũng đã có, và được dùng bên trong `_shared_step` trước khi forward chính của batch augment (`src/models/thesis_multitask.py:2268-2283`, `src/models/thesis_multitask.py:3123-3163`).

Memory initialization hôm nay đi theo logic sau. Ở đầu epoch, trainer gọi `maybe_initialize_memories_from_loader` nếu model có method đó (`src/engine/trainer.py:597-615`). Trong model, hook này chỉ chạy nếu prototype path đang dùng, memory chưa init, và `current_epoch_index >= bootstrap_encoder_epochs` (`src/models/thesis_multitask.py:1380-1391`). Sau đó model duyệt một số batch train, lấy hidden token pool cho continuous và discrete, rồi gọi `_initialize_memory_buffers_from_token_pool` (`src/models/thesis_multitask.py:1392-1413`).

Chi tiết token pool hiện tại là:

- Continuous pool lấy `selected_normal_hidden` từ synthetic batch, tức là chỉ giữ token normal ở những window synthetic train nếu `memory_initialization_with_synthetic_windows` bật (`src/models/thesis_multitask.py:1545-1568`).
- Discrete pool group hidden theo `classification_labels` của synthetic batch (`src/models/thesis_multitask.py:1569-1579`).

Chi tiết khởi tạo memory hiện tại là:

- Continuous memory dùng `_select_covering_vectors(continuous_hidden_tokens, continuous_num_prototypes)` (`src/models/thesis_multitask.py:1620-1625`).
- Discrete codebook chia đều số codeword theo class, fallback sang token pool hợp nhất nếu một class thiếu token, rồi cũng dùng `_select_covering_vectors` cho từng class (`src/models/thesis_multitask.py:1627-1669`).

Không thấy implementation k-means ở đoạn này. Design mới ghi rõ `k-means` cho cả continuous và discrete memory, nhưng code hiện tại chưa làm vậy.

Một lệch khác với design mới là topology orchestration. Config model active cho three-stage line là `configs/model/thesis_multitask_three_stage_window20.yaml`, trong đó `continuous_num_prototypes: 16`, `discrete_codebook_size: 60`, `enable_two_view_contrastive: true`, `bootstrap_encoder_epochs: 0`, `training_phase: multitask_pretraining`, `fusion_mode: task_specific_concat_projection`, `discrete_query_mode: cosine_topk`, `freeze_memories_after_initialization: true`, `discrete_memory_label_source: synthetic_train_labels` (`configs/model/thesis_multitask_three_stage_window20.yaml:1-54`).

Experiment YAML active exp4 vẫn là ba tầng rõ ràng: `stage1_classification`, `stage1_reconstruction`, `stage2_recovery`, `stage3_memory_initialization_and_fusion_warmup`, `multitask_pretraining`, với tổng `300` epoch (`configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml:1-48`). Script orchestration `scripts/run_three_stage_offline_pretraining.py` cũng encode cứng chính xác phase order này (`scripts/run_three_stage_offline_pretraining.py:37-47`) và build per-phase configs bằng cách gán `model.training_phase = phase_name` (`scripts/run_three_stage_offline_pretraining.py:177-220`).

### Evaluation

Offline evaluation entrypoint là `scripts/evaluate.py`. Nó rebuild dataset, rebuild model từ config, load checkpoint, nếu có `raw_sequences` thì rebuild data bundle bằng scaler state lưu trong checkpoint, rồi gọi `Evaluator.evaluate` trên test loader (`scripts/evaluate.py:144-218`). Sau đó nó còn lưu records, metrics, curves, protocol audit, và resolved config vào `output_dir` (`scripts/evaluate.py:240-260`).

`src/engine/evaluator.py` merge point scores của các window chồng lắp trở lại timeline gốc theo `entity_id`, `start_index`, `end_index`, rồi average score theo số lần cover (`src/engine/evaluator.py:68-167`). Sau đó evaluator lấy toàn bộ covered points để tính metric. Nếu checkpoint không mang threshold sẵn, evaluator tự chọn threshold ở quantile 0.95 của positive-support scores (`src/engine/evaluator.py:24-42`, `src/engine/evaluator.py:329-371`).

`src/metrics/pointwise.py` hiện tính `roc_auc`, `pr_auc`, `precision`, `recall`, `f1`, `fpr`, `affiliation_f1`, và nếu có `vus_max_buffer_size` thì tính thêm `vus_pr`, `vus_roc` (`src/metrics/pointwise.py:542-607`). Trong exp4 active, metric monitor là `val_synth_vus_pr` (`configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml:21-31`), nên plan implementation sau này phải coi đây là surface hiện đang quyết định best checkpoint.

## Code References

- `documents/design/offline_pretraining_two_stage_kmeans_memory_design.md:1` - approved SSOT mới cho rerun two-stage.
- `src/core/contracts.py:95` - batch contract `[B, L, D]` và field bắt buộc.
- `src/core/contracts.py:127` - model output contract với `hidden`, `recon`, `logits`, `point_scores`.
- `src/data/loaders.py:150` - scaler fit trên train split rồi transform trước windowing.
- `src/data/loaders.py:176` - build `WindowDataset` và `DataLoader` từ scaled sequences.
- `src/data/augment.py:21` - 11 RedLamp anomaly families + `normal`.
- `src/data/augment.py:866` - batch augmentation path thêm `classification_labels` và `synthetic_anomaly_mask`.
- `scripts/train.py:55` - merge `model` + `task` config để build runtime model.
- `src/engine/trainer.py:597` - trainer hook gọi `maybe_initialize_memories_from_loader` ở đầu epoch.
- `src/models/thesis_multitask.py:921` - continuous/discrete memory state được build trong model.
- `src/models/thesis_multitask.py:1380` - điều kiện memory initialization từ train loader.
- `src/models/thesis_multitask.py:1517` - token-pool collection cho memory initialization.
- `src/models/thesis_multitask.py:1609` - current memory initialization implementation.
- `src/models/thesis_multitask.py:1932` - active discrete `cosine_topk` query path.
- `src/models/thesis_multitask.py:1959` - fallback `gumbel_softmax` query path vẫn còn tồn tại.
- `src/models/thesis_multitask.py:2160` - two-view contrastive loss trên token normal.
- `src/models/thesis_multitask.py:3123` - `_shared_step` ghép contrastive + main forward + loss assembly.
- `configs/model/thesis_multitask_three_stage_window20.yaml:16` - current prototype sizes và runtime toggles.
- `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml:1` - active three-stage experiment surface.
- `scripts/run_three_stage_offline_pretraining.py:41` - encoded three-stage phase order.
- `src/engine/evaluator.py:24` - threshold selection logic.
- `src/metrics/pointwise.py:542` - pointwise metric computation including `vus_pr`.

## Pipeline Documentation

Pipeline offline hiện tại, nếu nhìn đúng code path chung, là:

1. Load experiment YAML rồi merge `data`, `model`, `task`.
2. Build SMD sequences, fit scaler trên train split, transform các split, rồi window hóa thành batch `[B, L, D]`.
3. Trong multitask model, train batch có thể bị inject synthetic anomaly ngay trong model step.
4. Encoder tạo `hidden` có shape `[B, L, H]`.
5. Nếu phase đang dùng prototype path và memory đã sẵn sàng, continuous branch đọc từ `continuous_prototype_bank`, discrete branch đọc từ `discrete_codebook` bằng `cosine_topk` hoặc Gumbel branch.
6. Fusion tạo `hidden_reconstruction` và `hidden_classification`.
7. Reconstruction head sinh `recon`, classification head sinh `logits`, point score là MSE theo timestep.
8. Loss tổng ghép reconstruction, classification, contrastive, và optional regularizers theo config.
9. Evaluation merge score từ overlapping windows về full timeline rồi tính pointwise metrics.

## Historical Context (from documents/)

`documents/design/idea.md` và `documents/design/design_starter.md` vẫn phản ánh triết lý chung của repo: giữ hidden contract ổn định, mô hình tự chứa trong một file, task heads lấy supervision từ fused representation, và objective nên modular nhưng codepath phải ít. Tuy nhiên, design cụ thể mới cho rerun hiện đã được khóa ở `documents/design/offline_pretraining_two_stage_kmeans_memory_design.md`, nơi hợp đồng mới là:

- two-stage thay cho three-stage,
- Stage A train shared multitask encoder từ scratch,
- cuối Stage A mới init cả hai memory bằng k-means,
- Stage B freeze encoder + both memories,
- Stage B chỉ train fusion heads và prediction heads,
- continuous size mới là 32, discrete size giữ nguyên 60.

Những ý đó hiện là design SSOT mới, chưa phải mô tả trung thực của runtime hôm nay.

## Open Questions

1. Trong plan implementation kế tiếp, “active config base” nên xuất phát từ line exp4 three-stage benchmark hiện tại hay từ exp2 single-run multitask configs. Cả hai đều còn tồn tại trong repo.
2. Design mới nói `k-means`, nhưng code hiện chỉ có `covering selection`. Chưa có file hay helper nào trong các đoạn đã đọc hiện thực k-means memory initialization cho `thesis_multitask.py`.
3. Design mới muốn loại bớt Gumbel-only machinery nếu discrete path chuẩn là `cosine_topk`, nhưng runtime hiện vẫn luôn build `discrete_assignment` khi discrete memory bật.
4. Với config `bootstrap_encoder_epochs: 0`, cần xác nhận ở bước plan kế tiếp chính xác giai đoạn nào sẽ được phép gọi memory init sau khi topology chuyển sang two-stage, để tránh init từ encoder chưa train.
