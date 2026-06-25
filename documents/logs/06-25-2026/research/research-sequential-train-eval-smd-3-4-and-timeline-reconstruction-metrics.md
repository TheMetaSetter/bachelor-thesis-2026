---
date: 2026-06-25 15:47:07 +0700
researcher: TheMetaSetter
git_commit: c66927b06d3b94f3505792cd3aaf66c0fc6b1064
branch: dev
repository: bachelor-thesis-2026
topic: "Sequential train+val -> test on SMD 3-4 for thesis_multitask.py and redlamp_mlp_baseline.py, with timeline reconstruction and requested metrics"
tags: [research, time-series, anomaly-detection, smd, evaluation, metrics]
status: complete
last_updated: 2026-06-25
last_updated_by: TheMetaSetter
---

# Research: Sequential train+val -> test on SMD 3-4 for thesis_multitask.py and redlamp_mlp_baseline.py, with timeline reconstruction and requested metrics

**Date**: 2026-06-25 15:47:07 +0700
**Researcher**: TheMetaSetter
**Git Commit**: c66927b06d3b94f3505792cd3aaf66c0fc6b1064
**Branch**: dev

## Research Question

Trong codebase hiện tại:

1. Có thể chạy tuần tự `train + val -> test` cho `src/models/thesis_multitask.py` và `src/models/redlamp_mlp_baseline.py` trên SMD `machine-3-4` hay không?
2. Có thể loop qua ba random seed `11`, `36`, `68` hay không?
3. Có thể ghép các vector anomaly score và vector label của từng test window về lại timeline gốc của từng chuỗi test như pseudo-code đã đính kèm hay không?
4. Ba metric test được yêu cầu là `VUS-PR`, `Affiliation-F1`, `VUS-ROC` hiện có được hiện thực trong codebase hay không?

## Summary

Codebase hiện tại **có hỗ trợ phần core để ghép lại pointwise anomaly scores và pointwise labels của các test windows về timeline gốc của từng entity**. Việc ghép này diễn ra trong evaluator bằng cách gom theo `entity_id` và `start_index`, sau đó:

- cộng dồn score theo từng timestep và chia cho số lần overlap, tức **score reduce = mean**;
- ghép nhãn bằng phép `max`, tức nếu bất kỳ window nào đánh dấu timestep đó là anomaly thì timeline label tại timestep đó là anomaly.

Cách làm này rất gần với pseudo-code mà anh đính kèm.

Về orchestration, codebase hiện tại **có thể train và test cả hai model trên SMD**, vì cả dataset `smd` lẫn hai model `thesis_multitask` và `redlamp_mlp_baseline` đều đã được đăng ký trong `train.py` và `evaluate.py`. Tuy nhiên, với generic single-stage offline experiments, train và test hiện là **hai bước tách rời**: `scripts/train.py` chỉ train + val, còn `scripts/evaluate.py` mới chạy test từ checkpoint. Script `scripts/run_multiseed_experiments.py` hiện chỉ loop nhiều config cho **training**, không tự động nối sang evaluation.

Về metric, codebase hiện tại **có VUS-PR**, nhưng **không có bằng chứng hiện thực `Affiliation-F1` hoặc `VUS-ROC`** trong active runtime.

## Detailed Findings

### Data Preparation

- SMD parser hỗ trợ lọc theo `entity_ids`, nên về mặt data path có thể nhắm riêng `machine-3-4`: `src/data/datasets/smd.py:82-107`.
- Data config riêng cho `machine-3-4` đã tồn tại sẵn: `configs/data/smd_rtx3090_machine_3_4_20_stride1.yaml:1-8`.
- Với SMD test split, parser gắn toàn bộ vector label gốc của chuỗi test vào `point_labels` của raw test sequence: `src/data/datasets/smd.py:118-165`.
- `WindowDataset` cắt test sequence thành các windows và giữ metadata `entity_id`, `start_index`, `end_index`, `absolute_start_index`, `absolute_end_index`, nên đủ thông tin để reconstruct timeline gốc: `src/data/loaders.py:204-244`.

### Modeling and Training

- `scripts/train.py` đăng ký dataset `smd` và cả hai model `thesis_multitask`, `redlamp_mlp_baseline`: `scripts/train.py:44-52`.
- `src/core/config.py` cũng cho phép `dataset_name = smd` và `model_name` thuộc `{thesis_multitask, redlamp_mlp_baseline}`: `src/core/config.py:299-317`.
- `scripts/train.py` chỉ build data bundle, build model, rồi gọi `trainer.train(...)` trên `train_loader` và `val_loader`; không có đoạn gọi test/evaluate ở cuối: `scripts/train.py:260-277` và `scripts/train.py:374-427`.
- `scripts/evaluate.py` mới là entrypoint test thật sự: nó load lại config, load checkpoint, rồi gọi `Evaluator.evaluate(...)` trên `data_bundle["loaders"]["test"]`: `scripts/evaluate.py:96-138`.
- Nghĩa là, với generic offline run của hai model này, workflow implemented today là:
  - bước 1: `train + val` bằng `scripts/train.py`;
  - bước 2: `test` bằng `scripts/evaluate.py`.

### Multi-seed Orchestration

- Script `scripts/run_multiseed_experiments.py` chỉ build lệnh `python scripts/train.py --experiment-config ...` cho mỗi config: `scripts/run_multiseed_experiments.py:125-132`.
- Nó chỉ chạy training commands, dù ở chế độ `sequential` hay `parallel`: `scripts/run_multiseed_experiments.py:207-222` và `scripts/run_multiseed_experiments.py:246-260`.
- Không có đoạn nào trong script này nối tiếp `scripts/evaluate.py`.
- Vì vậy:
  - **có thể** loop 3 seeds nếu chuẩn bị 3 experiment configs riêng;
  - nhưng **chưa có runner built-in** cho loop `train+val -> test` của 3 seeds cho generic single-stage path.

### Config Reality for SMD 3-4

- Config data cho `machine-3-4` có sẵn: `configs/data/smd_rtx3090_machine_3_4_20_stride1.yaml:1-8`.
- Nhưng baseline config active mà em đọc hiện vẫn trỏ vào `machine-2-1`, không phải `machine-3-4`: `configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml:8-12`.
- Thesis multitask single-stage config active em đọc cũng đang trỏ vào `machine-2-1`: `configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20__w20__seed11__default.yaml:8-12`.
- Thesis three-stage config cho `machine-3-4` và `seed=11` đã tồn tại: `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml:7-15`.
- Em không thấy config sẵn cho `seed=36` hay `seed=68` trên `machine-3-4` trong repo search hiện tại.

### Evaluation and Timeline Reconstruction

- `Evaluator.evaluate(...)` thu `point_scores` và `point_labels` của từng batch window test vào `pointwise_batch_payloads`: `src/engine/evaluator.py:214-239`.
- Sau đó evaluator gọi `reconstruct_pointwise_records_from_window_payload(...)`: `src/engine/evaluator.py:241-244`.
- Trong `accumulate_pointwise_window_payload(...)`, code duyệt từng window, lấy `entity_id`, `start_index`, `end_index`, rồi:
  - cộng score window vào `entity_score_sums[entity_id][start_index:end_index]`: `src/engine/evaluator.py:87-89`;
  - cộng bộ đếm overlap vào `entity_score_counts[entity_id][start_index:end_index]`: `src/engine/evaluator.py:90`;
  - ghép nhãn bằng `torch.maximum(...)`: `src/engine/evaluator.py:91-94`.
- Trong `reconstruct_pointwise_records_from_window_payload(...)`, điểm anomaly cuối cùng tại mỗi timestep là `averaged_scores = score_sum / counts`: `src/engine/evaluator.py:116-123`.
- Sau đó evaluator nối các record đã reconstruct của từng entity thành một vector score lớn và một vector label lớn để tính metrics global: `src/engine/evaluator.py:245-268`.

Điều này có nghĩa là:

- codebase **đã thực hiện đúng ý tưởng ghép từ window về timeline**;
- score reduce đang được hard-code là **mean**;
- label reduce đang được hard-code là **max**;
- phép ghép được làm **theo từng entity** trước, rồi mới concat toàn bộ entities để tính metric chung.

### Metric Support

- `compute_pointwise_metrics(...)` hiện trả về:
  - `roc_auc`
  - `pr_auc`
  - `precision`
  - `recall`
  - `f1`
  - `fpr`
  - và thêm `vus_pr` nếu `vus_max_buffer_size` khác `None`
  tại `src/metrics/pointwise.py:338-378`.
- Hàm `compute_vus_pr_exact_naive(...)` đã được hiện thực: `src/metrics/pointwise.py:246-287`.
- `scripts/evaluate.py` truyền `vus_max_buffer_size` và `vus_num_thresholds` vào `Evaluator`, nên VUS-PR test path thực sự được bật trong runtime: `scripts/evaluate.py:124-138`.
- Em **không tìm thấy** hiện thực nào cho:
  - `Affiliation-F1`
  - `VUS-ROC`
  trong `src/` và `tests/` bằng repo-wide search.
- `rg` cũng không trả về symbol hay hàm nào tên `affiliation` hoặc `vus_roc`.

## Code References

- `src/data/datasets/smd.py:82-107` - lọc SMD theo `entity_ids`
- `src/data/datasets/smd.py:118-165` - nạp test labels gốc của từng entity
- `src/data/loaders.py:204-244` - mỗi test window mang `point_labels` và metadata timeline
- `src/engine/evaluator.py:44-128` - ghép windows về lại timeline gốc theo entity
- `src/engine/evaluator.py:199-295` - evaluate trên test loader và concat records để tính metric
- `src/metrics/pointwise.py:246-287` - hiện thực `compute_vus_pr_exact_naive`
- `src/metrics/pointwise.py:338-378` - metric set pointwise hiện đang có
- `scripts/train.py:260-277` và `scripts/train.py:374-427` - train path chỉ train + val
- `scripts/evaluate.py:96-138` - test path load checkpoint rồi evaluate trên test
- `scripts/run_multiseed_experiments.py:125-132` và `scripts/run_multiseed_experiments.py:207-260` - multi-seed runner hiện chỉ launch training

## Pipeline Documentation

Pseudo-code anh đính kèm và codebase hiện tại khớp nhau ở logic lớn:

1. bắt đầu từ `Y_win` và `S_win` có shape `[N, L]`;
2. dùng `start_index` của từng window để map từng timestep trong window về timestep gốc của entity;
3. gộp overlap;
4. tạo lại `y` và `s` trên timeline gốc;
5. tính metric trên timeline-level vectors.

Khác biệt implemented today:

- score reduce hiện đang cố định là `mean`, chưa có switch `"mean"` hoặc `"max"`;
- label reduce hiện đang cố định là `max`;
- aggregation làm theo từng `entity_id` trước, sau đó concat tất cả entities để ra metric toàn test set.

## Historical Context (from documents/)

`documents/design/design_starter.md` và `documents/design/idea.md` đều mô tả batch contract là `point_labels: Tensor[B, L]`, nên evaluator reconstruct pointwise timeline từ windows là nhất quán với design contract hiện tại.

## Open Questions

- Nếu anh muốn đúng ba metric `VUS-PR`, `Affiliation-F1`, `VUS-ROC` ở test, thì hiện codebase còn thiếu ít nhất hai metric sau: `Affiliation-F1` và `VUS-ROC`.
- Nếu anh muốn loop đúng `2 models x 3 seeds x train+val->test` trong một runner thống nhất, hiện generic single-stage path chưa có orchestration script built-in cho cả train lẫn evaluate.
