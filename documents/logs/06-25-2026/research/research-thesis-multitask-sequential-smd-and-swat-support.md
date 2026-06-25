---
date: 2026-06-25 13:30:36 +07
researcher: TheMetaSetter
git_commit: c66927b06d3b94f3505792cd3aaf66c0fc6b1064
branch: dev
repository: bachelor-thesis-2026
topic: "Can thesis_multitask run train+val then test sequentially on SMD 3-4, SMD 2-1, and SWaT, and how are final metrics aggregated?"
tags: [research, time-series, anomaly-detection, thesis_multitask, smd, swat, evaluation]
status: complete
last_updated: 2026-06-25
last_updated_by: TheMetaSetter
---

# Research: Can thesis_multitask run train+val then test sequentially on SMD 3-4, SMD 2-1, and SWaT, and how are final metrics aggregated?

**Date**: 2026-06-25 13:30:36 +07  
**Researcher**: TheMetaSetter  
**Git Commit**: `c66927b06d3b94f3505792cd3aaf66c0fc6b1064`  
**Branch**: `dev`

## Research Question

Kiểm tra xem mô hình [`thesis_multitask.py`](../../../../src/models/thesis_multitask.py) trong codebase hiện tại có thể chạy tuần tự theo kiểu `train + val -> test` trên:

1. SMD `machine-3-4`
2. SMD `machine-2-1`
3. SWaT

Đồng thời xác định:

- pipeline hiện tại có thực sự support các dataset đó hay không;
- SWaT trong bối cảnh codebase hiện tại có được chia thành nhiều chuỗi riêng như SMD hay không;
- kết quả cuối cùng đang được tính theo công thức tổng hợp nào;
- các metric nào thực sự đã được implement, đặc biệt với ưu tiên `VUS-PR`, sau đó `VUS-ROC`, rồi `F1`.

## Summary

Kết luận ngắn gọn:

1. **SMD `machine-3-4`: có support đầy đủ trong active thesis pipeline**, và đã có config three-stage đang hoạt động.
2. **SMD `machine-2-1`: có support ở active thesis pipeline**, và đã có config `thesis_multitask` 300 epoch cho entity này.
3. **SWaT: chưa được support trong active thesis pipeline hiện tại**. Dữ liệu SWaT có mặt trong `data/SWaT/`, và reference codebase M2N2 có loader SWaT, nhưng active thesis runtime hiện chỉ whitelist `smd` và `anomaly_archive`.
4. **`scripts/train.py` không tự động chạy test ở cuối**. Với `thesis_multitask` single-stage, train/val và test đang là **hai bước riêng**: train bằng `scripts/train.py`, rồi test bằng `scripts/evaluate.py`. Riêng three-stage runner có orchestration evaluation riêng.
5. **Nếu chạy nhiều SMD entity cùng lúc**, evaluator hiện tại sẽ:
   - merge các window overlap trở lại timeline gốc theo từng `entity_id`;
   - sau đó **concat toàn bộ point score và point label của mọi entity**;
   - rồi tính **một bộ metric global duy nhất** trên vector concat đó.
6. **Metric test hiện có**: `roc_auc`, `pr_auc`, `precision`, `recall`, `f1`, `fpr`, và `vus_pr` nếu bật buffer VUS. **`VUS-ROC` chưa thấy được implement** trong active evaluator/metrics path.

## Detailed Findings

### Data Preparation

#### Active dataset support in the thesis runtime

Active config validation hiện chỉ cho phép:

- `smd`
- `anomaly_archive`

Điều này được hardcode trong [`src/core/config.py`](../../../../src/core/config.py) tại `supported_dataset_names = {"smd", "anomaly_archive"}` và validator sẽ raise ngay nếu `dataset_name` khác tập này.

#### SMD entity-level support

SMD parser đọc ba thư mục:

- `train/*.txt`
- `test/*.txt`
- `test_label/*.txt`

Nếu không filter entity, parser kỳ vọng đủ 28 máy. Nếu có `entity_ids`, parser chỉ lấy đúng các machine được chỉ định và kiểm tra từng entity có mặt ở cả train/test/test_label.

Do đó, về mặt loader, SMD hoàn toàn support các run theo từng machine như:

- `machine-3-4`
- `machine-2-1`
- hoặc nhiều machine cùng lúc nếu truyền nhiều `entity_ids`

#### Existing SMD configs already present

Config three-stage đang active cho `machine-3-4`:

- [`configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml`](../../../../configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml)

Config `thesis_multitask` active cho `machine-2-1`:

- [`configs/experiment/scale/smd__thesis_multitask__multitask-rtx3090-seed11-machine-2-1-300ep__w100__seed11__rtx3090.yaml`](../../../../configs/experiment/scale/smd__thesis_multitask__multitask-rtx3090-seed11-machine-2-1-300ep__w100__seed11__rtx3090.yaml)

Nghĩa là hai SMD run mà câu hỏi nêu ra đều đã có đường config thực tế trong repo.

#### SWaT status in the active thesis codebase

Trong active thesis runtime:

- không có `dataset_name: swat`;
- không có parser/builder SWaT được register trong `scripts/train.py` hay `scripts/evaluate.py`;
- không có support path trong validator hiện hành.

Tuy nhiên:

- local data hiện có thư mục `data/SWaT/` với `normal.csv`, `attack.csv`, `merged.csv`;
- reference codebase M2N2 có loader SWaT riêng và coi SWaT là một cặp `train normal` / `test attack`, không phải tập 28 entity như SMD.

Nói cách khác, **SWaT có dữ liệu và có reference semantics, nhưng chưa được nối vào active thesis pipeline**.

### Modeling and Training

#### thesis_multitask is registered in the offline runtime

`scripts/train.py` register:

- dataset builders cho `smd` và `anomaly_archive`
- model `thesis_multitask`

Sau đó script build dataset bundle, build model, rồi gọi trainer train loop. Không có lệnh evaluate nào ở cuối file train path này.

#### Single-stage `train + val -> test` is not one automatic command

Đối với active single-stage path:

- `scripts/train.py` chỉ train và validate
- `scripts/evaluate.py` load lại config + checkpoint rồi evaluate trên `data_bundle["loaders"]["test"]`

Vì vậy, nếu anh muốn chạy:

1. train + val trên `SMD 3-4`
2. test trên `SMD 3-4`
3. rồi lặp lại cho `SMD 2-1`
4. rồi lặp lại cho `SWaT`

thì với **SMD** điều này làm được, nhưng theo kiểu **nhiều bước hoặc một orchestration script riêng**. Với **SWaT** thì **chưa làm được trong active pipeline** vì thiếu dataset support.

#### Three-stage runner does include evaluation orchestration

Three-stage runner có manifest chứa `evaluation.checkpoint_path` lấy từ best checkpoint của stage train cuối, nên riêng nhánh three-stage đã có tư duy `train stages -> evaluation`.

Điểm này khác với single-stage `scripts/train.py`.

### Evaluation

#### How test windows are turned into final metrics

Evaluator hiện tại làm ba bước chính:

1. chạy `model.test_step(batch)` trên từng batch window của test loader;
2. với mỗi `entity_id`, cộng dồn score theo vùng overlap và chia trung bình để reconstruct lại **point score trên timeline gốc**;
3. concat toàn bộ `point_scores` và `point_labels` của mọi entity, rồi tính metric global.

Nói rõ hơn, công thức thực thi hiện tại là:

1. Với từng entity `e`, từng timestep `t`:
   - gom mọi window chứa `t`;
   - lấy trung bình anomaly score của các window đó.
2. Sau khi có một vector score theo timeline cho từng entity, evaluator nối:
   - `scores_all = concat(scores_entity_1, ..., scores_entity_k)`
   - `labels_all = concat(labels_entity_1, ..., labels_entity_k)`
3. Từ đó:
   - chọn `threshold = quantile(scores_all_positive_support, 0.95)` theo helper `select_point_score_threshold(...)`;
   - tính metric trên `scores_all` và `labels_all`.

Vì vậy, nếu anh chạy nhiều SMD machine trong cùng một config, kết quả cuối là **micro/global aggregation trên toàn bộ point đã concat**, không phải:

- macro-average theo entity;
- weighted-average theo entity;
- hay median theo entity.

#### Metrics actually implemented now

Pointwise metrics hiện có trong active path:

- `roc_auc`
- `pr_auc`
- `precision`
- `recall`
- `f1`
- `fpr`
- `vus_pr` nếu `vus_max_buffer_size` khác `None`

`VUS-PR` được implement qua `compute_vus_pr_exact_naive(...)`, tức là lấy trung bình area-under-range-aware-PR qua nhiều `buffer_size` từ `0` đến `max_buffer_size`.

#### VUS-ROC status

Trong active evaluator và `src/metrics/pointwise.py`, **em không thấy metric `vus_roc` / `VUS-ROC` được tính**.

Do đó, nếu ưu tiên mong muốn của anh là:

1. `VUS-PR`
2. `VUS-ROC`
3. `F1`

thì active pipeline hiện chỉ đáp ứng trọn vẹn:

1. `VUS-PR`
2. `F1`

còn `VUS-ROC` **chưa có implementation runtime tương ứng**.

#### Validation semantics versus true test semantics

Trong trainer:

- `val_realistic` vẫn chạy trên `val_loader`;
- chỉ khác là synthetic anomaly prior được calibrate từ thống kê test-window anomaly rate của SMD thông qua `compute_smd_test_window_anomaly_rate(...)`.

Nó **không phải** true test-set evaluation. True test nằm ở `scripts/evaluate.py`.

## Code References

- [`src/core/config.py:299`](../../../../src/core/config.py#L299) - active dataset whitelist chỉ gồm `smd` và `anomaly_archive`
- [`scripts/train.py:44`](../../../../scripts/train.py#L44) - register runtime datasets/models cho offline training
- [`scripts/train.py:260`](../../../../scripts/train.py#L260) - train script build dataset bundle và bắt đầu training
- [`scripts/evaluate.py:79`](../../../../scripts/evaluate.py#L79) - evaluation entrypoint
- [`scripts/evaluate.py:138`](../../../../scripts/evaluate.py#L138) - evaluator chạy trên `data_bundle["loaders"]["test"]`
- [`src/data/datasets/smd.py:61`](../../../../src/data/datasets/smd.py#L61) - SMD parser parse theo entity files
- [`src/data/datasets/smd.py:184`](../../../../src/data/datasets/smd.py#L184) - tính SMD test-window anomaly rate
- [`src/data/loaders.py:247`](../../../../src/data/loaders.py#L247) - SMD dataset builder truyền `entity_ids` vào parser
- [`src/engine/evaluator.py:97`](../../../../src/engine/evaluator.py#L97) - reconstruct pointwise records theo entity
- [`src/engine/evaluator.py:241`](../../../../src/engine/evaluator.py#L241) - merge window payload thành per-entity records
- [`src/engine/evaluator.py:252`](../../../../src/engine/evaluator.py#L252) - concat tất cả point scores/labels trên mọi entity
- [`src/metrics/pointwise.py:246`](../../../../src/metrics/pointwise.py#L246) - `compute_vus_pr_exact_naive(...)`
- [`src/metrics/pointwise.py:356`](../../../../src/metrics/pointwise.py#L356) - tập metric pointwise thực sự đang được tính
- [`configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml:7`](../../../../configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml#L7) - active `machine-3-4` thesis config
- [`configs/experiment/scale/smd__thesis_multitask__multitask-rtx3090-seed11-machine-2-1-300ep__w100__seed11__rtx3090.yaml:7`](../../../../configs/experiment/scale/smd__thesis_multitask__multitask-rtx3090-seed11-machine-2-1-300ep__w100__seed11__rtx3090.yaml#L7) - active `machine-2-1` thesis config
- [`bsc-thesis-ref-codebases/M2N2-master/README.md:35`](../../../../bsc-thesis-ref-codebases/M2N2-master/README.md#L35) - SWaT semantics: train normal, test abnormal
- [`bsc-thesis-ref-codebases/M2N2-master/data/load_data.py:195`](../../../../bsc-thesis-ref-codebases/M2N2-master/data/load_data.py#L195) - SWaT loader trong reference codebase

## Pipeline Documentation

### What is possible now

#### Case A. SMD `machine-3-4`

**Có thể chạy được.**

- Nếu dùng three-stage active path: đã có config riêng.
- Nếu muốn true test cuối: dùng three-stage orchestration hoặc train rồi evaluate.

#### Case B. SMD `machine-2-1`

**Có thể chạy được.**

- Đã có active thesis config 300 epoch riêng cho entity này.
- Test thật chạy qua `scripts/evaluate.py` sau khi có checkpoint.

#### Case C. SWaT

**Chưa chạy được trong active thesis pipeline hiện tại.**

Lý do:

- validator không cho `dataset_name: swat`;
- train/evaluate runtime không register SWaT dataset builder;
- active loader tree không có SWaT parser/builder.

### If the user wants all sequences shown in the image

Ảnh hiện nêu:

- `SWaT`
- `SMD_1-7`
- `SMD_1-8`
- `SMD_2-1`
- `SMD_2-4`
- `SMD_3-2`

Trong active thesis codebase:

- với **SMD**, về mặt data path có thể chạy theo từng machine đó nếu tạo config `entity_ids` tương ứng;
- với **SWaT**, chưa có active runtime path;
- nếu gom nhiều SMD machine trong cùng một run, metric cuối cùng hiện tại sẽ là **global concat-based metric**, không phải trung bình theo machine.

### SWaT sequence semantics relative to SMD

SMD trong codebase hiện tại là **multi-entity dataset**: mỗi machine là một entity độc lập.

SWaT theo local/reference semantics hiện tại là **một train normal stream** và **một test attack stream**. Nó không xuất hiện trong repo active dưới dạng nhiều entity file giống SMD.

Vì vậy, nếu sau này muốn “chia SWaT thành nhiều chuỗi riêng”, đó sẽ là **một quyết định thiết kế mới**, không phải hành vi đang tồn tại sẵn trong codebase.

## Historical Context (from documents/)

Design documents hiện hành vẫn giữ trọng tâm trên SMD và thesis multitask pipeline với:

- fixed windowing;
- prototype-based multitask modeling;
- `val_realistic_vus_pr` là metric validation quan trọng.

Điều này phù hợp với thực tế implementation: SMD là dataset first-class citizen, còn SWaT mới dừng ở mức dữ liệu cục bộ và reference codebase.

## Open Questions

1. Nếu muốn báo cáo kết quả cuối trên nhiều SMD machine, anh muốn:
   - chạy **mỗi machine một run riêng**, rồi tổng hợp ngoài pipeline;
   - hay cho phép **một config chứa nhiều `entity_ids`**, rồi lấy metric global kiểu concat như code hiện tại?
2. Nếu muốn hỗ trợ SWaT chính thức trong active thesis pipeline, cần chốt trước:
   - SWaT sẽ được coi là **một entity duy nhất** hay chia thành nhiều entity logic;
   - nếu chia, metric cuối cùng sẽ theo **concat global**, **macro-average theo entity**, hay công thức khác;
   - có cần thêm **`VUS-ROC`** vào evaluator để khớp ưu tiên metric đã nêu hay không.
