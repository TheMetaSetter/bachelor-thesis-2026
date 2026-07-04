# Plan: Offline Pretraining Two-Stage KMeans Memory Line

**Date**: 2026-07-04  
**Planner**: Codex  
**Input research note**: `documents/logs/07-04-2026/research/research-offline-pretraining-two-stage-kmeans-memory-design-current-codebase-state.md`  
**Design SSOT**: `documents/design/offline_pretraining_two_stage_kmeans_memory_design.md`  
**Related design guidance**: `documents/design/idea.md`, `documents/design/design_starter.md`

## Planning Goal

Mục tiêu của kế hoạch này là dựng **một line thí nghiệm mới ngay từ đầu** cho
`thesis_multitask.py`, bám đúng hợp đồng huấn luyện `two-stage` đã được khóa
trong design SSOT mới, thay vì sửa trực tiếp line `exp4 three-stage` đang active.

Hướng này ưu tiên ba điều:

1. giữ nguyên line `three-stage` cũ để đối chiếu và rollback;
2. tách semantic `two-stage` mới ra thành một surface config và orchestration
   riêng, không trộn với `three_stage`;
3. hiện thực một vertical slice tối thiểu nhưng chạy được end-to-end trước,
   rồi mới dọn các phần legacy như Gumbel-only machinery.

## Current State

Trạng thái hiện tại của codebase cho thấy line offline pretraining đang xoay
quanh semantic `three_stage`, không phải `two-stage`.

- `src/models/thesis_multitask.py` đã có shared encoder, continuous memory,
  discrete codebook, fusion, synthetic anomaly injection, contrastive loss, và
  hook khởi tạo memory.
- `src/engine/trainer.py:597` gọi
  `maybe_initialize_memories_from_loader(...)` ở đầu epoch. Điều này lệch với
  design mới vì design yêu cầu **chỉ khởi tạo memory ở cuối Stage A**.
- `src/models/thesis_multitask.py:1609` hiện khởi tạo memory bằng
  `covering selection`, chưa có k-means.
- `scripts/run_three_stage_offline_pretraining.py` và
  `src/core/config.py` đang encode cứng semantic `three_stage`.
- Active line `exp4` dùng
  `configs/model/thesis_multitask_three_stage_window20.yaml` và
  `configs/experiment/thesis/exp4/...three-stage...yaml`.

Các contract cần giữ nguyên:

- batch contract tại `src/core/contracts.py:95` với `x: [B, L, D]`;
- model output contract tại `src/core/contracts.py:127` với
  `hidden: [B, L, H]`, `recon`, `logits`, `point_scores`, `window_scores`;
- data path normalize-before-windowing trong `src/data/loaders.py`;
- synthetic label path RedLamp multiclass trong `src/data/augment.py`.

## Design Options

### Option A. New two-stage experiment line with dedicated orchestration script

Tạo một runner mới như `scripts/run_two_stage_offline_pretraining.py`, một
surface config mới `two_stage`, một model config mới, và một experiment YAML mới.
Runner sẽ gọi `scripts/train.py` hai lần theo đúng semantic:

- Stage A: train multitask encoder from scratch trong 80 epoch;
- cuối Stage A: build memory bằng latent pool từ train split;
- Stage B: load checkpoint Stage A, freeze encoder + memories, train fusion và
  prediction heads thêm 20 epoch.

Ưu điểm:

- tách sạch khỏi `three_stage`;
- blast radius nhỏ hơn so với sửa trainer/model để tự hiểu phase boundary;
- dễ audit artifact, checkpoint lineage, và epoch budget;
- dễ giữ line cũ nguyên vẹn để so sánh.

Nhược điểm:

- cần thêm một orchestration script mới;
- có thêm một surface config cần validate.

### Option B. Single experiment run with two-stage semantics embedded in trainer/model

Giữ một lần gọi `scripts/train.py`, nhưng thêm logic vào trainer hoặc model để:

- tự nhận biết khi hết `stage_a_epochs`;
- chạy memory initialization đúng một lần ở boundary đó;
- freeze tham số cần thiết rồi tiếp tục Stage B trong cùng process.

Ưu điểm:

- chỉ cần một run;
- ít script orchestration hơn.

Nhược điểm:

- trainer và model state machine sẽ phức tạp hơn;
- khó audit chính xác checkpoint boundary;
- dễ vô tình reuse logic `bootstrap_encoder_epochs` hiện tại và init memory sai
  thời điểm;
- khó rollback nếu logic stage mới va chạm với training loop cũ.

### Option C. Manual chained configs with no dedicated runner

Tạo hai experiment YAML riêng biệt và chạy thủ công:

- một config cho Stage A;
- một config cho Stage B đọc checkpoint của Stage A.

Ưu điểm:

- code mới ít nhất;
- phù hợp nếu chỉ cần chạy một lần rất thủ công.

Nhược điểm:

- không bền cho thesis workflow;
- dễ lỗi thao tác tay;
- artifact lineage và epoch accounting kém rõ ràng;
- không phù hợp nếu sau này cần smoke run, benchmark line, hoặc server launch.

## Recommended Direction

Khuyến nghị chọn **Option A**.

Ý tưởng chính ở đây là đây là phương án đơn giản nhất mà vẫn đúng semantic của
design mới. Nó không cố ép `three_stage` thành `two_stage`, không đẩy quá nhiều
trách nhiệm state machine vào trainer, và vẫn giữ được một vertical slice rõ:

1. Stage A train xong;
2. memory init một lần ở cuối Stage A;
3. Stage B freeze encoder + memories;
4. train fusion heads và prediction heads;
5. evaluate như pipeline hiện tại.

Nói ngắn gọn, Option A là phương án ít moving parts nhất mà vẫn đủ đúng cho
thesis rerun.

## Proposed Architecture

### 1. Experiment family separation

Tạo một family mới cho line này, ví dụ:

- model config:
  `configs/model/thesis_multitask_two_stage_window20.yaml`
- experiment config:
  `configs/experiment/thesis/exp5/`
  `smd__thesis_multitask__offline-pretraining-two-stage-kmeans-machine-3-4-window20__w20__seed11__rtx3090.yaml`
- smoke config:
  cùng thư mục `exp5` với phiên bản `...__smoke.yaml`
- runner:
  `scripts/run_two_stage_offline_pretraining.py`

Không reuse key `three_stage`. Surface config mới nên là:

```yaml
two_stage:
  stage_a_epochs: 80
  stage_b_epochs: 20
  expected_total_training_epochs: 100
  freeze_encoder_in_stage_b: true
  freeze_memories_in_stage_b: true
  continuous_memory_init_method: kmeans
  discrete_memory_init_method: kmeans_per_class
  discrete_memory_label_source: synthetic_train_labels
```

### 2. Model semantic separation

`src/models/thesis_multitask.py` nên được sửa theo hướng tách rõ ba semantic:

- `stage_a_multitask_encoder_training`
- `memory_initialization`
- `stage_b_fusion_head_training`

Không nên tiếp tục dùng `bootstrap_encoder_epochs` như cơ chế chính để xác định
thời điểm init memory cho line này. Field đó có thể giữ để tránh phá line cũ,
nhưng line mới không nên phụ thuộc vào nó.

### 3. Memory initialization interface

Thêm một interface rõ nghĩa trong model, ví dụ:

- `collect_memory_initialization_statistics_from_loader(...)`
- `initialize_memories_from_statistics(...)`
- `freeze_two_stage_stage_b_surfaces()`

Hướng này tốt hơn việc tiếp tục dùng
`maybe_initialize_memories_from_loader(...)` cho line mới, vì method cũ đang
ngụ ý “có thể init ở đầu epoch” thay vì “bắt buộc init tại boundary giữa Stage A
và Stage B”.

### 4. K-means implementation boundary

Không nhét k-means trực tiếp thành logic khó đọc trong một khối lớn của
`thesis_multitask.py`. Nên thêm một helper nhỏ, ví dụ:

- `src/models/components/kmeans_memory.py`

hoặc nếu muốn giữ repo ít file hơn:

- một helper section gọn trong `src/models/thesis_multitask.py`.

Khuyến nghị thực tế là thêm file helper riêng cho k-means vì đây là một đơn vị
trách nhiệm khá rõ:

- normalize input tokens nếu cần;
- chạy k-means cho continuous pool;
- chạy k-means per class cho discrete pool;
- fallback an toàn khi số token của một class thấp hơn số centroid yêu cầu.

### 5. Batch and output contract enforcement

Contract không đổi:

- batch vẫn đi qua `src/core/contracts.py`;
- output vẫn giữ `hidden`, `pooled`, `recon`, `logits`, `point_scores`,
  `window_scores`, `aux`.

Plan này không thay batch contract và không thay public output contract. Mọi đổi
khác chỉ nên nằm ở config/model internals và orchestration.

## File-Level Programming Plan

### A. Config validation and experiment loading

**Modify**

- `src/core/config.py`

**Responsibilities**

- thêm validator cho block `two_stage`;
- không để validator `three_stage` áp lên line mới;
- validate tổng epoch của `two_stage`;
- validate các boolean freeze flag;
- validate enum cho memory init method;
- validate `discrete_memory_label_source`.

**Why**

Nếu không có validation riêng, line mới sẽ rất dễ lẫn semantic với `three_stage`
và khó phát hiện config drift.

### B. New runner for the new experiment line

**Create**

- `scripts/run_two_stage_offline_pretraining.py`

**Responsibilities**

- đọc experiment config;
- validate `two_stage`;
- materialize run manifest riêng dưới `output_dir/two_stage/`;
- chạy Stage A qua `scripts/train.py`;
- gọi bước memory initialization đúng một lần sau Stage A;
- sinh checkpoint init cho Stage B;
- chạy Stage B với checkpoint đã được freeze đúng contract;
- ghi execution report và resolved configs.

**Why**

Đây là nơi tốt nhất để encode semantic boundary giữa Stage A và Stage B mà
không làm trainer chung phình thêm.

### C. Model changes for stage-aware two-stage semantics

**Modify**

- `src/models/thesis_multitask.py`

**Responsibilities**

- thêm semantic `training_phase` mới cho line `two-stage`;
- tách logic Stage A và Stage B rõ ràng;
- bỏ phụ thuộc của line mới vào `bootstrap_encoder_epochs`;
- thay interface init memory từ kiểu “maybe per epoch” sang “explicit boundary”;
- thêm helper freeze encoder + continuous memory + discrete memory cho Stage B;
- giữ `cosine_topk` là discrete query path chính;
- giảm phụ thuộc runtime vào `discrete_assignment` khi line mới không dùng
  Gumbel path.

**Important constraint**

Không được phá line `three-stage` đang tồn tại. Nghĩa là cần giữ backward
compatibility cho config cũ, nhưng route mới nên tách nhánh rõ, không trộn mơ hồ.

### D. K-means memory helper

**Create**

- `src/models/components/kmeans_memory.py`

**Responsibilities**

- torch-native mini-batch k-means hoặc full-batch k-means đơn giản cho tensor
  `[N, H]`;
- deterministic seeding theo seed runtime hiện có;
- per-class k-means wrapper cho discrete pool;
- fallback path khi `N < K`;
- utility trả về centroid có shape đúng với
  `continuous_num_prototypes` hoặc `discrete_codebook_size`.

**Why**

Thiết kế này tôn trọng single responsibility và tránh làm `thesis_multitask.py`
thành một file còn khó đọc hơn.

### E. Optional trainer cleanup

**Modify**

- `src/engine/trainer.py`

**Responsibilities**

- chặn hoặc bypass hook `maybe_initialize_memories_from_loader(...)` cho line
  `two-stage` mới;
- nếu cần, chỉ giữ hook cũ cho các line legacy.

**Why**

Nếu không chặn, line mới có nguy cơ init memory sớm ở đầu epoch như hiện tại.

### F. Experiment configs

**Create**

- `configs/model/thesis_multitask_two_stage_window20.yaml`
- `configs/experiment/thesis/exp5/smd__thesis_multitask__offline-pretraining-two-stage-kmeans-machine-3-4-window20__w20__seed11__rtx3090.yaml`
- `configs/experiment/thesis/exp5/smd__thesis_multitask__offline-pretraining-two-stage-kmeans-machine-3-4-window20-smoke__w20__seed11__smoke.yaml`

**Responsibilities**

- line mới phải kế thừa hyperparameter hợp lý từ line active `exp4`, nhưng đổi:
  - `continuous_num_prototypes: 32`
  - `discrete_codebook_size: 60`
  - `training_phase` mặc định cho Stage A
  - `two_stage.stage_a_epochs: 80`
  - `two_stage.stage_b_epochs: 20`
- giữ data config và task config ổn định ở lượt đầu để cô lập ảnh hưởng của
  topology mới.

### G. Comparative and smoke tooling

**Optional later, not in first vertical slice**

- `scripts/preflight_two_stage_server.py`
- `scripts/verify_two_stage_run.py`
- mở rộng `scripts/run_comparative_smd_experiments.py`

**Reason**

Không nên làm ngay trong vertical slice đầu tiên. Chỉ thêm khi line mới đã chạy
được end-to-end bằng runner riêng.

## Minimal Vertical Slice

Vertical slice tối thiểu nên được chia như sau.

### Slice 1. New orchestration without semantic cleanup

Mục tiêu là chạy được end-to-end line mới bằng runner/config mới, nhưng tạm thời
vẫn cho phép một số legacy internals còn tồn tại nếu chúng không tham gia path
chính.

Phạm vi:

- thêm `two_stage` config validation;
- thêm runner `run_two_stage_offline_pretraining.py`;
- thêm config model/experiment mới;
- thêm explicit Stage A -> memory init -> Stage B flow;
- freeze encoder + memories đúng contract;
- Stage B chỉ train fusion + prediction heads.

Tiêu chí hoàn thành:

- smoke run hoàn tất;
- artifact lineage rõ ràng;
- không còn init memory ở đầu epoch của Stage B;
- checkpoint tốt nhất vẫn evaluate được bằng `scripts/evaluate.py`.

### Slice 2. Replace covering selection with k-means

Phạm vi:

- tạo helper k-means;
- đổi continuous init sang `K=32` centroids từ clean train latent tokens;
- đổi discrete init sang `12 x 5 = 60` centroids từ synthetic train latent tokens
  theo class.

Tiêu chí hoàn thành:

- buffer shapes đúng;
- init reproducible theo seed;
- smoke và unit tests pass;
- logs ghi rõ số token thật sự dùng cho từng pool.

### Slice 3. Legacy surface cleanup

Phạm vi:

- giảm hoặc bỏ runtime dependency vào `discrete_assignment` cho line mới;
- hạn chế code path Gumbel-only trong checkpoint/runtime của line mới;
- thêm logging/ablation rõ hơn cho memory init và branch usage.

Tiêu chí hoàn thành:

- line mới dùng `cosine_topk` cleanly;
- line cũ vẫn chạy;
- code path ít mơ hồ hơn.

## Risk and Mitigation

### Risk 1. Two-stage line vô tình reuse semantic `three_stage`

**Mechanism**

Config validation, runner naming, và checkpoint layout cũ có thể làm line mới
trộn với line cũ.

**Mitigation**

- dùng block `two_stage` riêng;
- dùng runner riêng;
- output artifact dưới thư mục `two_stage/`;
- không cho script mới đọc field `three_stage`.

### Risk 2. Memory vẫn bị init sai thời điểm

**Mechanism**

Hook ở `trainer.py` hiện chạy đầu epoch.

**Mitigation**

- bypass hook cũ cho line mới;
- chỉ expose explicit init API ở boundary Stage A -> Stage B;
- thêm test khẳng định Stage B không reinitialize memory.

### Risk 3. K-means không ổn định khi token pool nhỏ hoặc lệch class

**Mechanism**

Một số lớp synthetic có thể có ít token hơn số centroid yêu cầu.

**Mitigation**

- fallback policy rõ ràng khi `N < K`;
- log token count theo class;
- smoke test cho class-thin path;
- giữ batch/data/task config ổn định ở line đầu.

### Risk 4. Fusion collapse sang một memory branch

**Mechanism**

Stage B chỉ train fusion và heads; nếu tín hiệu yếu, một branch có thể bị bỏ qua.

**Mitigation**

- log branch contribution hoặc similarity statistics trong `aux`;
- giữ `lambda_recon`, `lambda_cls`, contrastive settings từ line active lúc đầu;
- dùng evaluation và synth validation y như line active để so sánh công bằng.

### Risk 5. Cleanup Gumbel path làm hỏng line cũ

**Mechanism**

`thesis_multitask.py` đang dùng chung cho nhiều line config.

**Mitigation**

- cleanup Gumbel để ở Slice 3, không làm trong vertical slice đầu;
- line mới route bằng explicit config flags;
- regression test cho config legacy.

### Risk 6. Metric interpretation bị trộn giữa synth validation và real test

**Mechanism**

Line mới có thể đổi topology nhưng vẫn dùng monitor cũ.

**Mitigation**

- giữ metric surface cũ cho vòng đầu: `val_synth_vus_pr`;
- report riêng synth validation và real test;
- không đổi evaluator threshold logic trong line đầu.

## Open Questions

Các câu hỏi này không chặn plan sơ bộ, nhưng sẽ cần khóa trước khi qua bước
`prompts/3_structure_prompt.md`.

1. K-means backend nên là torch-native helper nội bộ hay thêm dependency ngoài.
   Khuyến nghị hiện tại là **torch-native helper nội bộ** để giảm phụ thuộc.
2. Continuous pool có nên lấy từ clean train windows thuần hay từ normal tokens
   trong synthetic-augmented train batch như code hiện tại. Design SSOT nghiêng
   về **clean train split**.
3. Stage B có nên giữ contrastive loss tắt hoàn toàn hay giữ cờ config nhưng
   default bằng `0.0`. Design hiện tại nghiêng về **tắt ở Stage B** vì Stage B
   chỉ train fusion + heads.
4. Discrete class pool có nên lấy mọi token trong synthetic window của class đó
   hay chỉ các token nằm trong đoạn bị inject anomaly. Nếu chưa có contract rõ,
   đây là điểm cần chốt ở bước structure/detail vì nó ảnh hưởng semantics của
   codebook.

## Validation and Test Plan

### Unit tests

**Create**

- `tests/test_two_stage_config_validation.py`
- `tests/test_two_stage_kmeans_initialization.py`
- `tests/test_two_stage_freeze_contract.py`

**Coverage**

- `two_stage` config hợp lệ và bị từ chối khi sai tổng epoch;
- continuous memory init trả đúng shape `[32, H]`;
- discrete memory init trả đúng shape `[60, H]`;
- per-class fallback hoạt động khi token count thấp;
- Stage B freeze đúng encoder + continuous memory + discrete memory.

### Integration / smoke tests

**Create**

- `tests/test_two_stage_runner_smoke.py`

**Coverage**

- runner mới materialize manifest đúng;
- Stage A xong mới init memory;
- Stage B không reinit memory;
- best checkpoint cuối cùng load được bằng `scripts/evaluate.py`.

### Regression tests

**Update or add**

- `tests/test_config_loading.py`
- test load cho một config `three_stage` legacy
- test load cho config `two_stage` mới

**Coverage**

- config cũ không vỡ;
- config mới không bị validator cũ chặn;
- model vẫn tôn trọng output contract chung.

## Proposed Implementation Order

1. Thêm `two_stage` config validation trong `src/core/config.py`.
2. Tạo model config và experiment config mới cho line `exp5`.
3. Tạo runner `scripts/run_two_stage_offline_pretraining.py`.
4. Sửa `thesis_multitask.py` để có explicit API cho Stage A end-of-stage memory
   initialization và Stage B freeze contract.
5. Bypass hook init memory đầu epoch trong `trainer.py` cho line mới.
6. Chạy smoke line mới với memory init cũ nếu cần để xác nhận orchestration
   boundary trước.
7. Thêm helper k-means và chuyển memory init sang k-means.
8. Bổ sung unit tests và smoke tests.
9. Sau khi line mới ổn định mới dọn Gumbel-only machinery cho path mới.

## Final Recommendation

Kế hoạch sơ bộ nên đi theo hướng:

- tạo **một line thí nghiệm mới `exp5 two-stage`**;
- thêm **runner riêng** thay vì ép semantic vào trainer chung;
- giữ **batch/output contract** và **data/task configs** ổn định;
- hoàn thành **vertical slice orchestration trước**;
- chỉ sau đó mới thay `covering selection` bằng **k-means** và dọn legacy path.

Đây là hướng đơn giản nhất, ít rủi ro nhất, và phù hợp nhất với trạng thái
codebase hiện tại lẫn design SSOT mới.
