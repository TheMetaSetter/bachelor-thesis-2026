---
date: 2026-06-24 18:51:13 +0700
researcher: Codex
git_commit: c1c3065ee611bab9b0d5c1071e7a58f62b99d6c7
branch: dev
repository: bachelor-thesis-2026
topic: "Audit current codebase for contradictions, reader confusion, and user-facing clarity in the three-stage offline pretraining pipeline"
tags: [research, codebase-audit, three-stage, semantics, usability]
status: complete
last_updated: 2026-06-24
last_updated_by: Codex
---

# Research: Audit current codebase for contradictions, reader confusion, and user-facing clarity in the three-stage offline pretraining pipeline

**Date**: 2026-06-24 18:51:13 +0700
**Researcher**: Codex
**Git Commit**: `c1c3065ee611bab9b0d5c1071e7a58f62b99d6c7`
**Branch**: `dev`

## Research Question

Rà soát lại codebase để tìm các chỗ mâu thuẫn, làm rối người đọc, hoặc không thân thiện với người dùng, đặc biệt quanh three-stage offline pretraining, synthetic multiclass balancing, validation semantics, và lệnh chạy bằng `tmux`.

## Summary

Codebase hiện đã khóa được contract chính cho run three-stage `300` epoch, stage naming canonical mới, WandB online logging, và bước evaluation sau khi train xong. Tuy nhiên vẫn còn một số điểm dễ gây hiểu lầm cho người đọc.

Điểm đáng chú ý nhất là có sự lệch giữa default runtime mới và default fallback cũ ở lớp config validation. Ngoài ra, một vài tên biến hoặc alias cũ vẫn còn sống trong code và test, nên dù runtime chính có thể chạy đúng, người đọc rất dễ hiểu sai semantics thực tế.

## Detailed Findings

### Data Preparation

- Synthetic anomaly injection hiện mặc định dùng taxonomy `redlamp_multiclass` và `train_balance_classes=True` trong injector runtime, nghĩa là ưu tiên chia gần đều trên 12 class thay vì dùng Bernoulli theo `anomaly_probability` khi train balanced multiclass [`src/data/augment.py:38-50`, `src/data/augment.py:796-856`].
- Tuy vậy, `anomaly_probability` vẫn được giữ ngay trong task YAML active và constructor defaults, nên người đọc rất dễ tưởng `0.5` nghĩa là “cân bằng 12 class”, trong khi thực tế khi `train_balance_classes=True` thì quota được tạo bằng `_balanced_class_quota()` và `anomaly_probability` không còn quyết định tỷ lệ class train nữa [`configs/task/multitask_tsad_redlamp_multiclass_window20.yaml:1-25`, `src/data/augment.py:821-856`].

### Modeling and Training

- Contract Stage 3 canonical đã là `stage3_memory_initialization_and_fusion_warmup`, và experiment active phân bổ đúng `50 + 70 + 20 + 20 + 140 = 300` [`configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml:42-51`].
- Runner cũng enforce chính xác budget 300 epoch và build manifest theo phase order canonical [`scripts/run_three_stage_offline_pretraining.py:40-46`, `scripts/run_three_stage_offline_pretraining.py:89-120`, `scripts/run_three_stage_offline_pretraining.py:243-298`].
- Tuy nhiên, lớp config vẫn còn giữ alias legacy `stage3_prototype_warmup_epochs`, normalize ngược ra cả key legacy, và test còn assert key legacy vẫn tồn tại trong loaded config. Điều này làm người đọc mới rất dễ nghĩ stage name cũ vẫn là contract chính [`src/core/config.py:19-22`, `src/core/config.py:120-143`, `tests/test_smd_machine_3_4_three_stage_config_loading.py:24-46`].
- Tài liệu discussion context cũng còn nhắc trực tiếp `stage3_prototype_warmup` trong logging contract, nên wording trong `documents/` và wording runtime hiện chưa sạch hoàn toàn [`documents/logs/06-17-2026/detail/detail-offline-pretraining-three-stage-discussion-context.md:376-383`].
- Có một lệch default quan trọng ở config validation: `train_balance_classes` fallback trong `src/core/config.py` vẫn là `False`, còn `classification_label_mode` fallback vẫn là `"binary"`, trong khi injector/model defaults hiện đã chuyển sang multiclass balanced làm mặc định. Nếu một config mới bỏ sót hai field này, người đọc có thể suy ra semantics khác với runtime mặc định mà model/injector đang thể hiện [`src/core/config.py:878-883`, `src/core/config.py:987-999`, `src/data/augment.py:41-50`, `src/models/redlamp_mlp_baseline.py:106-117`, `src/models/thesis_multitask.py:392-400`].
- Trong `ThesisMultitaskModel.from_flat_kwargs`, chỉ có nhánh fallback sang `"binary"` khi `num_classes == 2`; không có nhánh tương đương để tự ghi rõ `"redlamp_multiclass"` khi `num_classes == 12`. Code này không sai runtime chính, nhưng làm contract default bị thiếu đối xứng và khó đọc hơn [`src/models/thesis_multitask.py:566-571`].
- `RedLampMLPBaseline` vẫn còn tham số tên `balance_binary_classes_within_batch`, nhưng model lại ép `classification_label_mode="redlamp_multiclass"` khi tạo injector. Tên tham số này vì vậy không còn mô tả đúng use case active nữa và dễ làm người đọc tưởng baseline còn mode binary là main path [`src/models/redlamp_mlp_baseline.py:117`, `src/models/redlamp_mlp_baseline.py:241-267`].

### Evaluation

- Trainer hiện chạy `validation_step()` trên clean `val`, rồi mới chạy `realistic_validation_step()` hoặc `synthetic_validation_step()` như một validation phụ thứ hai. Tên gọi `val_realistic` là đúng theo prior anomaly, nhưng không phải là một loader khác tên “realistic validation set” [`src/engine/trainer.py:680-735`].
- Cụ thể hơn, `val_realistic` vẫn chạy trên chính `val_loader`; cái “realistic” đến từ việc trainer tính anomaly rate từ test prior SMD rồi truyền vào `prepare_realistic_validation_epoch(anomaly_probability=...)` trước khi inject synthetic anomalies lên batch validation [`src/engine/trainer.py:540-558`, `src/engine/trainer.py:698-719`, `src/models/thesis_multitask.py:1847-1861`, `src/models/redlamp_mlp_baseline.py:274-282`].
- Điều này có nghĩa là tên `prepare_realistic_validation_epoch(anomaly_probability)` cũng hơi nguy hiểm về UX: nó nghe như chuẩn bị một split realistic riêng, nhưng thật ra chỉ đổi prior injection cho synthetic validation trên `val_loader`.
- `validation_step()` của cả hai model vẫn tính clean `val_loss` với `classification_weight=0.0`, còn `realistic_validation_step()` mới mang classification component và metrics multiclass/VUS-PR tương ứng [`src/models/thesis_multitask.py:3228-3250`, `src/models/redlamp_mlp_baseline.py:673-680`].
- Checkpoint monitor của experiment three-stage đang đặt đúng vào `val_realistic_vus_pr`, và WandB logging đang bật `use_wandb: true`, `wandb_mode: online` trong config active [`configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml:26-34`].
- Runner three-stage thực sự có bước `evaluation` sau khi train 5 phase xong; manifest trỏ checkpoint cuối của `multitask_pretraining` vào `scripts/evaluate.py` [`scripts/run_three_stage_offline_pretraining.py:283-295`, `scripts/run_three_stage_offline_pretraining.py:301-320`, `scripts/run_three_stage_offline_pretraining.py:666-681`].

### User-Facing Launch and Logging

- Script `scripts/launch_tmux_three_stage_experiment.sh` hiện khá thân thiện cho việc chạy server: in ra `tmux session`, `log path`, `preflight summary`, `run verification summary`, và `attach command` ngay sau launch [`scripts/launch_tmux_three_stage_experiment.sh:182-230`].
- Tuy nhiên script mới in rõ phase names canonical ở `--dry-run`; còn đường chạy thường chỉ in đường dẫn và session name. Người dùng không nhìn thấy trực tiếp các hyperparameter stage nếu không mở config hoặc dry-run.
- Vì output hiện gói toàn bộ `preflight && train && verify` vào một command dài trong tmux, log file là nguồn sự thật thực tế; điều này ổn, nhưng vẫn đòi hỏi người dùng biết phải đọc file log thay vì chỉ attach tmux.

## Code References

- `src/core/config.py:19-22` - stage 3 canonical key và legacy key cùng tồn tại
- `src/core/config.py:878-883` - `train_balance_classes` fallback vẫn là `False`
- `src/core/config.py:987-999` - `classification_label_mode` fallback vẫn là `"binary"`
- `src/data/augment.py:38-50` - injector defaults đã là balanced multiclass
- `src/data/augment.py:796-856` - logic quota cân bằng 12 class
- `src/models/thesis_multitask.py:566-571` - fallback flat kwargs không đối xứng cho mode multiclass
- `src/models/thesis_multitask.py:1847-1861` - realistic validation chỉ đổi anomaly prior cho injector
- `src/models/thesis_multitask.py:3228-3250` - clean `val` tách khỏi `val_realistic`
- `src/models/redlamp_mlp_baseline.py:241-267` - baseline vẫn dùng tên `balance_binary_classes_within_batch` dù injector chạy multiclass
- `src/models/redlamp_mlp_baseline.py:673-680` - clean `val` và `val_realistic` tách riêng
- `src/engine/trainer.py:540-558` - realistic anomaly prior lấy từ SMD test window rate
- `src/engine/trainer.py:680-735` - trainer chạy clean validation trước, validation phụ sau
- `scripts/run_three_stage_offline_pretraining.py:89-120` - enforce exact 300-epoch budget
- `scripts/run_three_stage_offline_pretraining.py:283-295` - manifest chứa training stages và evaluation
- `scripts/run_three_stage_offline_pretraining.py:666-681` - execution thật sự chạy evaluation sau train
- `scripts/launch_tmux_three_stage_experiment.sh:182-230` - tmux launch UX và logging output

## Pipeline Documentation

Pipeline active hiện tại cho run three-stage machine `3-4` là:

1. preflight server check;
2. stage 1 classification;
3. stage 1 reconstruction;
4. stage 2 recovery;
5. stage 3 memory initialization and fusion warm-up;
6. multitask pretraining;
7. evaluation bằng `scripts/evaluate.py` trên checkpoint tốt nhất của phase cuối.

Validation runtime trong training loop được chia thành:

- clean `val` qua `validation_step()`;
- auxiliary validation qua `realistic_validation_step()` nếu model hỗ trợ;
- nếu không có realistic path thì fallback sang `synthetic_validation_step()`.

## Historical Context (from documents/)

Tài liệu design và discussion đã được chuyển khá nhiều sang wording canonical mới cho Stage 3, nhưng vẫn còn dư âm của tên `stage3_prototype_warmup` trong một số chỗ. Vì config loader còn duy trì alias này cho backward compatibility, nguy cơ “legacy wording quay lại thành wording tưởng như chính thức” vẫn còn hiện hữu.

## Open Questions

- Có nên tiếp tục giữ `anomaly_probability` trong task YAML active khi `train_balance_classes=True`, hay cần một comment/config-help nói rõ field này không còn điều khiển class balance trong mode balanced multiclass?
- Có nên đổi tên `prepare_realistic_validation_epoch(anomaly_probability)` sang một tên hẹp nghĩa hơn để tránh người đọc hiểu lầm rằng có một realistic loader riêng?
- Có nên dừng inject ngược lại legacy stage-3 alias vào loaded config, thay vào đó chỉ chấp nhận nó ở input rồi normalize sang canonical key duy nhất?

## Follow-up 2026-06-24 19:00 +0700

Người dùng yêu cầu liệt kê chính xác các dòng code liên quan đến bốn điểm audit chính. Các line quan trọng là:

### 1. Default runtime mới lệch với fallback cũ ở config layer

- `src/core/config.py:878-883` - `train_balance_classes` fallback vẫn là `False`
- `src/core/config.py:987-999` - `classification_label_mode` fallback vẫn là `"binary"`
- `src/data/augment.py:41-50` - injector defaults đã là `anomaly_probability=0.5`, `train_balance_classes=True`, `classification_label_mode="redlamp_multiclass"`
- `src/models/thesis_multitask.py:393-400` - thesis multitask synthetic defaults đã là balanced multiclass
- `src/models/redlamp_mlp_baseline.py:106-117` - baseline defaults đã là balanced multiclass
- `src/models/thesis_multitask.py:566-571` - chỉ còn fallback bất đối xứng sang `"binary"` khi `num_classes == 2`

### 2. `anomaly_probability: 0.5` dễ gây hiểu lầm khi train balanced multiclass

- `configs/task/multitask_tsad_redlamp_multiclass_window20.yaml:5-10` - task active đồng thời ghi `classification_label_mode: redlamp_multiclass`, `anomaly_probability: 0.5`, `train_balance_classes: true`
- `configs/task/multitask_tsad.yaml:5-10` - task common cũng ghi cùng pattern
- `src/data/augment.py:821-846` - chỉ khi `train_balance_classes=False` thì `anomaly_probability` mới quyết định injection decisions
- `src/data/augment.py:848-856` - khi `train_balance_classes=True` thì class labels được tạo từ `_balanced_class_quota()`

### 3. `val_realistic` thực ra vẫn chạy trên `val_loader`

- `src/engine/trainer.py:550-558` - realistic anomaly rate được suy ra từ test-window statistics
- `src/engine/trainer.py:686-690` - clean validation chạy trên `val_loader`
- `src/engine/trainer.py:698-719` - realistic validation cũng chạy trên chính `val_loader`, chỉ đổi step method và pointwise label key
- `src/models/thesis_multitask.py:1847-1855` - `prepare_realistic_validation_epoch()` chỉ set `synthetic_validation_injector.anomaly_probability`
- `src/models/redlamp_mlp_baseline.py:274-282` - baseline làm đúng cùng semantics
- `src/models/thesis_multitask.py:3228-3250` - clean `val` tách khỏi `val_realistic`
- `src/models/redlamp_mlp_baseline.py:673-680` - baseline cũng tách `validation_step()` khỏi `realistic_validation_step()`

### 4. Legacy Stage 3 wording vẫn còn sống

- `src/core/config.py:19-22` - khai báo đồng thời legacy key và canonical key
- `src/core/config.py:101-126` - normalize hai chiều giữa legacy key và canonical key
- `src/core/config.py:130-143` - validator vẫn cho phép cả hai key
- `tests/test_smd_machine_3_4_three_stage_config_loading.py:39-46` - test vẫn assert loaded config có cả canonical key lẫn legacy key
- `tests/test_smd_machine_3_4_three_stage_config_loading.py:64-94` - test vẫn kiểm tra behavior normalize/reject cho alias cũ
- `documents/logs/06-17-2026/detail/detail-offline-pretraining-three-stage-discussion-context.md:378-381` - wording tài liệu vẫn dùng `stage3_prototype_warmup`
