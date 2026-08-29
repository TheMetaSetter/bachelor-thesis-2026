---
date: 2026-08-22  Asia/Ho_Chi_Minh
researcher: OpenAI Codex
topic: "Chuẩn bị ablation direct branch routing cho THESIS"
status: complete
revision: dfb205b761ba1cc4c9f8bcbf43e81ec902ad8cd5
branch: dev
---

# Research: Chuẩn bị ablation direct branch routing cho THESIS

## Summary

Runtime hiện chưa hỗ trợ route trực tiếp từ hai prototype branch vào prediction head.
Runtime chỉ hỗ trợ `task_specific_concat_projection` và `learnable_sigmoid_scalars`.

Để chạy ablation mà không chạy lại Stage A, cần làm bốn việc:

1. Thêm mode `direct_branch_routing` vào model runtime.
2. Cho reconstruction head nhận continuous latent tensor và classification head nhận discrete latent tensor, kể cả trong đường Monte Carlo evaluation.
3. Giữ lại hai concat projection module trong model để checkpoint Stage A hoặc `stage_b_init.pt` cũ vẫn load được; nhưng freeze và không dùng chúng trong mode mới.
4. Dùng một config chỉ chạy Stage B từ checkpoint khởi tạo Stage B đã có. Không gọi runner two-stage hiện tại vì runner này luôn chạy Stage A.

Chưa có code hoặc config runtime nào được sửa trong lần nghiên cứu này.

## Research question

Xác định những phần codebase và thông tin cấu hình cần điều chỉnh để chạy thí nghiệm:

- bỏ qua hai fusion head;
- truyền continuous branch trực tiếp vào reconstruction head;
- truyền discrete branch trực tiếp vào classification head;
- giữ nguyên các thiết kế khác;
- không chạy lại Stage A.

## System context

Public entrypoint của model là `ThesisMultitaskModel`.
Offline training đi qua `scripts.cli.train`.
Runner benchmark two-stage là `scripts.experiments.run_two_stage_offline_pretraining` và wrapper `scripts.benchmarks.run_thesis_offline_benchmark`.

Stage A dùng passthrough. `_phase_uses_prototype_path()` chỉ trả về `True` cho `stage_b_fusion_finetuning`, và Stage A gọi `_build_phase_passthrough_outputs()`.
Vì vậy thay đổi này chỉ ảnh hưởng đường prototype/fusion của Stage B và các đường evaluation dùng model Stage B.

## Implemented behavior found in the code

### Forward chính

Trong `thesis_multitask_routing_forward_helpers.py`, Stage B lấy:

1. `continuous_outputs["prototype_context"]`;
2. `discrete_outputs["quantized_hidden"]`;
3. `_compute_fusion_outputs(...)`;
4. `reconstruction_head(hidden_reconstruction)`;
5. `classification_head(flattened_hidden_classification)`.

Hiện tại `_compute_fusion_outputs()` chỉ có hai nhánh:

- `task_specific_concat_projection`: nối hai tensor theo chiều cuối rồi dùng hai projection riêng;
- nhánh còn lại: trộn hai tensor bằng `alpha` và `beta`.

Code evidence: `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:357-461` và `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py:145-270`.

### Monte Carlo evaluation

Evaluation tạo `continuous_samples` và `discrete_samples` có shape `[B, M, L, H]`, sau đó gọi `_build_sampled_fusion_hidden()` trước khi chạy hai prediction head.
Mode mới phải được xử lý ở đây nữa; chỉ sửa forward chính sẽ làm train và MC evaluation dùng hai route khác nhau.

Code evidence: `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:14-67` và `:138-263`.

### Stage A

Stage A không dùng prototype path. `_build_phase_passthrough_outputs()` trả cùng encoder hidden cho hai task.
Do đó không cần sửa logic Stage A và không cần chạy lại Stage A cho ablation này.

Code evidence: `src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py:54-55` và `src/models/thesis_multitask_impl/thesis_multitask_state_passthrough_mixin.py:9-64`.

## Code adjustments required

### 1. Mở rộng config runtime

`ActiveRuntimeConfig` hiện chỉ chấp nhận hai giá trị fusion mode.
Thêm `direct_branch_routing` vào tập giá trị hợp lệ.

Code evidence: `src/models/thesis_multitask_impl/thesis_multitask_components.py:210-247`.

`config_model_validation.py` đã cho phép khóa `fusion_mode`, nên không cần thêm khóa mới ở validator cấp experiment.
Validator dataclass vẫn phải nhận giá trị mới.

### 2. Thêm route trực tiếp cho forward

Trong `_compute_fusion_outputs()`, mode mới cần trả:

```text
hidden_reconstruction = continuous_hidden
hidden_classification = discrete_hidden
```

`alpha` và `beta` nên trả tensor zero cùng batch size để giữ schema output hiện tại. `aux["fusion"]["fusion_mode"]` phải ghi rõ `direct_branch_routing`.

### 3. Thêm route trực tiếp cho MC evaluation

Trong `_build_sampled_fusion_hidden()`, mode mới cần trả:

```text
sampled_hidden_reconstruction = continuous_samples
sampled_hidden_classification = discrete_samples
```

Phần còn lại của MC evaluation giữ nguyên: reconstruction head tạo `reconstruction_samples`, classification head tạo `logits_samples`, rồi code tính mean, point score và uncertainty như hiện tại.

### 4. Không xóa projection module khỏi model

Model luôn tạo `classification_concat_projection` và `reconstruction_concat_projection` trong `_build_task_heads()`.
Checkpoint Stage A và `stage_b_init.pt` cũ chứa state của các module này.

Vì vậy mode mới nên bỏ qua các module trong forward nhưng vẫn giữ chúng trong state dict. Nếu xóa module, hàm `_load_stage_a_state_into_stage_b_model()` hiện chỉ cho phép hai unexpected keys của `discrete_assignment`; mismatch mới sẽ làm load checkpoint thất bại.

Code evidence: `src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py:293-342` và `scripts/experiments/run_two_stage_offline_pretraining.py:224-241`.

### 5. Freeze các module không còn dùng trong Stage B

`_configure_trainable_parameters_for_phase()` hiện freeze concat projection ở Stage A, nhưng ở Stage B vẫn để concat projection trainable.
Mode mới cần freeze ít nhất:

- `reconstruction_concat_projection`;
- `classification_concat_projection`;
- hai fusion gate;
- `alpha_logit` và `beta_logit`.

Encoder và hai memory bank tiếp tục giữ contract Stage B hiện tại.

Code evidence: `src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py:95-177` và danh sách trainable module ở `src/models/thesis_multitask_impl/thesis_multitask_state_schedule_mixin.py:137-155`.

## Configuration adjustments required

### Model override

Không cần sửa model base dùng cho các run cũ.
Config ablation nên dùng override:

```yaml
model_overrides:
  fusion_mode: direct_branch_routing
  training_phase: stage_b_fusion_finetuning
```

`load_experiment_config()` đã hỗ trợ `model_overrides`, nên cấu hình ablation có thể giữ nguyên model base và chỉ đổi mode.

Code evidence: `src/core/config.py:747-807`.

### Stage B only

Config chạy riêng Stage B cần có:

```yaml
epochs: 5
stage_name: stage_b_fusion_finetuning
initialization_checkpoint_path: <existing-stage-b-init.pt>
model_overrides:
  fusion_mode: direct_branch_routing
  training_phase: stage_b_fusion_finetuning
```

Các giá trị khác phải giữ nguyên run đối chứng: dataset, entity, seed, windowing, optimizer, loss weights, discrete query settings, memory settings, validation protocol và logging.

Không đặt `stage_a_multitask_epochs` trong config Stage B-only. Nếu giữ `two_stage` với epoch budget hiện tại, validator sẽ buộc `epochs` bằng tổng Stage A + Stage B.

### Variant name và output path

Không ghi đè O0 hoặc O1 cũ. Cần tên variant mới, ví dụ:

```yaml
experiment_variant: direct_branch_routing_v1
```

Output path cũng phải tách riêng để checkpoint, threshold và metrics không bị trộn với run concat projection.

## Why the current two-stage runner must not be used as-is

`execute_two_stage_plan()` luôn chạy `training_commands[0]`, tức Stage A, trước khi gọi `_prepare_stage_b_initialization_checkpoint()`.
Không có cờ hiện tại để bỏ qua riêng Stage A.

Code evidence: `scripts/experiments/run_two_stage_offline_pretraining.py:364-460`.

Vì yêu cầu không chạy lại Stage A, không dùng đường benchmark two-stage mặc định cho run này.
Có hai dữ liệu khởi tạo có thể tái sử dụng:

1. `stage_b_init.pt` cũ: phù hợp nhất vì memory đã được khởi tạo và direct routing không đổi encoder hay memory.
2. Stage A best checkpoint cũ: chỉ dùng nếu cần tạo lại memory initialization; thao tác này không phải train Stage A nhưng vẫn tạo thêm transition artifact.

Local checkout hiện không xác nhận có `outputs/...` checkpoint. Inventory trước đó chỉ ghi nhận các path historical như `outputs/benchmark/smd/thesis/O0/machine_1_6/seed6/two_stage/initializations/stage_b_init.pt`; cần kiểm tra file thật trước khi chạy.

## Online impact

Online adapter gọi `_compute_fusion_outputs()` của model THESIS, nên sau khi runtime có mode mới, online scoring sẽ tự dùng direct routing khi nó load checkpoint direct.

Code evidence: `src/models/online_impl/online_adaptation_helpers.py:50-104`.

Nếu chỉ chạy offline ablation thì không cần sửa online code.
Nếu chạy thêm A0/A1/A2, cần tạo online config trỏ tới offline variant mới và threshold artifact được tạo từ đúng direct Stage B checkpoint. `checkpoint_resolution.py` chọn checkpoint theo `offline_variant`, entity, seed, benchmark mode và stage name; online runtime còn kiểm tra checksum checkpoint trong threshold artifact.

Code evidence: `src/engine/online_tta/checkpoint_resolution.py:31-113` và `src/engine/online_tta/online_engine_run.py:103-145`.

## Tests and validation evidence

Đã chạy các test hẹp, không chạy Stage A:

```text
.venv/bin/python -m pytest -q \
  tests/models/test_multitask_shapes.py \
  tests/models/test_thesis_multitask_config_refactor.py \
  tests/benchmarks/test_two_stage_orchestration_dry_run.py
```

Kết quả: `21 passed, 1 skipped, 1 failed`.

Failure hiện có không liên quan đến direct routing: `test_multitask_model_uses_shared_three_layer_mlp_depth` giả định `model.encoder.network` là iterable MLP, nhưng config mặc định hiện tạo `SimpleWindowCnnEncoder`.

Một kiểm tra trực tiếp cũng xác nhận mode mới chưa được implement:

```text
ValueError: fusion_mode must be one of: task_specific_concat_projection, learnable_sigmoid_scalars
```

### Tests cần thêm sau khi implement

1. Config accepts `direct_branch_routing`.
2. Deterministic Stage B output có `hidden_reconstruction == continuous_branch["prototype_context"]`.
3. Deterministic Stage B output có `hidden_classification == discrete_branch["quantized_hidden"]`.
4. MC helper chuyển continuous samples vào reconstruction route và discrete samples vào classification route.
5. Stage B direct model vẫn load được state dict từ `stage_b_init.pt` cũ.
6. Trainable module report không liệt kê hai concat projection và hai fusion gate trong direct Stage B.
7. Stage B-only command không gọi Stage A subprocess.

## Evidence classification and limitations

- **Implemented:** hai fusion mode hiện tại, Stage A passthrough, Stage B prototype path, MC fusion path, checkpoint loading và config override.
- **Configured:** benchmark hiện chọn `task_specific_concat_projection` trong `configs/model/thesis_multitask_two_stage_window20.yaml`.
- **Tested:** các shape/config/orchestration tests nêu trên; một test nền đang fail do encoder-family mismatch.
- **Documented:** spec two-stage hiện mô tả Stage B train task-specific fusion heads và `fusion_mode: task_specific_concat_projection`.
- **Unknown:** checkpoint Stage A hoặc `stage_b_init.pt` thật có tồn tại trong local checkout hay chỉ ở remote; chưa được mở checkpoint trong lần nghiên cứu này.

## Conclusion

Phần bắt buộc phải sửa là model runtime ở ba điểm: giá trị config, deterministic fusion và MC fusion. Phần cấu hình bắt buộc là một experiment config mới cho Stage B-only, dùng `direct_branch_routing`, tên variant mới và checkpoint khởi tạo Stage B có sẵn.

Không cần chạy lại Stage A. Không nên xóa projection modules vì chúng cần cho checkpoint compatibility. Sau khi implement và kiểm tra hẹp, mới chạy một Stage B-only smoke flow; chỉ sau khi smoke flow đạt mới chạy benchmark chính.
