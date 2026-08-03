---
date: 2026-08-03T00:00:00+07:00
researcher: OpenAI Codex
topic: "Ontology, runtime flow, data flow, and IGCSE pseudocode for the THESIS offline pre-training phase"
status: complete
revision: e58602f45ee5439a1e001f060e8ea640aeddde9c
branch: dev
---

# Research: Offline pre-training ontology and flows

## Summary

Pha offline hiện hành có tên chính thức `offline_pretraining_phase`. Pha này có đúng hai stage theo thứ tự `stage_a_multitask_pretraining` rồi `stage_b_fusion_finetuning`. `stage_b_memory_initialization` là ranh giới chuyển tiếp giữa hai stage, không phải stage thứ ba. `offline_evaluation` chạy sau khi training kết thúc và cũng không phải một training stage.

Runtime thật bắt đầu ở `scripts.run_thesis_offline_benchmark`. Wrapper nạp experiment config và protocol config, tạo `two_stage_run_manifest`, chạy Stage A trong child process, tạo `stage_b_initialization_checkpoint`, chạy Stage B, chạy evaluation, rồi export score, metric, threshold và retention artifacts. Model public duy nhất của THESIS offline là `ThesisMultitaskModel`.

Code và specification hiện còn ba khác biệt cần giữ nguyên trạng thái trong ontology thay vì che đi:

1. Source dùng field `training_phase` và constant `TWO_STAGE_*_PHASE_NAME` để chứa tên stage. Đây là tên field lịch sử; semantics thực tế là `stage_name`.
2. Specification yêu cầu discrete memory pool của class 1–11 chỉ chứa injected anomaly tokens. Code hiện tại gom toàn bộ latent tokens của window thuộc từng class và dùng pooled fallback khi thiếu class.
3. Config offline O1 thể hiện variant trong `experiment_name` và `experiment_variant`, nhưng không có root field `offline_variant`. Artifact collector vì vậy có thể fallback thành `O0`.

## Research question

Nghiên cứu runtime và data contracts hiện hành để tạo một ontology thống nhất tên object của pha offline; sau đó dùng đúng các tên đó trong runtime flow, data flow và IGCSE pseudocode. Dùng ontology offline làm gốc để chuẩn hóa ontology và desired-flow pseudocode của pha online.

## System context

`offline_pretraining_phase` nhận raw SMD sequences, fit scaler chỉ bằng train split, tạo windows, huấn luyện `ThesisMultitaskModel`, khởi tạo hai memory banks từ train data, chọn Stage B checkpoint, hiệu chỉnh threshold bằng clean validation, rồi đánh giá synthetic validation và test.

Quan hệ lớn nhất là:

```text
offline_pretraining_phase
  contains stage_a_multitask_pretraining
  transitions through stage_b_memory_initialization
  contains stage_b_fusion_finetuning
  produces stage_b_best_checkpoint
  is followed by offline_evaluation
  produces threshold_artifact used again by online_tta_phase
```

## Execution path

1. `scripts.run_thesis_offline_benchmark` chuyển quyền cho benchmark wrapper.
2. `run_thesis_offline_benchmark(...)` nạp config, kiểm tra protocol và epoch budget, rồi tạo `two_stage_run_manifest`.
3. Orchestrator sinh một generated config cho mỗi stage. Cả `stage_name` và field tương thích cũ `training_phase` đều nhận cùng stage identifier.
4. Orchestrator chạy `scripts.train` cho `stage_a_multitask_pretraining`.
5. `scripts.cli.train.run_training_experiment(...)` tạo dataset bundle, model, optimizer, scheduler, logger và `Trainer`.
6. `Trainer.train(...)` lặp qua epoch và batch. Mỗi train batch đi qua `ThesisMultitaskModel.training_step(...)`, `loss.backward()`, gradient clipping và optimizer step. Mỗi epoch tiếp tục chạy clean validation, synthetic validation, metric aggregation và checkpoint selection.
7. Stage A dùng direct encoder-to-head path. Prototype path bị bypass. Loss gồm reconstruction, classification, two-view contrastive và thêm point-score loss nếu offline variant là O1.
8. Orchestrator nạp `stage_a_best_checkpoint` vào model cấu hình Stage B, rebuild deterministic train loader, thu latent token pools, chạy k-means, tạo continuous prototype bank, discrete codebook và anomaly verification metadata, rồi lưu `stage_b_initialization_checkpoint`.
9. Orchestrator chạy `scripts.train` cho `stage_b_fusion_finetuning`. Stage B nạp initialization checkpoint, freeze encoder và dùng frozen memory retrieval. Trainable surface hiện hành gồm hai concat projections và hai task heads.
10. Orchestrator chạy evaluation command bằng `stage_b_best_checkpoint`.
11. Benchmark wrapper lại nạp Stage B checkpoint để tạo artifact bundle. Evaluator chạy clean validation trước, khôi phục window point scores về entity timeline và hiệu chỉnh `offline_point_threshold` từ clean-validation scores.
12. Evaluator chạy synthetic validation và test bằng threshold đã cố định. Wrapper export scores, metrics, uncertainty summary, threshold artifact, provenance và benchmark report.

## Detailed findings

### Phase, stage, boundary, and operation are different object types

Code comment xác nhận offline pre-training là pha lớn; Stage A và Stage B là hai stage bên trong pha đó. Orchestrator vẫn gọi constants là `*_PHASE_NAME`, nhưng record mà nó tạo dùng key `stage_name`. Ontology phải ưu tiên object type thật thay vì lặp lại tên constant lịch sử.

Memory initialization diễn ra sau Stage A checkpoint và trước Stage B training. Nó tạo một checkpoint khởi tạo riêng. Vì vậy `stage_b_memory_initialization` là một transition operation, còn `stage_b_initialization_checkpoint` là output artifact của operation đó.

### Stage A data and computation

Offline batch chuẩn có `x`, `point_labels`, `mask`, `timestamps` và `meta`. Synthetic augmentation bổ sung `classification_labels`, `synthetic_anomaly_mask` và `augmentation_metadata`.

Stage A không dùng final prototype banks. Encoder tạo `latent_tokens`; reconstruction head tạo `reconstruction`; classification head tạo `classification_logits`. Model tính `reconstruction_loss`, `classification_loss`, `two_view_contrastive_loss`, và O1 có thêm `point_score_loss`. Model trả `stage_a_total_loss` cho trainer để backpropagate.

### Memory initialization semantics

Implemented continuous pool chỉ lấy tokens thuộc normal windows và normal positions của các synthetic batches. Code loại injected positions khỏi continuous pool.

Implemented discrete pool hiện gom toàn bộ tokens của mỗi synthetic class window. Điều này rộng hơn specification, vốn yêu cầu class 1–11 chỉ lấy injected anomaly positions. Khi một class không có token, code dùng `fallback_hidden_tokens`; specification lại yêu cầu fail thay vì fallback. Runtime flow document phải mô tả code hiện hành và ghi conflict kế bên.

### Stage B data and computation

Stage B forward bật prototype path. Encoder bị freeze. Continuous branch truy hồi `continuous_prototype_context`; discrete branch truy hồi `discrete_codeword_context`; `task_specific_concat_projection` tạo `reconstruction_fused_hidden` và `classification_fused_hidden`; hai task heads tạo reconstruction và classification logits.

Với config hiện hành, trainable modules thực tế là `reconstruction_concat_projection`, `classification_concat_projection`, `reconstruction_head` và `classification_head`. Các fusion gates chỉ thuộc mode khác và bị freeze trong Stage B.

### Evaluation and threshold lineage

Evaluator nhận per-window `point_scores`, dùng metadata `start_index` và `end_index` để đưa scores về entity timeline, rồi chỉ tính metrics trên covered points. Clean validation chạy trước test. `offline_point_threshold` được lấy từ clean-validation point-score timeline; test labels chỉ tham gia metric computation sau khi scores đã cố định.

Benchmark wrapper cũng lấy cùng clean-validation scores để tạo `online_point_ewma_threshold`. Vì vậy `threshold_artifact` là object chuyển tiếp chính thức từ offline sang online.

## Configuration observed

| Setting | Active main value | Meaning | Evidence |
| --- | --- | --- | --- |
| `window_size` | `20` | Số time points trong mỗi window | `configs/data/smd_benchmark_machine_1_6_window20.yaml:5` |
| `stage_a_multitask_epochs` | `25` | Epoch budget Stage A của main benchmark | `scripts/configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O0__machine_1_6__w20__seed8__main.yaml:46` |
| `stage_b_fusion_finetuning_epochs` | `5` | Epoch budget Stage B của main benchmark | `scripts/configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O0__machine_1_6__w20__seed8__main.yaml:47` |
| `continuous_num_prototypes` | `32` | Kích thước continuous prototype bank | `configs/model/thesis_multitask_two_stage_window20.yaml:21` |
| `discrete_codebook_size` | `60` | Kích thước discrete codebook | `configs/model/thesis_multitask_two_stage_window20.yaml:23` |
| `discrete_query_mode` | `cosine_topk` | Discrete retrieval mode của Stage B | `configs/model/thesis_multitask_two_stage_window20.yaml:80` |
| `monte_carlo_samples` | `10` | Số stochastic retrieval samples khi inference | `configs/model/thesis_multitask_two_stage_window20.yaml:25` |
| `checkpoint_monitor_metric` | `val_synth_vus_pr` | Metric chọn best checkpoint | `scripts/configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O0__machine_1_6__w20__seed8__main.yaml:21` |
| `offline_threshold_split` | `clean_validation` | Split hiệu chỉnh threshold offline | `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml:4` |
| `offline_threshold_quantile` | `0.99` | Quantile hiệu chỉnh threshold offline | `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml:5` |

## Evidence

- `scripts/experiments/run_two_stage_offline_pretraining.py:31-101` — định nghĩa hai stage, epoch validation và stage records.
- `scripts/experiments/run_two_stage_offline_pretraining.py:115-162` — materialize stage config và cho thấy `training_phase` đang giữ stage identifier.
- `scripts/experiments/run_two_stage_offline_pretraining.py:244-337` — tạo Stage B initialization checkpoint từ Stage A best checkpoint và train loader.
- `scripts/experiments/run_two_stage_offline_pretraining.py:364-459` — thứ tự thực thi Stage A, memory initialization, Stage B và evaluation qua child processes.
- `scripts/benchmarks/run_thesis_offline_benchmark.py:658-735` — full wrapper path từ config đến artifacts và report.
- `src/models/thesis_multitask.py:43-84` — public offline model entrypoint.
- `src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py:48-177` — stage-dependent prototype path và trainable/frozen surface.
- `src/models/thesis_multitask_impl/thesis_multitask_loss_step_mixin.py:218-356` — batch preparation, forward, loss assembly và public stage steps.
- `src/models/thesis_multitask_impl/thesis_multitask_state_memory_mixin.py:153-358` — implemented token pools, per-class k-means và fallback behavior.
- `scripts/benchmarks/run_thesis_offline_benchmark.py:421-471` — clean validation threshold chạy trước synthetic validation và test.
- `src/engine/evaluator.py:415-548` — inference, timeline reconstruction, threshold resolution và metrics.
- `documents/spec/full-spec-v3.md:430-512` — normative Stage A, memory boundary, Stage B và 25/5 epoch budget.
- `documents/spec/full-spec-v3.md:709-779` — normative threshold artifact và offline evaluation contract.
- `tests/models/test_multitask_memory_initialization.py:125-154` — test hiện hành xác nhận discrete pool chứa đủ ba tokens của class window, không chỉ injected token.

## Conflicts and uncertainties

| Topic | Implemented | Documented intent | Consequence |
| --- | --- | --- | --- |
| Stage field name | `training_phase` nhận stage identifier | Docs nói Stage A/B là stages trong offline phase | Ontology dùng `stage_name`; `training_phase` là compatibility alias. |
| Discrete pool membership | Tất cả tokens của class window | Chỉ eligible anomaly tokens cho class 1–11 | Pseudocode runtime ghi behavior hiện tại; ontology đánh dấu conflict. |
| Missing discrete class | Dùng pooled fallback | Raise nếu thiếu eligible tokens | Không được mô tả fallback là behavior đã được spec chấp thuận. |
| O1 artifact identity | Collector đọc root `offline_variant`, default `O0` | Config O1 chỉ có `experiment_name` và `experiment_variant` | `threshold_artifact.variant_name` có nguy cơ mang sai variant nếu config loader không bổ sung field. |
| Epoch budgets in older design note | `80 + 20` | Active full-spec-v3 và main config dùng `25 + 5` | Ontology không coi epoch count là identity của stage; flow dùng config-resolved budget. |

## Open questions

1. Developer có muốn sửa runtime discrete pool để khớp strict anomaly-position rule của `full-spec-v3`, hay sửa specification để chấp nhận whole-window class tokens?
2. Developer có muốn thêm root field `offline_variant: O0|O1` vào offline configs để artifact identity không fallback sai?
3. Field `training_phase` và constants `TWO_STAGE_*_PHASE_NAME` có được migrate sang stage terminology trong source hay chỉ giữ compatibility mapping ở documentation?
