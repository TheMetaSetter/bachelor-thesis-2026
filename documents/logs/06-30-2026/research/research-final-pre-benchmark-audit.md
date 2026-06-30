---
date: 2026-06-30 14:39:04 +07 +0700
researcher: TheMetaSetter
git_commit: 477007fb24c80c578dccf3c30c58acdf6355e081
branch: dev
repository: bachelor-thesis-2026
topic: "Final pre-benchmark audit of the active benchmark pipeline"
tags: [research, benchmark, smd, evaluation, synthetic-anomaly, preflight]
status: complete
last_updated: 2026-06-30
last_updated_by: TheMetaSetter
---

# Research: Final pre-benchmark audit of the active benchmark pipeline

**Date**: 2026-06-30 14:39:04 +07 +0700
**Researcher**: TheMetaSetter
**Git Commit**: `477007fb24c80c578dccf3c30c58acdf6355e081`
**Branch**: `dev`

## Research Question

Rà soát kĩ lại một lần cuối cùng toàn bộ benchmark pipeline active trước khi chạy benchmark, theo đúng tinh thần chỉ mô tả và kiểm tra những gì codebase đang làm thật sự ở thời điểm hiện tại.

## Summary

Kết luận ngắn gọn là codebase hiện tại đã sẵn sàng để chạy **benchmark SMD top-3 entities, 2 seeds, 2 methods** theo launcher active. Cụ thể, launcher active đang khóa vào `machine-1-6`, `machine-3-4`, `machine-3-9`, với hai model `redlamp_baseline` và `thesis_multitask`, mỗi config chạy `100` epochs. Đường đi benchmark này đã có test, có preflight, có manifest, có ràng buộc single-entity, có metric pointwise cùng `VUS-PR`, `VUS-ROC`, `affiliation F1`, và có đánh dấu non-comparable cho các run bị single-class hoặc truncated.

Tuy nhiên, codebase **chưa phải benchmark runtime đa dataset**. Runtime active hiện tại chỉ hỗ trợ `smd` và `anomaly_archive`. Vì vậy, nếu nói đến SWaT, IOPS, NASA, UCR, hay dataset mới như ICCAD, thì ở thời điểm audit này chưa có cơ sở để nói là benchmark runner active đã sẵn sàng chạy chúng giống SMD.

Một phát hiện rất quan trọng là `train_synth` và `val_synth` hiện tại đang cân bằng theo **12 class ở mức window**, không phải theo point. Vì có 1 class normal và 11 class anomaly, nên khoảng `11/12 ~= 91.65%` synthetic windows là anomalous. Mỗi anomalous window lại chỉ tiêm một đoạn liên tiếp dài khoảng `20%` đến `30%` của window length. Với `window_size = 20`, điều này làm cho **tỉ lệ anomaly point-level thực tế** trong synthetic train và synthetic validation rơi vào khoảng `22.8%`, chứ không phải `1%`, `3%`, hay `5%`.

## Detailed Findings

### Data Preparation

Runtime config validation hiện chỉ cho phép `dataset_name` thuộc tập `{smd, anomaly_archive}`. Nếu đưa dataset khác vào config active thì code sẽ fail sớm thay vì chạy mơ hồ. Điều này được chốt ở `src/core/config.py:300-310`.

Đường đi build dataset benchmark active hiện làm theo thứ tự: parse sequence theo split, kiểm tra test split phải có cả normal lẫn anomaly, fit `SequenceStandardScaler` trên `train`, transform cả `train/val/test`, rồi mới cắt window. Điều này nằm ở `src/data/loaders.py:150-176`. Nghĩa là benchmark SMD active hiện đang tuân thủ đúng hợp đồng “fit scaler trên train rồi apply sang future splits”.

Benchmark test split bị khóa phải là `mixed`, tức reconstructed timeline của test phải có cả `0` lẫn `1`. Nếu test chỉ có một lớp, code sẽ raise `ValueError`. Điều này nằm ở `src/data/split_protocol.py:47-58`. Nói ngắn gọn, benchmark active hiện không cho phép một test set toàn normal hoặc toàn anomaly mà vẫn giả vờ coi là benchmark chính.

Các config dữ liệu benchmark SMD active đang dùng `window_size: 20`, `train_stride: 10`, `val_stride: 1`, `test_stride: 1`, `shuffle_train: false`. Ví dụ rõ ở `configs/data/smd_benchmark_machine_1_6_window20.yaml`. Vì `test_stride = 1`, timeline test được phủ dày, không còn lỗi “bỏ sót nhiều điểm test vì stride quá lớn” trong benchmark main hiện tại.

### Synthetic Augmentation

Synthetic anomaly hiện vẫn được tiêm **sau windowization**, không phải sequence-level trước windowization. Điểm này thấy rất rõ ở `SyntheticAnomalyInjector.augment_batch`, vì nó nhận trực tiếp batch `x` có shape `[B, L, D]` và tiêm trên từng window (`src/data/augment.py:866-930`). Đoạn chọn span anomaly cũng dùng `window_size`, không dùng chiều dài cả chuỗi (`src/data/augment.py:198-218`).

`min_segment_fraction` và `max_segment_fraction` hiện được hiểu là tỉ lệ trên **window length**, không phải trên whole sequence. Với `window_size = 20` và config benchmark hiện tại là `0.2` đến `0.3`, mỗi anomalous window sẽ có một span anomaly dài khoảng `4` đến `6` timestep (`src/data/augment.py:203-217`, `configs/task/multitask_tsad_redlamp_multiclass_window20_benchmark_fixed_synth.yaml`).

Khi `train_balance_classes = true`, code bỏ qua Bernoulli `anomaly_probability` và chuyển sang chia quota gần đều cho các class active. Điều này được ghi rất rõ trong code ở `src/data/augment.py:52-57` và `src/data/augment.py:825-864`. Với taxonomy active hiện tại là `12` class (`normal` + `11` anomaly families), cân bằng theo class đồng nghĩa gần đúng với:

- `1/12` window là normal
- `11/12` window là anomalous

Em đã đo trực tiếp trên loader benchmark active bằng cách iterate qua `train` và `val`, rồi áp `SyntheticAnomalyInjector` đúng config benchmark. Kết quả:

- `machine-1-6`, `train`: `1736 / 1894` anomalous windows, tỷ lệ `0.91658`; anomaly point ratio `8652 / 37880 = 0.22841`
- `machine-1-6`, `val`: `4324 / 4718` anomalous windows, tỷ lệ `0.91649`; anomaly point ratio `21531 / 94360 = 0.22818`
- `machine-3-4`, `train`: giống `machine-1-6` do số window train trùng nhau
- `machine-3-4`, `val`: giống `machine-1-6` do số window val trùng nhau
- `machine-3-9`, `train`: anomaly point ratio `10451 / 45920 = 0.22759`
- `machine-3-9`, `val`: anomaly point ratio `26129 / 114460 = 0.22828`

Nói ngắn gọn: **synthetic train và synthetic validation hiện tại đang ở khoảng 22.8% anomaly timesteps** trên các window synthetic đã được tạo ra.

Hai model active đều dùng cùng logic injector này:

- `redlamp_baseline` xây `synthetic_anomaly_injector` và `synthetic_validation_injector` ở `src/models/redlamp_baseline.py:245-277`
- `thesis_multitask` xây injector tương tự ở `src/models/thesis_multitask.py:1044-1064`

Cả hai model đều reset RNG mỗi epoch:

- baseline: `src/models/redlamp_baseline.py:280-284`
- thesis: `src/models/thesis_multitask.py:1863-1867`

Trainer gọi các hook reset này đầu mỗi train epoch và đầu mỗi `val_synth` epoch (`src/engine/trainer.py:603-705`). Vì benchmark task config đang khóa `synthetic_train_seed: 7` và `synthetic_validation_seed: 7`, nên synthetic train/val hiện tại là **deterministic across epochs trong một run**.

### Modeling and Training

Benchmark baseline active là `redlamp_baseline`, không còn dùng tên cũ `redlamp_mlp_baseline` trong config benchmark active. Test active benchmark config hiện cũng đang kiểm tra đúng model name này ở `tests/test_redlamp_baseline_active_benchmark_config.py`.

Benchmark thesis active là `thesis_multitask` với `three_stage.expected_total_training_epochs = 100`, tách thành `15 + 25 + 5 + 5 + 50`. Ví dụ rõ ở `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-three-stage-machine_1_6__w20__seed6__main.yaml`.

Launcher benchmark active là shell script `scripts/launch_tmux_comparative_smd_experiment.sh`. Danh sách config main hiện được hard-code đúng `12` run:

- 3 entity SMD: `machine-1-6`, `machine-3-4`, `machine-3-9`
- 2 seed: `6`, `36`
- 2 method families: baseline và thesis

Điểm này nằm ở `scripts/launch_tmux_comparative_smd_experiment.sh:37-50`.

Launcher Python trung tâm là `scripts/run_comparative_smd_experiments.py`. Script này:

- ép mỗi config chỉ có đúng 1 `entity_id` (`scripts/run_comparative_smd_experiments.py:121-130`)
- phân biệt baseline single-stage với thesis three-stage (`scripts/run_comparative_smd_experiments.py:109-118`)
- baseline chạy `train.py` rồi `evaluate.py` (`scripts/run_comparative_smd_experiments.py:144-166`)
- thesis three-stage chạy `run_three_stage_offline_pretraining.py` (`scripts/run_comparative_smd_experiments.py:133-141`)

### Evaluation

Evaluator hiện reconstruct pointwise records trên từng entity, rồi chỉ dùng **covered points** để tính metrics. Điều này rất quan trọng vì nó tránh đưa các điểm chưa được model chạm tới vào metric một cách giả tạo. Đường đi này nằm ở:

- reconstruct records: `src/engine/evaluator.py:121-167`
- extract covered arrays: `src/engine/evaluator.py:170-199`
- evaluate and compute metrics: `src/engine/evaluator.py:318-371`

Threshold mặc định khi evaluate được chọn bằng quantile `0.95` trên `concatenated_scores` của validation/evaluation payload, với `threshold_source = positive_support_quantile_0.95` nếu không truyền threshold ngoài vào (`src/engine/evaluator.py:329-336`).

`compute_pointwise_metrics` hiện đã thật sự tính thêm:

- `affiliation_f1` ở `src/metrics/pointwise.py:469-491`
- `vus_pr` ở `src/metrics/pointwise.py:381-423` và gọi tại `src/metrics/pointwise.py:585-593`
- `vus_roc` ở `src/metrics/pointwise.py:425-466` và gọi tại `src/metrics/pointwise.py:593-598`

Nếu label chỉ có một lớp thì:

- `compute_vus_pr_exact_naive` trả `NaN` (`src/metrics/pointwise.py:393-395`)
- `compute_vus_roc_exact_naive` trả `NaN` (`src/metrics/pointwise.py:437-438`)
- `compute_affiliation_f1` trả `NaN` (`src/metrics/pointwise.py:478-479`)

Evaluator còn ghi thêm các cờ chẩn đoán:

- `raw_num_points`
- `evaluated_num_points`
- `num_entities_evaluated`
- `is_truncated_evaluation`
- `label_regime`
- `benchmark_comparability`
- `protocol_status`
- `threshold_source`

đều ở `src/engine/evaluator.py:347-371`.

Điều này có nghĩa là codebase hiện tại đã có cơ chế đánh dấu run “single-class” hoặc “truncated” là không comparable, thay vì âm thầm coi đó là benchmark bình thường.

## Verification Performed

### Focused tests

Đã chạy:

```bash
.venv/bin/python -m pytest -q \
  tests/test_config_loading.py \
  tests/test_comparative_config_loading.py \
  tests/test_redlamp_baseline_active_benchmark_config.py \
  tests/test_redlamp_baseline_active_config_paths.py \
  tests/test_smd_machine_3_4_three_stage_config_loading.py \
  tests/test_split_protocol.py \
  tests/test_synthetic_anomaly_injection.py \
  tests/test_multitask_validation_alignment.py \
  tests/test_evaluation_metrics_audit.py \
  tests/test_evaluation_protocol_audit.py \
  tests/test_evaluator_thresholding.py \
  tests/test_multitask_metrics_runtime.py \
  tests/test_three_stage_run_verifier.py \
  tests/test_three_stage_server_launcher.py \
  tests/test_three_stage_server_preflight.py \
  tests/test_one_redlamp_mlp_train_step.py \
  tests/test_one_multitask_train_step.py
```

Kết quả:

- `221 passed`
- `23 warnings`

Các warning chủ yếu đến từ test cố tình đụng vào single-class label regimes để kiểm tra behavior `ROC-AUC undefined`, `PR curve degenerate`, và các warning worker count trên máy local. Không có failed test trong cụm benchmark-critical này.

### Dry-run launcher

Đã chạy:

```bash
bash scripts/launch_tmux_comparative_smd_experiment.sh \
  --dry-run \
  --smoke-profile none \
  --data-num-workers-override 8
```

Dry-run xác nhận launcher đang chuẩn bị đúng `12` main runs và dự kiến sẽ dùng các đường dẫn artifact sau khi chạy thật:

- `outputs/benchmark_smd_reports/top3-two-seeds/comparative_server_preflight_summary.json`
- `outputs/benchmark_smd_reports/top3-two-seeds/comparative_manifest.json`
- `outputs/benchmark_smd_reports/top3-two-seeds/comparative_execution_report.json`

### Comparative preflight on local machine

Đã chạy local comparative preflight và lưu artifact tại:

- `outputs/benchmark_smd_reports/final-audit-local/comparative_server_preflight_summary.json`

Kết quả local:

- `all_devices_are_cuda = true` theo config
- `data_roots_exist = true`
- `artifact_paths_unique = true`
- `tmux_available = false` trên máy local
- `gpu_validation.status = cuda_unavailable` trên máy local
- `launch_readiness.status = not_ready_for_comparative_launch`

Kết luận phần này là: **config benchmark và artifact plan đều ổn**, nhưng local machine không phải target server nên preflight local không thể cho trạng thái ready. Cần chạy lại đúng script preflight này trên máy server RTX 3090 trước khi bấm tmux run thật.

## Code References

- `src/core/config.py:300-310` - runtime chỉ hỗ trợ `smd` và `anomaly_archive`
- `src/data/loaders.py:165-173` - validate test labels, fit scaler trên train, transform mọi split
- `src/data/split_protocol.py:47-58` - benchmark test phải có cả normal lẫn anomaly
- `src/data/augment.py:52-57` - balanced mode bỏ qua Bernoulli anomaly probability
- `src/data/augment.py:203-217` - anomaly span length được tính theo `window_size`
- `src/data/augment.py:800-864` - class quota gần đều trên taxonomy active
- `src/data/augment.py:866-930` - synthetic anomaly được tiêm ở mức window batch
- `src/models/redlamp_baseline.py:252-284` - build và reset synthetic injectors của baseline
- `src/models/thesis_multitask.py:1044-1064` - build injectors của thesis model
- `src/models/thesis_multitask.py:1863-1867` - reset synthetic RNG mỗi epoch
- `src/engine/trainer.py:603-705` - trainer gọi prepare hooks cho train và `val_synth`
- `src/engine/evaluator.py:318-371` - reconstruct, covered-only metrics, threshold, protocol flags
- `src/metrics/pointwise.py:381-491` - `VUS-PR`, `VUS-ROC`, `affiliation_f1`
- `src/metrics/pointwise.py:542-607` - pointwise metric payload và diagnostics
- `scripts/run_comparative_smd_experiments.py:121-130` - ràng buộc một config chỉ có một entity
- `scripts/run_comparative_smd_experiments.py:144-166` - baseline run = train rồi evaluate
- `scripts/launch_tmux_comparative_smd_experiment.sh:37-50` - danh sách 12 benchmark runs active
- `scripts/preflight_comparative_smd_server.py:96-133` - launch readiness logic

## Pipeline Documentation

Pipeline benchmark active hiện tại, nếu nói đúng theo code, là:

1. Load một trong các benchmark config SMD active.
2. Parse SMD raw train/test files cho đúng entity duy nhất được chọn.
3. Tách `train/val` từ official train timeline, giữ `test` là official labeled future timeline.
4. Kiểm tra `test` phải là mixed-label timeline.
5. Fit scaler trên `train`.
6. Transform `train/val/test`.
7. Cắt window với `train_stride = 10`, `val_stride = 1`, `test_stride = 1`.
8. Trong train và `val_synth`, model tự tiêm synthetic anomaly trên từng window.
9. Train baseline hoặc thesis three-stage theo config.
10. Reconstruct pointwise scores về timeline từng entity.
11. Chỉ lấy covered points để tính metrics.
12. Lưu threshold, diagnostics, curves, records, audit flags, và evaluation outputs.

## Historical Context (from documents/)

Audit này phù hợp với các quyết định đã chốt gần đây trong repo:

- benchmark chính phải có test timeline chứa cả normal lẫn anomaly
- smoke hoặc truncated runs phải bị đánh dấu non-comparable
- `stride = 1` cho test benchmark chính để tránh bỏ sót coverage
- synthetic train/val phải reproducible
- benchmark active trước mắt cần giữ đơn giản và thực dụng, tập trung vào SMD main path trước

## Open Questions

1. Synthetic train/val hiện đang ở khoảng `22.8%` anomaly points. Đây là behavior đúng theo code active, nhưng nó là một lựa chọn benchmark cần được chấp nhận có ý thức trước khi chạy.
2. Runtime benchmark active hiện chưa phải multi-dataset runtime. Nếu muốn chạy SWaT hoặc dataset khác theo cùng chuẩn benchmark, cần triển khai loader/runtime/config path riêng trước.
3. Cần chạy lại `scripts/preflight_comparative_smd_server.py` trên đúng máy server có `tmux` và `RTX 3090` để đổi trạng thái từ local `not_ready_for_comparative_launch` sang server-side readiness thực sự.
4. `src/metrics/pointwise.py` vẫn còn vài comment TODO cũ không còn khớp hoàn toàn với implementation hiện tại. Đây là nợ tài liệu, chưa phải bug runtime.
