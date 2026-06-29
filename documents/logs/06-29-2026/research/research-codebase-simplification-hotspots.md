---
date: 2026-06-29 23:08:38 +07 +0700
researcher: Codex
git_commit: e9380c0396ab57b7b6d564593cde701cb8773d06
branch: dev
repository: bachelor-thesis-2026
topic: "Codebase simplification audit for the active benchmark and validation pipeline"
tags: [research, time-series, anomaly-detection, codebase-simplification, benchmark]
status: complete
last_updated: 2026-06-29
last_updated_by: Codex
---

# Research: Codebase simplification audit for the active benchmark and validation pipeline

**Date**: 2026-06-29 23:08:38 +07 +0700  
**Researcher**: Codex  
**Git Commit**: `e9380c0396ab57b7b6d564593cde701cb8773d06`  
**Branch**: `dev`

## Research Question

Dò lại toàn bộ codebase để xem còn chỗ nào có thể đơn giản hoá hơn nữa hay không, với trọng tâm là benchmark path đang dùng để chuẩn bị chạy các thí nghiệm chính.

## Summary

Benchmark path hiện tại đã được đơn giản hơn rõ rệt so với nhánh cũ. Cụ thể, benchmark config mới đã khóa `epochs = 100`, dùng `val_synth_vus_pr` làm metric theo dõi, dùng `train_stride = 10` và `val_stride = test_stride = 1`, đồng thời tắt `val_realistic` ở task config. Tuy nhiên, phần code bao quanh benchmark path vẫn còn nhiều bề mặt cũ, nhất là `val_realistic` và họ config/script/test mang tên `comparative`, khiến codebase nhìn rối hơn nhu cầu thực tế hiện tại.

Điểm quan trọng nhất là benchmark path mới và codebase cũ đang cùng tồn tại trong một runtime. Điều này không nhất thiết làm sai logic ngay lập tức, nhưng nó làm tăng số lượng codepath, làm tên gọi kém rõ nghĩa, và khiến người đọc rất dễ hiểu nhầm đâu là đường chạy chính. Theo đúng tinh thần trong `documents/design/design_starter.md` và `codebase_preferences.md`, đây là phần có dư địa đơn giản hoá lớn nhất.

## Detailed Findings

### Data Preparation

Nhánh benchmark mới đã khóa task config theo hướng đơn giản hơn nhánh cũ. File [`configs/task/multitask_tsad_redlamp_multiclass_window20_benchmark_fixed_synth.yaml`](configs/task/multitask_tsad_redlamp_multiclass_window20_benchmark_fixed_synth.yaml) đặt `synthetic_train_seed = 7`, `synthetic_validation_seed = 7`, `train_balance_classes = true`, và `val_realistic = false` ngay trong task config. Điều này cho thấy benchmark path mới đang cố ý dùng synthetic validation cố định thay vì auxiliary realistic validation.

Tuy nhiên, cùng file này vẫn giữ lại `val_realistic_source` và `val_anomaly_rate_override` dù `val_realistic = false`. Đây là một dấu hiệu cho thấy cấu trúc task config hiện tại vẫn mang theo phần dư của nhánh validation cũ. Nó không gây lỗi trực tiếp, nhưng làm file cấu hình dài hơn mức cần thiết cho benchmark chính.

Synthetic anomaly injector hiện tại hỗ trợ cả hai chế độ: cân bằng lớp tổng hợp và Bernoulli anomaly sampling. Trong [`src/data/augment.py`](src/data/augment.py), tên biến `train_balance_classes` thực tế không chỉ ảnh hưởng train mà còn được dùng lại cho cả synthetic validation injector khi model khởi tạo hai injector. Bên trong injector, `anomaly_probability` chỉ thật sự điều khiển việc tiêm anomaly khi `train_balance_classes = false`. Khi `train_balance_classes = true`, injector bỏ qua Bernoulli sampling và tự phân phối gần đều các class tổng hợp trong batch. Như vậy, tên biến hiện tại diễn đạt chưa đủ đúng ý nghĩa runtime thật sự.

Window collation vẫn còn khá ồn về logging. Hàm [`src/data/collate.py`](src/data/collate.py) gọi `console_print` ngay lúc bắt đầu và sau khi tạo xong batch cho mọi batch được collate. Đây không phải lỗi protocol, nhưng là một nguồn nhiễu trong runtime logs, nhất là khi chạy benchmark dài.

### Modeling and Training

Phần phức tạp nhất còn sót lại nằm ở nhánh `val_realistic`. Trong [`src/engine/trainer.py`](src/engine/trainer.py), trainer vẫn còn giữ đầy đủ logic chọn threshold metric cho `val_realistic_*`, aggregate reconstruction diagnostics cho stage `val_realistic`, resolve anomaly rate từ test statistics cho realistic validation, và branch riêng trong vòng lặp epoch để chạy `realistic_validation_step`. Điều này có nghĩa là benchmark path mới đã tắt `val_realistic` ở config, nhưng runtime vẫn còn mang toàn bộ machinery cũ này.

Tình trạng đó cũng lặp lại trong model. Ở [`src/models/redlamp_mlp_baseline.py`](src/models/redlamp_mlp_baseline.py), model vẫn giữ:

- tham số legacy `balance_classes_within_batch`
- tham số legacy hơn nữa `balance_binary_classes_within_batch`
- method `prepare_realistic_validation_epoch`
- hỗ trợ `stage_name in {"val_synth", "val_realistic"}`
- method `realistic_validation_step`

Ở [`src/models/thesis_multitask.py`](src/models/thesis_multitask.py), phần dư còn nhiều hơn vì có:

- `SyntheticAnomalyConfig.val_realistic`
- `SyntheticAnomalyConfig.val_realistic_source`
- `SyntheticAnomalyConfig.val_anomaly_rate_override`
- các flat keys tương ứng trong `from_flat_kwargs`
- model attributes tương ứng
- method `prepare_realistic_validation_epoch`
- `prepare_synthetic_validation_epoch` hiện chỉ là wrapper quay lại `prepare_realistic_validation_epoch`
- các branch chấp nhận cả `val_synth` lẫn `val_realistic`
- method `realistic_validation_step`

Điểm cần nhấn mạnh là benchmark path mới không còn cần phần lớn lớp logic này. Nó vẫn hoạt động được vì `val_realistic` đã bị tắt ở config benchmark, nhưng codepath dư vẫn còn nguyên.

### Evaluation

Config validation vẫn cho phép toàn bộ hệ metric và scheduler monitor cũ cho `val_realistic_*`. Trong [`src/core/config.py`](src/core/config.py), `allowed_task_keys_by_task_name["multitask_tsad"]` vẫn chứa `val_realistic`, `val_realistic_source`, và `val_anomaly_rate_override`. Cùng file này cũng cho phép `checkpoint_monitor_metric` và `optimizer.scheduler.monitor_metric` dùng các metric `val_realistic_*`. Điều này cho thấy hệ config chính thức vẫn xem nhánh này là first-class runtime path.

Điều đó mâu thuẫn nhẹ với benchmark path mới. Benchmark tests trong [`tests/test_config_loading.py`](tests/test_config_loading.py) đã kiểm tra rõ rằng benchmark configs phải dùng `val_synth_vus_pr`, `epochs = 100`, `stride = 10`, `val_stride = 1`, `test_stride = 1`, và `task.val_realistic = false`. Nói cách khác, benchmark semantics mới đã có test riêng, nhưng hệ config chung vẫn chưa được dọn gọn theo benchmark semantics đó.

Test suite cũng phản ánh hai thế giới cùng tồn tại. Trong [`tests/test_multitask_validation_alignment.py`](tests/test_multitask_validation_alignment.py), một test vẫn xác nhận `val_realistic_*` được log khi không tắt nhánh realistic validation, và test khác xác nhận `val_synth_*` được dùng khi `task.val_realistic = false`. Đây là bằng chứng rằng codebase hiện có hai namespace validation phụ song song.

### Benchmark and Orchestration Surface

Tên gọi `comparative` vẫn phủ rất rộng lên scripts, tests, và artifact naming, dù launcher chính hiện đã trỏ sang benchmark configs mới. Script [`scripts/launch_tmux_comparative_smd_experiment.sh`](scripts/launch_tmux_comparative_smd_experiment.sh) có:

- session name mặc định theo benchmark mới
- main config paths trỏ đến `configs/experiment/benchmark/...`
- nhưng smoke configs vẫn trỏ sang `configs/experiment/comparative/...`
- report artifacts vẫn mang tên `comparative_manifest.json`, `comparative_execution_report.json`, `comparative_server_preflight_summary.json`
- runner bên dưới vẫn là `scripts/run_comparative_smd_experiments.py`

Điều này cho thấy orchestration layer đã được cập nhật một nửa. Chức năng thực tế đang hướng về benchmark mới, nhưng tên script, report, và smoke artifacts vẫn còn theo vocabulary cũ. Đây là một nguồn gây hiểu lầm rõ ràng cho người đọc và cho chính người chạy server sau này.

Tình trạng tương tự xuất hiện ở model config naming. Benchmark configs mới vẫn dùng:

- `configs/model/redlamp_mlp_baseline_comparative_smd.yaml`
- `configs/model/thesis_multitask_three_stage_comparative_smd.yaml`

Tên file không còn phản ánh đúng vai trò hiện tại của chúng, vì chúng đang được benchmark chính dùng trực tiếp.

### Historical and Documentation Surfaces

CLI help trong [`src/core/config_help.py`](src/core/config_help.py) hiện vẫn rất tối giản và chưa phản ánh rõ benchmark semantics mới. Nó chỉ nhắc `anomaly_probability must be in [0, 1]`, nhưng không giải thích rằng khi class balancing bật thì biến này không điều khiển anomaly ratio của benchmark balanced path.

Documentation và tests cũ vẫn còn nhiều dấu vết vocabulary cũ. Ví dụ:

- [`tests/test_comparative_config_loading.py`](tests/test_comparative_config_loading.py) vẫn khóa semantics cũ với `epochs = 300`, `val_realistic = true`, `stride = 1`, và danh sách entity cũ.
- `documents/design/experiment_config_organization_guideline.md` vẫn còn nhắc `val_realistic_*` như validation namespace chính.

Ở mức repo-wide grep tại commit hiện tại, có khoảng:

- `401` kết quả chứa `val_realistic`
- `306` kết quả chứa `comparative`
- `73` kết quả chứa `train_balance_classes`
- `17` kết quả chứa hai alias legacy `balance_binary_classes_within_batch` hoặc `balance_classes_within_batch`

Các con số này không tự thân chứng minh có bug, nhưng chúng cho thấy bề mặt khái niệm cũ vẫn còn rất rộng.

## Code References

- `src/engine/trainer.py:42-45` - threshold metadata vẫn có nhánh `val_realistic_threshold`
- `src/engine/trainer.py:202-203` - reconstruction diagnostics vẫn iterate qua `val_realistic`
- `src/engine/trainer.py:574-592` - realistic validation vẫn có logic ước lượng anomaly prior từ test statistics
- `src/engine/trainer.py:617-618` - trainer vẫn đọc `task.val_realistic`
- `src/engine/trainer.py:736-760` - branch realistic validation vẫn chạy riêng
- `src/core/config.py:453-470` - task config vẫn cho phép `val_realistic*`
- `src/core/config.py:696-713` - checkpoint monitor vẫn cho phép `val_realistic_*`
- `src/core/config.py:879-908` - default boolean fields vẫn chứa `val_realistic`
- `src/core/config.py:1097-1117` - validation logic riêng cho `val_realistic_source` và override anomaly rate
- `src/core/config.py:1369-1373` - logging diagnostics stages vẫn cho phép `val_realistic`
- `src/models/redlamp_mlp_baseline.py:242-276` - model baseline vẫn giữ aliases legacy cho class balancing
- `src/models/redlamp_mlp_baseline.py:285-296` - baseline vẫn có `prepare_realistic_validation_epoch`
- `src/models/redlamp_mlp_baseline.py:323-324` - baseline vẫn coi `val_realistic` là synthetic-validation stage hợp lệ
- `src/models/redlamp_mlp_baseline.py:709-713` - baseline vẫn có `realistic_validation_step`
- `src/models/thesis_multitask.py:392-400` - thesis synthetic config vẫn chứa `val_realistic*`
- `src/models/thesis_multitask.py:1062-1082` - `train_balance_classes` được dùng cho cả train injector và validation injector
- `src/models/thesis_multitask.py:1881-1902` - thesis model vẫn giữ realistic-validation wrapper logic
- `src/models/thesis_multitask.py:2300-2316` - thesis batch prep và contrastive logic vẫn chấp nhận `val_realistic`
- `src/models/thesis_multitask.py:3288-3292` - thesis model vẫn có `realistic_validation_step`
- `src/data/augment.py:52-56` - comment hiện đã nói rõ balanced path làm `anomaly_probability` không còn là điều khiển chính
- `src/data/augment.py:83-86` - injector vẫn giữ alias compatibility cho binary-era naming
- `src/data/augment.py:825-850` - chỉ khi `train_balance_classes = false` thì `anomaly_probability` mới trực tiếp quyết định anomaly injection
- `src/data/collate.py:11-31` - collate path đang log mọi batch
- `configs/task/multitask_tsad_redlamp_multiclass_window20_benchmark_fixed_synth.yaml:1-30` - benchmark task path mới đã khóa `val_realistic = false`
- `scripts/launch_tmux_comparative_smd_experiment.sh:5-10` - launcher đã mang default benchmark naming mới
- `scripts/launch_tmux_comparative_smd_experiment.sh:27-49` - launcher trộn smoke configs kiểu `comparative` với main configs kiểu `benchmark`
- `scripts/launch_tmux_comparative_smd_experiment.sh:182-207` - manifest/report naming và runner naming vẫn là `comparative`
- `tests/test_multitask_validation_alignment.py:167-217` - test vẫn khóa namespace `val_realistic_*`
- `tests/test_multitask_validation_alignment.py:220-258` - test benchmark-style path dùng `val_synth_*`
- `tests/test_comparative_config_loading.py:34-116` - test config cũ vẫn khóa semantics `comparative`
- `tests/test_config_loading.py:1804-1844` - test benchmark mới khóa semantics `val_synth`, `100 epochs`, và split-specific strides

## Pipeline Documentation

Nếu chỉ nhìn vào benchmark path mới đang chuẩn bị chạy, pipeline hiện tại đơn giản hơn trước ở ba điểm chính.

Thứ nhất, task path chính là:

- `train`: clean windows được synthetic augmentation nếu bật
- `val`: clean validation để theo dõi reconstruction và các metric sạch
- `val_synth`: auxiliary validation trên cùng `val_loader`, nhưng được tiêm synthetic anomaly cố định
- `test`: pointwise evaluation trên timeline test thật

Thứ hai, benchmark path mới dùng split-specific coverage rõ hơn:

- `train_stride = 10`
- `val_stride = 1`
- `test_stride = 1`

Thứ ba, benchmark configs mới đã đồng bộ sang budget `100` epochs và monitor `val_synth_vus_pr`.

Phần chưa đơn giản là runtime xung quanh pipeline này vẫn còn giữ nguyên logic của một pipeline cũ hơn:

- auxiliary realistic validation theo anomaly prior từ test
- naming `comparative`
- tests và docs khóa semantics cũ

## Historical Context (from documents/)

`documents/design/design_starter.md` nhấn mạnh triết lí “thin waist”, số lượng contract nhỏ, và “least amount of codepaths”. `documents/design/idea.md` cũng nhấn mạnh input contract rõ ràng, batch contract rõ ràng, và objective modular nhưng mặc định nên nhỏ. So với hai tài liệu này, benchmark path mới đang đi đúng hướng, còn các bề mặt `val_realistic` và `comparative` là phần còn lệch khỏi tinh thần “ít codepath”.

## Open Questions

1. Có muốn giữ `val_realistic` như một analysis-only legacy feature hay xoá hẳn khỏi runtime chính?
2. Có muốn đổi tên `train_balance_classes` thành một tên phản ánh đúng phạm vi dùng cho cả `train` và `val_synth` hay chỉ giữ tên cũ để giảm số lượng thay đổi?
3. Có muốn giữ họ config/test/script `comparative` như tài liệu lịch sử, hay chuyển toàn bộ orchestration hiện tại sang vocabulary `benchmark` cho đồng nhất?
4. Có muốn giữ các alias legacy `balance_binary_classes_within_batch` và `balance_classes_within_batch` để tương thích ngược, hay loại bỏ để giảm bớt bề mặt API?
