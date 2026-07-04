---
date: 2026-07-03 19:24:33 +0700
researcher: TheMetaSetter
git_commit: 31277c7afdd9e2d8dec3c39bf9c497fe1afab051
branch: dev
repository: bachelor-thesis-2026
topic: "Offline pre-training versus online adaptation window size, and synthetic anomaly realism relative to real anomaly span statistics"
tags: [research, time-series, anomaly-detection, window-size, synthetic-anomaly, online-adaptation]
status: complete
last_updated: 2026-07-03
last_updated_by: TheMetaSetter
---

# Research: Offline pre-training versus online adaptation window size, and synthetic anomaly realism relative to real anomaly span statistics

**Date**: 2026-07-03 19:24:33 +0700  
**Researcher**: TheMetaSetter  
**Git Commit**: `31277c7afdd9e2d8dec3c39bf9c497fe1afab051`  
**Branch**: `dev`

## Research Question

1. Window size trong pha offline pre-training liệu có cần phải khác với window size trong pha online test-time adaptation? Có 2 mức window size phổ biến là 20 và 100 time-steps.

2. Với các chỉ số thống kê như trên của anomaly span, quy trình tiêm bất thường nhân tạo hiện tại của codebase có đủ để tạo ra các anomaly có chất lượng tương xứng với các anomaly trong dữ liệu thật? Người dùng lo rằng các synthetic anomaly không đủ thật.

## Summary

Kết luận thực dụng từ code hiện tại là **không nên để offline pre-training và online test-time adaptation dùng window size khác nhau** trong pipeline đang có. Lý do không chỉ là lý thuyết, mà là ràng buộc implementation: `ThesisMultitaskModel` lưu `window_size` trong config, encoder kiểm tra chiều dài window, và classification head có input dimension bằng `window_size * hidden_dim`. Online adaptation lại load nguyên `ThesisMultitaskModel` từ checkpoint làm frozen reference model. Vì vậy, nếu offline checkpoint được train với `L=20` nhưng online stream cấp window `L=100`, model path hiện tại sẽ lệch shape hoặc fail validation.

Về synthetic anomaly, quy trình hiện tại tạo được các anomaly có ích cho bài toán multi-class synthetic classification, nhưng **chưa tương xứng với độ dài anomaly span trong dữ liệu thật**. Với active `window_size=20` và `min_segment_fraction=0.2`, `max_segment_fraction=0.3`, synthetic span chỉ dài `4`, `5`, hoặc `6` timestep trong từng anomalous window. Trong khi đó, thống kê dữ liệu thật vừa xuất ra cho thấy median span length của `SMD` là `11`, mean là `90`, p75 là `72.5`, max là `3161`; `NASA` và `SWaT` còn dài hơn nhiều. Do đó synthetic hiện tại đang mô phỏng anomaly ngắn, cục bộ, nằm trong một window, không mô phỏng tốt các anomaly kéo dài qua nhiều window.

## Detailed Findings

### Data Preparation

Design context mới nhất đã chuyển trọng tâm active thesis experiments sang `L=20`. `documents/design/design_starter.md` ghi rõ active window length hiện tại là `L = 20`, và `documents/design/offline_pretraining_phase_two_view_contrastive_design.md` cũng khóa active window length là `L = 20`. Đây là design context hiện hành, khác với phần mô tả cũ trong `prompts/1_research_prompt.md` còn nhắc đến fixed-length windows of one hundred time steps.

Offline three-stage experiment active dùng `window_size=20`. File `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml` trỏ tới `configs/data/smd_rtx3090_machine_3_4_20_stride1.yaml`, `configs/model/thesis_multitask_three_stage_window20.yaml`, và `configs/task/multitask_tsad_redlamp_multiclass_window20.yaml`. Data config tương ứng có `window_size: 20` và `stride: 1`.

Online adaptation config hiện tại vẫn nằm trong họ `w100`. `configs/experiment/baseline/smd__online_adaptation__online-adaptation__w100__seed7__default.yaml` trỏ tới `configs/data/smd_smoke.yaml`, và file data config này có `window_size: 100`, `stride: 10`. Tuy nhiên đây là một online slice mặc định/smoke, không phải bằng chứng rằng online adaptation nên lệch window size với offline checkpoint. Ngược lại, code online đang lấy `window_size` từ data config để tạo `SMDOnlineStream`, rồi model online lại load reference checkpoint của `ThesisMultitaskModel`.

Span statistics vừa tạo nằm ở `documents/logs/07-03-2026/research/anomaly_span_length_summary.csv`. Các số quan trọng:

| dataset | num_spans | mean | median | p75 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AnomalyArchive | 250 | 196.452 | 100.0 | 267.5 | 650.0 | 1700 |
| IOPS | 1470 | 37.116 | 8.0 | 21.0 | 143.55 | 1121 |
| NASA | 105 | 616.229 | 120.0 | 531.0 | 3240.0 | 4217 |
| SMD | 327 | 90.043 | 11.0 | 72.5 | 412.5 | 3161 |
| SWaT | 1 | 54621.0 | 54621.0 | 54621.0 | 54621.0 | 54621 |

Điểm cần nhớ là `SWaT/merged.csv` trong workspace hiện có một block `Attack` duy nhất rất dài. `AnomalyArchive` có một số interval filename tạo ra span length `0` theo half-open semantics; nghiên cứu này giữ nguyên annotation, không tự sửa dữ liệu.

### Modeling and Training

`ThesisMultitaskModel` hiện phụ thuộc trực tiếp vào `window_size`. Trong encoder wrapper, output hidden phải giữ đúng chiều `window_size`; nếu `hidden.shape[1]` khác `architecture.window_size`, code raise `ValueError`. Classification head cũng được tạo với input dimension bằng `architecture.window_size * architecture.hidden_dim`. Khi forward, model flatten `hidden_classification` thành shape `[B, self.window_size * self.hidden_dim]` trước khi đưa vào classifier.

Điều này làm cho `window_size` trở thành một phần của model contract, không chỉ là lựa chọn loader. Nếu checkpoint offline được train với `L=20`, các weight của classification head tương ứng với input dimension `20 * hidden_dim`; checkpoint đó không tương thích trực tiếp với model `L=100`, vì classifier input sẽ là `100 * hidden_dim`.

Online adaptation hiện load reference model từ checkpoint bằng `_load_reference_model(...)` trong `src/models/online_adaptation.py`. Hàm này lấy `config["model"]` và `config["task"]` từ checkpoint, dựng lại `ThesisMultitaskModel`, rồi `load_state_dict`. Nghĩa là reference model giữ nguyên `window_size` từ checkpoint. Trong cùng online path, `SMDOnlineStream` lại sinh window theo `experiment_config["data"]["window_size"]`. Nếu hai số này khác nhau, stream window và reference model không còn cùng một contract.

Quy trình synthetic anomaly hiện tại nằm trong `SyntheticAnomalyInjector`. Hàm `_sample_segment_bounds(...)` tính:

```text
min_segment_length = max(1, int(window_size * min_segment_fraction))
max_segment_length = max(min_segment_length, int(window_size * max_segment_fraction))
segment_length ~ randint(min_segment_length, max_segment_length)
```

Với active config `window_size=20`, `min_segment_fraction=0.2`, `max_segment_fraction=0.3`, synthetic span length là `4`, `5`, hoặc `6`. Khi `train_balance_classes=true`, code không còn dùng Bernoulli `anomaly_probability` để quyết định từng window, mà chia quota gần đều cho taxonomy active. Với `12` class gồm `1 normal + 11 anomaly class`, khoảng `11/12` synthetic windows là anomalous.

Các anomaly families hiện tại gồm `spike`, `flip`, `speedup`, `noise`, `cutoff`, `average`, `scale`, `wander`, `contextual`, `upsidedown`, và `mixture`. Đây là taxonomy tốt cho supervised synthetic anomaly-type classification, nhưng chúng vẫn được áp trên một segment ngắn nằm trong từng window riêng lẻ.

### Evaluation

Với câu hỏi window size, metric quan trọng nhất không phải metric score mà là **contract consistency** giữa data window, model checkpoint, và online stream. Offline `L=20` và online `L=100` không phải là một thay đổi evaluation đơn giản; trong code hiện tại nó đổi shape của model input contract và classifier head.

Với câu hỏi synthetic realism, span statistics là bằng chứng định lượng chính. Active synthetic `L=20` chỉ tạo segment `4-6` timestep. So với dữ liệu thật:

- `IOPS`: median `8`, p75 `21`; synthetic `4-6` gần nhóm rất ngắn nhưng vẫn thấp hơn p75.
- `SMD`: median `11`, mean `90`, p75 `72.5`; synthetic `4-6` chỉ mô phỏng phần cực ngắn.
- `AnomalyArchive`: median `100`, p75 `267.5`; synthetic `4-6` quá ngắn.
- `NASA`: median `120`, p75 `531`; synthetic `4-6` quá ngắn.
- `SWaT`: local `merged.csv` có một span `54621`; synthetic window-local không mô phỏng được loại span này.

Nếu dùng `L=100` với cùng fraction `0.2-0.3`, synthetic span length sẽ là `20-30`. Mức này gần hơn với `IOPS` p75 và một phần lower tail của `SMD`, nhưng vẫn thấp hơn rõ so với median của `AnomalyArchive`, median của `NASA`, và các long-span tails.

## Code References

- `documents/design/design_starter.md:14` - design starter ghi active window length hiện tại là `L = 20`.
- `documents/design/offline_pretraining_phase_two_view_contrastive_design.md:47` - offline pre-training two-view design ghi active window length là `L = 20`.
- `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml:12` - active three-stage offline experiment trỏ tới data config window20.
- `configs/data/smd_rtx3090_machine_3_4_20_stride1.yaml:5` - offline data config dùng `window_size: 20`.
- `configs/model/thesis_multitask_three_stage_window20.yaml:4` - offline thesis model config dùng `window_size: 20`.
- `configs/task/multitask_tsad_redlamp_multiclass_window20_benchmark_fixed_synth.yaml:10` - active synthetic policy dùng `anomaly_probability: 0.5`, class balancing, and segment fraction `0.2-0.3`.
- `configs/experiment/baseline/smd__online_adaptation__online-adaptation__w100__seed7__default.yaml:12` - online adaptation default experiment trỏ tới `configs/data/smd_smoke.yaml`.
- `configs/data/smd_smoke.yaml:3` - online smoke data config dùng `window_size: 100`.
- `scripts/run_online_adaptation.py:166` - online stream nhận `window_size` và `stride` từ data config.
- `src/data/stream.py:37` - `SMDOnlineStream` tạo sequential windows từ test sequences.
- `src/models/thesis_multitask.py:192` - encoder forward kiểm tra hidden window dimension.
- `src/models/thesis_multitask.py:998` - classification head input dimension là `window_size * hidden_dim`.
- `src/models/thesis_multitask.py:2395` - forward path kiểm tra `hidden_classification.shape[1] == self.window_size`.
- `src/models/online_adaptation.py:224` - online adaptation load reference `ThesisMultitaskModel` từ checkpoint config.
- `src/data/augment.py:198` - synthetic segment bounds được sample theo `window_size`.
- `src/data/augment.py:866` - synthetic anomaly được inject sau windowization trên batch `[B, L, D]`.
- `documents/logs/07-03-2026/research/anomaly_span_length_summary.csv:1` - dataset-level real anomaly span statistics.

## Pipeline Documentation

Pipeline offline active hiện nay nên được hiểu như sau:

```text
SMD raw sequence
-> train/val/test split
-> train-fitted scaling
-> windowization with L=20 for active three-stage offline configs
-> SyntheticAnomalyInjector on already-windowed batches
-> ThesisMultitaskModel with window_size=20
-> checkpoint containing model config and task config
```

Pipeline online adaptation hiện nay nên được hiểu như sau:

```text
load experiment config
-> build SMD test sequences
-> SMDOnlineStream(window_size=data_config.window_size)
-> OnlineWindowBatcher creates two online views
-> OnlineAdaptationModel loads frozen ThesisMultitaskModel from checkpoint
-> online projector alignment and optional prototype alignment
```

Vì online reference model được khôi phục từ checkpoint config, online stream window size phải khớp với checkpoint model window size nếu dùng cùng implementation path.

## Historical Context (from documents/)

`prompts/1_research_prompt.md` còn mô tả intended system với fixed-length windows of one hundred time steps. Tuy nhiên design documents mới hơn đã đồng bộ lại active thesis experiments về `L=20`. Điều này phản ánh tình trạng repo hiện tại: vẫn còn nhiều config `w100` cũ, nhưng active benchmark và three-stage thesis configs đã chuyển sang `w20`.

`documents/design/stream_design.md` nhấn mạnh rằng online evaluation phải là streaming/causal và xử lý từng sliding window theo thời gian. Tài liệu này không yêu cầu online adaptation phải dùng window size khác offline pre-training. Nó yêu cầu online path giữ được native window contract và causal evaluation.

Các research notes ngày 2026-06-30 đã ghi rõ synthetic anomaly hiện được inject sau windowization, với active `window_size=20` và segment length `4-6`. Nghiên cứu hiện tại mở rộng kết luận đó bằng cách đối chiếu trực tiếp với span statistics thật trong `documents/logs/07-03-2026/research/anomaly_span_length_summary.csv`.

## Open Questions

1. Nếu thesis muốn dùng `L=100` cho online adaptation, cần quyết định lại rằng offline checkpoint cũng phải được train với `L=100`, hoặc cần thiết kế một reference encoder/path không phụ thuộc classifier head shape. Đây chưa phải hành vi hiện tại.

2. Nếu synthetic anomaly cần tương xứng hơn với dữ liệu thật, cần một cơ chế sequence-level hoặc cross-window synthetic span. Cơ chế hiện tại chỉ tạo one-window local anomalies, nên không đại diện tốt cho long-span anomalies trong `SMD`, `NASA`, `AnomalyArchive`, và `SWaT`.

3. Nếu vẫn giữ `L=20`, cần xác định mục tiêu của synthetic augmentation là học anomaly-type perturbation cục bộ hay học real-span duration. Code hiện tại phù hợp hơn với mục tiêu thứ nhất.

