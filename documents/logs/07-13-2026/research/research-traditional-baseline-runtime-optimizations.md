---
date: 2026-07-13 22:14:39 +07:00
researcher: Artificial Intelligence Agent
git_commit: 818d386f3ace85cda174cdb162324cfd2f02fa41
branch: dev
repository: bachelor-thesis-2026
topic: "Reduce runtime of traditional ML baselines in the online streaming benchmark"
tags: [research, time-series, anomaly-detection, multi-class]
status: complete
last_updated: 2026-07-13
last_updated_by: Artificial Intelligence Agent
---

# Research: Reduce runtime of traditional ML baselines in the online streaming benchmark

**Date**: 2026-07-13 22:14:39 +07:00
**Researcher**: Artificial Intelligence Agent
**Git Commit**: `818d386f3ace85cda174cdb162324cfd2f02fa41`
**Branch**: `dev`

## Research Question
Anh thấy các traditional ML baseline trong online benchmark chạy chậm, kể cả smoke test. Em trace code thật để đề xuất 6 chiến lược khả dĩ giảm thời gian chạy.

## Summary
The current online benchmark launcher still performs full dataset bundle construction, including scaled sequences, `WindowDataset` objects, and `DataLoader` objects, even though the online benchmark only consumes `raw_sequences` from the bundle. For SMD `machine-1-6`, the raw train and test files each contain 23,688 to 23,689 rows, which means stride-1 streaming can reach roughly 23,670 windows per sequence before any cap is applied. The launcher already supports `task_overrides.max_online_steps`, but the benchmark configs inspected here do not pass that override, so smoke runs can still process the full stream. The biggest runtime costs are therefore data orchestration, window materialization, and per-window scoring rather than only the baseline estimator itself.

## Detailed Findings

### Data Preparation
- `configs/data/smd_benchmark_machine_1_6_window20.yaml` sets `window_size: 20`, `stride: 1`, `train_stride: 1`, `val_stride: 20`, `test_stride: 20`, `batch_size: 512`, and `num_workers: 12`.
- `src/data/datasets/smd.py` parses one train file, one test file, and one label file for the selected entity, then splits train into train/val by `validation_split_ratio`.
- `src/data/loaders.py` always builds scaled sequences, window datasets, and loaders. The online benchmark does not use the loaders, but it still pays for them.
- For `machine-1-6`, the train file and test file each have about 23.7k rows, so the stride-1 online stream can generate about 23.7k windows. That is already large enough to make a smoke run non-trivial.

### Modeling and Training
- `scripts/benchmarks/run_online_streaming_benchmark.py` loads the dataset bundle, selects exactly one train sequence and one clean validation sequence, calibrates the baseline, then runs the full test stream.
- `src/baselines/traditional/iforest.py` and `src/baselines/traditional/kmeans_ad.py` rebuild a full window matrix on each fit, calibration, and scoring pass.
- `src/baselines/traditional/stumpy_channel_ab.py` runs `stumpy.stump(...)` separately per channel, so its cost scales with both sequence length and channel count.
- `src/baselines/online/adaptive.py` scores stride-1 windows by looping across every possible window, then calibrates EWMA thresholds over the entire clean validation stream.
- `src/baselines/online/candi.py` and `src/baselines/online/m2n2.py` only differ in update policy; both inherit the same stride-1 scoring loop from `AdaptiveStreamingBaselineBase`.

### Evaluation
- The benchmark launcher truncates the test stream only when `task_overrides.max_online_steps` is present. It applies this cap before `baseline.run_sequence(...)`.
- `scripts/benchmarks/generate_online_benchmark_configs.py` already knows how to create smoke configs with `max_online_steps = 16` and smoke data overrides with `batch_size = 1` and `num_workers = 0`.
- The inspected hand-written smoke YAMLs under `configs/experiment/online_benchmark/...` do not appear to include the same `task_overrides` cap, so they can still run the full stream.

## Code References
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_online_streaming_benchmark.py:233-345` - benchmark control flow and `max_online_steps` truncation
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/data/loaders.py:150-228` - dataset bundle always builds scaled sequences, window datasets, and loaders
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/data/datasets/smd.py:62-182` - SMD parsing and train/val/test split
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/baselines/traditional/base.py:57-129` - non-overlap window matrix construction and robust calibration
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/baselines/traditional/iforest.py:58-143` - Isolation Forest fit and scoring path
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/baselines/traditional/kmeans_ad.py:61-147` - KMeansAD fit and scoring path
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/baselines/traditional/stumpy_channel_ab.py:55-260` - per-channel STUMPY AB-join path
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/baselines/online/adaptive.py:78-260` - stride-1 scoring loop and online updates
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/data/stream.py:38-250` - online stream materialization and batch assembly
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/configs/data/smd_benchmark_machine_1_6_window20.yaml:1-15` - smoke/runtime data config
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/configs/experiment/online_benchmark/iforest/smd__iforest__online_main__machine_1_6__w20__seed8__main.yaml:1-17` - main IForest config showing default `n_estimators: 100`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/configs/experiment/online_benchmark/kmeans_ad/smd__kmeans_ad__online_main__machine_1_6__w20__seed8__main.yaml:1-17` - main KMeansAD config showing default `n_clusters: 20`

## Pipeline Documentation
The online benchmark is not a tiny smoke harness by default. It parses the full SMD entity, builds the full bundle, calibrates on clean validation, then runs the test stream window by window. For adaptive baselines, each point requires a stride-1 window score, an EWMA score, a triage decision, and possibly a reference update. For traditional baselines, scoring is still repeated over every derived window and then projected back to point scores.

## Optimization Opportunities
1. Make smoke configs actually pass `task_overrides.max_online_steps` consistently, so smoke runs stop after a small prefix of the stream instead of the full test sequence.
2. Add a benchmark-only data path that returns `raw_sequences` without building `datasets` and `loaders`, because `run_online_streaming_benchmark.py` never consumes them.
3. Cache or reuse window matrices for a sequence across fit, calibrate, and score, so `build_window_matrix(...)` is not repeated for the same underlying array.
4. Use smaller estimator budgets for smoke and warm-up runs, especially `n_estimators` for Isolation Forest and the number of initializations for KMeans.
5. Replace per-window Python loops with vectorized sliding-window extraction or chunked scoring for stride-1 online baselines.
6. For STUMPY, reduce the work per call by limiting channels or by using a lighter benchmark subset, because the current channel-wise AB-join runs one full STUMPY query per channel.

## Open Questions
- Should smoke configs for the online benchmark be regenerated from `scripts/benchmarks/generate_online_benchmark_configs.py` so they inherit the `max_online_steps: 16` cap automatically?
- Should the benchmark launcher expose a raw-sequence-only loader path for any baseline that does not need PyTorch `DataLoader` objects?
- Should baseline smoke configs use a separate estimator budget from main configs, especially for `iforest` and `kmeans_ad`?
