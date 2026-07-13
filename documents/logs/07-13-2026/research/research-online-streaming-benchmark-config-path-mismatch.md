---
date: 2026-07-13T21:53:54+07:00
researcher: Codex
git_commit: 818d386f3ace85cda174cdb162324cfd2f02fa41
branch: dev
repository: bachelor-thesis-2026
topic: "Online streaming benchmark config path mismatch and similar path-name errors"
tags: [research, benchmark, config-path, online-streaming, file-not-found]
status: complete
last_updated: 2026-07-13
last_updated_by: Codex
---

# Research: Online streaming benchmark config path mismatch and similar path-name errors

**Date**: 2026-07-13T21:53:54+07:00
**Researcher**: Codex
**Git Commit**: 818d386f3ace85cda174cdb162324cfd2f02fa41
**Branch**: dev

## Research Question
Investigate the `FileNotFoundError` raised by `scripts/benchmarks/run_online_streaming_benchmark.py` for a `candi` smoke run, and identify similar path-name errors elsewhere in the codebase.

## Summary
The failure is not a loader bug in the narrow sense. The command used an invalid benchmark-config path:

`configs/experiment/online_benchmark/candi/smd__candi__online_main__machine_1_6__w20__seed8__smoke.yaml`

In this repository, `candi` and `m2n2` online-streaming configs are generated with `online_A0`, `online_A1`, and `online_A2`. The `online_main` variant is only used by `stumpy`, `kmeans_ad`, and `iforest`. Therefore, the failing path is a naming mismatch between the benchmark variant slot and the benchmark mode suffix.

The runner itself is strict and path-exact. It resolves the provided config relative to the repository root and forwards it to `load_yaml_config`; if the file is not present, it raises `FileNotFoundError` immediately. The same strict behavior appears in the offline benchmark runner.

## Detailed Findings

### Data Preparation
- Online-streaming benchmark configs are generated under `configs/experiment/online_benchmark/`.
- The generator distinguishes between method families:
  - `candi`, `m2n2` use variants `A0`, `A1`, `A2`.
  - `stumpy`, `kmeans_ad`, `iforest` use variant `main`.
- The generator also emits a mode suffix:
  - `main`
  - `smoke`

### Modeling and Training
- Not applicable to this failure. The error occurs before dataset loading or model instantiation.
- The launcher simply loads the benchmark config, then loads the data config and instantiates the baseline.

### Evaluation
- The failure happens in `scripts/benchmarks/run_online_streaming_benchmark.py` before evaluation starts.
- The loader path is exact:
  - if the config path is relative, it is joined to the repository root;
  - then `load_yaml_config()` is called directly.
- There is no fallback search across alternate config trees or aliasing for nearby filenames.

## Code References
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/generate_online_streaming_benchmark_configs.py:12-25` - method/variant mapping, including `candi` and `m2n2` as `A0/A1/A2` and `stumpy/kmeans_ad/iforest` as `main`.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/generate_online_streaming_benchmark_configs.py:124-183` - benchmark config construction and write path under `configs/experiment/online_benchmark/`.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_online_streaming_benchmark.py:60-64` - repository-root path resolution for benchmark configs.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_online_streaming_benchmark.py:239-244` - benchmark and protocol config loading.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/tests/online/test_online_streaming_benchmark_config_generation.py:17-40` - test snapshot that expects `candi` smoke files under `configs/experiment/online_benchmark/candi/` and uses `online_A0`.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_offline_benchmark.py:71-75` - same exact repository-root config resolution pattern for offline benchmarks.

## Pipeline Documentation
The current online-streaming benchmark surface is intentionally literal. The file path must match the generated config name exactly. For `candi` and `m2n2`, the correct pattern is:

`configs/experiment/online_benchmark/<method>/smd__<method>__online_A0|A1|A2__<entity>__w20__seed<seed>__main|smoke.yaml`

For `stumpy`, `kmeans_ad`, and `iforest`, the correct pattern is:

`configs/experiment/online_benchmark/<method>/smd__<method>__online_main__<entity>__w20__seed<seed>__main|smoke.yaml`

The observed failure used the second pattern for a first-pattern method, so the path did not exist.

## Historical Context (from documents/)
- `documents/spec/full-spec-v3.md` is the current SSOT. It explicitly frames the benchmark flow as CPU tests/preflight, then CUDA smoke gates, then main runs, and it uses causal A0/A1/A2 online execution as the normative online target.
- `documents/logs/07-13-2026/detail/detail-thesis-online-benchmark-checkpoint-metadata-resolver.md` documents the online THESIS checkpoint resolver contract, which is separate from this baseline streaming config issue.

## Open Questions
- Should the benchmark CLI emit a closer-match diagnostic when a config path is missing?
- Should the repository keep both `configs/experiment/...` and `scripts/configs/experiment/...` trees, or should one be treated as deprecated to reduce naming drift?
- Should there be a test that explicitly rejects `candi`/`m2n2` configs named with `online_main`?
