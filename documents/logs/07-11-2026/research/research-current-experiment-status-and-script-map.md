---
date: 2026-07-11T18:30:54+0700
researcher: Hermes Agent (GPT-5.6-Luna via OpenAI Codex)
git_commit: fbfd011ac85e94d559201fd2153161e5523ff8af
branch: dev
repository: bachelor-thesis-2026
topic: "Current experiment runability and script organization map"
tags: [research, time-series, anomaly-detection, multi-class, experiment-readiness, scripts]
status: complete
last_updated: 2026-07-11
last_updated_by: Hermes Agent
---

# Research: Current experiment status and script organization map

## Research Question

What is the current status of this codebase, can the full experiment described in `documents/spec/full-spec-v2.md` be run now, and what script-level material should be preserved for a later reorganization plan?

## Evidence snapshot

The repository was inspected at commit `fbfd011ac85e94d559201fd2153161e5523ff8af` on branch `dev`. The working tree was clean before and after inspection. The local environment is `.venv` with Python 3.12.12.

Available SMD arrays were found at `data/SMD/`:

- `SMD_train.npy`: `(708405, 38)`, `float32`
- `SMD_test.npy`: `(708420, 38)`, `float32`
- `SMD_test_label.npy`: `(708420,)`, `float32`

The current active protocol is window length `20`, offline stride `20`, online stride `1`, clean-validation quantile calibration, EWMA weights `0.9` current and `0.1` previous, and test-label usage limited to metrics.

## Executive conclusion

The codebase is runnable for structural validation, unit tests, dry-run planning, and local smoke artifacts. It is not yet demonstrably ready to claim completion of the full scientific experiment described by `full-spec-v2`.

The strongest verified status is:

- Full matrix preflight: passes structurally and reports `ready` for `18` THESIS offline, `54` THESIS online, `9` RedLamp offline, `27` traditional offline, and `81` online baseline configurations.
- Pytest: `398 passed, 22 warnings` in `45.37s`.
- THESIS offline and online smoke dry-run wrappers: execute successfully and write dry-run reports/integrity manifests.
- CUDA-required preflight: fails on this macOS host with `CUDA is required but no CUDA device is available`.
- Existing output tree: contains smoke/dry-run and historical artifacts, but no `completion_manifest.json` or `online_completion_manifest.json` was found, and no complete matrix artifact was verified by this audit.

Therefore, the answer to “can I run the full experiment now?” is:

> Not as an accepted full run from this machine and repository state. The launch configuration surface exists, but structural readiness is not the same as scientific completion. A CUDA-capable environment and the remaining full-spec runtime, artifact-integrity, resume, readability, and ordered smoke gates are still required before the 189-cell result can be treated as complete.

## Current implemented pipeline

### Data and protocol

The generated benchmark matrix covers three SMD entities (`machine-1-6`, `machine-3-4`, `machine-3-9`) and three seeds (`6`, `8`, `36`). The active benchmark window length is `20` with `38` input channels in the inspected SMD arrays.

The protocol file `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml` declares:

```yaml
window_size: 20
offline_window_stride: 20
online_window_stride: 1
offline_threshold_split: clean_validation
online_threshold_split: clean_validation
online_ewma_current_weight: 0.9
online_ewma_previous_weight: 0.1
test_label_usage: metrics_only
point_adjustment: false
```

### Offline THESIS path

`configs/experiment/offline_benchmark/thesis/` contains O0 and O1 main/smoke configurations. The full-matrix preflight verifies `18` main THESIS offline files and enforces a `25 + 5` Stage-A/Stage-B budget through the resolved two-stage configuration.

The active wrapper is `scripts/run_thesis_offline_benchmark.py`. It delegates Stage A and Stage B execution to `scripts/run_two_stage_offline_pretraining.py`, then evaluates clean validation, synthetic validation, and test splits, builds thresholds, and exports score/metric/protocol artifacts.

Relevant observed flow:

```text
run_thesis_offline_benchmark.py
  -> load_experiment_config()
  -> validate_protocol_config()
  -> validate_two_stage_epoch_budget()
  -> materialize_two_stage_run_manifest()
  -> execute_two_stage_plan()
  -> collect_offline_artifact_inputs()
  -> write thresholds, scores, metrics, resolved protocol, report
```

O1 is represented by separate generated configurations. The normative point-score-loss behavior remains specified in `documents/spec/full-spec-v2.md`; this audit did not treat a config count or dry-run report as proof that all scientific acceptance evidence exists.

### Online THESIS path

`configs/experiment/online_benchmark/thesis/` contains the O0/O1 × A0/A1/A2 × three-entity × three-seed configuration surface. The full-matrix preflight verifies `54` online THESIS main files and the expected six variant names.

The active wrapper is `scripts/run_thesis_online_benchmark.py`. It validates the protocol, calls `src.engine.online_tta.online_engine.run_thesis_online_tta_experiment()`, normalizes records, writes a benchmark report, and writes/verifies a report integrity manifest.

The current implementation and readiness documents still identify important full-spec-sensitive areas for verification/remediation: one-window source-forward semantics, exact latent score meaning, label-free verification, PNN/TTL ownership, exact A2 contrastive sets, full-stream completion, resume identity, and canonical completion artifacts. The presence of the online engine and tests does not by itself close these gates.

### Baselines and matrix

The repository also has separate wrappers for traditional offline baselines (`scripts/run_offline_benchmark.py`) and online streaming baselines (`scripts/run_online_streaming_benchmark.py`). The preflight reports the intended matrix as:

```text
THESIS offline:       18
THESIS online:        54
RedLamp offline:       9
Traditional offline:  27
Online baselines:     81
Total:               189
```

The online baseline wrapper has its own builder registry for CANDI, M2N2, STUMPY, KMeansAD, and Isolation Forest. It is not the THESIS online engine.

## Verification results

### Commands exercised

```text
.venv/bin/python scripts/preflight_full_benchmark_matrix.py --json
.venv/bin/python -m pytest -q
.venv/bin/python scripts/preflight_full_benchmark_matrix.py --require-cuda --json
.venv/bin/python scripts/run_thesis_offline_benchmark.py --experiment-config <O0 smoke> --dry-run
.venv/bin/python scripts/run_thesis_online_benchmark.py --experiment-config <O0_A0 smoke> --online-variant A0 --dry-run
```

Observed results:

- Matrix preflight exited `0` and returned `status: ready` with the counts above.
- Pytest exited `0`: `398 passed, 22 warnings`.
- CUDA-required preflight exited nonzero: no CUDA device is available.
- Offline dry-run exited `0` and materialized the two-stage plan with `1 + 1` smoke epochs.
- Online A0 dry-run exited `0` and reported verified report integrity.

The warnings were metric-library warnings for one-class slices and one STUMPY warning; they did not fail the test suite.

### Why this is not a full-run clearance

The normative specification explicitly says that a run is not complete merely because its configuration resolves or its matrix cell is enumerated. The current evidence still lacks, in this audit, a verified complete 189-cell execution with:

- CUDA smoke evidence in the target environment;
- full-stream causal coverage for every main online cell;
- interruption/resume equivalence evidence;
- completion manifests and checksum readback for every cell;
- identity-safe `--skip-completed` behavior;
- artifact-integrity and metric-availability aggregation;
- readability/compliance closure for all `src/` files and callables;
- a complete comparative report that marks every matrix cell completed, failed, or missing.

The existing smoke output tree should be understood as evidence that selected paths have been exercised, not as evidence that the main matrix is complete.

## Script organization map for later planning

This section records current boundaries and representative snippets. It is intentionally descriptive rather than a refactoring proposal.

### 1. Configuration generation

` scripts/generate_offline_benchmark_configs.py:1-150 `

Purpose observed: generate traditional offline benchmark YAML files from entity, seed, and method dimensions.

Representative boundary:

```python
def generate_offline_benchmark_configs(...):
    ...

def main():
    ...
```

` scripts/generate_online_benchmark_configs.py:1-274 `

Purpose observed: generate THESIS online configs. The file contains separate helpers for experiment names, output paths, reference checkpoint paths, model/task/data overrides, and the six O/A variants.

Representative path contract:

```python
_reference_checkpoint_path(...)
    -> outputs/.../two_stage/stage_b_fusion_finetuning/checkpoints/best.pt

_task_overrides(...)
    -> max_online_steps = 16 for smoke, None for main
```

` scripts/generate_online_streaming_benchmark_configs.py:1-200 `

Purpose observed: generate online baseline configs for CANDI, M2N2, STUMPY, KMeansAD, and Isolation Forest.

` scripts/generate_smd_benchmark_configs.py:1-244 `

Purpose observed: generate the older/general SMD THESIS offline benchmark configuration family, including O0/O1-like variant model overrides and output naming.

### 2. Offline execution

` scripts/run_two_stage_offline_pretraining.py:1-435 `

Purpose observed: owns the two-stage plan, stage-specific generated configs, Stage-A checkpoint transfer, Stage-B initialization, execution, and run manifest.

Representative boundary:

```python
build_two_stage_training_plan(...)
materialize_two_stage_run_manifest(...)
execute_two_stage_plan(...)
```

` scripts/run_thesis_offline_benchmark.py:1-390 `

Purpose observed: THESIS benchmark/report boundary around the two-stage owner. It does not train directly; it delegates to the two-stage runner and then performs evaluation/calibration/export.

Representative artifact boundary:

```python
artifact_inputs = collect_offline_artifact_inputs(...)
artifact_paths = _export_offline_artifacts(...)
report_path = _write_report(...)
```

` scripts/run_offline_benchmark.py:1-366 `

Purpose observed: traditional offline baseline runner. It builds a baseline from `BASELINE_BUILDERS`, fits on train, calibrates on clean validation, scores test, computes pointwise metrics, and writes threshold/report artifacts.

Representative registry:

```python
BASELINE_BUILDERS = {
    "stumpy_channel_ab": StumpyChannelABFrozenTrainRef,
    "kmeans_ad": KMeansADWindowBaseline,
    "iforest": IForestWindowBaseline,
}
```

` scripts/train.py:1-445 `

Purpose observed: shared registry/config-driven model construction, optimizer/scheduler construction, and generic training execution. It is imported by the THESIS offline wrapper.

Representative construction boundary:

```python
model = build_model_from_experiment_config(experiment_config)
optimizer = build_optimizer_from_experiment_config(model, experiment_config)
scheduler = build_scheduler_from_experiment_config(optimizer, experiment_config)
```

### 3. Online execution

` scripts/run_thesis_online_benchmark.py:1-175 `

Purpose observed: THESIS online report boundary. It accepts one experiment config and A0/A1/A2, delegates runtime behavior to `src.engine.online_tta`, normalizes records, and verifies the report manifest.

Representative boundary:

```python
online_outputs = run_thesis_online_tta_experiment(...)
report["online_execution"] = online_outputs
```

` scripts/run_online_adaptation.py:1-298 `

Purpose observed: lower-level generic online adaptation entrypoint that constructs the model/optimizer and calls the generic online adaptation experiment. This is a distinct script surface from the THESIS benchmark wrapper.

` scripts/run_online_streaming_benchmark.py:1-343 `

Purpose observed: online baseline runner. It fits/initializes a baseline, calibrates validation, iterates test sequences, calls `baseline.run_sequence()`, and writes online threshold, metrics, records, and report artifacts.

Representative loop:

```python
for sequence in test_sequences:
    sequence_metric_history, sequence_records = baseline.run_sequence(...)
    metric_history.extend(sequence_metric_history)
    records.extend(sequence_records)
```

### 4. Preflight and orchestration

` scripts/preflight_full_benchmark_matrix.py:1-181 `

Purpose observed: no-train structural matrix/config validator. It counts config files, validates resolved config semantics, checks THESIS two-stage epochs, checks variants, and emits the `18/54/9/27/81` report.

Representative contract:

```python
_validate_thesis_offline(paths)  # expected 18; enforces 25 + 5
_validate_thesis_online(paths)   # expected 54; enforces six O/A variants
```

` scripts/preflight_comparative_smd_server.py:1-233 `

Purpose observed: server-side launch preflight for comparative tmux execution. It checks GPU/device readiness, tmux availability, data roots, and launch readiness.

` scripts/preflight_three_stage_server.py:1-357 `

Purpose observed: older/parallel three-stage server preflight. It computes data readiness, uncapped-window counts, GPU requirements, and writes a preflight summary.

` scripts/run_comparative_smd_experiments.py:1-878 `

Purpose observed: large comparative orchestrator. It normalizes config paths, validates data roots and artifact paths, builds command plans, materializes worker overrides, executes groups, loads existing execution reports, checks required artifacts, and prints run records.

Representative planning boundary:

```python
plan = build_comparative_run_plan(...)
execute_comparative_run_plan(plan, ...)
```

` scripts/launch_tmux_comparative_smd_experiment.sh:1-302 `

Purpose observed: shell launcher. It defines smoke/main config arrays, parses GPU/report/session options, constructs preflight and runner commands, optionally prints them, or starts a tmux session with redirected logs.

Representative command composition:

```bash
PREFLIGHT_COMMAND=("${PYTHON_BIN}" scripts/preflight_comparative_smd_server.py ...)
RUNNER_COMMAND=("${PYTHON_BIN}" scripts/run_comparative_smd_experiments.py ...)
```

### 5. Reporting, analysis, visualization

` scripts/summarize_benchmark_results.py:1-357 ` reads report JSON files, extracts method/variant/entity/seed/threshold/metric fields, and writes summary JSON/CSV outputs.

` scripts/forensic_audit_run.py:1-200 ` builds a forensic markdown report from observed metrics.

` scripts/verify_three_stage_run.py:1-135 ` checks required three-stage artifacts and writes a verification summary.

` scripts/visualize_evaluation_results.py:1-353 ` loads evaluation outputs and test sequences to render entity-level result visualizations.

` scripts/visualize_synthetic_anomalies.py:1-279 ` builds a demonstration batch, injects synthetic anomalies, and plots affected channels/masks.

` scripts/compare_synthetic_profiles.py:1-625 ` compares synthetic profiles and produces galleries/plots; it is one of four script files above the repository's 500-line readability limit.

## Script-level facts relevant to future reorganization planning

1. There are `34` direct files under `scripts/` in the current inventory, including Python launchers and two shell launchers.
2. Four script files exceed the repository readability limit of 500 lines: `compare_synthetic_profiles.py` (626), `run_comparative_smd_experiments.py` (879), `run_three_stage_offline_pretraining.py` (822), and `summarize_anomaly_span_lengths.py` (580).
3. There are multiple execution families with overlapping responsibilities: generic training/adaptation, THESIS wrappers, traditional baseline wrappers, online baseline wrappers, comparative orchestration, tmux launchers, and legacy three-stage execution.
4. Configuration generation is split across three benchmark generators plus the older/general SMD generator.
5. The THESIS offline wrapper delegates to the two-stage runner, while the THESIS online wrapper delegates to the `src.engine.online_tta` engine. These are useful current ownership facts for any future reorganization map.
6. The matrix preflight is structural only; it does not train, execute the test stream, or prove output completion.
7. The shell launcher is CUDA/tmux-oriented and is not a substitute for the Python matrix preflight.
8. Existing smoke configuration caps use `max_online_steps: 16`; full main configuration generation currently emits an uncapped value (`None`) in the inspected generator, but full-stream completion still needs runtime evidence.

## Open questions

- Which CUDA host/environment will be used for the main 189-cell launch, and what exact PyTorch/CUDA lockfile will be recorded?
- Are all main Stage-B checkpoints and entity-specific threshold artifacts already present, valid, and identity-matched, or must the 18 THESIS offline runs be completed first?
- Which of the existing smoke outputs are accepted as current scientific evidence versus retained historical/debug artifacts?
- Should the older three-stage scripts remain supported as a legacy experiment family or be isolated from the active two-stage/full-spec path during future organization work?
- What is the authoritative single command for a complete matrix after CUDA, resume, artifact, and readability gates are closed?

## Code References

- `documents/spec/full-spec-v2.md` — normative offline/online experiment, artifact, demo, and acceptance contract.
- `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml` — active SMD window/threshold protocol.
- `scripts/preflight_full_benchmark_matrix.py:117-157` — structural matrix report.
- `scripts/run_thesis_offline_benchmark.py:322-367` — THESIS offline execution/report boundary.
- `scripts/run_thesis_online_benchmark.py:91-152` — THESIS online execution/report boundary.
- `src/engine/online_tta/online_engine.py:1308-` — THESIS online engine entrypoint.
- `documents/logs/07-11-2026/plan/plan-full-spec-v2-experiment-readiness-remediation.md:53-81` — latest documented implementation state and result-changing gaps.
- `documents/logs/07-11-2026/detail/detail-full-spec-v2-experiment-readiness-remediation.md:24-40` — accepted remediation batches and execution order.

## Status

Research complete. This note documents current executable evidence and preserves script ownership/boundary snippets for a later structure/reorganization planning task.
