---
date: 2026-06-30 13:21:29 +0700
researcher: Codex
git_commit: ddd20afb2f45c83a17fa93d54624789b783ca29d
branch: dev
repository: bachelor-thesis-2026
topic: "Code paths not yet simplified and easy to misunderstand"
tags: [research, codebase, readability, naming, runtime, configs]
status: complete
last_updated: 2026-06-30
last_updated_by: Codex
---

# Research: Code paths not yet simplified and easy to misunderstand

**Date**: 2026-06-30 13:21:29 +0700  
**Researcher**: Codex  
**Git Commit**: `ddd20afb2f45c83a17fa93d54624789b783ca29d`  
**Branch**: `dev`

## Research Question

Find parts of the current codebase that are still not simplified and are easy to misunderstand, using the repository research workflow and grounding every claim in the code as it exists now.

## Summary

The strongest current readability problem is no longer the main model logic itself. It is the mismatch between old and new surfaces around the RedLamp baseline family. One logical baseline is still exposed through multiple names, multiple config file identities, multiple registry aliases, and multiple legacy test/helper surfaces. This does not necessarily break runtime behavior, but it makes the codebase easier to misread.

The second strong problem is that some generic runtime layers still contain SMD-specific assumptions. The main example is realistic-validation anomaly-rate resolution, where the trainer, the task config surface, and the thesis model config still carry `test_smd_all` semantics even though the repository now discusses a broader multi-dataset benchmark direction.

The third strong problem is concentration of responsibility. `src/core/config.py` is currently a very large semantic choke point, and `validate_experiment_config(...)` has become a long monolithic validator. This is not automatically wrong, but it does create one file where naming migration, dataset support, task semantics, and training-policy checks all mix together.

## Detailed Findings

### 1. One RedLamp baseline still appears under too many names

This is the clearest current source of confusion.

- The canonical runtime model is now `redlamp_baseline`, but the compatibility shim still exists in `src/models/redlamp_mlp_baseline.py:1-13`.
- The training script registers both `redlamp_mlp_baseline` and `redlamp_baseline` to the same constructor in `scripts/train.py:44-52`.
- The evaluation script does the same in `scripts/evaluate.py:108-115`.
- The online adaptation script does the same in `scripts/run_online_adaptation.py:43-50`.
- The config validator still treats both names as supported model names in `src/core/config.py:301-307` and maps both names to the same allowed key set in `src/core/config.py:321-358`.
- The same duplication continues in numeric and boolean validation branches in `src/core/config.py:547-593` and `src/core/config.py:920-923`.

This means one real model family is still visible through at least these labels:

- `redlamp_baseline`
- `redlamp_mlp_baseline`
- `redlamp_cnn_baseline`

The extra confusion is that `configs/model/redlamp_cnn_baseline.yaml:1` says `model_name: redlamp_baseline`, while `configs/model/redlamp_baseline.yaml:1` also says `model_name: redlamp_baseline`, and `configs/model/redlamp_mlp_baseline.yaml:1` still says `model_name: redlamp_mlp_baseline`.

So the reader sees three config identities for what is now mostly one runtime family plus one legacy alias.

### 2. File names and config content no longer say the same thing

Several experiment files still carry the old baseline name in the file path, even though their internal content is already canonical.

Representative examples:

- `configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml:1-14`
- `configs/experiment/comparative/baseline/smd__redlamp_mlp_baseline__comparative-single-stage-machine_1_6__w20__seed6__main.yaml:1-13`

In both examples:

- the file name still says `redlamp_mlp_baseline`,
- but the internal owner is already `redlamp_baseline`,
- the `experiment_name` is already `...redlamp_baseline...`,
- and the `model_config_path` already points to a canonical `redlamp_baseline` model config.

So the outer filename and the inner runtime meaning no longer match. A reader who greps by path name can easily build the wrong mental model.

### 3. Comparative runner still accepts a name that resolved configs do not naturally produce

`scripts/run_comparative_smd_experiments.py:26-29` defines:

- `redlamp_baseline`
- `redlamp_mlp_baseline`
- `redlamp_cnn_baseline`

as supported baseline model names.

But the stage-family resolver uses `resolved_experiment_config["model"]["model_name"]` in `scripts/run_comparative_smd_experiments.py:113-118`.

At the same time:

- `configs/model/redlamp_cnn_baseline.yaml:1` resolves to `model_name: redlamp_baseline`.

So `redlamp_cnn_baseline` is a supported label in the comparative runner, but it is not the resolved runtime model name of the corresponding config preset. This is a subtle surface mismatch.

### 4. Script wiring is duplicated across multiple entry points

The same runtime wiring pattern appears in at least three scripts:

- `scripts/train.py:44-83`
- `scripts/evaluate.py:108-142`
- `scripts/run_online_adaptation.py:43-60`

Each script separately defines:

- dataset registration,
- model registration,
- and a `build_model_from_experiment_config(...)` helper.

This duplication is currently small enough to read, but it is still a real drift surface. The recent baseline rename already required coordinated changes across these parallel entry points.

### 5. SMD-specific realistic-validation semantics still leak into generic layers

This is the clearest remaining dataset-specific leakage.

- The trainer imports `compute_smd_test_window_anomaly_rate` directly from the SMD dataset module in `src/engine/trainer.py:36`.
- The realistic-validation anomaly-rate path uses `test_smd_all` as a task-level semantic token in `src/engine/trainer.py:603-610`.
- The task validator allows only `test_same_scope` and `test_smd_all` in `src/core/config.py:1110-1118`.
- The thesis multitask synthetic-anomaly config dataclass also hardcodes the same two values in `src/models/thesis_multitask.py:388-416`.

This means a generic trainer, a generic config validator, and a model-side configuration object all still know an explicitly SMD-named option.

After the recent anomaly-archive fallback fix, the runtime is safer than before. However, the semantic surface is still not fully dataset-agnostic.

### 6. The public dataset surface suggests less support than the repository layout suggests

The repository contains multiple dataset directories:

- `data/AnomalyArchive`
- `data/IOPS`
- `data/NASA`
- `data/SMD`
- `data/SWaT`
- `data/ibm-cloud-console-anomaly-dataset-iccad`

But the current runtime validator only supports two dataset names in `src/core/config.py:300-317`:

- `smd`
- `anomaly_archive`

The public data API also only exposes two loader families in `src/data/api.py:121-198`, and `src/data/__init__.py:3-13` only re-exports those two.

This is not inherently wrong if the repository intentionally supports only two active families. The confusing part is that the filesystem and surrounding discussion suggest a wider support surface than the actual validated runtime currently provides.

### 7. `src/core/config.py` has become a monolithic semantic choke point

`src/core/config.py` is currently `1481` lines long.

Top-level function boundaries show the main issue clearly:

- `load_yaml_config(...)` starts at `src/core/config.py:50`
- `validate_experiment_config(...)` starts at `src/core/config.py:199`
- the next top-level function, `load_experiment_config(...)`, only starts at `src/core/config.py:1409`

So `validate_experiment_config(...)` occupies roughly twelve hundred lines of one file.

Within that span, the same function is responsible for:

- allowed-key checking,
- dataset support decisions,
- model support decisions,
- training policy checks,
- three-stage config normalization,
- type checking,
- schedule semantics,
- realistic-validation semantics,
- and model-specific field semantics.

This centralization makes the code easy to locate, but hard to skim, because many unrelated semantic checks now share the same reading path.

### 8. Helper docs and launchers still teach the legacy surface

Two user-facing helper surfaces still advertise the old baseline path identity:

- `src/core/config_help.py:35-38` still shows train and evaluate examples using `configs/experiment/baseline/smd__redlamp_mlp_baseline__...`
- `scripts/launch_tmux_comparative_smd_experiment.sh:27-43` still hardcodes multiple legacy-named config paths

These helper surfaces may still run correctly because the files still exist. The confusion is that a new reader can follow these examples and conclude that the old name is still the main surface.

### 9. The test suite still preserves the old mental model in many places

Representative examples:

- `tests/test_cnn_encoder_config_loading.py:8`
- `tests/test_redlamp_gradient_conflict_metrics.py:8`
- `tests/test_redlamp_cnn_baseline_shapes.py:6`
- `tests/test_one_redlamp_mlp_train_step.py:6`
- `tests/test_redlamp_cnn_rerun_configs.py:4`

These tests still import `RedLampMLPBaseline` from the compatibility shim rather than the canonical `RedLampBaseline` class.

Other tests still explicitly use `model_name="redlamp_mlp_baseline"` in setup data, for example:

- `tests/test_comparative_preflight.py:84`
- `tests/test_comparative_runner.py:84`

This does not break correctness by itself. But it means the test suite still reinforces the legacy naming layer, so the codebase keeps two mental models alive at once.

### 10. The one-model-one-file rule now creates a readability tradeoff in the thesis model

This is a tradeoff rather than a mistake, but it is still important for readability.

- `src/models/thesis_multitask.py` is `3302` lines long.
- The main `ThesisMultitaskModel` class starts at `src/models/thesis_multitask.py:608`.
- Before that class starts, the same file already contains multiple helper functions, encoders, and many configuration dataclasses in `src/models/thesis_multitask.py:41-426`.

Under the repository rule "`1 model - 1 file`", this design is understandable. However, it means the file now behaves like a mini-subsystem inside one module. A new reader must hold a large amount of local context at once before reaching the main runtime class.

## Code References

- `src/models/redlamp_mlp_baseline.py:1-13` - compatibility shim for the renamed baseline.
- `scripts/train.py:44-83` - duplicated runtime registration and model-building logic.
- `scripts/evaluate.py:108-142` - parallel duplicated runtime registration and model-building logic.
- `scripts/run_online_adaptation.py:43-60` - third copy of baseline/runtime registration logic.
- `src/core/config.py:300-358` - supported names and alias-mapped model-key surface.
- `src/core/config.py:547-593` - repeated alias-aware numeric and float validation branches.
- `src/core/config.py:1110-1118` - SMD-specific `val_realistic_source` semantics in generic validation.
- `src/engine/trainer.py:570-610` - SMD-specific anomaly-rate logic inside the trainer.
- `src/models/thesis_multitask.py:388-416` - model-side config dataclass also knows `test_smd_all`.
- `src/data/api.py:121-198` - only two public dataset loader families are exported.
- `src/core/config_help.py:35-38` - helper examples still use legacy baseline config paths.
- `scripts/launch_tmux_comparative_smd_experiment.sh:27-43` - launcher still hardcodes legacy-named config paths.

## Pipeline Documentation

For this research question, the main runtime path that was inspected is:

1. Experiment config loading in `src/core/config.py`
2. Public and internal data loading surfaces in `src/data/api.py` and `src/data/loaders.py`
3. Baseline and thesis model naming/config surfaces in `src/models/`
4. Training and evaluation entry scripts in `scripts/train.py`, `scripts/evaluate.py`, and `scripts/run_online_adaptation.py`
5. Comparative launcher and preflight surfaces in `scripts/run_comparative_smd_experiments.py` and shell launchers
6. Test surfaces that still preserve legacy naming or runtime assumptions

## Historical Context (from documents/)

The current findings are consistent with the repository preferences in `codebase_preferences.md`:

- readability is the top priority,
- the number of code paths should stay low,
- and one model should stay in one file.

The current hotspots mostly appeared because the codebase has recently been migrating benchmark semantics and baseline naming while still preserving backward compatibility. So the main tension is not abstraction-heavy design. The main tension is transitional duplication.

## Open Questions

1. Is `redlamp_mlp_baseline` still required as a long-term compatibility surface, or is it now only temporary migration debt?
2. Should `redlamp_cnn_baseline` remain a preset-file name only, or should it also remain a meaningful runtime-facing identity?
3. Should `val_realistic_source` remain a shared task-level concept for future datasets, or should SMD-specific anomaly-rate estimation be isolated from generic trainer and generic task config surfaces?
4. Does the repository want the public runtime to continue supporting only `smd` and `anomaly_archive`, or should the code surface be updated to match the broader dataset inventory already present under `data/`?
