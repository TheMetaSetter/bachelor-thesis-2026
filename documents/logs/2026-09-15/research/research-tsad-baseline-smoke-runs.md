---
date: 2026-09-15
researcher: OpenAI Codex
topic: "Plan one local smoke batch for every TSAD-compatible tsad-lib baseline"
status: complete
---

# Research: TSAD Baseline Smoke Runs

## The story today

`tsad-lib` has one visible baseline catalog. `Catalog.list_models()` returns 41 entries. It contains Isolation Forest and 40 names copied from Time-Series-Library. It does not contain the special `thesis` method.

Today, 16 entries are `ready`: Isolation Forest plus Autoformer, Crossformer, DLinear, ETSformer, FEDformer, FiLM, Informer, KANAD, LightTS, MICN, Pyraformer, Reformer, TimesNet, Transformer, and iTransformer. The other 25 entries are `not_yet_tsad_ready`. They have no verified anomaly path in this library.

The current local environment passed the focused catalog and reference boundary suite: 19 passed. Those tests prove that every ready reference model can construct, reconstruct, train on one small synthetic batch, and produce finite scores. They do not prove a full real-data run for every model.

## The real baseline flow

`api.run()` creates an `ExperimentRunner`. `ExperimentRunner.run()` expands one request per selected model and runs requests one at a time. A request for an unready reference model ends as one failed `Result`; it does not silently disappear.

Isolation Forest fits its native runner and scores its native anomaly values. A ready reference model uses `build_reference_model()`, `ReferenceForwardAdapter`, and `TimeSeriesLibraryRunner`. The reference runner creates a reconstruction model, trains it with reconstruction MSE for `stage_a_epochs`, converts reconstruction error to scores, then uses the common metric writer.

Neither baseline path runs THESIS Stage A, memory initialization, Stage B, direct branch routing, or online TTA. The fixed THESIS smoke values of three Stage A epochs, two Stage B epochs, and 4,096 causal windows do not apply. The current shared `RunRequest` still creates an `O0-A0` baseline artifact identity. That identity is misleading: `O0/A0` are THESIS/MUSTARD ablation labels, not baseline adaptation policies. Do not run or report baseline cells under this identity; first make baseline requests omit those THESIS-only fields or use the selected method's own profile. Its resolved configuration also carries the shared `fusion_mode` field, but the baseline branch never reads that field for routing.

## One real source

`ServerMachineDataset/machine-1-3` is a suitable common smoke cell. Its train series has shape `(23702, 38)` and no supplied labels. Its test series has shape `(23703, 38)` and 817 anomaly labels. The active loader creates a train/validation split from the training series, fits scaling on the train portion, and uses 20-point offline windows for reference models. This is a real full-entity baseline smoke, not a tiny synthetic fixture.

## Artifacts and limits

Each completed baseline request writes `run_manifest.json`, `resolved_config.yaml`, `metrics.json`, `metrics.csv`, `metrics.md`, and `provenance.json`. The current baseline paths do not write THESIS checkpoints, threshold artifacts, online state, or online records.

`Report` writes `report.json`, `report.csv`, and `report.md` directly under its configured `output_root`. A new empty batch-specific output root prevents replacement of a previous aggregate report. The runtime then writes each model under that root's `benchmark_smoke` tree.

The local reference runner accepts only CPU through the public experiment path. The current CLI has no `--device` argument, while the notebook API accepts `device`; both paths use CPU for these baselines.

## Evidence

- `tsad-lib/src/tsad/catalog.py` — the visible model catalog begins with Isolation Forest and delegates the 40 reference names.
- `tsad-lib/src/tsad/models/time_series_library/registry.py` — readiness states and the 15 verified reference paths.
- `tsad-lib/src/tsad/experiment.py` — sequential request execution, unready-model failure rows, and separate THESIS and baseline branches.
- `tsad-lib/src/tsad/models/time_series_library/runner.py` — reconstruction-MSE fitting and score creation.
- `tsad-lib/src/tsad/reporting.py` — aggregate report files use `output_root`.
- `tsad-lib/src/tsad/data/adapters/server_machine.py` and `tsad-lib/src/tsad/pipeline.py` — real SMD loading, train-fitted scaling, and window preparation.
- `tsad-lib/tests/test_reference_models.py`, `test_reference_adapter.py`, and `test_model_registry.py` — the focused suite that passed 19 tests on 2026-09-15.

## Smoke-selection rule

The smoke batch must call `run()` with only models whose live catalog readiness is `ready`. The 25 unready names stay visible in `available(DATA)` and catalog inspection. They are not smoke-run inputs until their own TSAD path is certified.

## Terminology note

No runtime object is renamed. In this report, a **catalog entry** is any of the 41 visible non-THESIS model names. A **real smoke cell** is one of the 16 entries whose current readiness is `ready` and whose request completes on the real selected entity.
