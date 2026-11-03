---
date: 2026-09-15
planner: OpenAI Codex
topic: "Run one local smoke batch for every TSAD-compatible tsad-lib baseline"
status: approved_scope
related_research: documents/logs/2026-09-15/research/research-tsad-baseline-smoke-runs.md
---

# Plan: TSAD-Compatible Baseline Smoke Runs

## Summary

The batch will inspect the live baseline catalog on one real SMD machine. It will run only entries whose current readiness is `ready`. Today, this means 16 TSAD-compatible entries. It will not send the other 25 visible but unready entries to `run()`. It will not change code, model readiness, or the THESIS smoke contract.

The batch uses `ServerMachineDataset/machine-1-3`, seed `6`, CPU, and `benchmark_smoke`. It uses the current full entity because the baseline runtime has no smaller baseline-smoke selector. Reference models use their existing one-epoch reconstruction fit. Isolation Forest keeps its native fit. No baseline receives THESIS 3/2/4,096 work.

## Phase 1: freeze the honest selection

**Result:** The batch selects only real TSAD model cells.

**Tools:** `tsad-lib/.venv/bin/python`, pytest, the current `Catalog`.

### Stage 1.1: read the live catalog

1. Run `Catalog().list_models()` in `tsad-lib/.venv`.
2. Save the returned names and readiness states in the batch notes.
3. Select only names whose readiness is `ready`.
4. Confirm that exactly 16 names are `ready` before execution.
5. Record other catalog names as unavailable, without adding them to the batch.

**Complete when:** the selected list is derived from the live catalog and every selected model is TSAD-compatible.

### Stage 1.2: prove the local reference boundary

1. Run `tests/test_reference_models.py`, `tests/test_reference_adapter.py`, and `tests/test_model_registry.py` with `-p no:cacheprovider`.
2. Require every focused test to pass.
3. Stop if a ready model fails this suite.
4. Record the exact failing model and error before changing readiness or runtime code.

**Complete when:** the local environment proves the ready reference boundary before the real data batch begins.

### Stage 1.3: reserve one safe batch root

1. Choose one new empty `output_root` for this batch, such as `outputs/tsad-baseline-smoke-20260915-001`.
2. Confirm that the chosen directory does not exist.
3. Keep `experiment_type="benchmark_smoke"` inside that batch root.
4. Do not use `resume=True`.

**Complete when:** the aggregate `report.*` files and all model artifacts cannot replace an earlier batch.

## Phase 2: run every catalog entry once

**Result:** One report covers every currently compatible baseline.

**Tools:** public `tsad.run()` API, CPU, local `ServerMachineDataset` data.

### Stage 2.1: create the shared request

1. Set `data_root` to the thesis repository `data` directory.
2. Set `datasets=["ServerMachineDataset"]`.
3. Set `entities=["machine-1-3"]`.
4. Set `seed=6`.
5. Do not set THESIS-only `variant="O0"` or `online_variant="A0"` for baseline methods. The existing shared request's requirement for these fields is an API mismatch; the batch is blocked until baseline requests can omit them or use the selected method's own profile without encoding THESIS ablations.
6. Set `device="cpu"`.
7. Set `experiment_type="benchmark_smoke"`.

**Complete when:** every request has one unambiguous data entity, seed, device, and output root.

### Stage 2.2: execute the real batch

1. Read the live catalog.
2. Keep only model names whose readiness is `ready`.
3. Pass that compatible model list to one `run()` call.
4. Let `ExperimentRunner.run()` execute requests sequentially.
5. Let every selected entry train, score, evaluate, and write its own artifacts.
6. Do not retry a failed compatible model in the same output root.

**Complete when:** the aggregate report has one row for each selected compatible model. With today's catalog, it has 16 rows unless readiness changes before launch.

### Stage 2.3: keep method boundaries honest

1. Check the resolved configuration of one reference model.
2. Check that it records the existing one-epoch reference fit.
3. Check that it does not contain THESIS Stage B artifacts or online-state artifacts.
4. Check Isolation Forest separately because it does not use the reference reconstruction runner.
5. Treat the shared `fusion_mode` field as request metadata, not as baseline routing behavior.

**Complete when:** the output story preserves each baseline's native runtime boundary.

## Phase 3: verify the batch evidence

**Result:** A reader can distinguish completed TSAD runs from catalog names that were intentionally outside the batch.

**Tools:** JSON reader, SHA-free artifact existence checks, pytest only when a runtime issue appears.

### Stage 3.1: verify completed cells

1. For every completed row, read `run_manifest.json` and require `status="completed"`.
2. Require `resolved_config.yaml`, `metrics.json`, `metrics.csv`, `metrics.md`, and `provenance.json` in the matching model directory.
3. Require available values for `VUS-PR`, `VUS-ROC`, three `VUS-PR@FPR-budget` values, `Affiliation F1-score`, and `raw-FPR`.
4. Check that metric rows retain their availability state and warnings.

**Complete when:** every completed row has readable metric and provenance evidence.

### Stage 3.2: verify the selection boundary

1. Compare every report model name with the saved compatible model list.
2. Require every report model to have readiness `ready` at launch.
3. Record every unready catalog name outside the report.
4. Confirm that no unready model has a batch artifact directory.

**Complete when:** the report contains no incompatible model and no unavailable model is misreported as a zero-score baseline.

### Stage 3.3: close the smoke story

1. Read `report.json`, `report.csv`, and `report.md` from the batch root.
2. Confirm that all three contain the same compatible-model labels and statuses.
3. Record CPU, machine entity, seed, catalog counts, and focused-test result in a short execution note.
4. State only that these are local baseline smoke results for one real entity.
5. Do not claim all-dataset, all-entity, online-baseline, or wet-matrix readiness.

**Complete when:** the batch is reproducible and its limits are explicit.

## Execution command shape

The execution agent should run the public API from the sibling root. It should derive model names from the live catalog.

```python
from pathlib import Path

from tsad import run
from tsad.catalog import Catalog

models = [
    row["model"]
    for row in Catalog().list_models()
    if row["readiness"] == "ready"
]

report = run(
    data_root=Path("../bachelor-thesis-2026/data"),
    datasets=["ServerMachineDataset"],
    models=models,
    entities=["machine-1-3"],
    seed=6,
    output_root=Path("outputs/tsad-baseline-smoke-20260915-001"),
    experiment_type="benchmark_smoke",
    device="cpu",
)

report.show()
```

## Out of scope

This plan does not run or implement the 25 unready Time-Series-Library paths. It does not add RedLamp, CANDI, M2N2, Stumpy, or KMeansAD to `tsad-lib`. It does not execute THESIS, online TTA, the 1,764-cell SMD matrix, a wet run, or W&B logging.

## Terminology note

Do not label baseline smoke cells `O0-A0`. Those are THESIS/MUSTARD ablation labels. A baseline smoke identity contains the baseline method and its method-specific profile only. Resolve the shared API requirement above before running this batch.
