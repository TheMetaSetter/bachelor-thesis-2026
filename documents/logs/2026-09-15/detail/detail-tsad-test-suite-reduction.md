---
date: 2026-09-15
topic: "Reduce the tsad-lib pytest suite to 28 collected contract tests"
status: completed
source_structure: documents/logs/2026-09-15/structure/structure-tsad-test-suite-reduction.md
related_documents:
  - documents/logs/2026-09-15/research/research-tsad-test-suite-reduction.md
  - documents/logs/2026-09-15/plan/plan-tsad-test-suite-reduction.md
---

# Detail: A 28-Case TSAD Test Suite

## Summary

The agent replaced the 119-node pytest suite with 28 contract tests. The runtime code stayed unchanged. Each replacement contract ran before its old duplicate tests were removed.

Evidence on 2026-09-15:

```text
28 tests collected
28 passed
```

## Target files

| Proposed file | Cases | Responsibility |
| --- | ---: | --- |
| `tsad-lib/tests/test_data_contracts.py` | 5 | Data enters the library safely and metrics keep their meaning. |
| `tsad-lib/tests/test_baseline_contracts.py` | 4 | Each TSAD-compatible baseline path can be selected and scored. |
| `tsad-lib/tests/test_thesis_contracts.py` | 8 | THESIS offline and online variants keep their locked rules. |
| `tsad-lib/tests/test_public_contracts.py` | 6 | Notebook and CLI users reach the correct runtime. |
| `tsad-lib/tests/test_release_contracts.py` | 5 | Artifacts, matrix gates, documents, and tutorials stay release-ready. |

## Phase 1: make a safe replacement map

### Stage 1.1: record the old suite

1. Run the collection command in `tsad-lib`.
2. Record the 119 collected-node result.
3. Create the five proposed test files without removing an old test.

### Stage 1.2: define the count contract

1. Add one collection assertion procedure to the execution checklist.
2. Set its expected result to 28.
3. Set its failure condition to any result above 30.

## Phase 2: write the new tests before removal

### Stage 2.1: write five data contracts

1. Write `test_types_and_discovery_contract` in `test_data_contracts.py`.
2. Run that test and confirm it passes before removing `test_types.py` and `test_discovery.py` behavior.
3. Write `test_adapter_inspection_and_ambiguity_contract` with a loop over all adapter inspection cases.
4. Run that test and confirm each failure message includes the adapter name.
5. Write `test_adapter_loading_contract` with a loop over all supported adapter loading cases.
6. Run that test and confirm each loaded source has the expected shape and labels.
7. Write `test_pipeline_contract` for train-only scaling, half-open indices, causal stride, and hidden training labels.
8. Run that test and confirm it passes.
9. Write `test_scores_and_metrics_contract` for score reduction, VUS rows, affiliation F1, one-class unavailability, and uncovered points.
10. Run that test and confirm unavailable metrics remain `None` instead of zero.

### Stage 2.2: write four baseline contracts

1. Write `test_catalog_and_smoke_map_contract` in `test_baseline_contracts.py`.
2. Run that test and confirm the map prints only `readiness == "ready"` models.
3. Write `test_isolation_forest_contract` for fit, score, and prediction shapes.
4. Run that test and confirm it passes.
5. Write `test_reference_forward_adapter_contract` for input construction, model call, reconstruction extraction, and finite-shape rejection.
6. Run that test and confirm it passes.
7. Write `test_all_ready_reference_models_contract` with one loop over the 15 ready model names.
8. Run that test and confirm each model constructs, reconstructs, fits, and scores.

### Stage 2.3: write eight THESIS contracts

1. Write `test_o2_config_contract` in `test_thesis_contracts.py` for the three losses, no Balanced Point-Score Loss, direct routing, metric defaults, and invalid variant rejection.
2. Run that test and confirm it passes.
3. Write `test_two_view_contrastive_loss_contract` for clean batch points and the no-clean-point zero case.
4. Run that test and confirm it passes.
5. Write `test_o2_offline_stage_contract` for Stage A checkpoint, both memory banks, frozen encoder, skipped fusion blocks, and raw-input identity threshold.
6. Run that test and confirm it passes.
7. Write `test_fusion_override_contract` for the explicit non-default O0 routing record.
8. Run that test and confirm it passes.
9. Write `test_a0_online_contract` for causal absolute indices and unchanged projector values.
10. Run that test and confirm it passes.
11. Write `test_a1_online_contract` for verified PNN-only projector updates.
12. Run that test and confirm it passes.
13. Write `test_a2_online_contract` for hard-old contrastive updates.
14. Run that test and confirm it passes.
15. Write `test_online_state_contract` for threshold-identity rejection during state load.
16. Run that test and confirm it passes.

### Stage 2.4: write six public contracts

1. Write `test_notebook_discovery_contract` in `test_public_contracts.py` for public imports, available rows, inspect rows, load splits, ICCAD labels, SMD layout, and visible devices.
2. Run that test and confirm it passes.
3. Write `test_public_baseline_contract` for one completed Isolation Forest cell beside one explicit unready-model failure row.
4. Run that test and confirm the report writes `report.json`.
5. Write `test_public_reference_contract` for one ready reference model with VUS-PR.
6. Run that test and confirm it passes.
7. Write `test_public_thesis_smoke_contract` for O2, 3 Stage A epochs, 2 Stage B epochs, and 4,096 causal windows.
8. Run that test and confirm it passes.
9. Write `test_public_device_contract` for explicit CPU provenance and rejection of accelerated non-THESIS models.
10. Run that test and confirm it passes.
11. Write `test_cli_contract` for `list models`, `run`, and the same request fields as notebook execution.
12. Run that test and confirm it passes.

### Stage 2.5: write five release contracts

1. Write `test_artifact_and_report_contract` in `test_release_contracts.py` for JSON, CSV, Markdown, warnings, unavailable metric values, and required stage files.
2. Run that test and confirm it passes.
3. Write `test_output_root_resume_contract` for smoke and wet root separation, overwrite rejection, and valid resume.
4. Run that test and confirm it passes.
5. Write `test_smd_matrix_preflight_contract` for 28 entities, 1,764 logical units, anomaly-containing range, ready verified smoke gate, and blocking invalid smoke gate.
6. Run that test and confirm it passes.
7. Write `test_document_contract` for the O2, A1, and A2 ontology phrases.
8. Run that test and confirm it passes.
9. Write `test_tutorial_notebook_contract` with a loop over the four ICCAD tutorial notebooks.
10. Run that test and confirm each notebook has its expected TODO count and small code cells.

## Phase 3: remove only proven duplicates

### Stage 3.1: remove old data and baseline tests

1. Remove `test_types.py` after `test_types_and_discovery_contract` passes.
2. Remove `test_discovery.py` after `test_types_and_discovery_contract` passes.
3. Remove `test_adapter_inspection.py`, `test_array_text_adapters.py`, and `test_table_archive_mat_adapters.py` after the data contracts pass.
4. Remove `test_catalog.py`, `test_model_registry.py`, `test_method_runner.py`, `test_reference_adapter.py`, `test_reference_models.py`, and `test_map_tsad_baseline_smoke_script.py` after the baseline contracts pass.

### Stage 3.2: remove old THESIS, public, and release tests

1. Remove `test_config.py`, `test_variants.py`, `test_thesis_losses.py`, `test_thesis_runtime.py`, and `test_thesis_online.py` after the THESIS contracts pass.
2. Remove `test_api.py`, `test_cli.py`, and `test_public_runtime.py` after the public contracts pass.
3. Remove `test_artifacts.py`, `test_reporting.py`, `test_scores.py`, `test_metrics.py`, `test_matrix.py`, `test_document_contract.py`, and `test_tutorial_notebooks.py` after the release contracts pass.

## Phase 4: prove the cap

### Stage 4.1: collect and run

1. Run `PYTHONPATH=src .venv/bin/python -m pytest --collect-only -q -p no:cacheprovider tests` from `tsad-lib`.
2. Confirm pytest prints `28 tests collected`.
3. Run `PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests` from `tsad-lib`.
4. Confirm pytest reports 28 passing tests.

### Stage 4.2: record the result

1. Update `documents/spec/tsad-lib-development-spec.md` with the fresh count and command.
2. Mark the test-suite-minimization checklist item complete only after both commands pass.

## Constraints

- Do not change `src/tsad` behavior to make a test easier.
- Do not parameterize a test when it would create more collected pytest nodes.
- Do not hide a unique contract in a large unrelated test.
- Include a failing member name in every loop assertion.
- Run all test commands with `tsad-lib/.venv/bin/python`.
