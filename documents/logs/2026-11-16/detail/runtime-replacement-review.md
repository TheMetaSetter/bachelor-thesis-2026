# Runtime replacement review

Execution date: 2026-11-17.
Plan: [Implementation checklist](../plan/plan-implement-runtime-signatures.md).
Status: the replacement runtime and cleanup are complete. See the [completion report](#completion-report--2026-11-19) for checks, size counts, and limits.

## Starting state

Revision: `034833af231be2d6118861be38b331a584c35d44`. Branch: `simplifying`.

The following changes existed before this execution. Keep them unchanged.

```text
 D bsc-thesis-ref-codebases/MtsCID/README.md
 D bsc-thesis-ref-codebases/MtsCID/SMD_checkpoint.pth
 D bsc-thesis-ref-codebases/MtsCID/data/NIPS_TS_GECCO/NIPS_TS_Water_test.npy
 D bsc-thesis-ref-codebases/MtsCID/data/NIPS_TS_GECCO/NIPS_TS_Water_test_label.npy
 D bsc-thesis-ref-codebases/MtsCID/data/NIPS_TS_GECCO/NIPS_TS_Water_train.npy
 D bsc-thesis-ref-codebases/MtsCID/data_factory/data_loader.py
 D bsc-thesis-ref-codebases/MtsCID/figures/building_blocks.png
 D bsc-thesis-ref-codebases/MtsCID/figures/framework_overview.png
 D bsc-thesis-ref-codebases/MtsCID/main.py
 D bsc-thesis-ref-codebases/MtsCID/metrics/__init__.py
 D bsc-thesis-ref-codebases/MtsCID/metrics/_anomaly_detection.py
 D bsc-thesis-ref-codebases/MtsCID/metrics/_tsad_adjustment.py
 D bsc-thesis-ref-codebases/MtsCID/metrics/affiliation/_affiliation_zone.py
 D bsc-thesis-ref-codebases/MtsCID/metrics/affiliation/_integral_interval.py
 D bsc-thesis-ref-codebases/MtsCID/metrics/affiliation/_single_ground_truth_event.py
 D bsc-thesis-ref-codebases/MtsCID/metrics/affiliation/generics.py
 D bsc-thesis-ref-codebases/MtsCID/metrics/affiliation/metrics.py
 D bsc-thesis-ref-codebases/MtsCID/metrics/metrics.py
 D bsc-thesis-ref-codebases/MtsCID/metrics/vus/analysis/robustness_eval.py
 D bsc-thesis-ref-codebases/MtsCID/metrics/vus/metrics.py
 D bsc-thesis-ref-codebases/MtsCID/metrics/vus/models/distance.py
 D bsc-thesis-ref-codebases/MtsCID/metrics/vus/utils/metrics.py
 D bsc-thesis-ref-codebases/MtsCID/metrics/vus/utils/slidingWindows.py
 D bsc-thesis-ref-codebases/MtsCID/model/Conv_Blocks.py
 D bsc-thesis-ref-codebases/MtsCID/model/RevIN.py
 D bsc-thesis-ref-codebases/MtsCID/model/Transformer.py
 D bsc-thesis-ref-codebases/MtsCID/model/attn_layer.py
 D bsc-thesis-ref-codebases/MtsCID/model/embedding.py
 D bsc-thesis-ref-codebases/MtsCID/model/loss_functions.py
 D bsc-thesis-ref-codebases/MtsCID/model/lr.py
 D bsc-thesis-ref-codebases/MtsCID/model/multi_attention_blocks.py
 D bsc-thesis-ref-codebases/MtsCID/requirements.txt
 D bsc-thesis-ref-codebases/MtsCID/smd_experiment.txt
 D bsc-thesis-ref-codebases/MtsCID/solver.py
 D bsc-thesis-ref-codebases/MtsCID/test.sh
 D bsc-thesis-ref-codebases/MtsCID/utils/dataplot.py
 D bsc-thesis-ref-codebases/MtsCID/utils/utils.py
 D bsc-thesis-ref-codebases/mulan-main/CONFIG.md
 D bsc-thesis-ref-codebases/mulan-main/README.md
 D bsc-thesis-ref-codebases/mulan-main/algorithms.py
 D bsc-thesis-ref-codebases/mulan-main/augmentations.py
 D bsc-thesis-ref-codebases/mulan-main/config.py
 D bsc-thesis-ref-codebases/mulan-main/detection/configs/Base-RCNN-FPN.yaml
 D bsc-thesis-ref-codebases/mulan-main/detection/configs/coco_R50_MaskRCNN-FPN_ssl_L.yaml
 D bsc-thesis-ref-codebases/mulan-main/detection/convert-pretrain-to-detectron2.py
 D bsc-thesis-ref-codebases/mulan-main/detection/train_coco.py
 D bsc-thesis-ref-codebases/mulan-main/distributed.py
 D bsc-thesis-ref-codebases/mulan-main/eval_knn.py
 D bsc-thesis-ref-codebases/mulan-main/eval_linear.py
 D bsc-thesis-ref-codebases/mulan-main/figs/mulan_overview.jpg
 D bsc-thesis-ref-codebases/mulan-main/main.py
 D bsc-thesis-ref-codebases/mulan-main/models.py
 D bsc-thesis-ref-codebases/mulan-main/optim.py
 D bsc-thesis-ref-codebases/mulan-main/resnet.py
 D bsc-thesis-ref-codebases/mulan-main/resnet_jaxlike.py
 D bsc-thesis-ref-codebases/mulan-main/utils.py
 D bsc-thesis-ref-codebases/mulan-main/vision_transformer.py
 D bsc-thesis-ref-codebases/sto-transformer-main/.gitignore
 D bsc-thesis-ref-codebases/sto-transformer-main/AGENTS.md
 D bsc-thesis-ref-codebases/sto-transformer-main/CODE_OF_CONDUCT.md
 D bsc-thesis-ref-codebases/sto-transformer-main/CONTRIBUTING.md
 D bsc-thesis-ref-codebases/sto-transformer-main/README.md
 D bsc-thesis-ref-codebases/sto-transformer-main/code/CoLA/README.md
 D bsc-thesis-ref-codebases/sto-transformer-main/code/CoLA/Sto_Transformer.py
 D bsc-thesis-ref-codebases/sto-transformer-main/code/CoLA/train.py
 D bsc-thesis-ref-codebases/sto-transformer-main/code/IMDB/Config.py
 D bsc-thesis-ref-codebases/sto-transformer-main/code/IMDB/IMDBDataset.py
 D bsc-thesis-ref-codebases/sto-transformer-main/code/IMDB/Model_bert.py
 D bsc-thesis-ref-codebases/sto-transformer-main/code/IMDB/Model_sto_transformers.py
 D bsc-thesis-ref-codebases/sto-transformer-main/code/IMDB/Model_transformers.py
 D bsc-thesis-ref-codebases/sto-transformer-main/code/IMDB/README.md
 D bsc-thesis-ref-codebases/sto-transformer-main/code/IMDB/Run.py
 D bsc-thesis-ref-codebases/sto-transformer-main/code/IMDB/Run_evaluation.py
 D bsc-thesis-ref-codebases/sto-transformer-main/code/IMDB/Utils.py
 D bsc-thesis-ref-codebases/sto-transformer-main/code/IMDB/__init__.py
 D bsc-thesis-ref-codebases/sto-transformer-main/code/common/Constants.py
 D bsc-thesis-ref-codebases/sto-transformer-main/code/common/CumulativeTrainer.py
 D bsc-thesis-ref-codebases/sto-transformer-main/code/common/EMA.py
 D bsc-thesis-ref-codebases/sto-transformer-main/code/common/Utils.py
 D bsc-thesis-ref-codebases/sto-transformer-main/code/common/__init__.py
 D bsc-thesis-ref-codebases/sto-transformer-main/code/common/transformer.py
 D bsc-thesis-ref-codebases/sto-transformer-main/code/pre.sh
 D bsc-thesis-ref-codebases/sto-transformer-main/code/run.sh
 D bsc-thesis-ref-codebases/sto-transformer-main/code/setup.sh
 D documents/benchmark-flow-6-july-2026.md
 D documents/inventories/index.md
 D documents/inventories/ssh-command-notes.md
 D documents/learning-docs/how-to-form-a-research-question.md
 D documents/logs/2026-11-01/detail/baseline-restoration-backup.txt
 D documents/logs/2026-11-01/detail/removals.csv
 D documents/logs/2026-11-01/detail/verification.json
 D documents/logs/2026-11-01/plan/baseline-restoration.md
 D documents/logs/2026-11-01/plan/simplification.md
 D documents/logs/2026-11-03/detail/phase2-reexecution/phase2-rerun.md
 M documents/logs/2026-11-03/detail/phase2-reexecution/proposed-next-version-tree.md
 D documents/logs/2026-11-03/plan/codex-cleandocs-hook.md
 D documents/logs/2026-11-03/plan/cross-codebase-online-tta-terminology-audit.md
 D documents/logs/2026-11-03/plan/online-test-time-adaptation-wording-edits.md
 D documents/logs/2026-11-03/research/evaluation-metrics-test__evaluation_protocol_audit.md
 D documents/logs/2026-11-03/research/evaluation-reuse-test__evaluation_protocol_audit.md
 D documents/logs/2026-11-03/research/research-protocol-meaning.md
 M documents/logs/2026-11-09/plan/plan-config-signatures-and-fields.md
 D documents/logs/2026-11-09/plan/plan-explore-all-local-datasets.md
 D documents/logs/2026-11-09/research/research-local-datasets.md
 M documents/logs/2026-11-16/plan/decisions-tsad-preprocess.md
 M documents/logs/2026-11-16/plan/plan-tsad-preprocess.md
 D documents/notes/benchmark_metrics_and_data_directories.md
 D documents/notes/offline_report_collection_plan.md
 D documents/repository_architecture_analysis.md
 M documents/specs/definitions.md
 M documents/specs/desired-cli-20261109.md
 M documents/specs/protocol-validation-rules.md
 D documents/templates/feature-memo.md
 D notebooks/eda/iccad_eda.ipynb
 D notebooks/eda/smd_random_window_synthetic_anomaly_12_panel.ipynb
 D notebooks/eda/smd_random_window_synthetic_anomaly_12_panel_w100.ipynb
 D notebooks/eda/smd_three_entity_synthetic_profile_comparison.ipynb
 D notebooks/learning/smd_colab_window_preprocessing_template_exercise.ipynb
 D notebooks/templates/smd_colab_window_preprocessing_template.ipynb
 D notebooks/templates/time_series_loading_template.ipynb
 D reporting.tar.zst
 M reporting/README.md
 D reporting/online_phase_tables/online_metrics_99/baseline__candi__main__machine_1_6__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__candi__main__machine_1_6__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__candi__main__machine_1_6__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__candi__main__machine_3_4__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__candi__main__machine_3_4__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__candi__main__machine_3_4__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__candi__main__machine_3_9__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__candi__main__machine_3_9__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__candi__main__machine_3_9__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__iforest__main__machine_1_6__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__iforest__main__machine_1_6__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__iforest__main__machine_1_6__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__iforest__main__machine_3_4__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__iforest__main__machine_3_4__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__iforest__main__machine_3_4__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__iforest__main__machine_3_9__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__iforest__main__machine_3_9__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__iforest__main__machine_3_9__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__kmeans_ad__main__machine_1_6__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__kmeans_ad__main__machine_1_6__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__kmeans_ad__main__machine_1_6__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__kmeans_ad__main__machine_3_4__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__kmeans_ad__main__machine_3_4__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__kmeans_ad__main__machine_3_4__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__kmeans_ad__main__machine_3_9__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__kmeans_ad__main__machine_3_9__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__kmeans_ad__main__machine_3_9__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__m2n2__main__machine_1_6__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__m2n2__main__machine_1_6__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__m2n2__main__machine_1_6__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__m2n2__main__machine_3_4__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__m2n2__main__machine_3_4__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__m2n2__main__machine_3_4__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__m2n2__main__machine_3_9__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__m2n2__main__machine_3_9__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__m2n2__main__machine_3_9__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__stumpy__main__machine_1_6__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__stumpy__main__machine_1_6__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__stumpy__main__machine_1_6__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__stumpy__main__machine_3_4__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__stumpy__main__machine_3_4__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__stumpy__main__machine_3_4__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__stumpy__main__machine_3_9__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__stumpy__main__machine_3_9__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/baseline__stumpy__main__machine_3_9__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/online_metrics_99_manifest.csv
 D reporting/online_phase_tables/online_metrics_99/online_metrics_99_manifest.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A0__machine_1_6__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A0__machine_1_6__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A0__machine_1_6__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A0__machine_3_4__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A0__machine_3_4__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A0__machine_3_4__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A0__machine_3_9__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A0__machine_3_9__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A0__machine_3_9__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A1__machine_1_6__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A1__machine_1_6__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A1__machine_1_6__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A1__machine_3_4__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A1__machine_3_4__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A1__machine_3_4__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A1__machine_3_9__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A1__machine_3_9__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A1__machine_3_9__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A2__machine_1_6__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A2__machine_1_6__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A2__machine_1_6__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A2__machine_3_4__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A2__machine_3_4__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A2__machine_3_4__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A2__machine_3_9__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A2__machine_3_9__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O0_A2__machine_3_9__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A0__machine_1_6__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A0__machine_1_6__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A0__machine_1_6__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A0__machine_3_4__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A0__machine_3_4__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A0__machine_3_4__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A0__machine_3_9__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A0__machine_3_9__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A0__machine_3_9__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A1__machine_1_6__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A1__machine_1_6__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A1__machine_1_6__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A1__machine_3_4__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A1__machine_3_4__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A1__machine_3_4__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A1__machine_3_9__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A1__machine_3_9__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A1__machine_3_9__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A2__machine_1_6__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A2__machine_1_6__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A2__machine_1_6__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A2__machine_3_4__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A2__machine_3_4__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A2__machine_3_4__seed8__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A2__machine_3_9__seed36__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A2__machine_3_9__seed6__online_metrics.json
 D reporting/online_phase_tables/online_metrics_99/thesis__O1_A2__machine_3_9__seed8__online_metrics.json
 D scripts/plot_machine_3_4_anomaly_scores.py
 D scripts/plot_machine_3_9_configurable.py
 M scripts/preprocess.py
?? documents/logs/2026-11-16/detail/
?? documents/logs/2026-11-16/plan/decisions-runtime-replacement.md
?? documents/logs/2026-11-16/plan/plan-cleanup-next-version.md
?? documents/logs/2026-11-16/plan/plan-implement-runtime-signatures.md
?? documents/logs/2026-11-16/research/research-nasa-p2-annotations.md
?? documents/logs/2026-11-16/research/research-swat-source-timeline.md
?? documents/logs/2026-11-16/structure/
?? documents/specs/workflow-memo.md
```

## Instruction review

1. Read `AGENTS.md`, both preference files, and `python_semantics.md`.
2. The newer confirmed replacement decisions control the target layout and prepared-data reuse. Older path limits and raw loaders do not control this target.
3. The thesis specification index is absent, as stated in the plan. Use the named source documents.
4. The execution plan requires a stop at an open scientific decision. No later phase can start before Phase 1 passes.

## Inventory

1. `rg --files src scripts tests configs` returned 1,184 paths. The [path map](runtime-replacement-path-map.csv) records every path, its proposed owner, and its removal check.
2. A static search found 747 import and command references in source, scripts, tests, `README.md`, and `MAIN_FOLDERS.md`. Re-run `rg -n '^(from|import) (src|scripts)|scripts/(cli|benchmarks|launchers)/' src scripts tests README.md MAIN_FOLDERS.md` before retirement. A text match is not proof of execution.
3. Current configuration loading starts in `src/core/config.py`. Runtime data loading and window creation use `src/data/loaders.py` and `src/data/window.py`. The plan replaces their responsibilities through the new owners.
4. `src/data/` and `src/metrics/` currently exist as packages. The proposed `src/data.py` and `src/metrics.py` have the same import names. Replacement must resolve that conflict before importing new code. Do not assume that adding a file switches imports.
5. Some optional metadata paths were absent in the initial search. The search returned exit code 2 for those missing paths. This is inventory evidence, not a passed runtime check.

### Command ownership

1. `pyproject.toml`: absent.
1. `setup.py`: absent.
1. `setup.cfg`: absent.
1. `.venv/bin/tsad`: absent.
1. `tsad` on the current PATH: `None`.

## Repeatable size counts

Use regular text files returned by `rg --files src scripts tests configs`. Count physical lines with `len(text.splitlines())`, including comments and blank lines. Source and tests use `.py`; scripts use `.py` and `.sh`; YAML uses `.yaml` and `.yml` under `configs/`. Count Python function spans from `def` through `end_lineno`, including nested definitions and comments. Count file lines with the same rule.

| Group | Files | Lines |
| --- | ---: | ---: |
| Source | 126 | 26131 |
| Scripts | 118 | 22161 |
| Tests | 137 | 20791 |
| YAML | 801 | 35565 |

## Protected prepared manifests

On 2026-11-20, the prepared-data root changed to `data/preprocessed/`. The table records the paths at the time of the original checks. Use the same dataset and version names below the new root.

Hash algorithm: SHA-256 over file bytes. Protect every listed version. Do not change prepared values, scaler states, or manifests. These hashes prove file identity only; they do not prove all output files are valid.

| Manifest | SHA-256 |
| --- | --- |
| `outputs/preprocessed/iccad/3ceffc9b05e018ae91bdeb4424a204f5ff1cd7a63d282d02dc771bfea1a21ef1/manifest.json` | `3264b201e4795463ed115c9ce747f7fd1320090f00290d547b36b22729900435` |
| `outputs/preprocessed/iops/1a01993def9c82c12f4a4b9a77182b6a448e74cd1accaef59864fc23f467dc52/manifest.json` | `e3482303c9835bbfbec1bb6e01a2e978d200849a284d47c6ce4f748ea926eacc` |
| `outputs/preprocessed/nasa/fa7579b37219cb8136dcd0991b29058f21bb6a8cce0efc6d0a024d1945e6c2af/manifest.json` | `3ff7c6d4a29ca4383b0ce75303f6744203ad28b901bfdae0791856ff58c47cc6` |
| `outputs/preprocessed/smd/63dbcbb69a34a7ee3c0d82669f28461421c023b3427c23fbb7d9389b577d95f8/manifest.json` | `39b4f720b938dce53c9c5d4348ac0b13f31e8dbf9b57a4dc59eb9dfc50244ce5` |
| `outputs/preprocessed/swat/67e1e1753e66f44858d0b31ec38ccff60bb86e805b50a889d576fb02ca39e391/manifest.json` | `ff49451d41a71beb4a083dafe1e19f27b7ce83d91a05d1ace3e2143b7446f6c2` |
| `outputs/preprocessed/swat/83957f47be5f7b142ae0bec77a64aa43c306e6e258ef7bfa176f2e46e0daee06/manifest.json` | `c84272000d30061384e224377a342f01a4fa45dd079b8648369c7c927173f002` |

## Migration path map

[Exact path map](runtime-replacement-path-map.csv) contains one row for each of the 1184 inventoried paths. Each row names proposed owners and the required retirement check. Every old path remains protected from removal until the check passes and callers are reviewed. These mappings describe responsibility; they do not assert scientific equivalence.

| Old area | Proposed owners |
| --- | --- |
| `src/core/` | Configuration, artifacts, CLI, and protocols |
| `src/data/` | Existing preprocessing script, prepared loader, and training augmentation |
| `src/models/` | MUSTARD, RedLamp, and shared encoder |
| `src/baselines/`, `src/adapters/` | Independent baseline entrypoints and capability checks |
| `src/engine/` | Phase functions, scores, metrics, artifacts, and method updates |
| `src/protocols/` | Four validation contexts and score calculations |
| `scripts/` | CLI and workflow owners; keep preprocessing and document hook |
| `configs/` | Four profile folders, experiments, standalones, shared metrics |
| `tests/` | Owning replacement behavior tests |

Stage 1.1: complete. No runtime, source data, or prepared output was changed.

## First open step

Step: Phase 1, Stage 1.2, step 1.

Action: specify exact profile fields, defaults, references, and overrides.

1. The configuration plan defines the data-profile envelope but leaves method child fields open. Its evaluation settings also need a score policy before their applicability is fixed.
2. The signature proposal, sections 7, 12, and 21, explicitly leaves online alignment and commit times open. The confirmed replacement record, section 1, item 6, requires review of the same choice.
3. Existing `src/data/window.py` supports `drop` and `end_align`. Existing `src/models/neural_blocks.py` preserves time length and uses both sides of a position within its input window. These are reference behaviors; they do not select the next-version policy.
4. A full-block output may use all values in that block. It can be causal at block completion only when each result records that later commit time. It must not be reported as a prediction available earlier in the block.
5. An endpoint-only output from a trailing window gives another valid causal policy. It changes coverage and can change update frequency. That difference affects the scientific comparison.

The two options are recorded as an open question in [the decision file](../plan/decisions-runtime-replacement.md#execution-question-2026-11-17). The proposed default is full blocks with scores committed at block end. It requires acceptance of the delay. No answer is assumed.

### Current boundary

1. Stage 1.1 steps 1–7 are complete.
2. Stage 1.2 step 1 is in progress. Its checklist box remains open.
3. No subsequent step is complete. Do not advance to Stage 1.2 step 2 or Phase 2 before the required decision.
4. No code, configuration, prepared output, or historical result was removed or replaced.
5. Resume at Stage 1.2 step 1 after the answer. More scientific questions may arise during the required method review. The current question does not approve all remaining contract details.

## User reply: point-wise EWMA

1. The user requests scores for all points in each window, with EWMA applied per point. A newly observed last point keeps its raw score.
2. Inspection of `src/engine/online_tta/point_ewma.py` confirms that its existing recurrence matches this behavior. This is a source check, not a runtime test.
3. The earlier binary question omitted this valid overlapping-window behavior. Its recommendation is superseded by the user reply.
4. The remaining question is which estimate becomes the fixed reporting score: the first estimate, or a later estimate with a declared delay. The reply does not specify that choice.
5. Stage 1.2, step 1 remains open. No later checklist step is complete.

## User decision: last estimate and default weights

1. The user selected each point's last EWMA estimate for reports.
2. Use configurable defaults of `0.25` for the previous estimate and `0.75` for the new score. Explicit historical profiles retain their stated pair.
3. Record availability at the final contributing window endpoint. Preserve earlier window events and adaptation decisions. Finalized scores cannot change.
4. Stage 1.2 step 1 continues. The first score question is closed; the complete profile schema is not yet complete.

## Profile review: method window length

1. Read the MUSTARD mathematics, the method-review summaries, current SMD data/model/evaluation profiles, and the existing MOMENT adapter.
2. A single data-profile window length does not settle MOMENT's treatment. The method review describes 512-observation input; the current SMD profile uses 20.
3. Asked the user to choose method-specific length in one data profile, a shared 512-point length, or short input with MOMENT padding. This is a scientific input choice, not a package naming choice.
4. The proposed default is a method-specific MOMENT length in the same data profile. It is not yet approved.
5. Stage 1.2 step 1 remains open. No later step or implementation phase has started.

## User decision: MOMENT window length

1. Option A is confirmed. One data profile supplies the dataset default and `method_window_sizes: {moment: 512}`.
2. Method-specific lengths do not create another data profile or change prepared data.
3. Record effective window length in identity checks. No cell override is allowed.
4. Stage 1.2 step 1 continues.

## Source review: first method-contract conflict

1. Read the seven baseline mathematics notes and the current traditional-baseline input and scoring code.
2. Checked CANDI source selection, SANA, loss, and configuration. Checked RedLamp correction and loss code against the public author source.
3. The CANDI note adds an output correction; the author code subtracts it. The confirmed requirements ask for closeness to both sources but do not state their priority when they differ.
4. Asked for one source-priority rule to resolve this and later conflicts without unnecessary repeated questions. No answer is assumed.
5. Stage 1.2 step 1 remains in progress. No new execution checkbox is complete. No runtime source or prepared output changed.

## Scientific-intent rule and CANDI check

1. The user requests scientific intent as the basis for resolving paper/code differences. The earlier source-priority options are superseded.
2. Read CANDI's paper method pages and checked equations 12–18 against the official HTML and author code. Used the PDF skill for the equation review.
3. The output sign is related by reversing the unconstrained gate. Scalar and float64 PyTorch checks passed for the outputs and mapped gradients at three gate values, with absolute tolerance `1e-12`. This is an isolated mathematical check, not method parity.
4. Use the author subtract form with the signed gate. No sign configuration or second algorithm is required.
5. Read one validation label column per prepared dataset. Recorded counts in the decision file. PyArrow emitted CPU-cache and instruction-detection warnings because sandbox system calls were denied; all five reads and counts completed successfully.
6. Unknown-label calibration eligibility remains unresolved. Asked the user to allow unknown labels under a recorded assumption or require known-normal labels.
7. Stage 1.2 step 1 remains open. No later execution step has started.

## User decision: calibration eligibility

1. Option A is confirmed. Exclude whole windows that contain known anomalies. Permit unknown labels under a recorded normal-data assumption; keep labels unchanged.
2. Preserve source indices and save calibration-window counts. Stop if no eligible windows remain.
3. This closes the current calibration-label question. Stage 1.2 step 1 continues.

## Profile contract progress

1. Wrote [the working profile contract](runtime-profile-contract.md). It defines settled data and evaluation fields and default provenance.
2. Kept method and reporting child fields open. No unrestricted dictionary is accepted as a substitute for a schema.
3. The calibration-label question is closed. The exact method, stride, tail, and reporting fields still prevent completion of Stage 1.2 step 1.

## Current stop point: method reductions

1. The calibration-label decision is confirmed and recorded. Removed a redundant unknown-label switch from the working schema; one policy expresses the selected behavior.
2. Specified four reporting-profile fields: `phase`, `splits`, `metrics`, and optional `precision`. Kept the agreed fixed table arrangement and both CSV versions.
3. Asked two consequential project-specific questions: MOMENT channel reduction and CANDI CNN latent reduction. Neither follows uniquely from the reviewed original setup.
4. Stage 1.2 step 1 remains open until these choices and the remaining exact method fields are specified. No later stage has started.

## User decisions: reductions and contamination

1. MOMENT channel reduction: mean. CANDI CNN latent reduction: mean over time, followed by the reviewed normalization.
2. Recorded the user's caution that unknown-label training data need not be normal. Applied the same limitation to calibration without changing eligibility or source labels.
3. Recorded the limits of quantile-based false-positive claims. No new filtering, preprocessing, or method change follows from this caution.
4. Both reduction questions are closed. Continue Stage 1.2 step 1; the complete method schemas remain to be specified.

## Stage 1.2 step 1: field definitions complete

1. Specified data, method, evaluation, reporting, and synthetic fields in [the profile contract](runtime-profile-contract.md), including closed child keys and field types.
2. Removed duplicated M2N2 threshold-source configuration. The evaluation profile owns that source. Kept method trend state separate from the prepared input scaler.
3. Added the AdamW field shape because the inspected M2N2 reference trainer uses AdamW. This is schema support, not a claim of implementation or selected optimizer parity.
4. Listed exact profile references. The new method-profile `phases` list and additional MOMENT/evaluation files need the later signature and tree update in Stage 1.2 step 13.
5. Required scientific numeric values have no parser fallback; supplied YAML must state them. The later Phase 1 steps still bind those values and rules. The Phase 1 implementation gate is not passed.
6. Continue at Stage 1.2 step 2: capability and combination definitions. No Phase 2 code has started.

## Stage 1.2 step 2: target capabilities complete

1. Recorded all eight methods, both MOMENT variants, MUSTARD O/A variants, supported phases, and offline dependencies in the decision file.
2. The target covers all five prepared datasets under per-entity shape, window, calibration, and dependency checks. This is not observed runtime support.
3. Kept frozen online scoring separate from adaptation. Zero-shot MOMENT does not gain a dummy fit stage. CANDI requires a valid covariance input and cannot turn an empty reference into a match.
4. Kept combined MUSTARD reporting identity explicit, and retained native endpoint coverage for window-only baselines.
5. Continue at Stage 1.2 step 3: common CNN and method heads. No Phase 2 implementation has started.

## Stage 1.2 step 3: architecture bindings complete

1. Bound the common CNN settings and all method head shapes in the decision file. The changes are design bindings, not passed model tests.
2. Preserved MUSTARD's flattened classification context. Used the confirmed CANDI mean and a shared CANDI/M2N2 decoder shape. Recorded RedLamp's pooling change and corrected cross-entropy intention explicitly.
3. Added the missing MUSTARD `stochastic_inference` input to the field contract. Renamed its generic head-width field to the explicit `classifier_hidden_size`.
4. Selected an explicit proposed MOMENT Base revision from its published immutable configuration. The web tool could not read Hugging Face API metadata; weight-file availability and hashes are not verified. No model was downloaded.
5. Read SWaT manifests. Their prepared-file hash mappings and profile settings match. The source-preparation decision file explicitly identifies checked version `67e1e1753e66f44858d0b31ec38ccff60bb86e805b50a889d576fb02ca39e391`; use that as the proposed runtime default, with both versions retained. The other version matches the current script hash. Do not select by hash ordering.
6. Continue at Stage 1.2 step 4: window and stream policies. No replacement source or YAML files are implemented.

## Stage 1.2 step 4: gap evidence and pending decision

1. Read only timestamp columns for all three splits of ICCAD Console and the checked SWaT version. No prepared values or manifests were changed.
2. ICCAD has 2, 10, and 1 larger-than-cadence jumps in train, validation, and test. SWaT has one such test jump of 82 seconds. No negative or duplicate timestamp differences occurred in these files.
3. Asked whether windows should stop at known gaps or use consecutive saved rows across them. Recommend stopping at known gaps. Do not infer regular elapsed time for timestamp-free data.
4. Stage 1.2 step 4 remains unchecked. Stage 1.2 steps 1–3 and Stage 1.1 are complete. No replacement source or runtime YAML has been implemented.
5. The pinned MOMENT Base configuration was subsequently readable through its Hugging Face blob page. It confirms length 512, patch size 8, and hidden width 768 at the selected revision. The API and raw routes were unavailable. Model weights remain unverified and were not downloaded.

## Confirmed gap choice after pause

1. The user asked to pause, requested the gap question again, then confirmed option A and asked to record the real-world challenge.
2. Recorded breaking windows at known time gaps. Kept timestamp-free continuity unverified and left prepared data unchanged.
3. Recorded reduced usable windows and score coverage, unknown causes of the observed jumps, and the distinction from normal-data assumptions.
4. This update records the requested decision only. Execution resumed at the user request on 2026-11-17. Stage 1.2 step 4 remains unchecked because its other requirements are not complete.

## Stage 1.2 step 4: complete

1. The user requested a recap and continued implementation. Resumed from the paused step without skipping another item.
2. Bound lengths, per-method strides, tails, gaps, stream coordinates, offline alignment, and online availability/finality in the decision file.
3. Confirmed through `online_engine_run.py` that the old absolute interval selects the complete test sequence, not train/test concatenation. Kept source-row mapping separate.
4. Added `expected_interval_seconds` for known timestamp cadence and a cross-field restriction that online tails use `drop`. No new source-preparation path is added.
5. Changed native traditional evaluation to stride-1 endpoint scoring, with coverage recorded. This follows the target's prohibition on broadcasting scores and is an explicit departure from old non-overlap pointification.
6. Continue at Stage 1.2 step 5: metric and uncertainty contracts. No replacement runtime code has started.

## Stage 1.2 step 5: review in progress

1. Reviewed the current metric formulas, existing tests, raw Monte Carlo variance reductions, and summary-statistic meanings. Checked the public author VUS and affiliation interfaces.
2. Recorded precise pointwise eligibility, strict thresholds, missing-value rules, uncertainty axes, and prospective numerical tolerances in the decision record.
3. Confirmed with a small scalar example that concatenation across a gap can merge distinct anomaly events. Asked for a segment-summary rule before implementation.
4. Package inspection found NumPy 2.4.4, PyTorch 2.11.0, pandas 2.3.3, PyArrow 23.0.1, PyYAML 6.0.3, scikit-learn 1.7.2, SciPy 1.16.3, STUMPY 1.14.1, pytest 9.0.2, and W&B 0.25.1. VUS, affiliation, MOMENT, Kaggle, and Ruff were not installed as distributions in the checked environment. Nothing was installed or removed.
5. Dependency absence is a future readiness requirement, not a passed mock check. Version compatibility and reference outputs still need verification in their planned implementation stages.
6. Stage 1.2 step 5 remains open. No later step or replacement source implementation has started.

## Stage 1.2 step 5: complete

1. Recorded the user's segment-summary choice, exact weighting, distinct scalar names, omitted-segment counts, and coverage rules.
2. Recorded the required structured return from `compute_metrics` and segment identity fields. Update the public signatures in step 13 before implementation.
3. No metric code was run against an author implementation. Numerical tolerances and behavior contracts are specified; tests remain pending.
4. Continue at Stage 1.2 step 6: smoke limits and isolation.

## Stage 1.2 step 6: complete

1. Bound numeric smoke work, MOMENT-specific caps, mode identity, separate roots, parent isolation, report exclusion, and fake-provider verification.
2. Added the required `max_epochs_per_stage` limit to the design. No profiles or prepared files were modified.
3. Continue at Stage 1.2 step 7: run identities, status, and files.

## Stage 1.2 step 7: complete

1. Bound schema versions, UTC times, UUIDv7 generation/claims, collision handling, status transitions, state/result paths, file roles, and final tie order.
2. Checked Python 3.12.12, absence of standard UUIDv7, and absence of a registered `tsad` entrypoint in the thesis environment. Reviewed RFC 9562's v7 layout.
3. Selected separate whole-run state and split results, retaining the agreed result grouping. No registry or output artifact was created.
4. Continue at Stage 1.2 step 8: parent compatibility and artifact identity.

## Stage 1.2 step 8: complete

1. Specified parent filters, scientific identity fields, calibration compatibility, selected-artifact validation, and recorded-parent recovery.
2. Kept zero-shot and linear-probe dependencies distinct. Kept profile content identity separate from its filename.
3. Continue at Stage 1.2 step 9: exact recovery and the missing method continuation inputs.

## Stage 1.2 step 9: complete

1. Bound offline epoch/stage and online window recovery, a default eight-window checkpoint interval, complete state, writer publication, and same-parent/config recovery.
2. Identified missing `fit` continuation/callback inputs and immutable best-role generations. Record and apply the interface changes in step 13.
3. No recovery tests have run. These are implementation contracts for later interruption tests.
4. Continue at Stage 1.2 step 10: logging and backup behavior.

Stage 1.2 step 10 complete: optional providers, durable local authority, private existing Kaggle target, archive-free hard-link staging, and bounded failure handling are specified. API sources were checked; no network upload was made.

Stage 1.2 step 11 complete: inspected both package metadata paths and current console entrypoints. Defined isolated thesis installation and namespace. Recorded two module path changes to avoid retained legacy-package collisions. No package was installed.

Stage 1.2 step 12 complete: reviewed all received scientific replies and their bindings. No unanswered reply is treated as approval. Full-contract approval remains a separate plan step.

## Stage 1.2 step 13: complete

1. Updated continuation inputs, segment/metric records, package ownership, and module paths in the signature proposal. Added explicit review values to the profile contract. Native estimator fitting does not use an invented neural stage.
2. Aligned both tree proposals, the configuration plan, implementation ownership, and path map. No runtime code, raw data, prepared output, or package installation changed.
3. Checks passed: 22 Python examples parse; five updated contract links resolve; both trees contain the added profiles and renamed modules; all six protected manifest hashes match the starting record. Git whitespace checks pass. These checks are documentation evidence, not runtime tests.
4. Step 14 is pending. The implementation plan explicitly requires approval for the completed contract before Phase 2.

## Walkthrough amendments: 2026-11-17

1. Phase: inspect. Checked the score persistence, runtime record, recovery, and epoch-default statements. **Tools:** file reader and text search.
2. Phase: revise. Recorded both user decisions; changed saved scores to one nullable ordered column; kept point indices temporary; withdrew unequal epoch totals. **Tool:** apply_patch.
3. Phase: verify. Signature examples parse; whitespace checks pass; all six protected manifest hashes match. **Tools:** Python, SHA-256, and Git.
4. One clarification is pending: the common offline epoch budget. Proposed 30 total epochs with MUSTARD 25 + 5; another offered option is 10 total. The total-budget interpretation remains a proposal until the reply. No runtime tests or implementation changes were made.

## Epoch clarification closed: 2026-11-17

1. Phase: inspect. Read all active epoch statements in the contract, signatures, decisions, and checklist. **Tool:** file reader.
2. Phase: update. Recorded the user choice of 30 total offline epochs. Kept stage counts in method profiles and added equal-total validation to configuration tests. No new experiment field is required. **Tool:** apply_patch.
3. Phase: verify. Signature syntax, active epoch statements, all six protected manifest hashes, and Git whitespace checks passed. **Tools:** Python, SHA-256, and Git.
4. This entry closes the pending question in the earlier walkthrough ledger. Smoke limits, online adaptation steps, and methods without training epochs retain their separate meanings. Full-contract approval remains pending.

## Stage 1.2 step 14: complete

1. On 2026-11-18 the user directed execution of the full checklist with the updated runtime profile contract. This authorizes Phase 2.
2. The accepted updates include compact score storage and 30 total offline epochs per fitted neural method. No replacement runtime was present at the approval gate.

## Stage 2.1 steps 1–3: configuration

1. Added configuration tests. Confirmed that collection stops because `src.config` does not exist yet. This is the expected result before the resolver implementation.
2. Wrote the data, method, reporting, and confirmed evaluation profiles. The new default uses EWMA weights 0.25 for the previous estimate and 0.75 for the new estimate. Retained SMD profiles keep weights 0.1 and 0.9.
3. SMD `q995` is used by 90 old experiment definitions for CANDI, IForest, KMeans, M2N2, and STUMPY. The runner calls each baseline's own calibration and score methods without a shared point transform. This maps to `method_native` with `identity`. The legacy anomaly-ratio field has no runtime consumer; the active quantile remains 0.995. This mapping does not imply score parity with the old runner.
4. The shared native evaluation profiles declare `train` for M2N2's separate adaptation threshold. Only M2N2 uses this setting. For other methods the resolved setting is null.
5. SMD `q95` has no recorded caller that establishes a score space or point transform. The old YAML sets quantiles and EWMA weights, but omits both score fields. Step 3 remains open until these fields are settled. No later step has started.
6. All 30 new YAML files passed parsing and whitespace checks. The referenced prepared manifests exist. Static checks confirm 30 total epochs in each profile with training stages and the expected EWMA weights. `git diff --check` passed. Configuration tests still stop at collection until the resolver exists.

7. The user replaced the ambiguous SMD q95 translation with two MUSTARD scores: raw input MSE and latent MSE. Both use separate q95 thresholds from validation and point EWMA weights 0.25/0.75. The latent vector pair remains open. Step 3 remains open.
8. The user confirmed the latent pair: the normalized latent query and the corresponding reconstruction from the continuous prototype bank before fusion. The reduction over stochastic retrieval samples remains open.

9. The plan, the contract, the two tree proposals, and the signatures now identify the proposed q95 profiles. The stochastic sample reduction and the phase scope are open. Step 3 is still open. The YAML files and later steps remain pending.
10. The user confirmed the mean of per-sample latent MSE values within each window. Only the phase scope remains open for Step 3.
11. The user confirmed point EWMA across overlapping windows for both phases. Each score space gets an offline validation q95 threshold. The matching online run reuses that threshold. Wrote and checked the two YAML files. Step 3 is complete. Runtime checks still require the implementation.

## Stage 2.1 steps 4–5: metric names and examples

1. Wrote and reloaded 90 shared metric definitions with display precision 4.
2. Wrote the SMD raw-MSE offline/online example and the latent-MSE offline standalone. Checked their profile references and phase selections. The example uses `random_seeds`, `reporting_profiles`, and explicit `cells` to carry the existing seed, report, and run-template settings.

## Stage 2.1 steps 6–8: resolution

1. Added the read-only resolver and strict YAML loader. The resolver reads prepared manifest metadata and defers split materialization to Stage 2.2 and shared protocol checks to Stage 2.3.
2. Ran the configuration suite: 12 passed. Reviewed the configuration diff.

## Stage 2.2: prepared data and windows

1. The first data test run stopped at collection because `src.prepared_data` did not exist. The implementation passes all six data tests.
2. Loaded one prepared entity from each of the five datasets to check the file format. No raw sources were opened. Reviewed the data diff.

## Stage 2.3: validation contexts

1. The first protocol test run stopped at collection because the four modules did not exist. The configuration, data, and protocol suites now pass all 24 tests.
2. Connected checks to run resolution, entity loading, and batch assembly. Reviewed the diff and checked whitespace. Scoring and method update call sites will connect in their owning execution stages.

## Stage 3.1: run and checkpoint storage

1. The artifact tests first stopped at import because the module did not exist. Six artifact tests now pass. They check IDs, split paths, exact lookup, selected-parent failure, checkpoints, and the two-slot resume pointer.
2. Reviewed the diff and checked whitespace. Exact interruption parity and writer locking remain execution checks.

## Stage 3.2: shared scoring

1. Five scoring tests pass after reproducing the missing module. The tests cover raw units, mean per-sample latent error, fixed validation q95, overlap EWMA, eligible indices, and score-state isolation.
2. Reviewed the diff. The saved offline threshold is still to be connected and checked through the offline and online workflows.

## Stage 3.3: metrics

1. Installed [VUS](https://github.com/TheDatumOrg/VUS/commit/467bace0c103f5c2bdb4900306d82939bcac8ff9) and [affiliation](https://github.com/ahstat/affiliation-metrics-py/commit/8d8449858096bbade6a6e70848d05c9cc9b846fe) at the pinned revisions in the thesis environment. The first suite run confirmed the missing runtime module.
2. All 15 metric tests passed. Reviewed the diff. Ruling: use pinned author functions for unrestricted VUS and Affiliation F1. Reuse the reviewed local budgeted-VUS computation. Keep its required code through packaging and legacy cleanup to preserve its meaning.

## Stage 4.1: shared encoder

1. Reproduced the missing encoder module, implemented the reviewed time-preserving CNN, and passed two encoder tests. They check shape, gradients, and independent parameters. Reviewed the diff.

## Stage 4.2: training augmentation

1. Reproduced the missing augmentation entrypoint. Two tests passed with one window for each class. The entrypoint reuses the retained family formulas and uses the caller's CPU generator. It returns the next class rotation offset for the caller to retain.
2. Saved [the visual check](augmentation-check.png) and reviewed the diff. Keep the reviewed augmentation source available through packaging until its callers are migrated.

## Stage 5.1 step 1: MUSTARD review

1. The local source reference is thesis revision `034833af231be2d6118861be38b331a584c35d44`, with the project license in `LICENSE`. The [MUSTARD mathematics note](../../../notes/experiment-flows-and-equations.md) defines the compact comparison and cites the original runtime.
2. O0 uses reconstruction and classification. O1 adds the balanced point-score term to the classification branch in Stage A. O2 adds the two-view contrastive term once in Stage A. Source inspection confirms that the original contrastive path is active only in Stage A. The O2 comparison has no verified end-to-end legacy run in the reviewed configurations.
3. Stage B freezes encoder and memories and uses direct routing to the two heads. A0 scores without updates. A1 updates the projector for verified new normal observations. A2 adds the hard-old hinge and token contrastive term. Verification uses frozen-model prototype distances and recurrence from arrived windows. Test labels are available only to metrics.
4. The [profile contract](runtime-profile-contract.md) owns the accepted fields and training budgets. Save method weights, frozen memories and their metadata, optimizer state when applicable, augmentation RNG and class rotation, and the online verification buffer with its counters.
5. **Step 2 is blocked by one scientific choice.** The profile contract says to keep the O1 point-score term when a batch lacks one token group. The formula needs both normal and anomalous points. The legacy code records the term as unavailable and uses the full classification loss on that batch. Choose whether to keep the legacy fallback or retain a fixed classification weight and calculate the score term only across eligible batches. These rules can change gradients.
6. The combined configuration, data, protocol, artifact, scoring, metric, encoder, and augmentation suites passed 54 tests. No later MUSTARD step has started.

Stage 5.1, Step 2 is unblocked. The user selected the reference O1 fallback. The [decision and limitation](../plan/decisions-runtime-replacement.md#mustard-o1-batches-with-a-missing-point-group) give the rule and the measures for future analysis. The tests remain pending.

## Stage 5.1 steps 2–10: MUSTARD

1. The first test run stopped at import because the MUSTARD modules did not exist. The new implementation passes 11 tests. This includes the O1 fallback, a two-stage synthetic fit with replayable checkpoints, A0 source scoring, and an A1 update that changes only the projector.
2. Ruling: use ten normalized K-means iterations for both banks, with farthest-first seeds. The active bootstrap calls K-means, and the full specification names K-means for the continuous bank. The separate flow note describes seed selection alone. It predicts different centers for general inputs.
3. Ruling: retain deterministic sparse top-k in the reviewed cosine mode. Sample the continuous branch for stochastic scoring. Compare the A2 hard-old raw window error with its input-window threshold. Keep the reported point threshold out of that loss as required by the current comparison contract.
4. Reviewed the diff. The installed CLI, exact recovery, real MUSTARD experiment, and offline threshold reuse have separate later tests. Preserve the reviewed online verification and projector helper source until its installed-package dependencies are settled.

## Stage 5.2 step 1: RedLamp review

1. Reviewed the [author repository revision](https://github.com/KoheiObata/RedLamp/commit/326959f0c3646eb600501af2eb60a19af7828812) and the local annotated copy. The local copy has no separate RedLamp license file. Implement the equations in project code and import no author source.
2. Use a mean over time before the classification head. Apply the corrected 12-class cross entropy to logits. Compute the masked reconstruction mean over all cells, as in the author loss. Build the cell mask from per-channel changed positions in the shared augmentation record. Continue to skip singleton training batches.
3. On eligible validation windows, fix the selected anomaly classes and the min and max of the two score components. Average their validation-scaled values with equal weights. Use the existing epsilon floor. Score each window at its endpoint, as in the author index mapping. Ordinary point EWMA does not combine different endpoints. This changes the original smoothing and scaling rules.
4. Preserve model weights, classifier statistics, fitted score calibration, and the training augmentation generator and class rotation for continuation. The online method keeps all learned parameters fixed.

## Stage 5.2 steps 2–6: RedLamp

The first test run stopped because the replacement module was missing. Both tests now pass. They check corrected targets, cell-wise masks, the zero-range guard, endpoint alignment, and calibration that stays fixed when test input changes. Reviewed the diff. A real RedLamp fit and online run are later checks.

## Stage 5.3 step 1: STUMPY channel AB review

1. Use the installed STUMPY 1.14.1 package. The retained source identifies its license as BSD-3-Clause. The immutable release tag is v1.14.1.
2. Keep the complete saved training sequence and its separate continuous segments as the frozen reference. For each query window and channel, use the minimum STUMPY distance to a same-length reference window. With normalization enabled, use STUMPY's normalized distance. Otherwise use the configured p-norm. Keep native constant-window handling and do not use statistics from a complete test split.
3. Fit a channel median and interquartile range on eligible validation windows. Floor each range by the configured epsilon. Take the maximum calibrated channel score per window. Larger values indicate higher anomaly scores. Place the score at the endpoint with no broadcast. Save the reference segments, centers, and ranges. Online scoring makes no updates.

## Stage 5.3 steps 2–6: STUMPY

The test first failed because the module was missing. It now passes with an exact distance, preserved endpoint, restored state, and rejection of a short window. Reviewed the diff.

## Stage 5.4 step 1: KMeansAD review

Use scikit-learn 1.7.2, BSD-3-Clause, with the local thesis KMeansAD source at revision `034833af231be2d6118861be38b331a584c35d44` as the adapter reference. Flatten each selected window in time-major order. When enabled, subtract the row mean and divide by the row sample standard deviation. Use one for a constant or low-variance row. Fit only selected training windows with the configured cluster count, ten seeded initializations by default, and the standard K-means++ initializer. Reject an insufficient number of training windows. Score the Euclidean distance to the nearest center at each endpoint. Fit median and interquartile range on eligible validation scores. Keep cluster centers, scaling state, and the checkpoint identity.

## Stage 5.4 steps 2–6: KMeansAD

The first test failed because the module was missing. Its known-value center, distance, seed, and state checks now pass. Reviewed the diff.

## Stage 5.5 step 1: Isolation Forest review

Use scikit-learn 1.7.2, BSD-3-Clause. The local thesis adapter at revision `034833af231be2d6118861be38b331a584c35d44` uses time-major flattened windows and the same optional sample standardization as KMeansAD. Fit the configured forest to training windows only. Use the negative decision function as the anomaly score. Fit median and interquartile range on eligible validation scores. Keep one score at each endpoint. Save the full fitted estimator and validation parameters, with explicit scikit-learn and serialization format versions.

## Stage 5.5 steps 2–6: Isolation Forest

The first test failed because the module was missing. The final test passed with exact score direction, seeded fit, endpoint positions, and checkpoint restoration. The fitted estimator uses [skops](https://skops.readthedocs.io/en/stable/persistence.html) 0.13.0. The loader rejects unexpected types and incompatible library versions. Reviewed the diff and left the selected versions for the packaging step.

## Stage 5.6 step 1: CANDI review

The author reference is [revision 28c9679](https://github.com/kimanki/CANDI/commit/28c9679e503832f59e351208cde63657fcb51cad) with a modified non-commercial MIT license. Do not import the author checkout. Fit the approved time-mean CNN bottleneck and window decoder on training windows. Use validation reconstruction loss for checkpoint selection. Normalize the pooled validation vectors, fit their covariance and pseudoinverse, and select hard and moderate references under the approved quantile rules. For an arrived window use the recorded window score and its frozen-backbone latent vector. Scores greater than the threshold use hard references. Scores less than the threshold use moderate references. Equality and empty sets do not select a window. Require a squared distance strictly below the chi-square cutoff. Hold hard and moderate candidates in separate queues. Use the local SANA topology with signed input addition and output subtraction. Apply the configured optimizer only to SANA. Save its optimizer state, reference state, and pending queues. Test labels stay outside the method.

## Stage 5.6 steps 2–7: CANDI

The two tests first failed because the module was missing. They now pass for covariance, empty references, threshold equality, frozen parameters, pending state, and a label-free method call. Reviewed the diff.

## Stage 5.7 step 1: M2N2 review

The author reference is [revision 616b227](https://github.com/carrtesy/M2N2/commit/616b2270b6f2eab88ee5caa37c45507d2d041d22). The local author copy has no separate license file. Implement the reviewed equations without importing author code. Use zero-initialized channel trend. Update the trend from each whole arrived window before online scoring. Save the score before adapting learned parameters. Use the frozen training-score quantile for the predicted-normal mask. An error equal to the threshold is anomalous. Reduce the masked loss over all time positions. Skip the parameter step if the predicted-normal mask is empty. One arrived window advances the trend once. Use the approved nonoverlap default and preserve method trend, optimizer, weights, and the fitted training threshold on resume.

## Stage 5.7 steps 2–6: M2N2

The test first failed because the module was missing. The final test passed for causal trend state, threshold equality, an empty normal mask, a permitted parameter update, and repeated scores after state restoration. Reviewed the diff.

## Stage 5.8 step 1: MOMENT review

Use the [MIT-licensed official runtime revision](https://github.com/moment-timeseries-foundation-model/moment/commit/38f7310ad594100747ca2a8357e9c7ca7d323e0e) and the [MIT-licensed MOMENT-1-base weight revision](https://huggingface.co/AutonLab/MOMENT-1-base/tree/b0ae5751d8ef43d72ad48fb5128e2ddc93c94b53). Load the safetensors file and record its hash. Keep 512 observations per window and eight per patch. Pass all observed positions as valid and process channels independently. Use mean channel reconstruction error at each time point. Train only the reconstruction head in offline linear probing and use reconstruction MSE to select the validation checkpoint. Both online variants keep the pretrained or selected probing weights fixed. Save the pretrained dependency identity and the trained head only when applicable. No test labels enter the method.

## Stage 5.8 steps 2–6: MOMENT

1. The first test failed because the module was missing. Fixed the environment by restoring the project's PyTorch 2.9.1 pin to match TorchVision 0.24.1. Installed MOMENT 0.1.5 from the recorded source revision with Transformers 4.57.6 and Hugging Face Hub 0.36.2.
2. Downloaded only the pinned config and safetensors file. The weight SHA-256 is `1a436826ffe618273ec62b9656dc4cab8edc470364f104e90542a4ebc14fb825`. The config SHA-256 is `f1c66c2bb845229c0ed27a1600dbcc956b85ab21f9e5fd8a1663e6641bed7755`.
3. The actual cached checkpoint passed the zero-shot and linear-probing test in offline network mode. It produced finite point scores for two channels. Gradients reached only the head. Reviewed the diff.

## Stage 5.9: construction and the Phase 5 gate

1. Reproduced missing constructor dispatch. The lazy constructor table now covers all eight methods. All 14 profile construction and checkpoint round-trip checks passed. The signature checklist confirms fit, score, and checkpoint methods for every class. MUSTARD, CANDI, and M2N2 supply update methods. MUSTARD A0 skips parameter updates.
2. Added a known-value MUSTARD test for variance across retrieval samples. It checks the difference between point-score variance and variance of the window mean. Insufficient samples leave the variance unavailable.
3. Ran the combined replacement suite with network access disabled for MOMENT. All 89 tests passed. The system reported a joblib physical-core detection warning. The diff and line-limit checks passed.
4. The method reviews record departures from the author flows. Reuse of the pinned offline threshold and exact run recovery remain workflow checks.

## Stage 6.1 step 1: calibration source decision

1. On 2026-11-19, the user chose Option A.
   The retained normalized-input SMD q99 profile uses synthetic windows from the saved validation split, as in `scripts/cli/evaluate.py`.
2. The [decision record](../plan/decisions-runtime-replacement.md#smd-synthetic-calibration-choice--2026-11-19) records the reason and the limitation.
   The [profile contract](runtime-profile-contract.md#11-exact-profile-references) defines the source.
3. Stage 6.1 step 1 remains open for the offline tests.

## Stage 6.1 offline execution

1. Wrote the offline selection and calibration tests, then the epoch interruption test. All four cases showed the missing offline entry point before implementation.
2. Added selected-checkpoint loading, validation calibration, split results, and epoch/stage recovery. The synthetic SMD test checks q99 for both thresholds and separates the generated mask from source-label eligibility.
3. The replacement suite passed 93 tests with one joblib warning. After adding checkpoint candidate cleanup, the focused offline and MUSTARD review passed 16 tests. Ruff, line-limit, and whitespace checks passed.
4. One real SMD entity (`machine-1-1`) completed offline MUSTARD O0 with one epoch per stage and bounded windows. It used saved prepared data and produced all three split results. The latent q95 came from validation. This was a direct workflow check with temporary outputs.

## Stage 6.2 online execution

1. Wrote event order, future observation, continuation, frozen-parameter, and dependency tests in checklist order. Seven tests showed the missing online entry point before implementation.
2. Added causal score recording, checked updates, compact point score storage, and two-slot resume. The continuation tests check RNG, optimizer, buffer, point scores, and the pinned parent. A signal test checks the latest complete window.
3. Ran the bounded MUSTARD O0 to A2 workflow for prepared SMD entity `machine-1-1` with raw input MSE and latent MSE. Both runs loaded the matching offline q95 exactly and fitted triage separately from validation. A separate run verified the Option A synthetic SMD q99 profile.
4. The replacement suite passed 102 tests with the existing joblib warning. After freezing profile inputs in resolved settings and checking parent data/method compatibility, the focused suite passed 43 tests. Ruff and line-limit checks passed.
5. Ruling: an interrupted fit retains only referenced best checkpoint candidates. A successful run keeps the selected stage artifacts and discards its two working slots. Later cleanup still needs its own gate.

## Stage 7.1 definition editing and planning

1. Wrote five tests in checklist order. They showed that the experiment module was missing, then passed after implementation.
2. Planning checks the one-profile-per-dataset rule and the selected prepared windows. Smoke planning selects one combination per method and phase and inserts required offline variants. Planning creates no run IDs or result files.
3. Ruling: default planning uses the prepared manifest's entity order. An explicit cell can select a smaller entity set. Smoke selects the first eligible configured cell in that order.
4. Ruff, line-limit, and whitespace checks passed. Experiment execution and resume remain separate work.

## Stage 7.2 execution and experiment recovery

1. Wrote failure, dependency order, interruption, and preservation tests. They showed that experiment execution was missing before implementation. All 26 experiment, artifact, online, and offline tests then passed.
2. Runs execute in dependency order. A failed run does not prevent independent runs. A child with no eligible offline result remains blocked.
3. Ruling: an attempt records the allocated schedule once. A later resume uses those settings, IDs, and limits. It preserves successful runs and continues eligible unfinished runs across normal and smoke attempts.
4. Ruff, line-limit, and whitespace checks passed.

## Stage 8.1 reporting

1. The three report tests showed the missing module before implementation and passed after implementation.
2. Reports select one successful finite result per seed. Detailed and mean-only CSV files retain split and profile identity. Tests check missing seeds, incompatible settings, precision, and smoke exclusion.
3. Ruling: columns use `split/entity/metric`. Report metadata records the profile and effective precision. Incompatible selected seed results stop table creation so that the table cannot mix scientific settings.

## Stage 8.2 optional integrations

1. The provider fakes showed the missing integration module before implementation. Integration, experiment, and reporting tests then passed 14 checks.
2. W&B uses a native session and logs committed scalar metrics. Kaggle uses only the matched private dataset and the checked experiment files. Its temporary folder contains hard links. The manifest confirms a repeated upload without copying model data.
3. Ruling: read the private dataset reference from `TSAD_KAGGLE_DATASET` when no provider setting supplies it. The Kaggle SDK can retry API and file upload requests internally. The workflow makes one application-level submission and records an ambiguous outcome without making another submission for the same manifest.
4. No live provider calls were made. The W&B SDK in the thesis environment is 0.25.1. The Kaggle adapter uses the documented 2.2.5 interfaces. Ruff, line-limit, and whitespace checks passed.

## Stage 9.1 installed CLI

1. The five command tests showed the missing CLI before implementation. They passed after isolated installation, including a call from outside the checkout.
2. Installed `thesis-tsad` in the thesis environment after checking entrypoint ownership. Verified the installed preprocessing settings command and parsed the Bash and Zsh completion scripts.
3. Ruling: the package lists only the new runtime packages and the exact legacy numerical helper modules that it still uses. Preprocessing keeps its own entrypoint. Legacy entrypoints are not registered.
4. The install reused the checked environment. `pip check` reports five missing VUS dependencies: arch, Cython, hurst, tsfresh, and tslearn. The selected VUS metric module has separate runtime tests. This is a package environment limitation.

## Stage 10.1 installed SMD check

1. The installed CLI validated and ran MUSTARD O0 to A0 on prepared SMD entity `machine-1-1` with seed 6 and the approved smoke limits.
2. Reloaded both checkpoints, both test score arrays, and their metadata. Both phases used the same validation raw-input q95 of `0.013487492408603429`. The online run recorded the exact offline parent and the same prepared identity and scaler.
3. Recomputed FPR from saved scores and the aligned prepared labels. The offline smoke result was `0.00703125` on 1280 covered known-normal points. The online smoke result was zero on 51 covered known-normal points. These are bounded workflow checks.
4. The normal report excluded the smoke results. Three reporting tests passed. All six protected prepared manifest hashes matched their recorded values.

## Stage 10.2 retained coverage

1. Loaded and checked one prepared entity per dataset: ICCAD `console`, SMD `machine-1-1`, IOPS `KPI-05f10d3a-239c-3bef-9bdc-a2feeb0037aa`, NASA `A-1`, and SWaT `swat`.
2. All eight methods completed both phases on prepared SMD data through the installed CLI. A separate pair checked MOMENT linear probing with the cached pinned weights. All runs were bounded, with seed 6 and providers disabled.
3. The first result review found that the configured uncertainty fields did not reach the metric summary. A regression test showed the missing values. The fix uses the evaluation profile's requested field and statistic pairs. The metric suite passed 16 tests. The installed CLI wrote a finite MUSTARD validation point-score variance mean on repeat execution.
4. Real M2N2 continuation stopped after ten complete windows. The method made actual updates. The resumed run produced identical saved scores, network tensors, and trend tensor. The comparison used zero relative and absolute tolerance and preserved the triggered update count.
5. The MUSTARD offline and online point thresholds matched exactly in the bounded raw and latent checks. These checks confirm the selected contracts. The per-method review above records published-implementation departures and numerical limits.

6. All 123 replacement tests passed. The wider suite passed 632, skipped two, and failed four legacy checks. Those four use the original SMD source folder, which was removed as approved after preprocessing.
7. The required prepared entities and cached MOMENT weights were available. The VUS package metadata lists five unused dependencies absent from this environment. The selected metric functions passed the replacement suite. Provider verification used fakes.

## Stage 11.1 replacement cleanup

1. The path map records the caller review for every candidate. Removed 202 tracked files with a checked replacement and no retained caller: 23 source files, 53 scripts, 3 debug definitions, and 123 legacy tests.
2. Kept numerical support and paths with retained callers. Historical analysis, demonstration workflows, and scientific settings without an approved equivalent stay in the checkout. The no-caller check includes Python imports, shell references, notebook references, and YAML references.
3. Removed directories that became empty. All 123 replacement tests passed after cleanup. The maintained runtime and command search found no reference to a retired path.

## Final review corrections

1. Parent lookup now checks code and dependency identity before it selects the latest compatible run.
   An unreadable artifact in the selected run remains an error.
2. Online metrics use only the selected stream range.
   The saved segment bounds still use positions in the original prepared split.
3. Endpoint calibration checks all labels in each source window.
   A known anomaly excludes the window. Unknown labels remain unknown and contribute to the recorded assumption count.
4. A MUSTARD online cell can select O0, O1, or O2.
   The resolver binds that source profile before execution.
5. Method profile validation rejects optimizer names and parameter fields that the method cannot use.
   Requested gradient clipping uses the resolved values.
6. All 167 tests passed after these corrections. All 14 method profiles and the replacement code size checks passed.

Provider interface evidence: [W&B SDK settings](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py) and [Kaggle CLI package](https://pypi.org/project/kaggle/2.2.5/).

## Completion report — 2026-11-19

1. The thesis environment provides the installed `tsad` command. Verified discovery, settings, planning, execution, resume, reporting, and preprocessing argument forwarding.
2. All 167 tests passed after the final review changes. The installed command completed the final checks for all eight methods in both phases, MOMENT linear probing, Option A synthetic validation q99, and explicit MUSTARD O1 to A1 threshold reuse.
3. All six protected prepared manifests retained their recorded hashes. The real M2N2 check reproduced the saved scores and selected state after interruption.
4. Removed 202 paths cleared by the migration review. Kept paths with retained callers and scientific support code. The replacement runtime has 33 Python files and 5281 lines. Some retained legacy files exceed the new code size limits.
5. Local Markdown links, Ruff checks, whitespace checks, and the `.codex/` ignore rule passed. The provider checks used fakes. The wider VUS package still declares five dependencies absent from this environment. The exercised metric paths passed.
6. Results are from bounded SMD runs and method tests. They do not establish full-experiment accuracy or numerical parity with author implementations.

| Group | Start files | Start lines | Final files | Final lines |
| --- | ---: | ---: | ---: | ---: |
| src | 126 | 26131 | 136 | 28120 |
| scripts | 118 | 22161 | 65 | 15047 |
| tests | 137 | 20791 | 30 | 3746 |
| configs | 801 | 35565 | 833 | 36767 |
