# Source-baseline audit

Date: 2026-11-03  
Status: static review; no source changes or tests run.

## Scope and size

The agreed counter includes project-owned `.py` and `.sh` files under `src/` and `scripts/`; it excludes blank lines, comments, tests, configs, documents, and vendored code.

| Scope | Files | Counted lines |
|---|---:|---:|
| `src/` | 126 | 23,452 |
| `scripts/` | 118 | 18,845 |
| Total | 244 | 42,297 |

Across the total, 127 files exceed 100 counted lines, 83 exceed 200, and 48 have at most 10. Exact AST body matching found 52 cross-file groups, with roughly 712 copied counted lines beyond one body per group. This estimate may double-count nested/partial overlaps and is not a savings claim.

## Repeated-body review

Group numbers refer to the AST inventory captured during this audit.

| Groups | Repeated code | Caller/contract finding | Classification |
|---|---|---|---|
| 1–9 | Label-span normalization and dataset span-row helpers repeated between the CLI and helper module; binary-label normalization also appears in the suffix-count script. | Tests import the CLI helper names. The CLI already calls collection/summarization from the helper module. Same signatures and bodies permit module-level re-exports that preserve those import paths. | Safe deduplication candidate; about 195 lines from the CLI copies, plus one extra 11-line normalizer copy if shared. |
| 10 | Two JSON readers. | Both have the same body as `scripts/ops/_report_io.py:load_json_file`; local callers can keep aliases. | Safe deduplication candidate if the same error behavior is retained. |
| 11–12 | Two one-off matrix generators share `_model_overrides` and a short `main` body. | Their module constants and generated experiment matrices differ. | Share only the helper; retain or retire each generator as a separate experiment-scope decision. |
| 13–14 | Evaluation registration and protocol-validation delegates across benchmark runners. | Public runner names are used as test patch points. | Keep the public names; alias a single owner. |
| 15–17 | Offline/online config loading, data override, and baseline construction. | Called by distinct benchmark flows. The model registries and output contracts are method/phase-specific. | Share proven common transforms; keep each runner contract. |
| 18–28 | Helper functions duplicated between `run_thesis_offline_benchmark.py` and its `_internal` helper module. | The active runner imports only `_export_offline_retention_bundle` from that helper. The other helper workflows have no in-repository callers; tests import the public runner instead. The active export calls three local writers. | Remove uncalled helper workflows, retain the active export and its writers. The uncalled function bodies total 347 counted lines; import cleanup changes the final saving. |
| 29 | Four `_resolve_path` helpers in operational scripts. | Each is local to an operational CLI; one signature reverses the union annotation order but has the same body. | Safe to share if those operations remain; moot for retired scripts. |
| 30–31 | Offline/online report formatting and `main` bodies. | Each report uses different metric, method, and file constants. | Keep report outputs or retire the report feature; factor only formatting that accepts those values as inputs. |
| 32 | Positive-span extraction in protocol audit and visualization. | Same signature/body; each caller consumes half-open spans. | Safe shared helper if visualization remains. |
| 33–34 | KMeansAD and Isolation Forest `score_sequence`/`native_score` wrappers. | Each uses its own fitted scorer and calibration object; returned output keys and assembly are the same. | Share post-score assembly while retaining method-specific scoring and calibration. |
| 35 | 2-D sequence validation in STUMPY and baseline bases. | Same conversion to float64 and shape error behavior. | Safe shared helper. |
| 36 | Batch cloning in augmentation, RedLamp, and THESIS. | Same behavior: copy the mapping and clone tensor values. | Safe shared helper; keep batch contents and tensor identity semantics unchanged. |
| 37 | Maximum online-step parsing in engine and experiment entrypoint. | Same parser body; engine tests directly cover its `None` and integer cases. | One parser with a compatibility alias for the other path. |
| 38 | Stable JSON writer in three online runners. | Same parent-directory creation, sorted/indented JSON, newline, and returned path. | Safe shared helper. |
| 39–42 | RedLamp encoder-profile selection and step wrappers duplicated in its helper module and class. | The class owns the active methods; repository search found no calls to the helper copies. | Remove helper copies; retain the class methods. |
| 43–50 | Gradient-conflict profiling across RedLamp and THESIS. | Both implementations are live behind config flags and each has a focused test. | Contract-sensitive. Deduplicate into one owner or retire the diagnostic explicitly; do not call it dead. |
| 51 | Sample variance duplicated in two THESIS modules. | Both callers use the same validation and correction rule. | Safe deduplication candidate; keep the model method name as an alias if needed. |
| 52 | Tensor batch-to-device movement in three runtime modules. | Same mapping operation; callers pass a device and preserve non-tensor values. | Safe shared helper with compatibility aliases. |

Static caller checks do not replace numerical or runtime tests. No behavior change is claimed before implementation and parity checks.

### Static savings confirmed

- The repeated span helpers account for 206 gross counted lines across copies. Re-exporting functions through existing modules preserves the CLI import paths; imports/aliases reduce the eventual net saving slightly.
- In `_internal/run_thesis_offline_benchmark_helpers.py`, the active retention export and its three writers use 146 counted function lines. Other local function bodies total 347 lines and have no in-repository caller; retain the export path and remove only after checking import cleanup. This is uncalled-workflow removal, separate from duplicate savings.
- Four private RedLamp helper definitions copied in the class account for 41 lines and have no call sites; the class implementations remain active.

The remaining repeated groups are classified by caller and contract above, but their savings stay unconfirmed until a shared owner can preserve each caller path. The whole-source 712-line duplicate estimate must not be added to the figures above.

## Script-path audit

The Phase 1 matrix and current launch scripts identify these script responsibilities as active: training/evaluation CLI; two-stage offline training; THESIS offline and online runners; generic offline and online baseline runners; SMD matrix config generation; matrix launch; and metrics collection/summary.

The THESIS online runner must remain in the current matrix path. `scripts/benchmarks/run_remaining_smd_matrix.sh` invokes `scripts/benchmarks/run_thesis_online_benchmark.py`; `run_online_wet_matrix.sh` does as well. The Phase 1 caller table under-reports this direct path and should include both the specialized THESIS runner and the generic online-streaming runner.

`generate_remaining_smd_benchmark_configs.py` imports the offline, SMD, and online-streaming generators. Those dependencies cannot be removed independently while keeping the current matrix generator. Root-level `scripts/*.py` aliases are imported or monkeypatched by tests and mentioned in current docs; removing them requires migrating those references to canonical nested paths.

### Script-family inventory

| Script family | Files | Lines | Audit decision |
|---|---:|---:|---|
| `cli/` | 3 | 787 | Keep train and evaluate paths; package marker is a file-count candidate. |
| `experiments/` | 9 | 1,912 | Keep two-stage and online execution; compare/multiseed/ablation/direct-branch paths need matrix-by-matrix review. |
| `benchmarks/` | 33 | 6,755 | Keep current runners and config-generation dependencies; retire only alternate studies and one-off launchers after O2 and matrix contracts are resolved. |
| `ops/` | 25 | 4,909 | Keep the imported threshold-scoring adapter; review report, preflight, forensic, recalibration, backfill, and pruning tools individually. |
| `analysis/` | 9 | 1,987 | Post-hoc research analysis, outside runtime. Candidate for retirement after preserving needed reports and the SMD inspection workflow. |
| `visualization/` | 8 | 1,242 | Optional plots; retain synthetic-anomaly visualization for the required injection-inspection workflow. |
| `launchers/` | 3 | 468 | Keep one canonical matrix launcher if remote execution remains required; compare it with benchmark shell orchestration before removal. |
| Root script aliases and one-offs | 28 | 785 | Migrate public command/import references, then remove aliases and unrelated direct-branch/plot wrappers where not in the required matrix. |

The smallest current-matrix script set identified in this static review is 18 files and 6,672 counted lines: `scripts/cli/{train,evaluate}.py`; `scripts/experiments/run_two_stage_offline_pretraining.py`; `scripts/benchmarks/_config_generation_helpers.py`; `scripts/benchmarks/generate_{offline_benchmark,smd_benchmark,online_streaming_benchmark,remaining_smd_benchmark}_configs.py`; `scripts/benchmarks/run_{offline_benchmark,online_streaming_benchmark,thesis_offline_benchmark,thesis_online_benchmark}.py`; `scripts/benchmarks/_internal/run_thesis_offline_benchmark_helpers.py`; `scripts/benchmarks/{collect_remaining_smd_metrics,summarize_benchmark_results}.py`; `scripts/benchmarks/run_remaining_smd_matrix.sh`; `scripts/benchmarks/_remaining_smd_resource_env.sh`; and `scripts/ops/threshold_artifact_v4_online_scoring.py`. The total was recalculated with the agreed counter. This is a candidate minimum for the currently documented matrix, not a proposed target tree or proof that every file must remain after consolidation. It excludes optional cloud orchestration and includes a helper whose uncalled workflows are retirement candidates. Keeping the separate `scripts/experiments/run_online_adaptation.py` entrypoint and the standalone ablation CLI adds two files and 504 lines.

The set has no path to the 100-line per-file cap as written: 15 of its 18 files exceed 100 counted lines, and its largest file exceeds 1,000. Meeting the per-file cap requires removing responsibilities or consolidating code into fewer files; splitting alone would increase the file count. The three limits must be planned together.

## Feature-retirement candidates

These capabilities are outside the MUSTARD equations and method matrix, subject to preserving any final artifacts and user-facing behavior that still matter:

- Retire one-off synthetic-profile, cross-dataset span, drift-analysis, and visualization CLIs after their final figures/tables and synthetic-anomaly inspection workflow are retained. Keep `visualize_synthetic_anomalies.py` for the documented injection-inspection workflow unless an equivalent path is kept.
- Retire one-off operational report, forensic, backfill, pruning, and preflight CLIs after their final outputs and required artifact/provenance contracts are preserved. Do not remove `threshold_artifact_v4_online_scoring.py` while the THESIS offline runner imports it.
- Retire alternate reconstruction/VUS matrix generators and specific smoke/wet launch scripts that are outside the canonical matrix. Keep the direct-branch bridge until the matrix-required O2 path is resolved.
- Retire optional `retain_for_eda` trace/score bundles if selected raw traces are not required. Preserve report-ready metrics, checkpoint provenance, initialization/best checkpoints, and threshold identity.
- Retire optional gradient-conflict profiling and unused THESIS regularizer/CKA branches only as explicit experiment-feature removals; their focused tests/configs must be updated to the new supported scope.

### Known callers affected by these candidates

| Candidate | Existing callers or contracts to update if retired |
|---|---|
| Span-analysis CLIs/helpers | `tests/evaluation/test_anomaly_span_length_summary.py`; `tests/core/test_dated_log_paths.py`; preserve any required saved span summaries. |
| Visualization scripts | `tests/visualization/test_raw_input_mse_histograms.py`; `tests/evaluation/test_evaluation_visualization.py`; `tests/core/test_dated_log_paths.py`; retain the synthetic-anomaly inspection path required by `codebase_preferences.md`. |
| Operational scripts | Focused tests include `tests/ops/test_recalibrate_thesis_threshold_artifacts_v4.py`, `tests/ops/test_backfill_uq_summary.py`, `tests/ops/test_backfill_all_uq_summaries.py`, `tests/ops/test_aggregate_online_table3_metrics.py`, and benchmark preflight tests. The current notes document report collection and compact evaluation paths. |
| THESIS optional regularizers | `tests/models/test_one_multitask_train_step.py` enables diversity, variance, covariance, usage, and gate losses; online tests explicitly configure these off. No scanned model YAML enables them. |
| CKA-gated fusion | `tests/models/test_one_multitask_train_step.py` and `tests/models/test_exp2_two_view_cka.py` enable it; model YAMLs set it false. |
| Gradient-conflict profiling | `tests/models/test_thesis_multitask_gradient_profiling.py`, `tests/models/test_redlamp_cnn_baseline_shapes.py`, and `tests/models/test_redlamp_baseline_with_gradient_profiling_step.py` exercise it; one THESIS model YAML enables it. Removing it also changes the config and logged-metric contract. |
| Alternate direct/VUS experiments | `tests/benchmarks/test_pro_reconstruction_vus_budget_matrix.py`, `tests/experiments/test_direct_branch_routing_bridge.py`, and current research notes reference these paths. Keep the direct-branch bridge until the official O2 route is resolved. |

### Configuration and documented-command impact

- The analysis, visualization, and operations script retirements have no matching runtime config keys in the scan. Their Python APIs are imported by the tests listed above, and their commands are documented in `README.md`, `documents/notes/eval18_compact_output_paths.md`, `documents/notes/offline_runtime_and_data_flow.md`, `documents/notes/offline_report_collection_plan.md`, `documents/notes/online_runtime_flow_vscode_debug_guide.md`, and `documents/notes/smd-all-machines-experiment-matrix.md`. Update or remove those command references with any retirement.
- The THESIS feature candidates affect model configs. `enable_cka_gated_fusion` is false in the selected two-stage and RedLamp-aligned configs; `enable_gradient_conflict_profiling` is true in `configs/model/thesis_multitask_redlamp_multiclass_redlamp_aligned.yaml`. The two RedLamp CNN config files set profiling false. Optional loss weights are described as zero in `documents/notes/experiment-flows-and-equations.md`, while `tests/models/test_one_multitask_train_step.py` enables the optional losses and CKA branch. Retiring these features requires removing the corresponding parser/config fields and updating these tests and the enabled THESIS YAML, not merely deleting helper functions.
- Current matrix commands are documented in `documents/notes/smd-all-machines-experiment-matrix.md`. The THESIS offline runner imports the threshold-scoring tool, so that operational script is an active code dependency even though it is not a baseline method.

No config directory contains a baseline method's required runtime through these post-processing utilities; their impact is analysis/reporting and CLI behavior. A retirement plan must preserve any thesis tables, diagnostics, or provenance already selected as official artifacts.

## Limit arithmetic

Removing all of `scripts/analysis`, `scripts/ops`, and `scripts/visualization` would remove 42 files and 8,138 lines, but would still leave 202 files and 34,159 lines. Removing every script would still leave the 126-file, 23,452-line `src/` baseline. Therefore script deduplication and script-feature retirement alone cannot meet the 40-file/4,000-line limit. Reaching it while preserving the Phase 1 matrix requires substantial cuts to core runtime support or a narrower method/runtime scope; safe duplicate removal does not bridge that gap.

## Audit result and limits

The source and script baseline has been audited statically. The best-supported low-risk reductions are shared helpers with compatibility aliases, removal of uncalled helper workflows, and retiring individually selected analysis/operations tools after checking their output contracts. These changes are far below the target. The 18-file matrix-support candidate still exceeds 4,000 lines by 2,672 and has 15 files over 100 lines. Keeping online test-time adaptation and standalone ablation support increases this gap. This does not prove that the limits are impossible, but it shows that deduplication and script trimming alone do not meet them.

No source implementation was changed and no tests were run. Static caller searches can miss dynamic imports, external user commands, or artifact consumers. Runtime parity and scientific-output equivalence remain unverified.
