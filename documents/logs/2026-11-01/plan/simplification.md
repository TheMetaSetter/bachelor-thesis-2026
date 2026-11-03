> SUPERSEDED: This attempt wrongly removed scientific baselines. The source, scripts, configurations, tests, README and architecture tree have been restored to revision 753ad03ed60b2d41b078e53bf421429bb0bd6c0f. Earlier reduction figures no longer describe the current codebase. See baseline-restoration.md.

# First codebase simplification

Goal: fewer files and dispatch layers, with unchanged mathematical behavior.
Execution: sequential, in this checkout; preserve existing local edits. No commits or remote runs.
Tools: Python AST and pathlib for mechanical moves; Git diff and ripgrep for caller review; local PyTorch and pytest for verification.

## Design decisions

Keep datasets, all model variants, losses, injection, calibration, online state, ablations, diagnostics, reporting, demos, configs, and saved artifacts. These support research or reader inspection.
Remove duplicate command modules, wildcard forwarding, module replacement, and empty implementation packages. They add navigation work without adding research behavior. Existing top-level command names remain canonical. Imports through the removed internal packages migrate; no compatibility wrappers are added.
RedLamp owns forward, objective, and stage steps in its public file. Gradient diagnostics remain a named helper. Online adaptation owns its lifecycle; frozen encoder and projector are named neural helpers. THESIS calibration belongs in its public class; remove the empty loss aggregator and state aggregator while keeping specialized mathematics intact.
Alternatives: renaming everything to a new package adds migration without reducing computation; deleting mathematical variants risks losing research behavior. Neither is chosen.
Terminology: no mathematical objects renamed. Only Python module owners change, as listed below. Checkpoint tensor keys and configuration keys stay unchanged.

## Exact module migration

| Old file | Canonical file | Status |
|---|---|---|
| `scripts/cli/evaluate.py` | `scripts/evaluate.py` | merged into public owner |
| `scripts/benchmarks/generate_benchmark_smoke_configs.py` | `scripts/generate_benchmark_smoke_configs.py` | merged into public owner |
| `scripts/benchmarks/generate_offline_benchmark_configs.py` | `scripts/generate_offline_benchmark_configs.py` | merged into public owner |
| `scripts/benchmarks/generate_online_benchmark_configs.py` | `scripts/generate_online_benchmark_configs.py` | merged into public owner |
| `scripts/benchmarks/generate_online_streaming_benchmark_configs.py` | `scripts/generate_online_streaming_benchmark_configs.py` | merged into public owner |
| `scripts/benchmarks/generate_smd_benchmark_configs.py` | `scripts/generate_smd_benchmark_configs.py` | merged into public owner |
| `scripts/experiments/run_ablation.py` | `scripts/run_ablation.py` | merged into public owner |
| `scripts/experiments/run_comparative_smd_experiments.py` | `scripts/run_comparative_smd_experiments.py` | merged into public owner |
| `scripts/experiments/run_multiseed_experiments.py` | `scripts/run_multiseed_experiments.py` | merged into public owner |
| `scripts/benchmarks/run_offline_benchmark.py` | `scripts/run_offline_benchmark.py` | merged into public owner |
| `scripts/experiments/run_online_adaptation.py` | `scripts/run_online_adaptation.py` | merged into public owner |
| `scripts/benchmarks/run_online_streaming_benchmark.py` | `scripts/run_online_streaming_benchmark.py` | merged into public owner |
| `scripts/benchmarks/run_thesis_offline_benchmark.py` | `scripts/run_thesis_offline_benchmark.py` | merged into public owner |
| `scripts/benchmarks/run_thesis_online_benchmark.py` | `scripts/run_thesis_online_benchmark.py` | merged into public owner |
| `scripts/experiments/run_two_stage_offline_pretraining.py` | `scripts/run_two_stage_offline_pretraining.py` | merged into public owner |
| `scripts/benchmarks/summarize_benchmark_results.py` | `scripts/summarize_benchmark_results.py` | merged into public owner |
| `scripts/cli/train.py` | `scripts/train.py` | merged into public owner |
| `src/models/baseline_impl/redlamp_baseline.py` | `src/models/redlamp_baseline.py` | merged into public owner |
| `src/models/baseline_impl/redlamp_baseline_helpers.py` | `src/models/redlamp_gradient_metrics.py` | merged into public owner |
| `src/models/online_impl/online_adaptation.py` | `src/models/online_adaptation.py` | merged into public owner |
| `src/models/online_impl/online_adaptation_helpers.py` | `src/models/online_encoder.py` | merged into public owner |

Additional removals: `scripts/cli/__init__.py`, `src/models/baseline_impl/__init__.py`, `src/models/online_impl/__init__.py`, `src/models/thesis_multitask_impl/thesis_multitask_loss_mixin.py`, and `src/models/thesis_multitask_impl/thesis_multitask_state_mixin.py`.

## Retired operational features

Remove these five one-off programs: `scripts/ops/build_phase7_audit_artifacts.py`, `scripts/ops/debug_pilot_combination_uq_reporting.py`, `scripts/ops/re_evaluate_and_prune_thesis_runs.py`, `scripts/ops/re_evaluate_and_prune_thesis_runs_remaining.py`, `scripts/ops/run_thesis_eval_only_compact.py`.
They duplicate evaluation, inspection, report, or pruning commands and couple these actions to old campaign layouts. No retained source or tests import them; the remaining-run script calls only the retired batch program. Keep the independent evaluator, forensic inspector, summary collectors, and explicit pruning utility. Removal changes these operational command APIs, not model mathematics. Historical documents may still name the retired commands.

## Sequential checklist

### Phase 1: Design and baseline (Git, AST, pytest)
- [x] 1. Read repository rules and inspect tracked files and callers.
- [x] 2. Run the original test suite: 641 passed, 2 skipped.
- [x] 3. Record the complete target Python tree below before implementation.
- [x] 4. Capture deterministic model outputs, losses, gradients, and state in a temporary baseline artifact.

### Phase 2: Commands (Python AST/pathlib, ripgrep, pytest)
- [x] 5. Promote command implementations into their existing public script files.
- [x] 6. Update executable imports, command strings, and repository-root calculations.
- [x] 7. Remove the five retired operational programs, then run command/config/orchestration tests and resolve regressions before continuing.

### Phase 3: Models (Python AST/pathlib, PyTorch, pytest)
- [x] 8. Move RedLamp and online implementations into public model files and update callers.
- [x] 9. Inline RedLamp objective and stage steps; remove duplicate helper implementations.
- [x] 10. Move THESIS calibration into its public class and remove the two aggregator mixins.
- [x] 11. Compare deterministic before/after tensors and run model, online, and checkpoint tests.

### Phase 4: Documentation and verification (Markdown, pytest, Git)
- [x] 12. Replace the README with a short reader guide and module migration note.
- [x] 13. Update the design starter with the implemented file tree.
### Phase 5: Simpler physical tree (updated from the user's request; Python AST/pathlib)
- [x] 14. Record the exact additional moves and complete revised tree before implementation.
- [x] 15. Flatten dataset parsers and traditional baselines; remove redundant neural-baseline forwarding; flatten script support packages.
- [x] 16. Rename the THESIS helper folder and merge encoder, initialization, and passthrough fragments into their related owners.
- [x] 17. Update all executable callers and path-depth calculations; remove only stale bytecode in retired directories.
- [x] 18. Run numerical parity and the full suite; fix regressions before continuing.
### Phase 6: Reduce files and folders substantially (latest user instruction)
- [x] 19. Record the exact retirement list, module moves, and final shallow tree before implementation.
- [x] 20. Remove campaign tooling and only the tests of those retired operational features.
- [x] 21. Flatten core/runtime, datasets, models, evaluation, online helpers, scripts, and tests according to the table.
- [x] 22. Update imports, path-depth calculations, pytest discovery, and Git ignore exceptions.
- [x] 23. Compare mathematical outputs and run all retained tests; resolve regressions.
### Phase 7: At least 80% fewer repository files (latest user instruction)
Baseline: 2984 tracked files at `753ad03ed60b2d41b078e53bf421429bb0bd6c0f`. Target: at most 596 retained/new repository files. Ignore local datasets, virtual environments, caches, and pre-existing ignored personal assets when measuring the versioned library. Do not delete those ignored assets.

- [x] 24. Trace configurations through all 584 retained tests and record the exact retirement manifest in `detail/removals.csv`.
- [x] 25. Retain third-party licenses/provenance and consolidate all historical report JSON/CSV values without changing their values.
- [x] 26. Remove the manifest's exact tracked reference snapshots, generated report tables, slide/image/notebook assets, historical logs, and unused config copies.
- [x] 27. Flatten the 72 retained YAML files and migrate every active config reference.
### Phase 8: Rewrite project-owned core (latest instructions)
The core baseline is `src/` plus `scripts/`: 47,514 Python/shell lines at the starting Git revision. Third-party code, tests, YAML, and research assets do not count toward the 90% core target of at most 4,751 lines. Non-core research assets have been restored. The latest user instruction supersedes the broader file-pruning target.

- [x] 28. Record the 72-config equality result and the five obsolete framework/config-path failures; retain the before-change numerical artifact and a temporary source snapshot for comparison.
- [x] 29. Rewrite model internals, data operations, and causal online operations in disjoint scopes while preserving numerical behavior and native APIs.
- [x] 30. Replace the surrounding config/training/evaluation framework with a small explicit library path; retire unneeded operational features and their tests.
- [x] 31. Integrate the rewrites; run focused scientific tests and exact before/after tensor/gradient/state comparisons; fix every observed regression.
- [x] 32. Run one tiny complete training/calibration/evaluation/checkpoint flow through the new public path.
- [x] 33. Measure project-owned source lines and files against the original revision; record the remaining gap at the user-requested stopping point. The target was not reached.
- [x] 34. Update the reader guide, complete file tree, migration/feature-removal record, and verification report.
- [x] 35. Obtain independent review and run final checks; preserve original user edits.
- [x] 36. Report exact results, limits, and any unmet requirement honestly.

No minification, compressed source, archived source, or third-party deletions count as core simplification. Model equations, injection distributions/order, loss gradients, score calibration, and causal update order are the numerical comparison boundary. Removed orchestration/reporting/publishing interfaces must be named explicitly. Native method checkpoint tensor keys remain stable.

## Additional directory migrations

| Old file | New file |
|---|---|
| `src/models/thesis_multitask_impl/thesis_multitask_components.py` | `src/models/thesis/components.py` |
| `src/models/thesis_multitask_impl/thesis_multitask_config_parsing.py` | `src/models/thesis/parse_config.py` |
| `src/models/thesis_multitask_impl/thesis_multitask_loss_core_mixin.py` | `src/models/thesis/losses.py` |
| `src/models/thesis_multitask_impl/thesis_multitask_loss_gradient_mixin.py` | `src/models/thesis/gradient_metrics.py` |
| `src/models/thesis_multitask_impl/thesis_multitask_loss_step_mixin.py` | `src/models/thesis/steps.py` |
| `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py` | `src/models/thesis/forward.py` |
| `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py` | `src/models/thesis/geometry.py` |
| `src/models/thesis_multitask_impl/thesis_multitask_routing_helpers.py` | `src/models/thesis/sampling.py` |
| `src/models/thesis_multitask_impl/thesis_multitask_routing_mixin.py` | `src/models/thesis/routing.py` |
| `src/models/thesis_multitask_impl/thesis_multitask_setup_helpers.py` | `src/models/thesis/initialization.py` |
| `src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py` | `src/models/thesis/setup.py` |
| `src/models/thesis_multitask_impl/thesis_multitask_state_memory_mixin.py` | `src/models/thesis/memory.py` |
| `src/models/thesis_multitask_impl/thesis_multitask_state_schedule_mixin.py` | `src/models/thesis/schedule.py` |
| `src/models/thesis_multitask_impl/thesis_multitask_state_serialization_mixin.py` | `src/models/thesis/checkpoint.py` |
| `src/models/thesis_multitask_impl/__init__.py` | `src/models/thesis/__init__.py` |
| `src/data/datasets/anomaly_archive.py` | `src/data/anomaly_archive.py` |
| `src/data/datasets/smd.py` | `src/data/smd.py` |
| `src/baselines/traditional/__init__.py` | `src/baselines/__init__.py` |
| `src/baselines/traditional/stumpy_channel_ab.py` | `src/baselines/stumpy_channel_ab.py` |
| `src/baselines/traditional/kmeans_ad.py` | `src/baselines/kmeans_ad.py` |
| `src/baselines/traditional/base.py` | `src/baselines/base.py` |
| `src/baselines/traditional/iforest.py` | `src/baselines/iforest.py` |
| `scripts/experiments/run_direct_branch_routing_bridge.py` | `scripts/run_direct_branch_routing_bridge.py` |
| `scripts/experiments/_internal/run_comparative_smd_experiments_support.py` | `scripts/_comparative.py` |
| `scripts/benchmarks/_internal/run_thesis_offline_benchmark_helpers.py` | `scripts/_offline.py` |

Merge `thesis_multitask_encoder.py` into `thesis/components.py`; merge `thesis_multitask_state_memory_init_helpers.py` into `thesis/initialization.py`; merge the passthrough method from `thesis_multitask_state_passthrough_mixin.py` into `thesis/routing.py`. Remove the old neural-baseline forwarding package and empty script package initializers. Public mathematical class names remain unchanged. Dataset/traditional-baseline Python import paths change as listed; tensor/config schemas do not.

## Final retirements

Python/shell command files removed:

- `scripts/launch_tmux_comparative_smd_experiment.sh`
- `scripts/generate_offline_benchmark_configs.py`
- `scripts/run_direct_branch_routing_full.py`
- `scripts/generate_online_streaming_benchmark_configs.py`
- `scripts/generate_smd_benchmark_configs.py`
- `scripts/generate_online_benchmark_configs.py`
- `scripts/run_o1_recon075_cls025_machine_3_4_seed36_smoke.sh`
- `scripts/plot_machine_3_9_configurable.py`
- `scripts/generate_benchmark_smoke_configs.py`
- `scripts/plot_machine_3_4_anomaly_scores.py`
- `scripts/run_direct_branch_routing_bridge.py`
- `scripts/launch_tmux_two_stage_experiment.sh`
- `scripts/visualize_smd_12_classes_w20.py`
- `scripts/compress_data_zstd.sh`
- `scripts/run_comparative_smd_experiments.py`
- `scripts/run_direct_branch_routing_evaluation.py`
- `scripts/plot_iccad_a8.py`
- `scripts/_comparative.py`
- `scripts/run_multiseed_experiments.py`
- `scripts/run_direct_branch_routing_smoke.py`
- `scripts/visualization/__init__.py`
- `scripts/analysis/__init__.py`
- `scripts/benchmarks/run_online_wet_matrix.sh`
- `scripts/benchmarks/run_remaining_smd_smoke.sh`
- `scripts/benchmarks/generate_full_direct_recon075_cls025_matrix.py`
- `scripts/benchmarks/run_full_direct_recon075_cls025_offline_matrix.sh`
- `scripts/benchmarks/run_o0_recon075_cls025_direct_machine_1_6_seed6_smoke.sh`
- `scripts/benchmarks/run_o1_recon075_cls025_machine_3_4_seed36_smoke.sh`
- `scripts/benchmarks/_remaining_smd_resource_env.sh`
- `scripts/benchmarks/__init__.py`
- `scripts/benchmarks/run_main_method_smoke_tmux.sh`
- `scripts/benchmarks/_config_generation_helpers.py`
- `scripts/benchmarks/run_remaining_smd_cloud_tmux.sh`
- `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py`
- `scripts/benchmarks/run_remaining_smd_wet.sh`
- `scripts/benchmarks/collect_remaining_smd_metrics.py`
- `scripts/benchmarks/run_direct_branch_synthetic_validation_threshold_rerun.sh`
- `scripts/benchmarks/run_remaining_smd_matrix.sh`
- `scripts/benchmarks/run_o1_recon075_cls025_machine_3_4_seed36_wet.sh`
- `scripts/benchmarks/run_pro_reconstruction_vus_budget_matrix.sh`
- `scripts/benchmarks/generate_pro_reconstruction_vus_budget_matrix.py`
- `scripts/benchmarks/run_synthetic_validation_threshold_rerun.sh`
- `scripts/benchmarks/run_pro_reconstruction_vus_budget_smoke.sh`
- `scripts/launchers/launch_tmux_comparative_smd_experiment.sh`
- `scripts/launchers/__init__.py`
- `scripts/launchers/launch_tmux_two_stage_experiment.sh`
- `scripts/ops/render_online_phase_metric_table.py`
- `scripts/ops/backfill_all_uq_summaries.py`
- `scripts/ops/read_offline_uq_table2.py`
- `scripts/ops/prepare_raw_mse_offline_rerun.py`
- `scripts/ops/build_prune_manifest.py`
- `scripts/ops/build_remote_offline_report_data.py`
- `scripts/ops/write_offline_report_bundle.py`
- `scripts/ops/__init__.py`
- `scripts/ops/preflight_comparative_smd_server.py`
- `scripts/ops/collect_offline_report_data.py`
- `scripts/ops/inspect_thesis_run_artifacts.py`
- `scripts/ops/preflight_full_benchmark_matrix.py`
- `scripts/ops/render_offline_phase_metric_table.py`

Tests removed only for the retired operational features:

- `tests/benchmarks/test_offline_benchmark_config_generation.py`
- `tests/benchmarks/test_remaining_smd_benchmark_matrix.py`
- `tests/benchmarks/test_direct_branch_routing_evaluation_runner.py`
- `tests/benchmarks/test_comparative_tmux_launcher.py`
- `tests/benchmarks/test_pro_reconstruction_vus_budget_matrix.py`
- `tests/benchmarks/test_comparative_preflight.py`
- `tests/benchmarks/test_full_benchmark_matrix_preflight.py`
- `tests/benchmarks/test_comparative_runner.py`
- `tests/benchmarks/test_benchmark_config_generation.py`
- `tests/benchmarks/test_two_stage_tmux_launcher.py`
- `tests/benchmarks/test_direct_branch_routing_full_runner.py`
- `tests/benchmarks/test_raw_mse_rerun_configs.py`
- `tests/benchmarks/test_multiseed_launcher.py`
- `tests/benchmarks/test_direct_branch_routing_smoke_runner.py`
- `tests/ops/test_backfill_all_uq_summaries.py`
- `tests/online/test_online_benchmark_config_generation.py`
- `tests/online/test_online_streaming_benchmark_config_generation.py`

## Final shallow module moves

| Old file | New file |
|---|---|
| `src/core/console.py` | `src/console.py` |
| `src/core/config.py` | `src/config.py` |
| `src/core/registry.py` | `src/registry.py` |
| `src/core/config_aliases.py` | `src/config_aliases.py` |
| `src/core/config_help.py` | `src/config_help.py` |
| `src/core/artifact_naming.py` | `src/artifact_naming.py` |
| `src/core/config_experiment_validation.py` | `src/config_experiment_validation.py` |
| `src/core/uq_summary.py` | `src/uq_summary.py` |
| `src/core/evaluation_trace_compaction.py` | `src/evaluation_trace_compaction.py` |
| `src/core/seed.py` | `src/seed.py` |
| `src/core/contracts.py` | `src/contracts.py` |
| `src/core/config_model_validation.py` | `src/config_model_validation.py` |
| `src/core/artifact_integrity.py` | `src/artifact_integrity.py` |
| `src/core/runtime_components.py` | `src/runtime_components.py` |
| `src/engine/online_loop.py` | `src/online_loop.py` |
| `src/engine/checkpoint.py` | `src/checkpoint.py` |
| `src/engine/artifact_sinks.py` | `src/artifact_sinks.py` |
| `src/engine/thresholding.py` | `src/thresholding.py` |
| `src/engine/logger.py` | `src/logger.py` |
| `src/engine/trainer.py` | `src/trainer.py` |
| `src/engine/evaluator.py` | `src/evaluator.py` |
| `src/engine/runtime_resource_monitor.py` | `src/runtime_resource_monitor.py` |
| `src/engine/online_tta/verification_buffer.py` | `src/online/verification_buffer.py` |
| `src/engine/online_tta/online_engine_window_core.py` | `src/online/online_engine_window_core.py` |
| `src/engine/online_tta/verification_cycle.py` | `src/online/verification_cycle.py` |
| `src/engine/online_tta/non_overlap_guard.py` | `src/online/non_overlap_guard.py` |
| `src/engine/online_tta/signature_verification.py` | `src/online/signature_verification.py` |
| `src/engine/online_tta/online_optimizer.py` | `src/online/online_optimizer.py` |
| `src/engine/online_tta/online_engine_run.py` | `src/online/online_engine_run.py` |
| `src/engine/online_tta/__init__.py` | `src/online/__init__.py` |
| `src/engine/online_tta/runtime_state.py` | `src/online/runtime_state.py` |
| `src/engine/online_tta/timing_debug.py` | `src/online/timing_debug.py` |
| `src/engine/online_tta/online_losses.py` | `src/online/online_losses.py` |
| `src/engine/online_tta/triage.py` | `src/online/triage.py` |
| `src/engine/online_tta/online_calibration.py` | `src/online/online_calibration.py` |
| `src/engine/online_tta/checkpoint_resolution.py` | `src/online/checkpoint_resolution.py` |
| `src/engine/online_tta/verification_adapter.py` | `src/online/verification_adapter.py` |
| `src/engine/online_tta/online_engine_step.py` | `src/online/online_engine_step.py` |
| `src/engine/online_tta/point_ewma.py` | `src/online/point_ewma.py` |
| `src/engine/online_tta/online_engine.py` | `src/online/online_engine.py` |
| `src/engine/online_tta/online_engine_window_metrics.py` | `src/online/online_engine_window_metrics.py` |
| `src/engine/online_tta/online_engine_shared.py` | `src/online/online_engine_shared.py` |
| `src/metrics/classification_diagnostics.py` | `src/evaluation/classification_diagnostics.py` |
| `src/metrics/online_contract.py` | `src/evaluation/online_contract.py` |
| `src/metrics/pointwise.py` | `src/evaluation/pointwise.py` |
| `src/metrics/affiliation.py` | `src/evaluation/affiliation.py` |
| `src/protocols/synthetic_profile.py` | `src/evaluation/synthetic_profile.py` |
| `src/protocols/point_scores.py` | `src/evaluation/point_scores.py` |
| `src/protocols/threshold_artifact.py` | `src/evaluation/threshold_artifact.py` |
| `src/protocols/point_score_calibration.py` | `src/evaluation/point_score_calibration.py` |
| `src/protocols/reconstruction_scores.py` | `src/evaluation/reconstruction_scores.py` |
| `src/protocols/online_stream_range.py` | `src/evaluation/online_stream_range.py` |
| `src/protocols/smd_benchmark_protocol.py` | `src/evaluation/smd_benchmark_protocol.py` |
| `src/analysis/evaluation_protocol_audit.py` | `src/evaluation/evaluation_protocol_audit.py` |
| `src/analysis/anomaly_archive_kl.py` | `src/evaluation/anomaly_archive_kl.py` |
| `src/baselines/stumpy_channel_ab.py` | `src/models/stumpy_channel_ab.py` |
| `src/baselines/kmeans_ad.py` | `src/models/kmeans_ad.py` |
| `src/baselines/base.py` | `src/models/window_baselines.py` |
| `src/baselines/iforest.py` | `src/models/iforest.py` |
| `src/baselines/online/m2n2.py` | `src/models/m2n2.py` |
| `src/baselines/online/candi.py` | `src/models/candi.py` |
| `src/baselines/online/pre_tta.py` | `src/models/pre_tta.py` |
| `src/baselines/online/adaptive.py` | `src/models/adaptive_baselines.py` |
| `src/baselines/online/frozen.py` | `src/models/frozen_baselines.py` |
| `src/baselines/online/redlamp_encoder_checkpoint.py` | `src/models/redlamp_encoder_checkpoint.py` |
| `src/baselines/online/base.py` | `src/models/online_baselines.py` |
| `src/adapters/moment.py` | `src/models/moment.py` |
| `src/adapters/base.py` | `src/models/encoder_adapter.py` |
| `scripts/visualization/visualize_training_metrics.py` | `scripts/visualize_training_metrics.py` |
| `scripts/visualization/visualize_evaluation_results.py` | `scripts/visualize_evaluation_results.py` |
| `scripts/visualization/visualize_synthetic_anomalies.py` | `scripts/visualize_synthetic_anomalies.py` |
| `scripts/visualization/visualize_smd_benchmark_train_test.py` | `scripts/visualize_smd_benchmark_train_test.py` |
| `scripts/visualization/visualize_classification_diagnostics.py` | `scripts/visualize_classification_diagnostics.py` |
| `scripts/visualization/visualize_smd_train_test_entity.py` | `scripts/visualize_smd_train_test_entity.py` |
| `scripts/visualization/plot_raw_input_mse_histograms.py` | `scripts/plot_raw_input_mse_histograms.py` |
| `scripts/analysis/compare_synthetic_profiles_shared.py` | `scripts/compare_synthetic_profiles_shared.py` |
| `scripts/analysis/export_post_anomaly_suffix_counts.py` | `scripts/export_post_anomaly_suffix_counts.py` |
| `scripts/analysis/kl_divergence_anomaly_archive.py` | `scripts/kl_divergence_anomaly_archive.py` |
| `scripts/analysis/summarize_anomaly_span_lengths_helpers.py` | `scripts/summarize_anomaly_span_lengths_helpers.py` |
| `scripts/analysis/summarize_anomaly_span_lengths.py` | `scripts/summarize_anomaly_span_lengths.py` |
| `scripts/analysis/compare_synthetic_profiles.py` | `scripts/compare_synthetic_profiles.py` |
| `scripts/analysis/rank_smd_train_test_normal_drift.py` | `scripts/rank_smd_train_test_normal_drift.py` |
| `scripts/analysis/compare_synthetic_profiles_helpers.py` | `scripts/compare_synthetic_profiles_helpers.py` |
| `scripts/ops/_report_io.py` | `scripts/_report_io.py` |
| `scripts/ops/recalibrate_thesis_threshold_artifacts_v4.py` | `scripts/recalibrate_thesis_threshold_artifacts_v4.py` |
| `scripts/ops/aggregate_online_table3_metrics.py` | `scripts/aggregate_online_table3_metrics.py` |
| `scripts/ops/backfill_uq_summary.py` | `scripts/backfill_uq_summary.py` |
| `scripts/ops/threshold_artifact_v4_online_scoring.py` | `src/evaluation/threshold_artifact_v4_online_scoring.py` |
| `scripts/ops/forensic_audit_run.py` | `scripts/forensic_audit_run.py` |
| `scripts/ops/prune_raw_trace_artifacts.py` | `scripts/prune_raw_trace_artifacts.py` |
| `tests/demo/test_demo_state.py` | `tests/test_demo_demo_state.py` |
| `tests/demo/test_demo_app.py` | `tests/test_demo_demo_app.py` |
| `tests/demo/test_demo_stream_queue.py` | `tests/test_demo_demo_stream_queue.py` |
| `tests/demo/test_live_online_replay.py` | `tests/test_demo_live_online_replay.py` |
| `tests/visualization/test_raw_input_mse_histograms.py` | `tests/test_visualization_raw_input_mse_histograms.py` |
| `tests/core/test_contracts.py` | `tests/test_core_contracts.py` |
| `tests/core/test_uq_summary.py` | `tests/test_core_uq_summary.py` |
| `tests/core/test_evaluation_trace_compaction.py` | `tests/test_core_evaluation_trace_compaction.py` |
| `tests/core/test_artifact_naming.py` | `tests/test_core_artifact_naming.py` |
| `tests/experiments/test_direct_branch_routing_bridge.py` | `tests/test_experiments_direct_branch_routing_bridge.py` |
| `tests/contracts/test_online_runtime_contracts.py` | `tests/test_contracts_online_runtime_contracts.py` |
| `tests/contracts/test_configuration_data_contracts.py` | `tests/test_contracts_configuration_data_contracts.py` |
| `tests/contracts/test_model_baseline_contracts.py` | `tests/test_contracts_model_baseline_contracts.py` |
| `tests/contracts/test_evaluation_contracts.py` | `tests/test_contracts_evaluation_contracts.py` |
| `tests/contracts/test_engine_checkpoint_contracts.py` | `tests/test_contracts_engine_checkpoint_contracts.py` |
| `tests/contracts/test_public_release_contracts.py` | `tests/test_contracts_public_release_contracts.py` |
| `tests/runtime/test_console_instrumentation.py` | `tests/test_runtime_console_instrumentation.py` |
| `tests/runtime/test_learning_rate_scheduler_trainer.py` | `tests/test_runtime_learning_rate_scheduler_trainer.py` |
| `tests/runtime/test_raw_mse_training.py` | `tests/test_runtime_raw_mse_training.py` |
| `tests/runtime/test_kaggle_config_validation.py` | `tests/test_runtime_kaggle_config_validation.py` |
| `tests/runtime/test_artifact_sink_selection.py` | `tests/test_runtime_artifact_sink_selection.py` |
| `tests/runtime/test_learning_rate_scheduler.py` | `tests/test_runtime_learning_rate_scheduler.py` |
| `tests/runtime/test_checkpoint_roundtrip.py` | `tests/test_runtime_checkpoint_roundtrip.py` |
| `tests/runtime/test_logger_wandb.py` | `tests/test_runtime_logger_wandb.py` |
| `tests/runtime/test_learning_rate_scheduler_additional.py` | `tests/test_runtime_learning_rate_scheduler_additional.py` |
| `tests/runtime/test_trainer_checkpoint_fallback.py` | `tests/test_runtime_trainer_checkpoint_fallback.py` |
| `tests/models/test_redlamp_gradient_conflict_metrics.py` | `tests/test_models_redlamp_gradient_conflict_metrics.py` |
| `tests/models/test_multitask_metrics_runtime.py` | `tests/test_models_multitask_metrics_runtime.py` |
| `tests/models/test_registry.py` | `tests/test_models_registry.py` |
| `tests/models/test_redlamp_baseline_runtime.py` | `tests/test_models_redlamp_baseline_runtime.py` |
| `tests/models/test_thesis_multitask_classification_path_toggle.py` | `tests/test_models_thesis_multitask_classification_path_toggle.py` |
| `tests/models/test_one_multitask_train_step.py` | `tests/test_models_one_multitask_train_step.py` |
| `tests/models/test_one_redlamp_mlp_train_step.py` | `tests/test_models_one_redlamp_mlp_train_step.py` |
| `tests/models/test_exp2_two_view_cka.py` | `tests/test_models_exp2_two_view_cka.py` |
| `tests/models/test_redlamp_baseline_config_surface.py` | `tests/test_models_redlamp_baseline_config_surface.py` |
| `tests/models/test_traditional_baseline_contracts.py` | `tests/test_models_traditional_baseline_contracts.py` |
| `tests/models/test_thesis_multitask_cnn_shapes.py` | `tests/test_models_thesis_multitask_cnn_shapes.py` |
| `tests/models/test_redlamp_cnn_baseline_shapes.py` | `tests/test_models_redlamp_cnn_baseline_shapes.py` |
| `tests/models/test_thesis_multitask_point_score_loss.py` | `tests/test_models_thesis_multitask_point_score_loss.py` |
| `tests/models/test_neural_blocks.py` | `tests/test_models_neural_blocks.py` |
| `tests/models/test_redlamp_baseline_active_benchmark_config.py` | `tests/test_models_redlamp_baseline_active_benchmark_config.py` |
| `tests/models/test_one_train_step.py` | `tests/test_models_one_train_step.py` |
| `tests/models/test_redlamp_baseline_with_gradient_profiling_step.py` | `tests/test_models_redlamp_baseline_with_gradient_profiling_step.py` |
| `tests/models/test_multitask_memory_initialization.py` | `tests/test_models_multitask_memory_initialization.py` |
| `tests/models/test_fusion_ablation_modes.py` | `tests/test_models_fusion_ablation_modes.py` |
| `tests/models/test_multitask_shapes.py` | `tests/test_models_multitask_shapes.py` |
| `tests/models/test_optimizer_building.py` | `tests/test_models_optimizer_building.py` |
| `tests/models/test_multitask_validation_alignment.py` | `tests/test_models_multitask_validation_alignment.py` |
| `tests/models/test_multitask_shape_initialization.py` | `tests/test_models_multitask_shape_initialization.py` |
| `tests/models/test_raw_reconstruction_loss.py` | `tests/test_models_raw_reconstruction_loss.py` |
| `tests/models/test_multitask_memory_bootstrap.py` | `tests/test_models_multitask_memory_bootstrap.py` |
| `tests/models/test_moment_adapter_shapes.py` | `tests/test_models_moment_adapter_shapes.py` |
| `tests/models/test_multitask_objective_controls.py` | `tests/test_models_multitask_objective_controls.py` |
| `tests/models/test_model_shapes.py` | `tests/test_models_model_shapes.py` |
| `tests/models/test_redlamp_baseline.py` | `tests/test_models_redlamp_baseline.py` |
| `tests/models/test_thesis_multitask_gradient_profiling.py` | `tests/test_models_thesis_multitask_gradient_profiling.py` |
| `tests/models/test_temperature_schedule.py` | `tests/test_models_temperature_schedule.py` |
| `tests/models/test_thesis_multitask_config_refactor.py` | `tests/test_models_thesis_multitask_config_refactor.py` |
| `tests/models/test_direct_branch_routing.py` | `tests/test_models_direct_branch_routing.py` |
| `tests/models/test_multitask_memory_updates.py` | `tests/test_models_multitask_memory_updates.py` |
| `tests/compliance/test_codebase_compliance_scanner.py` | `tests/test_compliance_codebase_compliance_scanner.py` |
| `tests/compliance/test_src_refactor_contracts.py` | `tests/test_compliance_src_refactor_contracts.py` |
| `tests/benchmarks/test_benchmark_protocol_config.py` | `tests/test_benchmarks_benchmark_protocol_config.py` |
| `tests/benchmarks/test_evaluator_trace_payload.py` | `tests/test_benchmarks_evaluator_trace_payload.py` |
| `tests/benchmarks/test_benchmark_two_method_configs.py` | `tests/test_benchmarks_benchmark_two_method_configs.py` |
| `tests/benchmarks/test_two_stage_orchestration_dry_run.py` | `tests/test_benchmarks_two_stage_orchestration_dry_run.py` |
| `tests/benchmarks/test_thesis_offline_artifact_exports.py` | `tests/test_benchmarks_thesis_offline_artifact_exports.py` |
| `tests/benchmarks/test_run_offline_benchmark.py` | `tests/test_benchmarks_run_offline_benchmark.py` |
| `tests/benchmarks/test_smoke_wandb_logging.py` | `tests/test_benchmarks_smoke_wandb_logging.py` |
| `tests/benchmarks/test_full_spec_runtime_readiness.py` | `tests/test_benchmarks_full_spec_runtime_readiness.py` |
| `tests/benchmarks/test_thesis_online_benchmark_wrapper.py` | `tests/test_benchmarks_thesis_online_benchmark_wrapper.py` |
| `tests/benchmarks/test_summarize_benchmark_results.py` | `tests/test_benchmarks_summarize_benchmark_results.py` |
| `tests/evaluation/test_evaluation_metrics_audit.py` | `tests/test_evaluation_evaluation_metrics_audit.py` |
| `tests/evaluation/test_classification_diagnostics_metrics.py` | `tests/test_evaluation_classification_diagnostics_metrics.py` |
| `tests/evaluation/test_point_score_contracts.py` | `tests/test_evaluation_point_score_contracts.py` |
| `tests/evaluation/test_affiliation_metric.py` | `tests/test_evaluation_affiliation_metric.py` |
| `tests/evaluation/test_vus_pr_metric.py` | `tests/test_evaluation_vus_pr_metric.py` |
| `tests/evaluation/test_thresholding_helpers.py` | `tests/test_evaluation_thresholding_helpers.py` |
| `tests/evaluation/test_pointwise_range_metrics.py` | `tests/test_evaluation_pointwise_range_metrics.py` |
| `tests/evaluation/test_evaluation_visualization.py` | `tests/test_evaluation_evaluation_visualization.py` |
| `tests/evaluation/test_evaluator_thresholding.py` | `tests/test_evaluation_evaluator_thresholding.py` |
| `tests/evaluation/test_iforest_contract.py` | `tests/test_evaluation_iforest_contract.py` |
| `tests/evaluation/test_stumpy_channel_ab_contract.py` | `tests/test_evaluation_stumpy_channel_ab_contract.py` |
| `tests/evaluation/test_raw_input_mse_scores.py` | `tests/test_evaluation_raw_input_mse_scores.py` |
| `tests/evaluation/test_budgeted_vus_metrics.py` | `tests/test_evaluation_budgeted_vus_metrics.py` |
| `tests/evaluation/test_kmeansad_contract.py` | `tests/test_evaluation_kmeansad_contract.py` |
| `tests/evaluation/test_post_anomaly_suffix_export.py` | `tests/test_evaluation_post_anomaly_suffix_export.py` |
| `tests/evaluation/test_evaluation_protocol_audit.py` | `tests/test_evaluation_evaluation_protocol_audit.py` |
| `tests/evaluation/test_anomaly_span_length_summary.py` | `tests/test_evaluation_anomaly_span_length_summary.py` |
| `tests/evaluation/test_forensic_audit_markdown.py` | `tests/test_evaluation_forensic_audit_markdown.py` |
| `tests/ops/test_backfill_uq_summary.py` | `tests/test_ops_backfill_uq_summary.py` |
| `tests/ops/test_recalibrate_thesis_threshold_artifacts_v4.py` | `tests/test_ops_recalibrate_thesis_threshold_artifacts_v4.py` |
| `tests/ops/test_aggregate_online_table3_metrics.py` | `tests/test_ops_aggregate_online_table3_metrics.py` |
| `tests/online/test_online_stream.py` | `tests/test_online_online_stream.py` |
| `tests/online/test_artifact_integrity.py` | `tests/test_online_artifact_integrity.py` |
| `tests/online/test_redlamp_encoder_checkpoint.py` | `tests/test_online_redlamp_encoder_checkpoint.py` |
| `tests/online/test_online_calibration_contract.py` | `tests/test_online_online_calibration_contract.py` |
| `tests/online/test_online_timing_debug.py` | `tests/test_online_online_timing_debug.py` |
| `tests/online/test_online_verification_buffer.py` | `tests/test_online_online_verification_buffer.py` |
| `tests/online/test_online_stream_range.py` | `tests/test_online_online_stream_range.py` |
| `tests/online/test_online_adaptation_step.py` | `tests/test_online_online_adaptation_step.py` |
| `tests/online/test_online_streaming_benchmark_wrapper.py` | `tests/test_online_online_streaming_benchmark_wrapper.py` |
| `tests/online/test_online_reference_checkpoint.py` | `tests/test_online_online_reference_checkpoint.py` |
| `tests/online/test_online_tta_trainable_surface.py` | `tests/test_online_online_tta_trainable_surface.py` |
| `tests/online/test_full_spec_online_contract.py` | `tests/test_online_full_spec_online_contract.py` |
| `tests/online/test_online_entrypoint.py` | `tests/test_online_online_entrypoint.py` |
| `tests/online/test_entity_threshold_runtime.py` | `tests/test_online_entity_threshold_runtime.py` |
| `tests/online/test_online_signature_verification.py` | `tests/test_online_online_signature_verification.py` |
| `tests/online/test_online_state_roundtrip.py` | `tests/test_online_online_state_roundtrip.py` |
| `tests/online/test_online_engine_max_steps.py` | `tests/test_online_online_engine_max_steps.py` |
| `tests/online/test_online_runtime_state.py` | `tests/test_online_online_runtime_state.py` |
| `tests/online/test_online_ewma_threshold.py` | `tests/test_online_online_ewma_threshold.py` |
| `tests/online/test_online_streaming_baseline_contracts.py` | `tests/test_online_online_streaming_baseline_contracts.py` |
| `tests/online/test_online_tta_triage.py` | `tests/test_online_online_tta_triage.py` |
| `tests/online/test_threshold_artifact.py` | `tests/test_online_threshold_artifact.py` |
| `tests/online/test_full_spec_online_losses.py` | `tests/test_online_full_spec_online_losses.py` |
| `tests/online/test_online_method_fidelity.py` | `tests/test_online_online_method_fidelity.py` |
| `tests/online/test_online_prototype_metadata_contract.py` | `tests/test_online_online_prototype_metadata_contract.py` |
| `tests/online/test_verification_cycle.py` | `tests/test_online_verification_cycle.py` |
| `tests/online/test_online_tta_variants.py` | `tests/test_online_online_tta_variants.py` |
| `tests/data/test_synthetic_anomaly_injection.py` | `tests/test_data_synthetic_anomaly_injection.py` |
| `tests/data/test_synthetic_profile_comparison_plot.py` | `tests/test_data_synthetic_profile_comparison_plot.py` |
| `tests/data/test_split_protocol.py` | `tests/test_data_split_protocol.py` |
| `tests/data/test_smd_realistic_rate.py` | `tests/test_data_smd_realistic_rate.py` |
| `tests/data/test_public_data_api.py` | `tests/test_data_public_data_api.py` |
| `tests/data/test_anomaly_archive_kl.py` | `tests/test_data_anomaly_archive_kl.py` |
| `tests/data/test_smd_normal_drift_ranking.py` | `tests/test_data_smd_normal_drift_ranking.py` |
| `tests/data/test_smd_dataset_shapes.py` | `tests/test_data_smd_dataset_shapes.py` |
| `tests/data/test_data_cleaning_pipeline.py` | `tests/test_data_data_cleaning_pipeline.py` |
| `tests/data/test_nonoverlap_tail_windowing.py` | `tests/test_data_nonoverlap_tail_windowing.py` |
| `tests/data/test_anomaly_archive_dataset_loader.py` | `tests/test_data_anomaly_archive_dataset_loader.py` |
| `tests/data/test_loader_worker_resolution.py` | `tests/test_data_loader_worker_resolution.py` |
| `tests/data/test_synthetic_anomaly_visualization.py` | `tests/test_data_synthetic_anomaly_visualization.py` |
| `tests/data/test_sequence_standard_scaler_channel_mask.py` | `tests/test_data_sequence_standard_scaler_channel_mask.py` |
| `tests/data/test_windowizer.py` | `tests/test_data_windowizer.py` |
| `tests/data/test_smd_download_metadata.py` | `tests/test_data_smd_download_metadata.py` |
| `tests/data/test_smd_official_entity_data_configs.py` | `tests/test_data_smd_official_entity_data_configs.py` |
| `tests/data/test_smd_overlap_metadata_contract.py` | `tests/test_data_smd_overlap_metadata_contract.py` |
| `tests/data/test_smd_entity_visualization.py` | `tests/test_data_smd_entity_visualization.py` |
| `tests/data/test_synthetic_anomaly_visibility_profile.py` | `tests/test_data_synthetic_anomaly_visibility_profile.py` |
| `tests/engine/test_runtime_resource_monitor.py` | `tests/test_engine_runtime_resource_monitor.py` |
| `tests/engine/test_checkpoint_roundtrip.py` | `tests/test_engine_checkpoint_roundtrip.py` |
| `tests/engine/test_threshold_artifact.py` | `tests/test_engine_threshold_artifact.py` |
| `tests/compliance/fixtures/src_refactor_contracts.json` | `tests/fixtures/src_refactor_contracts.json` |

## Target Python tree

```text
demo/__init__.py
demo/app.py
demo/demo_state.py
demo/loaders.py
demo/offline_replay.py
demo/online_replay.py
demo/plotting.py
demo/stream_queue.py
scripts/__init__.py
scripts/_offline.py
scripts/_report_io.py
scripts/aggregate_online_table3_metrics.py
scripts/backfill_uq_summary.py
scripts/compare_synthetic_profiles.py
scripts/compare_synthetic_profiles_helpers.py
scripts/compare_synthetic_profiles_shared.py
scripts/evaluate.py
scripts/export_post_anomaly_suffix_counts.py
scripts/forensic_audit_run.py
scripts/kl_divergence_anomaly_archive.py
scripts/plot_raw_input_mse_histograms.py
scripts/prune_raw_trace_artifacts.py
scripts/rank_smd_train_test_normal_drift.py
scripts/recalibrate_thesis_threshold_artifacts_v4.py
scripts/run_ablation.py
scripts/run_offline_benchmark.py
scripts/run_online_adaptation.py
scripts/run_online_streaming_benchmark.py
scripts/run_thesis_offline_benchmark.py
scripts/run_thesis_online_benchmark.py
scripts/run_two_stage_offline_pretraining.py
scripts/summarize_anomaly_span_lengths.py
scripts/summarize_anomaly_span_lengths_helpers.py
scripts/summarize_benchmark_results.py
scripts/train.py
scripts/visualize_classification_diagnostics.py
scripts/visualize_evaluation_results.py
scripts/visualize_smd_benchmark_train_test.py
scripts/visualize_smd_train_test_entity.py
scripts/visualize_synthetic_anomalies.py
scripts/visualize_training_metrics.py
src/artifact_integrity.py
src/artifact_naming.py
src/artifact_sinks.py
src/checkpoint.py
src/config.py
src/config_aliases.py
src/config_experiment_validation.py
src/config_help.py
src/config_model_validation.py
src/console.py
src/contracts.py
src/data/__init__.py
src/data/anomaly_archive.py
src/data/api.py
src/data/augment.py
src/data/base.py
src/data/cleaning.py
src/data/collate.py
src/data/download.py
src/data/loaders.py
src/data/public_types.py
src/data/scalers.py
src/data/smd.py
src/data/split_protocol.py
src/data/stream.py
src/data/window.py
src/evaluation/affiliation.py
src/evaluation/anomaly_archive_kl.py
src/evaluation/classification_diagnostics.py
src/evaluation/evaluation_protocol_audit.py
src/evaluation/online_contract.py
src/evaluation/online_stream_range.py
src/evaluation/point_score_calibration.py
src/evaluation/point_scores.py
src/evaluation/pointwise.py
src/evaluation/reconstruction_scores.py
src/evaluation/smd_benchmark_protocol.py
src/evaluation/synthetic_profile.py
src/evaluation/threshold_artifact.py
src/evaluation/threshold_artifact_v4_online_scoring.py
src/evaluation_trace_compaction.py
src/evaluator.py
src/logger.py
src/models/adaptive_baselines.py
src/models/base_model.py
src/models/candi.py
src/models/encoder_adapter.py
src/models/frozen_baselines.py
src/models/iforest.py
src/models/kmeans_ad.py
src/models/m2n2.py
src/models/moment.py
src/models/neural_blocks.py
src/models/online_adaptation.py
src/models/online_adapter_modules.py
src/models/online_baselines.py
src/models/online_encoder.py
src/models/online_redlamp_reconstruction.py
src/models/pre_tta.py
src/models/redlamp_baseline.py
src/models/redlamp_encoder_checkpoint.py
src/models/redlamp_gradient_metrics.py
src/models/simple_window_cnn_autoencoder.py
src/models/stumpy_channel_ab.py
src/models/thesis/__init__.py
src/models/thesis/checkpoint.py
src/models/thesis/components.py
src/models/thesis/forward.py
src/models/thesis/geometry.py
src/models/thesis/gradient_metrics.py
src/models/thesis/initialization.py
src/models/thesis/losses.py
src/models/thesis/memory.py
src/models/thesis/parse_config.py
src/models/thesis/routing.py
src/models/thesis/sampling.py
src/models/thesis/schedule.py
src/models/thesis/setup.py
src/models/thesis/steps.py
src/models/thesis_multitask.py
src/models/window_baselines.py
src/online/__init__.py
src/online/checkpoint_resolution.py
src/online/non_overlap_guard.py
src/online/online_calibration.py
src/online/online_engine.py
src/online/online_engine_run.py
src/online/online_engine_shared.py
src/online/online_engine_step.py
src/online/online_engine_window_core.py
src/online/online_engine_window_metrics.py
src/online/online_losses.py
src/online/online_optimizer.py
src/online/point_ewma.py
src/online/runtime_state.py
src/online/signature_verification.py
src/online/timing_debug.py
src/online/triage.py
src/online/verification_adapter.py
src/online/verification_buffer.py
src/online/verification_cycle.py
src/online_loop.py
src/registry.py
src/runtime_components.py
src/runtime_resource_monitor.py
src/seed.py
src/thresholding.py
src/trainer.py
src/uq_summary.py
tests/codebase_compliance.py
tests/conftest.py
tests/test_benchmarks_benchmark_protocol_config.py
tests/test_benchmarks_benchmark_two_method_configs.py
tests/test_benchmarks_evaluator_trace_payload.py
tests/test_benchmarks_full_spec_runtime_readiness.py
tests/test_benchmarks_run_offline_benchmark.py
tests/test_benchmarks_smoke_wandb_logging.py
tests/test_benchmarks_summarize_benchmark_results.py
tests/test_benchmarks_thesis_offline_artifact_exports.py
tests/test_benchmarks_thesis_online_benchmark_wrapper.py
tests/test_benchmarks_two_stage_orchestration_dry_run.py
tests/test_compliance_codebase_compliance_scanner.py
tests/test_compliance_src_refactor_contracts.py
tests/test_contracts_configuration_data_contracts.py
tests/test_contracts_engine_checkpoint_contracts.py
tests/test_contracts_evaluation_contracts.py
tests/test_contracts_model_baseline_contracts.py
tests/test_contracts_online_runtime_contracts.py
tests/test_contracts_public_release_contracts.py
tests/test_core_artifact_naming.py
tests/test_core_contracts.py
tests/test_core_evaluation_trace_compaction.py
tests/test_core_uq_summary.py
tests/test_data_anomaly_archive_dataset_loader.py
tests/test_data_anomaly_archive_kl.py
tests/test_data_data_cleaning_pipeline.py
tests/test_data_loader_worker_resolution.py
tests/test_data_nonoverlap_tail_windowing.py
tests/test_data_public_data_api.py
tests/test_data_sequence_standard_scaler_channel_mask.py
tests/test_data_smd_dataset_shapes.py
tests/test_data_smd_download_metadata.py
tests/test_data_smd_entity_visualization.py
tests/test_data_smd_normal_drift_ranking.py
tests/test_data_smd_official_entity_data_configs.py
tests/test_data_smd_overlap_metadata_contract.py
tests/test_data_smd_realistic_rate.py
tests/test_data_split_protocol.py
tests/test_data_synthetic_anomaly_injection.py
tests/test_data_synthetic_anomaly_visibility_profile.py
tests/test_data_synthetic_anomaly_visualization.py
tests/test_data_synthetic_profile_comparison_plot.py
tests/test_data_windowizer.py
tests/test_demo_demo_app.py
tests/test_demo_demo_state.py
tests/test_demo_demo_stream_queue.py
tests/test_demo_live_online_replay.py
tests/test_engine_checkpoint_roundtrip.py
tests/test_engine_runtime_resource_monitor.py
tests/test_engine_threshold_artifact.py
tests/test_evaluation_affiliation_metric.py
tests/test_evaluation_anomaly_span_length_summary.py
tests/test_evaluation_budgeted_vus_metrics.py
tests/test_evaluation_classification_diagnostics_metrics.py
tests/test_evaluation_evaluation_metrics_audit.py
tests/test_evaluation_evaluation_protocol_audit.py
tests/test_evaluation_evaluation_visualization.py
tests/test_evaluation_evaluator_thresholding.py
tests/test_evaluation_forensic_audit_markdown.py
tests/test_evaluation_iforest_contract.py
tests/test_evaluation_kmeansad_contract.py
tests/test_evaluation_point_score_contracts.py
tests/test_evaluation_pointwise_range_metrics.py
tests/test_evaluation_post_anomaly_suffix_export.py
tests/test_evaluation_raw_input_mse_scores.py
tests/test_evaluation_stumpy_channel_ab_contract.py
tests/test_evaluation_thresholding_helpers.py
tests/test_evaluation_vus_pr_metric.py
tests/test_experiments_direct_branch_routing_bridge.py
tests/test_models_direct_branch_routing.py
tests/test_models_exp2_two_view_cka.py
tests/test_models_fusion_ablation_modes.py
tests/test_models_model_shapes.py
tests/test_models_moment_adapter_shapes.py
tests/test_models_multitask_memory_bootstrap.py
tests/test_models_multitask_memory_initialization.py
tests/test_models_multitask_memory_updates.py
tests/test_models_multitask_metrics_runtime.py
tests/test_models_multitask_objective_controls.py
tests/test_models_multitask_shape_initialization.py
tests/test_models_multitask_shapes.py
tests/test_models_multitask_validation_alignment.py
tests/test_models_neural_blocks.py
tests/test_models_one_multitask_train_step.py
tests/test_models_one_redlamp_mlp_train_step.py
tests/test_models_one_train_step.py
tests/test_models_optimizer_building.py
tests/test_models_raw_reconstruction_loss.py
tests/test_models_redlamp_baseline.py
tests/test_models_redlamp_baseline_active_benchmark_config.py
tests/test_models_redlamp_baseline_config_surface.py
tests/test_models_redlamp_baseline_runtime.py
tests/test_models_redlamp_baseline_with_gradient_profiling_step.py
tests/test_models_redlamp_cnn_baseline_shapes.py
tests/test_models_redlamp_gradient_conflict_metrics.py
tests/test_models_registry.py
tests/test_models_temperature_schedule.py
tests/test_models_thesis_multitask_classification_path_toggle.py
tests/test_models_thesis_multitask_cnn_shapes.py
tests/test_models_thesis_multitask_config_refactor.py
tests/test_models_thesis_multitask_gradient_profiling.py
tests/test_models_thesis_multitask_point_score_loss.py
tests/test_models_traditional_baseline_contracts.py
tests/test_online_artifact_integrity.py
tests/test_online_entity_threshold_runtime.py
tests/test_online_full_spec_online_contract.py
tests/test_online_full_spec_online_losses.py
tests/test_online_online_adaptation_step.py
tests/test_online_online_calibration_contract.py
tests/test_online_online_engine_max_steps.py
tests/test_online_online_entrypoint.py
tests/test_online_online_ewma_threshold.py
tests/test_online_online_method_fidelity.py
tests/test_online_online_prototype_metadata_contract.py
tests/test_online_online_reference_checkpoint.py
tests/test_online_online_runtime_state.py
tests/test_online_online_signature_verification.py
tests/test_online_online_state_roundtrip.py
tests/test_online_online_stream.py
tests/test_online_online_stream_range.py
tests/test_online_online_streaming_baseline_contracts.py
tests/test_online_online_streaming_benchmark_wrapper.py
tests/test_online_online_timing_debug.py
tests/test_online_online_tta_trainable_surface.py
tests/test_online_online_tta_triage.py
tests/test_online_online_tta_variants.py
tests/test_online_online_verification_buffer.py
tests/test_online_redlamp_encoder_checkpoint.py
tests/test_online_threshold_artifact.py
tests/test_online_verification_cycle.py
tests/test_ops_aggregate_online_table3_metrics.py
tests/test_ops_backfill_uq_summary.py
tests/test_ops_recalibrate_thesis_threshold_artifacts_v4.py
tests/test_runtime_artifact_sink_selection.py
tests/test_runtime_checkpoint_roundtrip.py
tests/test_runtime_console_instrumentation.py
tests/test_runtime_kaggle_config_validation.py
tests/test_runtime_learning_rate_scheduler.py
tests/test_runtime_learning_rate_scheduler_additional.py
tests/test_runtime_learning_rate_scheduler_trainer.py
tests/test_runtime_logger_wandb.py
tests/test_runtime_raw_mse_training.py
tests/test_runtime_trainer_checkpoint_fallback.py
tests/test_visualization_raw_input_mse_histograms.py
```

## Final YAML migration

| Old file | New file |
|---|---|
| `configs/data/smd_benchmark_machine_1_6_window20.yaml` | `configs/data__smd_benchmark_machine_1_6_window20.yaml` |
| `configs/data/smd_benchmark_machine_3_4_window20.yaml` | `configs/data__smd_benchmark_machine_3_4_window20.yaml` |
| `configs/data/smd_benchmark_machine_3_9_window20.yaml` | `configs/data__smd_benchmark_machine_3_9_window20.yaml` |
| `configs/data/smd_rtx3090_machine_1_6_20_stride1.yaml` | `configs/data__smd_rtx3090_machine_1_6_20_stride1.yaml` |
| `configs/data/smd_rtx3090_machine_3_1_20_stride1.yaml` | `configs/data__smd_rtx3090_machine_3_1_20_stride1.yaml` |
| `configs/data/smd_rtx3090_machine_3_9_20_stride1.yaml` | `configs/data__smd_rtx3090_machine_3_9_20_stride1.yaml` |
| `configs/data/smd_smoke.yaml` | `configs/data__smd_smoke.yaml` |
| `configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_1_6__w20__seed36__main.yaml` | `configs/experiment__smd__redlamp_baseline__benchmark-machine_1_6__w20__seed36__main.yaml` |
| `configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_1_6__w20__seed6__main.yaml` | `configs/experiment__smd__redlamp_baseline__benchmark-machine_1_6__w20__seed6__main.yaml` |
| `configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_1_6__w20__seed8__main.yaml` | `configs/experiment__smd__redlamp_baseline__benchmark-machine_1_6__w20__seed8__main.yaml` |
| `configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_4__w20__seed36__main.yaml` | `configs/experiment__smd__redlamp_baseline__benchmark-machine_3_4__w20__seed36__main.yaml` |
| `configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_4__w20__seed6__main.yaml` | `configs/experiment__smd__redlamp_baseline__benchmark-machine_3_4__w20__seed6__main.yaml` |
| `configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_4__w20__seed8__main.yaml` | `configs/experiment__smd__redlamp_baseline__benchmark-machine_3_4__w20__seed8__main.yaml` |
| `configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_9__w20__seed36__main.yaml` | `configs/experiment__smd__redlamp_baseline__benchmark-machine_3_9__w20__seed36__main.yaml` |
| `configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_9__w20__seed6__main.yaml` | `configs/experiment__smd__redlamp_baseline__benchmark-machine_3_9__w20__seed6__main.yaml` |
| `configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_9__w20__seed8__main.yaml` | `configs/experiment__smd__redlamp_baseline__benchmark-machine_3_9__w20__seed8__main.yaml` |
| `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_1_6__w20__seed36__main.yaml` | `configs/experiment__smd__thesis_multitask__benchmark-two-stage-machine_1_6__w20__seed36__main.yaml` |
| `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_1_6__w20__seed6__main.yaml` | `configs/experiment__smd__thesis_multitask__benchmark-two-stage-machine_1_6__w20__seed6__main.yaml` |
| `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_1_6__w20__seed8__main.yaml` | `configs/experiment__smd__thesis_multitask__benchmark-two-stage-machine_1_6__w20__seed8__main.yaml` |
| `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_4__w20__seed36__main.yaml` | `configs/experiment__smd__thesis_multitask__benchmark-two-stage-machine_3_4__w20__seed36__main.yaml` |
| `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_4__w20__seed6__main.yaml` | `configs/experiment__smd__thesis_multitask__benchmark-two-stage-machine_3_4__w20__seed6__main.yaml` |
| `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_4__w20__seed8__main.yaml` | `configs/experiment__smd__thesis_multitask__benchmark-two-stage-machine_3_4__w20__seed8__main.yaml` |
| `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_9__w20__seed36__main.yaml` | `configs/experiment__smd__thesis_multitask__benchmark-two-stage-machine_3_9__w20__seed36__main.yaml` |
| `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_9__w20__seed6__main.yaml` | `configs/experiment__smd__thesis_multitask__benchmark-two-stage-machine_3_9__w20__seed6__main.yaml` |
| `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_9__w20__seed8__main.yaml` | `configs/experiment__smd__thesis_multitask__benchmark-two-stage-machine_3_9__w20__seed8__main.yaml` |
| `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_1_6__w20__seed36__main.yaml` | `configs/experiment__smd__thesis_multitask__benchmark-two-stage-point-score-machine_1_6__w20__seed36__main.yaml` |
| `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_1_6__w20__seed6__main.yaml` | `configs/experiment__smd__thesis_multitask__benchmark-two-stage-point-score-machine_1_6__w20__seed6__main.yaml` |
| `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_1_6__w20__seed8__main.yaml` | `configs/experiment__smd__thesis_multitask__benchmark-two-stage-point-score-machine_1_6__w20__seed8__main.yaml` |
| `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_4__w20__seed36__main.yaml` | `configs/experiment__smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_4__w20__seed36__main.yaml` |
| `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_4__w20__seed6__main.yaml` | `configs/experiment__smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_4__w20__seed6__main.yaml` |
| `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_4__w20__seed8__main.yaml` | `configs/experiment__smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_4__w20__seed8__main.yaml` |
| `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_9__w20__seed36__main.yaml` | `configs/experiment__smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_9__w20__seed36__main.yaml` |
| `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_9__w20__seed6__main.yaml` | `configs/experiment__smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_9__w20__seed6__main.yaml` |
| `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_9__w20__seed8__main.yaml` | `configs/experiment__smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_9__w20__seed8__main.yaml` |
| `configs/experiment/benchmark_smoke/thesis/smd__thesis_multitask__benchmark-two-stage-machine_1_6__w20__seed6__smoke.yaml` | `configs/experiment__smd__thesis_multitask__benchmark-two-stage-machine_1_6__w20__seed6__smoke.yaml` |
| `configs/experiment/comparative/baseline/smd__redlamp_baseline__comparative-single-stage-machine_1_6__w20__seed6__smoke.yaml` | `configs/experiment__smd__redlamp_baseline__comparative-single-stage-machine_1_6__w20__seed6__smoke.yaml` |
| `configs/experiment/demo/demo_offline_replay.yaml` | `configs/experiment__demo_offline_replay.yaml` |
| `configs/experiment/demo/demo_online_stream.yaml` | `configs/experiment__demo_online_stream.yaml` |
| `configs/experiment/offline_ablation/thesis/smd__thesis__offline__direct_branch_routing__machine_1_6__w20__seed6__stage_b.yaml` | `configs/experiment__smd__thesis__offline__direct_branch_routing__machine_1_6__w20__seed6__stage_b.yaml` |
| `configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O0__machine_1_6__w20__seed6__main.yaml` | `configs/experiment__smd__thesis__offline__O0__machine_1_6__w20__seed6__main.yaml` |
| `configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O1__machine_1_6__w20__seed6__main.yaml` | `configs/experiment__smd__thesis__offline__O1__machine_1_6__w20__seed6__main.yaml` |
| `configs/experiment/online_benchmark/thesis/smd__thesis__online__O0_A0__machine_1_6__w20__seed6__main.yaml` | `configs/experiment__smd__thesis__online__O0_A0__machine_1_6__w20__seed6__main.yaml` |
| `configs/experiment/online_benchmark/thesis/smd__thesis__online__O0_A0__machine_1_6__w20__seed6__smoke.yaml` | `configs/experiment__smd__thesis__online__O0_A0__machine_1_6__w20__seed6__smoke.yaml` |
| `configs/experiment/online_diagnostic/thesis/smd__thesis__online__O0_A0__machine_1_6__w20__seed6__timing_5608_5909.yaml` | `configs/experiment__smd__thesis__online__O0_A0__machine_1_6__w20__seed6__timing_5608_5909.yaml` |
| `configs/model/online_adaptation.yaml` | `configs/model__online_adaptation.yaml` |
| `configs/model/online_adaptation_smoke_o0_machine_1_6_seed6.yaml` | `configs/model__online_adaptation_smoke_o0_machine_1_6_seed6.yaml` |
| `configs/model/redlamp_baseline.yaml` | `configs/model__redlamp_baseline.yaml` |
| `configs/model/redlamp_baseline_comparative_smd.yaml` | `configs/model__redlamp_baseline_comparative_smd.yaml` |
| `configs/model/redlamp_cnn_baseline.yaml` | `configs/model__redlamp_cnn_baseline.yaml` |
| `configs/model/thesis_multitask.yaml` | `configs/model__thesis_multitask.yaml` |
| `configs/model/thesis_multitask_redlamp_multiclass.yaml` | `configs/model__thesis_multitask_redlamp_multiclass.yaml` |
| `configs/model/thesis_multitask_redlamp_multiclass_cnn_simple.yaml` | `configs/model__thesis_multitask_redlamp_multiclass_cnn_simple.yaml` |
| `configs/model/thesis_multitask_redlamp_multiclass_redlamp_aligned.yaml` | `configs/model__thesis_multitask_redlamp_multiclass_redlamp_aligned.yaml` |
| `configs/model/thesis_multitask_two_stage_comparative_smd.yaml` | `configs/model__thesis_multitask_two_stage_comparative_smd.yaml` |
| `configs/model/thesis_multitask_two_stage_point_score_window20.yaml` | `configs/model__thesis_multitask_two_stage_point_score_window20.yaml` |
| `configs/model/thesis_multitask_two_stage_point_score_window20_recon075_cls025.yaml` | `configs/model__thesis_multitask_two_stage_point_score_window20_recon075_cls025.yaml` |
| `configs/model/thesis_multitask_two_stage_window20.yaml` | `configs/model__thesis_multitask_two_stage_window20.yaml` |
| `configs/protocol/smd_window20_cleanval_q95_ewma09.yaml` | `configs/protocol__smd_window20_cleanval_q95_ewma09.yaml` |
| `configs/protocol/smd_window20_cleanval_q995_ewma09.yaml` | `configs/protocol__smd_window20_cleanval_q995_ewma09.yaml` |
| `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml` | `configs/protocol__smd_window20_cleanval_q99_ewma09.yaml` |
| `configs/protocol/smd_window20_synthnormal_q99_ewma09.yaml` | `configs/protocol__smd_window20_synthnormal_q99_ewma09.yaml` |
| `configs/protocol/smd_window20_synthnormal_q99_normalized_input_mse_ewma09.yaml` | `configs/protocol__smd_window20_synthnormal_q99_normalized_input_mse_ewma09.yaml` |
| `configs/protocol/synthetic_redlamp12_legacy_window20.yaml` | `configs/protocol__synthetic_redlamp12_legacy_window20.yaml` |
| `configs/protocol/synthetic_redlamp12_visible_window20.yaml` | `configs/protocol__synthetic_redlamp12_visible_window20.yaml` |
| `configs/task/multitask_tsad.yaml` | `configs/task__multitask_tsad.yaml` |
| `configs/task/multitask_tsad_redlamp_multiclass_window20.yaml` | `configs/task__multitask_tsad_redlamp_multiclass_window20.yaml` |
| `configs/task/multitask_tsad_redlamp_multiclass_window20_balanced.yaml` | `configs/task__multitask_tsad_redlamp_multiclass_window20_balanced.yaml` |
| `configs/task/multitask_tsad_redlamp_multiclass_window20_benchmark_fixed_synth.yaml` | `configs/task__multitask_tsad_redlamp_multiclass_window20_benchmark_fixed_synth.yaml` |
| `configs/task/multitask_tsad_redlamp_multiclass_window20_comparative.yaml` | `configs/task__multitask_tsad_redlamp_multiclass_window20_comparative.yaml` |
| `configs/task/multitask_tsad_window10_binary.yaml` | `configs/task__multitask_tsad_window10_binary.yaml` |
| `configs/task/online_adaptation.yaml` | `configs/task__online_adaptation.yaml` |
| `configs/task/online_adaptation_smoke_o0_a2_machine_1_6_seed6.yaml` | `configs/task__online_adaptation_smoke_o0_a2_machine_1_6_seed6.yaml` |

## Final stopping point

The user requested ending work. See `../detail/simplification-result.md` for the measured result and unmet 90% target. Earlier broad retirement manifests were superseded: research/reference assets were restored.
