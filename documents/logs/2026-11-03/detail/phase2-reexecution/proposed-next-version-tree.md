# Proposed next-version directory tree

Date: 2026-11-03  
Status: design proposal based on the source and script audit. No source files are moved or removed here.

## Scope

Keep THESIS/MUSTARD, all Phase 1 comparison methods (RedLamp, STUMPY Channel AB, KMeansAD, Isolation Forest, CANDI, M2N2, and MOMENT), offline training/evaluation, online test-time adaptation with evaluation, config generation, launchers, checkpoint/calibration/resume handling, and report collection. MOMENT is a method. `O0/O1/O2` and `A0/A1/A2` are THESIS/MUSTARD ablations only.

This tree groups ownership and names audited files. It does not claim the 40-file, 4,000-line, or 100-lines-per-file targets are met. The audit shows the current-matrix script set alone is 6,672 lines. The standalone `run_online_adaptation.py` entrypoint is omitted because the required THESIS online path is already represented by `run_thesis_online_experiment.py`; confirm there are no distinct documented callers before retirement. Other file actions below still require caller and behavior checks.

## Proposed tree

```text
bachelor-thesis-2026/
├── configs/                              # Keep data, model, and experiment configs
├── data/                                 # Keep datasets and preprocessing inputs
├── documents/                            # Thesis, method math, specs, and research records
├── bsc-thesis-ref-codebases/             # Preserve reference implementations and licenses
├── src/
│   ├── data/                             # Dataset loading and transforms
│   │   ├── datasets/                     # Existing dataset modules
│   │   └── [existing preprocessing modules]
│   ├── methods/                          # One entrypoint per scientific method
│   │   ├── thesis_multitask.py            # THESIS runtime implementing MUSTARD
│   │   ├── redlamp.py
│   │   ├── stumpy_channel_ab.py
│   │   ├── kmeans_ad.py
│   │   ├── iforest.py
│   │   ├── candi.py
│   │   ├── m2n2.py
│   │   ├── moment.py                     # MOMENT method; no adapter identity
│   │   └── online_adaptation.py          # MUSTARD online test-time adaptation
│   ├── runtime/                          # Shared contracts and execution machinery
│   │   ├── config.py                     # Existing core config/validation ownership
│   │   ├── contracts.py
│   │   ├── engine/                       # Existing src/engine modules
│   │   ├── online_tta/                   # Online ordering, state, calibration, resume
│   │   └── artifacts/                    # Checkpoint and artifact integrity/naming
│   └── evaluation/                       # Scoring, metrics, protocols, analysis
│       ├── metrics/                      # Existing metrics modules
│       ├── protocols/                    # Existing protocol modules
│       └── analysis/                     # Reusable evaluation protocol audit
├── scripts/
│   ├── cli/
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── experiments/                      # Experiment config generation, runs, and reports
│   │   ├── run_two_stage_offline_pretraining.py
│   │   ├── run_ablation.py
│   │   ├── run_comparative_smd_experiments.py
│   │   └── run_multiseed_experiment.py
│   │   ├── generate_offline_experiment_configs.py
│   │   ├── generate_smd_experiment_configs.py
│   │   ├── generate_online_streaming_experiment_configs.py
│   │   ├── generate_remaining_smd_experiment_configs.py
│   │   ├── run_offline_experiment.py
│   │   ├── run_online_streaming_experiment.py
│   │   ├── run_thesis_offline_experiment.py
│   │   ├── run_thesis_online_experiment.py
│   │   ├── collect_remaining_smd_experiment_metrics.py
│   │   └── summarize_experiment_results.py
│   ├── ops/                              # Required checkpoint/calibration/report tools
│   │   └── threshold_artifact_v4_online_scoring.py
│   ├── launchers/                        # One canonical matrix launcher
│   ├── analysis/                         # Retain only selected analysis workflows
│   └── visualization/                    # Retain synthetic anomaly inspection workflow
├── tests/                                # Keep tests for every retained contract and method
└── pyproject.toml                        # Project tooling and dependencies
```

Bracketed entries describe existing groups whose exact filenames remain unchanged until a file-action map is approved. The tree intentionally shows the audited runtime surface instead of repeating every retained document, config, test, and reference file.

The experiment filenames shown here are proposed target names. Rename current `scripts/benchmarks/` paths and migrate their callers as part of implementation.

## File-level consolidation and retirement candidates

| Current file(s) | Proposed next-version action | Guard before applying |
|---|---|---|
| `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py`; `src/models/thesis_multitask_impl/thesis_multitask_routing_mixin.py` | Keep one sample-variance implementation; preserve the existing method alias if callers need it. | Compare validation and correction behavior; run focused tests. |
| `src/models/thesis_multitask_impl/thesis_multitask_state_memory_mixin.py`; routing modules | Share memory-vector normalization and compute the identical Stage B predicate once. | Preserve the two lifecycle output fields and all O/A ablations. |
| `scripts/analysis/summarize_anomaly_span_lengths.py`; `scripts/analysis/summarize_anomaly_span_lengths_helpers.py` | Keep helper bodies in the helper module and re-export public CLI names. | Preserve test imports, command behavior, and report output. |
| Current: `scripts/benchmarks/run_thesis_offline_benchmark.py`; `scripts/benchmarks/_internal/run_thesis_offline_benchmark_helpers.py` | Keep the active retention-bundle export and three writers; remove only verified uncalled helper workflows. | Preserve runner imports, test patch points, and checkpoint/report contents. |
| Current: `scripts/experiments/run_online_adaptation.py` | Omit the standalone entrypoint from the proposed tree; use the retained THESIS online experiment runner. | Before implementation, confirm the standalone command has no unique caller or behavior beyond the required online test-time adaptation and evaluation flow. |
| `src/baselines/neural/redlamp.py`; RedLamp helper modules | Remove the four unused copied helper definitions; keep active class methods. | Confirm references and preserve RedLamp experiment behavior. |
| `src/models/thesis_multitask_impl/`; model config and focused tests | Candidate removal: optional regularizers, CKA-gated fusion, model-summary diagnostic, and gradient-conflict profiling. | These are feature changes. Confirm final experiment scope and migrate configs/tests before removal. |
| Current root-level `scripts/*.py` forwarding aliases and duplicate launchers | Omit forwarding copies from the tree; keep the canonical `scripts/cli/` and `scripts/experiments/` owners. | The audit says aliases are imported/monkeypatched by tests and named in docs; migrate those references before source deletion. |
| `scripts/analysis/`, `scripts/ops/`, `scripts/visualization/` | Keep only selected report, calibration, and synthetic-anomaly inspection workflows. | Preserve official thesis outputs and documented artifact contracts. |
| 270 non-THESIS `A0/A1/A2` YAML profiles | Retire as described in the pruning proposal; reject these inputs for baselines. | Keep THESIS A-label profiles and historical result identities unchanged. |

Do not merge method implementations into one generic model. Keep method-specific scoring and calibration. Keep baseline profiles separate from THESIS ablations. Make source-count reductions from verified file actions only; do not count directory moves, tests, configs, or documentation as source savings.

## Open evidence

- The standard THESIS O2 lifecycle and matching checkpoint path remain unresolved; do not infer them from the direct-branch helper.
- MOMENT has no runtime implementation in this checkout. Its mathematical note does not establish runtime parity.
- The static audit did not verify external callers, artifact-consumer equivalence, or scientific-output parity.
- The exact baseline selection is taken from the later user decision in the session log; the earlier pruning proposal's “unresolved” sentence is stale.
