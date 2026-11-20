# Next-version runtime tree

Date: 2026-11-16
Updated: 2026-11-20

Status: Installed runtime and source cleanup checked on bounded SMD runs.
Project: `bachelor-thesis-2026`.

See the [runtime signatures](../detail/proposed-runtime-signatures.md) for the interface and the [execution record](../detail/runtime-replacement-review.md) for verified behavior.

The two MUSTARD q95 evaluation profiles use the offline threshold reuse rules in the [runtime profile contract](../detail/runtime-profile-contract.md#mustard-validation-q95-comparison).

## Confirmed runtime boundary: 2026-11-16

Follow the [runtime replacement decisions](../plan/decisions-runtime-replacement.md).

1. `encoders.py` owns one time-preserving CNN implementation shared by MUSTARD, RedLamp, CANDI, and M2N2. Each method owns separate weights, pooling, heads, and update state. CANDI's SANA stays method-specific; MOMENT retains its pretrained encoder.
2. Runtime loading reads prepared data without new splits, filling, or scaling. Keep the self-contained preprocessing script and active `scripts/cleandocs_hook.py`.
3. Preserve core method calculations where the confirmed decisions permit. Document departures from author code. RedLamp uses validation-fixed scoring and causal smoothing.
4. The agreed result hierarchy replaces the older depth maximum. New commands and checkpoints replace old interfaces. Remove each old implementation only after its replacement checks pass.

## 1. Design rules

1. Keep `scripts/preprocess.py` self-contained. It owns source reading, sorting, splitting, missing-value handling, scaling, and prepared-data versioning.
2. Read prepared data during method execution. Do not split it again or fit another input scaler.
3. Keep the four profile types and four protocol contexts. Profiles supply settings. Protocols check inputs and outputs.
4. Keep MUSTARD, its offline and online variants, and all seven comparison methods from the earlier proposal.
5. Keep one file for each small workflow. Keep focused helpers where the scientific method needs them.
6. Cleanup of confirmed unnecessary paths is complete. Remove further current code only after its required replacement is implemented and checked. Protect all specifications and detailed mathematics files, including files outside the usual document folders.

## 2. File-level tree

This tree lists the maintained runtime files. See the [cleanup map](../../2026-11-20/detail/runtime-cleanup-paths.csv) for retired paths. Dataset file names below use entity templates. Other protected specifications and detailed mathematics files are not expanded here.

```text
bachelor-thesis-2026/
├── README.md                         # Setup and main commands
├── AGENTS.md
├── codebase-agnostic-preferences.md
├── codebase-specific-preferences.md
├── python_semantics.md
├── MAIN_FOLDERS.md
├── LICENSE
├── .gitignore                        # Include .codex/ and generated outputs
├── pyproject.toml                    # Thesis package and installed tsad entrypoint
├── scripts/
│   ├── __init__.py
│   ├── cleandocs_hook.py             # Active report-only documentation hook
│   └── preprocess.py                 # Existing self-contained preprocessing
├── configs/
│   ├── data_profiles/
│   │   ├── iccad.yaml
│   │   ├── smd.yaml
│   │   ├── iops.yaml
│   │   ├── swat.yaml
│   │   ├── nasa.yaml
│   │   └── synthetic_profiles/
│   │       ├── redlamp_legacy.yaml
│   │       └── redlamp_visible.yaml
│   ├── method_profiles/
│   │   ├── mustard_o0_offline.yaml
│   │   ├── mustard_o1_offline.yaml
│   │   ├── mustard_o2_offline.yaml
│   │   ├── mustard_a0_online.yaml
│   │   ├── mustard_a1_online.yaml
│   │   ├── mustard_a2_online.yaml
│   │   └── baselines/
│   │       ├── redlamp.yaml
│   │       ├── stumpy_channel_ab.yaml
│   │       ├── kmeans_ad.yaml
│   │       ├── iforest.yaml
│   │       ├── candi.yaml
│   │       ├── m2n2.yaml
│   │       ├── moment.yaml
│   │       └── moment_linear_probing.yaml
│   ├── evaluation_profiles/
│   │   ├── raw_input_q99_ewma075.yaml
│   │   ├── native_q99_ewma075.yaml
│   │   ├── mustard_raw_input_q95_ewma075.yaml  # Raw MSE
│   │   ├── mustard_latent_q95_ewma075.yaml     # Latent MSE
│   │   └── smd/
│   │       ├── cleanval_q99_ewma09.yaml
│   │       ├── cleanval_q995_ewma09.yaml
│   │       ├── synthnormal_q99_ewma09.yaml
│   │       └── synthnormal_q99_normalized_input_mse_ewma09.yaml # q99 from synthetic validation windows
│   ├── reporting_profiles/
│   │   ├── offline_detection.yaml
│   │   ├── online_detection.yaml
│   │   └── uncertainty.yaml
│   ├── metrics.yaml                  # Shared metric names and display precision
│   ├── experiments/
│   │   └── smd_comparison.yaml        # Example experiment definition
│   └── standalones/
│       └── smd_mustard.yaml           # Example standalone definition
├── src/
│   ├── __init__.py
│   ├── artifacts.py
│   ├── budgeted_vus.py
│   ├── cli.py
│   ├── config.py
│   ├── dataset_picker.py
│   ├── dataset_selection.py
│   ├── encoders.py
│   ├── evaluation_metrics.py
│   ├── experiments.py
│   ├── integrations.py
│   ├── offline.py
│   ├── online.py
│   ├── prepared_data.py
│   ├── protocols.py
│   ├── reporting.py
│   ├── scoring.py
│   └── methods/
│       ├── __init__.py
│       ├── anomaly_families.py
│       ├── augmentation.py
│       ├── candi.py
│       ├── candi_adapter.py
│       ├── iforest.py
│       ├── kmeans_ad.py
│       ├── m2n2.py
│       ├── moment.py
│       ├── redlamp.py
│       ├── stumpy_channel_ab.py
│       └── mustard/
│           ├── __init__.py
│           ├── contrastive.py
│           ├── losses.py
│           ├── model.py
│           ├── online_update.py
│           ├── retrieval.py
│           ├── signature_verification.py
│           ├── state.py
│           └── verification_buffer.py
├── tests/
│   ├── conftest.py
│   ├── test_cli.py
│   ├── test_dataset_picker.py
│   ├── test_config.py
│   ├── test_experiments.py
│   ├── test_data.py
│   ├── test_offline.py
│   ├── test_online.py
│   ├── test_scoring.py
│   ├── test_metrics.py
│   ├── test_reporting.py
│   ├── test_artifacts.py
│   ├── test_integrations.py
│   ├── test_encoders.py
│   ├── test_augmentation.py
│   ├── test_mustard.py
│   ├── test_baselines.py
│   └── test_protocols.py
├── documents/
│   ├── specs/
│   │   ├── definitions.md
│   │   ├── desired-cli-20261109.md
│   │   ├── config-profiles-flow.md
│   │   ├── protocol-validation-rules.md
│   │   └── current-protocol-validation-rules.md
│   └── logs/
│       └── 2026-11-16/
│           ├── plan/
│           │   └── decisions-tsad-preprocess.md
│           ├── research/
│           │   └── research-missing-data-options.md
│           └── structure/
│               └── proposed-next-version-tree.md
├── data/
│   └── preprocessed/                 # Versioned prepared data and manifests
├── outputs/                          # New runs and historical results
└── bsc-thesis-ref-codebases/          # Only required references and their licenses
```

The experiment and standalone YAML names are examples. Add other named definitions when needed. A method profile represents one supported parameter set; add a separate file when another phase or parameter set needs one. Listing O2 or MOMENT does not establish runtime readiness.

The [cleanup record](../plan/plan-cleanup-next-version.md) lists completed removals. All required data is processed, and the five original source folders were removed after checks. Keep prepared outputs, required current code, protected specifications, and detailed mathematics files.

## 3. Preprocessing and data profiles

1. The script remains runnable without `src/`, external YAML files, or the future `tsad` command.
2. Its built-in profiles are the only default source for raw preprocessing settings. Do not create five YAML copies of those settings.
3. The proposed runtime data profiles select prepared data and hold window, stride, and batch settings. They may reference a separately supplied preprocessing override file when needed.
4. Do not pass a runtime data profile directly to the current script. Its optional YAML accepts only the fields shown by `--show-profile`. Runtime profile fields still need a schema before implementation.
5. Keep synthetic training augmentation separate. It must not silently change the saved source splits or refit their scaler. Preserve the agreed version and provenance rules for generated data.

```text
Raw dataset
    -> scripts/preprocess.py
    -> versioned Parquet files and scaler state
    -> src/prepared_data.py
    -> offline.py or online.py
    -> reporting.py
```

The future `tsad preprocess` route calls the existing script with explicit project paths. It must not copy the script's implementation. Registering the installed command and its name alongside `tsad-lib` still needs a packaging decision.

## 4. Prepared output files

This is a generated-path template. The script supports the five selected dataset names. Each version can contain several entity folders.

```text
data/preprocessed/<dataset>/<version_id>/
├── manifest.json
└── <entity>/
    ├── train.parquet
    ├── validation.parquet
    ├── test.parquet
    └── scaler.json
```

Keep the agreed experiment result hierarchy and its grouping rules. This proposal does not replace that hierarchy with the preprocessing layout. The confirmed hierarchy supersedes the older depth limit for the next thesis version. No existing result paths are migrated here.

## 5. What becomes simpler

| Earlier proposal | New proposed owner |
| --- | --- |
| Dataset source modules, split code, and input scalers | `scripts/preprocess.py` |
| Prepared-data loading, windows, batches, and stream iteration | `src/prepared_data.py` |
| Experiment definitions and matrix expansion | `src/experiments.py` |
| Registry and profile default lookup | `src/config.py` |
| Checkpoint files, run status, offline-parent lookup, and resume records | `src/artifacts.py` |
| Optional logging and backup wrappers | `src/integrations.py` |

1. The script already implements the preprocessing part of the data and stream protocol. `protocols.py` checks prepared inputs and runtime windows. Do not copy raw-data checks into a second pipeline.
2. MUSTARD's public entrypoint owns its lifecycle and checkpoint contents. Its helpers perform focused calculations. `artifacts.py` owns file storage and run identity, not the method's mathematics.
3. Eligible methods use the experiment's encoder choice. Each method owns separate weights and mutable state. The decision does not require equal initialization.
4. Required-method checks remain a Markdown checklist and simple text search. Do not add a fifth protocol or a mandatory inheritance hierarchy.
5. Preserve every retained behavior before retiring an old file. Review callers, configs, checkpoints, calibration, resume, and numerical results first.

## 6. Evidence and status

1. [Implemented preprocessing script](../../../../scripts/preprocess.py).
2. [Confirmed preprocessing decisions](../plan/decisions-tsad-preprocess.md).
3. [Agreed definitions](../../../specs/definitions.md).
4. [Desired CLI](../../../specs/desired-cli-20261109.md).
5. [Earlier CLI tree](../../2026-11-09/detail/proposed-cli-native-tree.md).
6. [Earlier config and reporting plan](../../2026-11-09/plan/plan-config-signatures-and-fields.md).

The latest recorded temporary suite has 46 contract tests. All required ICCAD, SMD, IOPS, NASA, and SWaT data is processed. Saved manifests cover 1, 28, 29, 80, and 1 entities, respectively. SMD and SWaT also have recorded independent scaling checks. The rest of this tree remains proposed. File consolidation needs contract tests before migration. Split a proposed module only when its actual code needs it; keep functions within 50 lines and code files within 500 lines.

No domain terms are renamed. `data profile`, `method profile`, `evaluation profile`, `reporting profile`, and the four protocol names retain their meanings. The proposed runtime data-profile schema separates prepared-data loading from the script's preprocessing settings; it does not change the current script's YAML interface.

## 7. Cleanup status and remaining migration

1. Cleanup of confirmed unnecessary files and empty folders is complete. No local archive was created.
2. All required source data is processed. Later execution reads prepared outputs, not the removed source folders.
3. Implement and verify the proposed runtime modules before retiring required current code, configs, tests, and callers.
4. The installed CLI, runtime profile schemas, and method migration remain proposed. This document update does not move or delete files.


## Phase 1 contract alignment: 2026-11-17

1. Follow the [profile contract](../detail/runtime-profile-contract.md) and its ordered decision record for fields and defaults. Full-contract approval is pending.
2. `prepared_data.py` and `evaluation_metrics.py` avoid retained old-package collisions. Public function names stay unchanged.
3. Distribution `thesis-tsad` maps `thesis_tsad` to `src/` and `thesis_scripts` to `scripts/`. Install only in the thesis environment after checking command ownership.
4. Preserve gap segments, delayed score finalization, segment summaries, and recovery state. Keep smoke outputs separate from normal reports.
5. Use the state branch and artifact bindings in Stage 1.2 steps 7–10 for whole-run status, required stage checkpoints, and committed continuation.
