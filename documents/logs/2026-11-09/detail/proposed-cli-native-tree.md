# Proposed CLI-native tree

Date: 2026-11-09  
Status: new design proposal. File names and responsibilities below are not implemented or verified. The older tree and current code are unchanged.

## Goal

Support the [desired CLI](../../../specs/desired-cli-20261109.md) with one clear owner for each workflow. Preserve the research scope, methods, four protocols, offline and online behavior, checkpoint and calibration contracts, recovery, reporting, and existing artifacts. This proposal follows the [simplicity audit](../research/audit-proposed-tree-cli-simplicity.md) and supersedes neither existing code nor an approved migration plan.

## Proposed tree

```text
bachelor-thesis-2026/
├── README.md
├── pyproject.toml                  # Proposes installed tsad command; ownership open
├── configs/
│   ├── data_profiles/
│   │   ├── smd.yaml
│   │   └── synthetic_profiles/
│   │       ├── redlamp_legacy.yaml
│   │       └── redlamp_visible.yaml
│   ├── method_profiles/
│   │   ├── mustard_<variant>_offline.yaml
│   │   ├── mustard_<variant>_online.yaml
│   │   └── baselines/<method-and-settings>.yaml
│   ├── evaluation_profiles/smd/
│   │   ├── cleanval_q95_ewma09.yaml
│   │   ├── cleanval_q99_ewma09.yaml
│   │   ├── cleanval_q995_ewma09.yaml
│   │   ├── synthnormal_q99_ewma09.yaml
│   │   └── synthnormal_q99_normalized_input_mse_ewma09.yaml
│   ├── reporting_profiles/<report>.yaml
│   ├── metrics.yaml                # Proposed shared metric definitions
│   ├── experiments/<experiment>.yaml
│   └── standalones/<run>.yaml
├── src/
│   ├── cli.py                      # Parse commands, completion, dispatch
│   ├── core/
│   │   ├── config.py               # Load and resolve profiles and settings
│   │   ├── contracts.py            # Shared data and artifact contracts
│   │   └── registry.py             # Dataset and method names and readiness
│   ├── experiments/
│   │   ├── definitions.py          # Create, list, read, and edit experiment YAML
│   │   ├── matrix.py               # Resolve defaults and create candidate runs
│   │   └── runner.py               # Execute, track dependencies, and orchestrate
│   ├── data/
│   │   ├── datasets/
│   │   │   ├── smd.py
│   │   │   ├── anomaly_archive.py
│   │   │   ├── nasa.py             # Proposed; NASA labels available
│   │   │   ├── swat.py             # Proposed; labels available
│   │   │   ├── iccad.py            # Proposed; interval labels
│   │   │   ├── tsb_ad_u.py         # Proposed; data currently archived
│   │   │   └── tsb_ad_m.py         # Proposed; data currently archived
│   │   ├── loaders.py
│   │   ├── split.py
│   │   ├── scalers.py
│   │   ├── window.py
│   │   └── stream.py
│   ├── methods/
│   │   ├── thesis/
│   │   │   ├── model.py
│   │   │   ├── encoder.py
│   │   │   ├── retrieval.py
│   │   │   ├── losses.py
│   │   │   ├── state.py
│   │   │   └── online_update.py
│   │   └── baselines/
│   │       ├── redlamp.py
│   │       ├── stumpy_channel_ab.py
│   │       ├── kmeans_ad.py
│   │       ├── iforest.py
│   │       ├── candi.py
│   │       ├── m2n2.py
│   │       └── moment.py           # Runnable path unverified
│   ├── protocols/
│   │   ├── run_evaluation.py
│   │   ├── data_stream.py
│   │   ├── scoring_calibration.py
│   │   └── method_updates.py
│   ├── runtime/
│   │   ├── offline.py
│   │   ├── online.py
│   │   ├── checkpoint.py
│   │   ├── calibration.py
│   │   └── resume.py
│   ├── evaluation/
│   │   ├── scoring.py
│   │   ├── metrics.py
│   │   ├── reporting.py            # Find results, aggregate, write CSV
│   │   └── protocol_audit.py
│   └── integrations/
│       ├── wandb.py                # Run logging
│       └── kaggle.py               # Experiment or standalone backup
├── scripts/
│   ├── compatibility/              # Only wrappers required by mapped callers
│   ├── ops/                        # Retained artifact maintenance
│   └── visualization/              # Retained research plots
├── tests/
│   ├── cli/
│   ├── core/
│   ├── experiments/
│   ├── data/
│   ├── methods/
│   ├── protocols/
│   ├── runtime/
│   ├── evaluation/
│   └── integrations/
├── documents/                      # Existing specs, research, and logs
├── bsc-thesis-ref-codebases/       # Existing reference code and licenses
├── data/                           # Existing local datasets and archives
│   ├── AnomalyArchive/              # Anomaly intervals encoded in filenames
│   ├── NASA/                       # train/, test/, labeled_anomalies.csv
│   ├── SMD/                        # Array data and SMD_test_label.npy
│   ├── ServerMachineDataset/       # train/, test/, test_label/
│   ├── SWaT/                       # CSV files include Normal/Attack labels
│   ├── ibm-cloud-console-anomaly-dataset-iccad/
│   │   └── anomaly_windows.csv     # Labeled anomaly intervals
│   ├── TSB-AD-U.zip                # Archived CSVs; sampled file has Label
│   ├── TSB-AD-M.zip                # Archived CSVs; sampled file has Label
│   └── IOPS/                       # Sample test files have two unnamed columns; labels unconfirmed
└── outputs/                        # Existing and new results; see config plan
```

Angle-bracket names are file patterns, not literal filenames. `configs/metrics.yaml`, profile filenames, and module names are proposed. Preserve other existing research materials and assets; omission from this compact tree is not a deletion decision.

## Existing dataset evidence

1. The tree above lists local labeled dataset sources observed on 2026-11-09. `AnomalyArchive` filenames carry anomaly interval indices; the current parser reads those indices. `NASA/labeled_anomalies.csv` and ICCAD's `anomaly_windows.csv` contain anomaly intervals. ICCAD's interval labels do not by themselves establish point-level labels.
2. `SMD/` and `ServerMachineDataset/` are two local representations with label files; their identity and preprocessing equivalence are not verified. `SWaT/attack.csv` and `SWaT/merged.csv` have a `Normal/Attack` column.
3. `TSB-AD-U.zip` and `TSB-AD-M.zip` are archives, not extracted directories; one CSV sampled from each contains a `Label` column. The contents and quality of every archived file were not audited.
4. `IOPS/` test files contain two unnamed columns in the inspected samples; their label semantics need confirmation. It is listed as a candidate, not as confirmed labeled data. Other local files and archives remain unclassified until their labels and intended task are verified; none are proposed for deletion.
5. The new dataset module names are proposals, not claims that loaders, profiles, downloads, or end-to-end runs exist. Dataset-specific config profiles and defaults for these sources require separate design. Adding labeled sources to the tree does not establish that all are selected for every experiment.

## Command ownership

| Desired CLI workflow | Proposed owner | Reads or changes |
| --- | --- | --- |
| List datasets and methods | `core/registry.py` | Registered names and readiness. |
| List, create, show, and edit experiments | `experiments/definitions.py` | Experiment YAML under `configs/experiments/`. |
| Complete names | `cli.py`, using registry and definitions | Known methods, datasets, and experiments. |
| Expand and show runs | `experiments/matrix.py` with `core/config.py` | Selected methods, datasets, defaults, profiles, and seeds. |
| Validate an experiment | `experiments/matrix.py` with `protocols/` and runtime checks | Resolved inputs, compatibility, and readiness; no training. |
| Run or smoke-run | `experiments/runner.py` with `runtime/` | Resolved runs, dependencies, checkpoints, and results. |
| Resume | `runtime/resume.py` with `experiments/runner.py` | Existing experiment state and artifacts. |
| Report | `evaluation/reporting.py` | Result JSON, reporting profiles, and metric definitions; writes CSV tables. |
| W&B and Kaggle settings | `integrations/wandb.py` and `integrations/kaggle.py` | Logging and backup for requested runs and experiments. |
| Dataset download | Future feature; no owner selected | Hugging Face download behavior is not designed. |

`cli.py` dispatches these operations; it does not own scientific computations. `pyproject.toml` is the proposed registration point for `tsad`. Whether this command is shared with or separate from the `tsad-lib` command remains open.

## Design boundaries

1. Generate applicable method–dataset combinations and expand them over the shared seed list. Resolve the defaults for entities, variants, and profiles before execution. The definition of applicability and error behavior remain open; do not silently discard requested combinations.
2. Keep the four protocol contexts: run and evaluation, data and stream, scoring and calibration, and method updates. Profiles contain values; protocols contain related input and output validation rules.
3. Keep explicit offline and online execution, checkpoint, calibration, and recovery responsibilities. Use the agreed offline-parent lookup and saved scaler state. Exact compatibility fields and artifact identities remain open.
4. Reporting uses JSON run metadata and metrics, and produces detailed and mean-only CSV tables by dataset. Preserve phase and split distinctions, valid seed counts, sample standard deviation, and profile provenance.
5. `--smoke` is a run mode, not a separately required smoke config. Its selection, limits, output identity, and protection against mixing with full results remain open.
6. Do not add a separate implementation for each CLI operation when a thin command can call the relevant owner. Retain existing script wrappers only after checking their callers and contracts; this is not a deletion list.
7. W&B logging and Kaggle backup remain distinct integrations. Their exact dependencies, failure behavior, and how they relate to `tsad-lib` need design before implementation.

## Shared encoder in an experiment

Methods in `tsad-lib/documents/specs/method-mathematics/` that are verified to be encoder-agnostic must use the same encoder in a specific experiment. The method notes do not by themselves establish which methods qualify; verify each method's mathematical and implementation contract before classifying it. Methods that require a particular encoder or do not use one are not forced into this rule.

The exact experiment-level configuration and encoder interface remain to be designed. Whether “same encoder” means the same architecture and initialization, or the same trained weights or shared model instance, is not yet established. Do not assume that sharing an encoder implies shared mutable state across runs or methods.

## Before implementation

1. Map every retained current method, variant, command, config, test, script caller, and artifact contract to a proposed owner. Resolve whether wrappers or extra modules are necessary.
2. Design signatures and config schemas, including experiment naming, lookup, mutation, defaults, completion, run IDs, reporting, and error behavior.
3. Reconcile automatic method–dataset combinations with the older explicit-template plan. Specify selection and evaluation behavior without losing scientific scope.
4. Confirm O2 and MOMENT end-to-end readiness before claiming they work. Compare score order, calibration, checkpoints, resume, and representative outputs against agreed references.
5. Establish an explicit migration and recovery plan that preserves tracked and local artifacts. No source, YAML, script, or artifact is changed by this proposal.
