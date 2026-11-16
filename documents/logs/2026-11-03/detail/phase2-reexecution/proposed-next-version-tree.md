# Proposed next-version tree

Updated: 2026-11-09
Status: design proposal; no source files changed.

## Goal

Keep the required research work runnable while giving each kind of code one clear home. This is a proposed layout, not evidence of behavior or parity.

Follow the agreed [config and reporting plan](../../../2026-11-09/plan/plan-config-signatures-and-fields.md). The next version is CLI-native; exact command names are open. Keep THESIS/MUSTARD, its O0/O1/O2 and A0/A1/A2 variants, the retained comparison methods, offline and online runs, checkpoint and resume handling, calibration, config generation, launchers, and result collection. Keep reference code and research records.

## Proposed tree

```text
bachelor-thesis-2026/
├── README.md
├── pyproject.toml
├── configs/
│   ├── data_profiles/
│   │   ├── smd.yaml                  # Dataset source and shared preprocessing
│   │   └── synthetic_profiles/
│   │       ├── redlamp_legacy.yaml
│   │       └── redlamp_visible.yaml
│   ├── method_profiles/
│   │   ├── mustard_<variant>_offline.yaml  # Proposed pattern; one parameter set per profile
│   │   ├── mustard_<variant>_online.yaml
│   │   └── baselines/
│   │       └── <method-and-settings>.yaml   # Separate profile for each parameter set
│   ├── evaluation_profiles/
│   │   └── smd/
│   │       ├── cleanval_q95_ewma09.yaml
│   │       ├── cleanval_q99_ewma09.yaml
│   │       ├── cleanval_q995_ewma09.yaml
│   │       ├── synthnormal_q99_ewma09.yaml
│   │       └── synthnormal_q99_normalized_input_mse_ewma09.yaml
│   ├── reporting_profiles/            # Proposed location; exact filenames open
│   │   └── <report>.yaml             # Table selection and display
│   ├── metrics.yaml                  # Proposed name; shared metric display defaults
│   ├── experiments/
│   │   ├── thesis.yaml               # Run templates and shared seeds
│   │   ├── baselines.yaml
│   │   ├── smoke.yaml                # Only when multiple runs
│   │   └── diagnostics.yaml          # Only when multiple runs
│   └── standalones/
│       └── <run>.yaml                # One standalone execution, including demos
├── data/                             # Local data; not source code
├── outputs/                          # Results; hierarchy described below
├── documents/
│   ├── specs/
│   │   ├── definitions.md
│   │   └── method-implementation-checklist.md
│   ├── research/                     # Papers, method notes, and decisions
│   └── logs/                         # Dated records and execution evidence
├── bsc-thesis-ref-codebases/         # Reference code and licenses
├── src/
│   ├── cli.py                        # Proposed CLI entry point; command names open
│   ├── core/
│   │   ├── config.py                # Load and validate general settings
│   │   ├── contracts.py             # Shared data and artifact contracts
│   │   └── registry.py              # Resolve method and dataset names
│   ├── data/
│   │   ├── datasets/
│   │   │   ├── smd.py
│   │   │   └── anomaly_archive.py
│   │   ├── loaders.py
│   │   ├── split.py
│   │   ├── scalers.py
│   │   ├── window.py
│   │   └── stream.py
│   ├── methods/
│   │   ├── thesis/
│   │   │   ├── model.py              # THESIS/MUSTARD model entry point
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
│   │       └── moment.py             # Proposed method file; runtime path is unverified
│   ├── protocols/
│   │   ├── run_evaluation.py         # Run inputs and evaluation outputs
│   │   ├── data_stream.py            # Sequence, window, batch, and stream checks
│   │   ├── scoring_calibration.py   # Score, calibration, and threshold checks
│   │   └── method_updates.py         # Update request and update result checks
│   ├── runtime/
│   │   ├── offline.py
│   │   ├── online.py
│   │   ├── checkpoint.py
│   │   ├── calibration.py
│   │   └── resume.py
│   ├── evaluation/
│   │   ├── scoring.py
│   │   ├── metrics.py
│   │   ├── reporting.py              # Aggregate run results and write tables
│   │   └── protocol_audit.py
│   └── analysis/                     # Reusable analysis code
├── scripts/
│   ├── cli/                          # Compatibility wrappers only if required
│   ├── experiments/
│   │   ├── generate_configs.py
│   │   ├── run_offline.py
│   │   ├── run_online.py
│   │   ├── run_comparison.py
│   │   ├── run_ablation.py
│   │   ├── collect_results.py
│   │   └── summarize_results.py
│   ├── launchers/
│   │   ├── run_offline.sh
│   │   └── run_online.sh
│   ├── ops/                          # Artifact inspection and maintenance
│   └── visualization/               # Retained research plots
├── tests/
│   ├── protocols/
│   │   ├── test_run_evaluation.py
│   │   ├── test_data_stream.py
│   │   ├── test_scoring_calibration.py
│   │   └── test_method_updates.py
│   ├── data/
│   ├── methods/
│   ├── runtime/
│   └── scripts/
```

## Config file details

This is a proposed layout, not an implemented schema. The earlier read-only inventory counted 808 YAML files; it does not establish a complete migration. See the [detailed plan](../../../2026-11-09/plan/plan-config-signatures-and-fields.md) for proposed fields and open decisions.

1. `configs/data_profiles/` holds dataset and synthetic-data profiles. One experiment uses one data profile per dataset. Fit the scaler on training data and reuse its saved state in the online run.
2. `configs/method_profiles/` holds method settings. Separate profiles represent different baseline parameter sets; do not put baseline parameter overrides in experiment cells. MUSTARD variant filenames are illustrative patterns, not confirmed files. O2 and MOMENT runtime paths remain unverified.
3. `configs/evaluation_profiles/` holds scoring, evaluation, and calibration settings. `configs/reporting_profiles/` holds table selection and display settings; its exact location and file names are proposed. A reporting profile can produce tables from existing results without changing the run or its underlying metric.
4. The proposed `configs/metrics.yaml` holds shared metric names and default display precision. A reporting profile can override precision for a table, but this is expected to be rare. The filename and full schema remain open.
5. `configs/experiments/` defines experiments with multiple runs. It lists run templates, one shared seed list, and reporting profile references. Expand each template once per seed; do not generate other unapproved combinations. One template with multiple seeds can produce an experiment. A resolved cell identifies one run.
6. `configs/standalones/` defines individual runs. A user may supply a unique `run_id`; otherwise generate UUIDv7. A missing standalone `output_dir` defaults to `outputs/standalones/<run_id>/`.
7. An experiment names its `experiment_id`; the program generates a project-unique UUIDv7 `run_id` for each execution. A repeated execution receives a new ID; resume keeps the existing ID. Experiment output paths are derived from the hierarchy below.
8. Four protocol files own validation rules, not YAML values. General config parsing and profile resolution belong in `src/core/config.py`. The signatures and exact integration are proposed in the detailed plan.

## Output and reporting layout

Experiment outputs use this full hierarchy; `phase` and `split` are always present, and dataset, entity, method, variant, and seed folders are omitted under the agreed single-value rules. Create only split folders with results. An execution has the same `run_id` in each split it produces.

```text
outputs/<experiment_id>/<phase>/<dataset>/<data_entity>/<method_name>/<ablation_variant>/<random_seed>/<split>/<run_id>/
```

1. Keep separate run ID folders for repeated executions. The aggregator uses the configured dimensions and result metadata to interpret omitted folders and select matching runs.
2. Each split records metadata and metrics in JSON. Each dataset table covers one phase and may contain multiple splits; the exact column names and file locations remain open.
3. Reporting cells aggregate one metric for an entity and method variant over the shared seeds. Select the latest eligible successful run by start time for each seed; exclude missing and non-finite values. Report the mean, sample standard deviation, and valid/expected count; show `N/A` where a value is unavailable.
4. Write a detailed CSV using `mean ± standard deviation (valid/expected)` and a simplified mean-only CSV. A method with no specified ablation uses `default`, traceable to the method profile path in run metadata.
5. For an online experiment run, find the latest successful compatible offline run within the same experiment for the required offline variant and seed. Load its best checkpoint and saved scaler; stop if a required calibration artifact is missing. Exact compatibility fields, artifact selection, and tie-breaking details remain open.

## Config terminology changes

1. The current `experiment_name` identifies a single run in the inspected loader. The proposed `experiment_id` identifies a multi-run experiment; generated `run_id` identifies an execution. These names are not automatically interchangeable.
2. Current `data_config_path` and `model_config_path` inform the proposed `data_profile` and `method_profile`. Current task settings may split between data, method, and run inputs. Current score and calibration settings inform `evaluation_profile`; these are mappings to verify, not automatic migration rules.
3. Current synthetic profiles are proposed under `configs/data_profiles/synthetic_profiles/`. Existing generated and debug runs require individual mapping to experiment templates or standalone files; do not assume all are interchangeable.
4. `reporting_profile`, shared metric definitions, and standalone config grouping are new proposed concepts. `offline_variant` and `online_variant` remain MUSTARD-specific names.
5. A template plus one seed resolves to a cell; the cell represents one run. An experiment is not an additional “experiment set.”

The earlier key scan found fields without exact source-string lookups. Their meaning and use remain unresolved; do not discard them merely because they were not found by that scan. Preserve existing artifacts and explicitly map old paths, settings, and consumers before migration.

## Protocol file details

Each file owns input and output validation rules for one context. Generic key, type, and tensor-shape helpers stay in `core/contracts.py`; protocol files use them and add context-specific checks. The current code spreads these checks across several modules, so this tree describes a proposed next-version layout.

1. `src/protocols/run_evaluation.py`
   1. Input checks: run identity, required and allowed config fields, variant and seed consistency, and evaluation restrictions such as test labels being used for metrics only.
   2. Output checks: required model-output fields and ranks, evaluation score/label alignment, and evaluation-record lengths and types.
   3. Keep general config parsing in `core/config.py`; this file owns rules that apply to a research run or its evaluation.
2. `src/protocols/data_stream.py`
   1. Input checks: data settings, raw sequence fields and shapes, split selection, stream-range bounds, supported stream mode, and scaler readiness.
   2. Output checks: window and batch structure, required metadata, and online absolute-index shape, type, and order.
   3. State any unchecked alignment rule explicitly. The current window validator, for example, does not check that every optional field has the same length as `x`.
3. `src/protocols/scoring_calibration.py`
   1. Input checks: score-space and transform choices, reconstruction shapes and finiteness, calibration inputs, threshold settings, and threshold-artifact identity and provenance.
   2. Output checks: finite reconstruction scores, per-window score lengths, EWMA weight sum, calibration values, and threshold-artifact fields and ordering.
   3. Preserve data-source limits: test labels do not select thresholds, and normalized-input scores use the required synthetic-normal validation source.
   4. The current aggregator returns a coverage mask but does not enforce full coverage. Keep that gap visible and add a coverage rule only where the run requires it.
4. `src/protocols/method_updates.py`
   1. Input checks: supported online variant, optimizer target and type, trainable-parameter rules, one-window input, triage thresholds, update-loss shapes, and verification inputs.
   2. Output checks: update-result fields, finite loss and parameters, allowed parameter changes, and consistency between the recorded score and the update from that window.
   3. The current code has no general post-update output validator. Add one here in the next version; do not describe the existing pre-update parameter check as post-update validation.

The four files validate values entering and leaving their context. They do not contain method implementations or run the experiment. Checking whether an implementation provides required methods stays in `documents/specs/method-implementation-checklist.md`, using a checklist and `grep`; it is a separate procedure.

## Simplification rules

1. Make main workflows accessible through the CLI. Choose its command names separately; retire forwarding copies only after their callers and documented commands are moved.
2. Keep each method's scoring and update behavior with that method. Share only identical helpers with matching inputs and outputs.
3. Keep checkpoint, calibration, and resume rules explicit. Do not merge them into a generic run helper.
4. Keep old result identities and reference code intact. Do not delete research records as part of source reorganization.
5. Keep all retained methods, variants, and tests. Do not remove a feature based only on its absence from the main config.

## Checks before implementation

1. Map every current source file, config, test, command, and artifact reader to its proposed owner.
2. Confirm each retained method has an end-to-end run path. MOMENT is listed as a baseline, but its runnable method path is unverified.
3. Resolve the THESIS O2 path and matching checkpoint evidence before claiming the full variant set is runnable.
4. Compare outputs, scoring order, calibration, checkpoint loading, and resume behavior after each move or consolidation.
5. Reconcile the exact CLI commands, profile schemas, reporting paths, compatibility checks, and artifact lookup before implementation.
6. Recount source files and lines after the file-action map is complete. This tree makes no reduction claim.
