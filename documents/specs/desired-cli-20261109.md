# Desired CLI end state

Date: 2026-11-09

Updated: 2026-11-20

Status: CLI design for `bachelor-thesis-2026`. See the [execution record](../logs/2026-11-16/detail/runtime-replacement-review.md) for the installed commands and verification results.

Use this command surface to guide the directory tree and the later design of class, method, and function signatures.

## Confirmed runtime requirements: 2026-11-16

Follow the [runtime replacement decisions](../logs/2026-11-16/plan/decisions-runtime-replacement.md). These decisions settle the corresponding open questions below.

1. Runtime commands reuse saved prepared data without raw sources, new splits, filling, or input scaling.
2. `show` and `validate` resolve common CNN settings for MUSTARD, RedLamp, CANDI, and M2N2. Separate instances own separate weights. MOMENT retains its pretrained encoder.
3. `validate` stops and lists unsupported combinations. It does not silently skip them.
4. `--smoke` selects the first configured dataset, entity, applicable variant, and seed for each method and supported phase. Include required offline dependencies. Limit work without changing prepared data; exclude smoke results from normal reports. Numeric limits and output isolation remain to be specified.
5. Test values support causal scoring and unlabeled adaptation after scoring. No future observations, complete-test statistics, or test-based tuning are allowed. Test labels remain with metric calculation.
6. RedLamp uses validation-fixed class selection and score scaling, with causal smoothing.
7. The next runtime uses new commands and checkpoints without old-command wrappers. Keep historical results as evidence.

## Commands

```bash
# Discover
tsad list datasets
tsad list methods
tsad list experiments

# Prepare data without an experiment
tsad preprocess --dataset <dataset_name>

# Create and configure
tsad new experiment --name <experiment_name>
tsad experiment --name <experiment_name> show
tsad experiment --name <experiment_name> methods add <method_1> [method_2 ...]
tsad experiment --name <experiment_name> methods remove <method_1> [method_2 ...]
tsad experiment --name <experiment_name> datasets add <dataset_1> [dataset_2 ...]
tsad experiment --name <experiment_name> datasets remove <dataset_1> [dataset_2 ...]
tsad experiment --name <experiment_name> set random_seeds <seed_1> [seed_2 ...]
tsad experiment --name <experiment_name> set wandb_logging true
tsad experiment --name <experiment_name> set kaggle_backup true

# Validate, run, and report
tsad validate experiment --name <experiment_name>
tsad run experiment --name <experiment_name> --smoke
tsad run experiment --name <experiment_name>
tsad resume experiment --name <experiment_name>
tsad report experiment --name <experiment_name>

# Future feature: download a dataset from Hugging Face
tsad datasets download <dataset_name>
```

The bracketed arguments mean one or more methods, datasets, or seeds may be supplied, as applicable. They are notation for this document, not literal shell arguments. `wandb_logging` and `kaggle_backup` can be enabled or disabled; `true` above is an example.

## Preprocessing

1. The installed command is `tsad preprocess --dataset <dataset_name>`. Supported datasets are `iccad`, `smd`, `iops`, `swat`, and `nasa`. It does not require an experiment.
2. The installed command uses the same script as `python scripts/preprocess.py --dataset <dataset_name>`.
3. Built-in settings are the defaults. Optional arguments are `--profile <settings.yaml>`, `--input-root <path>`, `--output-root <path>`, and `--entities <entity_1> [entity_2 ...]`. `--show-profile` prints resolved settings without reading source data.
4. Preprocessing owns sorting, source selection, chronological splitting, approved missing-value handling, and training-only input scaling. It does not create windows.
5. Save `manifest.json` at `data/preprocessed/<dataset>/<version_id>/`. Each entity folder contains `train.parquet`, `validation.parquet`, `test.parquet`, and `scaler.json`. An explicit output root replaces the default root.
6. With source files available, repeated execution checks a complete matching version before reuse. Incomplete or damaged versions stop without automatic overwrite or recovery.
7. Runtime data profiles select prepared versions and window settings. They are separate from the optional preprocessing override YAML. Runtime loading must not split or scale prepared observations again.
8. All required data is already processed. The five original source folders were removed after checks. The preprocessing command needs source files for a new execution or source-based repeat check; it cannot recreate them from prepared outputs. Source removal is a separate action, not script behavior.

```bash
# Current entrypoint; requires source data
python scripts/preprocess.py --dataset smd --entities machine-1-1

# Inspect built-in settings; does not require source data
python scripts/preprocess.py --dataset smd --show-profile
```

See [the confirmed preprocessing decisions](../logs/2026-11-16/plan/decisions-tsad-preprocess.md) for scientific rules, exclusions, and completed checks.

## Agreed behavior

In a terminal, `datasets add` opens one prepared-version selector per dataset.
Use Tab to reach Save, then press Enter. Save checks the selected data and
updates `prepared_root` in the shared data profile. New plans that use that
profile use the new version. With no terminal, the command applies the same
defaults. See the [dataset selection flow](../logs/2026-11-20/detail/dataset-selection-and-cleanup-decisions.md).

1. Shell completion suggests experiment names after `--name` and available names for method and dataset selection.
2. When the user does not specify entities, method variants, data profiles, or reporting profiles, the CLI uses defaults. The source and exact values of those defaults remain to be designed.
3. Adding multiple methods and datasets produces their applicable combinations. One dataset per experiment is expected to be common. Exact supported combinations still need method review. Validation stops and lists unsupported combinations.
4. `show` displays the selected settings, which choices are defaults, and the resulting runs. `validate` checks readiness before execution.
5. `--smoke` uses the selection rule above. Numeric work limits and output isolation remain to be specified; smoke results are excluded from normal reports.
6. A reporting command produces experiment tables from run results. Resume follows the agreed recovery rules; its detailed behavior remains to be designed.
7. Dataset download from Hugging Face is a future feature. The dataset source, selection, storage, and download behavior are not yet specified.

## Design work that follows

1. Reconcile automatic method–dataset combinations with the earlier config plan, which allowed automatic expansion over seeds only.
2. Define how the CLI selects or changes entities, variants, and profiles, and how it resolves their defaults.
3. Design a simple file-level directory tree around these workflows.
4. Specify clear signatures and responsibilities for the classes, methods, and functions in each file.
5. Resolve command naming and compatibility with the existing `tsad-lib` CLI proposal before claiming the two codebases share an implementation.

Related design: [config signatures and fields](../logs/2026-11-09/plan/plan-config-signatures-and-fields.md).

## Terminology changes

No domain terms changed. The CLI provides `tsad preprocess`. A runtime data profile selects prepared data; an optional preprocessing override YAML controls source preparation. The two files have different schemas.
