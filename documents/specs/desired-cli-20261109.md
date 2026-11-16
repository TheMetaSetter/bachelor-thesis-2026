# Desired CLI end state

Date: 2026-11-09  
Status: proposed end state for the next version of `bachelor-thesis-2026`; not implemented or an implementation specification.

Use this command surface to guide the directory tree and the later design of class, method, and function signatures.

## Commands

```bash
# Discover
tsad list datasets
tsad list methods
tsad list experiments

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

## Agreed behavior

1. Shell completion suggests experiment names after `--name` and available names for method and dataset selection.
2. When the user does not specify entities, method variants, data profiles, or reporting profiles, the CLI uses defaults. The source and exact values of those defaults remain to be designed.
3. Adding multiple methods and datasets produces their applicable combinations. One dataset per experiment is expected to be common. The definition of applicability and how unavailable combinations are reported remain open.
4. `show` displays the selected settings, which choices are defaults, and the resulting runs. `validate` checks readiness before execution.
5. `--smoke` provides a smoke execution. Its precise limits, run selection, and output identity remain open.
6. A reporting command produces experiment tables from run results. Resume follows the agreed recovery rules; its detailed behavior remains to be designed.
7. Dataset download from Hugging Face is a future feature. The dataset source, selection, storage, and download behavior are not yet specified.

## Design work that follows

1. Reconcile automatic method–dataset combinations with the earlier config plan, which allowed automatic expansion over seeds only.
2. Define how the CLI selects or changes entities, variants, and profiles, and how it resolves their defaults.
3. Design a simple file-level directory tree around these workflows.
4. Specify clear signatures and responsibilities for the classes, methods, and functions in each file.
5. Resolve command naming and compatibility with the existing `tsad-lib` CLI proposal before claiming the two codebases share an implementation.

Related design: [config signatures and fields](../logs/2026-11-09/plan/plan-config-signatures-and-fields.md).
