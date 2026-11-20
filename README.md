# MUSTARD time-series anomaly detection

This repository contains the thesis runtime and its comparison methods.
The runtime reads prepared data for ICCAD, SMD, IOPS, SWaT, and NASA.

## Use the thesis CLI

Run these commands from the thesis repository with its Python environment.

```sh
.venv/bin/python -m pip install -e .
.venv/bin/tsad list datasets
.venv/bin/tsad list methods
.venv/bin/tsad validate experiment --name smd_comparison
.venv/bin/tsad run experiment --name smd_comparison --smoke
.venv/bin/tsad report experiment --name smd_comparison --smoke
```

Use `tsad resume experiment --name smd_comparison` after an interruption.
Resume uses the saved run IDs, settings, and continuation state.
Use `tsad run standalone --name smd_mustard` for the named standalone definition.

For shell completion, activate `.venv` and load the script from `tsad completion bash` or `tsad completion zsh`.
Initialize Zsh completion with `compinit` first.

### Make reports from saved results

```sh
.venv/bin/tsad report experiment --name NAME
.venv/bin/tsad report experiment --name NAME --smoke
```

The first command reads normal results and writes to `outputs/experiments/NAME/reports/DATASET/`.
The second reads smoke results and writes to `outputs/smoke/NAME/reports/DATASET/`.
Each reporting profile produces an HTML file, two CSV files, and a JSON file.
In an experiment session, use `/report` or `/report --smoke` to run the same commands.

### Use the terminal interface

Run `.venv/bin/tsad` from the thesis repository in an interactive terminal.
Type each command in the text bar and press Enter.

1. Use `/list experiments` to see the experiment names.
2. Use `/experiment NAME` to open a session.
3. Use `/help` to see the slash commands.
4. Use `/datasets add smd swat` to open the prepared-version picker.
5. Use Tab to move to **Save** or **Cancel**. Press Enter to select it.
6. Read the CLI result. Press Enter to return to the command screen.
7. Select **Back** to return to the session.
8. Use `/quit` to close the interface.

Each session keeps its command history until the app closes.

### Select prepared dataset versions

```sh
.venv/bin/tsad new experiment --name my_experiment
.venv/bin/tsad experiment --name my_experiment datasets add smd swat
.venv/bin/tsad experiment --name my_experiment methods add mustard
```

Use Tab to move between fields. Press Enter to open a list. Use the arrow
keys to move through the list, and press Enter to select a version.
Press Esc to close an open list. Move to **Save** and press Enter to save.
Move to **Cancel** and press Enter to exit without changes.

The picker reads manifests to make the list. It selects the version in
the data profile if its manifest is valid. If that version is unavailable,
it selects the complete version with the latest manifest modification time.
The same defaults apply when the command runs without an interactive terminal.

Save checks the selected data files. It updates `prepared_root` in each
shared data profile. New plans that use an updated profile use the new version.
Saved run settings keep their original paths.

## Data and results

1. Inspect preprocessing settings with `.venv/bin/tsad preprocess --dataset smd --show-profile`.
   Preprocessing requires raw source files and saves versions in `data/preprocessed/<dataset>/<version_id>/` by default.
   Use `--output-root` to select a different folder. The runtime reads the version selected in each dataset profile.
2. Normal runs write to `outputs/experiments/`. Smoke runs write to `outputs/smoke/`.
   Reports use normal runs. Each table has a detailed CSV and a mean-only CSV.
3. MOMENT requires the pinned pretrained weights and the `moment` optional dependency group.
   Weights & Biases and Kaggle use their own optional dependency groups and are disabled by default.
   For Kaggle backup, set `TSAD_KAGGLE_DATASET` to the existing private dataset reference.

The checks cover bounded runs and exact recovery on prepared SMD data. See the [execution record](documents/logs/2026-11-16/detail/runtime-replacement-review.md) for results and dependency limits.
