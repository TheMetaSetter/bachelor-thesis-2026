# Main folders

1. `src/` contains the thesis CLI, workflows, validation rules, and methods.
2. `configs/` contains profiles, experiment definitions, and standalone definitions.
3. `scripts/` contains the self-contained preprocessing command and the documentation hook.
4. `data/preprocessed/` contains the prepared data and its manifests.
   `outputs/experiments/`, `outputs/smoke/`, and `outputs/standalones/` contain new run results when present.
5. `tests/` contains the runtime, terminal, and code size tests.
6. `documents/` contains decisions, specifications, and check results.
7. `bsc-thesis-ref-codebases/`, `papers/`, and `presentation/` contain research references and thesis material.
8. `notebooks/`, `images/`, `prompts/`, and `reporting/` contain retained material.

The [runtime tree](documents/logs/2026-11-16/structure/proposed-next-version-tree.md) lists the maintained files.
The [cleanup map](documents/logs/2026-11-20/detail/runtime-cleanup-paths.csv) records the retired paths.

## MemPalace indexing scope

Include these folders in the next indexing run for wing `bachelor_thesis_2026`:

- `src`
- `configs`
- `documents`
- `tests`
- `bsc-thesis-ref-codebases`
- `papers`
- `scripts`

Exclude `data`, `outputs`, and `reporting`. Exclude hidden files and folders inside the selected folders. Review any other file at least 100 MB before including it.
Run from the repository root with `mempalace mine --no-gitignore` so files in selected reference repositories are not lost to nested ignore rules; `mempalace.yaml` limits the scan to the seven folders above.
