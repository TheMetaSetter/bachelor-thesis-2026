# Main Folders

- `bsc-thesis-ref-codebases/`
- `configs/`
- `data/`
- `demo/`
- `documents/`
- `images/`
- `notebooks/`
- `outputs`/
- `papers/`
- `presentation/`
- `prompts`/
- `reporting/`
- `scripts/`
- `src/`
- `tests/`


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
