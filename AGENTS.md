# Repository Guidelines

Locally, the root of this repo is: `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/`.

On macos environment, please use `realpath` command to get the full file path everytime you cite a file or a folder.

All contributors and agents must strictly follow every requirement in `codebase_preferences.md`.
The single source of truth for this codebase is the `documents/design/` directory under `documents/`.

## Project Structure and Module Organization
This repository is research-first and currently centers on notebooks, design documents, and reference codebases.
- `documents/` holds design notes and research logs. Logs belong in `documents/logs/MM-DD-YYYY/<task_type>`.
- `prompts/` contains the prompt templates used for research and planning.
- `data/` stores raw datasets (for example `data/SMD`, `data/NASA`). Keep derived artifacts out of version control unless explicitly required.
- `bsc-thesis-ref-codebases/` contains external reference implementations (TSLib, CARLA, DALL-E, and others).
- Root-level `*.ipynb` notebooks are used for exploratory analysis and visualization.

When adding new source code, follow the guidance in `codebase_preferences.md`: one model per file, with inference and training logic colocated for readability.

## Build, Test, and Development Commands
- `conda env create -f environment.yml` creates a conda environment aligned with the research stack.
- `pip install -r requirements.txt` installs Python dependencies if using a virtual environment.
- `jupyter lab` runs local notebooks.
- `pytest` runs the test suite once tests are added.

## Coding Style and Naming Conventions
- Prefer explicit, descriptive names and short, linear control flow.
- Use four-space indentation in Python and follow PEP 8 conventions.
- Follow `python_semantics.md` whenever writing, reviewing, or refactoring Python code, especially for slicing, indexing, assignment, and mutability.
- Keep model logic self-contained in a single file. Avoid fragmented class hierarchies.
- Prefer adapter-style wrappers when integrating reference codebases.

## Testing Guidelines
Tests should use `pytest` and remain minimal and focused. Expected coverage includes data loader shapes, one forward and backward pass, checkpoint save and load, and synthetic anomaly injection behavior as described in `codebase_preferences.md`.

## Commit and Pull Request Guidelines
Commit messages in history use short, imperative summaries (for example, “Add data loaders”, “Fix type errors”). Follow the same pattern without prefixes or scopes.
Pull requests should include a concise summary, testing notes, and links to relevant design documents or notebooks. If data changes are introduced, describe the dataset versioning strategy.

## Research Workflow Notes
Follow the logging and planning conventions in `codebase_preferences.md`, including experiment tracking with Weights and Biases and the planned use of data version control for augmented datasets.
All contributors and agents must strictly follow every requirement in `codebase_preferences.md`.
