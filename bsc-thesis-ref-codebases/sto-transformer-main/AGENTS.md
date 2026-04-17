# Repository Guidelines

## Project Structure and Module Organization
- `code/` contains all source code and scripts.
- `code/IMDB/` implements the sentiment analysis experiments, including model definitions and the main runner.
- `code/CoLA/` implements the linguistic acceptability experiments.
- `code/common/` contains shared utilities and transformer components.
- Repository-level scripts (`code/setup.sh`, `code/pre.sh`, `code/run.sh`) support environment setup and experiment runs.

## Build, Test, and Development Commands
- Create a Python environment and install dependencies:
  - `conda create -n pytorch_latest_p37 python=3.7 anaconda`
  - `source activate pytorch_latest_p37`
  - `sh code/setup.sh`
- IMDB preprocessing and training:
  - `sh code/pre.sh -m IMDB -t tf -e default` (prepare datasets)
  - `sh code/run.sh -r train -m IMDB -t tf -e default -p '--n_epoch=5' -i 0` (train)
- IMDB uncertainty runs:
  - `sh code/run.sh -r uncertain-train-test -m IMDB -t tf-sto -e single_t1 -p '--n_epoch=50 --tau=1' -i 0`
- CoLA training:
  - `python code/CoLA/train.py --model_type sto_transformer --inference True --sto_transformer True --model_name dual --dual True`

## Coding Style and Naming Conventions
- Python is the primary language; follow standard Python formatting with four-space indentation.
- No automated formatter or linter is specified; keep changes minimal and consistent with surrounding code.
- Use descriptive snake_case for variables and functions; class names follow CapWords, matching existing modules.

## Testing Guidelines
- No dedicated test suite is present in the repository.
- Validate changes by running the relevant training or evaluation commands in `code/IMDB/README.md` and `code/CoLA/README.md`.

## Commit and Pull Request Guidelines
- Recent commit messages are short, imperative statements (for example, “Add debug code”). Use a similar style.
- Follow `CONTRIBUTING.md`: work from `main`, open an issue for significant changes, keep pull requests focused, and ensure local checks pass.
- Include a clear description of the change and any command output needed to reproduce results.

## Security and Responsible Disclosure
- For potential security issues, follow the private reporting instructions in `CONTRIBUTING.md` and do not open a public issue.
