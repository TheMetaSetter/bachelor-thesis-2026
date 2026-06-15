# setup guide

## 1. checkout the `dev` branch

```bash
git checkout dev
```

if you are not sure which branch you are on, check first:

```bash
git branch --show-current
```

## 2. run the setup script

from the repo root, run:

```bash
bash setup.sh
```

if your gpu server needs a specific pytorch cuda wheel, set the extra index url first:

```bash
TORCH_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu124" bash setup.sh
```

## 3. verify the environment

after the script finishes, activate the virtual environment if you want to work in the shell:

```bash
source $HOME/.local/bin/env
source .venv/bin/activate
```

quick checks:

```bash
python --version
python -c "import torch; print(torch.cuda.is_available())"
```

## notes

- the script installs `uv`, creates a python 3.12 virtual environment, and installs the python packages in `requirements.txt`.
- timezone is set to `asia/ho_chi_minh` during setup.
- if `dev` does not exist yet, use the branch name that your team keeps for shared setup work.
