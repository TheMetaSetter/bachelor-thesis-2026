#!/usr/bin/env bash
set -euo pipefail

# Install TSAD from the directory that contains this script.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if ! command -v git >/dev/null; then
    printf '%s\n' 'Git is required. Run sudo apt-get update and sudo apt-get install git.' >&2
    exit 1
fi

uv_bin_dir="${XDG_BIN_HOME:-${HOME}/.local/bin}"
if command -v uv >/dev/null; then
    uv_command=uv
else
    uv_command="${uv_bin_dir}/uv"
    if [[ ! -x "$uv_command" ]]; then
        if ! command -v curl >/dev/null; then
            printf '%s\n' 'curl is required. Run sudo apt-get update and sudo apt-get install curl ca-certificates.' >&2
            exit 1
        fi
        printf '%s\n' 'Install uv for the current user.'
        curl --proto '=https' --tlsv1.2 -fsSL https://astral.sh/uv/install.sh |
            UV_UNMANAGED_INSTALL="$uv_bin_dir" sh
    fi
fi

# Use Python 3.12 for a new environment. Keep an existing compatible environment.
if [[ ! -e .venv && ! -L .venv ]]; then
    printf '%s\n' 'Create .venv with Python 3.12.'
    "$uv_command" venv --python 3.12 .venv
fi

if [[ ! -x .venv/bin/python ]] ||
    ! .venv/bin/python -c 'import sys; sys.exit(sys.version_info < (3, 12))'; then
    printf '%s\n' 'Use Python 3.12 or newer in .venv. Move the old .venv and run setup again.' >&2
    exit 1
fi

printf '%s\n' 'Install TSAD and its required packages.'
"$uv_command" pip install --python .venv/bin/python --editable .
"$uv_command" pip check --python .venv/bin/python
.venv/bin/tsad list datasets

printf '\n%s\n' 'Setup complete. From this directory, run .venv/bin/tsad to open the terminal interface.'
