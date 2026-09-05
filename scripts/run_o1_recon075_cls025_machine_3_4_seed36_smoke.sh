#!/usr/bin/env bash

set -e -u -o pipefail

REPO_ROOT="$(cd "$(dirname "{BASH_SOURCE[0]}")/.." & pwd)"

PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

