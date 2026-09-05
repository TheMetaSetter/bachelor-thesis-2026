#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
MATRIX_RUNNER="${REPO_ROOT}/scripts/benchmarks/run_full_direct_recon075_cls025_offline_matrix.sh"

cd "$REPO_ROOT"
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"'
exec bash "$MATRIX_RUNNER"
