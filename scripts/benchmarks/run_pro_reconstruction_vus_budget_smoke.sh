#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
MATRIX_ROOT="${REPO_ROOT}/outputs/benchmark_pro_reconstruction_vus_budget"
CONFIG_ROOT="${MATRIX_ROOT}/generated_configs"
CONFIG_PATH="${CONFIG_ROOT}/smd__thesis__offline__O0_recon075_cls025_direct__machine_1_6__w20__seed6__fpr001__main.yaml"
PROTOCOL_CONFIG="configs/protocol/smd_window20_synthnormal_q99_normalized_input_mse_ewma09.yaml"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" -m scripts.benchmarks.generate_pro_reconstruction_vus_budget_matrix \
  --output-dir "${CONFIG_ROOT}"

"${PYTHON_BIN}" -m scripts.benchmarks.run_thesis_offline_benchmark \
  --experiment-config "${CONFIG_PATH}" \
  --protocol-config "${PROTOCOL_CONFIG}" \
  --dry-run
