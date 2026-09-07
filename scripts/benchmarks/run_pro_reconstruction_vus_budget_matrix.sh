#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
MATRIX_ROOT="${REPO_ROOT}/outputs/benchmark_pro_reconstruction_vus_budget"
CONFIG_ROOT="${MATRIX_ROOT}/generated_configs"
LOG_ROOT="${MATRIX_ROOT}/logs"
PROTOCOL_CONFIG="configs/protocol/smd_window20_synthnormal_q99_normalized_input_mse_ewma09.yaml"

cd "${REPO_ROOT}"
mkdir -p "${LOG_ROOT}"

"${PYTHON_BIN}" -m scripts.benchmarks.generate_pro_reconstruction_vus_budget_matrix \
  --output-dir "${CONFIG_ROOT}"

CONFIGS=()
while IFS= read -r config; do
  CONFIGS+=("${config}")
done < <(find "${CONFIG_ROOT}" -maxdepth 1 -type f -name '*.yaml' | sort)
if [[ "${#CONFIGS[@]}" -ne 54 ]]; then
  echo "Expected 54 generated configs, found ${#CONFIGS[@]}" >&2
  exit 1
fi

for config in "${CONFIGS[@]}"; do
  log_path="${LOG_ROOT}/$(basename "${config%.yaml}").preflight.log"
  "${PYTHON_BIN}" -m scripts.benchmarks.run_thesis_offline_benchmark \
    --experiment-config "${config}" \
    --protocol-config "${PROTOCOL_CONFIG}" \
    --dry-run | tee "${log_path}"
done

for config in "${CONFIGS[@]}"; do
  log_path="${LOG_ROOT}/$(basename "${config%.yaml}").log"
  "${PYTHON_BIN}" -m scripts.benchmarks.run_thesis_offline_benchmark \
    --experiment-config "${config}" \
    --protocol-config "${PROTOCOL_CONFIG}" \
    --skip-completed | tee "${log_path}"
done
