#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
PROTOCOL_CONFIG="configs/protocol/smd_window20_cleanval_q99_ewma09.yaml"
MATRIX_ROOT="outputs/benchmark_full_direct_recon075_cls025"
CONFIG_ROOT="${MATRIX_ROOT}/generated_configs"

cd "$REPO_ROOT"

"$PYTHON_BIN" -m scripts.benchmarks.generate_full_direct_recon075_cls025_matrix \
  --output-dir "$CONFIG_ROOT"

for variant in O0 O1; do
  for entity in machine_1_6 machine_3_4 machine_3_9; do
    for seed in 6 8 36; do
      config="${CONFIG_ROOT}/smd__thesis__offline__${variant}_recon075_cls025_direct__${entity}__w20__seed${seed}__main.yaml"
      run_root="${MATRIX_ROOT}/smd/thesis/${variant}/${entity}/seed${seed}"
      mkdir -p "${run_root}/logs"
      "$PYTHON_BIN" -m scripts.benchmarks.run_thesis_offline_benchmark \
        --experiment-config "$config" \
        --protocol-config "$PROTOCOL_CONFIG" \
        --dry-run | tee "${run_root}/logs/preflight.txt"
    done
  done
done

for variant in O0 O1; do
  for entity in machine_1_6 machine_3_4 machine_3_9; do
    for seed in 6 8 36; do
      config="${CONFIG_ROOT}/smd__thesis__offline__${variant}_recon075_cls025_direct__${entity}__w20__seed${seed}__main.yaml"
      run_root="${MATRIX_ROOT}/smd/thesis/${variant}/${entity}/seed${seed}"
      "$PYTHON_BIN" -m scripts.benchmarks.run_thesis_offline_benchmark \
        --experiment-config "$config" \
        --protocol-config "$PROTOCOL_CONFIG" \
        | tee "${run_root}/logs/run.txt"
    done
  done
done
