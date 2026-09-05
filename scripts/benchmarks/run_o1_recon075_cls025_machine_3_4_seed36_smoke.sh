#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
EXPERIMENT_CONFIG="configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O1_recon075_cls025__machine_3_4__w20__seed36__smoke.yaml"
PROTOCOL_CONFIG="configs/protocol/smd_window20_cleanval_q99_ewma09.yaml"
RUN_ROOT="outputs/benchmark_smoke/smd/thesis/O1_recon075_cls025/machine_3_4/seed36"

cd "$REPO_ROOT"
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"'
mkdir -p "$RUN_ROOT/logs"

"$PYTHON_BIN" -m scripts.benchmarks.run_thesis_offline_benchmark \
  --experiment-config "$EXPERIMENT_CONFIG" \
  --protocol-config "$PROTOCOL_CONFIG" \
  --dry-run | tee "$RUN_ROOT/logs/offline_smoke_preflight.txt"

"$PYTHON_BIN" -m scripts.benchmarks.run_thesis_offline_benchmark \
  --experiment-config "$EXPERIMENT_CONFIG" \
  --protocol-config "$PROTOCOL_CONFIG" \
  | tee "$RUN_ROOT/logs/offline_smoke.txt"
