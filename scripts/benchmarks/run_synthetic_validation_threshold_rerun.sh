#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
PROTOCOL_CONFIG="configs/protocol/smd_window20_synthnormal_q99_ewma09.yaml"
RERUN_ROOT="outputs/benchmark_rerun/smd/thesis_synthnormal_q99_$(date +%Y%m%d_%H%M%S)"

cd "$REPO_ROOT"

for variant in O0 O1; do
  for entity in machine_1_6 machine_3_4 machine_3_9; do
    for seed in 6 8 36; do
      checkpoint="outputs/benchmark/smd/thesis/${variant}/${entity}/seed${seed}/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt"
      test -f "$checkpoint" || { echo "missing: $checkpoint" >&2; exit 1; }
    done
  done
done

mkdir -p "$(dirname "$RERUN_ROOT")"
mkdir "$RERUN_ROOT"

for variant in O0 O1; do
  for entity in machine_1_6 machine_3_4 machine_3_9; do
    for seed in 6 8 36; do
      experiment_config="configs/experiment/offline_benchmark/thesis/smd__thesis__offline__${variant}__${entity}__w20__seed${seed}__main.yaml"
      checkpoint_path="outputs/benchmark/smd/thesis/${variant}/${entity}/seed${seed}/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt"
      output_dir="${RERUN_ROOT}/${variant}/${entity}/seed${seed}"

      "$PYTHON_BIN" -m scripts.benchmarks.run_thesis_offline_benchmark \
        --experiment-config "$experiment_config" \
        --protocol-config "$PROTOCOL_CONFIG" \
        --evaluation-only \
        --checkpoint-path "$checkpoint_path" \
        --output-dir "$output_dir"
    done
  done
done

echo "Rerun outputs: ${REPO_ROOT}/${RERUN_ROOT}"
