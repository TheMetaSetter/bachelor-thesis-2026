#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
LOG_DIR="${ONLINE_WET_LOG_DIR:-/tmp/online-wet-run-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$LOG_DIR"

declare -a FAILURES=()

run_one() {
    local label="$1"
    shift

    local log_file="$LOG_DIR/${label}.log"
    local status

    echo "START $label"
    if "$@" >"$log_file" 2>&1; then
        echo "PASS  $label"
        return 0
    fi

    status=$?
    echo "FAIL  $label (exit=$status)"
    tail -40 "$log_file"
    return "$status"
}

for offline_variant in O0 O1; do
    for online_variant in A0 A1 A2; do
        for entity_id in machine_1_6 machine_3_4 machine_3_9; do
            for seed in seed36 seed6 seed8; do
                label="thesis_${offline_variant}_${online_variant}_${entity_id}_${seed}"
                config="configs/experiment/online_benchmark/thesis/"
                config+="smd__thesis__online__${offline_variant}_${online_variant}"
                config+="__${entity_id}__w20__${seed}__main.yaml"

                if ! run_one "$label" "$PYTHON" \
                    scripts/benchmarks/run_thesis_online_benchmark.py \
                    --experiment-config "$config" \
                    --online-variant "$online_variant"; then
                    FAILURES+=("$label")
                fi
            done
        done
    done
done

for method in candi m2n2 iforest kmeans_ad stumpy; do
    for entity_id in machine_1_6 machine_3_4 machine_3_9; do
        for seed in seed36 seed6 seed8; do
            label="${method}_${entity_id}_${seed}"
            config="configs/experiment/online_benchmark/${method}/"
            config+="smd__${method}__online_main__${entity_id}__w20__${seed}__main.yaml"

            if ! run_one "$label" "$PYTHON" \
                scripts/benchmarks/run_online_streaming_benchmark.py \
                --benchmark-config "$config"; then
                FAILURES+=("$label")
            fi
        done
    done
done

echo
echo "Expected runs: 99"
echo "Failed runs: ${#FAILURES[@]}"
echo "Logs: $LOG_DIR"

if ((${#FAILURES[@]} > 0)); then
    printf 'Failed run: %s\n' "${FAILURES[@]}"
    exit 1
fi

echo "All 99 online wet runs completed successfully."
