#!/usr/bin/env bash
set -euo pipefail

MODE=""
ROLE="entrypoint"
GPU_INDEX=""
GPU_COUNT=2
WORKER_INDEX=""
WORKER_COUNT=""
RESOURCE_CLASS=""
PHASE_GROUP="all"
CPU_MASK=""
DATASET_ROOT="data/ServerMachineDataset"
OUTPUT_ROOT=""
SESSION_PREFIX="remaining-smd"
DRY_RUN=0
NO_TMUX=0
SKIP_COMPLETED=0
PRE_FLIGHT=0
COMPLETION_MARKER=""
ENTITY_IDS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --role) ROLE="$2"; shift 2 ;;
        --gpu-index) GPU_INDEX="$2"; shift 2 ;;
        --gpu-count) GPU_COUNT="$2"; shift 2 ;;
        --worker-index) WORKER_INDEX="$2"; shift 2 ;;
        --worker-count) WORKER_COUNT="$2"; shift 2 ;;
        --resource-class) RESOURCE_CLASS="$2"; shift 2 ;;
        --phase-group) PHASE_GROUP="$2"; shift 2 ;;
        --cpu-mask) CPU_MASK="$2"; shift 2 ;;
        --entity-id) ENTITY_IDS+=("$2"); shift 2 ;;
        --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --session-prefix) SESSION_PREFIX="$2"; shift 2 ;;
        --completion-marker) COMPLETION_MARKER="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --preflight) PRE_FLIGHT=1; shift ;;
        --no-tmux) NO_TMUX=1; shift ;;
        --skip-completed) SKIP_COMPLETED=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ "$MODE" != "smoke" && "$MODE" != "wet" ]]; then
    echo "--mode must be smoke or wet" >&2
    exit 2
fi
if ! [[ "$GPU_COUNT" =~ ^[0-9]+$ ]] || [[ "$GPU_COUNT" -lt 1 ]]; then
    echo "--gpu-count must be a positive integer" >&2
    exit 2
fi
if [[ "$PHASE_GROUP" != "all" && "$PHASE_GROUP" != "offline" \
    && "$PHASE_GROUP" != "online" ]]; then
    echo "--phase-group must be all, offline, or online" >&2
    exit 2
fi
if [[ -n "$RESOURCE_CLASS" && "$RESOURCE_CLASS" != "gpu" \
    && "$RESOURCE_CLASS" != "cpu" ]]; then
    echo "--resource-class must be gpu or cpu" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"
RESOURCE_ENV="$REPO_ROOT/scripts/benchmarks/_remaining_smd_resource_env.sh"
if [[ -z "$OUTPUT_ROOT" ]]; then
    if [[ "$MODE" == "smoke" ]]; then
        OUTPUT_ROOT="$REPO_ROOT/outputs/benchmark_smoke/smd_remaining"
    else
        OUTPUT_ROOT="$REPO_ROOT/outputs/benchmark/smd_remaining"
    fi
fi
if [[ "$DATASET_ROOT" != /* ]]; then
    DATASET_ROOT="$REPO_ROOT/$DATASET_ROOT"
fi
if [[ "$OUTPUT_ROOT" != /* ]]; then
    OUTPUT_ROOT="$REPO_ROOT/$OUTPUT_ROOT"
fi
MANIFEST="$OUTPUT_ROOT/remaining_smd_manifest.json"
PROTOCOL_CONFIG="$REPO_ROOT/configs/protocol/smd_window20_cleanval_q99_ewma09.yaml"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing project Python: $PYTHON" >&2
    exit 2
fi

generate_manifest() {
    local generator_args=(
        -m scripts.benchmarks.generate_remaining_smd_benchmark_configs
        --dataset-root "$DATASET_ROOT"
        --output-root "$OUTPUT_ROOT"
    )
    if [[ "$MODE" == "smoke" ]]; then
        generator_args+=(--smoke)
    fi
    if [[ "${#ENTITY_IDS[@]}" -gt 0 ]]; then
        for entity_id in "${ENTITY_IDS[@]}"; do
            generator_args+=(--entity-id "$entity_id")
        done
    fi
    if [[ "$DRY_RUN" -eq 1 || "$PRE_FLIGHT" -eq 1 ]]; then
        "$PYTHON" "${generator_args[@]}" --dry-run
    else
        "$PYTHON" "${generator_args[@]}"
    fi
}

run_command() {
    if [[ -n "$CPU_MASK" && ( "$RESOURCE_CLASS" == "gpu" || "$RESOURCE_CLASS" == "cpu" ) ]]; then
        source "$RESOURCE_ENV"
        run_with_remaining_smd_resources "$RESOURCE_CLASS" "$CPU_MASK" "$GPU_INDEX" "$@"
    else
        "$@"
    fi
}

run_one() {
    local runner="$1"
    local config_path="$2"
    local variant="$3"
    local output_dir="$4"
    local report_path="$5"
    if [[ "$SKIP_COMPLETED" -eq 1 && -f "$report_path" ]]; then
        echo "SKIP $report_path"
        return
    fi
    local -a command
    case "$runner" in
        thesis_offline)
            command=("$PYTHON" -m scripts.benchmarks.run_thesis_offline_benchmark
                --experiment-config "$config_path" --protocol-config "$PROTOCOL_CONFIG")
            if [[ "$SKIP_COMPLETED" -eq 1 ]]; then
                command+=(--skip-completed)
            fi
            run_command "${command[@]}"
            ;;
        redlamp)
            command=("$PYTHON" -m scripts.train --experiment-config "$config_path")
            run_command "${command[@]}"
            command=("$PYTHON" -m scripts.evaluate --experiment-config "$config_path"
                --checkpoint-path "$output_dir/checkpoints/best.pt"
                --protocol-config "$PROTOCOL_CONFIG")
            run_command "${command[@]}"
            ;;
        offline_baseline)
            command=("$PYTHON" -m scripts.benchmarks.run_offline_benchmark
                --benchmark-config "$config_path" --protocol-config "$PROTOCOL_CONFIG")
            run_command "${command[@]}"
            ;;
        thesis_online)
            command=("$PYTHON" -m scripts.benchmarks.run_thesis_online_benchmark
                --experiment-config "$config_path" --protocol-config "$PROTOCOL_CONFIG"
                --online-variant "${variant#*-}")
            run_command "${command[@]}"
            ;;
        online_baseline)
            command=("$PYTHON" -m scripts.benchmarks.run_online_streaming_benchmark
                --benchmark-config "$config_path" --protocol-config "$PROTOCOL_CONFIG")
            run_command "${command[@]}"
            ;;
        *)
            echo "Unknown runner: $runner" >&2
            return 2
            ;;
    esac
}

run_worker() {
    local worker_class="${RESOURCE_CLASS:-all}"
    local worker_index="${WORKER_INDEX:-${GPU_INDEX:-0}}"
    local worker_count="${WORKER_COUNT:-$GPU_COUNT}"
    if ! [[ "$worker_index" =~ ^[0-9]+$ && "$worker_count" =~ ^[1-9][0-9]*$ ]]; then
        echo "Worker index/count must be integers" >&2
        return 2
    fi
    if [[ "$worker_class" == "gpu" && -z "$GPU_INDEX" ]]; then
        echo "GPU worker requires --gpu-index" >&2
        return 2
    fi
    if [[ "$worker_class" == "cpu" && -n "$GPU_INDEX" ]]; then
        echo "CPU worker must not receive --gpu-index" >&2
        return 2
    fi
    if [[ -n "$COMPLETION_MARKER" ]]; then
        mkdir -p "$(dirname "$COMPLETION_MARKER")"
    fi
    echo "WORKER class=$worker_class index=$worker_index/$worker_count phase=$PHASE_GROUP cpu_mask=${CPU_MASK:-none} gpu=${GPU_INDEX:-none}"
    local selected_entities=""
    if [[ "${#ENTITY_IDS[@]}" -gt 0 ]]; then
        selected_entities="$(IFS=,; echo "${ENTITY_IDS[*]}")"
    fi
    local worker_status=0
    while IFS=$'\t' read -r run_id runner config_path variant output_dir report_path; do
        [[ -z "$run_id" ]] && continue
        echo "WORKER class=$worker_class START $run_id"
        if ! run_one "$runner" "$config_path" "$variant" "$output_dir" "$report_path"; then
            echo "WORKER class=$worker_class FAIL $run_id" >&2
            worker_status=1
            break
        fi
        echo "WORKER class=$worker_class DONE $run_id"
    done < <(
        "$PYTHON" - "$MANIFEST" "$PHASE_GROUP" "$worker_class" "$worker_index" "$worker_count" "$selected_entities" <<'PY'
import json
import sys

manifest_path, phase_group, resource_class, worker_index, worker_count, selected = sys.argv[1:]
manifest = json.loads(open(manifest_path, encoding="utf-8").read())
selected_entities = set(filter(None, selected.split(",")))
owned_index = 0
for run in manifest.get("runs", []):
    if phase_group != "all" and run.get("phase_group", run.get("phase")) != phase_group:
        continue
    if resource_class != "all" and run.get("resource_class") != resource_class:
        continue
    if selected_entities and run.get("entity_id") not in selected_entities:
        continue
    if owned_index % int(worker_count) != int(worker_index):
        owned_index += 1
        continue
    owned_index += 1
    print("\t".join(str(run.get(key) or "") for key in ("run_id", "runner", "config_path", "variant", "output_dir", "report_path")))
PY
    )
    return "$worker_status"
}

run_coordinator() {
    local worker_pids=()
    local gpu
    for gpu in 0 1; do
        (
            "$0" --mode "$MODE" --role worker --gpu-index "$gpu" --gpu-count 2 \
                --dataset-root "$DATASET_ROOT" --output-root "$OUTPUT_ROOT" \
                --no-tmux $([[ "$SKIP_COMPLETED" -eq 1 ]] && echo --skip-completed)
        ) &
        worker_pids+=("$!")
    done
    local status=0
    for pid in "${worker_pids[@]}"; do
        wait "$pid" || status=1
    done
    if [[ "$status" -eq 0 ]]; then
        "$PYTHON" -m scripts.benchmarks.collect_remaining_smd_metrics --manifest "$MANIFEST"
    fi
    return "$status"
}

if [[ "$ROLE" == "worker" ]]; then
    set +e
    run_worker
    status=$?
    set -e
    if [[ -n "$COMPLETION_MARKER" ]]; then
        printf '%s\n' "$status" > "$COMPLETION_MARKER"
    fi
    exit "$status"
fi

if [[ "$DRY_RUN" -eq 1 || "$PRE_FLIGHT" -eq 1 ]]; then
    generate_manifest
    for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
        echo "GPU $gpu: filtered queue"
    done
    echo "Metrics: VUS-PR@FPR-budget, VUS-PR, Affiliation F1-score, VUS-ROC, raw-FPR"
    exit 0
fi

generate_manifest
if [[ "$NO_TMUX" -eq 1 ]]; then
    run_coordinator
    exit $?
fi
if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required unless --no-tmux is set" >&2
    exit 2
fi
SESSION_NAME="$SESSION_PREFIX-$MODE"
coordinator_args=("$0" --mode "$MODE" --role coordinator
    --dataset-root "$DATASET_ROOT" --output-root "$OUTPUT_ROOT" --no-tmux)
if [[ "${#ENTITY_IDS[@]}" -gt 0 ]]; then
    for entity_id in "${ENTITY_IDS[@]}"; do
        coordinator_args+=(--entity-id "$entity_id")
    done
fi
if [[ "$SKIP_COMPLETED" -eq 1 ]]; then
    coordinator_args+=(--skip-completed)
fi
printf -v coordinator_command '%q ' "${coordinator_args[@]}"
tmux new-session -d -s "$SESSION_NAME" \
    "cd $(printf '%q' "$REPO_ROOT") && exec $coordinator_command"
echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
