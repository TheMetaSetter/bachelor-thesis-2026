#!/usr/bin/env bash
set -euo pipefail

MODE=""
ROLE="entrypoint"
GPU_COUNT=4
DATASET_ROOT="data/ServerMachineDataset"
OUTPUT_ROOT=""
SESSION_PREFIX="smd"
DRY_RUN=0
NO_TMUX=0
SKIP_COMPLETED=0
PRE_FLIGHT=0
ENTITY_IDS=()
GPU_MASKS=("0-7" "8-15" "16-23" "24-31")
CPU_MASKS=("32-37" "38-43")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --role) ROLE="$2"; shift 2 ;;
        --gpu-count) GPU_COUNT="$2"; shift 2 ;;
        --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --session-prefix) SESSION_PREFIX="$2"; shift 2 ;;
        --entity-id) ENTITY_IDS+=("$2"); shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --no-tmux) NO_TMUX=1; shift ;;
        --skip-completed) SKIP_COMPLETED=1; shift ;;
        --preflight) PRE_FLIGHT=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ "$MODE" != "smoke" && "$MODE" != "wet" ]]; then
    echo "--mode must be smoke or wet" >&2
    exit 2
fi
if ! [[ "$GPU_COUNT" =~ ^[0-9]+$ ]] || [[ "$GPU_COUNT" -ne 4 ]]; then
    echo "The cloud workflow requires exactly four GPUs" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"
MATRIX_SCRIPT="$REPO_ROOT/scripts/benchmarks/run_remaining_smd_matrix.sh"
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
LOG_ROOT="$OUTPUT_ROOT/tmux_logs/remaining_smd/$MODE"

run_generator_dry() {
    local -a args=(-m scripts.benchmarks.generate_remaining_smd_benchmark_configs
        --dataset-root "$DATASET_ROOT" --output-root "$OUTPUT_ROOT" --dry-run)
    if [[ "$MODE" == "smoke" ]]; then
        args+=(--smoke)
    fi
    if [[ "${#ENTITY_IDS[@]}" -gt 0 ]]; then
        for entity_id in "${ENTITY_IDS[@]}"; do
            args+=(--entity-id "$entity_id")
        done
    fi
    "$PYTHON" "${args[@]}"
}

write_manifest() {
    local -a args=(-m scripts.benchmarks.generate_remaining_smd_benchmark_configs
        --dataset-root "$DATASET_ROOT" --output-root "$OUTPUT_ROOT")
    if [[ "$MODE" == "smoke" ]]; then
        args+=(--smoke)
    fi
    if [[ "${#ENTITY_IDS[@]}" -gt 0 ]]; then
        for entity_id in "${ENTITY_IDS[@]}"; do
            args+=(--entity-id "$entity_id")
        done
    fi
    "$PYTHON" "${args[@]}"
}

preflight() {
    [[ -x "$PYTHON" ]] || { echo "Missing project Python: $PYTHON" >&2; return 2; }
    command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required" >&2; return 2; }
    command -v tmux >/dev/null 2>&1 || { echo "tmux is required" >&2; return 2; }
    command -v taskset >/dev/null 2>&1 || { echo "taskset is required" >&2; return 2; }
    local visible_gpu_count
    visible_gpu_count="$(nvidia-smi -L | awk 'NF { count += 1 } END { print count + 0 }')"
    if [[ "$visible_gpu_count" -lt 4 ]]; then
        echo "Expected at least four visible GPUs, found $visible_gpu_count" >&2
        return 2
    fi
    "$PYTHON" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() >= 4'
    local cpu_count
    cpu_count="$(nproc)"
    if [[ "$cpu_count" -lt 44 ]]; then
        echo "Expected at least 44 CPU cores, found $cpu_count" >&2
        return 2
    fi
    if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | awk 'NF { found = 1 } END { exit found }'; then
        :
    else
        echo "At least one target GPU already has a compute process" >&2
        return 2
    fi
    for split in train test test_label; do
        [[ -d "$DATASET_ROOT/$split" ]] || {
            echo "Missing dataset split: $DATASET_ROOT/$split" >&2
            return 2
        }
    done
    if [[ "${#ENTITY_IDS[@]}" -gt 0 ]]; then
        for entity_id in "${ENTITY_IDS[@]}"; do
            for split in train test test_label; do
                [[ -f "$DATASET_ROOT/$split/$entity_id.txt" ]] || {
                    echo "Missing entity file: $DATASET_ROOT/$split/$entity_id.txt" >&2
                    return 2
                }
            done
        done
    fi
    run_generator_dry
    echo "Preflight passed: GPUs=$visible_gpu_count CPUs=$cpu_count"
}

start_worker_session() {
    local phase="$1" resource_class="$2" index="$3" cpu_mask="$4" worker_count="$5"
    local session_name="${SESSION_PREFIX}-${phase}-${resource_class}-${index}"
    local marker="$LOG_ROOT/${session_name}.exit" log_path="$LOG_ROOT/${session_name}.log"
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "Refusing to overwrite existing tmux session: $session_name" >&2
        return 2
    fi
    local -a args=("$MATRIX_SCRIPT" --mode "$MODE" --role worker
        --gpu-count "$GPU_COUNT" --worker-index "$index" --worker-count "$worker_count"
        --resource-class "$resource_class" --phase-group "$phase" --cpu-mask "$cpu_mask"
        --dataset-root "$DATASET_ROOT" --output-root "$OUTPUT_ROOT"
        --completion-marker "$marker")
    if [[ "$resource_class" == "gpu" ]]; then
        args+=(--gpu-index "$index")
    fi
    if [[ "$SKIP_COMPLETED" -eq 1 ]]; then
        args+=(--skip-completed)
    fi
    printf -v worker_command '%q ' "${args[@]}"
    tmux new-session -d -s "$session_name" \
        "cd $(printf '%q' "$REPO_ROOT") && exec $worker_command"
    tmux pipe-pane -o -t "$session_name":0.0 "cat >> $(printf '%q' "$log_path")"
    echo "Started $session_name cpu_mask=$cpu_mask gpu=${index} log=$log_path"
}

start_phase() {
    local phase="$1"
    mkdir -p "$LOG_ROOT"
    local gpu cpu
    for gpu in 0 1 2 3; do
        start_worker_session "$phase" gpu "$gpu" "${GPU_MASKS[$gpu]}" 4
    done
    for cpu in 0 1; do
        start_worker_session "$phase" cpu "$cpu" "${CPU_MASKS[$cpu]}" 2
    done
}

wait_phase() {
    local phase="$1" status=0 class index session marker value limit
    for class in gpu cpu; do
        limit=4
        [[ "$class" == "cpu" ]] && limit=2
        for ((index = 0; index < limit; index++)); do
            session="${SESSION_PREFIX}-${phase}-${class}-${index}"
            marker="$LOG_ROOT/${session}.exit"
            while [[ ! -f "$marker" ]]; do
                sleep 2
            done
            value="$(tr -d '[:space:]' < "$marker")"
            echo "Completed $session status=$value"
            [[ "$value" == "0" ]] || status=1
        done
    done
    return "$status"
}

validate_dependencies() {
    "$PYTHON" - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
errors = []
for run in manifest.get("runs", []):
    if run.get("phase_group") != "online":
        continue
    entity = str(run["entity_id"]).replace("-", "_")
    seed = f"seed{int(run['seed'])}"
    for name, raw_path in run.get("dependencies", {}).items():
        path = Path(str(raw_path))
        if not path.is_file():
            errors.append(f"{run['run_id']}: missing {name}: {path}")
            continue
        if entity not in str(path) or seed not in str(path):
            errors.append(f"{run['run_id']}: mismatched {name}: {path}")
        if run.get("runner") == "thesis_online":
            offline_variant = str(run["variant"]).split("-", 1)[0]
            if offline_variant not in str(path) or "stage_b_fusion_finetuning" not in str(path):
                errors.append(f"{run['run_id']}: mismatched offline variant: {path}")
        if run.get("method") in {"candi", "m2n2"} and "redlamp_baseline" not in str(path):
            errors.append(f"{run['run_id']}: mismatched RedLamp dependency: {path}")
if errors:
    print("\n".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("Online dependencies validated")
PY
}

run_coordinator() {
    write_manifest
    start_phase offline
    if ! wait_phase offline; then
        echo "Offline phase failed; online phase was not started" >&2
        return 1
    fi
    validate_dependencies
    start_phase online
    if ! wait_phase online; then
        echo "Online phase failed; collector was not started" >&2
        return 1
    fi
    "$PYTHON" -m scripts.benchmarks.collect_remaining_smd_metrics --manifest "$MANIFEST"
}

if [[ "$DRY_RUN" -eq 1 ]]; then
    run_generator_dry
    for phase in offline online; do
        for gpu in 0 1 2 3; do
            echo "${SESSION_PREFIX}-${phase}-gpu-${gpu} cpu_mask=${GPU_MASKS[$gpu]} gpu=${gpu}"
        done
        for cpu in 0 1; do
            echo "${SESSION_PREFIX}-${phase}-cpu-${cpu} cpu_mask=${CPU_MASKS[$cpu]} gpu=none"
        done
    done
    exit 0
fi

if [[ "$PRE_FLIGHT" -eq 1 ]]; then
    preflight
    exit $?
fi
if [[ "$ROLE" == "coordinator" ]]; then
    run_coordinator
    exit $?
fi

preflight
if [[ "$NO_TMUX" -eq 1 ]]; then
    run_coordinator
    exit $?
fi
SESSION_NAME="${SESSION_PREFIX}-${MODE}"
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Refusing to overwrite existing tmux session: $SESSION_NAME" >&2
    exit 2
fi
coordinator_args=("$0" --mode "$MODE" --role coordinator --gpu-count "$GPU_COUNT"
    --dataset-root "$DATASET_ROOT" --output-root "$OUTPUT_ROOT")
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
echo "Started controller session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
