#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_EXPERIMENT_CONFIG="configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O0__machine_3_4__w20__seed6__main.yaml"
DEFAULT_PROTOCOL_CONFIG="configs/protocol/smd_window20_cleanval_q99_ewma09.yaml"
DEFAULT_SESSION_NAME="two-stage-benchmark-machine-3-4-seed6"
DEFAULT_PYTHON_BIN=".venv/bin/python"
DEFAULT_GPU_INDEX="0"
DEFAULT_REQUIRED_GPU_NAME_SUBSTRING="RTX 3090"

EXPERIMENT_CONFIG="${DEFAULT_EXPERIMENT_CONFIG}"
PROTOCOL_CONFIG="${DEFAULT_PROTOCOL_CONFIG}"
SESSION_NAME="${DEFAULT_SESSION_NAME}"
PYTHON_BIN="${DEFAULT_PYTHON_BIN}"
GPU_INDEX="${DEFAULT_GPU_INDEX}"
REQUIRED_GPU_NAME_SUBSTRING="${DEFAULT_REQUIRED_GPU_NAME_SUBSTRING}"
SKIP_COMPLETED=false
DRY_RUN=false
PREVIEW_ONLY=false
REPLACE_SESSION=false

LOG_DIR="${REPO_ROOT}/outputs/tmux_logs"
LOG_PATH=""

print_usage() {
  cat <<'EOF'
Usage: scripts/launch_tmux_two_stage_experiment.sh [options]

Options:
  --experiment-config PATH
  --protocol-config PATH
  --session-name NAME
  --python-bin PATH
  --gpu-index INDEX
  --required-gpu-name-substring TEXT
  --skip-completed
  --dry-run
  --preflight-only
  --replace-session
  --help
EOF
}

join_quoted_args() {
  local joined=""
  local argument
  for argument in "$@"; do
    printf -v joined '%s%q ' "${joined}" "${argument}"
  done
  printf '%s' "${joined% }"
}

join_display_args() {
  local joined=""
  local argument
  for argument in "$@"; do
    joined+="${argument} "
  done
  printf '%s' "${joined% }"
}

resolve_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s/%s\n' "${REPO_ROOT}" "${path}"
  fi
}

resolve_output_dir_path() {
  local resolved_config_path="$1"
  local output_dir
  output_dir="$(awk -F': ' '/^output_dir:/ {print $2; exit}' "${resolved_config_path}")"
  if [[ -z "${output_dir}" ]]; then
    printf '%s\n' "${REPO_ROOT}/outputs/unknown"
    return
  fi
  if [[ "${output_dir}" = /* ]]; then
    printf '%s\n' "${output_dir}"
  else
    printf '%s/%s\n' "${REPO_ROOT}" "${output_dir}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-config)
      EXPERIMENT_CONFIG="$2"
      shift 2
      ;;
    --protocol-config)
      PROTOCOL_CONFIG="$2"
      shift 2
      ;;
    --session-name)
      SESSION_NAME="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --gpu-index)
      GPU_INDEX="$2"
      shift 2
      ;;
    --required-gpu-name-substring)
      REQUIRED_GPU_NAME_SUBSTRING="$2"
      shift 2
      ;;
    --skip-completed)
      SKIP_COMPLETED=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --preflight-only)
      PREVIEW_ONLY=true
      shift
      ;;
    --replace-session)
      REPLACE_SESSION=true
      shift
      ;;
    --help|-h)
      print_usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      print_usage >&2
      exit 2
      ;;
  esac
done

if [[ "${DRY_RUN}" == true && "${PREVIEW_ONLY}" == true ]]; then
  printf '%s\n' "--dry-run and --preflight-only cannot be used together." >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${SESSION_NAME}.log"
RESOLVED_EXPERIMENT_CONFIG_PATH="$(resolve_path "${EXPERIMENT_CONFIG}")"
RESOLVED_PROTOCOL_CONFIG_PATH="$(resolve_path "${PROTOCOL_CONFIG}")"
RESOLVED_OUTPUT_DIR="$(resolve_output_dir_path "${RESOLVED_EXPERIMENT_CONFIG_PATH}")"
BENCHMARK_REPORT_PATH="${RESOLVED_OUTPUT_DIR}/benchmark/thesis_offline_benchmark_report.json"

BENCHMARK_COMMAND=(
  "${PYTHON_BIN}"
  "scripts/run_thesis_offline_benchmark.py"
  "--experiment-config"
  "${EXPERIMENT_CONFIG}"
  "--protocol-config"
  "${PROTOCOL_CONFIG}"
)
if [[ "${SKIP_COMPLETED}" == true ]]; then
  BENCHMARK_COMMAND+=("--skip-completed")
fi

BENCHMARK_COMMAND_STRING="$(join_quoted_args "${BENCHMARK_COMMAND[@]}")"
BENCHMARK_COMMAND_DISPLAY="$(join_display_args "${BENCHMARK_COMMAND[@]}")"

printf -v REPO_ROOT_QUOTED '%q' "${REPO_ROOT}"
printf -v LOG_PATH_QUOTED '%q' "${LOG_PATH}"
printf -v GPU_INDEX_QUOTED '%q' "${GPU_INDEX}"

TMUX_INNER_COMMAND="(cd ${REPO_ROOT_QUOTED} && CUDA_VISIBLE_DEVICES=${GPU_INDEX_QUOTED} ${BENCHMARK_COMMAND_STRING}) > ${LOG_PATH_QUOTED} 2>&1"

if [[ "${DRY_RUN}" == true ]]; then
  printf 'tmux session: %s\n' "${SESSION_NAME}"
  printf 'log path: %s\n' "${LOG_PATH}"
  printf 'benchmark report path: %s\n' "${BENCHMARK_REPORT_PATH}"
  printf 'experiment config: %s\n' "${RESOLVED_EXPERIMENT_CONFIG_PATH}"
  printf 'protocol config: %s\n' "${RESOLVED_PROTOCOL_CONFIG_PATH}"
  printf 'attach command: tmux attach -t %s\n' "${SESSION_NAME}"
  printf 'benchmark command: CUDA_VISIBLE_DEVICES=%s %s\n' "${GPU_INDEX}" "${BENCHMARK_COMMAND_DISPLAY}"
  printf 'tmux inner command: (cd %s && CUDA_VISIBLE_DEVICES=%s %s) > %s 2>&1\n' \
    "${REPO_ROOT}" \
    "${GPU_INDEX}" \
    "${BENCHMARK_COMMAND_DISPLAY}" \
    "${LOG_PATH}"
  exit 0
fi

if [[ "${PREVIEW_ONLY}" == true ]]; then
  cd "${REPO_ROOT}"
  exec env CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${BENCHMARK_COMMAND[@]}"
fi

if ! command -v tmux >/dev/null 2>&1; then
  printf 'tmux is required but was not found in PATH.\n' >&2
  exit 127
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  if [[ "${REPLACE_SESSION}" == true ]]; then
    tmux kill-session -t "${SESSION_NAME}"
  else
    printf 'tmux session already exists: %s\nUse --replace-session to replace it.\n' "${SESSION_NAME}" >&2
    exit 3
  fi
fi

tmux new-session -d -s "${SESSION_NAME}" "bash -lc $(printf '%q' "${TMUX_INNER_COMMAND}")"

printf 'tmux session: %s\n' "${SESSION_NAME}"
printf 'log path: %s\n' "${LOG_PATH}"
printf 'benchmark report path: %s\n' "${BENCHMARK_REPORT_PATH}"
printf 'attach command: tmux attach -t %s\n' "${SESSION_NAME}"
