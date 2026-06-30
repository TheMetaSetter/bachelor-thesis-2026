#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_SESSION_NAME="benchmark-smd-top3-two-seeds"
DEFAULT_PYTHON_BIN=".venv/bin/python"
DEFAULT_REPORT_DIR="outputs/benchmark_smd_reports/top3-two-seeds"
DEFAULT_GPU_INDEX="0"
DEFAULT_REQUIRED_GPU_NAME_SUBSTRING="RTX 3090"
DEFAULT_SMOKE_PROFILE="none"

SESSION_NAME="${DEFAULT_SESSION_NAME}"
PYTHON_BIN="${DEFAULT_PYTHON_BIN}"
REPORT_DIR="${DEFAULT_REPORT_DIR}"
GPU_INDEX="${DEFAULT_GPU_INDEX}"
REQUIRED_GPU_NAME_SUBSTRING="${DEFAULT_REQUIRED_GPU_NAME_SUBSTRING}"
DATA_NUM_WORKERS_OVERRIDE=""
SMOKE_PROFILE="${DEFAULT_SMOKE_PROFILE}"
SKIP_COMPLETED=false
DRY_RUN=false
PREFLIGHT_ONLY=false
REPLACE_SESSION=false

LOG_DIR="${REPO_ROOT}/outputs/tmux_logs"
LOG_PATH=""

FUNCTIONAL_SMOKE_CONFIG_PATHS=(
  "configs/experiment/comparative/baseline/smd__redlamp_baseline__comparative-single-stage-machine_1_6__w20__seed6__smoke.yaml"
  "configs/experiment/comparative/thesis/smd__thesis_multitask__comparative-three-stage-machine_1_6__w20__seed6__smoke.yaml"
)

STRESS_SMOKE_CONFIG_PATHS=(
  "configs/experiment/comparative_stress_smoke/baseline/smd__redlamp_baseline__comparative-single-stage-machine_1_6__w20__seed6__stress-smoke.yaml"
  "configs/experiment/comparative_stress_smoke/thesis/smd__thesis_multitask__comparative-three-stage-machine_1_6__w20__seed6__stress-smoke.yaml"
)

MAIN_CONFIG_PATHS=(
  "configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_1_6__w20__seed6__main.yaml"
  "configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_1_6__w20__seed36__main.yaml"
  "configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_4__w20__seed6__main.yaml"
  "configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_4__w20__seed36__main.yaml"
  "configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_9__w20__seed6__main.yaml"
  "configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_9__w20__seed36__main.yaml"
  "configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-three-stage-machine_1_6__w20__seed6__main.yaml"
  "configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-three-stage-machine_1_6__w20__seed36__main.yaml"
  "configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-three-stage-machine_3_4__w20__seed6__main.yaml"
  "configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-three-stage-machine_3_4__w20__seed36__main.yaml"
  "configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-three-stage-machine_3_9__w20__seed6__main.yaml"
  "configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-three-stage-machine_3_9__w20__seed36__main.yaml"
)

print_usage() {
  cat <<'EOF'
Usage: scripts/launch_tmux_comparative_smd_experiment.sh [options]

Options:
  --session-name NAME
  --python-bin PATH
  --report-dir PATH
  --gpu-index INDEX
  --required-gpu-name-substring TEXT
  --data-num-workers-override COUNT
  --smoke-profile functional|stress|none
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

resolve_report_dir_path() {
  local report_dir="$1"
  if [[ "${report_dir}" = /* ]]; then
    printf '%s\n' "${report_dir}"
  else
    printf '%s/%s\n' "${REPO_ROOT}" "${report_dir}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-name)
      SESSION_NAME="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --report-dir)
      REPORT_DIR="$2"
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
    --data-num-workers-override)
      DATA_NUM_WORKERS_OVERRIDE="$2"
      shift 2
      ;;
    --smoke-profile)
      SMOKE_PROFILE="$2"
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
      PREFLIGHT_ONLY=true
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

if [[ "${DRY_RUN}" == true && "${PREFLIGHT_ONLY}" == true ]]; then
  printf '%s\n' "--dry-run and --preflight-only cannot be used together." >&2
  exit 2
fi

SMOKE_CONFIG_PATHS=()
case "${SMOKE_PROFILE}" in
  functional)
    SMOKE_CONFIG_PATHS=("${FUNCTIONAL_SMOKE_CONFIG_PATHS[@]}")
    ;;
  stress)
    SMOKE_CONFIG_PATHS=("${STRESS_SMOKE_CONFIG_PATHS[@]}")
    ;;
  none)
    SMOKE_CONFIG_PATHS=()
    ;;
  *)
    printf 'Unknown --smoke-profile value: %s\n' "${SMOKE_PROFILE}" >&2
    exit 2
    ;;
esac

mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${SESSION_NAME}.log"
RESOLVED_REPORT_DIR="$(resolve_report_dir_path "${REPORT_DIR}")"
MANIFEST_PATH="${RESOLVED_REPORT_DIR}/comparative_manifest.json"
EXECUTION_REPORT_PATH="${RESOLVED_REPORT_DIR}/comparative_execution_report.json"
PREFLIGHT_SUMMARY_PATH="${RESOLVED_REPORT_DIR}/comparative_server_preflight_summary.json"

RUNNER_COMMAND=(
  "${PYTHON_BIN}"
  "scripts/run_comparative_smd_experiments.py"
  "--report-dir"
  "${REPORT_DIR}"
)
if [[ -n "${DATA_NUM_WORKERS_OVERRIDE}" ]]; then
  RUNNER_COMMAND+=("--data-num-workers-override" "${DATA_NUM_WORKERS_OVERRIDE}")
fi
if [[ "${SKIP_COMPLETED}" == true ]]; then
  RUNNER_COMMAND+=("--skip-completed")
fi
if [[ ${#SMOKE_CONFIG_PATHS[@]} -gt 0 ]]; then
  RUNNER_COMMAND+=("--smoke-config-paths")
  RUNNER_COMMAND+=("${SMOKE_CONFIG_PATHS[@]}")
fi
RUNNER_COMMAND+=("--config-paths")
RUNNER_COMMAND+=("${MAIN_CONFIG_PATHS[@]}")

PREFLIGHT_COMMAND=(
  "${PYTHON_BIN}"
  "scripts/preflight_comparative_smd_server.py"
  "--report-dir"
  "${REPORT_DIR}"
  "--gpu-index"
  "${GPU_INDEX}"
  "--required-gpu-name-substring"
  "${REQUIRED_GPU_NAME_SUBSTRING}"
)
if [[ -n "${DATA_NUM_WORKERS_OVERRIDE}" ]]; then
  PREFLIGHT_COMMAND+=("--data-num-workers-override" "${DATA_NUM_WORKERS_OVERRIDE}")
fi
if [[ ${#SMOKE_CONFIG_PATHS[@]} -gt 0 ]]; then
  PREFLIGHT_COMMAND+=("--smoke-config-paths")
  PREFLIGHT_COMMAND+=("${SMOKE_CONFIG_PATHS[@]}")
fi
PREFLIGHT_COMMAND+=("--config-paths")
PREFLIGHT_COMMAND+=("${MAIN_CONFIG_PATHS[@]}")
PREFLIGHT_COMMAND+=("--print-json")
if [[ "${PREFLIGHT_ONLY}" != true ]]; then
  PREFLIGHT_COMMAND+=("--require-launch-ready")
fi

PREFLIGHT_COMMAND_STRING="$(join_quoted_args "${PREFLIGHT_COMMAND[@]}")"
RUNNER_COMMAND_STRING="$(join_quoted_args "${RUNNER_COMMAND[@]}")"
PREFLIGHT_COMMAND_DISPLAY="$(join_display_args "${PREFLIGHT_COMMAND[@]}")"
RUNNER_COMMAND_DISPLAY="$(join_display_args "${RUNNER_COMMAND[@]}")"

printf -v REPO_ROOT_QUOTED '%q' "${REPO_ROOT}"
printf -v LOG_PATH_QUOTED '%q' "${LOG_PATH}"
printf -v GPU_INDEX_QUOTED '%q' "${GPU_INDEX}"
TMUX_INNER_COMMAND="(cd ${REPO_ROOT_QUOTED} && ${PREFLIGHT_COMMAND_STRING} && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU_INDEX_QUOTED} ${RUNNER_COMMAND_STRING}) > ${LOG_PATH_QUOTED} 2>&1"

if [[ "${DRY_RUN}" == true ]]; then
  printf 'tmux session: %s\n' "${SESSION_NAME}"
  printf 'log path: %s\n' "${LOG_PATH}"
  printf 'preflight summary path: %s\n' "${PREFLIGHT_SUMMARY_PATH}"
  printf 'comparative manifest path: %s\n' "${MANIFEST_PATH}"
  printf 'comparative execution report path: %s\n' "${EXECUTION_REPORT_PATH}"
  printf 'attach command: tmux attach -t %s\n' "${SESSION_NAME}"
  printf 'smoke profile: %s\n' "${SMOKE_PROFILE}"
  printf 'skip completed: %s\n' "${SKIP_COMPLETED}"
  printf 'preflight command: %s\n' "${PREFLIGHT_COMMAND_DISPLAY}"
  printf 'runner command: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=%s %s\n' "${GPU_INDEX}" "${RUNNER_COMMAND_DISPLAY}"
  printf 'tmux inner command: (cd %s && %s && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=%s %s) > %s 2>&1\n' \
    "${REPO_ROOT}" \
    "${PREFLIGHT_COMMAND_DISPLAY}" \
    "${GPU_INDEX}" \
    "${RUNNER_COMMAND_DISPLAY}" \
    "${LOG_PATH}"
  exit 0
fi

if [[ "${PREFLIGHT_ONLY}" == true ]]; then
  cd "${REPO_ROOT}"
  exec "${PREFLIGHT_COMMAND[@]}"
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
printf 'preflight summary path: %s\n' "${PREFLIGHT_SUMMARY_PATH}"
printf 'comparative manifest path: %s\n' "${MANIFEST_PATH}"
printf 'comparative execution report path: %s\n' "${EXECUTION_REPORT_PATH}"
printf 'attach command: tmux attach -t %s\n' "${SESSION_NAME}"
printf 'smoke profile: %s\n' "${SMOKE_PROFILE}"
printf 'skip completed: %s\n' "${SKIP_COMPLETED}"
