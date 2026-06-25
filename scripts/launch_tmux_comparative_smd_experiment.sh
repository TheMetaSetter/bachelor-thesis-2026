#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_SESSION_NAME="comparative-smd-top3-entities"
DEFAULT_PYTHON_BIN=".venv/bin/python"
DEFAULT_REPORT_DIR="outputs/comparative_smd_reports/top3-entities-three-seeds"

SESSION_NAME="${DEFAULT_SESSION_NAME}"
PYTHON_BIN="${DEFAULT_PYTHON_BIN}"
REPORT_DIR="${DEFAULT_REPORT_DIR}"
DRY_RUN=false
PREFLIGHT_ONLY=false
REPLACE_SESSION=false

LOG_DIR="${REPO_ROOT}/outputs/tmux_logs"
LOG_PATH=""

SMOKE_CONFIG_PATHS=(
  "configs/experiment/comparative/baseline/smd__redlamp_mlp_baseline__comparative-single-stage-machine_1_6__w20__seed6__smoke.yaml"
  "configs/experiment/comparative/thesis/smd__thesis_multitask__comparative-three-stage-machine_1_6__w20__seed6__smoke.yaml"
)

MAIN_CONFIG_PATHS=(
  "configs/experiment/comparative/baseline/smd__redlamp_mlp_baseline__comparative-single-stage-machine_1_6__w20__seed6__main.yaml"
  "configs/experiment/comparative/baseline/smd__redlamp_mlp_baseline__comparative-single-stage-machine_1_6__w20__seed36__main.yaml"
  "configs/experiment/comparative/baseline/smd__redlamp_mlp_baseline__comparative-single-stage-machine_1_6__w20__seed68__main.yaml"
  "configs/experiment/comparative/baseline/smd__redlamp_mlp_baseline__comparative-single-stage-machine_3_1__w20__seed6__main.yaml"
  "configs/experiment/comparative/baseline/smd__redlamp_mlp_baseline__comparative-single-stage-machine_3_1__w20__seed36__main.yaml"
  "configs/experiment/comparative/baseline/smd__redlamp_mlp_baseline__comparative-single-stage-machine_3_1__w20__seed68__main.yaml"
  "configs/experiment/comparative/baseline/smd__redlamp_mlp_baseline__comparative-single-stage-machine_3_9__w20__seed6__main.yaml"
  "configs/experiment/comparative/baseline/smd__redlamp_mlp_baseline__comparative-single-stage-machine_3_9__w20__seed36__main.yaml"
  "configs/experiment/comparative/baseline/smd__redlamp_mlp_baseline__comparative-single-stage-machine_3_9__w20__seed68__main.yaml"
  "configs/experiment/comparative/thesis/smd__thesis_multitask__comparative-three-stage-machine_1_6__w20__seed6__main.yaml"
  "configs/experiment/comparative/thesis/smd__thesis_multitask__comparative-three-stage-machine_1_6__w20__seed36__main.yaml"
  "configs/experiment/comparative/thesis/smd__thesis_multitask__comparative-three-stage-machine_1_6__w20__seed68__main.yaml"
  "configs/experiment/comparative/thesis/smd__thesis_multitask__comparative-three-stage-machine_3_1__w20__seed6__main.yaml"
  "configs/experiment/comparative/thesis/smd__thesis_multitask__comparative-three-stage-machine_3_1__w20__seed36__main.yaml"
  "configs/experiment/comparative/thesis/smd__thesis_multitask__comparative-three-stage-machine_3_1__w20__seed68__main.yaml"
  "configs/experiment/comparative/thesis/smd__thesis_multitask__comparative-three-stage-machine_3_9__w20__seed6__main.yaml"
  "configs/experiment/comparative/thesis/smd__thesis_multitask__comparative-three-stage-machine_3_9__w20__seed36__main.yaml"
  "configs/experiment/comparative/thesis/smd__thesis_multitask__comparative-three-stage-machine_3_9__w20__seed68__main.yaml"
)

print_usage() {
  cat <<'EOF'
Usage: scripts/launch_tmux_comparative_smd_experiment.sh [options]

Options:
  --session-name NAME
  --python-bin PATH
  --report-dir PATH
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

mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${SESSION_NAME}.log"
RESOLVED_REPORT_DIR="$(resolve_report_dir_path "${REPORT_DIR}")"
MANIFEST_PATH="${RESOLVED_REPORT_DIR}/comparative_manifest.json"
EXECUTION_REPORT_PATH="${RESOLVED_REPORT_DIR}/comparative_execution_report.json"

RUNNER_COMMAND=(
  "${PYTHON_BIN}"
  "scripts/run_comparative_smd_experiments.py"
  "--report-dir"
  "${REPORT_DIR}"
  "--smoke-config-paths"
)
RUNNER_COMMAND+=("${SMOKE_CONFIG_PATHS[@]}")
RUNNER_COMMAND+=("--config-paths")
RUNNER_COMMAND+=("${MAIN_CONFIG_PATHS[@]}")

PREFLIGHT_COMMAND=("${RUNNER_COMMAND[@]}" "--preflight-only")
PREFLIGHT_COMMAND_STRING="$(join_quoted_args "${PREFLIGHT_COMMAND[@]}")"
RUNNER_COMMAND_STRING="$(join_quoted_args "${RUNNER_COMMAND[@]}")"
PREFLIGHT_COMMAND_DISPLAY="$(join_display_args "${PREFLIGHT_COMMAND[@]}")"
RUNNER_COMMAND_DISPLAY="$(join_display_args "${RUNNER_COMMAND[@]}")"

printf -v REPO_ROOT_QUOTED '%q' "${REPO_ROOT}"
printf -v LOG_PATH_QUOTED '%q' "${LOG_PATH}"
TMUX_INNER_COMMAND="(cd ${REPO_ROOT_QUOTED} && ${PREFLIGHT_COMMAND_STRING} && ${RUNNER_COMMAND_STRING}) > ${LOG_PATH_QUOTED} 2>&1"

if [[ "${DRY_RUN}" == true ]]; then
  printf 'tmux session: %s\n' "${SESSION_NAME}"
  printf 'log path: %s\n' "${LOG_PATH}"
  printf 'comparative manifest path: %s\n' "${MANIFEST_PATH}"
  printf 'comparative execution report path: %s\n' "${EXECUTION_REPORT_PATH}"
  printf 'attach command: tmux attach -t %s\n' "${SESSION_NAME}"
  printf 'preflight command: %s\n' "${PREFLIGHT_COMMAND_DISPLAY}"
  printf 'runner command: %s\n' "${RUNNER_COMMAND_DISPLAY}"
  printf 'tmux inner command: (cd %s && %s && %s) > %s 2>&1\n' \
    "${REPO_ROOT}" \
    "${PREFLIGHT_COMMAND_DISPLAY}" \
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
printf 'comparative manifest path: %s\n' "${MANIFEST_PATH}"
printf 'comparative execution report path: %s\n' "${EXECUTION_REPORT_PATH}"
printf 'attach command: tmux attach -t %s\n' "${SESSION_NAME}"
