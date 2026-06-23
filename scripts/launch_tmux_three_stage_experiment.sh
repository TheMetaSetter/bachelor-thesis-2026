#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_EXPERIMENT_CONFIG="configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml"
DEFAULT_SESSION_NAME="three-stage-machine-3-4-seed11"
DEFAULT_PYTHON_BIN=".venv/bin/python"
DEFAULT_GPU_INDEX="0"
DEFAULT_REQUIRED_GPU_NAME_SUBSTRING="RTX 3090"

EXPERIMENT_CONFIG="${DEFAULT_EXPERIMENT_CONFIG}"
SESSION_NAME="${DEFAULT_SESSION_NAME}"
PYTHON_BIN="${DEFAULT_PYTHON_BIN}"
GPU_INDEX="${DEFAULT_GPU_INDEX}"
REQUIRED_GPU_NAME_SUBSTRING="${DEFAULT_REQUIRED_GPU_NAME_SUBSTRING}"
DRY_RUN=false
PREFLIGHT_ONLY=false
REPLACE_SESSION=false

LOG_DIR="${REPO_ROOT}/outputs/tmux_logs"
LOG_PATH=""

print_usage() {
  cat <<'EOF'
Usage: scripts/launch_tmux_three_stage_experiment.sh [options]

Options:
  --experiment-config PATH
  --session-name NAME
  --python-bin PATH
  --gpu-index INDEX
  --required-gpu-name-substring TEXT
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

resolve_experiment_config_path() {
  local config_path="$1"
  if [[ "${config_path}" = /* ]]; then
    printf '%s\n' "${config_path}"
  else
    printf '%s/%s\n' "${REPO_ROOT}" "${config_path}"
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
  printf '--dry-run and --preflight-only cannot be used together.\n' >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${SESSION_NAME}.log"
RESOLVED_EXPERIMENT_CONFIG_PATH="$(resolve_experiment_config_path "${EXPERIMENT_CONFIG}")"
RESOLVED_OUTPUT_DIR="$(resolve_output_dir_path "${RESOLVED_EXPERIMENT_CONFIG_PATH}")"
PREFLIGHT_SUMMARY_PATH="${RESOLVED_OUTPUT_DIR}/three_stage/server_preflight_summary.json"
RUN_VERIFICATION_SUMMARY_PATH="${RESOLVED_OUTPUT_DIR}/three_stage/three_stage_run_verification.json"

PREFLIGHT_COMMAND=(
  "${PYTHON_BIN}"
  "scripts/preflight_three_stage_server.py"
  "--experiment-config"
  "${EXPERIMENT_CONFIG}"
  "--gpu-index"
  "${GPU_INDEX}"
  "--required-gpu-name-substring"
  "${REQUIRED_GPU_NAME_SUBSTRING}"
  "--print-json"
)

if [[ "${PREFLIGHT_ONLY}" != true ]]; then
  PREFLIGHT_COMMAND+=("--require-launch-ready")
fi

TRAIN_COMMAND=(
  "${PYTHON_BIN}"
  "scripts/run_three_stage_offline_pretraining.py"
  "--experiment-config"
  "${EXPERIMENT_CONFIG}"
)

VERIFY_COMMAND=(
  "${PYTHON_BIN}"
  "scripts/verify_three_stage_run.py"
  "--output-dir"
  "${RESOLVED_OUTPUT_DIR}"
  "--require-success"
)

PREFLIGHT_COMMAND_STRING="$(join_quoted_args "${PREFLIGHT_COMMAND[@]}")"
TRAIN_COMMAND_STRING="$(join_quoted_args "${TRAIN_COMMAND[@]}")"
VERIFY_COMMAND_STRING="$(join_quoted_args "${VERIFY_COMMAND[@]}")"
PREFLIGHT_COMMAND_DISPLAY="$(join_display_args "${PREFLIGHT_COMMAND[@]}")"
TRAIN_COMMAND_DISPLAY="$(join_display_args "${TRAIN_COMMAND[@]}")"
VERIFY_COMMAND_DISPLAY="$(join_display_args "${VERIFY_COMMAND[@]}")"

printf -v REPO_ROOT_QUOTED '%q' "${REPO_ROOT}"
printf -v LOG_PATH_QUOTED '%q' "${LOG_PATH}"
printf -v GPU_INDEX_QUOTED '%q' "${GPU_INDEX}"

TMUX_INNER_COMMAND="(cd ${REPO_ROOT_QUOTED} && ${PREFLIGHT_COMMAND_STRING} && CUDA_VISIBLE_DEVICES=${GPU_INDEX_QUOTED} ${TRAIN_COMMAND_STRING} && ${VERIFY_COMMAND_STRING}) > ${LOG_PATH_QUOTED} 2>&1"

if [[ "${DRY_RUN}" == true ]]; then
  printf 'tmux session: %s\n' "${SESSION_NAME}"
  printf 'log path: %s\n' "${LOG_PATH}"
  printf 'preflight summary path: %s\n' "${PREFLIGHT_SUMMARY_PATH}"
  printf 'run verification summary path: %s\n' "${RUN_VERIFICATION_SUMMARY_PATH}"
  printf 'attach command: tmux attach -t %s\n' "${SESSION_NAME}"
  printf 'preflight command: %s\n' "${PREFLIGHT_COMMAND_DISPLAY}"
  printf 'training command: CUDA_VISIBLE_DEVICES=%s %s\n' "${GPU_INDEX}" "${TRAIN_COMMAND_DISPLAY}"
  printf 'verification command: %s\n' "${VERIFY_COMMAND_DISPLAY}"
  printf 'tmux inner command: (cd %s && %s && CUDA_VISIBLE_DEVICES=%s %s && %s) > %s 2>&1\n' \
    "${REPO_ROOT}" \
    "${PREFLIGHT_COMMAND_DISPLAY}" \
    "${GPU_INDEX}" \
    "${TRAIN_COMMAND_DISPLAY}" \
    "${VERIFY_COMMAND_DISPLAY}" \
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
printf 'run verification summary path: %s\n' "${RUN_VERIFICATION_SUMMARY_PATH}"
printf 'attach command: tmux attach -t %s\n' "${SESSION_NAME}"
