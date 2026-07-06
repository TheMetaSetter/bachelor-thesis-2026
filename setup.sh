#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

log() {
    printf '\n[%s] %s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$*"
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

require_command curl
require_command sudo

export DEBIAN_FRONTEND=noninteractive

log "Updating apt package metadata"
sudo apt-get update

log "Installing system packages"
sudo apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    tzdata \
    unar \
    zip

log "Setting timezone to Asia/Ho_Chi_Minh"
if command -v timedatectl >/dev/null 2>&1 && timedatectl status >/dev/null 2>&1; then
    sudo timedatectl set-timezone Asia/Ho_Chi_Minh
else
    sudo ln -snf /usr/share/zoneinfo/Asia/Ho_Chi_Minh /etc/localtime
    echo Asia/Ho_Chi_Minh | sudo tee /etc/timezone >/dev/null
fi

if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [[ -d "$HOME/.local/bin" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

source "$HOME/.local/bin/env"
require_command uv

log "Creating Python 3.12 virtual environment"
uv venv --python 3.12 --clear

# The activation is local to this script process. It keeps the later checks simple.
# shellcheck disable=SC1091
source .venv/bin/activate

log "Installing Python dependencies from requirements.txt"
if [[ -n "${TORCH_EXTRA_INDEX_URL:-}" ]]; then
    log "Using TORCH_EXTRA_INDEX_URL=${TORCH_EXTRA_INDEX_URL}"
    uv pip install --extra-index-url "$TORCH_EXTRA_INDEX_URL" -r requirements.txt
else
    uv pip install -r requirements.txt
fi

uv pip install gdown

log "Verifying installation"
python --version
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
PY

log "Setup complete"
