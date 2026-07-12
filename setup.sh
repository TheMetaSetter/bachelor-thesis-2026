#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

log() {
    printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

require() {
    command -v "$1" >/dev/null 2>&1 || {
        printf 'Missing required command: %s\n' "$1" >&2
        exit 1
    }
}

require curl
require sudo

export DEBIAN_FRONTEND=noninteractive

log "Installing system packages"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    tzdata \
    unzip

log "Setting timezone"
if command -v timedatectl >/dev/null &&
    timedatectl status >/dev/null 2>&1; then
    sudo timedatectl set-timezone Asia/Ho_Chi_Minh
else
    sudo ln -snf /usr/share/zoneinfo/Asia/Ho_Chi_Minh /etc/localtime
    printf '%s\n' Asia/Ho_Chi_Minh |
        sudo tee /etc/timezone >/dev/null
fi

if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

require uv

log "Creating Python environment"
uv venv --python 3.12 --clear

log "Installing Python dependencies"
install_args=(-r requirements.txt)

if [[ -n "${TORCH_EXTRA_INDEX_URL:-}" ]]; then
    install_args=(
        --extra-index-url "$TORCH_EXTRA_INDEX_URL"
        "${install_args[@]}"
    )
fi

uv pip install "${install_args[@]}"
uv pip install gdown

if [[ ! -d data ]]; then
    log "Downloading dataset"
    uv run gdown 1EJh7doCPL4lbGWTQ2YSul5QDfT-eIbwH -O data.zip
    unzip -q data.zip
    rm data.zip
fi

log "Verifying installation"
uv run python --version
uv run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
PY

log "Setup complete"