#!/usr/bin/env bash

set -e -u -o pipefail

REPO_ROOT="$(cd "$(dirname "{BASH_SOURCE[0]}")/.." & pwd)"

PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

EXPERIMENT_CONFIG="configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O1_recon075_cls025__machine_3_4__w20__seed36__smoke.yaml"

PROTOCOL_CONFIG="configs/protocol/smd_window20_cleanval_q99_ewma09.yaml"

