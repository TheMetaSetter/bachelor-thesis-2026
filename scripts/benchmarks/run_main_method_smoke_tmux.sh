#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLOUD_LAUNCHER="$REPO_ROOT/scripts/benchmarks/run_remaining_smd_cloud_tmux.sh"

# This wrapper keeps only THESIS O0/O1 offline runs and A0/A1/A2 online runs.
# Every selected run uses one of the four V100 GPUs, matching the wet layout.
exec bash "$CLOUD_LAUNCHER" \
    --mode smoke \
    --gpu-count 4 \
    --gpu-only \
    --main-method-only \
    --stage-a-epochs 4 \
    --stage-b-epochs 2 \
    --max-online-steps 16 \
    --skip-completed \
    --session-prefix main-smd-smoke \
    --output-root outputs/benchmark_smoke/smd_main_method \
    --entity-id machine-3-1 \
    --entity-id machine-3-5 \
    --entity-id machine-3-2 \
    --entity-id machine-3-11 \
    --entity-id machine-3-10 \
    --entity-id machine-1-3 \
    --entity-id machine-1-1 \
    --entity-id machine-2-8 \
    "$@"
