#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
PROTOCOL_CONFIG="configs/protocol/smd_window20_synthnormal_q99_ewma09.yaml"
RERUN_ROOT="${RERUN_ROOT:-outputs/benchmark_rerun/smd/direct_branch_synthnormal_q99_$(date +%Y%m%d_%H%M%S)}"

cd "$REPO_ROOT"
test -x "$PYTHON_BIN" || { echo "missing executable: $PYTHON_BIN" >&2; exit 1; }
test -f "$PROTOCOL_CONFIG" || { echo "missing: $PROTOCOL_CONFIG" >&2; exit 1; }

make_direct_config() {
  local base_config="$1"
  local config_path="$2"
  local experiment_name="$3"
  local configured_output_dir="$4"

  "$PYTHON_BIN" - "$base_config" "$config_path" "$experiment_name" \
    "$configured_output_dir" <<'PY'
import sys
from pathlib import Path

import yaml

base_path, output_path, experiment_name, configured_output_dir = sys.argv[1:]
config = yaml.safe_load(Path(base_path).read_text(encoding="utf-8"))
config["experiment_name"] = experiment_name
config["experiment_variant"] = "direct_branch_routing_v1"
config["output_dir"] = configured_output_dir
config["checkpoint_dir"] = f"{configured_output_dir}/checkpoints"
config.pop("two_stage", None)
model_overrides = dict(config.get("model_overrides", {}))
model_overrides.update(
    {
        "training_phase": "stage_b_fusion_finetuning",
        "fusion_mode": "direct_branch_routing",
    }
)
config["model_overrides"] = model_overrides
logging = dict(config.get("logging", {}))
logging.update(
    {
        "use_wandb": False,
        "wandb_mode": "disabled",
        "wandb_run_name": experiment_name,
        "wandb_job_type": "evaluate",
    }
)
config["logging"] = logging
Path(output_path).write_text(
    yaml.safe_dump(config, sort_keys=False),
    encoding="utf-8",
)
PY
}

run_one() {
  local base_config="$1"
  local checkpoint_path="$2"
  local run_name="$3"
  local output_dir="$4"
  local config_path="${RERUN_ROOT}/configs/${run_name}.yaml"

  make_direct_config \
    "$base_config" \
    "$config_path" \
    "$run_name" \
    "${RERUN_ROOT}/configured/${run_name}"

  "$PYTHON_BIN" -m scripts.benchmarks.run_thesis_offline_benchmark \
    --experiment-config "$config_path" \
    --protocol-config "$PROTOCOL_CONFIG" \
    --evaluation-only \
    --checkpoint-path "$checkpoint_path" \
    --output-dir "$output_dir"
}

validate_synthetic_threshold() {
  local output_dir="$1"
  "$PYTHON_BIN" - "$output_dir/thresholds/thresholds.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload["calibration_split"] == "clean_validation"
assert payload["thresholds"]["offline_point"]["source_split"] == (
    "synthetic_validation_normal"
)
PY
}

# Read-only preflight: 3 unsuffixed seed-6 checkpoints plus 18 O0/O1 checkpoints.
for entity in machine_1_6 machine_3_4 machine_3_9; do
  checkpoint="outputs/benchmark/smd/${entity}/seed6/thesis_direct_branch_routing/offline/stage_b/checkpoints/best.pt"
  test -f "$checkpoint" || { echo "missing: $checkpoint" >&2; exit 1; }
done
for variant in O0 O1; do
  for entity in machine_1_6 machine_3_4 machine_3_9; do
    for seed in 6 8 36; do
      checkpoint="outputs/benchmark/smd/${entity}/seed${seed}/thesis_direct_branch_routing_${variant}/offline/stage_b/checkpoints/best.pt"
      test -f "$checkpoint" || { echo "missing: $checkpoint" >&2; exit 1; }
    done
  done
done

if [[ -e "$RERUN_ROOT" ]]; then
  echo "rerun root already exists: $RERUN_ROOT" >&2
  exit 1
fi
mkdir -p "${RERUN_ROOT}/configs"

# One-combination gate before the remaining 20 evaluations.
run_name="direct_branch_routing_machine_1_6_seed6"
base_config="configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O0__machine_1_6__w20__seed6__main.yaml"
checkpoint="outputs/benchmark/smd/machine_1_6/seed6/thesis_direct_branch_routing/offline/stage_b/checkpoints/best.pt"
preflight_output_dir="${RERUN_ROOT}/direct/machine_1_6/seed6"
run_one "$base_config" "$checkpoint" "$run_name" "$preflight_output_dir"
validate_synthetic_threshold "$preflight_output_dir"
echo "Preflight passed; continuing with the remaining 20 direct-routing checkpoints."

for entity in machine_3_4 machine_3_9; do
  run_name="direct_branch_routing_${entity}_seed6"
  base_config="configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O0__${entity}__w20__seed6__main.yaml"
  checkpoint="outputs/benchmark/smd/${entity}/seed6/thesis_direct_branch_routing/offline/stage_b/checkpoints/best.pt"
  run_one \
    "$base_config" \
    "$checkpoint" \
    "$run_name" \
    "${RERUN_ROOT}/direct/${entity}/seed6"
done

for variant in O0 O1; do
  for entity in machine_1_6 machine_3_4 machine_3_9; do
    for seed in 6 8 36; do
      run_name="direct_branch_routing_${variant}_${entity}_seed${seed}"
      base_config="configs/experiment/offline_benchmark/thesis/smd__thesis__offline__${variant}__${entity}__w20__seed${seed}__main.yaml"
      checkpoint="outputs/benchmark/smd/${entity}/seed${seed}/thesis_direct_branch_routing_${variant}/offline/stage_b/checkpoints/best.pt"
      run_one \
        "$base_config" \
        "$checkpoint" \
        "$run_name" \
        "${RERUN_ROOT}/${variant}/${entity}/seed${seed}"
    done
  done
done

echo "Direct-routing rerun outputs: ${REPO_ROOT}/${RERUN_ROOT}"
