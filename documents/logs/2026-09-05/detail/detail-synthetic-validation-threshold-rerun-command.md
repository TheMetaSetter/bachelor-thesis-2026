---
date: 2026-09-05T18:00:00+07:00
topic: "Synthetic-validation threshold rerun command"
status: ready
related_detail: documents/logs/2026-09-05/detail/detail-synthetic-validation-threshold-rerun.md
---

# Rerun command

The command below evaluates the 18 Stage-B `best.pt` checkpoints for `O0` and
`O1`, three entities, and seeds `6`, `8`, and `36`. It uses the opt-in protocol
`smd_window20_synthnormal_q99_ewma09.yaml`. Every checkpoint writes to its own
directory below a new rerun root.

The SSH endpoint and remote repository path must be read from the current
`cloud-gpu.txt` or SSH instruction file immediately before use. Do not store
credentials in this document.

## Read-only inventory preflight

Run this on `cloud-gpu` after changing to the repository root. It checks the
exact 18 requested checkpoint paths without changing remote state.

```bash
for variant in O0 O1; do
  for entity in machine_1_6 machine_3_4 machine_3_9; do
    for seed in 6 8 36; do
      checkpoint="outputs/benchmark/smd/thesis/${variant}/${entity}/seed${seed}/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt"
      test -f "$checkpoint" || { echo "missing: $checkpoint"; exit 1; }
    done
  done
done
echo "18 Stage-B best checkpoints are present"
```

## One-combination preflight

Run one evaluation first. Inspect `thresholds/thresholds.json`,
`metrics/offline_metrics.json`, and `benchmark/thesis_offline_benchmark_report.json`
before starting the remaining 17 combinations.

```bash
variant=O0
entity=machine_1_6
seed=6
rerun_root="outputs/benchmark_rerun/smd/thesis_synthnormal_q99_$(date +%Y%m%d_%H%M%S)"
experiment_config="configs/experiment/offline_benchmark/thesis/smd__thesis__offline__${variant}__${entity}__w20__seed${seed}__main.yaml"
checkpoint_path="outputs/benchmark/smd/thesis/${variant}/${entity}/seed${seed}/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt"
output_dir="${rerun_root}/${variant}/${entity}/seed${seed}"

tmux new-session -d -s thesis-synth-q99-preflight \
  ".venv/bin/python -m scripts.run_thesis_offline_benchmark \
    --experiment-config \"${experiment_config}\" \
    --protocol-config configs/protocol/smd_window20_synthnormal_q99_ewma09.yaml \
    --evaluation-only \
    --checkpoint-path \"${checkpoint_path}\" \
    --output-dir \"${output_dir}\""
```

The preflight is valid only when the threshold artifact reports
`thresholds.offline_point.source_split: synthetic_validation_normal`, keeps
top-level `calibration_split: clean_validation`, and writes all files below
`${output_dir}`.

## Remaining 17 evaluations

After the preflight passes, run the same command for the remaining combinations.
The loop below skips `O0/machine_1_6/seed6`, so it starts exactly 17 jobs.

```bash
for variant in O0 O1; do
  for entity in machine_1_6 machine_3_4 machine_3_9; do
    for seed in 6 8 36; do
      if [ "$variant" = O0 ] && [ "$entity" = machine_1_6 ] && [ "$seed" = 6 ]; then
        continue
      fi
      experiment_config="configs/experiment/offline_benchmark/thesis/smd__thesis__offline__${variant}__${entity}__w20__seed${seed}__main.yaml"
      checkpoint_path="outputs/benchmark/smd/thesis/${variant}/${entity}/seed${seed}/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt"
      output_dir="${rerun_root}/${variant}/${entity}/seed${seed}"
      .venv/bin/python -m scripts.run_thesis_offline_benchmark \
        --experiment-config "$experiment_config" \
        --protocol-config configs/protocol/smd_window20_synthnormal_q99_ewma09.yaml \
        --evaluation-only \
        --checkpoint-path "$checkpoint_path" \
        --output-dir "$output_dir"
    done
  done
done
```

Do not run the matrix against a checkout that does not contain the new protocol
file and `--output-dir` implementation. Do not reuse an existing benchmark root.
