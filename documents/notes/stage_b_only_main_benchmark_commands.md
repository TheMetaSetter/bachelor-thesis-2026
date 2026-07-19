# Stage B init rewrite + Stage B train commands for old main benchmark

Remote tree checked:

- `ssh -p 20718 root@159.48.242.1`
- repo root: `/root/bachelor-thesis-2026`

The remote `outputs/benchmark/smd/thesis/.../two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml` files already contain:

- `initialization_checkpoint_path: .../two_stage/initializations/stage_b_init.pt`

What you want is:

1. Rebuild `stage_b_init.pt` from the existing Stage A checkpoint in the old main benchmark tree.
2. Train Stage B using that freshly rewritten init checkpoint.

Do **not** rerun Stage A.

## Command pattern

```bash
cd /root/bachelor-thesis-2026
THESIS_CONSOLE_QUIET=1 \
.venv/bin/python -m scripts.experiments.run_two_stage_offline_pretraining \
  --experiment-config <stage_a_config_path> \
  --stop-after-stage-b-init \
  2>&1 | tee <log_path>.txt
```

Then run Stage B with the Stage B config that points at the newly written init file:

```bash
cd /root/bachelor-thesis-2026
THESIS_CONSOLE_QUIET=1 \
.venv/bin/python -m scripts.train \
  --experiment-config <stage_b_config_path> \
  2>&1 | tee <log_path>.txt
```

## Main benchmark combo map

| Entity | Seed | O0 config | O1 config |
| --- | --- | --- | --- |
| machine_1_6 | 6 | `outputs/benchmark/smd/thesis/O0/machine_1_6/seed6/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml` | `outputs/benchmark/smd/thesis/O1/machine_1_6/seed6/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml` |
| machine_1_6 | 8 | `outputs/benchmark/smd/thesis/O0/machine_1_6/seed8/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml` | `outputs/benchmark/smd/thesis/O1/machine_1_6/seed8/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml` |
| machine_1_6 | 36 | `outputs/benchmark/smd/thesis/O0/machine_1_6/seed36/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml` | `outputs/benchmark/smd/thesis/O1/machine_1_6/seed36/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml` |
| machine_3_4 | 6 | `outputs/benchmark/smd/thesis/O0/machine_3_4/seed6/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml` | `outputs/benchmark/smd/thesis/O1/machine_3_4/seed6/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml` |
| machine_3_4 | 8 | `outputs/benchmark/smd/thesis/O0/machine_3_4/seed8/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml` | `outputs/benchmark/smd/thesis/O1/machine_3_4/seed8/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml` |
| machine_3_4 | 36 | `outputs/benchmark/smd/thesis/O0/machine_3_4/seed36/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml` | `outputs/benchmark/smd/thesis/O1/machine_3_4/seed36/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml` |
| machine_3_9 | 6 | `outputs/benchmark/smd/thesis/O0/machine_3_9/seed6/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml` | `outputs/benchmark/smd/thesis/O1/machine_3_9/seed6/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml` |
| machine_3_9 | 8 | `outputs/benchmark/smd/thesis/O0/machine_3_9/seed8/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml` | `outputs/benchmark/smd/thesis/O1/machine_3_9/seed8/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml` |
| machine_3_9 | 36 | `outputs/benchmark/smd/thesis/O0/machine_3_9/seed36/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml` | `outputs/benchmark/smd/thesis/O1/machine_3_9/seed36/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml` |

## Example

```bash
cd /root/bachelor-thesis-2026
THESIS_CONSOLE_QUIET=1 \
.venv/bin/python -m scripts.experiments.run_two_stage_offline_pretraining \
  --experiment-config outputs/benchmark/smd/thesis/O0/machine_1_6/seed6/two_stage/generated_configs/01_stage_a_multitask_pretraining.yaml \
  --stop-after-stage-b-init \
  2>&1 | tee outputs/tmux_logs/o0_machine_1_6_seed6_stage_b_init_rewrite.txt

cd /root/bachelor-thesis-2026
THESIS_CONSOLE_QUIET=1 \
.venv/bin/python -m scripts.train \
  --experiment-config outputs/benchmark/smd/thesis/O0/machine_1_6/seed6/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml \
  2>&1 | tee outputs/tmux_logs/o0_machine_1_6_seed6_stage_b_train.txt
```

## All-combo rerun

### 1) Rebuild `stage_b_init.pt` from the old Stage A checkpoint

```bash
cd /root/bachelor-thesis-2026
for variant in O0 O1; do
  for entity in machine_1_6 machine_3_4 machine_3_9; do
    for seed in 6 8 36; do
      stage_a_ckpt="outputs/benchmark/smd/thesis/${variant}/${entity}/seed${seed}/two_stage/stage_a_multitask_pretraining/checkpoints/best.pt"
      stage_b_cfg="outputs/benchmark/smd/thesis/${variant}/${entity}/seed${seed}/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml"
      stage_b_init="outputs/benchmark/smd/thesis/${variant}/${entity}/seed${seed}/two_stage/initializations/stage_b_init.pt"
      STAGE_A_CKPT="$stage_a_ckpt" \
      STAGE_B_CFG="$stage_b_cfg" \
      STAGE_B_INIT="$stage_b_init" \
      THESIS_DEBUG_VERIFICATION_INIT=1 \
      THESIS_CONSOLE_QUIET=1 \
      .venv/bin/python - <<'PY'
import os
from scripts.experiments.run_two_stage_offline_pretraining import _prepare_stage_b_initialization_checkpoint

manifest = {
    "training_stages": [
        {"best_checkpoint_path": os.environ["STAGE_A_CKPT"]},
        {
            "config_path": os.environ["STAGE_B_CFG"],
            "initialization_checkpoint_path": os.environ["STAGE_B_INIT"],
        },
    ]
}

_prepare_stage_b_initialization_checkpoint(manifest)
PY
    done
  done
done
```

### 2) Train Stage B with the rewritten init

```bash
cd /root/bachelor-thesis-2026
for variant in O0 O1; do
  for entity in machine_1_6 machine_3_4 machine_3_9; do
    for seed in 6 8 36; do
      stage_b_cfg="outputs/benchmark/smd/thesis/${variant}/${entity}/seed${seed}/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml"
      log_path="outputs/tmux_logs/${variant}_${entity}_seed${seed}_stage_b_train.txt"
      THESIS_CONSOLE_QUIET=1 \
      THESIS_DEBUG_VERIFICATION_INIT=1 \
      .venv/bin/python -m scripts.train \
        --experiment-config "$stage_b_cfg" \
        2>&1 | tee "$log_path"
    done
  done
done
```

### 3) Evaluate the new Stage B best checkpoint on the offline test set

```bash
cd /root/bachelor-thesis-2026
for variant in O0 O1; do
  for entity in machine_1_6 machine_3_4 machine_3_9; do
    for seed in 6 8 36; do
      stage_b_cfg="outputs/benchmark/smd/thesis/${variant}/${entity}/seed${seed}/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml"
      stage_b_best="outputs/benchmark/smd/thesis/${variant}/${entity}/seed${seed}/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt"
      log_path="outputs/tmux_logs/${variant}_${entity}_seed${seed}_stage_b_offline_test.txt"
      THESIS_CONSOLE_QUIET=1 \
      .venv/bin/python -m scripts.run_thesis_offline_benchmark \
        --experiment-config "$stage_b_cfg" \
        --protocol-config configs/protocol/smd_window20_cleanval_q99_ewma09.yaml \
        --evaluation-only \
        --checkpoint-path "$stage_b_best" \
        2>&1 | tee "$log_path"
    done
  done
done
```
