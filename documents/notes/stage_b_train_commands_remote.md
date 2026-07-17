# Stage B train commands verified on remote

Remote host checked:

- `ssh -p 20002 root@159.48.242.13`

The following `generated_configs/02_stage_b_fusion_finetuning.yaml` files exist on the remote benchmark tree.

## Command 1

```bash
THESIS_CONSOLE_QUIET=1 \
.venv/bin/python -m scripts.train \
  --experiment-config outputs/benchmark/smd/thesis/O0/machine_1_6/seed6/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml \
  2>&1 | tee outputs/tmux_logs/o0_machine_1_6_seed6_stage_b_train.txt
```

## Command 2

```bash
THESIS_CONSOLE_QUIET=1 \
.venv/bin/python -m scripts.train \
  --experiment-config outputs/benchmark/smd/thesis/O1/machine_3_9/seed8/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml \
  2>&1 | tee outputs/tmux_logs/o1_machine_3_9_seed8_stage_b_train.txt
```

## Pattern to reuse for other combos

Replace only the combo segment in the config path and the log filename.

