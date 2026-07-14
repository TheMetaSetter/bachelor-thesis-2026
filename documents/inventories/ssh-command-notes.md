# SSH Command Notes

Short, reusable read-only SSH commands that worked on the remote GPU server.

## Commands

```sh
ssh -p 20002 root@159.48.242.13
```

Use this for an interactive login when you want to inspect `tmux` or run commands manually.

```sh
cd /root/bachelor-thesis-2026 && find outputs/benchmark_smoke/smd/thesis/O0/machine_1_6/seed8/two_stage/stage_b_fusion_finetuning -maxdepth 2 \( -type f -o -type l \) | sort
```

Use this to inspect the exact files under one Stage B smoke run.

```sh
cd /root/bachelor-thesis-2026 && find outputs/benchmark_smoke/smd/thesis_multitask/machine_1_6/seed8/two_stage -maxdepth 4 \( -type f -o -type d \) | sort | sed -n '1,200p'
```

Use this to inspect the underlying run tree behind the `thesis/O0/...` symlink.
