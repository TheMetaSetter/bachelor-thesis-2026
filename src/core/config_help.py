from __future__ import annotations

"""Human-friendly CLI help for experiment configuration."""

from textwrap import dedent


def build_config_help_text(command_name: str) -> str:
    return dedent(
        f"""
        {command_name} config quick help

        1) Required structure
          - experiment_name, seed, device, output_dir, checkpoint_dir
          - data_config_path, model_config_path, task_config_path
          - optimizer, epochs

        2) Common fields to edit
          - data.window_size, data.stride, data.batch_size, data.entity_ids
          - optimizer.optimizer_name (adam|adamw), optimizer.learning_rate
          - optimizer.scheduler.scheduler_name (cosine|reduce_on_plateau)
          - checkpoint_monitor_metric (must match scheduler monitor for reduce_on_plateau)

        3) Wandb consistency rules (strict)
          - If logging.use_wandb is false, logging.wandb_mode must be 'disabled'
          - If logging.use_wandb is true, logging.wandb_mode must be 'online' or 'offline'

        4) Synthetic multitask hints
          - classification_label_mode='redlamp_multiclass' requires num_classes=12
          - anomaly_probability must be in [0, 1]

        5) Useful commands
          - Print this help:
              python scripts/{command_name}.py --print-config-help
          - Train:
              python scripts/train.py --experiment-config configs/experiment/baseline/smd__redlamp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml
          - Evaluate:
              python scripts/evaluate.py --experiment-config configs/experiment/baseline/smd__redlamp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml --checkpoint-path outputs/.../best.pt
        """
    ).strip()
