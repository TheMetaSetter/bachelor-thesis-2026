from __future__ import annotations

"""Entrypoint for offline training experiments.

A fresher can read this script as the shortest explanation of the runtime graph:
load config, register components, build data, build model, create the engine
objects, then hand everything to the trainer.
"""

import argparse
import math
from typing import Any

import torch

# Add src to path for imports
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.core.console import console_print
from src.core.config import load_experiment_config
from src.core.config_help import build_config_help_text
from src.core.registry import (
    build_dataset,
    build_model,
    register_dataset,
    register_model,
)
from src.core.seed import seed_everything
from src.data.loaders import (
    build_anomaly_archive_dataset_bundle,
    build_smd_dataset_bundle,
)
from src.engine.checkpoint import CheckpointManager
from src.engine.logger import ExperimentLogger
from src.engine.trainer import Trainer
from src.models.redlamp_baseline import RedLampBaseline
from src.models.reconstruction_mlp_ae import ReconstructionMLPAutoencoder
from src.models.thesis_multitask import ThesisMultitaskModel


def register_runtime_components() -> None:
    # Registration keeps script wiring explicit while still letting experiments
    # build datasets and models from names instead of hard-coded constructors.
    register_dataset("smd", build_smd_dataset_bundle)
    register_dataset("anomaly_archive", build_anomaly_archive_dataset_bundle)
    register_model("reconstruction_mlp_ae", ReconstructionMLPAutoencoder)
    register_model("thesis_multitask", ThesisMultitaskModel)
    register_model("redlamp_mlp_baseline", RedLampBaseline)
    register_model("redlamp_baseline", RedLampBaseline)
    console_print("REGISTRY", "Registered offline training runtime components")


def build_model_from_experiment_config(experiment_config: dict) -> torch.nn.Module:
    # Model and task settings are merged here because each active model file
    # owns both architecture and stage-step behavior.
    model_name = experiment_config["model"]["model_name"]
    model_kwargs = {
        key: value
        for key, value in experiment_config["model"].items()
        if key != "model_name"
    }
    model_kwargs.update(
        {
            key: value
            for key, value in experiment_config["task"].items()
            if key != "task_name"
        }
    )
    if (
        model_name in {"redlamp_baseline", "redlamp_mlp_baseline", "thesis_multitask"}
        and "window_size" not in model_kwargs
    ):
        model_kwargs["window_size"] = experiment_config["data"]["window_size"]
    console_print(
        "MODEL",
        "Building model from resolved experiment config",
        model_name=model_name,
        model_kwargs_keys=sorted(model_kwargs.keys()),
    )
    return build_model(model_name, **model_kwargs)


def build_optimizer_from_experiment_config(
    model: torch.nn.Module,
    experiment_config: dict[str, object],
) -> torch.optim.Optimizer:
    optimizer_config = experiment_config["optimizer"]
    optimizer_name = str(optimizer_config.get("optimizer_name", "adam"))
    optimizer_kwargs = {
        "lr": float(optimizer_config["learning_rate"]),
        "weight_decay": float(optimizer_config["weight_decay"]),
    }
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")


def maybe_initialize_model_from_checkpoint(
    model: torch.nn.Module,
    experiment_config: dict[str, object],
) -> None:
    initialization_checkpoint_path = experiment_config.get(
        "initialization_checkpoint_path"
    )
    if initialization_checkpoint_path is None:
        return
    checkpoint_path = Path(str(initialization_checkpoint_path))
    checkpoint_manager = CheckpointManager(checkpoint_path.parent)
    checkpoint_manager.load_checkpoint(checkpoint_path, model)
    console_print(
        "TRAIN",
        "Initialized model weights from checkpoint before training",
        initialization_checkpoint_path=checkpoint_path,
    )


def _compute_cosine_learning_rate_without_warmup(
    *,
    base_learning_rate: float,
    current_progress: float,
    total_epochs: int,
    cosine_end_lr: float,
    cosine_offset_epochs: float,
) -> float:
    cosine_progress = max(current_progress - cosine_offset_epochs, 0.0)
    cosine_duration = max(float(total_epochs) - cosine_offset_epochs, 1.0e-12)
    clamped_progress = min(cosine_progress / cosine_duration, 1.0)
    cosine_weight = 0.5 * (1.0 + math.cos(math.pi * clamped_progress))
    return cosine_end_lr + (base_learning_rate - cosine_end_lr) * cosine_weight


def compute_candi_style_cosine_learning_rate(
    *,
    base_learning_rate: float,
    current_progress: float,
    total_epochs: int,
    warmup_epochs: int,
    warmup_start_lr: float,
    cosine_end_lr: float,
    cosine_after_warmup: bool,
) -> float:
    cosine_offset_epochs = float(warmup_epochs) if cosine_after_warmup else 0.0
    if warmup_epochs > 0 and current_progress < warmup_epochs:
        cosine_warmup_end_lr = _compute_cosine_learning_rate_without_warmup(
            base_learning_rate=base_learning_rate,
            current_progress=float(warmup_epochs),
            total_epochs=total_epochs,
            cosine_end_lr=cosine_end_lr,
            cosine_offset_epochs=cosine_offset_epochs,
        )
        warmup_alpha = (cosine_warmup_end_lr - warmup_start_lr) / float(warmup_epochs)
        return current_progress * warmup_alpha + warmup_start_lr
    return _compute_cosine_learning_rate_without_warmup(
        base_learning_rate=base_learning_rate,
        current_progress=current_progress,
        total_epochs=total_epochs,
        cosine_end_lr=cosine_end_lr,
        cosine_offset_epochs=cosine_offset_epochs,
    )


def build_scheduler_from_experiment_config(
    optimizer: torch.optim.Optimizer,
    experiment_config: dict[str, object],
) -> tuple[Any | None, str | None]:
    optimizer_config = experiment_config["optimizer"]
    scheduler_config = optimizer_config.get("scheduler")
    if scheduler_config is None:
        console_print("TRAIN", "No learning rate scheduler configured")
        return None, None

    scheduler_name = scheduler_config["scheduler_name"]
    if scheduler_name == "cosine":
        console_print(
            "TRAIN",
            "Using arithmetic cosine learning rate policy",
            scheduler_name=scheduler_name,
        )
        return None, None
    if scheduler_name != "reduce_on_plateau":
        raise ValueError(f"Unsupported scheduler_name: {scheduler_name}")
    monitor_metric = str(scheduler_config["monitor_metric"])
    scheduler_mode_by_metric = {
        "val_loss": "min",
        "val_synth_loss": "min",
        "val_synth_roc_auc": "max",
        "val_synth_pr_auc": "max",
        "val_synth_vus_pr": "max",
        "val_realistic_loss": "min",
        "val_realistic_roc_auc": "max",
        "val_realistic_pr_auc": "max",
        "val_realistic_vus_pr": "max",
    }
    if monitor_metric not in scheduler_mode_by_metric:
        raise ValueError(f"Unsupported scheduler monitor metric: {monitor_metric}")
    scheduler_mode = scheduler_mode_by_metric[monitor_metric]

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode=scheduler_mode,
        factor=float(scheduler_config["factor"]),
        patience=int(scheduler_config["patience"]),
        threshold=float(scheduler_config["threshold"]),
        threshold_mode=str(scheduler_config["threshold_mode"]),
        cooldown=int(scheduler_config["cooldown"]),
        min_lr=float(scheduler_config["min_lr"]),
    )
    console_print(
        "TRAIN",
        "Initialized learning rate scheduler",
        scheduler_name=scheduler_name,
        monitor_metric=monitor_metric,
        scheduler_mode=scheduler_mode,
        factor=scheduler_config["factor"],
        patience=scheduler_config["patience"],
        threshold=scheduler_config["threshold"],
        threshold_mode=scheduler_config["threshold_mode"],
        cooldown=scheduler_config["cooldown"],
        min_lr=scheduler_config["min_lr"],
    )
    return scheduler, monitor_metric


def run_training_experiment(experiment_config: dict[str, object]) -> dict[str, object]:
    """Execute a complete training experiment from configuration.

    This helper is shared by the CLI and by tests or orchestration scripts.
    It orchestrates the entire training pipeline: seeding, component registration,
    data loading, model construction, and training execution.

    Args:
        experiment_config: Dictionary containing all experiment hyperparameters,
            data settings, model architecture, optimizer config, and paths.

    Returns:
        Training results dictionary from trainer.train() containing metrics,
            final losses, and other tracked experiment outcomes.
    """
    # Ensure reproducibility by seeding all RNG sources (Python, NumPy, PyTorch).
    # This must happen before any non-deterministic operations.
    seed_everything(int(experiment_config["seed"]))
    console_print(
        "TRAIN",
        "Starting training experiment",
        experiment_name=experiment_config["experiment_name"],
        device=experiment_config["device"],
        seed=experiment_config["seed"],
        output_dir=experiment_config["output_dir"],
        checkpoint_dir=experiment_config["checkpoint_dir"],
        epochs=experiment_config["epochs"],
    )

    # Register dataset and model builders into the global registry. This decouples
    # experiment configuration (which uses string names) from actual constructors,
    # allowing experiments to be defined in YAML without hardcoded imports.
    register_runtime_components()

    # Load and preprocess the dataset specified in config. Returns a bundle containing
    # train/val data loaders and a fitted scaler (for input normalization).
    data_bundle = build_dataset(
        experiment_config["data"]["dataset_name"], experiment_config["data"]
    )
    console_print(
        "DATA",
        "Built dataset bundle for training",
        dataset_name=experiment_config["data"]["dataset_name"],
        train_windows=len(data_bundle["datasets"]["train"]),
        val_windows=len(data_bundle["datasets"]["val"]),
        test_windows=len(data_bundle["datasets"]["test"]),
    )

    # Construct the model architecture and task logic from config. This combines
    # model-specific parameters (layer sizes, etc.) with task-specific logic
    # (loss weights, multitask heads, etc.) into a single PyTorch module.
    model = build_model_from_experiment_config(experiment_config)
    maybe_initialize_model_from_checkpoint(model, experiment_config)

    optimizer = build_optimizer_from_experiment_config(model, experiment_config)
    optimizer_name = str(experiment_config["optimizer"].get("optimizer_name", "adam"))
    console_print(
        "TRAIN",
        "Initialized optimizer",
        optimizer_type=type(optimizer).__name__,
        optimizer_name=optimizer_name,
        learning_rate=experiment_config["optimizer"]["learning_rate"],
        weight_decay=experiment_config["optimizer"]["weight_decay"],
    )
    scheduler, scheduler_monitor_metric = build_scheduler_from_experiment_config(
        optimizer, experiment_config
    )
    cosine_scheduler_config = None
    scheduler_config = experiment_config["optimizer"].get("scheduler")
    if scheduler_config is not None and scheduler_config["scheduler_name"] == "cosine":
        cosine_scheduler_config = {
            "base_learning_rate": float(
                experiment_config["optimizer"]["learning_rate"]
            ),
            "total_epochs": int(experiment_config["epochs"]),
            "warmup_epochs": int(scheduler_config["warmup_epochs"]),
            "warmup_start_lr": float(scheduler_config["warmup_start_lr"]),
            "cosine_end_lr": float(scheduler_config["cosine_end_lr"]),
            "cosine_after_warmup": bool(scheduler_config["cosine_after_warmup"]),
        }

    # Initialize experiment logger for tracking metrics, hyperparameters, and
    # artifacts. Logs are written to output_dir; config validates logging format.
    logging_config = dict(experiment_config.get("logging", {}))
    logging_config.setdefault("wandb_job_type", "train")
    logging_config.setdefault("wandb_run_name", experiment_config["experiment_name"])
    experiment_logger = ExperimentLogger(
        experiment_config["output_dir"],
        experiment_config=experiment_config,
        logging_config=logging_config,
    )
    console_print(
        "WANDB",
        "Prepared training logging config",
        use_wandb=logging_config.get("use_wandb", False),
        wandb_run_name=logging_config.get("wandb_run_name"),
        wandb_job_type=logging_config.get("wandb_job_type"),
        mirror_best_checkpoint_to_kaggle=logging_config.get(
            "mirror_best_checkpoint_to_kaggle", False
        ),
        mirror_output_dir_to_kaggle=logging_config.get(
            "mirror_output_dir_to_kaggle", False
        ),
    )

    # Initialize checkpoint manager for saving/loading model states, enabling
    # experiment resumption and best-model tracking across epochs.
    checkpoint_manager = CheckpointManager(
        experiment_config["checkpoint_dir"],
        artifact_sinks=experiment_logger.build_artifact_sinks(logging_config),
    )

    # Assemble the trainer with all engine components (model, optimizer, logging,
    # checkpointing). The trainer coordinates training loops, validation, and
    # checkpoint orchestration on the specified device (GPU/CPU).
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_monitor_metric=scheduler_monitor_metric,
        checkpoint_manager=checkpoint_manager,
        experiment_logger=experiment_logger,
        device=experiment_config["device"],
        cosine_scheduler_config=cosine_scheduler_config,
        gradient_clip_norm=experiment_config["optimizer"].get("gradient_clip_norm"),
        validation_evaluator_config=experiment_config.get("evaluation"),
        checkpoint_monitor_metric=experiment_config.get("checkpoint_monitor_metric"),
        enable_reconstruction_diagnostics=logging_config.get(
            "enable_reconstruction_diagnostics", False
        ),
        diagnostics_log_interval_steps=logging_config.get(
            "diagnostics_log_interval_steps", 1
        ),
        diagnostics_include_grad_norm=logging_config.get(
            "diagnostics_include_grad_norm", False
        ),
        diagnostics_stages_for_classification=logging_config.get(
            "diagnostics_stages_for_classification", []
        ),
        log_hard_prediction_ratio=logging_config.get(
            "log_hard_prediction_ratio", False
        ),
        log_row_normalized_confusion_matrix=logging_config.get(
            "log_row_normalized_confusion_matrix", False
        ),
        focus_metrics=logging_config.get("focus_metrics", []),
    )

    # Execute training with try-finally to ensure graceful logger shutdown even
    # if training fails or is interrupted. This flushes buffered metrics to disk.
    try:
        training_outputs = trainer.train(
            train_loader=data_bundle["loaders"]["train"],
            val_loader=data_bundle["loaders"]["val"],
            scaler_state=data_bundle["scaler"].state_dict(),
            config=experiment_config,
            epochs=int(experiment_config["epochs"]),
        )
        best_checkpoint_path = training_outputs["best_checkpoint_path"]
        console_print(
            "TRAIN",
            "Finished training experiment",
            best_checkpoint_path=best_checkpoint_path,
            num_logged_epochs=len(training_outputs["metric_history"]),
        )
        experiment_logger.log_summary(
            {
                "run/output_dir": str(experiment_config["output_dir"]),
                "run/checkpoint_dir": str(experiment_config["checkpoint_dir"]),
                "run/best_checkpoint_path": str(best_checkpoint_path)
                if best_checkpoint_path is not None
                else None,
            }
        )
        experiment_logger.log_artifact_file(
            file_path=experiment_logger.resolved_config_path,
            artifact_name=f"{experiment_config['experiment_name']}-resolved-config",
            artifact_type="config",
            aliases=["latest"],
            metadata={"experiment_name": experiment_config["experiment_name"]},
        )
        experiment_logger.log_artifact_file(
            file_path=experiment_logger.metrics_path,
            artifact_name=f"{experiment_config['experiment_name']}-metrics",
            artifact_type="metrics",
            aliases=["latest"],
            metadata={"experiment_name": experiment_config["experiment_name"]},
        )
        if best_checkpoint_path is not None:
            experiment_logger.log_artifact_file(
                file_path=best_checkpoint_path,
                artifact_name=f"{experiment_config['experiment_name']}-checkpoint",
                artifact_type="checkpoint",
                aliases=["best", "latest"],
                metadata={"experiment_name": experiment_config["experiment_name"]},
            )
        experiment_logger.mirror_output_directory(
            logging_config,
            metadata={
                "experiment_name": experiment_config["experiment_name"],
                "job_type": "train",
            },
        )
        return training_outputs
    finally:
        # Guarantee logger cleanup (flush buffers, close file handles) regardless
        # of training success or failure.
        experiment_logger.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-config",
        default="configs/experiment/baseline/smd__thesis_multitask__vertical-slice__w100__seed7__default.yaml",
    )
    parser.add_argument(
        "--print-config-help",
        action="store_true",
        help="Print a friendly config cheat sheet and exit.",
    )
    args = parser.parse_args()
    if args.print_config_help:
        print(build_config_help_text("train"))
        return

    experiment_config = load_experiment_config(args.experiment_config)
    console_print(
        "CONFIG",
        "Loaded CLI training experiment config",
        experiment_config_path=args.experiment_config,
    )
    run_training_experiment(experiment_config)


if __name__ == "__main__":
    main()
