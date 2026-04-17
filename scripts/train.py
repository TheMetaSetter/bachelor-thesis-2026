from __future__ import annotations
"""Entrypoint for offline training experiments.

A fresher can read this script as the shortest explanation of the runtime graph:
load config, register components, build data, build model, create the engine
objects, then hand everything to the trainer.
"""

import argparse
from typing import Any

import torch

# Add src to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.console import console_print
from src.core.config import load_experiment_config
from src.core.registry import build_dataset, build_model, register_dataset, register_model
from src.core.seed import seed_everything
from src.data.loaders import build_smd_dataset_bundle
from src.engine.checkpoint import CheckpointManager
from src.engine.logger import ExperimentLogger
from src.engine.trainer import Trainer
from src.models.reconstruction_mlp_ae import ReconstructionMLPAutoencoder
from src.models.thesis_multitask import ThesisMultitaskModel


def register_runtime_components() -> None:
    # Registration keeps script wiring explicit while still letting experiments
    # build datasets and models from names instead of hard-coded constructors.
    register_dataset("smd", build_smd_dataset_bundle)
    register_model("reconstruction_mlp_ae", ReconstructionMLPAutoencoder)
    register_model("thesis_multitask", ThesisMultitaskModel)
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
    console_print(
        "MODEL",
        "Building model from resolved experiment config",
        model_name=model_name,
        model_kwargs_keys=sorted(model_kwargs.keys()),
    )
    return build_model(model_name, **model_kwargs)


def build_scheduler_from_experiment_config(
    optimizer: torch.optim.Optimizer,
    experiment_config: dict[str, object],
) -> Any | None:
    optimizer_config = experiment_config["optimizer"]
    scheduler_config = optimizer_config.get("scheduler")
    if scheduler_config is None:
        console_print("TRAIN", "No learning rate scheduler configured")
        return None

    scheduler_name = scheduler_config["scheduler_name"]
    if scheduler_name != "reduce_on_plateau":
        raise ValueError(f"Unsupported scheduler_name: {scheduler_name}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
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
        monitor_metric=scheduler_config["monitor_metric"],
        factor=scheduler_config["factor"],
        patience=scheduler_config["patience"],
        threshold=scheduler_config["threshold"],
        threshold_mode=scheduler_config["threshold_mode"],
        cooldown=scheduler_config["cooldown"],
        min_lr=scheduler_config["min_lr"],
    )
    return scheduler


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
    data_bundle = build_dataset(experiment_config["data"]["dataset_name"], experiment_config["data"])
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
    
    # Create the Adam optimizer with learning rate and weight decay from config.
    # Adam is chosen for its adaptive learning rate and stable convergence properties.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(experiment_config["optimizer"]["learning_rate"]),
        weight_decay=float(experiment_config["optimizer"]["weight_decay"]),
    )
    console_print(
        "TRAIN",
        "Initialized optimizer",
        optimizer_type="Adam",
        learning_rate=experiment_config["optimizer"]["learning_rate"],
        weight_decay=experiment_config["optimizer"]["weight_decay"],
    )
    scheduler = build_scheduler_from_experiment_config(optimizer, experiment_config)
    
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
        mirror_best_checkpoint_to_kaggle=logging_config.get("mirror_best_checkpoint_to_kaggle", False),
        mirror_output_dir_to_kaggle=logging_config.get("mirror_output_dir_to_kaggle", False),
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
        checkpoint_manager=checkpoint_manager,
        experiment_logger=experiment_logger,
        device=experiment_config["device"],
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
                "run/best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
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
            metadata={"experiment_name": experiment_config["experiment_name"], "job_type": "train"},
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
        default="configs/experiment/smd_vertical_slice.yaml",
    )
    args = parser.parse_args()

    experiment_config = load_experiment_config(args.experiment_config)
    console_print("CONFIG", "Loaded CLI training experiment config", experiment_config_path=args.experiment_config)
    run_training_experiment(experiment_config)


if __name__ == "__main__":
    main()
