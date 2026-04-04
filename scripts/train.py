from __future__ import annotations
"""Entrypoint for offline training experiments.

A fresher can read this script as the shortest explanation of the runtime graph:
load config, register components, build data, build model, create the engine
objects, then hand everything to the trainer.
"""

import argparse

import torch

# Add src to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

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
    return build_model(model_name, **model_kwargs)


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
    
    # Register dataset and model builders into the global registry. This decouples
    # experiment configuration (which uses string names) from actual constructors,
    # allowing experiments to be defined in YAML without hardcoded imports.
    register_runtime_components()

    # Load and preprocess the dataset specified in config. Returns a bundle containing
    # train/val data loaders and a fitted scaler (for input normalization).
    data_bundle = build_dataset(experiment_config["data"]["dataset_name"], experiment_config["data"])
    
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
    
    # Initialize checkpoint manager for saving/loading model states, enabling
    # experiment resumption and best-model tracking across epochs.
    checkpoint_manager = CheckpointManager(experiment_config["checkpoint_dir"])
    
    # Initialize experiment logger for tracking metrics, hyperparameters, and
    # artifacts. Logs are written to output_dir; config validates logging format.
    experiment_logger = ExperimentLogger(
        experiment_config["output_dir"],
        experiment_config=experiment_config,
        logging_config=experiment_config.get("logging"),
    )
    
    # Assemble the trainer with all engine components (model, optimizer, logging,
    # checkpointing). The trainer coordinates training loops, validation, and
    # checkpoint orchestration on the specified device (GPU/CPU).
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        checkpoint_manager=checkpoint_manager,
        experiment_logger=experiment_logger,
        device=experiment_config["device"],
    )
    
    # Execute training with try-finally to ensure graceful logger shutdown even
    # if training fails or is interrupted. This flushes buffered metrics to disk.
    try:
        return trainer.train(
            train_loader=data_bundle["loaders"]["train"],
            val_loader=data_bundle["loaders"]["val"],
            scaler_state=data_bundle["scaler"].state_dict(),
            config=experiment_config,
            epochs=int(experiment_config["epochs"]),
        )
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
    run_training_experiment(experiment_config)


if __name__ == "__main__":
    main()
