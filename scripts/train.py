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
    # This helper is shared by the CLI and by tests or orchestration scripts.
    seed_everything(int(experiment_config["seed"]))
    register_runtime_components()

    data_bundle = build_dataset(experiment_config["data"]["dataset_name"], experiment_config["data"])
    model = build_model_from_experiment_config(experiment_config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(experiment_config["optimizer"]["learning_rate"]),
        weight_decay=float(experiment_config["optimizer"]["weight_decay"]),
    )
    checkpoint_manager = CheckpointManager(experiment_config["checkpoint_dir"])
    experiment_logger = ExperimentLogger(
        experiment_config["output_dir"],
        experiment_config=experiment_config,
        logging_config=experiment_config.get("logging"),
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        checkpoint_manager=checkpoint_manager,
        experiment_logger=experiment_logger,
        device=experiment_config["device"],
    )
    try:
        return trainer.train(
            train_loader=data_bundle["loaders"]["train"],
            val_loader=data_bundle["loaders"]["val"],
            scaler_state=data_bundle["scaler"].state_dict(),
            config=experiment_config,
            epochs=int(experiment_config["epochs"]),
        )
    finally:
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
