from __future__ import annotations

import argparse

import torch

# Add src to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import load_experiment_config
from src.core.registry import build_model, build_task, register_dataset, register_model, register_task
from src.core.seed import seed_everything
from src.data.loaders import build_smd_dataloaders
from src.engine.checkpoint import CheckpointManager
from src.engine.logger import ExperimentLogger
from src.engine.trainer import Trainer
from src.models.reconstruction_mlp_ae import ReconstructionMLPAutoencoder
from src.tasks.reconstruction_task import ReconstructionTask


def register_phase_one_components() -> None:
    register_dataset("smd", build_smd_dataloaders)
    register_model("reconstruction_mlp_ae", ReconstructionMLPAutoencoder)
    register_task("reconstruction", ReconstructionTask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-config",
        default="configs/experiment/smd_vertical_slice.yaml",
    )
    args = parser.parse_args()

    experiment_config = load_experiment_config(args.experiment_config)
    seed_everything(int(experiment_config["seed"]))
    register_phase_one_components()

    data_bundle = build_smd_dataloaders(experiment_config["data"])
    model = build_model(experiment_config["model"]["model_name"], **{
        key: value
        for key, value in experiment_config["model"].items()
        if key != "model_name"
    })
    task = build_task(experiment_config["task"]["task_name"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(experiment_config["optimizer"]["learning_rate"]),
        weight_decay=float(experiment_config["optimizer"]["weight_decay"]),
    )
    checkpoint_manager = CheckpointManager(experiment_config["checkpoint_dir"])
    experiment_logger = ExperimentLogger(experiment_config["output_dir"])
    trainer = Trainer(
        model=model,
        task=task,
        optimizer=optimizer,
        checkpoint_manager=checkpoint_manager,
        experiment_logger=experiment_logger,
        device=experiment_config["device"],
    )
    trainer.train(
        train_loader=data_bundle["loaders"]["train"],
        val_loader=data_bundle["loaders"]["val"],
        scaler_state=data_bundle["scaler"].state_dict(),
        config=experiment_config,
        epochs=int(experiment_config["epochs"]),
    )


if __name__ == "__main__":
    main()

