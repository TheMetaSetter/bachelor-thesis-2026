from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# Add the src directory to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import load_experiment_config
from src.core.registry import build_model, build_task, register_dataset, register_model, register_task
from src.data.loaders import build_smd_dataloaders
from src.data.scalers import SequenceStandardScaler
from src.engine.checkpoint import CheckpointManager
from src.engine.evaluator import Evaluator
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
    parser.add_argument(
        "--checkpoint-path",
        default="outputs/smd_vertical_slice/checkpoints/best.pt",
    )
    args = parser.parse_args()

    experiment_config = load_experiment_config(args.experiment_config)
    register_phase_one_components()

    data_bundle = build_smd_dataloaders(experiment_config["data"])
    scaler = SequenceStandardScaler()
    model = build_model(experiment_config["model"]["model_name"], **{
        key: value
        for key, value in experiment_config["model"].items()
        if key != "model_name"
    })
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_manager = CheckpointManager(experiment_config["checkpoint_dir"])
    loaded_checkpoint = checkpoint_manager.load_checkpoint(args.checkpoint_path, model, optimizer)
    scaler.load_state_dict(loaded_checkpoint["scaler_state_dict"])
    task = build_task(experiment_config["task"]["task_name"])
    evaluator = Evaluator(task=task, device=experiment_config["device"])
    evaluation_outputs = evaluator.evaluate(model, data_bundle["loaders"]["test"])

    output_dir = Path(experiment_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "evaluation_records.json"
    metrics_path = output_dir / "evaluation_metrics.json"

    serializable_records = [
        {
            "entity_id": record["entity_id"],
            "point_scores": record["point_scores"].tolist(),
            "point_labels": record["point_labels"].tolist(),
            "num_points": record["num_points"],
        }
        for record in evaluation_outputs["records"]
    ]
    records_path.write_text(json.dumps(serializable_records), encoding="utf-8")
    metrics_path.write_text(json.dumps(evaluation_outputs["metrics"]), encoding="utf-8")


if __name__ == "__main__":
    main()

