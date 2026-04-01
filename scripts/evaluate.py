from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# Add the src directory to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import load_experiment_config
from src.core.registry import build_dataset, build_model, register_dataset, register_model
from src.data.loaders import build_smd_dataset_bundle
from src.data.scalers import SequenceStandardScaler
from src.engine.checkpoint import CheckpointManager
from src.engine.evaluator import Evaluator
from src.models.reconstruction_mlp_ae import ReconstructionMLPAutoencoder
from src.models.thesis_multitask import ThesisMultitaskModel


def register_runtime_components() -> None:
    register_dataset("smd", build_smd_dataset_bundle)
    register_model("reconstruction_mlp_ae", ReconstructionMLPAutoencoder)
    register_model("thesis_multitask", ThesisMultitaskModel)


def build_model_from_experiment_config(experiment_config: dict) -> torch.nn.Module:
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
    register_runtime_components()

    data_bundle = build_dataset(experiment_config["data"]["dataset_name"], experiment_config["data"])
    scaler = SequenceStandardScaler()
    model = build_model_from_experiment_config(experiment_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_manager = CheckpointManager(experiment_config["checkpoint_dir"])
    loaded_checkpoint = checkpoint_manager.load_checkpoint(args.checkpoint_path, model, optimizer)
    scaler.load_state_dict(loaded_checkpoint["scaler_state_dict"])
    evaluator = Evaluator(device=experiment_config["device"])
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
