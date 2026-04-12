from __future__ import annotations
"""Entrypoint for offline checkpoint evaluation.

This script mirrors the training script closely on purpose. A new reader should
be able to compare the two files and immediately see that evaluation reuses the
same config-driven experiment graph, then swaps the trainer for the evaluator.
"""

import argparse
import json
from pathlib import Path

import torch

# Add the src directory to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.core.console import console_print
from src.core.config import load_experiment_config
from src.core.registry import build_dataset, build_model, register_dataset, register_model
from src.data.loaders import build_smd_dataset_bundle
from src.data.scalers import SequenceStandardScaler
from src.engine.checkpoint import CheckpointManager
from src.engine.evaluator import Evaluator
from src.engine.logger import ExperimentLogger
from src.models.reconstruction_mlp_ae import ReconstructionMLPAutoencoder
from src.models.thesis_multitask import ThesisMultitaskModel


def register_runtime_components() -> None:
    register_dataset("smd", build_smd_dataset_bundle)
    register_model("reconstruction_mlp_ae", ReconstructionMLPAutoencoder)
    register_model("thesis_multitask", ThesisMultitaskModel)
    console_print("REGISTRY", "Registered evaluation runtime components")


def build_model_from_experiment_config(experiment_config: dict) -> torch.nn.Module:
    # Evaluation rebuilds the model from config first, then checkpoint loading
    # restores the learned weights on top of that exact architecture.
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
        "Building evaluation model from experiment config",
        model_name=model_name,
        model_kwargs_keys=sorted(model_kwargs.keys()),
    )
    return build_model(model_name, **model_kwargs)


def run_evaluation_experiment(
    experiment_config: dict[str, object],
    checkpoint_path: str,
) -> dict[str, object]:
    # Persisting both metrics and the resolved config makes later thesis figures
    # easier to reproduce without hidden notebook state.
    register_runtime_components()
    console_print(
        "EVAL",
        "Starting evaluation experiment",
        experiment_name=experiment_config["experiment_name"],
        checkpoint_path=checkpoint_path,
        device=experiment_config["device"],
    )

    data_bundle = build_dataset(experiment_config["data"]["dataset_name"], experiment_config["data"])
    console_print(
        "DATA",
        "Built dataset bundle for evaluation",
        dataset_name=experiment_config["data"]["dataset_name"],
        test_windows=len(data_bundle["datasets"]["test"]),
    )
    scaler = SequenceStandardScaler()
    model = build_model_from_experiment_config(experiment_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_manager = CheckpointManager(experiment_config["checkpoint_dir"])
    loaded_checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path, model, optimizer)
    scaler.load_state_dict(loaded_checkpoint["scaler_state_dict"])
    console_print("CHECKPOINT", "Loaded checkpoint for evaluation", checkpoint_path=checkpoint_path)
    evaluator = Evaluator(device=experiment_config["device"])
    evaluation_outputs = evaluator.evaluate(model, data_bundle["loaders"]["test"])
    logging_config = dict(experiment_config.get("logging", {}))
    logging_config.setdefault("wandb_job_type", "evaluate")
    logging_config.setdefault("wandb_run_name", f"{experiment_config['experiment_name']}-evaluate")
    experiment_logger = ExperimentLogger(
        experiment_config["output_dir"],
        experiment_config=experiment_config,
        logging_config=logging_config,
        write_run_start_record=False,
        write_resolved_config=False,
    )
    console_print(
        "WANDB",
        "Prepared evaluation logging config",
        use_wandb=logging_config.get("use_wandb", False),
        wandb_run_name=logging_config.get("wandb_run_name"),
        wandb_job_type=logging_config.get("wandb_job_type"),
    )

    output_dir = Path(experiment_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "evaluation_records.json"
    metrics_path = output_dir / "evaluation_metrics.json"
    curves_path = output_dir / "evaluation_curves.json"
    resolved_config_path = output_dir / "resolved_experiment_config.json"

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
    curves_path.write_text(json.dumps(evaluation_outputs["curves"]), encoding="utf-8")
    resolved_config_path.write_text(
        json.dumps(experiment_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    prefixed_metrics = {
        f"evaluation/{metric_name}": metric_value
        for metric_name, metric_value in evaluation_outputs["metrics"].items()
    }
    experiment_logger.log_metrics(prefixed_metrics)
    experiment_logger.log_summary(prefixed_metrics | {"evaluation/checkpoint_path": checkpoint_path})
    experiment_logger.log_artifact_file(
        file_path=resolved_config_path,
        artifact_name=f"{experiment_config['experiment_name']}-resolved-config",
        artifact_type="config",
        aliases=["latest"],
        metadata={"experiment_name": experiment_config["experiment_name"], "job_type": "evaluate"},
    )
    experiment_logger.log_artifact_file(
        file_path=metrics_path,
        artifact_name=f"{experiment_config['experiment_name']}-evaluation-metrics",
        artifact_type="evaluation",
        aliases=["latest"],
        metadata={"experiment_name": experiment_config["experiment_name"], "job_type": "evaluate"},
    )
    experiment_logger.log_artifact_file(
        file_path=records_path,
        artifact_name=f"{experiment_config['experiment_name']}-evaluation-records",
        artifact_type="evaluation",
        aliases=["latest"],
        metadata={"experiment_name": experiment_config["experiment_name"], "job_type": "evaluate"},
    )
    experiment_logger.log_artifact_file(
        file_path=curves_path,
        artifact_name=f"{experiment_config['experiment_name']}-evaluation-curves",
        artifact_type="evaluation",
        aliases=["latest"],
        metadata={"experiment_name": experiment_config["experiment_name"], "job_type": "evaluate"},
    )
    experiment_logger.close()
    console_print(
        "EVAL",
        "Finished evaluation experiment",
        metrics=evaluation_outputs["metrics"],
        curves_path=curves_path,
    )
    return evaluation_outputs


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
    console_print(
        "CONFIG",
        "Loaded CLI evaluation experiment config",
        experiment_config_path=args.experiment_config,
        checkpoint_path=args.checkpoint_path,
    )
    run_evaluation_experiment(experiment_config, args.checkpoint_path)


if __name__ == "__main__":
    main()
