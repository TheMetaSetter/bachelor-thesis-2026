from __future__ import annotations

"""Entrypoint for the first online adaptation slice.

Read this script after the offline train and evaluate scripts. It follows the
same config-driven graph, then swaps the offline dataloader and trainer for the
online stream, batcher, and online loop.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.append(str(Path(__file__).parent.parent))

from src.core.console import console_print
from src.core.config import load_experiment_config
from src.core.registry import (
    build_dataset,
    build_model,
    register_dataset,
    register_model,
)
from src.core.seed import seed_everything
from src.data.loaders import build_smd_dataset_bundle
from src.data.stream import OnlineWindowBatcher, SMDOnlineStream
from src.engine.checkpoint import CheckpointManager
from src.engine.logger import ExperimentLogger
from src.engine.online_loop import OnlineLoop
from src.models.online_adaptation import OnlineAdaptationModel
from src.models.redlamp_mlp_baseline import RedLampMLPBaseline
from src.models.reconstruction_mlp_ae import ReconstructionMLPAutoencoder
from src.models.thesis_multitask import ThesisMultitaskModel


def register_runtime_components() -> None:
    register_dataset("smd", build_smd_dataset_bundle)
    register_model("reconstruction_mlp_ae", ReconstructionMLPAutoencoder)
    register_model("thesis_multitask", ThesisMultitaskModel)
    register_model("redlamp_mlp_baseline", RedLampMLPBaseline)
    register_model("online_adaptation", OnlineAdaptationModel)
    console_print("REGISTRY", "Registered online adaptation runtime components")


def build_model_from_experiment_config(
    experiment_config: dict[str, Any],
) -> torch.nn.Module:
    # Only the online-specific task keys are passed into the online model here,
    # because the online file owns the adaptation boundary directly.
    model_name = experiment_config["model"]["model_name"]
    model_kwargs = {
        key: value
        for key, value in experiment_config["model"].items()
        if key != "model_name"
    }
    task_keys_for_model = {
        "reference_checkpoint_path",
        "warm_start_projector",
        "target_param_group",
        "clean_stream_only",
        "reset_policy",
        "reset_alignment_threshold",
    }
    model_kwargs.update(
        {
            key: value
            for key, value in experiment_config["task"].items()
            if key in task_keys_for_model
        }
    )
    console_print(
        "MODEL",
        "Building online adaptation model from experiment config",
        model_name=model_name,
        model_kwargs_keys=sorted(model_kwargs.keys()),
    )
    return build_model(model_name, **model_kwargs)


def build_optimizer_from_experiment_config(
    model: torch.nn.Module,
    experiment_config: dict[str, Any],
) -> torch.optim.Optimizer:
    optimizer_config = experiment_config["optimizer"]
    optimizer_name = str(optimizer_config.get("optimizer_name", "adam"))
    optimizer_parameters = model.get_parameter_group(
        experiment_config["task"]["target_param_group"]
    )
    optimizer_kwargs = {
        "lr": float(optimizer_config["learning_rate"]),
        "weight_decay": float(optimizer_config["weight_decay"]),
    }
    if optimizer_name == "adam":
        return torch.optim.Adam(optimizer_parameters, **optimizer_kwargs)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(optimizer_parameters, **optimizer_kwargs)
    raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")


def run_online_adaptation_experiment(
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    # The first accepted online runtime is intentionally conservative:
    # build a clean stream, adapt a small parameter group, and checkpoint often.
    seed_everything(int(experiment_config["seed"]))
    register_runtime_components()
    console_print(
        "ONLINE",
        "Starting online adaptation experiment",
        experiment_name=experiment_config["experiment_name"],
        device=experiment_config["device"],
        seed=experiment_config["seed"],
        output_dir=experiment_config["output_dir"],
        checkpoint_dir=experiment_config["checkpoint_dir"],
    )

    data_bundle = build_dataset(
        experiment_config["data"]["dataset_name"], experiment_config["data"]
    )
    console_print(
        "DATA",
        "Built dataset bundle for online adaptation",
        dataset_name=experiment_config["data"]["dataset_name"],
        test_sequences=len(data_bundle["scaled_sequences"]["test"]),
    )
    model = build_model_from_experiment_config(experiment_config)
    optimizer = build_optimizer_from_experiment_config(model, experiment_config)
    optimizer_name = str(experiment_config["optimizer"].get("optimizer_name", "adam"))
    console_print(
        "ONLINE",
        "Initialized online optimizer",
        optimizer_type=type(optimizer).__name__,
        optimizer_name=optimizer_name,
        target_param_group=experiment_config["task"]["target_param_group"],
        learning_rate=experiment_config["optimizer"]["learning_rate"],
        weight_decay=experiment_config["optimizer"]["weight_decay"],
    )
    logging_config = dict(experiment_config.get("logging", {}))
    logging_config.setdefault("wandb_job_type", "online_adaptation")
    logging_config.setdefault("wandb_run_name", experiment_config["experiment_name"])
    experiment_logger = ExperimentLogger(
        experiment_config["output_dir"],
        experiment_config=experiment_config,
        logging_config=logging_config,
    )
    console_print(
        "WANDB",
        "Prepared online adaptation logging config",
        use_wandb=logging_config.get("use_wandb", False),
        wandb_run_name=logging_config.get("wandb_run_name"),
        wandb_job_type=logging_config.get("wandb_job_type"),
    )
    checkpoint_manager = CheckpointManager(
        experiment_config["checkpoint_dir"],
        artifact_sinks=experiment_logger.build_artifact_sinks(logging_config),
    )

    online_stream = SMDOnlineStream(
        sequences=data_bundle["scaled_sequences"]["test"],
        window_size=int(experiment_config["data"]["window_size"]),
        stride=int(experiment_config["data"]["stride"]),
        clean_stream_only=bool(experiment_config["task"]["clean_stream_only"]),
        max_windows=experiment_config["task"]["max_online_steps"],
    )
    online_batcher = OnlineWindowBatcher(
        stream=online_stream,
        batch_size=int(experiment_config["data"]["batch_size"]),
        view_noise_std=float(experiment_config["task"]["view_noise_std"]),
        view_dropout_probability=float(
            experiment_config["task"]["view_dropout_probability"]
        ),
    )
    console_print(
        "ONLINE",
        "Prepared online stream and batcher",
        online_windows=len(online_stream),
        batch_size=experiment_config["data"]["batch_size"],
        view_noise_std=experiment_config["task"]["view_noise_std"],
        view_dropout_probability=experiment_config["task"]["view_dropout_probability"],
    )

    online_loop = OnlineLoop(
        model=model,
        optimizer=optimizer,
        checkpoint_manager=checkpoint_manager,
        experiment_logger=experiment_logger,
        device=experiment_config["device"],
    )
    try:
        online_outputs = online_loop.run(
            online_batcher=online_batcher,
            scaler_state=data_bundle["scaler"].state_dict(),
            config=experiment_config,
            max_online_steps=int(experiment_config["task"]["max_online_steps"]),
            log_every_n_steps=int(experiment_config["task"]["log_every_n_steps"]),
            checkpoint_every_n_steps=int(
                experiment_config["task"]["checkpoint_every_n_steps"]
            ),
        )
        console_print(
            "ONLINE",
            "Finished online adaptation experiment",
            final_checkpoint_path=online_outputs["final_checkpoint_path"],
            num_logged_steps=len(online_outputs["metric_history"]),
        )
        output_dir = Path(experiment_config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "online_metrics.json"
        records_path = output_dir / "online_records.json"

        metrics_path.write_text(
            json.dumps(online_outputs["metric_history"], indent=2), encoding="utf-8"
        )
        records_path.write_text(
            json.dumps(online_outputs["records"], indent=2), encoding="utf-8"
        )
        experiment_logger.log_summary(
            {
                "online/final_checkpoint_path": str(
                    online_outputs["final_checkpoint_path"]
                ),
                "online/num_logged_steps": len(online_outputs["metric_history"]),
            }
        )
        experiment_logger.log_artifact_file(
            file_path=experiment_logger.resolved_config_path,
            artifact_name=f"{experiment_config['experiment_name']}-resolved-config",
            artifact_type="config",
            aliases=["latest"],
            metadata={
                "experiment_name": experiment_config["experiment_name"],
                "job_type": "online_adaptation",
            },
        )
        experiment_logger.log_artifact_file(
            file_path=experiment_logger.metrics_path,
            artifact_name=f"{experiment_config['experiment_name']}-metrics",
            artifact_type="metrics",
            aliases=["latest"],
            metadata={
                "experiment_name": experiment_config["experiment_name"],
                "job_type": "online_adaptation",
            },
        )
        experiment_logger.log_artifact_file(
            file_path=metrics_path,
            artifact_name=f"{experiment_config['experiment_name']}-online-metrics",
            artifact_type="online-evaluation",
            aliases=["latest"],
            metadata={
                "experiment_name": experiment_config["experiment_name"],
                "job_type": "online_adaptation",
            },
        )
        experiment_logger.log_artifact_file(
            file_path=records_path,
            artifact_name=f"{experiment_config['experiment_name']}-online-records",
            artifact_type="online-evaluation",
            aliases=["latest"],
            metadata={
                "experiment_name": experiment_config["experiment_name"],
                "job_type": "online_adaptation",
            },
        )
        experiment_logger.log_artifact_file(
            file_path=online_outputs["final_checkpoint_path"],
            artifact_name=f"{experiment_config['experiment_name']}-checkpoint",
            artifact_type="checkpoint",
            aliases=["final", "latest"],
            metadata={
                "experiment_name": experiment_config["experiment_name"],
                "job_type": "online_adaptation",
            },
        )
        experiment_logger.mirror_output_directory(
            logging_config,
            metadata={
                "experiment_name": experiment_config["experiment_name"],
                "job_type": "online_adaptation",
            },
        )
        return online_outputs
    finally:
        experiment_logger.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-config",
        default="configs/experiment/baseline/smd__online_adaptation__online-adaptation__w100__seed7__default.yaml",
    )
    args = parser.parse_args()

    experiment_config = load_experiment_config(args.experiment_config)
    console_print(
        "CONFIG",
        "Loaded CLI online adaptation config",
        experiment_config_path=args.experiment_config,
    )
    run_online_adaptation_experiment(experiment_config)


if __name__ == "__main__":
    main()
