from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import load_experiment_config
from src.core.registry import build_dataset, build_model, register_dataset, register_model
from src.core.seed import seed_everything
from src.data.loaders import build_smd_dataset_bundle
from src.data.stream import OnlineWindowBatcher, SMDOnlineStream
from src.engine.checkpoint import CheckpointManager
from src.engine.logger import ExperimentLogger
from src.engine.online_loop import OnlineLoop
from src.models.online_adaptation import OnlineAdaptationModel
from src.models.reconstruction_mlp_ae import ReconstructionMLPAutoencoder
from src.models.thesis_multitask import ThesisMultitaskModel


def register_runtime_components() -> None:
    register_dataset("smd", build_smd_dataset_bundle)
    register_model("reconstruction_mlp_ae", ReconstructionMLPAutoencoder)
    register_model("thesis_multitask", ThesisMultitaskModel)
    register_model("online_adaptation", OnlineAdaptationModel)


def build_model_from_experiment_config(experiment_config: dict[str, Any]) -> torch.nn.Module:
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
    return build_model(model_name, **model_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-config",
        default="configs/experiment/smd_online_adaptation.yaml",
    )
    args = parser.parse_args()

    experiment_config = load_experiment_config(args.experiment_config)
    seed_everything(int(experiment_config["seed"]))
    register_runtime_components()

    data_bundle = build_dataset(experiment_config["data"]["dataset_name"], experiment_config["data"])
    model = build_model_from_experiment_config(experiment_config)
    optimizer = torch.optim.Adam(
        model.get_parameter_group(experiment_config["task"]["target_param_group"]),
        lr=float(experiment_config["optimizer"]["learning_rate"]),
        weight_decay=float(experiment_config["optimizer"]["weight_decay"]),
    )
    checkpoint_manager = CheckpointManager(experiment_config["checkpoint_dir"])
    experiment_logger = ExperimentLogger(experiment_config["output_dir"])

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
        view_dropout_probability=float(experiment_config["task"]["view_dropout_probability"]),
    )

    online_loop = OnlineLoop(
        model=model,
        optimizer=optimizer,
        checkpoint_manager=checkpoint_manager,
        experiment_logger=experiment_logger,
        device=experiment_config["device"],
    )
    online_outputs = online_loop.run(
        online_batcher=online_batcher,
        scaler_state=data_bundle["scaler"].state_dict(),
        config=experiment_config,
        max_online_steps=int(experiment_config["task"]["max_online_steps"]),
        log_every_n_steps=int(experiment_config["task"]["log_every_n_steps"]),
        checkpoint_every_n_steps=int(experiment_config["task"]["checkpoint_every_n_steps"]),
    )

    output_dir = Path(experiment_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "online_metrics.json"
    records_path = output_dir / "online_records.json"

    metrics_path.write_text(json.dumps(online_outputs["metric_history"], indent=2), encoding="utf-8")
    records_path.write_text(json.dumps(online_outputs["records"], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
