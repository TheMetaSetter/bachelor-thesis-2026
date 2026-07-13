from __future__ import annotations

from pathlib import Path

import torch

from scripts.run_online_adaptation import (
    build_optimizer_from_experiment_config,
    run_online_adaptation_experiment,
)
from src.engine.online_tta.online_engine_run import _build_runtime_online_context


class _FakeOnlineModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projector_weight = torch.nn.Parameter(torch.ones(1))
        self.target_param_group = "projector_params"

    def get_parameter_group(self, target_param_group: str) -> list[torch.nn.Parameter]:
        assert target_param_group == "projector_params"
        return [self.projector_weight]


class _FakeOnlineLoop:
    def __init__(
        self, model, optimizer, checkpoint_manager, experiment_logger, device: str
    ) -> None:
        self.model = model

    def run(
        self,
        online_batcher,
        scaler_state: dict[str, object],
        config: dict[str, object],
        max_online_steps: int,
        log_every_n_steps: int,
        checkpoint_every_n_steps: int,
    ) -> dict[str, object]:
        return {
            "final_checkpoint_path": Path(config["checkpoint_dir"]) / "online_final.pt",
            "metric_history": [{"online/step": 1, "online/alignment_loss": 0.1}],
            "records": [{"step": 1, "entity_ids": ["machine-1"]}],
        }


class _SpyOnlineBenchmarkModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projector_weight = torch.nn.Parameter(torch.ones(1))
        self.to_called_with: str | None = None

    def to(self, *args, **kwargs):  # type: ignore[override]
        if args:
            self.to_called_with = str(args[0])
        elif "device" in kwargs:
            self.to_called_with = str(kwargs["device"])
        return self

    def get_parameter_group(self, target_param_group: str) -> list[torch.nn.Parameter]:
        assert target_param_group == "projector_params"
        return [self.projector_weight]


def test_build_online_optimizer_supports_adamw() -> None:
    model = _FakeOnlineModel()

    optimizer = build_optimizer_from_experiment_config(
        model,
        {
            "task": {"target_param_group": "projector_params"},
            "optimizer": {
                "optimizer_name": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
            },
        },
    )

    assert isinstance(optimizer, torch.optim.AdamW)


def test_run_online_adaptation_experiment_writes_summary_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "scripts.run_online_adaptation.register_runtime_components", lambda: None
    )
    monkeypatch.setattr(
        "scripts.run_online_adaptation.build_dataset",
        lambda dataset_name, data_config: {
            "scaled_sequences": {
                "test": [
                    {
                        "x": torch.randn(120, 38),
                        "point_labels": torch.zeros(120, dtype=torch.long),
                        "mask": torch.ones(120, 38),
                        "timestamps": torch.arange(120),
                        "meta": {
                            "dataset_name": "smd",
                            "entity_id": "machine-1",
                            "split": "test",
                            "num_channels": 38,
                            "sequence_length": 120,
                        },
                    }
                ]
            },
            "scaler": type(
                "Scaler",
                (),
                {
                    "state_dict": staticmethod(
                        lambda: {
                            "feature_mean": torch.zeros(38),
                            "feature_std": torch.ones(38),
                        }
                    )
                },
            )(),
        },
    )
    monkeypatch.setattr(
        "scripts.run_online_adaptation.build_model_from_experiment_config",
        lambda experiment_config: _FakeOnlineModel(),
    )
    monkeypatch.setattr("scripts.run_online_adaptation.OnlineLoop", _FakeOnlineLoop)

    experiment_config = {
        "experiment_name": "online-smoke",
        "seed": 7,
        "device": "cpu",
        "output_dir": str(tmp_path / "outputs"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "data": {
            "dataset_name": "smd",
            "window_size": 100,
            "stride": 10,
            "batch_size": 1,
        },
        "model": {"model_name": "online_adaptation"},
        "task": {
            "target_param_group": "projector_params",
            "clean_stream_only": True,
            "max_online_steps": 1,
            "log_every_n_steps": 1,
            "checkpoint_every_n_steps": 1,
            "view_noise_std": 0.0,
            "view_dropout_probability": 0.0,
        },
        "optimizer": {"learning_rate": 0.001, "weight_decay": 0.0},
        "epochs": 1,
    }

    outputs = run_online_adaptation_experiment(experiment_config)

    assert outputs["metric_history"][0]["online/step"] == 1
    assert (tmp_path / "outputs" / "online_metrics.json").exists()
    assert (tmp_path / "outputs" / "online_records.json").exists()


def test_online_benchmark_moves_model_to_device_before_calibration(
    monkeypatch, tmp_path: Path
) -> None:
    model = _SpyOnlineBenchmarkModel()

    monkeypatch.setattr(
        "src.engine.online_tta.online_engine_run.build_dataset",
        lambda dataset_name, data_config: {
            "scaled_sequences": {
                "val": [
                    {
                        "x": torch.randn(30, 38),
                        "point_labels": torch.zeros(30, dtype=torch.long),
                        "mask": torch.ones(30, 38),
                        "timestamps": torch.arange(30),
                        "meta": {
                            "dataset_name": "smd",
                            "entity_id": "machine-1-6",
                            "split": "val",
                            "num_channels": 38,
                            "sequence_length": 30,
                        },
                    }
                ],
                "test": [
                    {
                        "x": torch.randn(30, 38),
                        "point_labels": torch.zeros(30, dtype=torch.long),
                        "mask": torch.ones(30, 38),
                        "timestamps": torch.arange(30),
                        "meta": {
                            "dataset_name": "smd",
                            "entity_id": "machine-1-6",
                            "split": "test",
                            "num_channels": 38,
                            "sequence_length": 30,
                        },
                    }
                ],
            },
            "scaler": type(
                "Scaler",
                (),
                {
                    "state_dict": staticmethod(
                        lambda: {
                            "feature_mean": torch.zeros(38),
                            "feature_std": torch.ones(38),
                        }
                    )
                },
            )(),
        },
    )
    monkeypatch.setattr(
        "src.engine.online_tta.online_engine_run._build_model_from_experiment_config",
        lambda experiment_config: model,
    )
    monkeypatch.setattr(
        "src.engine.online_tta.online_engine_run._build_optimizer_from_experiment_config",
        lambda model, experiment_config: torch.optim.Adam(
            model.get_parameter_group("projector_params"), lr=1e-3
        ),
    )
    monkeypatch.setattr(
        "src.engine.online_tta.online_engine_run.assert_only_projector_is_trainable",
        lambda model: None,
    )
    monkeypatch.setattr(
        "src.engine.online_tta.online_engine_run.calibrate_entity_threshold_artifacts",
        lambda **kwargs: {
            "machine-1-6": {
                "entity_id": "machine-1-6",
                "thresholds": {"online_ewma_point": {"value": 0.5}},
            }
        },
    )
    monkeypatch.setattr(
        "src.engine.online_tta.online_engine_run.write_threshold_artifact",
        lambda artifact, path: path,
    )
    monkeypatch.setattr(
        "src.engine.online_tta.online_engine_run.build_online_runtime_state",
        lambda **kwargs: type(
            "RuntimeState",
            (),
            {
                "stream_cursor": 0,
                "previous_ewma_score": None,
                "signature_history": [],
                "recurrent_signatures": [],
                "verification_history": [],
                "verification_entries": [],
                "hard_old_intervals": [],
                "to_dict": lambda self: {"stream_cursor": 0},
            },
        )(),
    )
    monkeypatch.setattr(
        "src.engine.online_tta.online_engine_run.CheckpointManager",
        type(
            "FakeCheckpointManager",
            (),
            {
                "__init__": lambda self, checkpoint_dir: setattr(
                    self, "checkpoint_dir", Path(checkpoint_dir)
                ),
            },
        ),
    )

    context = _build_runtime_online_context(
        experiment_config={
            "experiment_name": "online-device-spy",
            "seed": 7,
            "device": "cuda",
            "output_dir": str(tmp_path / "outputs"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "data": {"dataset_name": "smd", "window_size": 20, "batch_size": 1},
            "model": {"model_name": "online_adaptation"},
            "task": {
                "target_param_group": "projector_params",
                "clean_stream_only": True,
                "max_online_steps": 1,
            },
            "optimizer": {"learning_rate": 0.001, "weight_decay": 0.0},
        },
        protocol_config={
            "window_size": 20,
            "online_ewma_current_weight": 0.9,
            "online_ewma_previous_weight": 0.1,
            "online_threshold_quantile": 0.99,
            "offline_threshold_quantile": 0.99,
        },
        online_variant="A0",
    )

    assert model.to_called_with == "cuda"
    assert context["device"] == "cuda"
