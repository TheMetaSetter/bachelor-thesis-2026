from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import load_experiment_config, validate_experiment_config
from src.models.redlamp_baseline import RedLampBaseline
from src.models.thesis_multitask import ThesisMultitaskModel

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_experiment_config(
    tmp_path: Path,
    *,
    model_name: str,
    model_config_path: str,
) -> Path:
    experiment_path = tmp_path / f"{model_name}_experiment.yaml"
    experiment_path.write_text(
        "\n".join(
            [
                "experiment_name: cnn-config-regression",
                "seed: 7",
                "device: cpu",
                "output_dir: outputs/test",
                "checkpoint_dir: outputs/test/checkpoints",
                f"data_config_path: {REPO_ROOT / 'configs/data/smd_rtx3090_machine_2_1_20.yaml'}",
                f"model_config_path: {model_config_path}",
                f"task_config_path: {REPO_ROOT / 'configs/task/multitask_tsad_redlamp_multiclass_window20.yaml'}",
                "optimizer:",
                "  optimizer_name: adamw",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "  scheduler:",
                "    scheduler_name: reduce_on_plateau",
                "    monitor_metric: val_synth_vus_pr",
                "    factor: 0.5",
                "    patience: 5",
                "    threshold: 0.0001",
                "    threshold_mode: rel",
                "    cooldown: 0",
                "    min_lr: 1.0e-6",
                "checkpoint_monitor_metric: val_synth_vus_pr",
                "epochs: 1",
            ]
        ),
        encoding="utf-8",
    )
    return experiment_path


def _write_cnn_model_config(
    tmp_path: Path,
    *,
    source_path: Path,
    encoder_family: str,
) -> Path:
    model_path = tmp_path / f"{source_path.stem}_{encoder_family}.yaml"
    model_text = source_path.read_text(encoding="utf-8")
    model_text = model_text.replace(
        "encoder_family: mlp", f"encoder_family: {encoder_family}"
    )
    model_path.write_text(model_text, encoding="utf-8")
    return model_path


def test_mlp_and_cnn_model_configs_load_and_surface_encoder_family(
    tmp_path: Path,
) -> None:
    baseline_experiment_path = _write_experiment_config(
        tmp_path,
        model_name="redlamp_baseline",
        model_config_path=str(REPO_ROOT / "configs/model/redlamp_baseline.yaml"),
    )
    cnn_model_config_path = _write_cnn_model_config(
        tmp_path,
        source_path=REPO_ROOT
        / "configs/model/thesis_multitask_redlamp_multiclass.yaml",
        encoder_family="cnn_simple",
    )
    thesis_experiment_path = _write_experiment_config(
        tmp_path,
        model_name="thesis_multitask",
        model_config_path=str(cnn_model_config_path),
    )

    baseline_config = load_experiment_config(baseline_experiment_path)
    thesis_config = load_experiment_config(thesis_experiment_path)

    validate_experiment_config(baseline_config)
    validate_experiment_config(thesis_config)

    baseline_model = RedLampBaseline(**baseline_config["model"])
    thesis_model_kwargs = dict(thesis_config["model"])
    thesis_model_kwargs.pop("model_name", None)
    thesis_model = ThesisMultitaskModel(**thesis_model_kwargs)

    assert baseline_model.encoder_family == "mlp"
    assert thesis_model.model_config.architecture.encoder_family == "cnn_simple"


def test_invalid_encoder_family_is_rejected(tmp_path: Path) -> None:
    experiment_path = _write_experiment_config(
        tmp_path,
        model_name="redlamp_baseline",
        model_config_path=str(REPO_ROOT / "configs/model/redlamp_baseline.yaml"),
    )
    experiment_config = load_experiment_config(experiment_path)
    experiment_config["model"]["encoder_family"] = "not-a-valid-family"

    with pytest.raises(ValueError, match="encoder_family"):
        validate_experiment_config(experiment_config)
