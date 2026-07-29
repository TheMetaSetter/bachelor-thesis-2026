from __future__ import annotations

from pathlib import Path

from scripts.train import run_training_experiment
from src.core.console import console_print
from src.data.loaders import build_smd_dataset_bundle
from src.models.thesis_multitask import ThesisMultitaskModel


def test_console_print_formats_prefixed_message(capsys) -> None:
    console_print("TRAIN", "Test message", epoch=1, loss=0.5)

    captured = capsys.readouterr()

    assert "[TRAIN] Test message" in captured.out
    assert "epoch=1" in captured.out
    assert "loss=0.500000" in captured.out


def test_smd_builder_emits_console_dataset_summary(capsys) -> None:
    build_smd_dataset_bundle(
        {
            "dataset_name": "smd",
            "root_dir": "data/ServerMachineDataset",
            "window_size": 100,
            "stride": 10,
            "batch_size": 2,
            "num_workers": 0,
            "validation_split_ratio": 0.2,
            "shuffle_train": True,
            "max_train_windows": 4,
            "max_val_windows": 2,
            "max_test_windows": 2,
        }
    )

    captured = capsys.readouterr()

    assert "[DATA] Resolved SMD dataset root" in captured.out
    assert "[DATA] Completed SMD parsing" in captured.out
    assert "[DATA] Built data loaders" in captured.out


def test_model_construction_emits_parameter_summary(capsys) -> None:
    ThesisMultitaskModel(
        input_dim=38,
        window_size=100,
        encoder_dim=16,
        hidden_dim=8,
        num_classes=2,
        dropout=0.0,
        continuous_enabled=True,
        continuous_num_prototypes=4,
        discrete_enabled=True,
        discrete_codebook_size=6,
        gumbel_temperature=1.0,
        temperature_start=1.0,
        temperature_end=0.8,
        temperature_anneal_fraction=1.0,
        alpha_logit_init=0.0,
        beta_logit_init=0.0,
        lambda_cls=1.0,
        lambda_div=0.0,
        lambda_var=0.0,
        lambda_cov=0.0,
        lambda_use=0.01,
        lambda_gate=0.0,
        use_synthetic_augmentation=False,
        freeze_fusion_for_epochs=0,
        warmup_alpha_value=0.5,
        warmup_beta_value=0.5,
        anomaly_probability=0.5,
        min_segment_fraction=0.1,
        max_segment_fraction=0.2,
        spike_scale=3.0,
    )

    captured = capsys.readouterr()

    assert "Parameter summary for ThesisMultitaskModel" in captured.out
    assert "Component parameters: encoder" in captured.out
    assert "Component parameters: continuous_prototype_bank" in captured.out
    assert "lambda_use=0.010000" in captured.out


def test_run_training_experiment_emits_runtime_console_messages(
    capsys, tmp_path: Path
) -> None:
    experiment_config = {
        "experiment_name": "console_instrumentation_smoke",
        "seed": 7,
        "device": "cpu",
        "output_dir": str(tmp_path / "outputs"),
        "checkpoint_dir": str(tmp_path / "outputs" / "checkpoints"),
        "epochs": 1,
        "data": {
            "dataset_name": "smd",
            "root_dir": "data/ServerMachineDataset",
            "window_size": 100,
            "stride": 10,
            "batch_size": 2,
            "num_workers": 0,
            "validation_split_ratio": 0.2,
            "shuffle_train": True,
            "max_train_windows": 4,
            "max_val_windows": 2,
            "max_test_windows": 2,
        },
        "model": {
            "model_name": "thesis_multitask",
            "input_dim": 38,
            "encoder_dim": 16,
            "hidden_dim": 8,
            "dropout": 0.0,
            "num_classes": 2,
            "training_phase": "stage_a_multitask_pretraining",
            "bootstrap_encoder_epochs": 0,
            "gumbel_temperature": 1.0,
            "temperature_start": 1.0,
            "temperature_end": 1.0,
            "temperature_anneal_fraction": 1.0,
            "alpha_logit_init": 0.0,
            "beta_logit_init": 0.0,
            "lambda_cls": 1.0,
            "lambda_div": 0.0,
            "lambda_var": 0.0,
            "lambda_cov": 0.0,
            "lambda_use": 0.0,
            "lambda_gate": 0.0,
        },
        "task": {
            "task_name": "multitask_tsad",
            "use_synthetic_augmentation": False,
            "anomaly_probability": 0.5,
            "min_segment_fraction": 0.1,
            "max_segment_fraction": 0.2,
            "spike_scale": 3.0,
            "anomaly_families": ["spike"],
        },
        "optimizer": {
            "learning_rate": 0.001,
            "weight_decay": 0.0,
        },
        "logging": {
            "use_wandb": False,
        },
    }

    training_outputs = run_training_experiment(experiment_config)

    captured = capsys.readouterr()

    assert training_outputs["best_checkpoint_path"] is not None
    assert "[TRAIN] Completed optimizer step" in captured.out
    assert "[CHECKPOINT] Saving checkpoint" in captured.out
    assert "[WANDB] Logged metrics to JSONL" in captured.out
