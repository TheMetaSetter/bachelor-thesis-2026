from __future__ import annotations

from src.core.config import load_yaml_config, validate_experiment_config
from src.data.augment import REDLAMP_ANOMALY_FAMILIES


def test_redlamp_baseline_model_config_file_is_canonical() -> None:
    model_config = load_yaml_config("configs/model/redlamp_baseline.yaml")

    assert model_config["model_name"] == "redlamp_baseline"


def test_validate_experiment_config_accepts_redlamp_baseline_model_name() -> None:
    experiment_config = {
        "experiment_name": "redlamp-baseline-config-ok",
        "seed": 7,
        "device": "cpu",
        "output_dir": "outputs/redlamp-baseline-config-ok",
        "checkpoint_dir": "outputs/redlamp-baseline-config-ok/checkpoints",
        "data": {
            "dataset_name": "smd",
            "root_dir": "data/ServerMachineDataset",
            "window_size": 20,
            "stride": 1,
            "batch_size": 2,
            "num_workers": 0,
            "validation_split_ratio": 0.2,
            "shuffle_train": False,
        },
        "model": {
            "model_name": "redlamp_baseline",
            "input_dim": 38,
            "window_size": 20,
            "latent_dim": 16,
            "encoder_family": "cnn_simple",
            "mlp_num_linear_layers": 3,
            "cnn_num_layers": 3,
            "cnn_kernel_size": 3,
            "cnn_hidden_channels": 8,
            "cnn_dropout": 0.1,
            "classifier_dim": 8,
            "num_classes": 12,
            "dropout": 0.1,
            "lambda_recon": 0.9,
            "lambda_cls": 0.1,
            "use_label_refurbishment": True,
            "refurbishment_alpha": 0.1,
            "refurbishment_beta": 0.01,
            "enable_gradient_conflict_profiling": False,
            "gradient_profiling_scope": "encoder_all",
            "gradient_focus_layer_name": "encoder_last_affine",
            "gradient_log_every_n_steps": 1,
            "gradient_ema_alpha": 0.1,
            "gradient_sma_window": 50,
            "gradient_profile_include_bias": False,
        },
        "task": {
            "task_name": "multitask_tsad",
            "use_synthetic_augmentation": True,
            "use_synthetic_validation": True,
            "synthetic_train_seed": 17,
            "synthetic_validation_seed": 7,
            "classification_label_mode": "redlamp_multiclass",
            "freeze_fusion_for_epochs": 0,
            "warmup_alpha_value": 0.5,
            "warmup_beta_value": 0.5,
            "anomaly_probability": 0.5,
            "train_balance_classes": True,
            "val_realistic": False,
            "val_realistic_source": "test_same_scope",
            "val_anomaly_rate_override": None,
            "min_segment_fraction": 0.2,
            "max_segment_fraction": 0.3,
            "spike_scale": 3.0,
            "anomaly_visibility_boost": 1.5,
            "anomaly_families": list(REDLAMP_ANOMALY_FAMILIES),
        },
        "optimizer": {
            "optimizer_name": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "gradient_clip_norm": 0.5,
        },
        "epochs": 1,
        "checkpoint_monitor_metric": "val_synth_vus_pr",
    }

    validate_experiment_config(experiment_config)
