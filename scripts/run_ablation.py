from __future__ import annotations

"""Thin orchestration script for YAML-driven offline ablations.

This file exists so ablation comparison does not depend on ad hoc notebook code.
It repeatedly calls the same train and evaluate helpers, then writes a compact
summary artifact that is easy to compare across runs.

A new reader should notice that the default thesis starting point is still the
simple reconstruction-plus-classification objective. The configs typically fed
into this script are later comparison runs that intentionally enable extra loss
terms or branch constraints.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).parent.parent))

from src.core.console import console_print
from src.core.config import load_experiment_config
from src.engine.logger import ExperimentLogger
from scripts.evaluate import run_evaluation_experiment
from scripts.train import run_training_experiment


def _build_summary_row(
    experiment_config_path: str,
    experiment_config: dict[str, Any],
    training_outputs: dict[str, Any],
    evaluation_outputs: dict[str, Any],
) -> dict[str, Any]:
    # The summary row keeps only the comparison fields that are most useful when
    # reading ablation outcomes side by side.
    final_epoch_metrics = training_outputs["metric_history"][-1]
    model_config = experiment_config["model"]
    task_config = experiment_config["task"]
    return {
        "experiment_config_path": experiment_config_path,
        "experiment_name": experiment_config["experiment_name"],
        "best_checkpoint_path": str(training_outputs["best_checkpoint_path"]),
        "final_train_loss": final_epoch_metrics.get("train_loss"),
        "final_val_loss": final_epoch_metrics.get("val_loss"),
        "final_train_alpha": final_epoch_metrics.get("train_alpha"),
        "final_train_beta": final_epoch_metrics.get("train_beta"),
        "final_train_temperature": final_epoch_metrics.get("train_temperature"),
        "final_train_discrete_usage_concentration": final_epoch_metrics.get(
            "train_discrete_usage_concentration"
        ),
        "roc_auc": evaluation_outputs["metrics"].get(
            "roc_auc", evaluation_outputs["metrics"].get("auroc")
        ),
        "f1": evaluation_outputs["metrics"].get(
            "f1", evaluation_outputs["metrics"].get("f1_at_threshold")
        ),
        "threshold": evaluation_outputs["metrics"]["threshold"],
        "lambda_cls": model_config["lambda_cls"],
        "lambda_div": model_config["lambda_div"],
        "lambda_var": model_config["lambda_var"],
        "lambda_cov": model_config["lambda_cov"],
        "lambda_use": model_config["lambda_use"],
        "lambda_gate": model_config["lambda_gate"],
        "freeze_fusion_for_epochs": task_config["freeze_fusion_for_epochs"],
        "warmup_alpha_value": task_config["warmup_alpha_value"],
        "warmup_beta_value": task_config["warmup_beta_value"],
        "use_synthetic_augmentation": task_config["use_synthetic_augmentation"],
        "anomaly_families": ",".join(task_config.get("anomaly_families", [])),
    }


def run_ablation_suite(
    experiment_config_paths: list[str],
    summary_output_dir: str | Path,
) -> dict[str, Any]:
    # Each ablation run still goes through the same train/evaluate codepath.
    # The orchestration layer only handles repetition and summarization.
    summary_dir = Path(summary_output_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    console_print(
        "TRAIN",
        "Starting ablation suite",
        summary_output_dir=summary_dir,
        experiment_config_paths=experiment_config_paths,
    )
    first_experiment_config = load_experiment_config(experiment_config_paths[0])
    suite_logging_config = dict(first_experiment_config.get("logging", {}))
    suite_logging_config.setdefault("wandb_job_type", "ablation_summary")
    suite_logging_config.setdefault(
        "wandb_run_name",
        f"{first_experiment_config['experiment_name']}-ablation-summary",
    )
    suite_experiment_config = {
        "experiment_name": f"{first_experiment_config['experiment_name']}_ablation_suite",
        "task": {"task_name": "ablation_suite"},
        "ablation_experiment_configs": experiment_config_paths,
    }
    suite_logger = ExperimentLogger(
        summary_dir,
        experiment_config=suite_experiment_config,
        logging_config=suite_logging_config,
    )
    console_print(
        "WANDB",
        "Prepared ablation suite logger",
        use_wandb=suite_logging_config.get("use_wandb", False),
        wandb_run_name=suite_logging_config.get("wandb_run_name"),
        wandb_job_type=suite_logging_config.get("wandb_job_type"),
    )

    try:
        for experiment_config_path in experiment_config_paths:
            console_print(
                "TRAIN",
                "Running ablation member",
                experiment_config_path=experiment_config_path,
            )
            experiment_config = load_experiment_config(experiment_config_path)
            training_outputs = run_training_experiment(experiment_config)
            evaluation_outputs = run_evaluation_experiment(
                experiment_config=experiment_config,
                checkpoint_path=str(training_outputs["best_checkpoint_path"]),
            )
            summary_rows.append(
                _build_summary_row(
                    experiment_config_path=experiment_config_path,
                    experiment_config=experiment_config,
                    training_outputs=training_outputs,
                    evaluation_outputs=evaluation_outputs,
                )
            )

        summary_json_path = summary_dir / "ablation_summary.json"
        summary_csv_path = summary_dir / "ablation_summary.csv"
        summary_json_path.write_text(
            json.dumps(summary_rows, indent=2), encoding="utf-8"
        )

        if summary_rows:
            fieldnames = list(summary_rows[0].keys())
            with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_rows)

        suite_logger.log_summary(
            {
                "ablation/num_runs": len(summary_rows),
                "ablation/summary_json_path": str(summary_json_path),
                "ablation/summary_csv_path": str(summary_csv_path),
            }
        )
        suite_logger.log_artifact_file(
            file_path=suite_logger.resolved_config_path,
            artifact_name=f"{suite_experiment_config['experiment_name']}-resolved-config",
            artifact_type="config",
            aliases=["latest"],
            metadata={"job_type": "ablation_summary"},
        )
        suite_logger.log_artifact_file(
            file_path=suite_logger.metrics_path,
            artifact_name=f"{suite_experiment_config['experiment_name']}-metrics",
            artifact_type="metrics",
            aliases=["latest"],
            metadata={"job_type": "ablation_summary"},
        )
        suite_logger.log_artifact_file(
            file_path=summary_json_path,
            artifact_name=f"{suite_experiment_config['experiment_name']}-summary-json",
            artifact_type="ablation-summary",
            aliases=["latest"],
            metadata={"job_type": "ablation_summary"},
        )
        suite_logger.log_artifact_file(
            file_path=summary_csv_path,
            artifact_name=f"{suite_experiment_config['experiment_name']}-summary-csv",
            artifact_type="ablation-summary",
            aliases=["latest"],
            metadata={"job_type": "ablation_summary"},
        )

        return {
            "summary_rows": summary_rows,
            "summary_json_path": summary_json_path,
            "summary_csv_path": summary_csv_path,
        }
    finally:
        suite_logger.close()
        console_print(
            "TRAIN", "Closed ablation suite logger", summary_output_dir=summary_dir
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-config",
        action="append",
        dest="experiment_configs",
        default=None,
        help="Repeat this flag to run multiple ablation configs.",
    )
    parser.add_argument(
        "--summary-output-dir",
        default="outputs/ablation_runs",
    )
    args = parser.parse_args()

    experiment_config_paths = args.experiment_configs or [
        "configs/experiment/smd_multitask_smoke.yaml"
    ]
    console_print(
        "CONFIG",
        "Loaded CLI ablation arguments",
        experiment_config_paths=experiment_config_paths,
        summary_output_dir=args.summary_output_dir,
    )
    run_ablation_suite(
        experiment_config_paths=experiment_config_paths,
        summary_output_dir=args.summary_output_dir,
    )


if __name__ == "__main__":
    main()
