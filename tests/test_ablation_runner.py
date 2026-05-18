from __future__ import annotations

from pathlib import Path

from scripts.run_ablation import run_ablation_suite


def test_run_ablation_suite_writes_compact_summary_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_train(experiment_config: dict[str, object]) -> dict[str, object]:
        return {
            "best_checkpoint_path": tmp_path / "best.pt",
            "metric_history": [
                {
                    "train_loss": 1.0,
                    "val_loss": 0.8,
                    "train_alpha": 0.4,
                    "train_beta": 0.6,
                    "train_temperature": 1.1,
                    "train_discrete_usage_concentration": 0.2,
                }
            ],
        }

    def fake_evaluate(
        experiment_config: dict[str, object], checkpoint_path: str
    ) -> dict[str, object]:
        return {
            "metrics": {
                "roc_auc": 0.75,
                "pr_auc": 0.65,
                "vus_pr": 0.61,
                "f1": 0.5,
                "threshold": 0.12,
            }
        }

    monkeypatch.setattr("scripts.run_ablation.run_training_experiment", fake_train)
    monkeypatch.setattr("scripts.run_ablation.run_evaluation_experiment", fake_evaluate)

    outputs = run_ablation_suite(
        experiment_config_paths=["configs/experiment/smd_multitask_smoke.yaml"],
        summary_output_dir=tmp_path / "summary",
    )

    assert outputs["summary_json_path"].exists()
    assert outputs["summary_csv_path"].exists()
    assert outputs["summary_rows"][0]["experiment_name"] == "smd_multitask_smoke"
    assert outputs["summary_rows"][0]["pr_auc"] == 0.65
    assert outputs["summary_rows"][0]["vus_pr"] == 0.61
    assert outputs["summary_rows"][0]["bootstrap_encoder_epochs"] == 10
    assert outputs["summary_rows"][0]["anomaly_families"]
