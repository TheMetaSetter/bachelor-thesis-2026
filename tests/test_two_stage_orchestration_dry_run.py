from __future__ import annotations

from pathlib import Path

from scripts.run_two_stage_offline_pretraining import execute_two_stage_plan


def test_two_stage_dry_run_lists_evaluation_once(tmp_path: Path) -> None:
    (tmp_path / "two_stage").mkdir()
    manifest = {
        "manifest_root": str(tmp_path / "two_stage"),
        "training_stages": [
            {
                "stage_name": "stage_a_multitask_pretraining",
                "config_path": str(tmp_path / "stage_a.yaml"),
            },
            {
                "stage_name": "stage_b_fusion_finetuning",
                "config_path": str(tmp_path / "stage_b.yaml"),
            },
        ],
        "evaluation": {
            "checkpoint_path": str(tmp_path / "best.pt"),
            "config_path": str(tmp_path / "stage_b.yaml"),
        },
    }

    report = execute_two_stage_plan(manifest, dry_run=True)

    assert report["completed_stage_names"] == [
        "stage_a_multitask_pretraining",
        "stage_b_fusion_finetuning",
        "evaluation",
    ]
