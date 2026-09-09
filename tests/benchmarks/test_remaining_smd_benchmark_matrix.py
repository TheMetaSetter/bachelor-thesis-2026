from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

import scripts.benchmarks.generate_remaining_smd_benchmark_configs as generator
from scripts.benchmarks.collect_remaining_smd_metrics import (
    extract_requested_metrics,
)
from src.metrics.online_contract import compute_final_online_metrics
from scripts.benchmarks.generate_remaining_smd_benchmark_configs import (
    EXCLUDED_ENTITY_IDS,
    build_matrix_plan,
    discover_remaining_entities,
    mode_settings,
    select_short_online_range,
    write_matrix_configs,
)
from src.core.artifact_naming import is_valid_wandb_smoke_run_name


def _write_entity_files(dataset_root: Path, entity_ids: list[str]) -> None:
    for split in ("train", "test", "test_label"):
        split_dir = dataset_root / split
        split_dir.mkdir(parents=True)
        for entity_id in entity_ids:
            (split_dir / f"{entity_id}.txt").write_text("0\n", encoding="utf-8")


def test_discovery_excludes_completed_entities(tmp_path: Path) -> None:
    entity_ids = [*EXCLUDED_ENTITY_IDS, "machine-2-1"]
    _write_entity_files(tmp_path, entity_ids)

    assert discover_remaining_entities(tmp_path) == ("machine-2-1",)


def test_discovery_accepts_only_requested_valid_entities(tmp_path: Path) -> None:
    entity_ids = [*EXCLUDED_ENTITY_IDS, "machine-2-1", "machine-2-2"]
    _write_entity_files(tmp_path, entity_ids)

    assert discover_remaining_entities(
        tmp_path, selected_entity_ids=("machine-2-2", "machine-2-1")
    ) == ("machine-2-1", "machine-2-2")


def test_discovery_rejects_excluded_selected_entity(tmp_path: Path) -> None:
    _write_entity_files(tmp_path, [*EXCLUDED_ENTITY_IDS, "machine-2-1"])

    with pytest.raises(ValueError, match="explicitly excluded"):
        discover_remaining_entities(tmp_path, selected_entity_ids=("machine-1-6",))


def test_mode_settings_keep_requested_thesis_epochs() -> None:
    assert mode_settings(smoke=True)["stage_a_epochs"] == 3
    assert mode_settings(smoke=True)["stage_b_epochs"] == 2
    assert mode_settings(smoke=False)["stage_a_epochs"] == 25
    assert mode_settings(smoke=False)["stage_b_epochs"] == 5
    assert mode_settings(smoke=True)["max_online_steps"] == 16
    assert mode_settings(smoke=False)["max_online_steps"] is None


def test_short_online_range_contains_the_densest_ground_truth_region(
    tmp_path: Path,
) -> None:
    label_path = tmp_path / "machine-2-1.txt"
    labels = [0] * 32
    labels[3:6] = [1, 1, 1]
    labels[20:22] = [1, 1]
    label_path.write_text("\n".join(map(str, labels)) + "\n", encoding="utf-8")

    assert select_short_online_range(label_path, length=8) == {
        "absolute_start_index": 0,
        "absolute_end_index": 8,
        "length": 8,
        "anomaly_points": 3,
    }


def test_matrix_plan_has_expected_method_counts() -> None:
    plan = build_matrix_plan(
        entity_ids=("machine-2-1",),
        seed_values=(6,),
        smoke=True,
        output_root=Path("outputs/benchmark_smoke/smd_remaining"),
    )
    counts: dict[tuple[str, str], int] = {}
    for run in plan:
        key = (run["phase"], run["method"])
        counts[key] = counts.get(key, 0) + 1

    assert counts["offline", "thesis"] == 2
    assert counts["offline", "redlamp_baseline"] == 1
    assert counts["offline", "iforest"] == 1
    assert counts["offline", "kmeans_ad"] == 1
    assert counts["offline", "stumpy_channel_ab"] == 1
    assert counts["online", "thesis"] == 6
    assert counts["online", "candi"] == 1
    assert counts["online", "m2n2"] == 1
    assert counts["online", "stumpy"] == 1
    assert counts["online", "kmeans_ad"] == 1
    assert counts["online", "iforest"] == 1


def test_main_method_plan_excludes_all_baselines() -> None:
    plan = build_matrix_plan(
        entity_ids=("machine-2-1",),
        seed_values=(6,),
        smoke=True,
        main_method_only=True,
        output_root=Path("outputs/benchmark_smoke/smd_main_method"),
    )

    assert [(run["phase"], run["variant"]) for run in plan] == [
        ("offline", "O0"),
        ("offline", "O1"),
        ("online", "O0-A0"),
        ("online", "O0-A1"),
        ("online", "O0-A2"),
        ("online", "O1-A0"),
        ("online", "O1-A1"),
        ("online", "O1-A2"),
    ]
    assert {run["method"] for run in plan} == {"thesis"}


def test_mode_settings_accept_custom_thesis_smoke_epochs() -> None:
    settings = mode_settings(
        smoke=True,
        stage_a_epochs=4,
        stage_b_epochs=2,
        max_online_steps=16,
    )

    assert settings["stage_a_epochs"] == 4
    assert settings["stage_b_epochs"] == 2
    assert settings["max_online_steps"] == 16


def test_metric_extraction_uses_exact_report_names() -> None:
    payload = {
        "offline_metrics": {
            "fpr": 0.012,
            "affiliation_f1": 0.3,
            "vus_pr": 0.4,
            "vus_roc": 0.5,
            "vus_pr_at_fpr_budget": {
                "0.001": 0.11,
                "0.005": 0.22,
                "0.01": 0.33,
            },
        }
    }

    assert extract_requested_metrics(payload) == {
        "VUS-PR@FPR-budget": {"0.1%": 0.11, "0.5%": 0.22, "1%": 0.33},
        "VUS-PR": 0.4,
        "Affiliation F1-score": 0.3,
        "VUS-ROC": 0.5,
        "raw-FPR": 0.012,
    }


def test_metric_extraction_reads_nested_online_final_metrics() -> None:
    payload = {
        "online_execution": {
            "final_metrics": {
                "online/step": 16,
                "fpr": 0.02,
                "affiliation_f1": 0.25,
                "vus_pr": 0.35,
                "vus_roc": 0.45,
                "vus_pr_at_fpr_budget": {
                    "0.001": 0.1,
                    "0.005": 0.2,
                    "0.01": 0.3,
                },
            }
        }
    }

    assert extract_requested_metrics(payload)["raw-FPR"] == 0.02


def test_metric_extraction_marks_incomplete_online_metrics_as_missing() -> None:
    payload = {
        "online_execution": {
            "final_metrics": {
                "online/step": 16,
                "vus_pr": 0.35,
                "vus_roc": 0.45,
            }
        }
    }

    assert extract_requested_metrics(payload) == {}


def test_final_online_metric_contract_contains_all_requested_metrics() -> None:
    metrics = compute_final_online_metrics(
        point_labels=[0, 1, 0, 1, 0, 0],
        point_scores=[0.1, 0.9, 0.2, 0.8, 0.05, 0.1],
        threshold=0.5,
        vus_max_buffer_size=2,
        vus_num_thresholds=5,
    )

    assert set(metrics) == {
        "vus_pr_at_fpr_budget",
        "vus_pr",
        "affiliation_f1",
        "vus_roc",
        "fpr",
    }
    assert set(metrics["vus_pr_at_fpr_budget"]) == {"0.001", "0.005", "0.01"}


def test_smoke_launcher_dry_run_mentions_two_gpus(tmp_path: Path) -> None:
    _write_entity_files(tmp_path, [*EXCLUDED_ENTITY_IDS, "machine-2-1"])
    script = Path("scripts/benchmarks/run_remaining_smd_smoke.sh")
    completed = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path),
            "--dry-run",
            "--no-tmux",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "GPU 0" in completed.stdout
    assert "GPU 1" in completed.stdout
    assert "VUS-PR@FPR-budget" in completed.stdout


def test_cloud_launcher_dry_run_lists_four_gpu_and_two_cpu_queues(tmp_path: Path) -> None:
    _write_entity_files(tmp_path, [*EXCLUDED_ENTITY_IDS, "machine-2-1"])
    script = Path("scripts/benchmarks/run_remaining_smd_cloud_tmux.sh")
    completed = subprocess.run(
        [
            "bash",
            str(script),
            "--mode",
            "smoke",
            "--gpu-count",
            "4",
            "--entity-id",
            "machine-2-1",
            "--dataset-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "outputs"),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    for queue in (
        "offline-gpu-0",
        "offline-gpu-1",
        "offline-gpu-2",
        "offline-gpu-3",
        "offline-cpu-0",
        "offline-cpu-1",
    ):
        assert queue in completed.stdout


def test_cloud_launcher_main_method_dry_run_lists_only_gpu_queues(tmp_path: Path) -> None:
    _write_entity_files(tmp_path, [*EXCLUDED_ENTITY_IDS, "machine-2-1"])
    script = Path("scripts/benchmarks/run_remaining_smd_cloud_tmux.sh")
    completed = subprocess.run(
        [
            "bash",
            str(script),
            "--mode",
            "smoke",
            "--gpu-count",
            "4",
            "--main-method-only",
            "--gpu-only",
            "--stage-a-epochs",
            "4",
            "--stage-b-epochs",
            "2",
            "--max-online-steps",
            "16",
            "--entity-id",
            "machine-2-1",
            "--dataset-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "outputs"),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"runs": 24' in completed.stdout
    for queue in ("offline-gpu-0", "offline-gpu-1", "offline-gpu-2", "offline-gpu-3"):
        assert f"smd-{queue}" in completed.stdout
    for queue in ("online-gpu-0", "online-gpu-1", "online-gpu-2", "online-gpu-3"):
        assert f"smd-{queue}" in completed.stdout
    assert "cpu-0" not in completed.stdout
    assert "cpu-1" not in completed.stdout


def test_cloud_resource_orchestration_uses_runtime_allowed_cpu_ids() -> None:
    launcher = Path("scripts/benchmarks/run_remaining_smd_cloud_tmux.sh").read_text(
        encoding="utf-8"
    )
    resource_env = Path(
        "scripts/benchmarks/_remaining_smd_resource_env.sh"
    ).read_text(encoding="utf-8")

    assert "Cpus_allowed_list" in launcher
    assert 'rm -f "$marker"' in launcher
    assert 'taskset -c "$cpu_mask" true' in resource_env


def test_config_builder_rejects_disabled_wandb_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        generator,
        "_common_logging",
        lambda **_: {
            "use_wandb": False,
            "wandb_project": "bachelor-thesis-2026",
            "wandb_mode": "disabled",
        },
    )
    run = {
        "runner": "thesis_offline",
        "run_id": "off-thesis-machine-2-1-s6",
        "variant": "O0",
        "entity_id": "machine-2-1",
        "seed": 6,
        "output_dir": "outputs/test",
    }

    with pytest.raises(ValueError, match="W&B logging must be enabled"):
        generator._build_config(
            run,
            Path("configs/data/smd_benchmark_machine_2_1_window20.yaml"),
            generator.mode_settings(smoke=True),
        )


def test_manifest_records_resource_and_dependency_metadata(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_entity_files(dataset_root, [*EXCLUDED_ENTITY_IDS, "machine-2-1"])
    labels = [0] * 2048
    labels[10] = 1
    (dataset_root / "test_label" / "machine-2-1.txt").write_text(
        "\n".join(map(str, labels)) + "\n", encoding="utf-8"
    )

    manifest_path = write_matrix_configs(
        dataset_root=dataset_root,
        output_root=tmp_path / "outputs",
        smoke=True,
        seed_values=(6,),
        selected_entity_ids=("machine-2-1",),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    thesis_online = next(
        run for run in manifest["runs"] if run["runner"] == "thesis_online"
    )
    candi = next(
        run for run in manifest["runs"] if run["method"] == "candi"
    )
    iforest = next(
        run for run in manifest["runs"] if run["method"] == "iforest" and run["phase"] == "offline"
    )

    assert thesis_online["resource_class"] == "gpu"
    assert thesis_online["phase_group"] == "online"
    assert set(thesis_online["dependencies"]) == {
        "stage_b_checkpoint",
        "threshold_artifact",
    }
    assert candi["resource_class"] == "gpu"
    assert set(candi["dependencies"]) == {"redlamp_checkpoint"}
    assert iforest["resource_class"] == "cpu"
    assert iforest["dependencies"] == {}


def test_generated_configs_use_resource_specific_worker_limits(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_entity_files(dataset_root, [*EXCLUDED_ENTITY_IDS, "machine-2-1"])
    labels = [0] * 2048
    labels[10] = 1
    (dataset_root / "test_label" / "machine-2-1.txt").write_text(
        "\n".join(map(str, labels)) + "\n", encoding="utf-8"
    )

    manifest_path = write_matrix_configs(
        dataset_root=dataset_root,
        output_root=tmp_path / "outputs",
        smoke=True,
        seed_values=(6,),
        selected_entity_ids=("machine-2-1",),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configs = {
        (run["runner"], run["method"]): yaml.safe_load(
            Path(run["config_path"]).read_text(encoding="utf-8")
        )
        for run in manifest["runs"]
    }

    assert configs[("thesis_offline", "thesis")]["data_config_path"]
    data_config = yaml.safe_load(
        next(
            path for path in (tmp_path / "outputs" / "generated_configs" / "data").glob("*.yaml")
        ).read_text(encoding="utf-8")
    )
    assert data_config["num_workers"] == 4
    assert configs[("redlamp", "redlamp_baseline")]["data_overrides"]["num_workers"] == 4
    assert configs[("thesis_online", "thesis")]["data_overrides"]["num_workers"] == 2
    assert configs[("offline_baseline", "iforest")]["data_overrides"]["num_workers"] == 0
    assert configs[("online_baseline", "candi")]["device"] == "cuda"
    assert configs[("online_baseline", "iforest")]["device"] == "cpu"


def test_all_remaining_smd_configs_enable_online_wandb_for_smoke_and_wet(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    _write_entity_files(dataset_root, [*EXCLUDED_ENTITY_IDS, "machine-2-1"])
    labels = [0] * 2048
    labels[10] = 1
    (dataset_root / "test_label" / "machine-2-1.txt").write_text(
        "\n".join(map(str, labels)) + "\n", encoding="utf-8"
    )

    for smoke in (True, False):
        manifest_path = write_matrix_configs(
            dataset_root=dataset_root,
            output_root=tmp_path / ("smoke" if smoke else "wet"),
            smoke=smoke,
            seed_values=(6,),
            selected_entity_ids=("machine-2-1",),
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        for run in manifest["runs"]:
            config = yaml.safe_load(
                Path(run["config_path"]).read_text(encoding="utf-8")
            )
            assert config["logging"]["use_wandb"] is True
            assert config["logging"]["wandb_mode"] == "online"
            if smoke:
                assert is_valid_wandb_smoke_run_name(
                    config["logging"]["wandb_run_name"]
                )


def test_online_thesis_config_points_to_matching_stage_b_output(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_entity_files(dataset_root, [*EXCLUDED_ENTITY_IDS, "machine-2-1"])
    labels = [0] * 2048
    labels[10] = 1
    (dataset_root / "test_label" / "machine-2-1.txt").write_text(
        "\n".join(map(str, labels)) + "\n", encoding="utf-8"
    )
    manifest_path = write_matrix_configs(
        dataset_root=dataset_root,
        output_root=tmp_path / "outputs",
        smoke=True,
        seed_values=(6,),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = next(
        item
        for item in manifest["runs"]
        if item["runner"] == "thesis_online" and item["variant"] == "O0-A0"
    )
    config_text = Path(run["config_path"]).read_text(encoding="utf-8")
    expected = "machine_2_1/seed6/thesis/O0/offline/two_stage"
    assert expected in config_text
    assert run["report_path"].endswith("thesis_online_A0_benchmark_report.json")
    assert run["online_range"]["length"] == 2048
    assert "absolute_start_index: 0" in config_text
    assert "absolute_end_index: 2048" in config_text


def test_online_baseline_configs_share_range_and_device_policy(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_entity_files(dataset_root, [*EXCLUDED_ENTITY_IDS, "machine-2-1"])
    labels = [0] * 2048
    labels[10] = 1
    (dataset_root / "test_label" / "machine-2-1.txt").write_text(
        "\n".join(map(str, labels)) + "\n", encoding="utf-8"
    )
    manifest_path = write_matrix_configs(
        dataset_root=dataset_root,
        output_root=tmp_path / "outputs",
        smoke=True,
        seed_values=(6,),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = {
        item["method"]: item
        for item in manifest["runs"]
        if item["runner"] == "online_baseline"
    }
    assert runs["candi"]["online_range"] == runs["iforest"]["online_range"]
    assert "device: cuda" in Path(runs["candi"]["config_path"]).read_text()
    assert "device: cpu" in Path(runs["iforest"]["config_path"]).read_text()
