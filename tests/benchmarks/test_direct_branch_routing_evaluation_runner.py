from __future__ import annotations

from pathlib import Path


def test_evaluation_runner_uses_machine_3_4_three_seed_direct_checkpoints(
    monkeypatch,
) -> None:
    import scripts.run_direct_branch_routing_evaluation as runner

    captured: list[tuple[str, str, dict[str, object]]] = []

    def fake_build_config(config_path: Path) -> dict[str, object]:
        return {
            "experiment_name": config_path.stem,
            "output_dir": "outputs/direct",
            "logging": {"wandb_project": "bachelor-thesis-2026"},
        }

    def fake_evaluate(config: dict[str, object], checkpoint_path: str) -> None:
        captured.append(
            (str(config["experiment_name"]), checkpoint_path, dict(config["logging"]))
        )

    monkeypatch.setattr(runner, "build_direct_experiment_config", fake_build_config)
    monkeypatch.setattr(runner, "run_evaluation_experiment", fake_evaluate)
    monkeypatch.setattr(Path, "exists", lambda _: True)

    runner.main()

    assert len(captured) == 3
    assert ["seed6", "seed8", "seed36"] == [
        name.split("__")[-2] for name, _, _ in captured
    ]
    assert all("machine_3_4" in checkpoint for _, checkpoint, _ in captured)
    assert all(
        "thesis_direct_branch_routing_O0" in checkpoint for _, checkpoint, _ in captured
    )
    assert all(logging["use_wandb"] is True for _, _, logging in captured)
    assert all(logging["wandb_mode"] == "online" for _, _, logging in captured)
    assert all(logging["wandb_job_type"] == "evaluate" for _, _, logging in captured)
