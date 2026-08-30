from __future__ import annotations

from scripts.run_direct_branch_routing_full import (
    EXPERIMENT_CONFIGS,
    build_run_configs,
)


def test_full_runner_builds_three_gpu_direct_routing_configs() -> None:
    configs = build_run_configs()

    assert configs == list(EXPERIMENT_CONFIGS)
    assert len(configs) == 3
    assert [path.name for path in configs] == [
        "smd__thesis__offline__direct_branch_routing__machine_1_6__w20__seed6__stage_b.yaml",
        "smd__thesis__offline__direct_branch_routing__machine_3_4__w20__seed6__stage_b.yaml",
        "smd__thesis__offline__direct_branch_routing__machine_3_9__w20__seed6__stage_b.yaml",
    ]
