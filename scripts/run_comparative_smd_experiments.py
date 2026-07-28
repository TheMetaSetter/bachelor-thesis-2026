from __future__ import annotations

import subprocess

from scripts.experiments import (
    run_comparative_smd_experiments as _implementation_module,
)

from scripts.experiments._internal.run_comparative_smd_experiments_support import (
    REPOSITORY_ROOT,
    SUPPORTED_BASELINE_MODEL_NAMES,
    _build_baseline_single_stage_commands,
    _build_run_record,
    _build_thesis_two_stage_commands,
    _load_run_records,
    _normalize_artifact_path,
    _utc_now_iso,
    _validate_single_entity_contract,
    load_experiment_config,
    normalize_config_path,
    parse_args,
    resolve_dataset_root,
    resolve_stage_family,
    validate_dataset_roots,
    validate_unique_artifact_paths,
)

build_comparative_run_plan = _implementation_module.build_comparative_run_plan
execute_comparative_run_plan = _implementation_module.execute_comparative_run_plan
main = _implementation_module.main

__all__ = [
    "build_comparative_run_plan",
    "execute_comparative_run_plan",
    "main",
]


if __name__ == "__main__":
    main()
