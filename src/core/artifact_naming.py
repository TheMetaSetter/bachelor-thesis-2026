from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Sequence


WANDB_ARTIFACT_NAME_LIMIT = 128

_ALLOWED_ROLES = {
    "cfg",
    "met",
    "ckpt",
    "eval",
    "records",
    "curves",
    "traces",
    "audit",
    "on-met",
    "on-rec",
    "thr",
    "rpt",
    "bnd",
    "abl",
    "out",
    "run",
}

_ALLOWED_SMOKE_PHASE_TOKENS = {"off", "on"}
_SMOKE_RUN_NAME_PATTERN = re.compile(
    r"smk-(?:off|on)-[A-Za-z0-9_.-]+-e[A-Za-z0-9_.-]+-s-?\d+"
)
_WANDB_METHOD_DISPLAY_TOKENS = {"kmeans_ad": "KA"}


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is not None and str(value).strip():
            return value
    return None


def _required_text(field_name: str, value: Any) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"artifact identity requires {field_name}")
    return str(value).strip()


def _required_seed(value: Any) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError("artifact identity requires seed")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("artifact identity seed must be an integer") from exc


def _compact_stage(stage: str) -> str:
    normalized = stage.strip()
    lowered = normalized.lower().replace("-", "_")
    if lowered in {"stagea", "stage_a", "stage_a_multitask_pretraining"}:
        return "stageA"
    if lowered in {"stageb", "stage_b", "stage_b_fusion_finetuning"}:
        return "stageB"
    if lowered in {"online", "online_adaptation", "online_tta"}:
        return "online"
    if lowered in {"evaluation", "evaluate"}:
        return "eval"
    return normalized


def _token_from_experiment_name(experiment_name: Any, pattern: str) -> str | None:
    if experiment_name is None:
        return None
    match = re.search(pattern, str(experiment_name), flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def build_artifact_identity(
    experiment_config: Mapping[str, Any],
    *,
    stage: str | None = None,
    online_variant: str | None = None,
) -> dict[str, str | int]:
    """Extract the smallest stable identity shared by artifact producers."""
    if not isinstance(experiment_config, Mapping):
        raise ValueError("experiment_config must be a mapping")

    data_config = experiment_config.get("data")
    data_config = data_config if isinstance(data_config, Mapping) else {}
    task_config = experiment_config.get("task")
    task_config = task_config if isinstance(task_config, Mapping) else {}
    evaluation_config = experiment_config.get("evaluation")
    evaluation_config = (
        evaluation_config if isinstance(evaluation_config, Mapping) else {}
    )

    entity_ids = data_config.get("entity_ids")
    entity_from_list = None
    if isinstance(entity_ids, list) and len(entity_ids) == 1:
        entity_from_list = entity_ids[0]

    resolved_stage = _first_non_empty(
        stage,
        experiment_config.get("stage_name"),
        task_config.get("stage_name"),
        experiment_config.get("model", {}).get("stage_name")
        if isinstance(experiment_config.get("model"), Mapping)
        else None,
    )
    resolved_online_variant = _first_non_empty(
        online_variant,
        experiment_config.get("online_variant"),
        task_config.get("online_variant"),
        _token_from_experiment_name(
            experiment_config.get("experiment_name"), r"(?:^|[_-])((?:A[0-2])|main)(?:[_-]|$)"
        ),
    )
    resolved_variant = _first_non_empty(
        experiment_config.get("offline_variant"),
        task_config.get("offline_variant"),
        experiment_config.get("variant"),
        task_config.get("variant"),
        _token_from_experiment_name(
            experiment_config.get("experiment_name"), r"(?:^|[_-])(O[01])(?:[_-]|$)"
        ),
        experiment_config.get("experiment_variant"),
    )
    model_config = experiment_config.get("model")
    if resolved_variant is None and (
        task_config.get("task_name") == "online_adaptation"
        or isinstance(model_config, Mapping)
        and model_config.get("model_name") == "online_adaptation"
    ):
        resolved_variant = "base"
    resolved_fpr_budget = _first_non_empty(
        experiment_config.get("fpr_budget"),
        task_config.get("fpr_budget"),
        evaluation_config.get("fpr_budget"),
        _token_from_experiment_name(
            experiment_config.get("experiment_variant"),
            r"(?:^|[_-])(fpr[0-9]+)(?:[_-]|$)",
        ),
    )

    identity: dict[str, str | int] = {
        "dataset": _required_text(
            "dataset", _first_non_empty(experiment_config.get("dataset"), data_config.get("dataset_name"))
        ),
        "variant": _required_text("variant", resolved_variant),
        "entity": _required_text(
            "entity",
            _first_non_empty(
                experiment_config.get("entity_id"),
                task_config.get("entity_id"),
                entity_from_list,
            ),
        ),
        "seed": _required_seed(
            _first_non_empty(experiment_config.get("seed"), task_config.get("seed"))
        ),
    }
    if resolved_stage is not None:
        identity["stage"] = _compact_stage(str(resolved_stage))
    if resolved_online_variant is not None:
        identity["online_variant"] = str(resolved_online_variant).strip()
    if resolved_fpr_budget is not None:
        identity["fpr_budget"] = str(resolved_fpr_budget).strip()
    return identity


def _identity_text(identity: Mapping[str, str | int], field_name: str) -> str:
    value = identity.get(field_name)
    if value is None or not str(value).strip():
        raise ValueError(f"artifact identity requires {field_name}")
    return str(value).strip()


def _budget_token(value: str) -> str:
    if value.lower().startswith("fpr"):
        return value.lower()
    compact = value.replace(".", "").replace("%", "")
    return f"b{compact}"


def build_wandb_artifact_name(
    *,
    role: str,
    identity: Mapping[str, str | int],
) -> str:
    """Build and validate a short, role-first W&B artifact name."""
    if role not in _ALLOWED_ROLES:
        raise ValueError(f"unsupported W&B artifact role: {role}")

    variant = _identity_text(identity, "variant")
    entity = _identity_text(identity, "entity")
    seed = _identity_text(identity, "seed")
    stage_or_variant = _identity_text(identity, "stage") if identity.get("stage") else variant

    parts = [role, stage_or_variant]
    if identity.get("online_variant"):
        parts.append(_identity_text(identity, "online_variant"))
    parts.extend([variant, entity, f"s{seed}"])
    if identity.get("fpr_budget"):
        parts.append(_budget_token(_identity_text(identity, "fpr_budget")))

    return validate_wandb_artifact_name("-".join(parts))


def build_wandb_run_name(
    experiment_config: Mapping[str, Any],
    *,
    stage: str | None = None,
    online_variant: str | None = None,
) -> str:
    """Build the same compact identity vocabulary for a W&B run name."""
    identity = build_artifact_identity(
        experiment_config,
        stage=stage,
        online_variant=online_variant,
    )
    return build_wandb_artifact_name(role="run", identity=identity)


def wandb_entity_token(entity_id: Any) -> str:
    """Return the compact entity token used by smoke W&B display names."""
    normalized = _required_text("entity", entity_id).replace("-", "_")
    if normalized.startswith("machine_"):
        return f"e{normalized.removeprefix('machine_')}"
    return normalized


def wandb_method_display_token(method: Any) -> str:
    """Return a selected short method token without changing method identity."""
    normalized = _required_text("method", method)
    return _WANDB_METHOD_DISPLAY_TOKENS.get(normalized, normalized)


def build_wandb_smoke_run_name(
    *,
    phase_token: str,
    identity_tokens: Sequence[str],
    entity_token: str,
    seed: Any,
) -> str:
    """Build the selected short W&B display name for a smoke run."""
    if phase_token not in _ALLOWED_SMOKE_PHASE_TOKENS:
        raise ValueError(f"unsupported smoke phase token: {phase_token!r}")
    if isinstance(identity_tokens, str) or not identity_tokens:
        raise ValueError("smoke run name requires at least one identity token")
    normalized_identity_tokens = [
        _required_text("identity token", token) for token in identity_tokens
    ]
    normalized_entity_token = _required_text("entity", entity_token)
    normalized_seed = _required_seed(seed)
    return validate_wandb_artifact_name(
        "-".join(
            [
                "smk",
                phase_token,
                *normalized_identity_tokens,
                normalized_entity_token,
                f"s{normalized_seed}",
            ]
        )
    )


def is_valid_wandb_smoke_run_name(name: Any) -> bool:
    """Return whether a value follows the selected smoke W&B grammar."""
    if not isinstance(name, str):
        return False
    try:
        validated_name = validate_wandb_artifact_name(name)
    except ValueError:
        return False
    return _SMOKE_RUN_NAME_PATTERN.fullmatch(validated_name) is not None


def resolve_wandb_run_name(
    logging_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
    *,
    stage: str | None = None,
    online_variant: str | None = None,
) -> str:
    """Preserve a valid configured smoke name and retain legacy fallbacks."""
    configured_name = logging_config.get("wandb_run_name")
    if is_valid_wandb_smoke_run_name(configured_name):
        return str(configured_name).strip()
    return build_wandb_run_name(
        experiment_config,
        stage=stage,
        online_variant=online_variant,
    )


def validate_wandb_artifact_name(name: str) -> str:
    """Validate an already composed W&B artifact name without changing it."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("W&B artifact name must be a non-empty string")
    name = name.strip()
    if len(name) > WANDB_ARTIFACT_NAME_LIMIT:
        raise ValueError(
            "W&B artifact name is longer than "
            f"{WANDB_ARTIFACT_NAME_LIMIT} characters: {name!r}"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"W&B artifact name contains unsupported characters: {name!r}")
    return name
