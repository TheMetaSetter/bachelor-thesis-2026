"""Check prepared data, method updates, calibration, and run results."""


from math import isfinite

import numpy as np


def _check_identity(identity, config):
    for key in ("dataset", "entity"):
        if identity.get(key) != config.get(key):
            raise ValueError(f"data: {key} mismatch")
    if identity.get("split") not in {"train", "validation", "test"}:
        raise ValueError("data: unknown split")


def validate_data_input(data: dict, config: dict) -> None:
    """Check an entity or a prepared split."""
    if "splits" in data:
        if set(data["splits"]) != {"train", "validation", "test"}:
            raise ValueError("data: expected train, validation, and test")
        for name, split in data["splits"].items():
            if split["identity"]["split"] != name or split["x"].shape[1] != len(data["features"]):
                raise ValueError(f"data: {name} identity or feature count mismatch")
            validate_data_input(split, config)
        return
    _check_identity(data.get("identity", {}), config)
    x = np.asarray(data.get("x"))
    if x.ndim != 2 or not all(x.shape) or not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
        raise ValueError("data: expected a nonempty matrix of finite features")
    for field in ("labels", "source_row", "source_file"):
        if np.shape(data.get(field)) != (len(x),):
            raise ValueError(f"data: {field} does not align with features")
    if "timestamps" in data and np.shape(data["timestamps"]) != (len(x),):
        raise ValueError("data: timestamps do not align with features")


def validate_data_output(batch: dict, config: dict) -> None:
    """Check batch shape, point positions, and optional aligned values."""
    _check_identity(batch.get("identity", {}), config)
    x, indices = np.asarray(batch.get("x")), np.asarray(batch.get("indices"))
    if x.ndim != 3 or not all(x.shape) or x.shape[1] != config["data"]["window_size"]:
        raise ValueError("batch: invalid window shape")
    if (indices.shape != x.shape[:2] or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(np.diff(indices, axis=1) != 1)):
        raise ValueError("batch: invalid point indices")
    if np.shape(batch.get("segment_ids")) != (len(x),):
        raise ValueError("batch: segment IDs do not align with windows")
    if config.get("phase") == "online" and len(x) != 1:
        raise ValueError("batch: online execution requires one window")
    for key in ("labels", "timestamps"):
        if key in batch and np.shape(batch[key]) != indices.shape:
            raise ValueError(f"batch: {key} does not align with indices")


def validate_update_input(method, batch: dict, context: dict) -> None:
    """Check that an enabled update uses one arrived unlabeled window."""
    if "labels" in batch:
        raise ValueError("update: labels must be absent")
    if not context.get("allow_update", False):
        return
    if not callable(getattr(method, "update", None)) or getattr(method, "optimizer", None) is None:
        raise ValueError("update: enabled method needs an update function and optimizer")
    x = np.asarray(batch.get("x"))
    if x.ndim != 3 or len(x) != 1 or not np.isfinite(x).all():
        raise ValueError("update: expected one finite window")
    score = context.get("recorded_score")
    if score is None or not np.isfinite(score).all():
        raise ValueError("update: a finite pre-update score is required")
    if "arrived_index" in context and np.max(batch["indices"]) > context["arrived_index"]:
        raise ValueError("update: observations must have arrived")


def validate_update_output(result: dict, context: dict) -> None:
    """Check update outcome, loss, and allowed parameter changes."""
    if type(result.get("did_update")) is not bool:
        raise ValueError("update: did_update must be a boolean")
    loss = result.get("loss")
    if loss is not None and (type(loss) not in (int, float) or not isfinite(loss)):
        raise ValueError("update: loss must be finite")
    changed = set(result.get("changed_parameters", ()))
    if not changed <= set(context.get("allowed_parameters", ())):
        raise ValueError("update: changed a parameter outside the allowed set")
    if (result["did_update"] and not context.get("allow_update")) or (changed and not result["did_update"]):
        raise ValueError("update: changes do not match the recorded decision")


_PARAMETER_FIELDS = {
    "mustard_offline": "classifier_hidden_size head_dropout continuous discrete memory loss_weights "
                       "label_correction contrastive_temperature stochastic_inference monte_carlo_samples variance_correction",
    "mustard_online": "projector_hidden_size projector_dropout verification contrastive_weight contrastive_temperature",
    "redlamp": "classifier_hidden_size classifier_dropout label_correction classification_weight "
               "class_selection_probability zero_range_epsilon",
    "stumpy_channel_ab": "normalize p channel_aggregation calibration_epsilon",
    "kmeans_ad": "n_clusters n_init normalize_windows normalization_epsilon",
    "iforest": "n_estimators max_samples max_features contamination normalize_windows normalization_epsilon",
    "candi": "latent_pooling latent_normalization_epsilon covariance_rcond similarity_quantile minimum_samples sana",
    "m2n2": "trend_decay",
    "moment": "checkpoint revision channel_aggregation patch_length",
}
_NESTED_FIELDS = {
    "continuous": "enabled num_prototypes temperature",
    "discrete": "enabled codebook_size top_k temperature",
    "memory": "normalization_epsilon initialization_batches freeze_after_initialization",
    "loss_weights": "reconstruction classification contrastive diversity variance covariance usage gate",
    "label_correction": "alpha beta",
    "verification": "buffer_size cycle_size non_overlap_gap signature_top_k",
    "sana": "hidden_size attention_heads feedforward_size layers dropout gate_initialization",
}
_BOOL = set("enabled stochastic_inference freeze_after_initialization normalize normalize_windows".split())
_INTEGER = set("classifier_hidden_size projector_hidden_size monte_carlo_samples num_prototypes codebook_size top_k "
               "initialization_batches buffer_size cycle_size signature_top_k non_overlap_gap hidden_size "
               "attention_heads feedforward_size layers n_clusters n_init n_estimators minimum_samples patch_length "
               "variance_correction".split())
_FIXED = {"channel_aggregation": {"moment": "mean", "stumpy_channel_ab": "max"}, "latent_pooling": "mean_time"}


def _require_fields(value, fields, context, *, optional=()):
    required = set(fields.split())
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - set(optional):
        raise ValueError(f"{context}: expected fields {', '.join(sorted(required))}")


def _value(value, field, *, minimum=0, maximum=float("inf"), allow_minimum=False):
    if type(value) not in (int, float) or not isfinite(value) or value > maximum or (
        value < minimum if allow_minimum else value <= minimum
    ):
        raise ValueError(f"{field}: invalid number")


def _count(value, field, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field}: expected an integer >= {minimum}")


def _optimizer(values, expected):
    if not isinstance(values, dict) or values.get("name") != expected:
        raise ValueError(f"optimizer: this method requires {expected}")
    extra = "momentum dampening nesterov" if expected == "sgd" else "betas epsilon"
    _require_fields(values, "name learning_rate weight_decay " + extra, "optimizer")
    _value(values["learning_rate"], "learning_rate")
    _value(values["weight_decay"], "weight_decay", allow_minimum=True)
    if expected == "sgd":
        momentum, dampening, nesterov = (values[key] for key in extra.split())
        _value(momentum, "momentum", allow_minimum=True, maximum=1)
        _value(dampening, "dampening", allow_minimum=True)
        if type(nesterov) is not bool or momentum == 1 or (nesterov and (momentum == 0 or dampening != 0)):
            raise ValueError("optimizer: invalid SGD momentum, dampening, or Nesterov")
    else:
        betas = values["betas"]
        if not isinstance(betas, list) or len(betas) != 2:
            raise ValueError("optimizer: supply two betas")
        for beta in betas:
            _value(beta, "betas", allow_minimum=True, maximum=1)
            if beta == 1:
                raise ValueError("optimizer: betas must be less than one")
        _value(values["epsilon"], "epsilon")


def _training(profile, expected):
    stages = profile.get("training")
    if not expected:
        if stages is not None:
            raise ValueError("training: this profile has no epoch-based fit")
        return
    _require_fields(stages, "stages", "training")
    stages = stages["stages"]
    if not isinstance(stages, list) or len(stages) != len(expected):
        raise ValueError("training: incorrect number of method stages")
    name = profile["method"]
    for stage, role in zip(stages, expected, strict=True):
        _require_fields(stage, "name epochs optimizer monitor gradient_clip_norm", "training stage", optional=("scheduler",))
        monitor = "validation_total_loss" if name in {"mustard", "redlamp"} else "validation_reconstruction_loss"
        if stage["name"] != role or stage["monitor"] != monitor:
            raise ValueError(f"training: expected {role} and {monitor}")
        _count(stage["epochs"], "epochs")
        _optimizer(stage["optimizer"], "adamw" if name == "m2n2" else "adam")
        if stage["gradient_clip_norm"] is not None:
            _value(stage["gradient_clip_norm"], "gradient_clip_norm")
        schedule = stage.setdefault("scheduler", None)
        if schedule is not None:
            _require_fields(schedule, "name warmup_epochs minimum_learning_rate interval", "scheduler")
            _count(schedule["warmup_epochs"], "warmup_epochs", 0)
            _value(schedule["minimum_learning_rate"], "minimum_learning_rate", allow_minimum=True)
            if name != "candi" or schedule["name"] != "cosine" or schedule["interval"] != "epoch":
                raise ValueError("scheduler: the selected method does not support this schedule")
            if schedule["warmup_epochs"] >= stage["epochs"]:
                raise ValueError("scheduler: warmup must end before the stage ends")


def _parameter_value(name, value, method):
    if name in _BOOL:
        if type(value) is not bool:
            raise ValueError(f"{name}: expected a boolean")
    elif name in _INTEGER:
        _count(value, name, 0 if name in {"non_overlap_gap", "variance_correction"} else 1)
        if name == "variance_correction" and value > 1:
            raise ValueError("variance_correction: expected 0 or 1")
    elif name in _FIXED:
        expected = _FIXED[name]
        if value != (expected[method] if isinstance(expected, dict) else expected):
            raise ValueError(f"{name}: unsupported value")
    elif name in {"checkpoint", "revision"}:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name}: expected text")
    elif name == "p":
        _value(value, name, minimum=1, allow_minimum=True)
    elif name == "gate_initialization":
        _value(value, name, minimum=-float("inf"))
    elif name in {"max_samples", "max_features", "contamination", "classification_weight",
                  "class_selection_probability", "similarity_quantile"} or name.endswith("dropout") or name == "trend_decay":
        if name in {"max_samples", "contamination"} and value == "auto":
            return
        if name == "max_samples" and type(value) is int:
            _count(value, name)
            return
        _value(value, name, maximum=0.5 if name == "contamination" else 1,
               allow_minimum=name not in {"max_samples", "max_features", "contamination", "similarity_quantile"})
        if value == 1 and (name.endswith("dropout") or name in {"trend_decay", "similarity_quantile"}):
            raise ValueError(f"{name}: must be less than one")
    else:
        _value(value, name, allow_minimum=name in {"alpha", "beta", "covariance_rcond", "contrastive_weight"})


def _parameters(profile, kind):
    params = profile["parameters"]
    name, variant = profile["method"], profile["variant"]
    _require_fields(params, _PARAMETER_FIELDS[kind], f"{name} parameters")
    for key, value in params.items():
        if key not in _NESTED_FIELDS:
            _parameter_value(key, value, name)
            continue
        if key == "verification" and variant == "A0" and value is None:
            continue
        _require_fields(value, _NESTED_FIELDS[key], key)
        for field, setting in value.items():
            if key == "loss_weights":
                _value(setting, field, allow_minimum=True)
                permitted = {"reconstruction", "classification", "contrastive"} if variant == "O2" else {
                    "reconstruction", "classification"
                }
                if field not in permitted and setting != 0:
                    raise ValueError(f"{variant}: unsupported loss weight {field}")
            else:
                _parameter_value(field, setting, name)
    if kind == "mustard_offline" and (
        not params["memory"]["freeze_after_initialization"] or params["discrete"]["top_k"] > params["discrete"]["codebook_size"]
    ):
        raise ValueError("MUSTARD requires fixed initialized memory and valid discrete top-k")
    if kind == "mustard_online":
        verification = params["verification"]
        if (variant == "A0") != (verification is None) or (
            verification is not None and verification["cycle_size"] > verification["buffer_size"]
        ):
            raise ValueError("invalid MUSTARD verification settings")
        if variant != "A2" and params["contrastive_weight"] != 0:
            raise ValueError("contrastive updates require MUSTARD A2")
    if "label_correction" in params:
        alpha, beta = (params["label_correction"][key] for key in ("alpha", "beta"))
        if alpha + 12 * beta > 1:
            raise ValueError("invalid 12-class label correction")
    if "sana" in params and params["sana"]["hidden_size"] % params["sana"]["attention_heads"]:
        raise ValueError("SANA hidden size must be divisible by its head count")


def validate_method_profile(profile):
    """Reject settings that the selected method cannot apply."""
    _require_fields(profile, "method variant phases parameters", "method profile",
                    optional=("training", "update", "required_offline_variant"))
    name, variant, phases = (profile[key] for key in ("method", "variant", "phases"))
    if not isinstance(name, str) or not isinstance(variant, str):
        raise ValueError("method and variant must be names")
    if not isinstance(phases, list) or not phases or any(phase not in ("offline", "online") for phase in phases):
        raise ValueError("method: expected a nonempty list of supported phases")
    if len(set(phases)) != len(phases):
        raise ValueError("method: duplicate phase")
    if name == "mustard":
        if variant in ("O0", "O1", "O2") and phases == ["offline"]:
            kind, stages = "mustard_offline", ("stage_a", "stage_b")
        elif variant in ("A0", "A1", "A2") and phases == ["online"]:
            kind, stages = "mustard_online", ()
        else:
            raise ValueError("MUSTARD variant and phase do not match")
    elif name in _PARAMETER_FIELDS and not name.startswith("mustard_"):
        supported = ("zero_shot", "linear_probing") if name == "moment" else ("default",)
        if variant not in supported:
            raise ValueError(f"{name}: unsupported variant")
        kind = name
        fits = name in {"redlamp", "candi", "m2n2"} or name == "moment" and variant == "linear_probing"
        stages = ("fit",) if "offline" in phases and fits else ()
    else:
        raise ValueError(f"unsupported method: {name!r}")
    _parameters(profile, kind)
    _training(profile, stages)
    update = profile.get("update")
    adapts = variant in ("A1", "A2") or name in {"m2n2", "candi"} and "online" in phases
    if adapts:
        _require_fields(update, "steps optimizer gradient_clip_norm", "method update")
        _count(update["steps"], "update steps")
        _optimizer(update["optimizer"], "adam" if name == "mustard" else "sgd")
        if update["gradient_clip_norm"] is not None:
            _value(update["gradient_clip_norm"], "gradient_clip_norm")
    elif update is not None:
        raise ValueError("update: selected variant does not adapt online")
    source = profile.get("required_offline_variant")
    expected = ("O0", "O1", "O2") if name == "mustard" else (variant,)
    needs_parent = "online" in phases and not (name == "moment" and variant == "zero_shot")
    if (needs_parent and source not in expected) or (not needs_parent and source is not None):
        raise ValueError("method profile: incompatible required_offline_variant")


IDENTITY_KEYS = ("dataset", "entity", "method_name", "variant", "phase", "seed")
_STATUSES = {"running", "blocked", "failed", "interrupted", "successful"}


def validate_run_input(config: dict) -> None:
    """Check the run or the run and its aligned evaluation inputs."""
    missing = set(IDENTITY_KEYS) - config.keys()
    if missing:
        raise ValueError(f"run: missing identity fields {sorted(missing)}")
    if config["phase"] not in {"offline", "online"} or config["phase"] not in config["method"]["phases"]:
        raise ValueError(f"run: unsupported phase {config['phase']}")
    identity = config["prepared_identity"]
    for key in ("dataset", "entity"):
        if identity.get(key) != config[key]:
            raise ValueError(f"run: prepared {key} mismatch")
    kind = config.get("kind", "resolved_run")
    if kind == "evaluation":
        scores, labels = np.asarray(config.get("scores")), np.asarray(config.get("labels"))
        if scores.ndim != 1 or scores.shape != labels.shape:
            raise ValueError("evaluation: scores and labels must align")
    elif kind != "resolved_run":
        raise ValueError(f"run: unsupported record kind {kind!r}")


def validate_run_output(result: dict, config: dict) -> None:
    """Check result identity, status, and any defined scalar metrics."""
    for key in IDENTITY_KEYS:
        if result.get(key) != config[key]:
            raise ValueError(f"result: {key} does not match the run")
    if result.get("status") not in _STATUSES:
        raise ValueError("result: unsupported status")
    for name, value in result.get("metrics", {}).items():
        if value is not None and (type(value) not in (int, float) or not isfinite(value)):
            raise ValueError(f"result: {name} must be finite or undefined")


_CALIBRATION_SPLITS = {"train", "validation", "synthetic_validation", "synthetic_validation_normal"}


def _calibration_source(values):
    if values.get("source_split") not in _CALIBRATION_SPLITS:
        raise ValueError("calibration: invalid source split")


def _check_scores(values):
    kind = values.get("score_kind")
    dimensions = {"point": 1, "window": 2}
    scores = np.asarray(values.get("raw_scores"))
    indices = np.asarray(values.get("indices"))
    if kind not in dimensions or scores.ndim != dimensions[kind] or scores.shape != indices.shape:
        raise ValueError("scores: invalid score and index shapes")
    if not np.issubdtype(scores.dtype, np.number) or not np.isfinite(scores).all():
        raise ValueError("scores: expected finite values")
    if not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0):
        raise ValueError("scores: expected nonnegative integer indices")


def validate_calibration_input(values: dict, config: dict) -> None:
    """Check an input score record or the source for threshold fitting."""
    if values.get("kind") == "calibration":
        _calibration_source(values)
        return
    _check_scores(values)


def validate_calibration_output(values: dict, config: dict) -> None:
    """Check generated scores or thresholds before use."""
    if values.get("kind") != "calibration":
        _check_scores(values)
        return
    _calibration_source(values)
    if values.get("score_space") != config["evaluation"]["score_space"]:
        raise ValueError("calibration: score space mismatch")
    if not values.get("checkpoint_sha256"):
        raise ValueError("calibration: missing checkpoint identity")
    for key in ("point_threshold", "window_threshold", "latent_low", "latent_high"):
        if key == "point_threshold" or key in values:
            value = values.get(key)
            if type(value) not in (int, float) or not isfinite(value):
                raise ValueError(f"calibration: {key} must be finite")
    latent_keys = {"latent_low", "latent_high"}
    if values.keys() & latent_keys:
        if not latent_keys <= values.keys() or values["latent_low"] > values["latent_high"]:
            raise ValueError("calibration: invalid latent threshold order")
