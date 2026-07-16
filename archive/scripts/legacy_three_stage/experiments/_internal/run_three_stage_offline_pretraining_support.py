from scripts.experiments._internal.run_three_stage_offline_pretraining_shared import *  # noqa: F401,F403


def _build_stage_experiment_config(
    experiment_config: dict[str, Any],
    phase_record: dict[str, Any],
) -> dict[str, Any]:
    stage_config = copy.deepcopy(experiment_config)
    phase_name = str(phase_record["phase_name"])
    stage_experiment_name = _to_stage_experiment_name(
        str(experiment_config["experiment_name"]),
        phase_name,
    )
    stage_output_dir = Path(
        _to_stage_output_dir(str(experiment_config["output_dir"]), phase_name)
    )
    stage_checkpoint_dir = stage_output_dir / "checkpoints"
    stage_config["experiment_name"] = stage_experiment_name
    stage_config["output_dir"] = str(stage_output_dir)
    stage_config["checkpoint_dir"] = str(stage_checkpoint_dir)
    stage_config["epochs"] = int(phase_record["epochs"])
    for reference_field in [
        "data_config_path",
        "model_config_path",
        "task_config_path",
    ]:
        if reference_field in stage_config:
            stage_config[reference_field] = _resolve_repo_config_reference(
                str(stage_config[reference_field])
            )
    stage_config["three_stage_phase"] = phase_name
    stage_config["three_stage_global_epoch_start"] = int(
        phase_record["global_epoch_start"]
    )
    stage_config["three_stage_global_epoch_end"] = int(phase_record["global_epoch_end"])
    if (
        STAGE3_WARMUP_EPOCHS_CANONICAL_KEY in stage_config["three_stage"]
        and STAGE3_WARMUP_EPOCHS_LEGACY_KEY in stage_config["three_stage"]
    ):
        stage_config["three_stage"].pop(STAGE3_WARMUP_EPOCHS_LEGACY_KEY)
    stage_config["model"]["training_phase"] = phase_name
    model_overrides = copy.deepcopy(stage_config.get("model_overrides", {}))
    model_overrides["training_phase"] = phase_name
    model_overrides["freeze_memories_after_initialization"] = bool(
        stage_config["three_stage"]["freeze_memories_after_initialization"]
    )
    model_overrides["freeze_recovered_zipped_encoder_during_warmup"] = bool(
        stage_config["three_stage"]["freeze_recovered_zipped_encoder_during_warmup"]
    )
    model_overrides["discrete_memory_label_source"] = str(
        stage_config["three_stage"]["discrete_memory_label_source"]
    )
    stage_config["model_overrides"] = model_overrides
    initialization_checkpoint_path = _to_stage_initialization_checkpoint_path(
        str(experiment_config["output_dir"]),
        phase_name,
    )
    if initialization_checkpoint_path is not None:
        stage_config["initialization_checkpoint_path"] = initialization_checkpoint_path
    logging_config = copy.deepcopy(stage_config.get("logging", {}))
    logging_config["wandb_job_type"] = phase_name
    logging_config["wandb_run_name"] = stage_experiment_name
    stage_config["logging"] = logging_config
    return stage_config


def _stage_manifest_root(experiment_config: dict[str, Any]) -> Path:
    return Path(str(experiment_config["output_dir"])) / "three_stage"


def materialize_three_stage_run_manifest(
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    training_plan = build_three_stage_training_plan(experiment_config)
    manifest_root = _stage_manifest_root(experiment_config)
    generated_configs_dir = manifest_root / "generated_configs"
    generated_configs_dir.mkdir(parents=True, exist_ok=True)

    training_stages: list[dict[str, Any]] = []
    for phase_index, phase_record in enumerate(training_plan, start=1):
        phase_name = str(phase_record["phase_name"])
        stage_config = _build_stage_experiment_config(experiment_config, phase_record)
        stage_config_path = (
            generated_configs_dir / f"{phase_index:02d}_{phase_name}.yaml"
        )
        with stage_config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(stage_config, handle, sort_keys=False)
        training_stages.append(
            {
                "phase_name": phase_name,
                "epochs": int(phase_record["epochs"]),
                "global_epoch_start": int(phase_record["global_epoch_start"]),
                "global_epoch_end": int(phase_record["global_epoch_end"]),
                "config_path": str(stage_config_path),
                "output_dir": stage_config["output_dir"],
                "checkpoint_dir": stage_config["checkpoint_dir"],
                "initialization_checkpoint_path": stage_config.get(
                    "initialization_checkpoint_path"
                ),
                "best_checkpoint_path": str(
                    Path(stage_config["checkpoint_dir"]) / "best.pt"
                ),
                **_build_semantic_stage_metadata(phase_name),
            }
        )

    evaluation_config_path = generated_configs_dir / "06_evaluation_reference.yaml"
    with evaluation_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(experiment_config, handle, sort_keys=False)

    manifest = {
        "experiment_name": str(experiment_config["experiment_name"]),
        "total_training_epochs": compute_three_stage_total_training_epochs(
            experiment_config["three_stage"]
        ),
        "optimizer_training_phase_names": _optimizer_training_phase_names(),
        "statistical_procedure_names": list(STATISTICAL_PROCEDURE_NAMES),
        "training_stages": training_stages,
        "evaluation": {
            "config_path": str(evaluation_config_path),
            "checkpoint_path": training_stages[-1]["best_checkpoint_path"],
        },
    }
    manifest_path = manifest_root / "three_stage_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_three_stage_execution_commands(
    manifest: dict[str, Any],
) -> dict[str, list[list[str]] | list[str]]:
    training_commands: list[list[str]] = []
    for stage_record in manifest["training_stages"]:
        training_commands.append(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts" / "train.py"),
                "--experiment-config",
                str(stage_record["config_path"]),
            ]
        )
    evaluation_command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "evaluate.py"),
        "--experiment-config",
        str(manifest["evaluation"]["config_path"]),
        "--checkpoint-path",
        str(manifest["evaluation"]["checkpoint_path"]),
    ]
    return {
        "training": training_commands,
        "evaluation": evaluation_command,
    }


def _resolve_manifest_root_from_manifest(manifest: dict[str, Any]) -> Path:
    first_stage_config_path = Path(str(manifest["training_stages"][0]["config_path"]))
    return first_stage_config_path.parent.parent


def _collect_cnn_conv_module_names(model: torch.nn.Module) -> list[str]:
    conv_module_names: list[str] = []
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv1d) and module_name.startswith("encoder."):
            conv_module_names.append(module_name)
    if not conv_module_names:
        raise ValueError("Stage 2 MTZ approximation requires a CNN encoder")
    return conv_module_names


def _compute_cnn_activation_signatures(
    model: torch.nn.Module,
    train_loader: Any,
    *,
    max_batches: int,
) -> dict[str, torch.Tensor]:
    conv_module_names = _collect_cnn_conv_module_names(model)
    activations_by_layer: dict[str, list[torch.Tensor]] = {
        module_name: [] for module_name in conv_module_names
    }
    hook_handles = []
    modules_by_name = dict(model.named_modules())

    for module_name in conv_module_names:
        module = modules_by_name[module_name]

        def _capture_activation(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            _module_name: str = module_name,
        ) -> None:
            flattened_output = (
                output.detach()
                .cpu()
                .permute(1, 0, 2)
                .reshape(
                    output.shape[1],
                    -1,
                )
            )
            activations_by_layer[_module_name].append(flattened_output)

        hook_handles.append(module.register_forward_hook(_capture_activation))

    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(train_loader, start=1):
            model.encoder(batch)
            if batch_index >= max_batches:
                break

    for hook_handle in hook_handles:
        hook_handle.remove()

    signatures: dict[str, torch.Tensor] = {}
    for module_name, captured_batches in activations_by_layer.items():
        if not captured_batches:
            raise ValueError(
                "Stage 2 MTZ approximation could not capture encoder activations"
            )
        signatures[module_name] = torch.cat(captured_batches, dim=1)
    return signatures


def _match_channel_signatures_by_cosine_similarity(
    classification_signatures: dict[str, torch.Tensor],
    reconstruction_signatures: dict[str, torch.Tensor],
) -> dict[str, list[tuple[int, int]]]:
    channel_matches: dict[str, list[tuple[int, int]]] = {}
    for module_name, classification_signature in classification_signatures.items():
        reconstruction_signature = reconstruction_signatures[module_name]
        if classification_signature.shape[0] != reconstruction_signature.shape[0]:
            raise ValueError(
                "Stage 2 first-pass MTZ requires equal channel counts per matched layer"
            )
        similarity_matrix = torch.matmul(
            F.normalize(classification_signature, dim=1),
            F.normalize(reconstruction_signature, dim=1).transpose(0, 1),
        )
        unmatched_rows = set(range(similarity_matrix.shape[0]))
        unmatched_cols = set(range(similarity_matrix.shape[1]))
        layer_matches: list[tuple[int, int]] = []
        while unmatched_rows and unmatched_cols:
            best_pair: tuple[int, int] | None = None
            best_similarity = float("-inf")
            for row_index in sorted(unmatched_rows):
                for col_index in sorted(unmatched_cols):
                    similarity_value = float(similarity_matrix[row_index, col_index])
                    if similarity_value > best_similarity:
                        best_similarity = similarity_value
                        best_pair = (row_index, col_index)
            if best_pair is None:
                raise ValueError(
                    "Failed to compute deterministic Stage 2 channel match"
                )
            unmatched_rows.remove(best_pair[0])
            unmatched_cols.remove(best_pair[1])
            layer_matches.append(best_pair)
        channel_matches[module_name] = sorted(layer_matches, key=lambda pair: pair[0])
    return channel_matches


def _zip_cnn_encoder_state_dicts_with_matches(
    *,
    classification_state_dict: dict[str, torch.Tensor],
    reconstruction_state_dict: dict[str, torch.Tensor],
    channel_matches: dict[str, list[tuple[int, int]]],
) -> dict[str, torch.Tensor]:
    zipped_encoder_state_dict: dict[str, torch.Tensor] = {}
    previous_reconstruction_channel_order: list[int] | None = None

    for module_name in sorted(
        channel_matches.keys(),
        key=lambda name: int(name.rsplit(".", maxsplit=1)[-1]),
    ):
        layer_matches = channel_matches[module_name]
        classification_indices = [pair[0] for pair in layer_matches]
        reconstruction_indices = [pair[1] for pair in layer_matches]
        weight_key = f"{module_name}.weight"
        bias_key = f"{module_name}.bias"

        classification_weight = classification_state_dict[weight_key]
        reconstruction_weight = reconstruction_state_dict[weight_key]
        if previous_reconstruction_channel_order is not None:
            reconstruction_weight = reconstruction_weight[
                :,
                previous_reconstruction_channel_order,
                :,
            ]

        zipped_encoder_state_dict[weight_key] = 0.5 * (
            classification_weight[classification_indices]
            + reconstruction_weight[reconstruction_indices]
        )
        if (
            bias_key in classification_state_dict
            and bias_key in reconstruction_state_dict
        ):
            zipped_encoder_state_dict[bias_key] = 0.5 * (
                classification_state_dict[bias_key][classification_indices]
                + reconstruction_state_dict[bias_key][reconstruction_indices]
            )

        previous_reconstruction_channel_order = reconstruction_indices

    return zipped_encoder_state_dict


def _build_stage2_mtz_approximation_encoder_state_dict(
    classification_state_dict: dict[str, torch.Tensor],
    reconstruction_state_dict: dict[str, torch.Tensor],
    stage2_config: dict[str, Any],
    classification_config: dict[str, Any],
    reconstruction_config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    register_runtime_components()
    classification_model = build_model_from_experiment_config(classification_config)
    reconstruction_model = build_model_from_experiment_config(reconstruction_config)
    classification_model.load_state_dict(classification_state_dict)
    reconstruction_model.load_state_dict(reconstruction_state_dict)

    stage2_data_config = copy.deepcopy(stage2_config["data"])
    stage2_data_config["shuffle_train"] = False
    stage2_data_config["num_workers"] = 0
    data_bundle = build_dataset(stage2_data_config["dataset_name"], stage2_data_config)
    train_loader = data_bundle["loaders"]["train"]

    classification_signatures = _compute_cnn_activation_signatures(
        classification_model,
        train_loader,
        max_batches=STAGE2_ZIPPING_ACTIVATION_BATCHES,
    )
    reconstruction_signatures = _compute_cnn_activation_signatures(
        reconstruction_model,
        train_loader,
        max_batches=STAGE2_ZIPPING_ACTIVATION_BATCHES,
    )
    channel_matches = _match_channel_signatures_by_cosine_similarity(
        classification_signatures,
        reconstruction_signatures,
    )
    zipped_encoder_state_dict = _zip_cnn_encoder_state_dicts_with_matches(
        classification_state_dict=classification_state_dict,
        reconstruction_state_dict=reconstruction_state_dict,
        channel_matches=channel_matches,
    )
    zip_metadata = {
        "zipping_strategy": "mtz_approximation_activation_matching",
        "matching_policy": "greedy_cosine_channel_matching",
        "shared_scope": "encoder_only",
        "reused_head_policy": "stage1_task_specific_heads",
        "activation_source_split": "train",
        "activation_batches_used": STAGE2_ZIPPING_ACTIVATION_BATCHES,
        "matched_layer_names": list(channel_matches.keys()),
    }
    return zipped_encoder_state_dict, zip_metadata


def _prepare_stage2_recovery_initialization_checkpoint(
    manifest: dict[str, Any],
) -> Path:
    stage_records_by_phase = {
        stage_record["phase_name"]: stage_record
        for stage_record in manifest["training_stages"]
    }
    classification_checkpoint_path = Path(
        str(stage_records_by_phase["stage1_classification"]["best_checkpoint_path"])
    )
    reconstruction_checkpoint_path = Path(
        str(stage_records_by_phase["stage1_reconstruction"]["best_checkpoint_path"])
    )
    stage2_record = stage_records_by_phase["stage2_recovery"]
    initialization_checkpoint_path = Path(
        str(stage2_record["initialization_checkpoint_path"])
    )
    initialization_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    classification_checkpoint = torch.load(
        classification_checkpoint_path,
        map_location="cpu",
    )
    reconstruction_checkpoint = torch.load(
        reconstruction_checkpoint_path,
        map_location="cpu",
    )
    classification_config = load_experiment_config(
        stage_records_by_phase["stage1_classification"]["config_path"]
    )
    reconstruction_config = load_experiment_config(
        stage_records_by_phase["stage1_reconstruction"]["config_path"]
    )
    stage2_config = load_experiment_config(stage2_record["config_path"])
    register_runtime_components()
    stage2_model = build_model_from_experiment_config(stage2_config)
    zipped_state_dict = stage2_model.state_dict()
    classification_state_dict = classification_checkpoint["model_state_dict"]
    reconstruction_state_dict = reconstruction_checkpoint["model_state_dict"]
    zipped_encoder_state_dict, zip_metadata = (
        _build_stage2_mtz_approximation_encoder_state_dict(
            classification_state_dict,
            reconstruction_state_dict,
            stage2_config,
            classification_config,
            reconstruction_config,
        )
    )

    for state_key in list(zipped_state_dict.keys()):
        if state_key.startswith("encoder."):
            zipped_state_dict[state_key] = zipped_encoder_state_dict[state_key]
        elif state_key.startswith("classification_head."):
            zipped_state_dict[state_key] = classification_state_dict[state_key]
        elif state_key.startswith("reconstruction_head."):
            zipped_state_dict[state_key] = reconstruction_state_dict[state_key]

    stage2_model.load_state_dict(zipped_state_dict)
    initialization_payload = {
        "model_state_dict": stage2_model.state_dict(),
        "optimizer_state_dict": {},
        "scaler_state_dict": classification_checkpoint["scaler_state_dict"],
        "config": stage2_config,
        "epoch": 0,
        "metric_history": [],
        "extra_state": {
            "memory_initialized": False,
            "memory_training_enabled": False,
            "memory_ready_for_initialization": False,
            "memory_initialization_epoch": None,
            "stage2_zip_metadata": zip_metadata,
        },
    }
    torch.save(initialization_payload, initialization_checkpoint_path)
    console_print(
        "THREE_STAGE",
        "Prepared Stage 2 recovery initialization checkpoint",
        initialization_checkpoint_path=initialization_checkpoint_path,
        classification_checkpoint_path=classification_checkpoint_path,
        reconstruction_checkpoint_path=reconstruction_checkpoint_path,
        **zip_metadata,
    )
    return initialization_checkpoint_path
