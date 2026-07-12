from scripts.analysis.compare_synthetic_profiles_shared import *  # noqa: F401,F403

def _plot_first_sample(
    clean_batch: dict[str, Any],
    legacy_batch: dict[str, Any],
    visible_batch: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_sample, _ = _select_most_visible_sample_channel(
        clean_batch,
        legacy_batch,
    )
    visible_sample, _ = _select_most_visible_sample_channel(
        clean_batch,
        visible_batch,
    )
    clean = clean_batch["x"].detach().cpu()
    legacy = legacy_batch["x"].detach().cpu()
    visible = visible_batch["x"].detach().cpu()
    legacy_annotation = _build_sample_plot_annotation(
        "Legacy profile",
        legacy_batch,
        legacy_sample,
    )
    visible_annotation = _build_sample_plot_annotation(
        "Visible profile",
        visible_batch,
        visible_sample,
    )
    legacy_mask = legacy_batch["synthetic_anomaly_mask"][legacy_sample].detach().cpu()
    visible_mask = (
        visible_batch["synthetic_anomaly_mask"][visible_sample].detach().cpu()
    )
    legacy_channels = _select_most_visible_channels(
        clean[legacy_sample],
        legacy[legacy_sample],
        legacy_mask.bool(),
    )
    visible_channels = _select_most_visible_channels(
        clean[visible_sample],
        visible[visible_sample],
        visible_mask.bool(),
    )

    figure, axes = plt.subplots(2, 2, figsize=(15, 7), constrained_layout=True)
    _plot_clean_reference_channels(
        axes[0, 0],
        clean[legacy_sample],
        legacy_channels,
        "Legacy",
    )
    _plot_profile_window(
        axes[1, 0],
        legacy[legacy_sample],
        legacy_annotation,
        legacy_channels,
    )
    _plot_clean_reference_channels(
        axes[0, 1],
        clean[visible_sample],
        visible_channels,
        "Visible",
    )
    _plot_profile_window(
        axes[1, 1],
        visible[visible_sample],
        visible_annotation,
        visible_channels,
    )
    for column_index in range(2):
        axes[1, column_index].set_xlabel("point index inside window")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    base_seed = _resolve_visualization_seed(args.seed)
    if bool(args.use_smd_entities):
        clean_batch = build_random_smd_entity_window_batch(
            entity_ids=list(SMD_ENTITY_CONFIG_PATHS),
            split_name=str(args.split),
            seed=base_seed,
        )
    else:
        clean_batch = _ensure_demo_batch_has_enough_channels(
            build_demo_batch(args.experiment_config),
            min_channels=12,
        )
    legacy_profile = _load_profile(args.legacy_profile)
    visible_profile = _load_profile(args.visible_profile)
    output_dir = _resolve_runtime_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_mode == "full_taxonomy":
        selected_profile = visible_profile
        gallery_batches = _build_family_gallery_batches(
            profile=selected_profile,
            clean_batch=clean_batch,
            seed=base_seed,
        )
        summary = {
            "requested_seed": args.seed,
            "base_visualization_seed": base_seed,
            "source": "smd_entities" if bool(args.use_smd_entities) else "demo_batch",
            "split": args.split if bool(args.use_smd_entities) else None,
            "plot_mode": args.plot_mode,
            "profile_name": selected_profile["profile_name"],
            "families": [
                {"family_name": item["family_name"], "seed": item["seed"]}
                for item in gallery_batches
            ],
        }
        summary_path = output_dir / "synthetic_profile_comparison_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), "utf-8")
        _plot_family_gallery(
            clean_batch,
            gallery_batches,
            output_dir / "synthetic_profile_comparison_first_sample.png",
        )
        return summary

    legacy_batch, legacy_seed = _augment_until_three_visible_channels(
        legacy_profile,
        clean_batch,
        base_seed,
    )
    visible_batch, visible_seed = _augment_until_three_visible_channels(
        visible_profile,
        clean_batch,
        base_seed,
    )

    summary = {
        "requested_seed": args.seed,
        "base_visualization_seed": base_seed,
        "experiment_config": args.experiment_config,
        "source": "smd_entities" if bool(args.use_smd_entities) else "demo_batch",
        "split": args.split if bool(args.use_smd_entities) else None,
        "plot_mode": args.plot_mode,
        "visualization_seeds": {
            "legacy": legacy_seed,
            "visible": visible_seed,
        },
        "legacy": _profile_summary("legacy", clean_batch, legacy_batch),
        "visible": _profile_summary("visible", clean_batch, visible_batch),
    }
    summary_path = output_dir / "synthetic_profile_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), "utf-8")
    _plot_first_sample(
        clean_batch,
        legacy_batch,
        visible_batch,
        output_dir / "synthetic_profile_comparison_first_sample.png",
    )
    return summary
