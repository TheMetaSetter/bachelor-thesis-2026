from scripts.analysis.compare_synthetic_profiles_shared import *  # noqa: F401,F403
from scripts.analysis.compare_synthetic_profiles_helpers import *  # noqa: F401,F403
from scripts.analysis.compare_synthetic_profiles_shared import (
    _build_family_gallery_batches,
    _build_sample_plot_annotation,
    _ensure_demo_batch_has_enough_channels,
    _injected_point_indices,
    _resolve_visualization_seed,
    _select_most_visible_channels,
    _select_most_visible_sample_channel,
    _select_random_window_indices,
)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", default=None)
    parser.add_argument(
        "--legacy-profile",
        default="configs/protocol/synthetic_redlamp12_legacy_window20.yaml",
    )
    parser.add_argument(
        "--visible-profile",
        default="configs/protocol/synthetic_redlamp12_visible_window20.yaml",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fix the visualization seed. Omit this to show a different window each run.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/synthetic_profile_comparison",
    )
    parser.add_argument(
        "--use-smd-entities",
        action="store_true",
        help="Sample one real SMD window from each benchmark entity.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split to sample when --use-smd-entities is enabled.",
    )
    parser.add_argument(
        "--plot-mode",
        choices=["profile_comparison", "full_taxonomy"],
        default="profile_comparison",
    )
    summary = run_comparison(parser.parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
