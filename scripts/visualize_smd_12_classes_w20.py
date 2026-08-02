"""Create a 2-column x 6-row view of one clean SMD window and 11 variants."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.augment import (  # noqa: E402
    REDLAMP_ANOMALY_FAMILIES,
    REDLAMP_MULTICLASS_CLASS_NAMES,
    SyntheticAnomalyInjector,
)
from src.data.collate import collate_windows  # noqa: E402
from src.data.loaders import build_smd_dataset_bundle  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
ENTITY_ID = "machine-3-9"
WINDOW_SIZE = 20
OUTPUT_DIR = ROOT / "documents" / "logs" / "07-20-2026" / "detail"


def main() -> None:
    bundle = build_smd_dataset_bundle(
        {
            "dataset_name": "smd",
            "root_dir": str(ROOT / "data" / "ServerMachineDataset"),
            "entity_ids": [ENTITY_ID],
            "validation_split_ratio": 0.2,
            "window_size": WINDOW_SIZE,
            "stride": 1,
            "batch_size": 1,
            "num_workers": 0,
            "shuffle_train": False,
        }
    )
    train_dataset = bundle["datasets"]["train"]
    rng = np.random.default_rng()
    window_index = int(rng.integers(0, len(train_dataset)))
    clean_batch = collate_windows([train_dataset[window_index]])
    clean_window = clean_batch["x"][0].detach().cpu().numpy()

    entries = [("normal", clean_window, clean_batch["meta"][0], [])]
    for family in REDLAMP_ANOMALY_FAMILIES:
        injector = SyntheticAnomalyInjector(
            anomaly_probability=1.0,
            min_segment_fraction=0.2,
            max_segment_fraction=0.3,
            anomaly_visibility_boost=2.0,
            anomaly_families=(family,),
            family_intensity={"speedup": {"factors": [3.0]}},
            train_balance_classes=False,
            classification_label_mode="redlamp_multiclass",
            deterministic_seed=int(rng.integers(0, 2**31 - 1)),
        )
        augmented = injector.augment_batch(clean_batch)
        metadata = augmented["augmentation_metadata"][0]
        entries.append(
            (
                family,
                augmented["x"][0].detach().cpu().numpy(),
                metadata,
                metadata.get("affected_channels", []),
            )
        )

    # Prioritize channels affected by speedup/cutoff so those two families remain
    # visible even when the injector samples different channels per family.
    prioritized_channels = [
        int(channel)
        for name, _, _, affected in entries
        if name in {"speedup", "cutoff"}
        for channel in affected
    ]
    all_affected_channels = [
        int(channel) for _, _, _, affected in entries for channel in affected
    ]
    channels = list(dict.fromkeys(prioritized_channels + all_affected_channels))[:3]
    if not channels:
        channels = list(range(min(3, clean_window.shape[-1])))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for start, end in ((0, 6), (6, 12)):
        figure, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
        axes = axes.ravel()
        for axis, (name, window, metadata, affected) in zip(axes, entries[start:end]):
            for color_index, channel in enumerate(channels):
                axis.plot(
                    window[:, channel],
                    color=colors[color_index],
                    linewidth=1.5,
                    label=f"ch {channel}",
                )
            if name != "normal":
                changed = metadata.get("family_parameters_by_channel", {})
                for channel in channels:
                    positions = changed.get(str(channel), {}).get(
                        "changed_positions", []
                    )
                    if positions:
                        axis.scatter(
                            positions,
                            window[positions, channel],
                            s=18,
                            color=colors[channels.index(channel)],
                        )
            label_index = (
                0 if name == "normal" else REDLAMP_MULTICLASS_CLASS_NAMES.index(name)
            )
            axis.set_title(f"class {label_index}: {name}", loc="left", fontsize=10)
            axis.set_ylabel("value")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8, ncol=len(channels), loc="upper right")
        for axis in axes[-2:]:
            axis.set_xlabel("time-step")
        figure.suptitle(f"SMD {ENTITY_ID} — classes {start}–{end - 1}", fontsize=14)
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        output = OUTPUT_DIR / f"smd_3-9_classes_{start}-{end - 1}_w20.png"
        figure.savefig(output, dpi=180)
        plt.close(figure)
        print(f"Saved: {output.resolve()}")
    print(
        f"Source: entity={ENTITY_ID}, train_window_index={window_index}, channels={channels}"
    )


if __name__ == "__main__":
    main()
