from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_diagnostics_records(
    classification_diagnostics_dir: Path,
) -> dict[str, list[dict]]:
    records_by_stage: dict[str, list[dict]] = {"train": [], "val_synth": []}
    for json_path in sorted(classification_diagnostics_dir.glob("epoch_*_*.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        stage = record.get("stage")
        if stage in records_by_stage:
            records_by_stage[stage].append(record)
    for stage_name in records_by_stage:
        records_by_stage[stage_name].sort(key=lambda record: int(record["epoch"]))
    return records_by_stage


def _plot_confusion_heatmap(
    *,
    normalized_matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(normalized_matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_title(title)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(np.arange(len(class_names)))
    axis.set_yticks(np.arange(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticklabels(class_names)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Row-normalized value")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_class_ratio_stacked_area(
    *,
    supports_by_epoch: list[list[int]],
    epochs: list[int],
    class_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    support_matrix = np.asarray(supports_by_epoch, dtype=np.float64)
    row_sums = support_matrix.sum(axis=1, keepdims=True)
    ratios = np.divide(
        support_matrix,
        np.clip(row_sums, a_min=1.0, a_max=None),
        out=np.zeros_like(support_matrix, dtype=np.float64),
        where=row_sums > 0,
    )

    color_palette = list(plt.get_cmap("tab20").colors)
    class_colors = color_palette[: len(class_names)]

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.stackplot(epochs, ratios.T, labels=class_names, colors=class_colors)
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Synthetic class ratio")
    axis.set_ylim(0.0, 1.0)
    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def visualize_experiment_output(experiment_output_dir: Path) -> None:
    classification_diagnostics_dir = (
        experiment_output_dir / "classification_diagnostics"
    )
    visualization_dir = (
        experiment_output_dir / "classification_diagnostics_visualizations"
    )
    visualization_dir.mkdir(parents=True, exist_ok=True)

    if not classification_diagnostics_dir.exists():
        no_data_path = visualization_dir / "README_no_classification_diagnostics.txt"
        no_data_path.write_text(
            "No classification_diagnostics directory found. "
            "This run likely disabled classification path or diagnostics logging.\n",
            encoding="utf-8",
        )
        print(f"[INFO] {experiment_output_dir}: no diagnostics found.")
        return

    records_by_stage = _load_diagnostics_records(classification_diagnostics_dir)
    for stage_name, records in records_by_stage.items():
        if not records:
            continue
        latest_record = records[-1]
        class_names = latest_record["class_names"]
        normalized_matrix = np.asarray(
            latest_record["row_normalized"], dtype=np.float64
        )
        heatmap_path = visualization_dir / f"{stage_name}_confusion_heatmap_latest.png"
        _plot_confusion_heatmap(
            normalized_matrix=normalized_matrix,
            class_names=class_names,
            output_path=heatmap_path,
            title=f"{experiment_output_dir.name} | {stage_name} | latest epoch {latest_record['epoch']}",
        )

        supports_by_epoch = [record["support"] for record in records]
        epochs = [int(record["epoch"]) for record in records]
        ratio_path = (
            visualization_dir / f"{stage_name}_synthetic_class_ratio_over_epochs.png"
        )
        _plot_class_ratio_stacked_area(
            supports_by_epoch=supports_by_epoch,
            epochs=epochs,
            class_names=class_names,
            output_path=ratio_path,
            title=f"{experiment_output_dir.name} | {stage_name} | synthetic anomaly class ratio",
        )
    print(f"[INFO] {experiment_output_dir}: visualization export completed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-output-dir",
        action="append",
        required=True,
        help="Experiment output directory. Pass this argument multiple times for multiple runs.",
    )
    args = parser.parse_args()

    for experiment_output_dir in args.experiment_output_dir:
        visualize_experiment_output(Path(experiment_output_dir))


if __name__ == "__main__":
    main()
