from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(
    "/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/"
    "ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/"
    "Khoá luận tốt nghiệp/bachelor-thesis-2026/"
)
sys.path.insert(0, str(REPO))

from src.data.augment import SyntheticAnomalyInjector

ANOMALY_FAMILY = "wander"
WINDOW_SIZE = 20
WINDOW_STARTS = (200, 1200, 1400, 2200, 3200, 4200)
DPI = 600
SVG_DIR = REPO / "images/visualize-in-out-svg"
PNG_DIR = REPO / "images/visualize-in-out-png"
SVG_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR.mkdir(parents=True, exist_ok=True)

series = np.loadtxt(
    REPO / "data/ServerMachineDataset/train/machine-3-9.txt", delimiter=","
)
validation_length = int(series.shape[0] * 0.2)
validation = series[-validation_length:]
std = validation.std(axis=0)
channels = np.argsort(std)[::-1][:3]
time = np.arange(WINDOW_SIZE)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:3]

for window_start in WINDOW_STARTS:
    window = torch.tensor(
        validation[window_start : window_start + WINDOW_SIZE],
        dtype=torch.float32,
    )[None]
    batch = SyntheticAnomalyInjector(
        anomaly_probability=1.0,
        anomaly_families=(ANOMALY_FAMILY,),
        train_balance_classes=False,
        deterministic_seed=7,
    ).augment_batch({"x": window})
    augmented = batch["x"][0].numpy()
    point_mask = batch["synthetic_anomaly_mask"][0].numpy().astype(bool)

    for values, suffix, mark_anomalies in (
        (window[0].numpy(), "clean", False),
        (augmented, "synthetic-anomaly", True),
    ):
        fig, axis = plt.subplots(figsize=(4.5, 2.2))
        for color, channel in zip(colors, channels):
            axis.plot(time, values[:, channel], color=color)
            if mark_anomalies:
                axis.scatter(
                    time[point_mask],
                    values[point_mask, channel],
                    color="red",
                    zorder=3,
                )
        axis.set_axis_off()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        filename = f"machine-3-9-window-{window_start:04d}-{suffix}"
        for output_dir, file_format in ((SVG_DIR, "svg"), (PNG_DIR, "png")):
            output = output_dir / f"{filename}.{file_format}"
            fig.savefig(output, format=file_format, dpi=DPI, bbox_inches="tight")
            print(f"Đã lưu: {output.resolve()}")
        plt.close(fig)
