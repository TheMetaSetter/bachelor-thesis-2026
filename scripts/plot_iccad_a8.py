from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data/ibm-cloud-console-anomaly-dataset-iccad/anomaly_windows.csv"
DATA = ROOT / "data/ibm-cloud-console-anomaly-dataset-iccad/pivoted_data_all.parquet"
SPAN_ID = "a8"
POINTS_BEFORE = 200
POINTS_AFTER = 200
CHANNEL_CHUNK = 256


def read_span():
    for chunk in pd.read_csv(LABELS, chunksize=5):
        rows = chunk[chunk["number"] == SPAN_ID]
        if not rows.empty:
            return rows.iloc[0]
    raise ValueError(f"Không tìm thấy anomaly span {SPAN_ID}.")


def select_channels(parquet, anomaly_mask):
    # ICCAD gán nhãn theo thời gian, không gán nhãn riêng cho từng channel.
    # Vì vậy, chọn channel có dữ liệu trong a8 và std(a8) cao nhất.
    names = [name for name in parquet.schema_arrow.names if name.endswith("_avg")]
    scores = []
    for start in range(0, len(names), CHANNEL_CHUNK):
        chunk = parquet.read(columns=names[start : start + CHANNEL_CHUNK]).to_pandas()
        values = chunk.loc[anomaly_mask]
        standard_deviations = values.std(axis=0, ddof=0)
        for name, score in standard_deviations.items():
            if values[name].count() >= 2 and np.isfinite(score):
                scores.append((name, score))
    return sorted(scores, key=lambda item: item[1], reverse=True)[:3]


def main():
    span = read_span()
    start_time = pd.Timestamp(span.anomaly_start)
    end_time = pd.Timestamp(span.anomaly_end)
    parquet = pq.ParquetFile(DATA)
    timestamps = parquet.read(columns=["interval_start"]).column(0).to_numpy()
    anomaly_mask = (timestamps >= start_time.timestamp()) & (
        timestamps <= end_time.timestamp()
    )
    anomaly_indices = np.flatnonzero(anomaly_mask)
    if len(anomaly_indices) == 0:
        raise ValueError(f"Span {SPAN_ID} không chứa timestep trong Parquet.")

    selected = select_channels(parquet, anomaly_mask)
    left = max(anomaly_indices[0] - POINTS_BEFORE, 0)
    right = min(anomaly_indices[-1] + POINTS_AFTER + 1, len(timestamps))
    names = [name for name, _ in selected]
    data = parquet.read(columns=["interval_start", *names]).to_pandas().iloc[left:right]
    time = pd.to_datetime(data.pop("interval_start"), unit="s", utc=True)
    time = time.dt.tz_convert(start_time.tz)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for axis, (name, score) in zip(axes, selected):
        axis.plot(time, data[name], color="tab:blue", linewidth=0.8)
        axis.axvspan(
            start_time,
            end_time,
            color="tab:red",
            alpha=0.18,
            label=f"anomaly span {SPAN_ID}",
        )
        axis.set_title(
            f"{name}  |  std(a8)={score:.3g}", loc="left", pad=16, fontsize=8
        )
        axis.grid(alpha=0.2)

    axes[0].legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 1.55))
    fig.suptitle("ICCAD — anomaly span a8", y=0.98)
    fig.supxlabel("Thời gian", y=0.02)
    fig.supylabel("Giá trị", x=0.01)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.90, bottom=0.10, hspace=1.0)
    plt.show()


if __name__ == "__main__":
    main()
