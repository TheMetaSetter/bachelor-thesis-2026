from __future__ import annotations

"""Small plotting helpers for demo replays."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from demo.demo_state import OfflineReplayState, OnlineReplayState


def _save_figure(figure, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_offline_replay(state: OfflineReplayState, output_path: str | Path) -> Path:
    figure, axes = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, constrained_layout=True
    )
    time_index = np.arange(state.raw_values.shape[0])
    for channel_index in range(min(3, state.raw_values.shape[1])):
        axes[0].plot(
            time_index,
            state.raw_values[:, channel_index],
            linewidth=0.9,
            label=f"ch {channel_index}",
        )
    axes[0].set_title(f"{state.entity_id} | offline replay")
    axes[0].legend(loc="upper right")
    axes[1].plot(
        time_index, state.point_scores, color="navy", linewidth=1.0, label="point score"
    )
    axes[1].axhline(
        state.threshold, color="black", linestyle="--", linewidth=1.0, label="threshold"
    )
    axes[1].fill_between(
        time_index,
        state.threshold,
        state.point_scores,
        where=state.predicted_mask,
        color="red",
        alpha=0.2,
    )
    axes[1].set_ylabel("score")
    axes[1].set_xlabel("time index")
    axes[1].legend(loc="upper right")
    return _save_figure(figure, output_path)


def plot_online_replay(state: OnlineReplayState, output_path: str | Path) -> Path:
    figure, axes = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, constrained_layout=True
    )
    time_index = np.arange(state.raw_values.shape[0])
    for channel_index in range(min(3, state.raw_values.shape[1])):
        axes[0].plot(
            time_index,
            state.raw_values[:, channel_index],
            linewidth=0.9,
            label=f"ch {channel_index}",
        )
    axes[0].set_title(f"{state.entity_id} | online replay")
    axes[0].legend(loc="upper right")
    score_time = state.score_indices
    axes[1].plot(
        score_time,
        state.raw_point_scores,
        color="steelblue",
        linewidth=0.9,
        label="raw score",
    )
    axes[1].plot(
        score_time,
        state.ewma_point_scores,
        color="navy",
        linewidth=1.0,
        label="EWMA score",
    )
    axes[1].axhline(
        state.threshold, color="black", linestyle="--", linewidth=1.0, label="threshold"
    )
    axes[1].fill_between(
        score_time,
        state.threshold,
        state.ewma_point_scores,
        where=state.predicted_mask,
        color="red",
        alpha=0.2,
    )
    axes[1].set_ylabel("score")
    axes[1].set_xlabel("absolute index")
    axes[1].legend(loc="upper right")
    return _save_figure(figure, output_path)
