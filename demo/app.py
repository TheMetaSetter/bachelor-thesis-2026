from __future__ import annotations

"""Tiny demo entrypoint for offline and online replay images."""

import argparse
from pathlib import Path
from typing import Any

import yaml

from demo.offline_replay import build_offline_replay_state
from demo.online_replay import build_online_replay_state
from demo.plotting import plot_offline_replay, plot_online_replay


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def run_demo(
    *,
    offline_report_path: str | None,
    online_report_path: str | None,
    output_dir: str,
) -> dict[str, str]:
    output_root = Path(output_dir)
    outputs: dict[str, str] = {}
    if offline_report_path is not None:
        offline_state = build_offline_replay_state(offline_report_path)
        outputs["offline_replay"] = str(
            plot_offline_replay(
                offline_state,
                output_root / "offline_replay.png",
            )
        )
    if online_report_path is not None:
        online_state = build_online_replay_state(online_report_path)
        outputs["online_replay"] = str(
            plot_online_replay(
                online_state,
                output_root / "online_replay.png",
            )
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--offline-report", default=None)
    parser.add_argument("--online-report", default=None)
    parser.add_argument("--output-dir", default="outputs/demo")
    args = parser.parse_args()

    config = _load_yaml(args.config) if args.config is not None else {}
    offline_report_path = args.offline_report or config.get("offline_report_path")
    online_report_path = args.online_report or config.get("online_report_path")
    output_dir = (
        args.output_dir
        if args.output_dir != "outputs/demo"
        else config.get(
            "output_dir",
            "outputs/demo",
        )
    )
    run_demo(
        offline_report_path=offline_report_path,
        online_report_path=online_report_path,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
