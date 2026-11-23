"""Score each channel against a fixed STUMPY training reference."""

from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..artifacts import save_checkpoint
from ..prepared_data import _segments, iter_batches


class StumpyChannelAB:
    """Store per-channel distances and validation calibration."""

    def __init__(self, config: dict, *, encoder=None):
        if encoder is not None:
            raise ValueError("STUMPY does not use an encoder")
        self.config = deepcopy(config)
        self.settings = self.config["method"]["parameters"]
        self.length = config["data"]["window_size"]
        if self.length < 3:
            raise ValueError("STUMPY requires a window length of at least 3")
        self.reference = []
        self.center = self.scale = None

    def _distances(self, values):
        from stumpy import mass

        if not self.reference:
            raise ValueError("STUMPY requires a fitted reference")
        x = np.asarray(values, dtype=np.float64)
        if (
            x.ndim != 3
            or x.shape[1] != self.length
            or x.shape[2] != self.reference[0].shape[1]
        ):
            raise ValueError(
                "STUMPY input must match the reference length and channels"
            )
        if not np.isfinite(x).all():
            raise ValueError("STUMPY requires finite inputs")
        return np.array(
            [
                [
                    min(
                        mass(
                            window[:, channel],
                            segment[:, channel],
                            normalize=self.settings["normalize"],
                            p=self.settings["p"],
                        ).min()
                        for segment in self.reference
                    )
                    for channel in range(x.shape[2])
                ]
                for window in x
            ]
        )

    def score(self, batch: dict) -> dict:
        """Return the largest calibrated channel score at each endpoint."""
        if "labels" in batch or "point_labels" in batch:
            raise ValueError("STUMPY scoring does not accept labels")
        if self.center is None or self.scale is None:
            raise ValueError("STUMPY requires validation calibration")
        scores = ((self._distances(batch["x"]) - self.center) / self.scale).max(axis=1)
        result = {
            "score_kind": "point",
            "raw_scores": scores,
            "indices": np.asarray(batch["indices"])[:, -1],
        }
        result.update(
            {key: batch[key] for key in ("segment_ids", "identity") if key in batch}
        )
        return result

    def state_dict(self) -> dict:
        return deepcopy(
            {
                "schema_version": 1,
                "method": "stumpy_channel_ab",
                "length": self.length,
                "settings": self.settings,
                "reference": self.reference,
                "center": self.center,
                "scale": self.scale,
            }
        )

    def load_state_dict(self, state):
        if (
            state["schema_version"],
            state["method"],
            state["length"],
            state["settings"],
        ) != (1, "stumpy_channel_ab", self.length, self.settings):
            raise ValueError("STUMPY checkpoint identity mismatch")
        self.reference = deepcopy(state["reference"])
        self.center, self.scale = deepcopy(state["center"]), deepcopy(state["scale"])

    def _calibrate(self, validation):
        data = self.config["data"] | {"limits": self.config.get("limits", {})}
        collected = []
        for batch in iter_batches(
            validation, data, phase="offline", seed=self.config["seed"]
        ):
            labels = np.asarray(
                batch.get("labels", np.full(batch["indices"].shape, np.nan))
            )
            eligible = ~(labels == 1).any(axis=1)
            if eligible.any():
                collected.append(self._distances(batch["x"][eligible]))
        if not collected:
            raise ValueError("STUMPY requires eligible validation windows")
        values = np.concatenate(collected)
        low, center, high = np.quantile(values, [0.25, 0.5, 0.75], axis=0)
        self.center = center
        self.scale = np.maximum(high - low, self.settings["calibration_epsilon"])

    def fit(
        self,
        train: dict,
        validation: dict,
        *,
        checkpoint_dir: Path,
        resume: dict | None = None,
        on_checkpoint=None,
    ) -> dict:
        """Keep the training segments and fit channel calibration."""
        if self.config["phase"] != "offline":
            raise ValueError("STUMPY fitting requires an offline run")
        identity = {
            "method": "stumpy_channel_ab",
            "prepared_identity": self.config["prepared_identity"],
        }
        initial = (
            resume["initial"]
            if resume
            else save_checkpoint(
                checkpoint_dir / f"fit.initial.{uuid4().hex}.pt",
                self.state_dict(),
                identity=identity,
                role="fit.initial",
            )
        )
        if on_checkpoint is not None and not resume:
            on_checkpoint(
                {
                    "initial": initial,
                    "method_state": self.state_dict(),
                    "fit_complete": False,
                }
            )
        if resume and resume["fit_complete"]:
            self.load_state_dict(resume["method_state"])
            selected = resume["selected_checkpoint"]
        else:
            self.reference = [
                np.asarray(train["x"][start:stop], dtype=np.float64).copy()
                for start, stop in _segments(train, self.config["data"])
                if stop - start >= self.length
            ]
            self._calibrate(validation)
            selected = save_checkpoint(
                checkpoint_dir / f"fit.best.{uuid4().hex}.pt",
                self.state_dict(),
                identity=identity,
                role="fit.best",
            )
            if on_checkpoint is not None:
                on_checkpoint(
                    {
                        "initial": initial,
                        "method_state": self.state_dict(),
                        "fit_complete": True,
                        "selected_checkpoint": selected,
                    }
                )
        return {
            "selected_checkpoint": selected,
            "stages": [
                {
                    "name": "fit",
                    "initial_checkpoint": initial,
                    "best_checkpoint": selected,
                }
            ],
        }
