"""Score each channel against a fixed STUMPY training reference."""

from copy import deepcopy
from pathlib import Path

import numpy as np

from ..method_runtime import (
    build_score_record,
    eligible_validation_batches,
    fit_endpoint_model,
    require_no_labels,
)
from ..prepared_data import _segments


class StumpyChannelAB:
    """Store per-channel distances and validation calibration."""

    name = "stumpy_channel_ab"

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

    def _distances(self, values: np.ndarray) -> np.ndarray:
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
                            window[:, ch],
                            seg[:, ch],
                            normalize=self.settings["normalize"],
                            p=self.settings["p"],
                        ).min()
                        for seg in self.reference
                    )
                    for ch in range(x.shape[2])
                ]
                for window in x
            ]
        )

    def score(self, batch: dict) -> dict:
        """Return the largest calibrated channel score at each endpoint."""
        require_no_labels(batch, "STUMPY")
        if self.center is None or self.scale is None:
            raise ValueError("STUMPY requires validation calibration")
        scores = ((self._distances(batch["x"]) - self.center) / self.scale).max(axis=1)
        return build_score_record(batch, scores, score_kind="point")

    def state_dict(self) -> dict:
        return deepcopy(
            {
                "schema_version": 1,
                "method": self.name,
                "length": self.length,
                "settings": self.settings,
                "reference": self.reference,
                "center": self.center,
                "scale": self.scale,
            }
        )

    def load_state_dict(self, state: dict) -> None:
        if (
            state["schema_version"],
            state["method"],
            state["length"],
            state["settings"],
        ) != (
            1,
            self.name,
            self.length,
            self.settings,
        ):
            raise ValueError("STUMPY checkpoint identity mismatch")
        self.reference = deepcopy(state["reference"])
        self.center, self.scale = deepcopy(state["center"]), deepcopy(state["scale"])

    def _fit_train(self, train: dict) -> None:
        self.reference = [
            np.asarray(train["x"][start:stop], dtype=np.float64).copy()
            for start, stop in _segments(train, self.config["data"])
            if stop - start >= self.length
        ]

    def _calibrate(self, validation: dict) -> None:
        collected = [
            self._distances(values)
            for values in eligible_validation_batches(validation, self.config)
        ]
        if not collected:
            raise ValueError("STUMPY requires eligible validation windows")
        low, center, high = np.quantile(
            np.concatenate(collected), [0.25, 0.5, 0.75], axis=0
        )
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
        return fit_endpoint_model(
            self,
            train,
            validation,
            method_name=self.name,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            on_checkpoint=on_checkpoint,
            fit_fn=self._fit_train,
            calibrate_fn=self._calibrate,
        )
