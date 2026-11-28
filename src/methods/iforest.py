"""Fit and restore an Isolation Forest for prepared time-series windows."""

from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np

from ..method_runtime import (
    build_score_record,
    eligible_validation_batches,
    fit_endpoint_model,
    require_no_labels,
    window_matrix,
)
from ..prepared_data import iter_batches


class IsolationForestAD:
    """Store the fitted forest and its fixed validation scaling."""

    name = "iforest"

    def __init__(self, config: dict, *, encoder=None):
        if encoder is not None:
            raise ValueError("Isolation Forest does not use an encoder")
        self.config = deepcopy(config)
        self.settings = self.config["method"]["parameters"]
        self.length = config["data"]["window_size"]
        self.model = self.center = self.scale = None

    def _fit_windows(self, x: np.ndarray) -> None:
        from sklearn.ensemble import IsolationForest

        keys = ("n_estimators", "max_samples", "max_features", "contamination")
        settings = {key: self.settings[key] for key in keys}
        rows = window_matrix(x, self.length, self.settings)
        if type(settings["max_samples"]) is int and settings["max_samples"] > len(rows):
            raise ValueError(
                "Isolation Forest has fewer training windows than max_samples"
            )
        self.model = IsolationForest(**settings, random_state=self.config["seed"]).fit(
            rows
        )

    def _raw_scores(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Isolation Forest requires a fitted model")
        return -self.model.decision_function(
            window_matrix(x, self.length, self.settings)
        )

    def score(self, batch: dict) -> dict:
        require_no_labels(batch, "Isolation Forest")
        if self.scale is None:
            raise ValueError("Isolation Forest requires validation calibration")
        return build_score_record(
            batch,
            (self._raw_scores(batch["x"]) - self.center) / self.scale,
            score_kind="point",
        )

    def state_dict(self) -> dict:
        from skops.io import dumps

        return {
            "schema_version": 1,
            "method": self.name,
            "length": self.length,
            "settings": deepcopy(self.settings),
            "estimator": None if self.model is None else dumps(self.model),
            "sklearn_version": version("scikit-learn"),
            "skops_version": version("skops"),
            "center": self.center,
            "scale": self.scale,
        }

    def load_state_dict(self, state: dict) -> None:
        from sklearn.ensemble import IsolationForest
        from skops.io import get_untrusted_types, loads

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
            raise ValueError("Isolation Forest checkpoint identity mismatch")
        if state["sklearn_version"] != version("scikit-learn") or state[
            "skops_version"
        ] != version("skops"):
            raise ValueError("Isolation Forest checkpoint dependency version mismatch")
        payload = state["estimator"]
        if payload is not None and get_untrusted_types(data=payload):
            raise ValueError(
                "Isolation Forest checkpoint has unexpected estimator types"
            )
        model = loads(payload) if payload is not None else None
        if model is not None and type(model) is not IsolationForest:
            raise ValueError("Isolation Forest checkpoint has the wrong estimator")
        self.model, self.center, self.scale = model, state["center"], state["scale"]

    def _fit_train(self, train: dict) -> None:
        data = self.config["data"] | {"limits": self.config.get("limits", {})}
        windows = np.concatenate(
            [
                batch["x"]
                for batch in iter_batches(
                    train, data, phase="offline", seed=self.config["seed"]
                )
            ]
        )
        self._fit_windows(windows)

    def _calibrate(self, validation: dict) -> None:
        scores = [
            self._raw_scores(values)
            for values in eligible_validation_batches(validation, self.config)
        ]
        if not scores:
            raise ValueError("Isolation Forest requires eligible validation windows")
        low, center, high = np.quantile(np.concatenate(scores), [0.25, 0.5, 0.75])
        self.center, self.scale = float(center), max(float(high - low), 1e-8)

    def fit(
        self,
        train: dict,
        validation: dict,
        *,
        checkpoint_dir: Path,
        resume: dict | None = None,
        on_checkpoint=None,
    ) -> dict:
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
