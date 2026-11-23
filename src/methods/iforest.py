"""Fit and restore an Isolation Forest for prepared time-series windows."""

from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

from .kmeans_ad import _endpoint_record, _fit_endpoint_estimator, _window_matrix


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

    def _fit_windows(self, x):
        from sklearn.ensemble import IsolationForest

        keys = ("n_estimators", "max_samples", "max_features", "contamination")
        settings = {key: self.settings[key] for key in keys}
        rows = _window_matrix(x, self.length, self.settings)
        if type(settings["max_samples"]) is int and settings["max_samples"] > len(rows):
            raise ValueError(
                "Isolation Forest has fewer training windows than max_samples"
            )
        self.model = IsolationForest(**settings, random_state=self.config["seed"]).fit(
            rows
        )

    def _raw_scores(self, x):
        if self.model is None:
            raise ValueError("Isolation Forest requires a fitted model")
        return -self.model.decision_function(
            _window_matrix(x, self.length, self.settings)
        )

    def score(self, batch: dict) -> dict:
        if "labels" in batch or "point_labels" in batch:
            raise ValueError("Isolation Forest scoring does not accept labels")
        if self.scale is None:
            raise ValueError("Isolation Forest requires validation calibration")
        return _endpoint_record(
            batch, (self._raw_scores(batch["x"]) - self.center) / self.scale
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

    def load_state_dict(self, state):
        from sklearn.ensemble import IsolationForest
        from skops.io import get_untrusted_types, loads

        if (
            state["schema_version"],
            state["method"],
            state["length"],
            state["settings"],
        ) != (1, self.name, self.length, self.settings):
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

    def fit(
        self,
        train: dict,
        validation: dict,
        *,
        checkpoint_dir: Path,
        resume: dict | None = None,
        on_checkpoint=None,
    ) -> dict:
        return _fit_endpoint_estimator(
            self, train, validation, checkpoint_dir, resume, on_checkpoint
        )
