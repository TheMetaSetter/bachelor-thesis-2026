"""Fit K-means to prepared windows and score distance to the nearest center."""

from copy import deepcopy
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


class KMeansAD:
    """Own train-fitted centers and validation scaling for point scores."""

    name = "kmeans_ad"

    def __init__(self, config: dict, *, encoder=None):
        if encoder is not None:
            raise ValueError("KMeansAD does not use an encoder")
        self.config = deepcopy(config)
        self.settings = self.config["method"]["parameters"]
        self.length = config["data"]["window_size"]
        self.centers = self.center = self.scale = None

    def _fit_windows(self, x: np.ndarray) -> None:
        from sklearn.cluster import KMeans

        rows = window_matrix(x, self.length, self.settings)
        clusters = self.settings["n_clusters"]
        if len(rows) < clusters:
            raise ValueError(
                "KMeansAD has fewer training windows than configured clusters"
            )
        model = KMeans(
            n_clusters=clusters,
            n_init=self.settings["n_init"],
            random_state=self.config["seed"],
        )
        self.centers = model.fit(rows).cluster_centers_.copy()

    def _raw_scores(self, x: np.ndarray) -> np.ndarray:
        from sklearn.metrics import pairwise_distances_argmin_min

        if self.centers is None:
            raise ValueError("KMeansAD requires fitted centers")
        rows = window_matrix(x, self.length, self.settings)
        return pairwise_distances_argmin_min(rows, self.centers, metric="euclidean")[1]

    def score(self, batch: dict) -> dict:
        require_no_labels(batch, "KMeansAD")
        if self.scale is None:
            raise ValueError("KMeansAD requires validation calibration")
        return build_score_record(
            batch,
            (self._raw_scores(batch["x"]) - self.center) / self.scale,
            score_kind="point",
        )

    def state_dict(self) -> dict:
        return deepcopy(
            {
                "schema_version": 1,
                "method": self.name,
                "length": self.length,
                "settings": self.settings,
                "centers": self.centers,
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
            raise ValueError("KMeansAD checkpoint identity mismatch")
        self.centers, self.center, self.scale = (
            deepcopy(state["centers"]),
            state["center"],
            state["scale"],
        )

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
            raise ValueError("KMeansAD requires eligible validation windows")
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
