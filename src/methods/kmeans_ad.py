"""Fit K-means to prepared windows and score distance to the nearest center."""

from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..artifacts import save_checkpoint
from ..prepared_data import iter_batches


def _window_matrix(x, length, settings):
    values = np.asarray(x, dtype=float)
    if values.ndim != 3 or values.shape[1] != length or not len(values):
        raise ValueError("estimator requires nonempty windows of the configured length")
    if not np.isfinite(values).all():
        raise ValueError("estimator requires finite feature values")
    rows = values.reshape(len(values), -1)
    if not settings["normalize_windows"]:
        return rows
    spread = (
        rows.std(axis=1, keepdims=True, ddof=1)
        if rows.shape[1] > 1
        else np.zeros((len(rows), 1))
    )
    scale = np.where(spread >= settings["normalization_epsilon"], spread, 1)
    return (rows - rows.mean(axis=1, keepdims=True)) / scale


def _endpoint_record(batch, scores):
    result = {
        "score_kind": "point",
        "raw_scores": scores,
        "indices": np.asarray(batch["indices"])[:, -1],
    }
    result.update(
        {key: batch[key] for key in ("segment_ids", "identity") if key in batch}
    )
    return result


def _fit_endpoint_estimator(
    method, train, validation, checkpoint_dir, resume, on_checkpoint
):
    config = method.config
    if config["phase"] != "offline":
        raise ValueError(f"{method.name} fitting requires an offline run")
    identity = {"method": method.name, "prepared_identity": config["prepared_identity"]}
    initial = (
        resume["initial"]
        if resume
        else save_checkpoint(
            checkpoint_dir / f"fit.initial.{uuid4().hex}.pt",
            method.state_dict(),
            identity=identity,
            role="fit.initial",
        )
    )
    if on_checkpoint is not None and resume is None:
        on_checkpoint(
            {
                "initial": initial,
                "method_state": method.state_dict(),
                "fit_complete": False,
            }
        )
    if resume and resume["fit_complete"]:
        method.load_state_dict(resume["method_state"])
        selected = resume["selected_checkpoint"]
    else:
        data = config["data"] | {"limits": config.get("limits", {})}
        method._fit_windows(
            np.concatenate(
                [
                    batch["x"]
                    for batch in iter_batches(
                        train, data, phase="offline", seed=config["seed"]
                    )
                ]
            )
        )
        validation_scores = []
        for batch in iter_batches(
            validation, data, phase="offline", seed=config["seed"]
        ):
            labels = np.asarray(
                batch.get("labels", np.full(batch["indices"].shape, np.nan))
            )
            eligible = ~(labels == 1).any(axis=1)
            if eligible.any():
                validation_scores.append(method._raw_scores(batch["x"][eligible]))
        if not validation_scores:
            raise ValueError(f"{method.name} requires eligible validation windows")
        low, center, high = np.quantile(
            np.concatenate(validation_scores), [0.25, 0.5, 0.75]
        )
        method.center, method.scale = float(center), max(float(high - low), 1e-8)
        selected = save_checkpoint(
            checkpoint_dir / f"fit.best.{uuid4().hex}.pt",
            method.state_dict(),
            identity=identity,
            role="fit.best",
        )
        if on_checkpoint is not None:
            on_checkpoint(
                {
                    "initial": initial,
                    "method_state": method.state_dict(),
                    "fit_complete": True,
                    "selected_checkpoint": selected,
                }
            )
    return {
        "selected_checkpoint": selected,
        "stages": [
            {"name": "fit", "initial_checkpoint": initial, "best_checkpoint": selected}
        ],
    }


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

    def _fit_windows(self, x):
        from sklearn.cluster import KMeans

        rows = _window_matrix(x, self.length, self.settings)
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

    def _raw_scores(self, x):
        from sklearn.metrics import pairwise_distances_argmin_min

        if self.centers is None:
            raise ValueError("KMeansAD requires fitted centers")
        rows = _window_matrix(x, self.length, self.settings)
        return pairwise_distances_argmin_min(rows, self.centers, metric="euclidean")[1]

    def score(self, batch: dict) -> dict:
        if "labels" in batch or "point_labels" in batch:
            raise ValueError("KMeansAD scoring does not accept labels")
        if self.scale is None:
            raise ValueError("KMeansAD requires validation calibration")
        return _endpoint_record(
            batch, (self._raw_scores(batch["x"]) - self.center) / self.scale
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

    def load_state_dict(self, state):
        if (
            state["schema_version"],
            state["method"],
            state["length"],
            state["settings"],
        ) != (1, self.name, self.length, self.settings):
            raise ValueError("KMeansAD checkpoint identity mismatch")
        self.centers, self.center, self.scale = (
            deepcopy(state["centers"]),
            state["center"],
            state["scale"],
        )

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
