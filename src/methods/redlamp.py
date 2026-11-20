"""Train RedLamp and score windows with validation-fixed settings."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..artifacts import save_checkpoint
from ..encoders import create_encoder
from ..prepared_data import iter_batches
from .augmentation import augment


def _corrected_targets(labels, classes, correction, *, dtype):
    alpha, beta = correction["alpha"], correction["beta"]
    one_hot = F.one_hot(labels.long(), classes).to(dtype)
    result = one_hot * (1 - alpha - classes * beta) + beta
    result[:, 0] += alpha
    return result


def _cell_mask(x, metadata):
    mask = torch.zeros_like(x, dtype=torch.bool)
    for row, item in enumerate(metadata):
        for channel, details in item["family_parameters_by_channel"].items():
            mask[row, details["changed_positions"], int(channel)] = True
    return mask


def _masked_reconstruction(x, reconstructed, mask):
    return (x - reconstructed).square().masked_fill(mask, 0).mean()


def _score_record(batch, values):
    record = {"score_kind": "point", "raw_scores": values,
              "indices": np.asarray(batch["indices"])[:, -1]}
    for key in ("segment_ids", "identity"):
        if key in batch:
            record[key] = batch[key]
    return record


class RedLamp:
    """Own the reconstruction model and frozen score calibration."""

    def __init__(self, config: dict, *, encoder: nn.Module | None = None):
        self.config = deepcopy(config)
        self.device = torch.device(config["device"])
        self.settings = self.config["method"]["parameters"]
        features = len(config["prepared_identity"]["features"])
        width = config["encoder"]["output_dim"]
        length = config.get("data", {}).get("window_size")
        self.encoder = encoder if encoder is not None else create_encoder(
            config["encoder"], input_features=features, window_size=length
        )
        dropout = self.settings["classifier_dropout"]
        self.reconstruction = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, features),
        )
        classifier_width = self.settings["classifier_hidden_size"]
        self.classifier = nn.Sequential(
            nn.Linear(width, classifier_width), nn.BatchNorm1d(classifier_width),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(classifier_width, 12),
        )
        self.network = nn.ModuleDict({"encoder": self.encoder, "reconstruction": self.reconstruction,
                                      "classifier": self.classifier}).to(self.device)
        if config["phase"] == "online":
            self.network.requires_grad_(False)
        self.rng = torch.Generator().manual_seed(config["seed"])
        self.class_rotation_offset = 0
        self.calibration = None

    def _forward(self, x):
        encoded = self.encoder(x)
        return self.reconstruction(encoded), self.classifier(encoded.mean(dim=1))

    @torch.no_grad()
    def _components(self, x):
        self.network.eval()
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        reconstruction, logits = self._forward(x)
        return (reconstruction - x).square().mean(dim=(1, 2)).cpu().numpy(), logits.softmax(-1).cpu().numpy()

    def _fit_score_calibration(self, reconstruction, probabilities):
        if not len(reconstruction):
            raise ValueError("RedLamp requires eligible validation windows")
        classes = np.flatnonzero(probabilities.mean(axis=0)[1:] <= self.settings["class_selection_probability"]) + 1
        classification = probabilities[:, classes].sum(axis=1)
        self.calibration = {
            "classes": classes.tolist(), "reconstruction_min": float(reconstruction.min()),
            "reconstruction_max": float(reconstruction.max()), "classification_min": float(classification.min()),
            "classification_max": float(classification.max()),
        }

    def score(self, batch: dict) -> dict:
        """Return the combined score at each window endpoint."""
        if "labels" in batch or "point_labels" in batch:
            raise ValueError("RedLamp scoring does not accept labels")
        if self.calibration is None:
            raise ValueError("RedLamp needs fitted validation score calibration")
        reconstruction, probabilities = self._components(batch["x"])
        calibration = self.calibration
        classification = probabilities[:, calibration["classes"]].sum(axis=1)
        epsilon = self.settings["zero_range_epsilon"]
        terms = []
        for key, value in (("reconstruction", reconstruction), ("classification", classification)):
            minimum = calibration[f"{key}_min"]
            scale = max(calibration[f"{key}_max"] - minimum, epsilon)
            terms.append((value - minimum) / scale)
        return _score_record(batch, (terms[0] + terms[1]) / 2)

    def state_dict(self) -> dict:
        return deepcopy({"schema_version": 1, "method": "redlamp", "network": self.network.state_dict(),
                         "calibration": self.calibration, "rng": self.rng.get_state(),
                         "class_rotation_offset": self.class_rotation_offset})

    def load_state_dict(self, state: Mapping) -> None:
        if state["schema_version"] != 1 or state["method"] != "redlamp":
            raise ValueError("RedLamp checkpoint identity mismatch")
        self.network.load_state_dict(state["network"])
        self.calibration = deepcopy(state["calibration"])
        self.rng.set_state(state["rng"].cpu())
        self.class_rotation_offset = state["class_rotation_offset"]

    def _batches(self, split):
        data = self.config["data"] | {"limits": self.config.get("limits", {})}
        return iter_batches(split, data, phase="offline", seed=self.config["seed"])

    def _batch_loss(self, x, metadata):
        altered, logits = self._forward(x)
        label = _corrected_targets(metadata["classification_labels"], 12, self.settings["label_correction"],
                                   dtype=logits.dtype)
        classification = -(label * logits.log_softmax(dim=1)).sum(dim=1).mean()
        reconstruction = _masked_reconstruction(x, altered, _cell_mask(x, metadata["augmentation_metadata"]))
        weight = self.settings["classification_weight"]
        return weight * classification + (1 - weight) * reconstruction

    def _epoch(self, split, optimizer=None):
        training = optimizer is not None
        self.network.train(training)
        rng = self.rng if training else torch.Generator().manual_seed(self.config["seed"])
        offset = self.class_rotation_offset if training else 0
        loss_total, count, skipped = 0.0, 0, 0
        for batch in self._batches(split):
            x = torch.as_tensor(batch["x"], device=self.device, dtype=torch.float32)
            if training and len(x) == 1:
                skipped += 1
                continue
            x, supervision = augment(x, self.config["synthetic"] | {"class_rotation_offset": offset}, generator=rng)
            offset = supervision["class_rotation_offset"]
            with torch.set_grad_enabled(training):
                loss = self._batch_loss(x, supervision)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    clip = self.config["method"]["training"]["stages"][0]["gradient_clip_norm"]
                    if clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.network.parameters(), clip, error_if_nonfinite=True)
                    optimizer.step()
            loss_total += float(loss.detach()) * len(x)
            count += len(x)
        if training:
            self.class_rotation_offset = offset
        if not count:
            raise ValueError("RedLamp: no eligible loss batches")
        return {"total_loss": loss_total / count, "skipped_windows": skipped}

    def _calibrate_validation(self, validation):
        errors, probabilities = [], []
        for batch in self._batches(validation):
            labels = np.asarray(batch.get("labels", np.full(batch["indices"].shape, np.nan)))
            selected = ~(labels == 1).any(axis=1)
            if selected.any():
                error, distribution = self._components(batch["x"][selected])
                errors.append(error)
                probabilities.append(distribution)
        if not errors:
            raise ValueError("RedLamp requires eligible validation windows")
        self._fit_score_calibration(np.concatenate(errors), np.concatenate(probabilities))

    def fit(self, train: dict, validation: dict, *, checkpoint_dir: Path, resume: dict | None = None,
            on_checkpoint=None) -> dict:
        """Fit the model, select its best state, and calibrate on validation."""
        if self.config["phase"] != "offline":
            raise ValueError("RedLamp fitting requires an offline run")
        stage = self.config["method"]["training"]["stages"][0]
        options = stage["optimizer"]
        optimizer = torch.optim.Adam(self.network.parameters(), lr=options["learning_rate"],
                                     weight_decay=options["weight_decay"], betas=tuple(options["betas"]),
                                     eps=options["epsilon"])
        identity = {"method": "redlamp", "prepared_identity": self.config["prepared_identity"]}
        if resume:
            self.load_state_dict(resume["method_state"])
            optimizer.load_state_dict(resume["optimizer_state"])
            initial, best, best_loss = resume["initial"], resume["best"], resume["best_loss"]
            history, start = list(resume["history"]), resume["next_epoch"]
        else:
            initial = save_checkpoint(checkpoint_dir / f"fit.initial.{uuid4().hex}.pt", self.state_dict(),
                                      identity=identity, role="fit.initial")
            history, start, best, best_loss = [], 0, None, float("inf")
        limit = self.config.get("limits", {}).get("max_epochs_per_stage", stage["epochs"])
        for epoch in range(start, min(limit, stage["epochs"])):
            train_record = self._epoch(train, optimizer)
            validation_record = self._epoch(validation)
            history.append({"epoch": epoch + 1, "train": train_record, "validation": validation_record})
            if validation_record["total_loss"] < best_loss:
                best_loss, best = validation_record["total_loss"], self.state_dict()
            if on_checkpoint is not None:
                on_checkpoint({"method_state": self.state_dict(), "optimizer_state": optimizer.state_dict(),
                               "initial": initial, "history": list(history), "next_epoch": epoch + 1,
                               "best": best, "best_loss": best_loss})
        if best is None:
            raise ValueError("RedLamp has no finite best validation loss")
        self.load_state_dict(best)
        self._calibrate_validation(validation)
        selected = save_checkpoint(checkpoint_dir / f"fit.best.{uuid4().hex}.pt", self.state_dict(),
                                   identity=identity, role="fit.best")
        return {"selected_checkpoint": selected, "stages": [{"name": "fit", "initial_checkpoint": initial,
                "best_checkpoint": selected, "best_validation_loss": best_loss, "epochs": history}]}
