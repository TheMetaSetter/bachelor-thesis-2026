"""RedLamp model computation flow: multiclass anomaly classifier and masked reconstruction."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..encoders import create_encoder
from ..method_runtime import (
    build_adam,
    build_score_record,
    corrected_cross_entropy,
    eligible_validation_batches,
    fit_neural_stages,
    require_no_labels,
    to_device_tensor,
)
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


class RedLamp:
    """Own the reconstruction/classification network and frozen score calibration."""

    def __init__(self, config: dict, *, encoder: nn.Module | None = None):
        self.config = deepcopy(config)
        self.device = torch.device(config["device"])
        self.settings = self.config["method"]["parameters"]
        features = len(config["prepared_identity"]["features"])
        width = config["encoder"]["output_dim"]
        length = config.get("data", {}).get("window_size")
        self.encoder = (
            encoder
            if encoder is not None
            else create_encoder(
                config["encoder"], input_features=features, window_size=length
            )
        )
        dropout = self.settings["classifier_dropout"]
        self.reconstruction = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, features),
        )
        classifier_width = self.settings["classifier_hidden_size"]
        self.classifier = nn.Sequential(
            nn.Linear(width, classifier_width),
            nn.BatchNorm1d(classifier_width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_width, 12),
        )
        self.network = nn.ModuleDict(
            {
                "encoder": self.encoder,
                "reconstruction": self.reconstruction,
                "classifier": self.classifier,
            }
        ).to(self.device)
        if config["phase"] == "online":
            self.network.requires_grad_(False)
        self.rng = torch.Generator().manual_seed(config["seed"])
        self.class_rotation_offset = 0
        self.calibration = None

    def _forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        return self.reconstruction(encoded), self.classifier(encoded.mean(dim=1))

    @torch.no_grad()
    def _components(self, x):
        self.network.eval()
        tensor = to_device_tensor(x, self.device)
        reconstruction, logits = self._forward(tensor)
        return (reconstruction - tensor).square().mean(
            dim=(1, 2)
        ).cpu().numpy(), logits.softmax(-1).cpu().numpy()

    def _fit_score_calibration(
        self, reconstruction: np.ndarray, probabilities: np.ndarray
    ) -> None:
        if not len(reconstruction):
            raise ValueError("RedLamp requires eligible validation windows")
        classes = (
            np.flatnonzero(
                probabilities.mean(axis=0)[1:]
                <= self.settings["class_selection_probability"]
            )
            + 1
        )
        classification = probabilities[:, classes].sum(axis=1)
        self.calibration = {
            "classes": classes.tolist(),
            "reconstruction_min": float(reconstruction.min()),
            "reconstruction_max": float(reconstruction.max()),
            "classification_min": float(classification.min()),
            "classification_max": float(classification.max()),
        }

    def score(self, batch: dict) -> dict:
        """Return the combined score at each window endpoint."""
        require_no_labels(batch, "RedLamp")
        if self.calibration is None:
            raise ValueError("RedLamp needs fitted validation score calibration")
        reconstruction, probabilities = self._components(batch["x"])
        classification = probabilities[:, self.calibration["classes"]].sum(axis=1)
        epsilon = self.settings["zero_range_epsilon"]
        terms = [
            (value - self.calibration[f"{key}_min"])
            / max(
                self.calibration[f"{key}_max"] - self.calibration[f"{key}_min"], epsilon
            )
            for key, value in (
                ("reconstruction", reconstruction),
                ("classification", classification),
            )
        ]
        return build_score_record(batch, (terms[0] + terms[1]) / 2, score_kind="point")

    def state_dict(self) -> dict:
        return deepcopy(
            {
                "schema_version": 1,
                "method": "redlamp",
                "network": self.network.state_dict(),
                "calibration": self.calibration,
                "rng": self.rng.get_state(),
                "class_rotation_offset": self.class_rotation_offset,
            }
        )

    def load_state_dict(self, state: Mapping) -> None:
        if state["schema_version"] != 1 or state["method"] != "redlamp":
            raise ValueError("RedLamp checkpoint identity mismatch")
        self.network.load_state_dict(state["network"])
        self.calibration = deepcopy(state["calibration"])
        self.rng.set_state(state["rng"].cpu())
        self.class_rotation_offset = state["class_rotation_offset"]

    def _batch_loss(self, x: torch.Tensor, metadata: dict) -> torch.Tensor:
        altered, logits = self._forward(x)
        classification = corrected_cross_entropy(
            logits, metadata["classification_labels"], self.settings["label_correction"]
        )
        mask = metadata.get("cell_mask")
        if mask is None:
            mask = _cell_mask(x, metadata["augmentation_metadata"])
        reconstruction = _masked_reconstruction(x, altered, mask)
        weight = self.settings["classification_weight"]
        return weight * classification + (1 - weight) * reconstruction

    def _epoch(self, split: dict, optimizer=None) -> dict:
        training = optimizer is not None
        self.network.train(training)
        rng = (
            self.rng if training else torch.Generator().manual_seed(self.config["seed"])
        )
        offset = self.class_rotation_offset if training else 0
        data = self.config["data"] | {"limits": self.config.get("limits", {})}
        loss_total, count, skipped = 0.0, 0, 0
        for batch in iter_batches(
            split, data, phase="offline", seed=self.config["seed"]
        ):
            x = to_device_tensor(batch["x"], self.device)
            if training and len(x) == 1:
                skipped += 1
                continue
            x, supervision = augment(
                x,
                self.config["synthetic"] | {"class_rotation_offset": offset},
                generator=rng,
            )
            offset = supervision["class_rotation_offset"]
            with torch.set_grad_enabled(training):
                loss = self._batch_loss(x, supervision)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    clip = self.config["method"]["training"]["stages"][0][
                        "gradient_clip_norm"
                    ]
                    if clip is not None:
                        nn.utils.clip_grad_norm_(
                            self.network.parameters(), clip, error_if_nonfinite=True
                        )
                    optimizer.step()
            loss_total += float(loss.detach()) * len(x)
            count += len(x)
        if training:
            self.class_rotation_offset = offset
        if not count:
            raise ValueError("RedLamp: no eligible loss batches")
        return {"total_loss": loss_total / count, "skipped_windows": skipped}

    def _calibrate_validation(self, validation: dict) -> None:
        pairs = [
            self._components(values)
            for values in eligible_validation_batches(validation, self.config)
        ]
        if not pairs:
            raise ValueError("RedLamp requires eligible validation windows")
        self._fit_score_calibration(
            np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])
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
        if self.config["phase"] != "offline":
            raise ValueError("RedLamp fitting requires an offline run")
        stage = self.config["method"]["training"]["stages"][0]
        optimizer = build_adam(self.network.parameters(), stage["optimizer"])
        if resume:
            optimizer.load_state_dict(resume["optimizer_state"])
        identity = {
            "method": "redlamp",
            "prepared_identity": self.config["prepared_identity"],
        }
        return fit_neural_stages(
            self,
            [stage | {"name": "fit"}],
            checkpoint_dir=checkpoint_dir,
            identity=identity,
            run_epoch=lambda _, epoch: {
                "epoch": epoch + 1,
                "train": self._epoch(train, optimizer),
                "validation": self._epoch(validation),
            },
            resume=resume,
            on_checkpoint=None
            if on_checkpoint is None
            else lambda s: on_checkpoint(
                s | {"optimizer_state": optimizer.state_dict()}
            ),
            post_fit=lambda: self._calibrate_validation(validation),
        )
