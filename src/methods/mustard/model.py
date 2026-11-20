"""Train and score MUSTARD with direct routing for its two task heads."""

from collections.abc import Mapping
from copy import deepcopy
from itertools import islice
from pathlib import Path
from uuid import uuid4

import torch
from torch import nn
from torch.nn import functional as F

from ...artifacts import load_checkpoint, save_checkpoint
from ...config import load_yaml
from ...encoders import NearIdentityMLPProjector, create_encoder
from .signature_verification import nearest_discrete_codeword
from .verification_buffer import VerificationBuffer
from ...prepared_data import iter_batches
from ...scoring import reconstruction_scores
from ..augmentation import augment
from .losses import compute_loss
from .online_update import update_window
from .retrieval import _kmeans, retrieve
from .state import pack_state, restore_state


def _head(dimensions, dropout):
    layers = []
    for index, (width, next_width) in enumerate(zip(dimensions[:-1], dimensions[1:], strict=True)):
        layers.append(nn.Linear(width, next_width))
        if index < len(dimensions) - 2:
            layers.extend((nn.ReLU(), nn.Dropout(dropout)))
    return nn.Sequential(*layers)


def _adam(parameters, config):
    return torch.optim.Adam(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"],
                            betas=tuple(config["betas"]), eps=config["epsilon"])


class Mustard:
    """Own the trainable network, memory banks, and online update state."""

    def __init__(self, config: dict, *, encoder: nn.Module | None = None):
        self.config = deepcopy(config)
        self.variant = config["variant"]
        self.device = torch.device(config["device"])
        self.parameters_config = self.config["method"]["parameters"]
        self.source_variant = config.get("required_offline_variant") or config["method"].get(
            "required_offline_variant") or self.variant
        self.source_parameters = self.parameters_config
        if config["phase"] == "online":
            source = config.get("source_method")
            if source is None:
                path = Path(config["project_root"]) / "configs/method_profiles" / f"mustard_{self.source_variant.lower()}_offline.yaml"
                source = load_yaml(path)
            self.source_parameters = source["parameters"]
        inputs = len(config["prepared_identity"]["features"])
        length = config["data"]["window_size"]
        hidden = config["encoder"]["output_dim"]
        self.shape = (length, inputs, hidden)
        self.encoder = encoder if encoder is not None else create_encoder(
            config["encoder"], input_features=inputs, window_size=length
        )
        classifier = self.source_parameters["classifier_hidden_size"]
        dropout = self.source_parameters["head_dropout"]
        self.reconstruction_head = _head((hidden, hidden, hidden, inputs), dropout)
        self.classification_head = _head((length * hidden, classifier, classifier, 12), dropout)
        self.network = nn.ModuleDict({"encoder": self.encoder, "reconstruction_head": self.reconstruction_head,
                                      "classification_head": self.classification_head})
        self.projector = None
        self.verification_buffer = None
        self.memory: dict[str, torch.Tensor] = {}
        self.rng = torch.Generator().manual_seed(config["seed"])
        self.class_rotation_offset = 0
        self.optimizer = None
        if config["phase"] == "online" and self.variant != "A0":
            self._initialize_online(hidden)
        self.network.to(self.device)
        self._set_stage("stage_a" if config["phase"] == "offline" else "online")

    def _initialize_online(self, hidden):
        settings = self.parameters_config
        self.projector = NearIdentityMLPProjector(
            hidden, settings["projector_hidden_size"], settings["projector_dropout"]
        )
        self.network["projector"] = self.projector
        verification = settings["verification"]
        self.verification_buffer = VerificationBuffer(
            max_size=verification["buffer_size"], non_overlap_gap=verification["non_overlap_gap"],
            verification_capacity=verification["cycle_size"], default_ttl=2,
        )

    def _set_stage(self, stage):
        self.stage = stage
        self.encoder.requires_grad_(stage == "stage_a")
        self.reconstruction_head.requires_grad_(stage != "online")
        self.classification_head.requires_grad_(stage != "online")
        if stage == "online":
            self.network.eval()
            self.optimizer = None if self.projector is None else _adam(
                self.projector.parameters(), self.config["method"]["update"]["optimizer"]
            )
        else:
            stages = self.config["method"]["training"]["stages"]
            definition = next(item for item in stages if item["name"] == stage)
            self.optimizer = _adam((p for p in self.network.parameters() if p.requires_grad), definition["optimizer"])

    @torch.no_grad()
    def _initialize_memory(self, features, labels, masks):
        params = self.source_parameters
        epsilon = params["memory"]["normalization_epsilon"]
        normal = features[(labels[:, None] == 0) & ~masks.bool()]
        if params["continuous"]["enabled"]:
            self.memory["continuous"] = _kmeans(normal, params["continuous"]["num_prototypes"], epsilon)
        if params["discrete"]["enabled"]:
            count = params["discrete"]["codebook_size"]
            groups = [features[labels == code].reshape(-1, features.shape[-1]) for code in range(12)]
            fallback = features.flatten(0, 1)
            centers, classes = [], []
            for code, group in enumerate(groups):
                size = count // 12 + int(code < count % 12)
                if size:
                    centers.append(_kmeans(group if len(group) else fallback, size, epsilon))
                    classes.extend([code] * size)
            self.memory["discrete"] = torch.cat(centers)
            self.memory["codeword_class_ids"] = torch.tensor(classes, device=self.device)
            self.memory["anomalous_mask"] = self.memory["codeword_class_ids"] != 0
            self._calibrate_radii(features[labels != 0].flatten(0, 1))

    def _calibrate_radii(self, anomaly_tokens):
        bank = self.memory["discrete"]
        radii = bank.new_zeros(len(bank))
        counts = torch.zeros(len(bank), dtype=torch.long, device=bank.device)
        if len(anomaly_tokens):
            nearest, distance = nearest_discrete_codeword(anomaly_tokens[None], bank)
            nearest, distance = nearest.flatten(), distance.flatten()
            counts = torch.bincount(nearest, minlength=len(bank))
            for index in nearest.unique().tolist():
                radii[index] = torch.quantile(distance[nearest == index], 0.99).clamp_min(0)
        self.memory.update(radii=radii, contributing_token_counts=counts)

    def _forward(self, x, *, stochastic=False):
        hidden = self.encoder(x)
        projected = self.projector(hidden) if self.projector is not None else hidden
        settings = self.source_parameters | ({"generator": self.rng} if stochastic else {})
        branches = retrieve(projected, self.memory if self.stage != "stage_a" else {}, settings)
        return {
            **branches, "hidden": hidden, "projected": projected,
            "reconstruction": self.reconstruction_head(branches["continuous"]),
            "logits": self.classification_head(branches["discrete"].flatten(1)),
        }

    @torch.no_grad()
    def score(self, batch: dict) -> dict:
        """Return unsmoothed point scores and separate online triage scores."""
        if "labels" in batch or "point_labels" in batch:
            raise ValueError("MUSTARD scoring does not accept labels")
        self.network.eval()
        window = torch.as_tensor(batch["x"], dtype=torch.float32, device=self.device)
        evaluation = self.config["evaluation"]
        space = evaluation["score_space"]
        if space == "latent" and "continuous" not in self.memory:
            raise ValueError("latent MSE requires the continuous prototype bank")
        stochastic = self.source_parameters["stochastic_inference"]
        count = self.source_parameters["monte_carlo_samples"] if stochastic else 1
        passes = [self._forward(window, stochastic=stochastic) for _ in range(count)]
        if space == "latent":
            samples = torch.stack([item["continuous"] for item in passes], dim=1)
            errors = (samples - passes[0]["query"][:, None]).square().mean(dim=-1)
        else:
            samples = torch.stack([item["reconstruction"] for item in passes], dim=1)
            errors = self._point_errors(window, samples, space)
        result = {"score_kind": "window", "raw_scores": errors.mean(dim=1).cpu().numpy()}
        result.update({key: batch[key] for key in ("indices", "segment_ids", "identity") if key in batch})
        if self.variant in {"A1", "A2"}:
            result["auxiliary"] = self._triage_scores(window, passes)
        if evaluation.get("uncertainty_fields"):
            result["uncertainty"] = self._uncertainty(passes, errors, evaluation["uncertainty_fields"])
        return result

    def _point_errors(self, window, reconstructions, space):
        settings = {"score_space": space, "scaler": self.config.get("scaler")}
        return torch.stack([reconstruction_scores(window, sample, settings)
                            for sample in reconstructions.unbind(dim=1)], dim=1)

    def _uncertainty(self, passes, errors, requested):
        correction = self.source_parameters["variance_correction"]
        if len(passes) <= correction:
            return {}
        def variance(name):
            values = [item[name] for item in passes]
            if name == "logits":
                values = [value.softmax(dim=-1) for value in values]
            return torch.stack(values, dim=1).var(dim=1, correction=correction)

        result = {
            "point_anomaly_score_variance": errors.var(dim=1, correction=correction),
            "window_anomaly_score_variance": errors.mean(dim=-1).var(dim=1, correction=correction),
        }
        for name, prefix in (("continuous", "continuous_retrieval"), ("discrete", "discrete_retrieval"),
                             ("reconstruction", "reconstruction")):
            full = variance(name)
            if name == "reconstruction":
                result["reconstruction_variance_full"] = full
            result[f"{prefix}_variance_point"] = full.mean(dim=-1)
            result[f"{prefix}_variance_window"] = full.mean(dim=(-1, -2))
        result["classification_probability_variance"] = variance("logits")
        result["classification_variance_mean"] = result["classification_probability_variance"].mean(dim=-1)
        return {name: result[name].cpu().numpy() for name in requested}

    def _triage_scores(self, window, passes):
        recon = torch.stack([item["reconstruction"] for item in passes], dim=1)
        raw = self._point_errors(window, recon, "raw_input")
        features, bank = F.normalize(passes[0]["projected"], dim=-1), self.memory["continuous"]
        distance = 1 - features @ F.normalize(bank, dim=-1).T
        return {"input_window_score": raw.mean(dim=(1, 2)).cpu().numpy(),
                "latent_window_score": distance.amin(dim=-1).mean(dim=-1).cpu().numpy()}

    def update(self, batch: dict, context: dict) -> dict:
        """Apply an allowed test-time projector update."""
        if self.config["phase"] != "online":
            raise ValueError("MUSTARD updates require an online variant")
        x = torch.as_tensor(batch["x"], dtype=torch.float32, device=self.device)
        return update_window(self, batch | {"x": x}, context)

    def state_dict(self) -> dict:
        return pack_state(self)

    def load_state_dict(self, state: Mapping) -> None:
        restore_state(self, state)

    def _batches(self, split):
        data = self.config["data"] | {"limits": self.config.get("limits", {})}
        return iter_batches(split, data, phase="offline", seed=self.config["seed"])

    def _augment(self, batch, rng, offset):
        clean = torch.as_tensor(batch["x"], dtype=torch.float32, device=self.device)
        settings = self.config["synthetic"] | {"class_rotation_offset": offset}
        altered, labels = augment(clean, settings, generator=rng)
        return clean, altered, labels

    @torch.no_grad()
    def _initialize_from_training(self, train):
        features, labels, masks = [], [], []
        batches = self.source_parameters["memory"]["initialization_batches"]
        self.encoder.eval()
        for batch in islice(self._batches(train), batches):
            _, altered, supervision = self._augment(batch, self.rng, self.class_rotation_offset)
            self.class_rotation_offset = supervision["class_rotation_offset"]
            features.append(self.encoder(altered))
            labels.append(supervision["classification_labels"])
            masks.append(supervision["synthetic_anomaly_mask"])
        self._initialize_memory(torch.cat(features), torch.cat(labels), torch.cat(masks))

    def _epoch(self, split, settings, *, fitting):
        self.network.train(fitting)
        if self.stage == "stage_b":
            self.encoder.eval()
        rng = self.rng if fitting else torch.Generator().manual_seed(self.config["seed"])
        offset = self.class_rotation_offset if fitting else 0
        total, count, batches, unavailable = 0.0, 0, 0, 0
        for batch in self._batches(split):
            clean, altered, targets = self._augment(batch, rng, offset)
            offset = targets["class_rotation_offset"]
            with torch.set_grad_enabled(fitting):
                output = self._forward(altered)
                if self.variant == "O2" and self.stage == "stage_a":
                    output["clean_features"] = self.encoder(clean)
                losses = compute_loss(output, targets | {"x": altered}, self.source_parameters | {
                    "variant": self.variant, "stage": self.stage,
                })
                if fitting:
                    self.optimizer.zero_grad(set_to_none=True)
                    losses["total"].backward()
                    clip = settings["gradient_clip_norm"]
                    if clip is not None:
                        nn.utils.clip_grad_norm_(self.network.parameters(), clip, error_if_nonfinite=True)
                    self.optimizer.step()
            total += float(losses["total"].detach()) * len(clean)
            count += len(clean)
            batches += 1
            if self.variant == "O1" and self.stage == "stage_a" and "point_score" not in losses:
                unavailable += 1
        if fitting:
            self.class_rotation_offset = offset
        if not count:
            raise ValueError("MUSTARD epoch requires at least one window")
        return {"total_loss": total / count, "batch_count": batches,
                "point_score_unavailable_batches": unavailable}

    def _checkpoint_identity(self):
        return {"method": "mustard", "variant": self.variant, "shape": self.shape,
                "prepared_identity": self.config["prepared_identity"]}

    def _fit_stage(self, train, validation, definition, directory, stages, resume, on_checkpoint):
        stage = definition["name"]
        previous = resume if resume and resume["stage"] == stage else None
        if previous is None:
            self._set_stage(stage)
            if stage == "stage_b":
                self._initialize_from_training(train)
        else:
            self.load_state_dict(previous["method_state"])
        identity = self._checkpoint_identity()
        initial = Path(previous["initial"]) if previous else save_checkpoint(
            directory / f"{stage}.initial.{uuid4().hex}.pt", self.state_dict(), identity=identity, role=f"{stage}.initial"
        )
        best = previous["best"] if previous else None
        best_loss = previous["best_loss"] if previous else float("inf")
        start = previous["next_epoch"] if previous else 0
        count = min(definition["epochs"], self.config.get("limits", {}).get("max_epochs_per_stage", definition["epochs"]))
        history = list(previous["history"]) if previous else []
        for epoch in range(start, count):
            training = self._epoch(train, definition, fitting=True)
            evaluation = self._epoch(validation, definition, fitting=False)
            value = evaluation["total_loss"]
            if value < best_loss:
                best_loss = value
                best = save_checkpoint(directory / f"{stage}.best.{uuid4().hex}.pt", self.state_dict(),
                                       identity=identity, role=f"{stage}.best")
            history.append({"epoch": epoch + 1, "train": training, "validation": evaluation})
            if on_checkpoint is not None:
                on_checkpoint({"stage": stage, "next_epoch": epoch + 1, "method_state": self.state_dict(),
                               "best": best, "best_loss": best_loss, "initial": initial,
                               "stages": list(stages), "history": list(history)})
        if best is None:
            raise ValueError(f"MUSTARD {stage}: no finite best validation loss")
        self.load_state_dict(load_checkpoint(Path(best), expected_identity=identity | {"role": f"{stage}.best"}))
        return {"name": stage, "initial_checkpoint": initial, "best_checkpoint": best,
                "best_validation_loss": best_loss, "epochs": history}

    def fit(self, train: dict, validation: dict, *, checkpoint_dir: Path, resume: dict | None = None,
            on_checkpoint=None) -> dict:
        """Fit both offline stages and retain each initial and best checkpoint."""
        if self.config["phase"] != "offline":
            raise ValueError("MUSTARD fitting requires an offline variant")
        stages = list(resume["stages"]) if resume else []
        definitions = self.config["method"]["training"]["stages"]
        for definition in definitions[len(stages):]:
            stage = self._fit_stage(train, validation, definition, checkpoint_dir, stages, resume, on_checkpoint)
            stages.append(stage)
            if on_checkpoint is not None:
                on_checkpoint({"stage": stage["name"], "stages": list(stages), "method_state": self.state_dict()})
        self.network.eval()
        return {"selected_checkpoint": stages[-1]["best_checkpoint"], "stages": stages}
