"""MUSTARD model computation flow: encoder, dual-branch retrieval, heads, and projector."""

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from itertools import islice
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ...encoders import create_encoder
from ...method_runtime import (
    build_adam,
    build_score_record,
    fit_neural_stages,
    require_no_labels,
    resolve_source_parameters,
    to_device_tensor,
)
from ...prepared_data import iter_batches
from ..augmentation import augment
from .losses import compute_loss
from .online_update import update_window
from .retrieval import _kmeans, retrieve
from .score_outputs import point_errors, uncertainty
from .signature_verification import nearest_discrete_codeword
from .state import pack_state, restore_state
from .verification_buffer import VerificationBuffer


def _mlp_head(dimensions: tuple[int, ...], dropout: float) -> nn.Sequential:
    layers = []
    for index, (width, next_width) in enumerate(
        zip(dimensions[:-1], dimensions[1:], strict=True)
    ):
        layers.append(nn.Linear(width, next_width))
        if index < len(dimensions) - 2:
            layers.extend((nn.ReLU(), nn.Dropout(dropout)))
    return nn.Sequential(*layers)


class NearIdentityMLPProjector(nn.Module):
    """Residual projector initialized near identity for online adaptation."""

    def __init__(
        self, hidden_dim: int, projector_hidden_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, projector_hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(projector_hidden_dim, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(1.0e-3))
        nn.init.kaiming_uniform_(self.fc1.weight, a=5**0.5)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0e-4)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.alpha * self.fc2(
            self.dropout(self.activation(self.fc1(hidden)))
        )


class Mustard:
    """Own the trainable network, memory banks, and online update state."""

    def __init__(self, config: dict, *, encoder: nn.Module | None = None):
        self.config = deepcopy(config)
        self.variant = config["variant"]
        self.device = torch.device(config["device"])
        self.parameters_config = self.config["method"]["parameters"]
        self.source_variant = (
            config.get("required_offline_variant")
            or config["method"].get("required_offline_variant")
            or self.variant
        )
        self.source_parameters = resolve_source_parameters(
            self.config, self.source_variant
        )
        inputs = len(config["prepared_identity"]["features"])
        length, hidden = config["data"]["window_size"], config["encoder"]["output_dim"]
        self.shape = (length, inputs, hidden)
        self.encoder = (
            encoder
            if encoder is not None
            else create_encoder(
                config["encoder"], input_features=inputs, window_size=length
            )
        )
        classifier, dropout = (
            self.source_parameters["classifier_hidden_size"],
            self.source_parameters["head_dropout"],
        )
        self.reconstruction_head = _mlp_head((hidden, hidden, hidden, inputs), dropout)
        self.classification_head = _mlp_head(
            (length * hidden, classifier, classifier, 12), dropout
        )
        self.network = nn.ModuleDict(
            {
                "encoder": self.encoder,
                "reconstruction_head": self.reconstruction_head,
                "classification_head": self.classification_head,
            }
        )
        self.projector, self.verification_buffer, self.optimizer = None, None, None
        self.memory: dict[str, torch.Tensor] = {}
        self.rng = torch.Generator().manual_seed(config["seed"])
        self.class_rotation_offset = 0
        if config["phase"] == "online" and self.variant != "A0":
            self._initialize_online(hidden)
        self.network.to(self.device)
        self._set_stage("stage_a" if config["phase"] == "offline" else "online")

    def _initialize_online(self, hidden: int) -> None:
        settings = self.parameters_config
        self.projector = NearIdentityMLPProjector(
            hidden, settings["projector_hidden_size"], settings["projector_dropout"]
        )
        self.network["projector"] = self.projector
        v = settings["verification"]
        self.verification_buffer = VerificationBuffer(
            max_size=v["buffer_size"],
            non_overlap_gap=v["non_overlap_gap"],
            verification_capacity=v["cycle_size"],
        )

    def _set_stage(self, stage: str) -> None:
        self.stage = stage
        self.encoder.requires_grad_(stage == "stage_a")
        self.reconstruction_head.requires_grad_(stage != "online")
        self.classification_head.requires_grad_(stage != "online")
        if stage == "online":
            self.network.eval()
            self.optimizer = (
                None
                if self.projector is None
                else build_adam(
                    self.projector.parameters(),
                    self.config["method"]["update"]["optimizer"],
                )
            )
            return
        stages = self.config["method"]["training"]["stages"]
        definition = next(item for item in stages if item["name"] == stage)
        self.optimizer = build_adam(
            (p for p in self.network.parameters() if p.requires_grad),
            definition["optimizer"],
        )

    def _forward(
        self, x: torch.Tensor, *, stochastic: bool = False, generator=None
    ) -> dict:
        hidden = self.encoder(x)
        projected = self.projector(hidden) if self.projector is not None else hidden
        settings = self.source_parameters | (
            {"generator": self.rng if generator is None else generator}
            if stochastic
            else {}
        )
        branches = retrieve(
            projected, self.memory if self.stage != "stage_a" else {}, settings
        )
        return {
            **branches,
            "hidden": hidden,
            "projected": projected,
            "reconstruction": self.reconstruction_head(branches["continuous"]),
            "logits": self.classification_head(branches["discrete"].flatten(1)),
        }

    @torch.no_grad()
    def _initialize_memory(
        self, features: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor
    ) -> None:
        params = self.source_parameters
        epsilon = params["memory"]["normalization_epsilon"]
        normal = features[(labels[:, None] == 0) & ~masks.bool()]
        if params["continuous"]["enabled"]:
            self.memory["continuous"] = _kmeans(
                normal, params["continuous"]["num_prototypes"], epsilon
            )
        if not params["discrete"]["enabled"]:
            return
        count, fallback = params["discrete"]["codebook_size"], features.flatten(0, 1)
        centers, classes = [], []
        for code in range(12):
            group = features[labels == code].reshape(-1, features.shape[-1])
            size = count // 12 + int(code < count % 12)
            if size:
                centers.append(
                    _kmeans(group if len(group) else fallback, size, epsilon)
                )
                classes.extend([code] * size)
        self.memory["discrete"] = torch.cat(centers)
        self.memory["codeword_class_ids"] = torch.tensor(classes, device=self.device)
        self.memory["anomalous_mask"] = self.memory["codeword_class_ids"] != 0
        self._calibrate_radii(features[labels != 0].flatten(0, 1))

    def _calibrate_radii(self, anomaly_tokens: torch.Tensor) -> None:
        bank = self.memory["discrete"]
        radii = bank.new_zeros(len(bank))
        counts = torch.zeros(len(bank), dtype=torch.long, device=bank.device)
        if len(anomaly_tokens):
            nearest, distance = nearest_discrete_codeword(anomaly_tokens[None], bank)
            nearest, distance = nearest.flatten(), distance.flatten()
            counts = torch.bincount(nearest, minlength=len(bank))
            for index in nearest.unique().tolist():
                radii[index] = torch.quantile(
                    distance[nearest == index], 0.99
                ).clamp_min(0)
        self.memory.update(radii=radii, contributing_token_counts=counts)

    @torch.no_grad()
    def score(
        self, batch: dict, *, score_space=None, generator=None, uncertainty_fields=None
    ) -> dict:
        """Return unsmoothed point scores and optional triage/uncertainty outputs."""
        require_no_labels(batch, "MUSTARD")
        self.network.eval()
        window = to_device_tensor(batch["x"], self.device)
        evaluation = self.config["evaluation"]
        space = evaluation["score_space"] if score_space is None else score_space
        if space == "latent" and "continuous" not in self.memory:
            raise ValueError("latent MSE requires the continuous prototype bank")
        stochastic = self.source_parameters["stochastic_inference"]
        count = self.source_parameters["monte_carlo_samples"] if stochastic else 1
        passes = [
            self._forward(window, stochastic=stochastic, generator=generator)
            for _ in range(count)
        ]
        if space == "latent":
            samples = torch.stack([item["continuous"] for item in passes], dim=1)
            errors = (samples - passes[0]["query"][:, None]).square().mean(dim=-1)
        else:
            samples = torch.stack([item["reconstruction"] for item in passes], dim=1)
            errors = point_errors(window, samples, space, self.config.get("scaler"))
        result = build_score_record(
            batch, errors.mean(dim=1).cpu().numpy(), score_kind="window"
        )
        if self.variant in {"A1", "A2"}:
            result["auxiliary"] = self._triage_scores(window, passes)
        fields = (
            evaluation.get("uncertainty_fields")
            if uncertainty_fields is None
            else uncertainty_fields
        )
        if fields:
            result["uncertainty"] = self._uncertainty(passes, errors, fields)
        return result

    def _uncertainty(
        self, passes: list[dict], errors: torch.Tensor, requested: list[str]
    ) -> dict:
        return uncertainty(
            passes, errors, requested, self.source_parameters["variance_correction"]
        )

    def _triage_scores(self, window: torch.Tensor, passes: list[dict]) -> dict:
        recon = torch.stack([item["reconstruction"] for item in passes], dim=1)
        raw = point_errors(window, recon, "raw_input", self.config.get("scaler"))
        features = F.normalize(passes[0]["projected"], dim=-1)
        distance = 1 - features @ F.normalize(self.memory["continuous"], dim=-1).T
        return {
            "input_window_score": raw.mean(dim=(1, 2)).cpu().numpy(),
            "latent_window_score": distance.amin(dim=-1).mean(dim=-1).cpu().numpy(),
        }

    def update(self, batch: dict, context: dict) -> dict:
        """Apply an allowed test-time projector update."""
        if self.config["phase"] != "online":
            raise ValueError("MUSTARD updates require an online variant")
        return update_window(
            self, batch | {"x": to_device_tensor(batch["x"], self.device)}, context
        )

    def state_dict(self) -> dict:
        return pack_state(self)

    def load_state_dict(self, state: Mapping) -> None:
        restore_state(self, state)

    def _batches(self, split: dict):
        data = self.config["data"] | {"limits": self.config.get("limits", {})}
        return iter_batches(split, data, phase="offline", seed=self.config["seed"])

    def _augment(self, batch: dict, rng: torch.Generator, offset: int):
        clean = to_device_tensor(batch["x"], self.device)
        altered, labels = augment(
            clean,
            self.config["synthetic"] | {"class_rotation_offset": offset},
            generator=rng,
        )
        return clean, altered, labels

    @torch.no_grad()
    def _initialize_from_training(self, train: dict) -> None:
        features, labels, masks = [], [], []
        self.encoder.eval()
        for batch in islice(
            self._batches(train),
            self.source_parameters["memory"]["initialization_batches"],
        ):
            _, altered, supervision = self._augment(
                batch, self.rng, self.class_rotation_offset
            )
            self.class_rotation_offset = supervision["class_rotation_offset"]
            features.append(self.encoder(altered))
            labels.append(supervision["classification_labels"])
            masks.append(supervision["synthetic_anomaly_mask"])
        self._initialize_memory(
            torch.cat(features), torch.cat(labels), torch.cat(masks)
        )

    def _batch_loss(
        self,
        clean: torch.Tensor,
        altered: torch.Tensor,
        targets: dict,
        settings: dict,
        fitting: bool,
    ):
        with torch.set_grad_enabled(fitting):
            output = self._forward(altered)
            if self.variant == "O2" and self.stage == "stage_a":
                output["clean_features"] = self.encoder(clean)
            losses = compute_loss(
                output,
                targets | {"x": altered},
                self.source_parameters | {"variant": self.variant, "stage": self.stage},
            )
            if fitting:
                self.optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                if settings["gradient_clip_norm"] is not None:
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        settings["gradient_clip_norm"],
                        error_if_nonfinite=True,
                    )
                self.optimizer.step()
        return losses

    def _epoch(self, split: dict, settings: dict, *, fitting: bool) -> dict:
        self.network.train(fitting)
        if self.stage == "stage_b":
            self.encoder.eval()
        rng = (
            self.rng if fitting else torch.Generator().manual_seed(self.config["seed"])
        )
        offset = self.class_rotation_offset if fitting else 0
        total, count, batches, unavailable = 0.0, 0, 0, 0
        term_totals, term_counts = defaultdict(float), defaultdict(int)
        for batch in self._batches(split):
            clean, altered, targets = self._augment(batch, rng, offset)
            offset = targets["class_rotation_offset"]
            losses = self._batch_loss(clean, altered, targets, settings, fitting)
            total += float(losses["total"].detach()) * len(clean)
            for name, loss in losses.items():
                if name != "total":
                    term_totals[name] += float(loss.detach()) * len(clean)
                    term_counts[name] += len(clean)
            count, batches = count + len(clean), batches + 1
            if (
                self.variant == "O1"
                and self.stage == "stage_a"
                and "point_score" not in losses
            ):
                unavailable += 1
        if fitting:
            self.class_rotation_offset = offset
        if not count:
            raise ValueError("MUSTARD epoch requires at least one window")
        return {
            "total_loss": total / count,
            "loss_terms": {
                name: value / term_counts[name] for name, value in term_totals.items()
            },
            "loss_term_counts": dict(term_counts),
            "batch_count": batches,
            "point_score_unavailable_batches": unavailable,
        }

    def fit(
        self,
        train: dict,
        validation: dict,
        *,
        checkpoint_dir: Path,
        resume: dict | None = None,
        on_checkpoint=None,
        monitor_epoch=None,
    ) -> dict:
        """Fit both offline stages and retain each initial and best checkpoint."""
        if self.config["phase"] != "offline":
            raise ValueError("MUSTARD fitting requires an offline variant")
        identity = {
            "method": "mustard",
            "variant": self.variant,
            "shape": self.shape,
            "prepared_identity": self.config["prepared_identity"],
        }
        result = fit_neural_stages(
            self,
            self.config["method"]["training"]["stages"],
            checkpoint_dir=checkpoint_dir,
            identity=identity,
            run_epoch=lambda definition, _: {
                "epoch": _ + 1,
                "train": self._epoch(train, definition, fitting=True),
                "validation": self._epoch(validation, definition, fitting=False),
            },
            resume=resume,
            on_checkpoint=on_checkpoint,
            on_stage_start=lambda stage: (
                self._set_stage(stage),
                self._initialize_from_training(train) if stage == "stage_b" else None,
            ),
            monitor_epoch=monitor_epoch,
            save_best_per_epoch=True,
        )
        self.network.eval()
        return result
