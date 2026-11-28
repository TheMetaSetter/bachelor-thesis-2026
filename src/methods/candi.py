"""CANDI model computation flow: shared-CNN backbone, validation references, and SANA updates."""

from copy import deepcopy
from math import cos, pi
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..encoders import create_encoder
from ..method_runtime import (
    build_adam,
    build_score_record,
    eligible_validation_batches,
    fit_neural_stages,
    require_no_labels,
    to_device_tensor,
)
from ..prepared_data import iter_batches
from .candi_adapter import SANA


def _decoder(width: int, length: int, channels: int) -> nn.Sequential:
    size = length * channels
    if size < 4:
        raise ValueError("window decoder requires at least four cells")
    return nn.Sequential(
        nn.Linear(width, size // 4),
        nn.ReLU(),
        nn.Linear(size // 4, size // 2),
        nn.ReLU(),
        nn.Linear(size // 2, size),
    )


def _learning_rate(epoch: int, stage: dict) -> float:
    peak = stage["optimizer"]["learning_rate"]
    schedule = stage["scheduler"]
    if schedule is None:
        return peak
    warmup = schedule["warmup_epochs"]
    if epoch < warmup:
        return peak * (epoch + 1) / warmup
    minimum = schedule["minimum_learning_rate"]
    progress = (epoch - warmup) / (stage["epochs"] - warmup)
    return minimum + (peak - minimum) * (1 + cos(pi * progress)) / 2


class CANDI:
    """Keep the backbone frozen during online SANA updates."""

    def __init__(self, config: dict, *, encoder=None):
        self.config = deepcopy(config)
        self.settings = self.config["method"]["parameters"]
        self.device = torch.device(config["device"])
        self.length = config["data"]["window_size"]
        self.channels = len(config["prepared_identity"]["features"])
        width = config["encoder"]["output_dim"]
        self.encoder = (
            encoder
            if encoder is not None
            else create_encoder(
                config["encoder"], input_features=self.channels, window_size=self.length
            )
        )
        self.decoder = _decoder(width, self.length, self.channels)
        self.backbone = nn.ModuleDict(
            {"encoder": self.encoder, "decoder": self.decoder}
        ).to(self.device)
        self.references = None
        self.pending = {"hard": [], "moderate": []}
        self.adapters = None
        self.optimizer = None
        if config["phase"] == "online":
            self._configure_online()

    def _configure_online(self) -> None:
        options = self.settings["sana"]
        if options["layers"] != 1:
            raise ValueError("CANDI supports one reviewed SANA layer")
        args = dict(
            input_dim=self.channels,
            window_size=self.length,
            d_model=options["hidden_size"],
            n_heads=options["attention_heads"],
            d_ff=options["feedforward_size"],
            dropout=options["dropout"],
            gating_init=options["gate_initialization"],
        )
        self.adapters = nn.ModuleDict(
            {"input_adapter": SANA(**args), "output_adapter": SANA(**args)}
        ).to(self.device)
        self.backbone.requires_grad_(False).eval()
        opt = self.config["method"]["update"]["optimizer"]
        self.optimizer = torch.optim.SGD(
            self.adapters.parameters(),
            lr=opt["learning_rate"],
            momentum=opt["momentum"],
            dampening=opt["dampening"],
            weight_decay=opt["weight_decay"],
            nesterov=opt["nesterov"],
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        adapted = (
            x + self.adapters["input_adapter"](x) if self.adapters is not None else x
        )
        hidden = self.encoder(adapted).mean(dim=1)
        reconstructed = self.decoder(hidden).reshape_as(x)
        if self.adapters is not None:
            reconstructed = reconstructed - self.adapters["output_adapter"](
                reconstructed
            )
        return reconstructed

    def _fit_references(
        self, scores: np.ndarray, latents: np.ndarray, quantile: float
    ) -> None:
        from scipy.stats import chi2

        if len(latents) < 2:
            raise ValueError(
                "CANDI covariance requires at least two eligible validation windows"
            )
        covariance = np.atleast_2d(np.cov(latents, rowvar=False))
        inverse = np.linalg.pinv(
            covariance, rcond=self.settings["covariance_rcond"], hermitian=True
        )
        count = int(np.floor(len(scores) * (1 - quantile) + 1e-12))
        q1, q3 = np.quantile(scores, [0.25, 0.75])
        self.references = {
            "hard": latents[np.argsort(-scores, kind="stable")[:count]].copy(),
            "moderate": latents[(scores > q1) & (scores < q3)].copy(),
            "inverse": inverse,
            "threshold": float(np.quantile(scores, quantile)),
            "cutoff": float(
                chi2.ppf(self.settings["similarity_quantile"], df=latents.shape[1])
            ),
        }

    def _select(self, score: float, latent: np.ndarray) -> str | None:
        if self.references is None:
            raise ValueError("CANDI requires saved validation references")
        threshold = self.references["threshold"]
        if score == threshold:
            return None
        group = "hard" if score > threshold else "moderate"
        if not len(self.references[group]):
            return None
        delta = latent - self.references[group]
        distances = np.einsum("ni,ij,nj->n", delta, self.references["inverse"], delta)
        return group if np.any(distances < self.references["cutoff"]) else None

    @torch.no_grad()
    def score(self, batch: dict) -> dict:
        require_no_labels(batch, "CANDI")
        self.backbone.eval()
        if self.adapters is not None:
            self.adapters.eval()
        x = to_device_tensor(batch["x"], self.device)
        raw = (self._forward(x) - x).square().mean(dim=-1)
        result = build_score_record(batch, raw.cpu().numpy(), score_kind="window")
        if self.adapters is not None:
            latent = F.normalize(
                self.encoder(x).mean(1),
                dim=-1,
                eps=self.settings["latent_normalization_epsilon"],
            )
            result["auxiliary"] = {
                "window_score": raw.mean(dim=1).cpu().numpy(),
                "latent": latent.cpu().numpy(),
            }
        return result

    def update(self, batch: dict, context: dict) -> dict:
        require_no_labels(batch, "CANDI")
        if self.optimizer is None or "recorded_score" not in context:
            raise ValueError(
                "CANDI adaptation requires an online optimizer and a recorded score"
            )
        group = self._select(
            float(context["recorded_score"]), np.asarray(context["latent"])
        )
        if group is None:
            return {"did_update": False, "loss": None, "group": None}
        self.pending[group].append(
            to_device_tensor(batch["x"], self.device).detach().clone()
        )
        if len(self.pending[group]) < self.settings["minimum_samples"]:
            return {"did_update": False, "loss": None, "group": group}
        selected = torch.cat(self.pending[group])
        self.adapters.train()
        cfg = self.config["method"]["update"]
        for _ in range(cfg["steps"]):
            self.optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(self._forward(selected), selected)
            loss.backward()
            if cfg["gradient_clip_norm"] is not None:
                nn.utils.clip_grad_norm_(
                    self.adapters.parameters(),
                    cfg["gradient_clip_norm"],
                    error_if_nonfinite=True,
                )
            self.optimizer.step()
        self.pending[group].clear()
        return {"did_update": True, "loss": float(loss.detach()), "group": group}

    def state_dict(self) -> dict:
        return deepcopy(
            {
                "schema_version": 1,
                "method": "candi",
                "backbone": self.backbone.state_dict(),
                "references": self.references,
                "pending": self.pending,
                "adapters": None
                if self.adapters is None
                else self.adapters.state_dict(),
                "optimizer": None
                if self.optimizer is None
                else self.optimizer.state_dict(),
            }
        )

    def load_state_dict(self, state: dict) -> None:
        if state["schema_version"] != 1 or state["method"] != "candi":
            raise ValueError("CANDI checkpoint identity mismatch")
        self.backbone.load_state_dict(state["backbone"])
        self.references = deepcopy(state["references"])
        self.pending = {
            k: [v.to(self.device).clone() for v in q]
            for k, q in state["pending"].items()
        }
        if state["adapters"] is not None:
            if self.adapters is None:
                raise ValueError(
                    "CANDI cannot restore online state into its offline model"
                )
            self.adapters.load_state_dict(state["adapters"])
            self.optimizer.load_state_dict(state["optimizer"])

    def _epoch(self, split: dict, optimizer=None) -> float:
        self.backbone.train(optimizer is not None)
        data = self.config["data"] | {"limits": self.config.get("limits", {})}
        total = count = 0
        for batch in iter_batches(
            split, data, phase="offline", seed=self.config["seed"]
        ):
            x = to_device_tensor(batch["x"], self.device)
            with torch.set_grad_enabled(optimizer is not None):
                loss = F.mse_loss(self._forward(x), x)
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    clip = self.config["method"]["training"]["stages"][0][
                        "gradient_clip_norm"
                    ]
                    if clip is not None:
                        nn.utils.clip_grad_norm_(
                            self.backbone.parameters(), clip, error_if_nonfinite=True
                        )
                    optimizer.step()
            total += float(loss.detach()) * len(x)
            count += len(x)
        return total / count

    @torch.no_grad()
    def _validation_references(self, validation: dict) -> None:
        self.backbone.eval()
        scores, vectors = [], []
        for values in eligible_validation_batches(validation, self.config):
            x = to_device_tensor(values, self.device)
            scores.append(
                (self._forward(x) - x).square().mean(dim=(1, 2)).cpu().numpy()
            )
            hidden = F.normalize(
                self.encoder(x).mean(1),
                dim=-1,
                eps=self.settings["latent_normalization_epsilon"],
            )
            vectors.append(hidden.cpu().numpy())
        if not scores:
            raise ValueError("CANDI requires eligible validation windows")
        self._fit_references(
            np.concatenate(scores),
            np.concatenate(vectors),
            self.config["evaluation"]["offline_threshold_quantile"],
        )

    def fit(
        self,
        train: dict,
        validation: dict,
        *,
        checkpoint_dir: Path,
        resume=None,
        on_checkpoint=None,
    ) -> dict:
        if self.config["phase"] != "offline":
            raise ValueError("CANDI fitting requires an offline run")
        stage = self.config["method"]["training"]["stages"][0]
        optimizer = build_adam(self.backbone.parameters(), stage["optimizer"])
        if resume:
            optimizer.load_state_dict(resume["optimizer_state"])
        identity = {
            "method": "candi",
            "prepared_identity": self.config["prepared_identity"],
        }
        return fit_neural_stages(
            self,
            [stage | {"name": "fit"}],
            checkpoint_dir=checkpoint_dir,
            identity=identity,
            run_epoch=lambda definition, epoch: (
                optimizer.param_groups[0].__setitem__(
                    "lr", _learning_rate(epoch, definition)
                ),
                {
                    "epoch": epoch + 1,
                    "train_loss": self._epoch(train, optimizer),
                    "validation": {"total_loss": self._epoch(validation)},
                },
            )[1],
            resume=resume,
            on_checkpoint=None
            if on_checkpoint is None
            else lambda s: on_checkpoint(
                s | {"optimizer_state": optimizer.state_dict()}
            ),
            post_fit=lambda: self._validation_references(validation),
        )
