"""Fit CANDI reference state and adapt its signed residual modules."""

from copy import deepcopy
from math import cos, pi
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..artifacts import save_checkpoint
from ..encoders import create_encoder
from .candi_adapter import SANA
from ..prepared_data import iter_batches


def _decoder(width, length, channels):
    size = length * channels
    if size < 4:
        raise ValueError("window decoder requires at least four cells")
    return nn.Sequential(nn.Linear(width, size // 4), nn.ReLU(),
                         nn.Linear(size // 4, size // 2), nn.ReLU(),
                         nn.Linear(size // 2, size))


class CANDI:
    """Keep the backbone frozen during online updates."""

    def __init__(self, config: dict, *, encoder=None):
        self.config = deepcopy(config)
        self.settings = self.config["method"]["parameters"]
        self.device = torch.device(config["device"])
        self.length = config["data"]["window_size"]
        self.channels = len(config["prepared_identity"]["features"])
        width = config["encoder"]["output_dim"]
        self.encoder = encoder if encoder is not None else create_encoder(
            config["encoder"], input_features=self.channels, window_size=self.length
        )
        self.decoder = _decoder(width, self.length, self.channels)
        self.backbone = nn.ModuleDict({"encoder": self.encoder, "decoder": self.decoder}).to(self.device)
        self.references = None
        self.pending = {"hard": [], "moderate": []}
        self.adapters = None
        self.optimizer = None
        if config["phase"] == "online":
            self._configure_online()

    def _configure_online(self):
        options = self.settings["sana"]
        if options["layers"] != 1:
            raise ValueError("CANDI supports one reviewed SANA layer")
        args = dict(input_dim=self.channels, window_size=self.length, d_model=options["hidden_size"],
                    n_heads=options["attention_heads"], d_ff=options["feedforward_size"],
                    dropout=options["dropout"], gating_init=options["gate_initialization"])
        self.adapters = nn.ModuleDict({"input_adapter": SANA(**args), "output_adapter": SANA(**args)}).to(self.device)
        self.backbone.requires_grad_(False).eval()
        options = self.config["method"]["update"]["optimizer"]
        self.optimizer = torch.optim.SGD(
            self.adapters.parameters(), lr=options["learning_rate"], momentum=options["momentum"],
            dampening=options["dampening"], weight_decay=options["weight_decay"], nesterov=options["nesterov"],
        )

    def _forward(self, x):
        adapted = x + self.adapters["input_adapter"](x) if self.adapters is not None else x
        hidden = self.encoder(adapted).mean(dim=1)
        reconstructed = self.decoder(hidden).reshape_as(x)
        if self.adapters is not None:
            reconstructed = reconstructed - self.adapters["output_adapter"](reconstructed)
        return reconstructed

    def _fit_references(self, scores, latents, quantile):
        from scipy.stats import chi2

        if len(latents) < 2:
            raise ValueError("CANDI covariance requires at least two eligible validation windows")
        covariance = np.atleast_2d(np.cov(latents, rowvar=False))
        inverse = np.linalg.pinv(covariance, rcond=self.settings["covariance_rcond"], hermitian=True)
        count = int(np.floor(len(scores) * (1 - quantile) + 1e-12))
        q1, q3 = np.quantile(scores, [0.25, 0.75])
        self.references = {
            "hard": latents[np.argsort(-scores, kind="stable")[:count]].copy(),
            "moderate": latents[(scores > q1) & (scores < q3)].copy(),
            "inverse": inverse, "threshold": float(np.quantile(scores, quantile)),
            "cutoff": float(chi2.ppf(self.settings["similarity_quantile"], df=latents.shape[1])),
        }

    def _select(self, score, latent):
        references = self.references
        if references is None:
            raise ValueError("CANDI requires saved validation references")
        threshold = references["threshold"]
        if score == threshold:
            return None
        group = "hard" if score > threshold else "moderate"
        if not len(references[group]):
            return None
        delta = latent - references[group]
        distances = np.einsum("ni,ij,nj->n", delta, references["inverse"], delta)
        return group if np.any(distances < references["cutoff"]) else None

    @torch.no_grad()
    def score(self, batch: dict) -> dict:
        if "labels" in batch or "point_labels" in batch:
            raise ValueError("CANDI scoring does not accept labels")
        self.backbone.eval()
        if self.adapters is not None:
            self.adapters.eval()
        x = torch.as_tensor(batch["x"], device=self.device, dtype=torch.float32)
        raw = (self._forward(x) - x).square().mean(dim=-1)
        result = {"score_kind": "window", "raw_scores": raw.cpu().numpy()}
        result.update({key: batch[key] for key in ("indices", "segment_ids", "identity") if key in batch})
        if self.adapters is not None:
            latent = F.normalize(self.encoder(x).mean(1), dim=-1, eps=self.settings["latent_normalization_epsilon"])
            result["auxiliary"] = {"window_score": raw.mean(dim=1).cpu().numpy(), "latent": latent.cpu().numpy()}
        return result

    def update(self, batch: dict, context: dict) -> dict:
        if "labels" in batch or "point_labels" in batch:
            raise ValueError("CANDI adaptation does not accept labels")
        if self.optimizer is None or "recorded_score" not in context:
            raise ValueError("CANDI adaptation requires an online optimizer and a recorded score")
        group = self._select(float(context["recorded_score"]), np.asarray(context["latent"]))
        if group is None:
            return {"did_update": False, "loss": None, "group": None}
        x = torch.as_tensor(batch["x"], device=self.device, dtype=torch.float32)
        self.pending[group].append(x.detach().clone())
        if len(self.pending[group]) < self.settings["minimum_samples"]:
            return {"did_update": False, "loss": None, "group": group}
        selected = torch.cat(self.pending[group])
        self.adapters.train()
        config = self.config["method"]["update"]
        for _ in range(config["steps"]):
            self.optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(self._forward(selected), selected)
            loss.backward()
            if config["gradient_clip_norm"] is not None:
                nn.utils.clip_grad_norm_(self.adapters.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True)
            self.optimizer.step()
        self.pending[group].clear()
        return {"did_update": True, "loss": float(loss.detach()), "group": group}

    def state_dict(self):
        return deepcopy({"schema_version": 1, "method": "candi", "backbone": self.backbone.state_dict(),
                         "references": self.references, "pending": self.pending,
                         "adapters": self.adapters.state_dict() if self.adapters is not None else None,
                         "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None})

    def load_state_dict(self, state):
        if state["schema_version"] != 1 or state["method"] != "candi":
            raise ValueError("CANDI checkpoint identity mismatch")
        self.backbone.load_state_dict(state["backbone"])
        self.references = deepcopy(state["references"])
        self.pending = {key: [value.to(self.device).clone() for value in queue]
                        for key, queue in state["pending"].items()}
        if state["adapters"] is not None:
            if self.adapters is None:
                raise ValueError("CANDI cannot restore online state into its offline model")
            self.adapters.load_state_dict(state["adapters"])
            self.optimizer.load_state_dict(state["optimizer"])

    def _batches(self, split):
        data = self.config["data"] | {"limits": self.config.get("limits", {})}
        return iter_batches(split, data, phase="offline", seed=self.config["seed"])

    def _epoch(self, split, optimizer=None):
        self.backbone.train(optimizer is not None)
        total = count = 0
        for batch in self._batches(split):
            x = torch.as_tensor(batch["x"], device=self.device, dtype=torch.float32)
            with torch.set_grad_enabled(optimizer is not None):
                loss = F.mse_loss(self._forward(x), x)
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    clip = self.config["method"]["training"]["stages"][0]["gradient_clip_norm"]
                    if clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), clip, error_if_nonfinite=True)
                    optimizer.step()
            total += float(loss.detach()) * len(x)
            count += len(x)
        return total / count

    @torch.no_grad()
    def _validation_references(self, validation):
        self.backbone.eval()
        scores, vectors = [], []
        for batch in self._batches(validation):
            labels = np.asarray(batch.get("labels", np.full(batch["indices"].shape, np.nan)))
            eligible = ~(labels == 1).any(axis=1)
            if eligible.any():
                x = torch.as_tensor(batch["x"][eligible], device=self.device, dtype=torch.float32)
                scores.append((self._forward(x) - x).square().mean(dim=(1, 2)).cpu().numpy())
                hidden = F.normalize(self.encoder(x).mean(1), dim=-1, eps=self.settings["latent_normalization_epsilon"])
                vectors.append(hidden.cpu().numpy())
        if not scores:
            raise ValueError("CANDI requires eligible validation windows")
        self._fit_references(np.concatenate(scores), np.concatenate(vectors),
                             self.config["evaluation"]["offline_threshold_quantile"])

    def fit(self, train: dict, validation: dict, *, checkpoint_dir: Path, resume=None,
            on_checkpoint=None) -> dict:
        if self.config["phase"] != "offline":
            raise ValueError("CANDI fitting requires an offline run")
        stage = self.config["method"]["training"]["stages"][0]
        settings = stage["optimizer"]
        optimizer = torch.optim.Adam(
            self.backbone.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
            betas=tuple(settings["betas"]), eps=settings["epsilon"],
        )
        identity = {"method": "candi", "prepared_identity": self.config["prepared_identity"]}
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
            optimizer.param_groups[0]["lr"] = _learning_rate(epoch, stage)
            train_loss = self._epoch(train, optimizer)
            validation_loss = self._epoch(validation)
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation_loss})
            if validation_loss < best_loss:
                best_loss, best = validation_loss, self.state_dict()
            if on_checkpoint is not None:
                on_checkpoint({"method_state": self.state_dict(), "optimizer_state": optimizer.state_dict(),
                               "initial": initial, "history": list(history), "next_epoch": epoch + 1,
                               "best": best, "best_loss": best_loss})
        if best is None:
            raise ValueError("CANDI has no finite best validation loss")
        self.load_state_dict(best)
        self._validation_references(validation)
        selected = save_checkpoint(checkpoint_dir / f"fit.best.{uuid4().hex}.pt", self.state_dict(),
                                   identity=identity, role="fit.best")
        return {"selected_checkpoint": selected, "stages": [{"name": "fit", "initial_checkpoint": initial,
                "best_checkpoint": selected, "best_validation_loss": best_loss, "epochs": history}]}


def _learning_rate(epoch, stage):
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
