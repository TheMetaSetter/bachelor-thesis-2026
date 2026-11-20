"""Fit M2N2 and adapt to arrived windows with a predicted-normal mask."""

from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from torch import nn

from ..artifacts import save_checkpoint
from ..encoders import create_encoder
from ..prepared_data import iter_batches
from .candi import _decoder


class M2N2:
    """Own the backbone, causal trend, and training-derived threshold."""

    def __init__(self, config: dict, *, encoder=None):
        self.config = deepcopy(config)
        self.device = torch.device(config["device"])
        self.length = config["data"]["window_size"]
        self.channels = len(config["prepared_identity"]["features"])
        width = config["encoder"]["output_dim"]
        self.encoder = encoder if encoder is not None else create_encoder(
            config["encoder"], input_features=self.channels, window_size=self.length
        )
        self.decoder = _decoder(width, self.length, self.channels)
        self.network = nn.ModuleDict({"encoder": self.encoder, "decoder": self.decoder}).to(self.device)
        self.trend = torch.zeros(self.channels, device=self.device)
        self.adaptation_threshold = None
        self.optimizer = None
        if config["phase"] == "online":
            settings = config["method"]["update"]["optimizer"]
            self.optimizer = torch.optim.SGD(self.network.parameters(), lr=settings["learning_rate"],
                                             weight_decay=settings["weight_decay"], momentum=settings["momentum"],
                                             dampening=settings["dampening"], nesterov=settings["nesterov"])

    def _errors(self, x):
        code = self.encoder(x - self.trend).mean(1)
        reconstruction = self.decoder(code).reshape_as(x) + self.trend
        return (reconstruction - x).square().mean(dim=-1)

    @torch.no_grad()
    def score(self, batch: dict) -> dict:
        if "labels" in batch or "point_labels" in batch:
            raise ValueError("M2N2 scoring does not accept labels")
        self.network.eval()
        x = torch.as_tensor(batch["x"], device=self.device, dtype=torch.float32)
        if self.config["phase"] == "online":
            if len(x) != 1:
                raise ValueError("M2N2 online scoring requires one arrived window")
            self.trend.lerp_(x.mean(dim=(0, 1)), 1 - self.config["method"]["parameters"]["trend_decay"])
        result = {"score_kind": "window", "raw_scores": self._errors(x).cpu().numpy()}
        result.update({key: batch[key] for key in ("indices", "segment_ids", "identity") if key in batch})
        return result

    def update(self, batch: dict, context: dict) -> dict:
        if "labels" in batch or "point_labels" in batch:
            raise ValueError("M2N2 adaptation does not accept labels")
        if self.optimizer is None or self.adaptation_threshold is None:
            raise ValueError("M2N2 needs a fitted training threshold and online optimizer")
        x = torch.as_tensor(batch["x"], device=self.device, dtype=torch.float32)
        recorded = torch.as_tensor(context["recorded_scores"], device=self.device)
        if recorded.shape != x.shape[:2] or not torch.isfinite(recorded).all():
            raise ValueError("M2N2 update requires aligned and finite raw scores")
        normal = recorded < self.adaptation_threshold
        count = int(normal.sum())
        if not count:
            return {"did_update": False, "loss": None, "normal_count": 0}
        settings = self.config["method"]["update"]
        self.network.eval()
        for _ in range(settings["steps"]):
            self.optimizer.zero_grad(set_to_none=True)
            loss = (self._errors(x) * normal).mean()
            if not torch.isfinite(loss):
                raise ValueError("M2N2 adaptation loss must be finite")
            loss.backward()
            clip = settings["gradient_clip_norm"]
            if clip is not None:
                nn.utils.clip_grad_norm_(self.network.parameters(), clip, error_if_nonfinite=True)
            self.optimizer.step()
        return {"did_update": True, "loss": float(loss.detach()), "normal_count": count}

    def state_dict(self) -> dict:
        return deepcopy({"schema_version": 1, "method": "m2n2", "network": self.network.state_dict(),
                         "trend": self.trend, "adaptation_threshold": self.adaptation_threshold,
                         "optimizer": None if self.optimizer is None else self.optimizer.state_dict()})

    def load_state_dict(self, state):
        if state["schema_version"] != 1 or state["method"] != "m2n2":
            raise ValueError("M2N2 checkpoint identity mismatch")
        self.network.load_state_dict(state["network"])
        self.trend = state["trend"].to(self.device).clone()
        self.adaptation_threshold = state["adaptation_threshold"]
        if state["optimizer"] is not None:
            if self.optimizer is None:
                raise ValueError("M2N2 cannot restore online optimizer into an offline run")
            self.optimizer.load_state_dict(state["optimizer"])

    def _batches(self, split):
        data = self.config["data"] | {"limits": self.config.get("limits", {})}
        return iter_batches(split, data, phase="offline", seed=self.config["seed"])

    def _epoch(self, split, optimizer=None):
        self.network.train(optimizer is not None)
        total = count = 0
        for batch in self._batches(split):
            x = torch.as_tensor(batch["x"], device=self.device, dtype=torch.float32)
            with torch.set_grad_enabled(optimizer is not None):
                loss = self._errors(x).mean()
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    clip = self.config["method"]["training"]["stages"][0]["gradient_clip_norm"]
                    if clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.network.parameters(), clip, error_if_nonfinite=True)
                    optimizer.step()
            total += float(loss.detach()) * len(x)
            count += len(x)
        return total / count

    @torch.no_grad()
    def _fit_threshold(self, train):
        evaluation = self.config["evaluation"]
        if evaluation.get("adaptation_threshold_split") != "train" or train["identity"]["split"] != "train":
            raise ValueError("M2N2 adaptation threshold requires the training split")
        self.network.eval()
        scores = []
        for batch in self._batches(train):
            values = torch.as_tensor(batch["x"], device=self.device, dtype=torch.float32)
            scores.append(self._errors(values).flatten().cpu().numpy())
        self.adaptation_threshold = float(np.quantile(np.concatenate(scores), evaluation["online_threshold_quantile"]))

    def fit(self, train: dict, validation: dict, *, checkpoint_dir: Path, resume=None,
            on_checkpoint=None) -> dict:
        if self.config["phase"] != "offline":
            raise ValueError("M2N2 fitting requires an offline run")
        stage = self.config["method"]["training"]["stages"][0]
        options = stage["optimizer"]
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=options["learning_rate"],
                                      weight_decay=options["weight_decay"], betas=tuple(options["betas"]),
                                      eps=options["epsilon"])
        identity = {"method": "m2n2", "prepared_identity": self.config["prepared_identity"]}
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
            raise ValueError("M2N2 has no finite best validation loss")
        self.load_state_dict(best)
        self._fit_threshold(train)
        selected = save_checkpoint(checkpoint_dir / f"fit.best.{uuid4().hex}.pt", self.state_dict(),
                                   identity=identity, role="fit.best")
        return {"selected_checkpoint": selected, "stages": [{"name": "fit", "initial_checkpoint": initial,
                "best_checkpoint": selected, "best_validation_loss": best_loss, "epochs": history}]}
