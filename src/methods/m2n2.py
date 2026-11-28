"""M2N2 model computation flow: detrended reconstruction and predicted-normal updates."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..encoders import create_encoder
from ..method_runtime import (
    build_adam,
    build_score_record,
    fit_neural_stages,
    require_no_labels,
    to_device_tensor,
)
from ..prepared_data import iter_batches


def _window_decoder(width: int, length: int, channels: int) -> nn.Sequential:
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


class M2N2:
    """Own the backbone, causal trend, and training-derived threshold."""

    def __init__(self, config: dict, *, encoder=None):
        self.config = deepcopy(config)
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
        self.decoder = _window_decoder(width, self.length, self.channels)
        self.network = nn.ModuleDict(
            {"encoder": self.encoder, "decoder": self.decoder}
        ).to(self.device)
        self.trend = torch.zeros(self.channels, device=self.device)
        self.adaptation_threshold = None
        self.optimizer = None
        if config["phase"] == "online":
            opt = config["method"]["update"]["optimizer"]
            self.optimizer = torch.optim.SGD(
                self.network.parameters(),
                lr=opt["learning_rate"],
                weight_decay=opt["weight_decay"],
                momentum=opt["momentum"],
                dampening=opt["dampening"],
                nesterov=opt["nesterov"],
            )

    def _errors(self, x: torch.Tensor) -> torch.Tensor:
        code = self.encoder(x - self.trend).mean(1)
        reconstruction = self.decoder(code).reshape_as(x) + self.trend
        return (reconstruction - x).square().mean(dim=-1)

    @torch.no_grad()
    def score(self, batch: dict) -> dict:
        require_no_labels(batch, "M2N2")
        self.network.eval()
        x = to_device_tensor(batch["x"], self.device)
        if self.config["phase"] == "online":
            if len(x) != 1:
                raise ValueError("M2N2 online scoring requires one arrived window")
            self.trend.lerp_(
                x.mean(dim=(0, 1)),
                1 - self.config["method"]["parameters"]["trend_decay"],
            )
        return build_score_record(
            batch, self._errors(x).cpu().numpy(), score_kind="window"
        )

    def update(self, batch: dict, context: dict) -> dict:
        require_no_labels(batch, "M2N2")
        if self.optimizer is None or self.adaptation_threshold is None:
            raise ValueError(
                "M2N2 needs a fitted training threshold and online optimizer"
            )
        x = to_device_tensor(batch["x"], self.device)
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
            if settings["gradient_clip_norm"] is not None:
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    settings["gradient_clip_norm"],
                    error_if_nonfinite=True,
                )
            self.optimizer.step()
        return {"did_update": True, "loss": float(loss.detach()), "normal_count": count}

    def state_dict(self) -> dict:
        return deepcopy(
            {
                "schema_version": 1,
                "method": "m2n2",
                "network": self.network.state_dict(),
                "trend": self.trend,
                "adaptation_threshold": self.adaptation_threshold,
                "optimizer": None
                if self.optimizer is None
                else self.optimizer.state_dict(),
            }
        )

    def load_state_dict(self, state: dict) -> None:
        if state["schema_version"] != 1 or state["method"] != "m2n2":
            raise ValueError("M2N2 checkpoint identity mismatch")
        self.network.load_state_dict(state["network"])
        self.trend = state["trend"].to(self.device).clone()
        self.adaptation_threshold = state["adaptation_threshold"]
        if state["optimizer"] is not None:
            if self.optimizer is None:
                raise ValueError(
                    "M2N2 cannot restore online optimizer into an offline run"
                )
            self.optimizer.load_state_dict(state["optimizer"])

    def _epoch(self, split: dict, optimizer=None) -> float:
        self.network.train(optimizer is not None)
        data = self.config["data"] | {"limits": self.config.get("limits", {})}
        total = count = 0
        for batch in iter_batches(
            split, data, phase="offline", seed=self.config["seed"]
        ):
            x = to_device_tensor(batch["x"], self.device)
            with torch.set_grad_enabled(optimizer is not None):
                loss = self._errors(x).mean()
                if optimizer is not None:
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
            total += float(loss.detach()) * len(x)
            count += len(x)
        return total / count

    @torch.no_grad()
    def _fit_threshold(self, train: dict) -> None:
        evaluation = self.config["evaluation"]
        if (
            evaluation.get("adaptation_threshold_split") != "train"
            or train["identity"]["split"] != "train"
        ):
            raise ValueError("M2N2 adaptation threshold requires the training split")
        self.network.eval()
        data = self.config["data"] | {"limits": self.config.get("limits", {})}
        scores = [
            self._errors(to_device_tensor(batch["x"], self.device))
            .flatten()
            .cpu()
            .numpy()
            for batch in iter_batches(
                train, data, phase="offline", seed=self.config["seed"]
            )
        ]
        self.adaptation_threshold = float(
            np.quantile(np.concatenate(scores), evaluation["online_threshold_quantile"])
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
            raise ValueError("M2N2 fitting requires an offline run")
        stage = self.config["method"]["training"]["stages"][0]
        optimizer = build_adam(
            self.network.parameters(), stage["optimizer"], adamw=True
        )
        if resume:
            optimizer.load_state_dict(resume["optimizer_state"])
        identity = {
            "method": "m2n2",
            "prepared_identity": self.config["prepared_identity"],
        }
        return fit_neural_stages(
            self,
            [stage | {"name": "fit"}],
            checkpoint_dir=checkpoint_dir,
            identity=identity,
            run_epoch=lambda _, epoch: {
                "epoch": epoch + 1,
                "train_loss": self._epoch(train, optimizer),
                "validation": {"total_loss": self._epoch(validation)},
            },
            resume=resume,
            on_checkpoint=None
            if on_checkpoint is None
            else lambda s: on_checkpoint(
                s | {"optimizer_state": optimizer.state_dict()}
            ),
            post_fit=lambda: self._fit_threshold(train),
        )
