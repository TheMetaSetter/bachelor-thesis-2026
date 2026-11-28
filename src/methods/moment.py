"""Use the pinned MOMENT model for zero-shot scoring or linear probing."""

from copy import deepcopy
from hashlib import file_digest
from pathlib import Path
import re

import torch

from ..method_runtime import (
    build_adam,
    build_score_record,
    fit_neural_stages,
    require_no_labels,
    to_device_tensor,
)
from ..prepared_data import iter_batches


def _pretrained_files(settings: dict, project: Path) -> Path:
    from huggingface_hub import snapshot_download

    name, revision = settings["checkpoint"], settings["revision"]
    supplied = project / name
    if supplied.is_dir():
        return supplied
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("MOMENT requires an immutable weight revision")
    return Path(
        snapshot_download(
            name,
            revision=revision,
            allow_patterns=["model.safetensors", "config.json"],
            cache_dir=project / "outputs/model_cache",
        )
    )


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


class Moment:
    """Own pretrained identity and the optional fitted reconstruction head."""

    def __init__(self, config: dict, *, encoder=None):
        from momentfm import MOMENTPipeline

        if encoder is not None:
            raise ValueError("MOMENT uses its pretrained encoder")
        self.config = deepcopy(config)
        self.variant = config["variant"]
        self.device = torch.device(config["device"])
        self.length = config["data"]["window_size"]
        settings = config["method"]["parameters"]
        if self.length != 512 or settings["patch_length"] != 8:
            raise ValueError("MOMENT-1-base requires length 512 and patch length eight")
        directory = _pretrained_files(settings, Path(config["project_root"]))
        self.pretrained_identity = {
            "checkpoint": settings["checkpoint"],
            "revision": settings["revision"],
            "weights_sha256": _sha256(directory / "model.safetensors"),
            "config_sha256": _sha256(directory / "config.json"),
        }
        self.model = MOMENTPipeline.from_pretrained(
            str(directory), model_kwargs={"task_name": "reconstruction"}, strict=True
        )
        self.model.init()
        self.model.to(self.device).requires_grad_(False).eval()
        if self.variant == "linear_probing" and config["phase"] == "offline":
            self.model.head.requires_grad_(True)

    def _reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.length:
            raise ValueError("MOMENT requires input shape [batch, 512, features]")
        mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.long)
        scores = self.model.reconstruct(
            x_enc=x.transpose(1, 2), input_mask=mask, mask=mask
        )
        return scores.reconstruction.transpose(1, 2)

    @torch.no_grad()
    def score(self, batch: dict) -> dict:
        """Return channel-mean reconstruction error for each observed point."""
        require_no_labels(batch, "MOMENT")
        self.model.eval()
        x = to_device_tensor(batch["x"], self.device)
        scores = (x - self._reconstruct(x)).square().mean(dim=-1).cpu().numpy()
        return build_score_record(batch, scores, score_kind="window")

    def state_dict(self) -> dict:
        head = (
            self.model.head.state_dict() if self.variant == "linear_probing" else None
        )
        return deepcopy(
            {
                "schema_version": 1,
                "method": "moment",
                "variant": self.variant,
                "pretrained_identity": self.pretrained_identity,
                "head": head,
            }
        )

    def load_state_dict(self, state: dict) -> None:
        if (
            state["schema_version"],
            state["method"],
            state["variant"],
            state["pretrained_identity"],
        ) != (1, "moment", self.variant, self.pretrained_identity):
            raise ValueError("MOMENT checkpoint identity mismatch")
        if self.variant == "linear_probing":
            self.model.head.load_state_dict(state["head"])

    def _epoch(self, split: dict, optimizer=None) -> float:
        self.model.eval()
        if optimizer is not None:
            self.model.head.train()
        data = self.config["data"] | {"limits": self.config.get("limits", {})}
        total = count = 0
        for batch in iter_batches(
            split, data, phase="offline", seed=self.config["seed"]
        ):
            x = to_device_tensor(batch["x"], self.device)
            with torch.set_grad_enabled(optimizer is not None):
                loss = (x - self._reconstruct(x)).square().mean()
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    clip = self.config["method"]["training"]["stages"][0][
                        "gradient_clip_norm"
                    ]
                    if clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.head.parameters(), clip, error_if_nonfinite=True
                        )
                    optimizer.step()
            total += float(loss.detach()) * len(x)
            count += len(x)
        return total / count

    def fit(
        self,
        train: dict,
        validation: dict,
        *,
        checkpoint_dir: Path,
        resume=None,
        on_checkpoint=None,
    ) -> dict:
        """Train only the head and select it by validation reconstruction loss."""
        if self.variant != "linear_probing" or self.config["phase"] != "offline":
            raise ValueError("MOMENT fitting requires offline linear probing")
        stage = self.config["method"]["training"]["stages"][0]
        optimizer = build_adam(self.model.head.parameters(), stage["optimizer"])
        if resume:
            optimizer.load_state_dict(resume["optimizer_state"])
        identity = {
            "method": "moment",
            "variant": self.variant,
            "pretrained_identity": self.pretrained_identity,
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
        )
