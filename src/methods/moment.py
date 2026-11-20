"""Use the pinned MOMENT model for zero-shot scoring or linear probing."""

from copy import deepcopy
from hashlib import file_digest
from pathlib import Path
import re
from uuid import uuid4

import torch

from ..artifacts import save_checkpoint
from ..prepared_data import iter_batches


def _pretrained_files(settings, project):
    from huggingface_hub import snapshot_download

    name, revision = settings["checkpoint"], settings["revision"]
    supplied = project / name
    if supplied.is_dir():
        return supplied
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("MOMENT requires an immutable weight revision")
    return Path(snapshot_download(
        name, revision=revision, allow_patterns=["model.safetensors", "config.json"],
        cache_dir=project / "outputs/model_cache",
    ))


def _sha256(path):
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
        root = Path(config["project_root"])
        directory = _pretrained_files(settings, root)
        self.pretrained_identity = {
            "checkpoint": settings["checkpoint"], "revision": settings["revision"],
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

    def _reconstruct(self, x):
        if x.ndim != 3 or x.shape[1] != self.length:
            raise ValueError("MOMENT requires input shape [batch, 512, features]")
        mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.long)
        scores = self.model.reconstruct(x_enc=x.transpose(1, 2), input_mask=mask, mask=mask)
        return scores.reconstruction.transpose(1, 2)

    @torch.no_grad()
    def score(self, batch: dict) -> dict:
        """Return channel-mean reconstruction error for each observed point."""
        if "labels" in batch or "point_labels" in batch:
            raise ValueError("MOMENT scoring does not accept labels")
        self.model.eval()
        x = torch.as_tensor(batch["x"], device=self.device, dtype=torch.float32)
        scores = (x - self._reconstruct(x)).square().mean(dim=-1).cpu().numpy()
        record = {"score_kind": "window", "raw_scores": scores}
        record.update({key: batch[key] for key in ("indices", "segment_ids", "identity") if key in batch})
        return record

    def state_dict(self) -> dict:
        head = self.model.head.state_dict() if self.variant == "linear_probing" else None
        return deepcopy({"schema_version": 1, "method": "moment", "variant": self.variant,
                         "pretrained_identity": self.pretrained_identity, "head": head})

    def load_state_dict(self, state):
        if (state["schema_version"], state["method"], state["variant"], state["pretrained_identity"]) != (
            1, "moment", self.variant, self.pretrained_identity
        ):
            raise ValueError("MOMENT checkpoint identity mismatch")
        if self.variant == "linear_probing":
            self.model.head.load_state_dict(state["head"])

    def _epoch(self, split, optimizer=None):
        self.model.eval()
        if optimizer is not None:
            self.model.head.train()
        data = self.config["data"] | {"limits": self.config.get("limits", {})}
        total = count = 0
        for batch in iter_batches(split, data, phase="offline", seed=self.config["seed"]):
            x = torch.as_tensor(batch["x"], device=self.device, dtype=torch.float32)
            with torch.set_grad_enabled(optimizer is not None):
                loss = (x - self._reconstruct(x)).square().mean()
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    clip = self.config["method"]["training"]["stages"][0]["gradient_clip_norm"]
                    if clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.head.parameters(), clip, error_if_nonfinite=True)
                    optimizer.step()
            total += float(loss.detach()) * len(x)
            count += len(x)
        return total / count

    def fit(self, train: dict, validation: dict, *, checkpoint_dir: Path, resume=None,
            on_checkpoint=None) -> dict:
        """Train only the head and select it by validation reconstruction loss."""
        if self.variant != "linear_probing" or self.config["phase"] != "offline":
            raise ValueError("MOMENT fitting requires offline linear probing")
        stage = self.config["method"]["training"]["stages"][0]
        options = stage["optimizer"]
        optimizer = torch.optim.Adam(self.model.head.parameters(), lr=options["learning_rate"],
                                     weight_decay=options["weight_decay"], betas=tuple(options["betas"]),
                                     eps=options["epsilon"])
        identity = {"method": "moment", "variant": self.variant, "pretrained_identity": self.pretrained_identity}
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
            raise ValueError("MOMENT has no finite best validation loss")
        self.load_state_dict(best)
        selected = save_checkpoint(checkpoint_dir / f"fit.best.{uuid4().hex}.pt", self.state_dict(),
                                   identity=identity, role="fit.best")
        return {"selected_checkpoint": selected, "stages": [{"name": "fit", "initial_checkpoint": initial,
                "best_checkpoint": selected, "best_validation_loss": best_loss, "epochs": history}]}
