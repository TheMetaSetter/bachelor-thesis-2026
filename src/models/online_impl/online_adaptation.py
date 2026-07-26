"""Projector-first online adaptation model for the first accepted Phase 4 slice.

This file should be read after the offline multitask model. The online path is
deliberately conservative: it reuses the offline encoder geometry, keeps the
reference encoder frozen, and adapts only a small residual projector by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.console import (
    console_print,
    print_parameter_summary,
    summarize_batch,
    summarize_tensor,
)
from src.core.contracts import validate_model_outputs, validate_online_batch
from src.engine.online_tta.checkpoint_resolution import (
    resolve_legacy_reference_checkpoint_path,
)
from src.models.base_model import BaseModel
from src.models.thesis_multitask import ThesisMultitaskModel
from src.models.online_impl.online_adaptation_helpers import (
    NearIdentityMLPProjector,
    ThesisMultitaskEncoderAdapter,
)


def _resolve_reference_checkpoint_path(checkpoint_path: str | Path) -> Path:
    """Resolve legacy flat paths to the two-stage Stage-B checkpoint."""
    requested = Path(checkpoint_path)
    if requested.exists():
        return requested
    resolved = resolve_legacy_reference_checkpoint_path(requested)
    console_print(
        "MODEL",
        "Resolved legacy online reference checkpoint path",
        requested_path=requested,
        resolved_path=resolved,
    )
    return resolved

class OnlineAdaptationModel(BaseModel):
    def __init__(
        self,
        input_dim: int,
        encoder_dim: int,
        hidden_dim: int,
        projector_hidden_dim: int,
        projector_dropout: float,
        enable_prototype_alignment: bool,
        lambda_align: float,
        lambda_proto: float,
        lambda_anchor: float,
        score_source: str,
        reference_checkpoint_path: str,
        warm_start_projector: bool,
        target_param_group: str,
        clean_stream_only: bool,
        reset_policy: str = "disabled",
        reset_alignment_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        # The first online slice is intentionally narrow so that a new reader
        # can reason about one adaptation mechanism at a time.
        if not clean_stream_only:
            raise ValueError(
                "The first online adaptation slice supports only clean_stream_only=True"
            )
        if score_source != "projected_hidden":
            raise ValueError(
                "The first online adaptation slice supports only score_source='projected_hidden'"
            )

        self.input_dim = input_dim
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.enable_prototype_alignment = enable_prototype_alignment
        self.lambda_align = lambda_align
        self.lambda_proto = lambda_proto
        self.lambda_anchor = lambda_anchor
        self.score_source = score_source
        self.reference_checkpoint_path = str(reference_checkpoint_path)
        self.warm_start_projector = warm_start_projector
        self.target_param_group = target_param_group
        self.clean_stream_only = clean_stream_only
        self.reset_policy = reset_policy
        self.reset_alignment_threshold = reset_alignment_threshold
        self.alignment_temperature = 0.1

        # The offline multitask checkpoint is the source of truth for the
        # representation geometry used by both the reference and online encoders.
        frozen_multitask_model = self._load_reference_model(reference_checkpoint_path)
        self.reference_encoder = ThesisMultitaskEncoderAdapter(
            frozen_multitask_model, freeze_parameters=True
        )
        self.online_encoder = ThesisMultitaskEncoderAdapter(
            frozen_multitask_model, freeze_parameters=True
        )
        self.online_mlp_projector = NearIdentityMLPProjector(
            hidden_dim=hidden_dim,
            projector_hidden_dim=projector_hidden_dim,
            dropout=projector_dropout,
        )
        self.projector = self.online_mlp_projector
        self.projector_anchor_state_dict = self._clone_projector_state_dict()
        self._set_trainable_parameter_group(target_param_group)
        print_parameter_summary(
            "MODEL",
            "OnlineAdaptationModel",
            self,
            {
                "reference_encoder": self.reference_encoder,
                "online_encoder": self.online_encoder,
                "online_mlp_projector": self.online_mlp_projector,
            },
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            hidden_dim=hidden_dim,
            projector_hidden_dim=projector_hidden_dim,
            enable_prototype_alignment=enable_prototype_alignment,
            lambda_align=lambda_align,
            lambda_proto=lambda_proto,
            lambda_anchor=lambda_anchor,
            target_param_group=target_param_group,
            trainable_group_parameters=sum(
                parameter.numel()
                for parameter in self.get_parameter_group(target_param_group)
            ),
        )

    def _load_reference_model(
        self, checkpoint_path: str | Path
    ) -> ThesisMultitaskModel:
        # The online runtime is defined only for multitask checkpoints. Failing
        # early here prevents confusing baseline-versus-online mismatches later.
        resolved_checkpoint_path = _resolve_reference_checkpoint_path(checkpoint_path)
        loaded_checkpoint = torch.load(resolved_checkpoint_path, map_location="cpu")
        config = loaded_checkpoint["config"]
        model_name = config.get("model", {}).get("model_name")
        if model_name != "thesis_multitask":
            raise ValueError(
                "reference_checkpoint_path must point to a thesis_multitask checkpoint, "
                f"but found model_name={model_name!r}"
            )
        model_kwargs = {
            key: value for key, value in config["model"].items() if key != "model_name"
        }
        model_kwargs.update(
            {
                key: value
                for key, value in config.get("task", {}).items()
                if key != "task_name"
            }
        )
        reference_model = ThesisMultitaskModel(**model_kwargs)
        reference_model.load_state_dict(loaded_checkpoint["model_state_dict"])
        if hasattr(reference_model, "load_checkpoint_extra_state"):
            reference_model.load_checkpoint_extra_state(
                loaded_checkpoint.get("extra_state")
            )
        reference_model.eval()
        return reference_model

    def _clone_projector_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            parameter_name: parameter.detach().cpu().clone()
            for parameter_name, parameter in self.online_mlp_projector.state_dict().items()
        }

    def get_projector_anchor_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            parameter_name: parameter.clone()
            for parameter_name, parameter in self.projector_anchor_state_dict.items()
        }

    def load_projector_anchor_state_dict(
        self, state_dict: dict[str, torch.Tensor]
    ) -> None:
        self.projector_anchor_state_dict = {
            parameter_name: parameter.detach().cpu().clone()
            for parameter_name, parameter in state_dict.items()
        }

    def _parameters_for_target_group(
        self, target_param_group: str
    ) -> list[nn.Parameter]:
        if target_param_group == "projector_params":
            return list(self.online_mlp_projector.parameters())
        if target_param_group == "online_encoder_params":
            return self.online_encoder.encoder_parameters()
        raise ValueError(
            "target_param_group must be either 'projector_params' or 'online_encoder_params'"
        )

    def _set_trainable_parameter_group(self, target_param_group: str) -> None:
        # Parameter groups are explicit because the design docs treat the online
        # optimization boundary as part of the architecture, not a small detail.
        for parameter in self.reference_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.online_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.online_mlp_projector.parameters():
            parameter.requires_grad = False

        for parameter in self._parameters_for_target_group(target_param_group):
            parameter.requires_grad = True
        console_print(
            "MODEL",
            "Set trainable parameter group",
            target_param_group=target_param_group,
        )

    def get_parameter_group(self, target_param_group: str) -> list[nn.Parameter]:
        return self._parameters_for_target_group(target_param_group)

    def _replace_batch_x(
        self, batch: dict[str, Any], x_tensor: torch.Tensor
    ) -> dict[str, Any]:
        replaced_batch = dict(batch)
        replaced_batch["x"] = x_tensor
        return replaced_batch

    def _compute_alignment_loss(
        self,
        reference_hidden: torch.Tensor,
        projected_hidden: torch.Tensor,
    ) -> torch.Tensor:
        # Alignment compares pooled representations so the online path first
        # learns to match reference geometry before widening adaptation scope.
        pooled_reference = F.normalize(reference_hidden.mean(dim=1), dim=-1)
        pooled_projected = F.normalize(projected_hidden.mean(dim=1), dim=-1)
        similarity_logits = (
            pooled_projected @ pooled_reference.T / self.alignment_temperature
        )
        labels = torch.arange(
            similarity_logits.shape[0], device=similarity_logits.device
        )
        return 0.5 * (
            F.cross_entropy(similarity_logits, labels)
            + F.cross_entropy(similarity_logits.T, labels)
        )

    def _compute_prototype_alignment_loss(
        self,
        reference_hidden: torch.Tensor,
        projected_hidden: torch.Tensor,
    ) -> torch.Tensor:
        # Prototype alignment is optional because the first accepted slice keeps
        # the online objective small unless this extra term is explicitly enabled.
        if not self.enable_prototype_alignment:
            return projected_hidden.new_zeros(())
        prototype_target = self.reference_encoder.compute_prototype_target(
            reference_hidden
        )
        return torch.mean((projected_hidden - prototype_target) ** 2)

    def _compute_anchor_loss(self) -> torch.Tensor:
        # The anchor term measures drift away from the projector's initial state.
        anchor_loss = None
        for parameter_name, parameter in self.online_mlp_projector.named_parameters():
            anchor_parameter = self.projector_anchor_state_dict[parameter_name].to(
                parameter.device
            )
            parameter_loss = torch.mean((parameter - anchor_parameter) ** 2)
            anchor_loss = (
                parameter_loss if anchor_loss is None else anchor_loss + parameter_loss
            )
        if anchor_loss is None:
            return torch.zeros((), dtype=torch.float32)
        return anchor_loss

    def _compute_projector_drift(self) -> torch.Tensor:
        drift_value = None
        for parameter_name, parameter in self.online_mlp_projector.named_parameters():
            anchor_parameter = self.projector_anchor_state_dict[parameter_name].to(
                parameter.device
            )
            parameter_drift = torch.mean((parameter - anchor_parameter) ** 2)
            drift_value = (
                parameter_drift
                if drift_value is None
                else drift_value + parameter_drift
            )
        if drift_value is None:
            return torch.zeros((), dtype=torch.float32)
        return torch.sqrt(drift_value)

    def forward_source(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Score the frozen source path directly for the A0 variant."""
        validate_online_batch(batch)
        source_hidden = self.reference_encoder.encode_source(batch)
        scored = self.reference_encoder.score_source(source_hidden, batch["x"])
        outputs = {
            "hidden": source_hidden,
            "pooled": scored["pooled"],
            "recon": scored["recon"],
            "logits": scored["logits"],
            "point_scores": scored["point_scores"],
            "window_scores": scored["window_scores"],
            "aux": {
                "reference_hidden": source_hidden,
                "online_hidden": source_hidden,
                "projected_hidden": None,
                "scoring": scored["aux"],
                "latent_window_score": scored["aux"]["latent_window_score"],
            },
        }
        validate_model_outputs(outputs)
        return outputs

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        # One frozen source encoding feeds the residual projector. This keeps
        # calibration and adaptation in the same latent geometry.
        validate_online_batch(batch)
        console_print(
            "MODEL", "Online adaptation forward input batch", **summarize_batch(batch)
        )
        reference_hidden = self.reference_encoder.encode_source(batch)
        online_hidden = reference_hidden
        projected_hidden = self.online_mlp_projector(reference_hidden)

        scored_outputs = self.reference_encoder.score_projected(
            projected_hidden, batch["x"]
        )
        alignment_loss = self._compute_alignment_loss(
            reference_hidden, projected_hidden
        )
        prototype_alignment_loss = self._compute_prototype_alignment_loss(
            reference_hidden, projected_hidden
        )
        anchor_loss = self._compute_anchor_loss()
        projector_drift = self._compute_projector_drift()

        outputs = {
            "hidden": projected_hidden,
            "pooled": scored_outputs["pooled"],
            "recon": scored_outputs["recon"],
            "logits": scored_outputs["logits"],
            "point_scores": scored_outputs["point_scores"],
            "window_scores": scored_outputs["window_scores"],
            "aux": {
                "reference_hidden": reference_hidden,
                "online_hidden": online_hidden,
                "projected_hidden": projected_hidden,
                "alignment_loss": alignment_loss,
                "prototype_alignment_loss": prototype_alignment_loss,
                "anchor_loss": anchor_loss,
                "projector_drift": projector_drift,
                "target_param_group": self.target_param_group,
                "scoring": scored_outputs["aux"],
                "latent_window_score": scored_outputs["aux"]["latent_window_score"],
            },
        }
        validate_model_outputs(outputs)
        console_print(
            "MODEL",
            "Online adaptation forward outputs",
            reference_hidden=summarize_tensor(reference_hidden),
            online_hidden=summarize_tensor(online_hidden),
            projected_hidden=summarize_tensor(projected_hidden),
            recon=summarize_tensor(outputs["recon"]),
            logits=summarize_tensor(outputs["logits"]),
            point_scores=summarize_tensor(outputs["point_scores"]),
            window_scores=summarize_tensor(outputs["window_scores"]),
            alignment_loss=float(alignment_loss.detach().cpu()),
            prototype_alignment_loss=float(prototype_alignment_loss.detach().cpu()),
            anchor_loss=float(anchor_loss.detach().cpu()),
            projector_drift=float(projector_drift.detach().cpu()),
        )
        return outputs

    def _build_stage_log(
        self,
        stage_name: str,
        outputs: dict[str, Any],
        total_loss: torch.Tensor,
    ) -> dict[str, float]:
        return {
            f"{stage_name}_loss": float(total_loss.detach().cpu()),
            f"{stage_name}_alignment_loss": float(
                outputs["aux"]["alignment_loss"].detach().cpu()
            ),
            f"{stage_name}_prototype_alignment_loss": float(
                outputs["aux"]["prototype_alignment_loss"].detach().cpu()
            ),
            f"{stage_name}_anchor_loss": float(
                outputs["aux"]["anchor_loss"].detach().cpu()
            ),
            f"{stage_name}_projector_drift": float(
                outputs["aux"]["projector_drift"].detach().cpu()
            ),
            f"{stage_name}_window_score_mean": float(
                outputs["window_scores"].mean().detach().cpu()
            ),
        }

    def _shared_step(self, batch: dict[str, Any], stage_name: str) -> dict[str, Any]:
        # The online objective is intentionally smaller than the offline one:
        # align first, optionally use prototypes, and regularize projector drift.
        outputs = self.forward(batch)
        total_loss = (
            self.lambda_align * outputs["aux"]["alignment_loss"]
            + self.lambda_proto * outputs["aux"]["prototype_alignment_loss"]
            + self.lambda_anchor * outputs["aux"]["anchor_loss"]
        )
        loss_terms = {
            "total_loss": total_loss,
            "alignment_loss": outputs["aux"]["alignment_loss"],
            "prototype_alignment_loss": outputs["aux"]["prototype_alignment_loss"],
            "anchor_loss": outputs["aux"]["anchor_loss"],
        }
        console_print(
            stage_name.upper(),
            "Completed online adaptation stage step",
            batch_size=batch["x"].shape[0],
            total_loss=float(total_loss.detach().cpu()),
            alignment_loss=float(outputs["aux"]["alignment_loss"].detach().cpu()),
            prototype_alignment_loss=float(
                outputs["aux"]["prototype_alignment_loss"].detach().cpu()
            ),
            anchor_loss=float(outputs["aux"]["anchor_loss"].detach().cpu()),
            projector_drift=float(outputs["aux"]["projector_drift"].detach().cpu()),
        )
        return {
            "loss": total_loss,
            "log": self._build_stage_log(stage_name, outputs, total_loss),
            "outputs": outputs,
            "loss_terms": loss_terms,
            "batch": batch,
        }

    def training_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(batch=batch, stage_name="train")

    def validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(batch=batch, stage_name="val")

    def test_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(batch=batch, stage_name="test")
