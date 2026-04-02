from __future__ import annotations
"""Projector-first online adaptation model for the first accepted Phase 4 slice.

This file should be read after the offline multitask model. The online path is
deliberately conservative: it reuses the offline encoder geometry, keeps the
reference encoder frozen, and adapts only a small residual projector by default.
"""

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.contracts import validate_model_outputs, validate_online_batch
from src.models.base_model import BaseModel
from src.models.thesis_multitask import ThesisMultitaskModel


class ThesisMultitaskEncoderAdapter(nn.Module):
    # The adapter keeps the online file readable by reusing the offline encoder
    # without forcing the rest of the online logic back into the multitask file.
    def __init__(self, thesis_model: ThesisMultitaskModel, freeze_parameters: bool = True) -> None:
        super().__init__()
        self.model = copy.deepcopy(thesis_model)
        if freeze_parameters:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self.model.encoder(batch)

    def score_from_hidden(self, hidden: torch.Tensor, x_tensor: torch.Tensor) -> dict[str, Any]:
        continuous_outputs = self.model._continuous_prototype_lookup(hidden)
        discrete_outputs = self.model._discrete_prototype_lookup(hidden)
        fusion_outputs = self.model._compute_fusion_outputs(
            continuous_hidden=continuous_outputs["prototype_context"],
            discrete_hidden=discrete_outputs["quantized_hidden"],
        )
        hidden_reconstruction = fusion_outputs["hidden_reconstruction"]
        hidden_classification = fusion_outputs["hidden_classification"]
        recon = self.model.reconstruction_head(hidden_reconstruction)
        pooled_hidden = hidden_classification.mean(dim=1)
        logits = self.model.classification_head(pooled_hidden)
        point_scores = torch.mean((recon - x_tensor) ** 2, dim=-1)
        return {
            "pooled": pooled_hidden,
            "recon": recon,
            "logits": logits,
            "point_scores": point_scores,
            "window_scores": point_scores.mean(dim=1),
            "aux": {
                "continuous_branch": continuous_outputs,
                "discrete_branch": discrete_outputs,
                "fusion": fusion_outputs["aux"],
                "hidden_reconstruction": hidden_reconstruction,
                "hidden_classification": hidden_classification,
                "alpha": fusion_outputs["alpha"],
                "beta": fusion_outputs["beta"],
            },
        }

    def compute_prototype_target(self, hidden: torch.Tensor) -> torch.Tensor:
        prototype_targets: list[torch.Tensor] = []
        continuous_outputs = self.model._continuous_prototype_lookup(hidden)
        prototype_targets.append(continuous_outputs["prototype_context"])
        discrete_outputs = self.model._discrete_prototype_lookup(hidden)
        prototype_targets.append(discrete_outputs["quantized_hidden"])
        return torch.stack(prototype_targets, dim=0).mean(dim=0)

    def encoder_parameters(self) -> list[nn.Parameter]:
        return list(self.model.encoder.parameters())


class ResidualProjector(nn.Module):
    # The projector is residual and near-identity on purpose. That makes it the
    # safest first parameter group to adapt online.
    def __init__(self, hidden_dim: int, projector_hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, projector_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projector_hidden_dim, hidden_dim),
        )
        final_layer = self.network[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.network(hidden)


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
            raise ValueError("The first online adaptation slice supports only clean_stream_only=True")
        if score_source != "projected_hidden":
            raise ValueError("The first online adaptation slice supports only score_source='projected_hidden'")

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
        self.reference_encoder = ThesisMultitaskEncoderAdapter(frozen_multitask_model, freeze_parameters=True)
        self.online_encoder = ThesisMultitaskEncoderAdapter(frozen_multitask_model, freeze_parameters=True)
        self.projector = ResidualProjector(
            hidden_dim=hidden_dim,
            projector_hidden_dim=projector_hidden_dim,
            dropout=projector_dropout,
        )
        self.projector_anchor_state_dict = self._clone_projector_state_dict()
        self._set_trainable_parameter_group(target_param_group)

    def _load_reference_model(self, checkpoint_path: str | Path) -> ThesisMultitaskModel:
        # The online runtime is defined only for multitask checkpoints. Failing
        # early here prevents confusing baseline-versus-online mismatches later.
        loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = loaded_checkpoint["config"]
        model_name = config.get("model", {}).get("model_name")
        if model_name != "thesis_multitask":
            raise ValueError(
                "reference_checkpoint_path must point to a thesis_multitask checkpoint, "
                f"but found model_name={model_name!r}"
            )
        model_kwargs = {
            key: value
            for key, value in config["model"].items()
            if key != "model_name"
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
        reference_model.eval()
        return reference_model

    def _clone_projector_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            parameter_name: parameter.detach().cpu().clone()
            for parameter_name, parameter in self.projector.state_dict().items()
        }

    def get_projector_anchor_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            parameter_name: parameter.clone()
            for parameter_name, parameter in self.projector_anchor_state_dict.items()
        }

    def load_projector_anchor_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.projector_anchor_state_dict = {
            parameter_name: parameter.detach().cpu().clone()
            for parameter_name, parameter in state_dict.items()
        }

    def _set_trainable_parameter_group(self, target_param_group: str) -> None:
        # Parameter groups are explicit because the design docs treat the online
        # optimization boundary as part of the architecture, not a small detail.
        for parameter in self.reference_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.online_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.projector.parameters():
            parameter.requires_grad = False

        if target_param_group == "projector_params":
            for parameter in self.projector.parameters():
                parameter.requires_grad = True
            return
        if target_param_group == "online_encoder_params":
            for parameter in self.online_encoder.encoder_parameters():
                parameter.requires_grad = True
            return
        raise ValueError("target_param_group must be either 'projector_params' or 'online_encoder_params'")

    def get_parameter_group(self, target_param_group: str) -> list[nn.Parameter]:
        if target_param_group == "projector_params":
            return list(self.projector.parameters())
        if target_param_group == "online_encoder_params":
            return self.online_encoder.encoder_parameters()
        raise ValueError("target_param_group must be either 'projector_params' or 'online_encoder_params'")

    def _replace_batch_x(self, batch: dict[str, Any], x_tensor: torch.Tensor) -> dict[str, Any]:
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
        similarity_logits = pooled_projected @ pooled_reference.T / self.alignment_temperature
        labels = torch.arange(similarity_logits.shape[0], device=similarity_logits.device)
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
        prototype_target = self.reference_encoder.compute_prototype_target(reference_hidden)
        return torch.mean((projected_hidden - prototype_target) ** 2)

    def _compute_anchor_loss(self) -> torch.Tensor:
        # The anchor term measures drift away from the projector's initial state.
        anchor_loss = None
        for parameter_name, parameter in self.projector.named_parameters():
            anchor_parameter = self.projector_anchor_state_dict[parameter_name].to(parameter.device)
            parameter_loss = torch.mean((parameter - anchor_parameter) ** 2)
            anchor_loss = parameter_loss if anchor_loss is None else anchor_loss + parameter_loss
        if anchor_loss is None:
            return torch.zeros((), dtype=torch.float32)
        return anchor_loss

    def _compute_projector_drift(self) -> torch.Tensor:
        drift_value = None
        for parameter_name, parameter in self.projector.named_parameters():
            anchor_parameter = self.projector_anchor_state_dict[parameter_name].to(parameter.device)
            parameter_drift = torch.mean((parameter - anchor_parameter) ** 2)
            drift_value = parameter_drift if drift_value is None else drift_value + parameter_drift
        if drift_value is None:
            return torch.zeros((), dtype=torch.float32)
        return torch.sqrt(drift_value)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        # The forward path follows the design narrative literally:
        # build two views, encode them separately, project the online one, then score in reference space.
        validate_online_batch(batch)
        reference_outputs = self.reference_encoder(self._replace_batch_x(batch, batch["view_a"]))
        online_outputs = self.online_encoder(self._replace_batch_x(batch, batch["view_b"]))
        reference_hidden = reference_outputs["hidden"]
        online_hidden = online_outputs["hidden"]
        projected_hidden = self.projector(online_hidden)

        scored_outputs = self.reference_encoder.score_from_hidden(projected_hidden, batch["x"])
        alignment_loss = self._compute_alignment_loss(reference_hidden, projected_hidden)
        prototype_alignment_loss = self._compute_prototype_alignment_loss(reference_hidden, projected_hidden)
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
            },
        }
        validate_model_outputs(outputs)
        return outputs

    def _build_stage_log(
        self,
        stage_name: str,
        outputs: dict[str, Any],
        total_loss: torch.Tensor,
    ) -> dict[str, float]:
        return {
            f"{stage_name}_loss": float(total_loss.detach().cpu()),
            f"{stage_name}_alignment_loss": float(outputs["aux"]["alignment_loss"].detach().cpu()),
            f"{stage_name}_prototype_alignment_loss": float(
                outputs["aux"]["prototype_alignment_loss"].detach().cpu()
            ),
            f"{stage_name}_anchor_loss": float(outputs["aux"]["anchor_loss"].detach().cpu()),
            f"{stage_name}_projector_drift": float(outputs["aux"]["projector_drift"].detach().cpu()),
            f"{stage_name}_window_score_mean": float(outputs["window_scores"].mean().detach().cpu()),
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
