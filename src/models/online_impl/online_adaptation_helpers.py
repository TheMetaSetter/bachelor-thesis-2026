from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.engine.online_tta.signature_verification import (
    PrototypeVerificationMetadata,
)
from src.models.neural_blocks import build_multilayer_perceptron
from src.models.thesis_multitask import ThesisMultitaskModel

class ThesisMultitaskEncoderAdapter(nn.Module):
    # The adapter keeps the online file readable by reusing the offline encoder
    # without forcing the rest of the online logic back into the multitask file.
    def __init__(
        self, thesis_model: ThesisMultitaskModel, freeze_parameters: bool = True
    ) -> None:
        super().__init__()
        self.model = copy.deepcopy(thesis_model)
        if freeze_parameters:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self.model.encoder(batch)

    def encode_source(self, batch: dict[str, Any]) -> torch.Tensor:
        """Encode one input window with the frozen source encoder."""
        return self.forward(batch)["hidden"].detach()

    def score_source(
        self, source_hidden: torch.Tensor, x_tensor: torch.Tensor
    ) -> dict[str, Any]:
        return self.score_from_hidden(source_hidden, x_tensor)

    def score_projected(
        self, projected_hidden: torch.Tensor, x_tensor: torch.Tensor
    ) -> dict[str, Any]:
        return self.score_from_hidden(projected_hidden, x_tensor)

    def score_from_hidden(
        self, hidden: torch.Tensor, x_tensor: torch.Tensor
    ) -> dict[str, Any]:
        # The online adapter reuses the frozen offline scoring heads on the same
        # latent geometry. That assumption is deliberate and should stay visible.
        continuous_outputs = self.model._continuous_prototype_lookup(
            hidden,
            stage_name="test",
        )
        discrete_outputs = self.model._discrete_prototype_lookup(
            hidden,
            stage_name="test",
        )
        fusion_outputs = self.model._compute_fusion_outputs(
            continuous_hidden=continuous_outputs["prototype_context"],
            discrete_hidden=discrete_outputs["quantized_hidden"],
        )
        hidden_reconstruction = fusion_outputs["hidden_reconstruction"]
        hidden_classification = fusion_outputs["hidden_classification"]
        recon = self.model.reconstruction_head(hidden_reconstruction)
        if hidden_classification.shape[1] != self.model.window_size:
            raise ValueError(
                "hidden_classification must have window dimension "
                f"{self.model.window_size}, but received "
                f"{hidden_classification.shape[1]}"
            )
        flattened_classification_hidden = hidden_classification.reshape(
            hidden_classification.shape[0],
            self.model.window_size * self.model.hidden_dim,
        )
        logits = self.model.classification_head(flattened_classification_hidden)
        point_scores = torch.mean((recon - x_tensor) ** 2, dim=-1)
        latent_window_score = self._compute_latent_memory_score(hidden)
        return {
            "pooled": flattened_classification_hidden,
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
                "latent_window_score": latent_window_score,
            },
        }

    def prototype_verification_metadata(self) -> PrototypeVerificationMetadata:
        """Expose frozen prototype metadata for the online verification path."""
        return PrototypeVerificationMetadata.from_model(self.model)

    def _compute_latent_memory_score(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return mean nearest-normal-prototype cosine distance per window."""
        prototypes = self.model.continuous_prototype_bank
        if not isinstance(prototypes, torch.Tensor):
            raise ValueError("continuous prototype bank is required for latent scoring")
        normalized_hidden = F.normalize(hidden, dim=-1)
        normalized_prototypes = F.normalize(prototypes, dim=-1)
        distances = 1.0 - normalized_hidden @ normalized_prototypes.T
        return distances.min(dim=-1).values.mean(dim=-1)

    def compute_prototype_target(self, hidden: torch.Tensor) -> torch.Tensor:
        prototype_targets: list[torch.Tensor] = []
        continuous_outputs = self.model._continuous_prototype_lookup(
            hidden,
            stage_name="test",
        )
        prototype_targets.append(continuous_outputs["prototype_context"])
        discrete_outputs = self.model._discrete_prototype_lookup(
            hidden,
            stage_name="test",
        )
        prototype_targets.append(discrete_outputs["quantized_hidden"])
        return torch.stack(prototype_targets, dim=0).mean(dim=0)

    def encoder_parameters(self) -> list[nn.Parameter]:
        return list(self.model.encoder.parameters())


class NearIdentityMLPProjector(nn.Module):
    # The projector is residual and near-identity on purpose. That keeps the
    # first online step close to the calibrated latent space.
    def __init__(
        self,
        hidden_dim: int,
        projector_hidden_dim: int,
        dropout: float = 0.0,
        init_alpha: float = 1.0e-3,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, projector_hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(projector_hidden_dim, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

        nn.init.kaiming_uniform_(self.fc1.weight, a=5**0.5)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0e-4)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # At initialization the residual branch is tiny, so the projector is
        # almost an identity map but still trainable.
        residual = self.fc1(hidden)
        residual = self.activation(residual)
        residual = self.dropout(residual)
        residual = self.fc2(residual)
        return hidden + self.alpha * residual
