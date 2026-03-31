from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.core.contracts import validate_batch, validate_model_outputs
from src.models.base_encoder import BaseEncoder
from src.models.base_model import BaseModel
from src.models.modules.continuous_prototypes import ContinuousPrototypeLookup
from src.models.modules.discrete_prototypes import DiscretePrototypeLookup
from src.models.modules.fusion import TaskFusion


class MultitaskWindowEncoder(BaseEncoder):
    def __init__(
        self,
        input_dim: int,
        encoder_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        hidden = self.network(batch["x"])
        return {
            "hidden": hidden,
            "pooled": hidden.mean(dim=1),
            "aux": {"encoder_name": "multitask_window_encoder"},
        }


class ThesisMultitaskModel(BaseModel):
    def __init__(
        self,
        input_dim: int,
        encoder_dim: int,
        hidden_dim: int,
        num_classes: int = 2,
        dropout: float = 0.0,
        continuous_enabled: bool = False,
        continuous_num_prototypes: int = 0,
        discrete_enabled: bool = False,
        discrete_codebook_size: int = 0,
        fusion_mode: str = "identity",
    ) -> None:
        super().__init__()
        self.encoder = MultitaskWindowEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.continuous_prototypes = ContinuousPrototypeLookup(
            hidden_dim=hidden_dim,
            num_prototypes=continuous_num_prototypes,
            enabled=continuous_enabled,
        )
        self.discrete_prototypes = DiscretePrototypeLookup(
            hidden_dim=hidden_dim,
            codebook_size=discrete_codebook_size,
            enabled=discrete_enabled,
        )
        self.fusion = TaskFusion(mode=fusion_mode)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, input_dim),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        validate_batch(batch)
        encoder_outputs = self.encoder(batch)
        continuous_outputs = self.continuous_prototypes(encoder_outputs["hidden"])
        discrete_outputs = self.discrete_prototypes(encoder_outputs["hidden"])
        fused_outputs = self.fusion(
            base_hidden=encoder_outputs["hidden"],
            continuous_branch=continuous_outputs,
            discrete_branch=discrete_outputs,
        )

        reconstruction_hidden = fused_outputs["hidden_reconstruction"]
        classification_hidden = fused_outputs["hidden_classification"]
        recon = self.reconstruction_head(reconstruction_hidden)
        pooled_classification_hidden = classification_hidden.mean(dim=1)
        logits = self.classification_head(pooled_classification_hidden)
        point_scores = torch.mean((recon - batch["x"]) ** 2, dim=-1)

        outputs = {
            "hidden": encoder_outputs["hidden"],
            "pooled": encoder_outputs["pooled"],
            "recon": recon,
            "logits": logits,
            "point_scores": point_scores,
            "window_scores": point_scores.mean(dim=1),
            "aux": {
                "encoder": encoder_outputs["aux"],
                "continuous_branch": continuous_outputs,
                "discrete_branch": discrete_outputs,
                "fusion": fused_outputs["aux"],
                "hidden_reconstruction": reconstruction_hidden,
                "hidden_classification": classification_hidden,
            },
        }
        validate_model_outputs(outputs)
        return outputs
