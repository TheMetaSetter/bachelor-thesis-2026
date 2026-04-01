from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.contracts import validate_batch, validate_model_outputs
from src.data.augment import SyntheticAnomalyInjector
from src.models.base_model import BaseModel


class MultitaskWindowEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
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
        continuous_enabled: bool = True,
        continuous_num_prototypes: int = 8,
        discrete_enabled: bool = True,
        discrete_codebook_size: int = 16,
        gumbel_temperature: float = 1.0,
        alpha_logit_init: float = 0.0,
        beta_logit_init: float = 0.0,
        lambda_cls: float = 1.0,
        enable_diversity_loss: bool = False,
        enable_variance_loss: bool = False,
        enable_covariance_loss: bool = False,
        enable_usage_loss: bool = False,
        enable_gate_loss: bool = False,
        lambda_div: float = 0.0,
        lambda_var: float = 0.0,
        lambda_cov: float = 0.0,
        lambda_use: float = 0.0,
        lambda_gate: float = 0.0,
        variance_floor_gamma: float = 1.0,
        gate_barrier_margin: float = 0.25,
        use_synthetic_augmentation: bool = True,
        anomaly_probability: float = 0.5,
        min_segment_fraction: float = 0.1,
        max_segment_fraction: float = 0.2,
        spike_scale: float = 3.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.continuous_num_prototypes = continuous_num_prototypes
        self.discrete_codebook_size = discrete_codebook_size
        self.gumbel_temperature = gumbel_temperature
        self.lambda_cls = lambda_cls
        self.lambda_div = lambda_div
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.lambda_use = lambda_use
        self.lambda_gate = lambda_gate
        self.enable_diversity_loss = enable_diversity_loss
        self.enable_variance_loss = enable_variance_loss
        self.enable_covariance_loss = enable_covariance_loss
        self.enable_usage_loss = enable_usage_loss
        self.enable_gate_loss = enable_gate_loss
        self.variance_floor_gamma = variance_floor_gamma
        self.gate_barrier_margin = gate_barrier_margin
        self.use_synthetic_augmentation = use_synthetic_augmentation
        self.epsilon = 1e-6

        # Encoder block
        self.encoder = MultitaskWindowEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Continuous prototype block
        if continuous_enabled and continuous_num_prototypes > 0:
            self.continuous_prototype_bank = nn.Parameter(
                torch.randn(continuous_num_prototypes, hidden_dim)
            )
        else:
            self.register_parameter("continuous_prototype_bank", None)

        # Discrete prototype block
        if discrete_enabled and discrete_codebook_size > 0:
            self.discrete_assignment = nn.Linear(hidden_dim, discrete_codebook_size)
            self.discrete_codebook = nn.Parameter(torch.randn(discrete_codebook_size, hidden_dim))
        else:
            self.discrete_assignment = None
            self.register_parameter("discrete_codebook", None)

        # Fusion scalars and fusion equations
        self.alpha_logit = nn.Parameter(torch.tensor(float(alpha_logit_init)))
        self.beta_logit = nn.Parameter(torch.tensor(float(beta_logit_init)))

        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, input_dim),
        )

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Offline objective helpers
        self.branch_layer_norm = nn.LayerNorm(hidden_dim)
        self.synthetic_anomaly_injector = SyntheticAnomalyInjector(
            anomaly_probability=anomaly_probability,
            min_segment_fraction=min_segment_fraction,
            max_segment_fraction=max_segment_fraction,
            spike_scale=spike_scale,
        )
        self.optional_loss_configs: dict[str, dict[str, Any]] = {
            "diversity_loss": {
                "enabled": self.enable_diversity_loss,
                "weight": self.lambda_div,
                "compute_fn": self._compute_cross_branch_diversity_loss,
            },
            "variance_loss": {
                "enabled": self.enable_variance_loss,
                "weight": self.lambda_var,
                "compute_fn": self._compute_variance_floor_loss,
            },
            "covariance_loss": {
                "enabled": self.enable_covariance_loss,
                "weight": self.lambda_cov,
                "compute_fn": self._compute_covariance_reduction_loss,
            },
            "usage_loss": {
                "enabled": self.enable_usage_loss,
                "weight": self.lambda_use,
                "compute_fn": self._compute_prototype_usage_loss,
            },
            "gate_loss": {
                "enabled": self.enable_gate_loss,
                "weight": self.lambda_gate,
                "compute_fn": self._compute_gate_regularization_loss,
            },
        }

    def _zero_loss(self, reference_tensor: torch.Tensor) -> torch.Tensor:
        return reference_tensor.new_zeros(())

    def _continuous_prototype_lookup(self, hidden: torch.Tensor) -> dict[str, Any]:
        continuous_hidden = hidden
        attention_logits = None
        attention_weights = None

        if self.continuous_prototype_bank is not None:
            attention_logits = torch.einsum(
                "blh,kh->blk",
                hidden,
                self.continuous_prototype_bank,
            ) / math.sqrt(self.hidden_dim)
            attention_weights = torch.softmax(attention_logits, dim=-1)
            continuous_hidden = torch.einsum(
                "blk,kh->blh",
                attention_weights,
                self.continuous_prototype_bank,
            )

        return {
            "hidden": hidden,
            "prototype_context": continuous_hidden,
            "prototype_logits": attention_logits,
            "prototype_weights": attention_weights,
            "aux": {
                "branch_name": "continuous",
                "enabled": self.continuous_prototype_bank is not None,
                "num_prototypes": self.continuous_num_prototypes,
            },
        }

    def _discrete_prototype_lookup(self, hidden: torch.Tensor) -> dict[str, Any]:
        discrete_hidden = hidden
        assignment_logits = None
        assignment_probabilities = None
        code_indices = None

        if self.discrete_assignment is not None and self.discrete_codebook is not None:
            assignment_logits = self.discrete_assignment(hidden)
            assignment_probabilities = F.gumbel_softmax(
                assignment_logits,
                tau=self.gumbel_temperature,
                hard=False,
                dim=-1,
            )
            discrete_hidden = torch.einsum(
                "blk,kh->blh",
                assignment_probabilities,
                self.discrete_codebook,
            )
            code_indices = torch.argmax(assignment_probabilities, dim=-1)

        return {
            "hidden": hidden,
            "quantized_hidden": discrete_hidden,
            "assignment_logits": assignment_logits,
            "assignment_probabilities": assignment_probabilities,
            "code_indices": code_indices,
            "aux": {
                "branch_name": "discrete",
                "enabled": self.discrete_assignment is not None,
                "codebook_size": self.discrete_codebook_size,
                "temperature": self.gumbel_temperature,
            },
        }

    def _compute_fusion_outputs(
        self,
        continuous_hidden: torch.Tensor,
        discrete_hidden: torch.Tensor,
    ) -> dict[str, Any]:
        alpha = torch.sigmoid(self.alpha_logit)
        beta = torch.sigmoid(self.beta_logit)

        # TODO: Think of strategy to adjust alpha and beta
        # such that it tackles branch collapsing.
        hidden_reconstruction = beta * discrete_hidden + (1.0 - beta) * continuous_hidden
        hidden_classification = alpha * discrete_hidden + (1.0 - alpha) * continuous_hidden

        return {
            "hidden_reconstruction": hidden_reconstruction,
            "hidden_classification": hidden_classification,
            "alpha": alpha,
            "beta": beta,
            "aux": {
                "fusion_mode": "learnable_sigmoid_scalars",
                "alpha": float(alpha.detach().cpu()),
                "beta": float(beta.detach().cpu()),
                "alpha_logit": float(self.alpha_logit.detach().cpu()),
                "beta_logit": float(self.beta_logit.detach().cpu()),
            },
        }

    def _clone_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        cloned_batch: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                cloned_batch[key] = value.clone()
            elif isinstance(value, list):
                cloned_batch[key] = [dict(item) if isinstance(item, dict) else item for item in value]
            else:
                cloned_batch[key] = value
        return cloned_batch

    def _prepare_batch(self, batch: dict[str, Any], stage_name: str) -> dict[str, Any]:
        if (
            "classification_labels" in batch
            and "synthetic_anomaly_mask" in batch
            and "augmentation_metadata" in batch
        ):
            return self._clone_batch(batch)

        if stage_name == "train" and self.use_synthetic_augmentation:
            return self.synthetic_anomaly_injector.augment_batch(batch)

        prepared_batch = self._clone_batch(batch)
        batch_size, window_size, _ = prepared_batch["x"].shape
        prepared_batch["classification_labels"] = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=prepared_batch["x"].device,
        )
        prepared_batch["synthetic_anomaly_mask"] = torch.zeros(
            batch_size,
            window_size,
            dtype=torch.long,
            device=prepared_batch["x"].device,
        )
        prepared_batch["augmentation_metadata"] = [
            {
                "is_synthetic_anomaly": False,
                "anomaly_family": "clean",
                "start_index": None,
                "end_index": None,
                "affected_channels": [],
                "family_parameters_by_channel": {},
            }
            for _ in range(batch_size)
        ]
        if prepared_batch["point_labels"] is None:
            prepared_batch["point_labels"] = prepared_batch["synthetic_anomaly_mask"].clone()
        return prepared_batch

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        validate_batch(batch)
        encoder_outputs = self.encoder(batch)
        hidden = encoder_outputs["hidden"]

        continuous_outputs = self._continuous_prototype_lookup(hidden)
        discrete_outputs = self._discrete_prototype_lookup(hidden)
        fusion_outputs = self._compute_fusion_outputs(
            continuous_hidden=continuous_outputs["prototype_context"],
            discrete_hidden=discrete_outputs["quantized_hidden"],
        )

        hidden_reconstruction = fusion_outputs["hidden_reconstruction"]
        hidden_classification = fusion_outputs["hidden_classification"]
        recon = self.reconstruction_head(hidden_reconstruction)
        pooled_classification_hidden = hidden_classification.mean(dim=1)
        logits = self.classification_head(pooled_classification_hidden)
        point_scores = torch.mean((recon - batch["x"]) ** 2, dim=-1)

        outputs = {
            "hidden": hidden,
            "pooled": pooled_classification_hidden,
            "recon": recon,
            "logits": logits,
            "point_scores": point_scores,
            "window_scores": point_scores.mean(dim=1),
            "aux": {
                "encoder": encoder_outputs["aux"],
                "continuous_branch": continuous_outputs,
                "discrete_branch": discrete_outputs,
                "fusion": fusion_outputs["aux"],
                "hidden_reconstruction": hidden_reconstruction,
                "hidden_classification": hidden_classification,
                "alpha": fusion_outputs["alpha"],
                "beta": fusion_outputs["beta"],
            },
        }
        validate_model_outputs(outputs)
        return outputs

    def _normalize_branch_tokens(self, branch_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_hidden = self.branch_layer_norm(branch_hidden).reshape(-1, self.hidden_dim)
        feature_mean = normalized_hidden.mean(dim=0, keepdim=True)
        feature_std = normalized_hidden.std(dim=0, unbiased=False, keepdim=True)
        standardized_hidden = (normalized_hidden - feature_mean) / (feature_std + self.epsilon)
        return normalized_hidden, standardized_hidden

    def _compute_reconstruction_loss(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> torch.Tensor:
        return torch.mean((outputs["recon"] - batch["x"]) ** 2)

    def _compute_classification_loss(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> torch.Tensor:
        return F.cross_entropy(outputs["logits"], batch["classification_labels"].long())

    def _compute_cross_branch_diversity_loss(self, outputs: dict[str, Any]) -> torch.Tensor:
        continuous_hidden = outputs["aux"]["continuous_branch"]["prototype_context"]
        discrete_hidden = outputs["aux"]["discrete_branch"]["quantized_hidden"]
        _, standardized_continuous = self._normalize_branch_tokens(continuous_hidden)
        _, standardized_discrete = self._normalize_branch_tokens(discrete_hidden)
        num_tokens = standardized_continuous.shape[0]
        cross_branch_correlation = standardized_continuous.T @ standardized_discrete / num_tokens
        return cross_branch_correlation.pow(2).mean()

    def _compute_variance_floor_loss(self, outputs: dict[str, Any]) -> torch.Tensor:
        variance_losses: list[torch.Tensor] = []
        for branch_name in ["continuous_branch", "discrete_branch"]:
            branch_hidden = outputs["aux"][branch_name][
                "prototype_context" if branch_name == "continuous_branch" else "quantized_hidden"
            ]
            normalized_hidden, _ = self._normalize_branch_tokens(branch_hidden)
            feature_std = normalized_hidden.std(dim=0, unbiased=False)
            variance_losses.append(F.relu(self.variance_floor_gamma - feature_std).pow(2).mean())
        return torch.stack(variance_losses).sum()

    def _compute_covariance_reduction_loss(self, outputs: dict[str, Any]) -> torch.Tensor:
        covariance_losses: list[torch.Tensor] = []
        for branch_name in ["continuous_branch", "discrete_branch"]:
            branch_hidden = outputs["aux"][branch_name][
                "prototype_context" if branch_name == "continuous_branch" else "quantized_hidden"
            ]
            _, standardized_hidden = self._normalize_branch_tokens(branch_hidden)
            num_tokens = standardized_hidden.shape[0]
            covariance_matrix = standardized_hidden.T @ standardized_hidden / num_tokens
            diagonal_matrix = torch.diag(torch.diag(covariance_matrix))
            off_diagonal_matrix = covariance_matrix - diagonal_matrix
            if self.hidden_dim == 1:
                covariance_losses.append(self._zero_loss(branch_hidden))
            else:
                covariance_losses.append(
                    off_diagonal_matrix.pow(2).sum() / (self.hidden_dim * (self.hidden_dim - 1))
                )
        return torch.stack(covariance_losses).sum()

    def _compute_prototype_usage_loss(self, outputs: dict[str, Any]) -> torch.Tensor:
        assignment_probabilities = outputs["aux"]["discrete_branch"]["assignment_probabilities"]
        if assignment_probabilities is None or self.discrete_codebook_size <= 0:
            return self._zero_loss(outputs["hidden"])
        average_usage = assignment_probabilities.mean(dim=(0, 1))
        target_usage = torch.full_like(average_usage, 1.0 / self.discrete_codebook_size)
        return torch.sum((average_usage - target_usage) ** 2)

    def _compute_gate_regularization_loss(self, outputs: dict[str, Any]) -> torch.Tensor:
        alpha = outputs["aux"]["alpha"]
        beta = outputs["aux"]["beta"]
        alpha_barrier = F.relu(torch.abs(alpha - 0.5) - self.gate_barrier_margin).pow(2)
        beta_barrier = F.relu(torch.abs(beta - 0.5) - self.gate_barrier_margin).pow(2)
        return alpha_barrier + beta_barrier

    def _compute_optional_loss_terms(self, outputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        optional_loss_values: dict[str, torch.Tensor] = {}
        for loss_name, loss_config in self.optional_loss_configs.items():
            compute_fn: Callable[[dict[str, Any]], torch.Tensor] = loss_config["compute_fn"]
            if loss_config["enabled"]:
                optional_loss_values[loss_name] = compute_fn(outputs)
            else:
                optional_loss_values[loss_name] = self._zero_loss(outputs["hidden"])
        return optional_loss_values

    def _compute_total_loss(
        self,
        reconstruction_loss: torch.Tensor,
        classification_loss: torch.Tensor,
        optional_loss_values: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        total_loss = reconstruction_loss + self.lambda_cls * classification_loss
        for loss_name, loss_value in optional_loss_values.items():
            total_loss = total_loss + self.optional_loss_configs[loss_name]["weight"] * loss_value
        return total_loss

    def _build_stage_log(
        self,
        stage_name: str,
        outputs: dict[str, Any],
        loss_terms: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> dict[str, float]:
        predicted_labels = torch.argmax(outputs["logits"], dim=-1)
        classification_accuracy = float(
            (predicted_labels == batch["classification_labels"]).float().mean().detach().cpu()
        )
        return {
            f"{stage_name}_loss": float(loss_terms["total_loss"].detach().cpu()),
            f"{stage_name}_reconstruction_loss": float(loss_terms["reconstruction_loss"].detach().cpu()),
            f"{stage_name}_classification_loss": float(loss_terms["classification_loss"].detach().cpu()),
            f"{stage_name}_diversity_loss": float(loss_terms["diversity_loss"].detach().cpu()),
            f"{stage_name}_variance_loss": float(loss_terms["variance_loss"].detach().cpu()),
            f"{stage_name}_covariance_loss": float(loss_terms["covariance_loss"].detach().cpu()),
            f"{stage_name}_usage_loss": float(loss_terms["usage_loss"].detach().cpu()),
            f"{stage_name}_gate_loss": float(loss_terms["gate_loss"].detach().cpu()),
            f"{stage_name}_classification_accuracy": classification_accuracy,
            f"{stage_name}_alpha": float(outputs["aux"]["alpha"].detach().cpu()),
            f"{stage_name}_beta": float(outputs["aux"]["beta"].detach().cpu()),
            f"{stage_name}_continuous_norm": float(
                outputs["aux"]["continuous_branch"]["prototype_context"].norm(dim=-1).mean().detach().cpu()
            ),
            f"{stage_name}_discrete_norm": float(
                outputs["aux"]["discrete_branch"]["quantized_hidden"].norm(dim=-1).mean().detach().cpu()
            ),
        }

    def _shared_step(self, batch: dict[str, Any], stage_name: str) -> dict[str, Any]:
        prepared_batch = self._prepare_batch(batch, stage_name)
        outputs = self.forward(prepared_batch)

        reconstruction_loss = self._compute_reconstruction_loss(outputs, prepared_batch)
        classification_loss = self._compute_classification_loss(outputs, prepared_batch)
        optional_loss_values = self._compute_optional_loss_terms(outputs)
        total_loss = self._compute_total_loss(
            reconstruction_loss=reconstruction_loss,
            classification_loss=classification_loss,
            optional_loss_values=optional_loss_values,
        )

        loss_terms = {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "classification_loss": classification_loss,
            **optional_loss_values,
        }
        return {
            "loss": total_loss,
            "log": self._build_stage_log(stage_name, outputs, loss_terms, prepared_batch),
            "outputs": outputs,
            "loss_terms": loss_terms,
            "batch": prepared_batch,
        }

    def training_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(batch=batch, stage_name="train")

    def validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(batch=batch, stage_name="val")

    def test_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(batch=batch, stage_name="test")
