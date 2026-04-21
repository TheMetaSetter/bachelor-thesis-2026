from __future__ import annotations

"""Self-contained multitask prototype-fusion model.

This is the main offline thesis model, so the file is intentionally long and
intentionally self-contained. A fresher should read it in this order: encoder,
continuous branch, discrete branch, fusion, optional losses, then the shared
stage step that assembles the training objective.
"""

import math
import time
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.console import (
    console_print,
    print_parameter_summary,
    summarize_batch,
    summarize_label_distribution,
    summarize_tensor,
)
from src.core.contracts import validate_batch, validate_model_outputs
from src.data.augment import REDLAMP_ANOMALY_FAMILIES, SyntheticAnomalyInjector
from src.models.base_model import BaseModel


def build_multilayer_perceptron(
    *,
    input_dim: int,
    intermediate_dim: int,
    output_dim: int,
    num_linear_layers: int,
    dropout: float,
    apply_output_activation: bool,
) -> nn.Sequential:
    """Build a readable MLP with one explicit shared depth contract.

    The repository keeps one thesis model in one file, so this helper exists
    only to avoid repeating the same layer-construction pattern three times for
    the encoder, reconstruction head, and classification head.
    """
    if num_linear_layers < 2:
        raise ValueError("num_linear_layers must be at least 2")

    layer_dims = (
        [input_dim] + [intermediate_dim] * (num_linear_layers - 1) + [output_dim]
    )
    network_layers: list[nn.Module] = []
    for layer_index, (layer_input_dim, layer_output_dim) in enumerate(
        zip(layer_dims[:-1], layer_dims[1:])
    ):
        is_last_linear_layer = layer_index == num_linear_layers - 1
        network_layers.append(nn.Linear(layer_input_dim, layer_output_dim))
        if not is_last_linear_layer:
            network_layers.append(nn.ReLU())
            network_layers.append(nn.Dropout(dropout))
        elif apply_output_activation:
            network_layers.append(nn.ReLU())

    return nn.Sequential(*network_layers)


class MultitaskWindowEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_dim: int,
        hidden_dim: int,
        num_linear_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # The encoder depth is shared with both task heads so the offline model
        # can form a symmetric MLP contract from YAML instead of hard-coding
        # different depths in different submodules.
        self.network = build_multilayer_perceptron(
            input_dim=input_dim,
            intermediate_dim=encoder_dim,
            output_dim=hidden_dim,
            num_linear_layers=num_linear_layers,
            dropout=dropout,
            apply_output_activation=True,
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
        mlp_num_linear_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.0,
        continuous_enabled: bool = True,
        continuous_num_prototypes: int = 8,
        discrete_enabled: bool = True,
        discrete_codebook_size: int = 16,
        gumbel_temperature: float = 1.0,
        temperature_start: float = 1.0,
        temperature_end: float = 1.0,
        temperature_anneal_fraction: float = 1.0,
        temperature_hold_fraction: float = 0.0,
        alpha_logit_init: float = 0.0,
        beta_logit_init: float = 0.0,
        use_label_refurbishment: bool = False,
        refurbishment_alpha: float = 0.0,
        refurbishment_beta: float = 0.0,
        reconstruction_normal_only: bool = False,
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
        usage_lambda_start: float | None = None,
        usage_lambda_end: float | None = None,
        usage_lambda_schedule_fraction: float = 1.0,
        variance_floor_gamma: float = 1.0,
        gate_barrier_margin: float = 0.25,
        use_synthetic_augmentation: bool = True,
        use_synthetic_validation: bool = True,
        synthetic_validation_seed: int = 7,
        freeze_fusion_for_epochs: int = 0,
        warmup_alpha_value: float = 0.5,
        warmup_beta_value: float = 0.5,
        anomaly_probability: float = 0.5,
        min_segment_fraction: float = 0.1,
        max_segment_fraction: float = 0.2,
        spike_scale: float = 3.0,
        anomaly_families: tuple[str, ...] | list[str] = REDLAMP_ANOMALY_FAMILIES,
    ) -> None:
        super().__init__()
        # This constructor stores both the architecture and the experiment
        # switches because the repository follows the one-model-one-file rule.
        self.hidden_dim = hidden_dim
        self.mlp_num_linear_layers = mlp_num_linear_layers
        self.num_classes = num_classes
        self.continuous_num_prototypes = continuous_num_prototypes
        self.discrete_codebook_size = discrete_codebook_size
        self.default_gumbel_temperature = gumbel_temperature
        self.gumbel_temperature = gumbel_temperature
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.temperature_anneal_fraction = temperature_anneal_fraction
        self.temperature_hold_fraction = temperature_hold_fraction
        self.use_label_refurbishment = use_label_refurbishment
        self.refurbishment_alpha = refurbishment_alpha
        self.refurbishment_beta = refurbishment_beta
        self.reconstruction_normal_only = reconstruction_normal_only
        self.lambda_cls = lambda_cls
        self.lambda_div = lambda_div
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.lambda_use = lambda_use
        self.lambda_gate = lambda_gate
        self.usage_lambda_start = (
            lambda_use if usage_lambda_start is None else usage_lambda_start
        )
        self.usage_lambda_end = (
            lambda_use if usage_lambda_end is None else usage_lambda_end
        )
        self.usage_lambda_schedule_fraction = usage_lambda_schedule_fraction
        self.current_usage_lambda = self.usage_lambda_start
        self.enable_diversity_loss = enable_diversity_loss
        self.enable_variance_loss = enable_variance_loss
        self.enable_covariance_loss = enable_covariance_loss
        self.enable_usage_loss = enable_usage_loss
        self.enable_gate_loss = enable_gate_loss
        self.variance_floor_gamma = variance_floor_gamma
        self.gate_barrier_margin = gate_barrier_margin
        self.use_synthetic_augmentation = use_synthetic_augmentation
        self.use_synthetic_validation = use_synthetic_validation
        self.synthetic_validation_seed = synthetic_validation_seed
        self.freeze_fusion_for_epochs = freeze_fusion_for_epochs
        self.warmup_alpha_value = warmup_alpha_value
        self.warmup_beta_value = warmup_beta_value
        self.epsilon = 1e-6
        self.current_epoch_index = 0
        self.current_total_epochs = 1
        self.active_alpha_override: float | None = None
        self.active_beta_override: float | None = None
        self.schedule_state = {
            "epoch": 1,
            "warmup_active": False,
            "freeze_fusion_for_epochs": self.freeze_fusion_for_epochs,
            "temperature": self.gumbel_temperature,
            "usage_lambda": self.current_usage_lambda,
        }
        if self.use_label_refurbishment and self.num_classes != 2:
            raise ValueError(
                "label refurbishment currently supports only binary classification"
            )

        # Encoder block.
        # This produces the common hidden state that both prototype branches see.
        self.encoder = MultitaskWindowEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            hidden_dim=hidden_dim,
            num_linear_layers=mlp_num_linear_layers,
            dropout=dropout,
        )

        # Continuous branch.
        # This branch retrieves a soft prototype context from a learned bank.
        if continuous_enabled and continuous_num_prototypes > 0:
            self.continuous_prototype_bank = nn.Parameter(
                torch.randn(continuous_num_prototypes, hidden_dim)
            )
        else:
            self.register_parameter("continuous_prototype_bank", None)

        # Discrete branch.
        # This branch assigns tokens to a codebook through Gumbel-Softmax.
        if discrete_enabled and discrete_codebook_size > 0:
            self.discrete_assignment = nn.Linear(hidden_dim, discrete_codebook_size)
            self.discrete_codebook = nn.Parameter(
                torch.randn(discrete_codebook_size, hidden_dim)
            )
        else:
            self.discrete_assignment = None
            self.register_parameter("discrete_codebook", None)

        # Fusion scalars.
        # `alpha` controls the classification mix and `beta` controls the
        # reconstruction mix so the two tasks can prefer different geometry.
        self.alpha_logit = nn.Parameter(torch.tensor(float(alpha_logit_init)))
        self.beta_logit = nn.Parameter(torch.tensor(float(beta_logit_init)))

        # Task heads.
        # Supervision lives on the fused task-specific hidden states, not on the
        # branch-local states. That keeps the branches observable but not separate predictors.
        self.reconstruction_head = build_multilayer_perceptron(
            input_dim=hidden_dim,
            intermediate_dim=encoder_dim,
            output_dim=input_dim,
            num_linear_layers=mlp_num_linear_layers,
            dropout=dropout,
            apply_output_activation=False,
        )

        self.classification_head = build_multilayer_perceptron(
            input_dim=hidden_dim,
            intermediate_dim=hidden_dim,
            output_dim=num_classes,
            num_linear_layers=mlp_num_linear_layers,
            dropout=dropout,
            apply_output_activation=False,
        )

        # Offline objective helpers.
        # Optional losses are activated by `lambda_*` so ablations can stay on
        # one codepath instead of branching into separate model variants. The
        # intended starting point is still only reconstruction plus
        # classification loss until observed failure modes justify more terms.
        self.branch_layer_norm = nn.LayerNorm(hidden_dim)
        self.synthetic_anomaly_injector = SyntheticAnomalyInjector(
            anomaly_probability=anomaly_probability,
            min_segment_fraction=min_segment_fraction,
            max_segment_fraction=max_segment_fraction,
            spike_scale=spike_scale,
            anomaly_families=anomaly_families,
        )
        self.synthetic_validation_injector = SyntheticAnomalyInjector(
            anomaly_probability=anomaly_probability,
            min_segment_fraction=min_segment_fraction,
            max_segment_fraction=max_segment_fraction,
            spike_scale=spike_scale,
            anomaly_families=anomaly_families,
            deterministic_seed=synthetic_validation_seed,
        )

        self.optional_loss_configs: dict[str, dict[str, Any]] = {
            "diversity_loss": {
                "enabled": self.lambda_div > 0.0,
                "weight": self.lambda_div,
                "compute_fn": self._compute_cross_branch_diversity_loss,
            },
            "variance_loss": {
                "enabled": self.lambda_var > 0.0,
                "weight": self.lambda_var,
                "compute_fn": self._compute_variance_floor_loss,
            },
            "covariance_loss": {
                "enabled": self.lambda_cov > 0.0,
                "weight": self.lambda_cov,
                "compute_fn": self._compute_covariance_reduction_loss,
            },
            "usage_loss": {
                "enabled": max(self.usage_lambda_start, self.usage_lambda_end) > 0.0,
                "weight": self.lambda_use,
                "compute_fn": self._compute_prototype_usage_loss,
            },
            "gate_loss": {
                "enabled": self.lambda_gate > 0.0,
                "weight": self.lambda_gate,
                "compute_fn": self._compute_gate_regularization_loss,
            },
        }
        self.set_epoch_context(epoch_index=0, total_epochs=1)
        print_parameter_summary(
            "MODEL",
            "ThesisMultitaskModel",
            self,
            {
                "encoder": self.encoder,
                "continuous_prototype_bank": self.continuous_prototype_bank,
                "discrete_assignment": self.discrete_assignment,
                "discrete_codebook": self.discrete_codebook,
                "reconstruction_head": self.reconstruction_head,
                "classification_head": self.classification_head,
                "alpha_logit": self.alpha_logit,
                "beta_logit": self.beta_logit,
            },
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            hidden_dim=hidden_dim,
            mlp_num_linear_layers=mlp_num_linear_layers,
            num_classes=num_classes,
            use_label_refurbishment=use_label_refurbishment,
            refurbishment_alpha=refurbishment_alpha,
            refurbishment_beta=refurbishment_beta,
            reconstruction_normal_only=reconstruction_normal_only,
            lambda_cls=lambda_cls,
            lambda_div=lambda_div,
            lambda_var=lambda_var,
            lambda_cov=lambda_cov,
            lambda_use=lambda_use,
            lambda_gate=lambda_gate,
            temperature_start=temperature_start,
            temperature_end=temperature_end,
            temperature_hold_fraction=temperature_hold_fraction,
            usage_lambda_start=self.usage_lambda_start,
            usage_lambda_end=self.usage_lambda_end,
            usage_lambda_schedule_fraction=usage_lambda_schedule_fraction,
            use_synthetic_validation=use_synthetic_validation,
            synthetic_validation_seed=synthetic_validation_seed,
        )

    def _zero_loss(self, reference_tensor: torch.Tensor) -> torch.Tensor:
        return reference_tensor.new_zeros(())

    def _compute_temperature_for_epoch(
        self, epoch_index: int, total_epochs: int
    ) -> float:
        # The temperature schedule is kept inside the model because it changes
        # the discrete branch behavior, not the generic trainer behavior.
        hold_epochs = math.ceil(total_epochs * self.temperature_hold_fraction)
        if epoch_index < hold_epochs:
            return float(self.temperature_start)

        anneal_epoch_index = max(epoch_index - hold_epochs, 0)
        anneal_epochs = max(
            1, math.ceil(total_epochs * self.temperature_anneal_fraction)
        )
        if anneal_epochs == 1:
            progress = 0.0
        else:
            progress = min(anneal_epoch_index / float(anneal_epochs - 1), 1.0)
        if progress <= 0.0:
            return float(self.temperature_start)
        if progress >= 1.0:
            return float(self.temperature_end)
        return float(
            self.temperature_start
            + progress * (self.temperature_end - self.temperature_start)
        )

    def _compute_usage_lambda_for_epoch(
        self, epoch_index: int, total_epochs: int
    ) -> float:
        usage_schedule_epochs = max(
            1, math.ceil(total_epochs * self.usage_lambda_schedule_fraction)
        )
        if usage_schedule_epochs == 1:
            progress = 1.0
        else:
            progress = min(epoch_index / float(usage_schedule_epochs - 1), 1.0)
        if progress <= 0.0:
            return float(self.usage_lambda_start)
        if progress >= 1.0:
            return float(self.usage_lambda_end)
        return float(
            self.usage_lambda_start
            + progress * (self.usage_lambda_end - self.usage_lambda_start)
        )

    def set_epoch_context(self, epoch_index: int, total_epochs: int) -> None:
        # Warm-up can temporarily pin fusion to a known regime so ablations can
        # compare continuous-only, discrete-only, and fused behavior cleanly.
        self.current_epoch_index = epoch_index
        self.current_total_epochs = total_epochs
        self.gumbel_temperature = self._compute_temperature_for_epoch(
            epoch_index, total_epochs
        )
        self.current_usage_lambda = self._compute_usage_lambda_for_epoch(
            epoch_index, total_epochs
        )
        warmup_active = epoch_index < self.freeze_fusion_for_epochs
        self.active_alpha_override = self.warmup_alpha_value if warmup_active else None
        self.active_beta_override = self.warmup_beta_value if warmup_active else None
        self.schedule_state = {
            "epoch": epoch_index + 1,
            "warmup_active": warmup_active,
            "freeze_fusion_for_epochs": self.freeze_fusion_for_epochs,
            "temperature": self.gumbel_temperature,
            "usage_lambda": self.current_usage_lambda,
        }
        console_print(
            "MODEL",
            "Updated multitask epoch context",
            epoch=epoch_index + 1,
            total_epochs=total_epochs,
            warmup_active=warmup_active,
            temperature=self.gumbel_temperature,
            usage_lambda=self.current_usage_lambda,
            alpha_override=self.active_alpha_override,
            beta_override=self.active_beta_override,
        )

    def get_schedule_state(self) -> dict[str, Any]:
        return dict(self.schedule_state)

    def prepare_synthetic_validation_epoch(self) -> None:
        # The synthetic validation path must replay the same corruption pattern
        # every epoch so the auxiliary classification curves are comparable.
        self.synthetic_validation_injector.reset_rng()

    def _continuous_prototype_lookup(self, hidden: torch.Tensor) -> dict[str, Any]:
        # The continuous branch keeps a soft weighted prototype mixture.
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
        # The discrete branch keeps a quantized codebook view of the same tokens.
        discrete_hidden = hidden
        assignment_logits = None
        assignment_probabilities = None
        code_indices = None

        if self.discrete_assignment is not None and self.discrete_codebook is not None:
            
            # Hiện tại, discrete assignment là một lớp linear
            # với tham số học được.
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
        # Fusion is expressed as exact limiting cases of the same equations.
        # That is why continuous-only and discrete-only ablations need no second model.
        if self.active_alpha_override is None:
            alpha = torch.sigmoid(self.alpha_logit)
        else:
            alpha = continuous_hidden.new_tensor(float(self.active_alpha_override))
        if self.active_beta_override is None:
            beta = torch.sigmoid(self.beta_logit)
        else:
            beta = continuous_hidden.new_tensor(float(self.active_beta_override))

        # Beta là mức độ mà tác vụ tái tạo chuỗi (reconstruction) sử dụng
        # nhánh các vec-tơ nguyên mẫu rời rạc (discrete prototype).
        # Mình kì vọng giá trị này sẽ nhỏ.
        hidden_reconstruction = (
            beta * discrete_hidden + (1.0 - beta) * continuous_hidden
        )

        # Alpha là mức độ mà tác vụ phân loại (classification) sử dụng nhánh các
        # vec-tơ nguyên mẫu liên tục.
        # Mình kì vọng giá trị này sẽ lớn.
        hidden_classification = (
            alpha * discrete_hidden + (1.0 - alpha) * continuous_hidden
        )

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
                "warmup_active": self.schedule_state["warmup_active"],
                "temperature": self.gumbel_temperature,
            },
        }

    def _clone_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        cloned_batch: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                cloned_batch[key] = value.clone()
            elif isinstance(value, list):
                cloned_batch[key] = [
                    dict(item) if isinstance(item, dict) else item for item in value
                ]
            else:
                cloned_batch[key] = value
        return cloned_batch

    def _prepare_clean_batch(
        self, batch: dict[str, Any], stage_name: str
    ) -> dict[str, Any]:
        # The model owns augmentation timing because synthetic supervision is
        # part of the multitask objective, not a separate preprocessing pipeline.
        if (
            "classification_labels" in batch
            and "synthetic_anomaly_mask" in batch
            and "augmentation_metadata" in batch
        ):
            console_print(
                stage_name.upper(),
                "Received pre-augmented multitask batch",
                **summarize_batch(batch),
                classification_label_distribution=summarize_label_distribution(
                    batch["classification_labels"]
                ),
            )
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
                "anomaly_family_index": None,
                "start_index": None,
                "end_index": None,
                "affected_channels": [],
                "family_parameters_by_channel": {},
            }
            for _ in range(batch_size)
        ]
        if prepared_batch["point_labels"] is None:
            prepared_batch["point_labels"] = prepared_batch[
                "synthetic_anomaly_mask"
            ].clone()
        console_print(
            stage_name.upper(),
            "Prepared clean multitask batch",
            **summarize_batch(prepared_batch),
            classification_label_distribution=summarize_label_distribution(
                prepared_batch["classification_labels"]
            ),
        )
        return prepared_batch

    def _prepare_batch(self, batch: dict[str, Any], stage_name: str) -> dict[str, Any]:
        if stage_name == "val_synth" and self.use_synthetic_validation:
            return self.synthetic_validation_injector.augment_batch(batch)
        return self._prepare_clean_batch(batch, stage_name)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        # The forward pass is the main representation story of the thesis model:
        # encode once, build two branch views, fuse per task, then score.
        validate_batch(batch)
        console_print(
            "MODEL", "Multitask forward input batch", **summarize_batch(batch)
        )
        forward_start_time = time.perf_counter()

        # Truyền lô dữ liệu qua encoder để mã hoá
        # Thu được các vec-tơ ẩn (hidden vector)
        # Một vec-tơ ẩn cho mỗi bước thời gian (timestep)
        encoder_outputs = self.encoder(batch)
        hidden = encoder_outputs["hidden"]

        # Tái tạo các vec-tơ ẩn sử dụng các vec-tơ nguyên mẫu liên tục
        continuous_outputs = self._continuous_prototype_lookup(hidden)

        # Tái tạo các vec-tơ ẩn sử dụng các vec-tơ nguyên mẫu rời rạc
        discrete_outputs = self._discrete_prototype_lookup(hidden)

        # Kết hợp vec-tơ ẩn từ hai nhánh lại với nhau
        fusion_outputs = self._compute_fusion_outputs(
            continuous_hidden=continuous_outputs["prototype_context"],
            discrete_hidden=discrete_outputs["quantized_hidden"],
        )

        # Lấy ra vec-tơ kết hợp dùng cho từng tác vụ
        hidden_reconstruction = fusion_outputs["hidden_reconstruction"]
        hidden_classification = fusion_outputs["hidden_classification"]

        # Truyền vec-tơ kết hợp dùng cho
        # tác vụ tái tạo qua mạng tái tạo (reconstruction head)
        recon = self.reconstruction_head(hidden_reconstruction)

        # Xét vec-tơ kết hợp dùng cho tác vụ phân loại,
        # cộng tất cả các vec-tơ ở từng bước thời gian lại với nhau.
        # Xong, chia cho số bước thời gian để lấy vec-tơ trung bình.
        # dim=1 nghĩa là xem mỗi bước thời gian là một hạng tử.
        # Rồi, lấy trung bình.
        pooled_classification_hidden = hidden_classification.mean(dim=1)
        logits = self.classification_head(pooled_classification_hidden)

        # Độ bất thường được tính bằng cách
        # Tính toán độ lỗi (error) giữa bản gốc và bản tái tạo
        # ở từng bước thời gian.
        # Xong, bình phương tất cả độ lỗi lên rồi cộng lại chia lấy trung bình.
        point_scores = torch.mean((recon - batch["x"]) ** 2, dim=-1)

        # Chuẩn bị đầu ra theo thoả thuận (contract) đã được thiết lập.
        # Xem: src/core/contracts.py
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
                "forward_pass_seconds": time.perf_counter() - forward_start_time,
            },
        }
        validate_model_outputs(outputs)
        console_print(
            "MODEL",
            "Multitask forward outputs",
            hidden=summarize_tensor(outputs["hidden"]),
            pooled=summarize_tensor(outputs["pooled"]),
            recon=summarize_tensor(outputs["recon"]),
            logits=summarize_tensor(outputs["logits"]),
            point_scores=summarize_tensor(outputs["point_scores"]),
            window_scores=summarize_tensor(outputs["window_scores"]),
            continuous_hidden=summarize_tensor(
                outputs["aux"]["continuous_branch"]["prototype_context"]
            ),
            discrete_hidden=summarize_tensor(
                outputs["aux"]["discrete_branch"]["quantized_hidden"]
            ),
            assignment_probabilities=summarize_tensor(
                outputs["aux"]["discrete_branch"]["assignment_probabilities"]
            ),
            hidden_reconstruction=summarize_tensor(
                outputs["aux"]["hidden_reconstruction"]
            ),
            hidden_classification=summarize_tensor(
                outputs["aux"]["hidden_classification"]
            ),
            forward_pass_seconds=outputs["aux"]["forward_pass_seconds"],
        )
        return outputs

    def _normalize_branch_tokens(
        self, branch_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_hidden = self.branch_layer_norm(branch_hidden).reshape(
            -1, self.hidden_dim
        )
        feature_mean = normalized_hidden.mean(dim=0, keepdim=True)
        feature_std = normalized_hidden.std(dim=0, unbiased=False, keepdim=True)
        standardized_hidden = (normalized_hidden - feature_mean) / (
            feature_std + self.epsilon
        )
        return normalized_hidden, standardized_hidden

    def _compute_reconstruction_loss(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> torch.Tensor:
        squared_reconstruction_error = (outputs["recon"] - batch["x"]) ** 2
        if not self.reconstruction_normal_only or "synthetic_anomaly_mask" not in batch:
            return torch.mean(squared_reconstruction_error)

        normal_time_step_mask = self._build_normal_time_step_mask(
            batch, squared_reconstruction_error
        )
        expanded_normal_mask = normal_time_step_mask.unsqueeze(-1).expand_as(
            squared_reconstruction_error
        )
        active_normal_cells = torch.count_nonzero(expanded_normal_mask)
        if int(active_normal_cells.item()) == 0:
            return torch.mean(squared_reconstruction_error)

        return (
            torch.sum(squared_reconstruction_error * expanded_normal_mask)
            / expanded_normal_mask.sum()
        )

    def _compute_classification_loss(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> torch.Tensor:
        if self.use_label_refurbishment:
            target_probabilities = self._build_refurbished_binary_targets(
                batch["classification_labels"],
                outputs["logits"].dtype,
            )
            log_probabilities = F.log_softmax(outputs["logits"], dim=-1)
            return torch.mean(
                torch.sum(-target_probabilities * log_probabilities, dim=-1)
            )

        return F.cross_entropy(outputs["logits"], batch["classification_labels"].long())

    def _build_normal_time_step_mask(
        self,
        batch: dict[str, Any],
        reference_tensor: torch.Tensor,
    ) -> torch.Tensor:
        anomaly_mask = batch["synthetic_anomaly_mask"].to(
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
        )
        normal_time_step_mask = 1.0 - anomaly_mask
        return torch.clamp(normal_time_step_mask, min=0.0, max=1.0)

    def _build_refurbished_binary_targets(
        self,
        classification_labels: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.num_classes != 2:
            raise ValueError("Binary label refurbishment requires num_classes == 2")

        hard_labels = classification_labels.long()
        target_probabilities = torch.zeros(
            hard_labels.shape[0],
            self.num_classes,
            device=hard_labels.device,
            dtype=target_dtype,
        )
        target_probabilities[:, 0] = torch.where(
            hard_labels == 0,
            1.0 - self.refurbishment_beta,
            self.refurbishment_alpha,
        )
        target_probabilities[:, 1] = torch.where(
            hard_labels == 0,
            self.refurbishment_beta,
            1.0 - self.refurbishment_alpha,
        )
        return target_probabilities

    def _compute_cross_branch_diversity_loss(
        self, outputs: dict[str, Any]
    ) -> torch.Tensor:
        # This discourages the two branches from collapsing onto the same signal.
        continuous_hidden = outputs["aux"]["continuous_branch"]["prototype_context"]
        discrete_hidden = outputs["aux"]["discrete_branch"]["quantized_hidden"]
        _, standardized_continuous = self._normalize_branch_tokens(continuous_hidden)
        _, standardized_discrete = self._normalize_branch_tokens(discrete_hidden)
        num_tokens = standardized_continuous.shape[0]
        cross_branch_correlation = (
            standardized_continuous.T @ standardized_discrete / num_tokens
        )
        return cross_branch_correlation.pow(2).mean()

    def _compute_variance_floor_loss(self, outputs: dict[str, Any]) -> torch.Tensor:
        variance_losses: list[torch.Tensor] = []
        for branch_name in ["continuous_branch", "discrete_branch"]:
            branch_hidden = outputs["aux"][branch_name][
                "prototype_context"
                if branch_name == "continuous_branch"
                else "quantized_hidden"
            ]
            normalized_hidden, _ = self._normalize_branch_tokens(branch_hidden)
            feature_std = normalized_hidden.std(dim=0, unbiased=False)
            variance_losses.append(
                F.relu(self.variance_floor_gamma - feature_std).pow(2).mean()
            )
        return torch.stack(variance_losses).sum()

    def _compute_covariance_reduction_loss(
        self, outputs: dict[str, Any]
    ) -> torch.Tensor:
        covariance_losses: list[torch.Tensor] = []
        for branch_name in ["continuous_branch", "discrete_branch"]:
            branch_hidden = outputs["aux"][branch_name][
                "prototype_context"
                if branch_name == "continuous_branch"
                else "quantized_hidden"
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
                    off_diagonal_matrix.pow(2).sum()
                    / (self.hidden_dim * (self.hidden_dim - 1))
                )
        return torch.stack(covariance_losses).sum()

    # Version 1 with pure simple loss (only classification and reconstruction) was trained.
    # Based on real diagnostic, this usage loss will be added to prevent over-centralizing in few discrete prototypes.
    # So in version 2, the final loss function has 3 terms: one for classification, one for reconstruction
    # and one for usage of discrete prototypes.
    def _compute_prototype_usage_loss(self, outputs: dict[str, Any]) -> torch.Tensor:
        # Usage balancing is the main protection against dead or ignored codes.
        assignment_probabilities = outputs["aux"]["discrete_branch"][
            "assignment_probabilities"
        ]
        if assignment_probabilities is None or self.discrete_codebook_size <= 0:
            return self._zero_loss(outputs["hidden"])
        average_usage = assignment_probabilities.mean(dim=(0, 1))
        target_usage = torch.full_like(average_usage, 1.0 / self.discrete_codebook_size)
        return torch.sum((average_usage - target_usage) ** 2)

    def _compute_gate_regularization_loss(
        self, outputs: dict[str, Any]
    ) -> torch.Tensor:
        # Gate entropy regularization keeps the fusion scalars from collapsing
        # too confidently unless the data actually supports that decision.
        alpha = outputs["aux"]["alpha"]
        beta = outputs["aux"]["beta"]
        max_entropy = math.log(2.0)
        alpha_clamped = torch.clamp(alpha, self.epsilon, 1.0 - self.epsilon)
        beta_clamped = torch.clamp(beta, self.epsilon, 1.0 - self.epsilon)
        alpha_entropy = -(
            alpha_clamped * torch.log(alpha_clamped)
            + (1.0 - alpha_clamped) * torch.log(1.0 - alpha_clamped)
        )
        beta_entropy = -(
            beta_clamped * torch.log(beta_clamped)
            + (1.0 - beta_clamped) * torch.log(1.0 - beta_clamped)
        )
        alpha_penalty = 1.0 - alpha_entropy / max_entropy
        beta_penalty = 1.0 - beta_entropy / max_entropy
        return 0.5 * (alpha_penalty + beta_penalty)

    def _compute_optional_loss_terms(
        self, outputs: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        optional_loss_values: dict[str, torch.Tensor] = {}
        for loss_name, loss_config in self.optional_loss_configs.items():
            compute_fn: Callable[[dict[str, Any]], torch.Tensor] = loss_config[
                "compute_fn"
            ]
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
        classification_weight: float,
    ) -> torch.Tensor:
        # The weighted sum is intentionally explicit so readers can map each
        # `lambda_*` config field directly to one line of the objective. The
        # default beginning of training is still the small objective
        # `L_recon + lambda_cls * L_cls`.
        total_loss = reconstruction_loss + classification_weight * classification_loss
        for loss_name, loss_value in optional_loss_values.items():
            loss_weight = self.optional_loss_configs[loss_name]["weight"]
            if loss_name == "usage_loss":
                loss_weight = self.current_usage_lambda
            total_loss = total_loss + loss_weight * loss_value
        return total_loss

    def _build_stage_log(
        self,
        stage_name: str,
        outputs: dict[str, Any],
        loss_terms: dict[str, torch.Tensor],
        batch: dict[str, Any],
        *,
        include_classification_metrics: bool,
    ) -> dict[str, float]:
        # These logs are part of the branch-collapse observability surface, not
        # just convenience metrics. They are meant to support ablation reading.
        assignment_probabilities = outputs["aux"]["discrete_branch"][
            "assignment_probabilities"
        ]
        if assignment_probabilities is None or self.discrete_codebook_size <= 0:
            discrete_usage_top1 = 0.0
            discrete_usage_entropy = 0.0
            discrete_usage_concentration = 0.0
            discrete_usage_active_codes = 0.0
        else:
            average_usage = assignment_probabilities.mean(dim=(0, 1))
            average_usage = average_usage / average_usage.sum().clamp_min(self.epsilon)
            discrete_usage_top1 = float(average_usage.max().detach().cpu())
            discrete_usage_entropy = float(
                (
                    -(
                        average_usage * torch.log(average_usage.clamp_min(self.epsilon))
                    ).sum()
                )
                .detach()
                .cpu()
            )
            discrete_usage_concentration = float(
                torch.sum(average_usage.pow(2)).detach().cpu()
            )
            discrete_usage_active_codes = float(
                torch.sum(
                    (
                        average_usage > (1.0 / max(self.discrete_codebook_size * 2, 1))
                    ).float()
                )
                .detach()
                .cpu()
            )
        stage_log = {
            f"{stage_name}_loss": float(loss_terms["total_loss"].detach().cpu()),
            f"{stage_name}_reconstruction_loss": float(
                loss_terms["reconstruction_loss"].detach().cpu()
            ),
            f"{stage_name}_diversity_loss": float(
                loss_terms["diversity_loss"].detach().cpu()
            ),
            f"{stage_name}_variance_loss": float(
                loss_terms["variance_loss"].detach().cpu()
            ),
            f"{stage_name}_covariance_loss": float(
                loss_terms["covariance_loss"].detach().cpu()
            ),
            f"{stage_name}_usage_loss": float(loss_terms["usage_loss"].detach().cpu()),
            f"{stage_name}_gate_loss": float(loss_terms["gate_loss"].detach().cpu()),
            f"{stage_name}_alpha": float(outputs["aux"]["alpha"].detach().cpu()),
            f"{stage_name}_beta": float(outputs["aux"]["beta"].detach().cpu()),
            f"{stage_name}_continuous_norm": float(
                outputs["aux"]["continuous_branch"]["prototype_context"]
                .norm(dim=-1)
                .mean()
                .detach()
                .cpu()
            ),
            f"{stage_name}_discrete_norm": float(
                outputs["aux"]["discrete_branch"]["quantized_hidden"]
                .norm(dim=-1)
                .mean()
                .detach()
                .cpu()
            ),
            f"{stage_name}_discrete_usage_top1": discrete_usage_top1,
            f"{stage_name}_discrete_usage_entropy": discrete_usage_entropy,
            f"{stage_name}_discrete_usage_concentration": discrete_usage_concentration,
            f"{stage_name}_discrete_usage_active_codes": discrete_usage_active_codes,
            f"{stage_name}_temperature": float(self.gumbel_temperature),
            f"{stage_name}_usage_lambda": float(self.current_usage_lambda),
            f"{stage_name}_warmup_active": float(self.schedule_state["warmup_active"]),
        }
        if include_classification_metrics:
            predicted_labels = torch.argmax(outputs["logits"], dim=-1)
            classification_accuracy = float(
                (predicted_labels == batch["classification_labels"])
                .float()
                .mean()
                .detach()
                .cpu()
            )
            stage_log[f"{stage_name}_classification_loss"] = float(
                loss_terms["classification_loss"].detach().cpu()
            )
            stage_log[f"{stage_name}_classification_accuracy"] = classification_accuracy
        return stage_log

    def _shared_step(
        self,
        batch: dict[str, Any],
        stage_name: str,
        *,
        classification_weight: float,
        include_classification_metrics: bool,
    ) -> dict[str, Any]:
        # This is the one place where the actual multitask training objective is assembled.
        prepared_batch = self._prepare_batch(batch, stage_name)
        outputs = self.forward(prepared_batch)

        # Tính toán các hàm loss thành phần
        reconstruction_loss = self._compute_reconstruction_loss(outputs, prepared_batch)
        classification_loss = self._compute_classification_loss(outputs, prepared_batch)
        optional_loss_values = self._compute_optional_loss_terms(outputs)

        # Tính toán hàm loss tổng
        total_loss = self._compute_total_loss(
            reconstruction_loss=reconstruction_loss,
            classification_loss=classification_loss,
            optional_loss_values=optional_loss_values,
            classification_weight=classification_weight,
        )

        loss_terms = {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "classification_loss": classification_loss,
            **optional_loss_values,
        }
        console_print(
            stage_name.upper(),
            "Completed multitask stage step",
            batch_size=prepared_batch["x"].shape[0],
            total_loss=float(total_loss.detach().cpu()),
            reconstruction_loss=float(reconstruction_loss.detach().cpu()),
            classification_loss=float(classification_loss.detach().cpu()),
            diversity_loss=float(optional_loss_values["diversity_loss"].detach().cpu()),
            variance_loss=float(optional_loss_values["variance_loss"].detach().cpu()),
            covariance_loss=float(
                optional_loss_values["covariance_loss"].detach().cpu()
            ),
            usage_loss=float(optional_loss_values["usage_loss"].detach().cpu()),
            gate_loss=float(optional_loss_values["gate_loss"].detach().cpu()),
            classification_label_distribution=summarize_label_distribution(
                prepared_batch["classification_labels"]
            ),
            alpha=float(outputs["aux"]["alpha"].detach().cpu()),
            beta=float(outputs["aux"]["beta"].detach().cpu()),
            forward_pass_seconds=outputs["aux"]["forward_pass_seconds"],
        )
        return {
            "loss": total_loss,
            "log": self._build_stage_log(
                stage_name,
                outputs,
                loss_terms,
                prepared_batch,
                include_classification_metrics=include_classification_metrics,
            ),
            "outputs": outputs,
            "loss_terms": loss_terms,
            "batch": prepared_batch,
        }

    def training_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(
            batch=batch,
            stage_name="train",
            classification_weight=self.lambda_cls,
            include_classification_metrics=True,
        )

    def validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(
            batch=batch,
            stage_name="val",
            classification_weight=0.0,
            include_classification_metrics=False,
        )

    def synthetic_validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(
            batch=batch,
            stage_name="val_synth",
            classification_weight=self.lambda_cls,
            include_classification_metrics=True,
        )

    def test_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(
            batch=batch,
            stage_name="test",
            classification_weight=0.0,
            include_classification_metrics=False,
        )
