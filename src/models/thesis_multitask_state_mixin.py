from __future__ import annotations

"""Mixin extracted from the thesis multitask model.

This file keeps constructor and configuration plumbing together so the main
model file can stay below the code-size limit without changing runtime
behavior.
"""

import math
import time
from collections import OrderedDict, deque
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
from src.data.augment import SyntheticAnomalyInjector
from src.models.thesis_multitask_components import (
    STAGE3_PHASE_CANONICAL_NAME,
    STAGE3_PHASE_LEGACY_NAME,
    TWO_STAGE_A_PHASE_NAME,
    TWO_STAGE_B_PHASE_NAME,
    TWO_STAGE_PHASE_NAMES,
    REDLAMP_ANOMALY_FAMILIES,
    REDLAMP_MULTICLASS_CLASS_NAMES,
    MultitaskArchitectureConfig,
    MultitaskWindowEncoder,
    ObjectiveConfig,
    MemoryInitializationConfig,
    PrototypeBranchConfig,
    ScheduleAndWarmupConfig,
    SyntheticAnomalyConfig,
    ThesisMultitaskModelConfig,
    build_multilayer_perceptron,
)


class ThesisMultitaskStateMixin:
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
            "bootstrap_active": self._is_bootstrap_active(),
            "train_memory_mode": float(
                not self._should_bypass_memory_for_stage("train")
            ),
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
            bootstrap_active=self.schedule_state["bootstrap_active"],
            train_memory_mode=self.schedule_state["train_memory_mode"],
        )

    def get_schedule_state(self) -> dict[str, Any]:
        return dict(self.schedule_state)

    def _is_bootstrap_active(self) -> bool:
        return (
            self.bootstrap_encoder_epochs > 0
            and self.current_epoch_index < self.bootstrap_encoder_epochs
            and not self.memory_initialized
        )

    def _should_bypass_memory_for_stage(self, stage_name: str) -> bool:
        del stage_name
        if not self._phase_uses_prototype_path():
            return True
        return self._is_bootstrap_active() or not self.memory_initialized

    def _should_update_memory(self, stage_name: str) -> bool:
        return (
            stage_name == "train"
            and self.memory_training_enabled
            and self.memory_initialized
            and self._phase_uses_prototype_path()
            and not self.freeze_memories_after_initialization
        )

    def _semantic_stage_label(self) -> str:
        # Active two-stage runs should read as Stage A or Stage B.
        # The label is stage-facing even when the runtime field still uses the
        # historical training_phase name internally.
        if self.training_phase == TWO_STAGE_A_PHASE_NAME:
            return "Stage A: Multitask Pretraining"
        if self.training_phase == TWO_STAGE_B_PHASE_NAME:
            return "Stage B: Fusion Finetuning"
        if self.training_phase == STAGE3_PHASE_CANONICAL_NAME:
            return "Stage 3: Memory Initialization and Fusion Warm-Up"
        return self.training_phase

    def _memory_initialization_substep_active(self) -> bool:
        return self.training_phase in {
            STAGE3_PHASE_CANONICAL_NAME,
            TWO_STAGE_B_PHASE_NAME,
        }

    def _fusion_warmup_substep_active(self) -> bool:
        return self.training_phase in {
            STAGE3_PHASE_CANONICAL_NAME,
            TWO_STAGE_B_PHASE_NAME,
        }

    def _trainable_module_names(self) -> list[str]:
        trainable_module_names: list[str] = []
        trainable_modules = {
            "classification_concat_projection": self.classification_concat_projection,
            "classification_fusion_gate": self.classification_fusion_gate,
            "classification_head": self.classification_head,
            "continuous_update_gate": self.continuous_update_gate,
            "discrete_assignment": self.discrete_assignment,
            "encoder": self.encoder,
            "reconstruction_concat_projection": self.reconstruction_concat_projection,
            "reconstruction_fusion_gate": self.reconstruction_fusion_gate,
            "reconstruction_head": self.reconstruction_head,
        }
        for module_name, module in trainable_modules.items():
            if module is None:
                continue
            if any(parameter.requires_grad for parameter in module.parameters()):
                trainable_module_names.append(module_name)
        return sorted(trainable_module_names)

    def get_memory_lifecycle_state(self) -> dict[str, Any]:
        return {
            "bootstrap_encoder_epochs": self.bootstrap_encoder_epochs,
            "current_epoch": self.current_epoch_index + 1,
            # The lifecycle state is stage-facing for active two-stage runs.
            "semantic_stage_label": self._semantic_stage_label(),
            "memory_initialization_substep": (
                self._memory_initialization_substep_active()
            ),
            "fusion_warmup_substep": self._fusion_warmup_substep_active(),
            "recovered_zipped_encoder_frozen_during_warmup": (
                self._phase_freezes_encoder()
            ),
            "trainable_module_names": self._trainable_module_names(),
            "memory_initialized": self.memory_initialized,
            "memory_training_enabled": self.memory_training_enabled,
            "memory_ready_for_initialization": self.memory_ready_for_initialization,
            "memory_initialization_epoch": self.memory_initialization_epoch,
            "continuous_memory_source_label": (
                "normal_only_recovered_training_features"
            ),
            "discrete_memory_source_label": (
                "class_stratified_recovered_training_features"
            ),
            "discrete_memory_label_source": self.discrete_memory_label_source,
            "memory_mode": float(not self._should_bypass_memory_for_stage("train")),
            "train_memory_mode": float(
                not self._should_bypass_memory_for_stage("train")
            ),
        }

    def get_checkpoint_extra_state(self) -> dict[str, Any]:
        state = self.get_memory_lifecycle_state()
        if isinstance(self.anomalous_codeword_mask, torch.Tensor):
            state["anomalous_codeword_mask"] = (
                self.anomalous_codeword_mask.detach().cpu().tolist()
            )
        if isinstance(self.anomaly_radii, torch.Tensor):
            state["anomaly_radii"] = self.anomaly_radii.detach().cpu().tolist()
        state["verification_metadata_source"] = self.verification_metadata_source
        state["verification_metadata_schema_version"] = int(
            getattr(self, "verification_metadata_schema_version", 1)
        )
        state["verification_metadata_split"] = str(
            getattr(self, "verification_metadata_split", "synthetic_train")
        )
        state["verification_metadata_initialization_seed"] = int(
            getattr(self, "verification_metadata_initialization_seed", 0)
        )
        if isinstance(self.verification_codeword_class_ids, torch.Tensor):
            state["verification_codeword_class_ids"] = (
                self.verification_codeword_class_ids.detach().cpu().tolist()
            )
        if isinstance(self.verification_contributing_token_counts, torch.Tensor):
            state["verification_contributing_token_counts"] = (
                self.verification_contributing_token_counts.detach().cpu().tolist()
            )
        state["verification_radius_quantile"] = 0.99
        state["verification_metadata_label_source"] = (
            self.discrete_memory_label_source
        )
        state["stochastic_inference"] = self.stochastic_inference
        state["monte_carlo_samples"] = self.monte_carlo_samples
        state["continuous_temperature"] = self.continuous_temperature
        state["discrete_temperature"] = self.discrete_temperature
        state["variance_correction"] = self.variance_correction
        state["return_mc_samples"] = self.return_mc_samples
        state["sample_retention_policy"] = self.sample_retention_policy
        return state

    def get_memory_tensor_state(self) -> dict[str, torch.Tensor | None]:
        return {
            "continuous_prototype_bank": (
                None
                if self.continuous_prototype_bank is None
                else self.continuous_prototype_bank.detach().clone()
            ),
            "discrete_codebook": (
                None
                if self.discrete_codebook is None
                else self.discrete_codebook.detach().clone()
            ),
            "discrete_ema_counts": (
                None
                if self.discrete_ema_counts is None
                else self.discrete_ema_counts.detach().clone()
            ),
            "discrete_ema_sums": (
                None
                if self.discrete_ema_sums is None
                else self.discrete_ema_sums.detach().clone()
            ),
        }

    def mark_memories_initialized(
        self, initialization_epoch: int | None = None
    ) -> None:
        self.memory_initialized = True
        self.memory_training_enabled = not self.freeze_memories_after_initialization
        self.memory_ready_for_initialization = False
        self.memory_initialization_epoch = initialization_epoch
        console_print(
            "MODEL",
            "Marked prototype memories as initialized",
            initialization_epoch=initialization_epoch,
            memory_state=self.get_memory_lifecycle_state(),
        )

    def maybe_initialize_memories_from_loader(
        self,
        train_loader: Any,
        device: str,
    ) -> bool:
        if not self._phase_uses_prototype_path():
            return False
        if self.memory_initialized:
            return False
        if self.current_epoch_index < self.bootstrap_encoder_epochs:
            return False
        self.memory_ready_for_initialization = True
        token_pool = self._collect_memory_initialization_token_pool_from_loader(
            train_loader,
            device,
        )
        continuous_hidden_tokens = token_pool["continuous_hidden_tokens"]
        if continuous_hidden_tokens.shape[0] == 0:
            console_print(
                "MODEL",
                "No normal hidden tokens were available for memory initialization",
                epoch=self.current_epoch_index + 1,
                num_batches_used=token_pool["num_batches_used"],
            )
            return False
        self._initialize_memory_buffers_from_token_pool(
            continuous_hidden_tokens=continuous_hidden_tokens,
            discrete_hidden_tokens_by_class=token_pool[
                "discrete_hidden_tokens_by_class"
            ],
        )
        self.mark_memories_initialized(
            initialization_epoch=self.current_epoch_index + 1
        )
        console_print(
            "MODEL",
            "Initialized frozen prototype memories from recovered training features",
            epoch=self.current_epoch_index + 1,
            bootstrap_encoder_epochs=self.bootstrap_encoder_epochs,
            num_batches_used=token_pool["num_batches_used"],
            num_continuous_normal_tokens=token_pool["num_continuous_normal_tokens"],
            discrete_classes_with_tokens=sorted(
                token_pool["num_discrete_class_tokens_by_class"].keys()
            ),
            continuous_memory_source_label=("normal_only_recovered_training_features"),
            discrete_memory_source_label=(
                "class_stratified_recovered_training_features"
            ),
            discrete_memory_label_source=self.discrete_memory_label_source,
        )
        return True

    def load_checkpoint_extra_state(self, extra_state: dict[str, Any] | None) -> None:
        if not extra_state:
            return
        if "verification_metadata_schema_version" in extra_state and int(
            extra_state["verification_metadata_schema_version"]
        ) != 1:
            raise ValueError("verification_metadata_schema_version must be 1")
        if "verification_metadata_split" in extra_state and extra_state.get(
            "verification_metadata_split"
        ) != "synthetic_train":
            raise ValueError("verification_metadata_split must be synthetic_train")
        if "verification_radius_quantile" in extra_state and not (
            0.0 < float(extra_state["verification_radius_quantile"]) <= 1.0
        ):
            raise ValueError("verification_radius_quantile must be in (0, 1]")
        if "verification_metadata_label_source" in extra_state and extra_state.get(
            "verification_metadata_label_source"
        ) != self.discrete_memory_label_source:
            raise ValueError(
                "verification_metadata_label_source must match discrete_memory_label_source"
            )
        if "verification_metadata_initialization_seed" in extra_state and int(
            extra_state["verification_metadata_initialization_seed"]
        ) < 0:
            raise ValueError("verification_metadata_initialization_seed must be non-negative")
        self.memory_initialized = bool(
            extra_state.get("memory_initialized", self.memory_initialized)
        )
        self.memory_training_enabled = bool(
            extra_state.get("memory_training_enabled", self.memory_training_enabled)
        )
        self.memory_ready_for_initialization = bool(
            extra_state.get(
                "memory_ready_for_initialization",
                self.memory_ready_for_initialization,
            )
        )
        self.memory_initialization_epoch = extra_state.get(
            "memory_initialization_epoch",
            self.memory_initialization_epoch,
        )
        mask = extra_state.get("anomalous_codeword_mask")
        radii = extra_state.get("anomaly_radii")
        if mask is not None and radii is not None:
            if not isinstance(self.discrete_codebook, torch.Tensor):
                raise ValueError("verification metadata requires a discrete codebook")
            mask = torch.as_tensor(mask, dtype=torch.bool)
            radii = torch.as_tensor(radii, dtype=torch.float32)
            if mask.shape != (self.discrete_codebook.shape[0],):
                raise ValueError("anomalous_codeword_mask checkpoint shape mismatch")
            if (
                radii.shape != mask.shape
                or not torch.isfinite(radii).all().item()
                or (radii < 0).any().item()
            ):
                raise ValueError("anomaly_radii checkpoint shape or value mismatch")
            self.anomalous_codeword_mask = mask.detach().bool().clone()
            self.anomaly_radii = radii.detach().float().clone()
            self.verification_metadata_source = str(
                extra_state.get("verification_metadata_source", "checkpoint")
            )
            if "verification_codeword_class_ids" in extra_state:
                codeword_class_ids = torch.as_tensor(
                    extra_state["verification_codeword_class_ids"], dtype=torch.long
                )
                if codeword_class_ids.shape != mask.shape:
                    raise ValueError(
                        "verification_codeword_class_ids checkpoint shape mismatch"
                    )
                self.verification_codeword_class_ids = (
                    codeword_class_ids.detach().clone()
                )
            if "verification_contributing_token_counts" in extra_state:
                contributing_token_counts = torch.as_tensor(
                    extra_state["verification_contributing_token_counts"],
                    dtype=torch.float32,
                )
                if contributing_token_counts.shape != mask.shape:
                    raise ValueError(
                        "verification_contributing_token_counts checkpoint shape mismatch"
                    )
                self.verification_contributing_token_counts = (
                    contributing_token_counts.detach().clone()
                )
            self.verification_metadata_schema_version = int(
                extra_state.get("verification_metadata_schema_version", 1)
            )
            self.verification_metadata_split = str(
                extra_state.get("verification_metadata_split", "synthetic_train")
            )
            self.verification_metadata_initialization_seed = int(
                extra_state.get("verification_metadata_initialization_seed", 0)
            )
        if "stochastic_inference" in extra_state:
            if bool(extra_state["stochastic_inference"]) != self.stochastic_inference:
                raise ValueError("checkpoint stochastic_inference does not match model")
        if "monte_carlo_samples" in extra_state:
            if int(extra_state["monte_carlo_samples"]) != self.monte_carlo_samples:
                raise ValueError("checkpoint monte_carlo_samples does not match model")
        if "variance_correction" in extra_state:
            if int(extra_state["variance_correction"]) != self.variance_correction:
                raise ValueError("checkpoint variance_correction does not match model")
        if "sample_retention_policy" in extra_state:
            if str(extra_state["sample_retention_policy"]) != self.sample_retention_policy:
                raise ValueError(
                    "checkpoint sample_retention_policy does not match model"
                )

    def _move_initialization_batch_to_device(
        self,
        batch: dict[str, Any],
        device: str,
    ) -> dict[str, Any]:
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _normalize_memory_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        return F.normalize(vectors, dim=-1, eps=self.memory_norm_epsilon)

    def _normalize_hidden_for_memory(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.normalize(hidden, dim=-1, eps=self.memory_norm_epsilon)

    def _select_covering_vectors(
        self,
        candidate_vectors: torch.Tensor,
        num_vectors: int,
    ) -> torch.Tensor:
        if candidate_vectors.shape[0] == 0:
            raise ValueError("candidate_vectors must contain at least one token")

        normalized_vectors = self._normalize_memory_vectors(candidate_vectors)
        if normalized_vectors.shape[0] <= num_vectors:
            repeated_indices = (
                torch.arange(
                    num_vectors,
                    device=normalized_vectors.device,
                )
                % normalized_vectors.shape[0]
            )
            return normalized_vectors.index_select(0, repeated_indices)

        mean_vector = normalized_vectors.mean(dim=0, keepdim=True)
        squared_distances_to_mean = torch.sum(
            (normalized_vectors - mean_vector) ** 2,
            dim=1,
        )
        first_index = int(torch.argmin(squared_distances_to_mean).item())
        selected_indices = [first_index]
        minimum_squared_distances = torch.sum(
            (normalized_vectors - normalized_vectors[first_index]) ** 2,
            dim=1,
        )

        while len(selected_indices) < num_vectors:
            next_index = int(torch.argmax(minimum_squared_distances).item())
            selected_indices.append(next_index)
            next_squared_distances = torch.sum(
                (normalized_vectors - normalized_vectors[next_index]) ** 2,
                dim=1,
            )
            minimum_squared_distances = torch.minimum(
                minimum_squared_distances,
                next_squared_distances,
            )

        selected_index_tensor = torch.tensor(
            selected_indices,
            device=normalized_vectors.device,
        )
        return normalized_vectors.index_select(0, selected_index_tensor)

    def _run_kmeans(
        self,
        tokens: torch.Tensor,
        k: int,
        *,
        num_iterations: int,
    ) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [N, H]")
        if tokens.shape[0] == 0:
            raise ValueError("tokens must contain at least one row")
        if k <= 0:
            raise ValueError("k must be positive")
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")

        normalized_tokens = self._normalize_memory_vectors(tokens)
        num_tokens = int(normalized_tokens.shape[0])
        if num_tokens <= k:
            repeated_indices = (
                torch.arange(k, device=normalized_tokens.device) % num_tokens
            )
            return normalized_tokens.index_select(0, repeated_indices)

        # Deterministic initialization:
        # 1. choose the token closest to the mean as the first center
        # 2. iteratively choose the farthest token from the current center set
        token_mean = normalized_tokens.mean(dim=0, keepdim=True)
        squared_distances_to_mean = torch.sum(
            (normalized_tokens - token_mean) ** 2,
            dim=1,
        )
        first_index = int(torch.argmin(squared_distances_to_mean).item())
        selected_indices = [first_index]

        minimum_squared_distances = torch.sum(
            (normalized_tokens - normalized_tokens[first_index]) ** 2,
            dim=1,
        )
        while len(selected_indices) < k:
            next_index = int(torch.argmax(minimum_squared_distances).item())
            selected_indices.append(next_index)
            next_squared_distances = torch.sum(
                (normalized_tokens - normalized_tokens[next_index]) ** 2,
                dim=1,
            )
            minimum_squared_distances = torch.minimum(
                minimum_squared_distances,
                next_squared_distances,
            )

        centers = normalized_tokens.index_select(
            0,
            torch.tensor(selected_indices, device=normalized_tokens.device),
        )

        for _ in range(num_iterations):
            pairwise_distances = torch.cdist(normalized_tokens, centers, p=2)
            assignments = torch.argmin(pairwise_distances, dim=1)
            updated_centers: list[torch.Tensor] = []
            for center_index in range(k):
                cluster_mask = assignments == center_index
                if not torch.any(cluster_mask):
                    updated_centers.append(centers[center_index])
                    continue
                cluster_tokens = normalized_tokens[cluster_mask]
                updated_centers.append(cluster_tokens.mean(dim=0))
            centers = torch.stack(updated_centers, dim=0)
            centers = self._normalize_memory_vectors(centers)

        return centers

    def _collect_memory_initialization_token_pool_from_loader(
        self,
        train_loader: Any,
        device: str,
    ) -> dict[str, Any]:
        continuous_hidden_token_groups: list[torch.Tensor] = []
        discrete_hidden_tokens_by_class: dict[int, list[torch.Tensor]] = {}
        num_batches_used = 0
        previous_training_mode = self.training

        self.eval()
        with torch.no_grad():
            for batch_index, raw_batch in enumerate(train_loader):
                if batch_index >= self.memory_initialization_batches:
                    break
                num_batches_used += 1
                batch_on_device = self._move_initialization_batch_to_device(
                    raw_batch,
                    device,
                )
                clean_batch = self._prepare_clean_batch(
                    batch_on_device,
                    stage_name="memory_init",
                )
                clean_hidden = self.encoder(clean_batch)["hidden"].reshape(
                    -1,
                    self.hidden_dim,
                )
                if not (
                    self.memory_initialization_with_synthetic_windows
                    and self.use_synthetic_augmentation
                ):
                    continuous_hidden_token_groups.append(clean_hidden)
                    discrete_hidden_tokens_by_class.setdefault(0, []).append(
                        clean_hidden
                    )
                    continue

                synthetic_batch = self.synthetic_anomaly_injector.augment_batch(
                    self._clone_batch(batch_on_device)
                )
                synthetic_hidden = self.encoder(synthetic_batch)["hidden"]
                synthetic_labels = synthetic_batch["classification_labels"].long()
                normal_window_mask = synthetic_labels == 0
                normal_time_step_mask = synthetic_batch["synthetic_anomaly_mask"] == 0
                if int(normal_window_mask.sum().item()) > 0:
                    normal_hidden = synthetic_hidden[normal_window_mask]
                    normal_position_mask = normal_time_step_mask[normal_window_mask]
                    selected_normal_hidden = normal_hidden[normal_position_mask]
                    if selected_normal_hidden.numel() > 0:
                        continuous_hidden_token_groups.append(selected_normal_hidden)

                for class_index in synthetic_labels.unique(sorted=True).tolist():
                    class_mask = synthetic_labels == int(class_index)
                    class_hidden = synthetic_hidden[class_mask].reshape(
                        -1,
                        self.hidden_dim,
                    )
                    if class_hidden.numel() == 0:
                        continue
                    discrete_hidden_tokens_by_class.setdefault(
                        int(class_index), []
                    ).append(class_hidden)

        self.train(previous_training_mode)

        if continuous_hidden_token_groups:
            continuous_hidden_tokens = torch.cat(continuous_hidden_token_groups, dim=0)
        else:
            continuous_hidden_tokens = torch.empty(0, self.hidden_dim, device=device)

        finalized_discrete_hidden_tokens_by_class: dict[int, torch.Tensor] = {}
        for class_index, class_hidden_groups in discrete_hidden_tokens_by_class.items():
            finalized_discrete_hidden_tokens_by_class[class_index] = torch.cat(
                class_hidden_groups,
                dim=0,
            )

        return {
            "continuous_hidden_tokens": continuous_hidden_tokens,
            "discrete_hidden_tokens_by_class": finalized_discrete_hidden_tokens_by_class,
            "num_batches_used": num_batches_used,
            "num_continuous_normal_tokens": sum(
                int(hidden_group.shape[0])
                for hidden_group in continuous_hidden_token_groups
            ),
            "num_discrete_class_tokens_by_class": {
                class_index: int(class_hidden.shape[0])
                for class_index, class_hidden in finalized_discrete_hidden_tokens_by_class.items()
            },
        }

    def _initialize_memory_buffers_from_token_pool(
        self,
        *,
        continuous_hidden_tokens: torch.Tensor,
        discrete_hidden_tokens_by_class: dict[int, torch.Tensor],
    ) -> None:
        if continuous_hidden_tokens.shape[0] == 0:
            raise ValueError(
                "continuous_hidden_tokens must contain at least one normal token"
            )

        if self.continuous_prototype_bank is not None:
            continuous_seed_vectors = self._run_kmeans(
                continuous_hidden_tokens,
                self.continuous_num_prototypes,
                num_iterations=10,
            )
            self.continuous_prototype_bank.copy_(continuous_seed_vectors)

        if self.discrete_codebook is not None:
            available_class_indices = sorted(discrete_hidden_tokens_by_class)
            if not available_class_indices:
                raise ValueError(
                    "discrete_hidden_tokens_by_class must contain at least one class"
                )
            per_class_counts = [
                self.discrete_codebook_size // self.num_classes
                + (
                    1
                    if class_index < self.discrete_codebook_size % self.num_classes
                    else 0
                )
                for class_index in range(self.num_classes)
            ]
            fallback_hidden_tokens = torch.cat(
                [
                    discrete_hidden_tokens_by_class[class_index]
                    for class_index in available_class_indices
                ],
                dim=0,
            )
            class_stratified_vectors: list[torch.Tensor] = []
            for class_index, class_target_count in enumerate(per_class_counts):
                if class_target_count == 0:
                    continue
                class_hidden_tokens = discrete_hidden_tokens_by_class.get(class_index)
                if class_hidden_tokens is None or class_hidden_tokens.shape[0] == 0:
                    class_hidden_tokens = fallback_hidden_tokens
                class_stratified_vectors.append(
                    self._run_kmeans(
                        class_hidden_tokens,
                        class_target_count,
                        num_iterations=10,
                    )
                )
            discrete_seed_vectors = torch.cat(class_stratified_vectors, dim=0)
            if discrete_seed_vectors.shape[0] != self.discrete_codebook_size:
                raise ValueError(
                    "class-stratified discrete initialization must exactly fill "
                    f"discrete_codebook_size={self.discrete_codebook_size}, "
                    f"but produced {discrete_seed_vectors.shape[0]} vectors"
                )
            self.discrete_codebook.copy_(discrete_seed_vectors)
            if self.discrete_ema_counts is not None:
                self.discrete_ema_counts.fill_(1.0)
            if self.discrete_ema_sums is not None:
                self.discrete_ema_sums.copy_(discrete_seed_vectors)
            self._calibrate_anomaly_verification_metadata(
                discrete_hidden_tokens_by_class=discrete_hidden_tokens_by_class
            )

    def _calibrate_anomaly_verification_metadata(
        self, *, discrete_hidden_tokens_by_class: dict[int, torch.Tensor]
    ) -> None:
        """Mark anomaly codewords and store q99 train-token radii."""
        if not isinstance(self.discrete_codebook, torch.Tensor):
            return
        counts = [
            self.discrete_codebook_size // self.num_classes
            + (1 if index < self.discrete_codebook_size % self.num_classes else 0)
            for index in range(self.num_classes)
        ]
        mask = torch.zeros(self.discrete_codebook_size, dtype=torch.bool)
        codeword_class_ids = torch.zeros(
            self.discrete_codebook_size, dtype=torch.long
        )
        contributing_token_counts = torch.zeros(
            self.discrete_codebook_size, dtype=torch.float32
        )
        offset = 0
        for class_index, count in enumerate(counts):
            if class_index > 0:
                mask[offset : offset + count] = True
            codeword_class_ids[offset : offset + count] = class_index
            offset += count
        radii = torch.zeros(self.discrete_codebook_size)
        anomaly_groups = [
            values.reshape(-1, self.hidden_dim)
            for class_index, values in discrete_hidden_tokens_by_class.items()
            if class_index > 0 and values.numel() > 0
        ]
        if anomaly_groups:
            anomaly_tokens = torch.cat(anomaly_groups, dim=0)
            distances = (
                1.0
                - F.normalize(anomaly_tokens, dim=-1)
                @ F.normalize(self.discrete_codebook, dim=-1).T
            )
            nearest_ids = distances.argmin(dim=-1)
            nearest_distances = distances.gather(1, nearest_ids[:, None]).squeeze(1)
            contributing_token_counts += torch.bincount(
                nearest_ids,
                minlength=self.discrete_codebook_size,
            ).to(dtype=torch.float32)
            for codeword_id in torch.unique(nearest_ids).tolist():
                assigned = nearest_distances[nearest_ids == codeword_id]
                radii[codeword_id] = torch.quantile(assigned, 0.99)
        self.anomalous_codeword_mask = mask
        self.anomaly_radii = radii
        self.verification_codeword_class_ids = codeword_class_ids
        self.verification_contributing_token_counts = contributing_token_counts
        self.verification_metadata_source = "train_anomaly_tokens_q99"
        self.verification_metadata_schema_version = 1
        self.verification_metadata_split = "synthetic_train"
        self.verification_metadata_initialization_seed = int(
            self.synthetic_train_seed
            if getattr(self, "synthetic_train_seed", None) is not None
            else getattr(self, "synthetic_validation_seed", 0)
        )

    def _update_continuous_memory_bank(
        self,
        hidden: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.continuous_prototype_bank is None:
            raise ValueError("continuous_prototype_bank is not available")

        normalized_hidden = self._normalize_hidden_for_memory(hidden)
        if token_mask is not None:
            selected_hidden = normalized_hidden[token_mask]
            if selected_hidden.numel() == 0:
                return self._normalize_memory_vectors(self.continuous_prototype_bank)
            normalized_hidden = selected_hidden.reshape(1, -1, self.hidden_dim)
        normalized_memory = self._normalize_memory_vectors(
            self.continuous_prototype_bank
        )

        # k prototypes with each prototype having h dimensions (containing h numbers)
        # b windows with each window having l timesteps,
        # with each timestep having h dimensions (containing h numbers)
        # k @ b.T is making each prototype attending to each timestep in a window
        # across all b windows in a batch.
        prototype_to_token_logits = torch.einsum(
            "kh,blh->kbl",
            normalized_memory,  # (n_continuous_prototypes, d_model)
            normalized_hidden,  # (batch_size, n_timesteps, d_model)
        ) / math.sqrt(
            self.hidden_dim
        )  # (n_continuous_prototypes, batch_size, n_timesteps)

        # n_continuous_prototypes là self.continuous_num_prototypes
        # d_model là h

        prototype_to_token_weights = torch.softmax(
            prototype_to_token_logits.reshape(self.continuous_num_prototypes, -1),
            # (n_continuous_prototypes, batch_size * n_timesteps)
            dim=-1,
        ).reshape_as(
            prototype_to_token_logits
        )  # (n_continuous_prototypes, batch_size, n_timesteps)

        weighted_hidden_summary = torch.einsum(
            "kbl,blh->kh",
            prototype_to_token_weights,
            normalized_hidden,
        )

        weighted_hidden_summary = self._normalize_memory_vectors(
            weighted_hidden_summary
        )

        gate_input = torch.cat(
            [normalized_memory, weighted_hidden_summary],
            dim=-1,
        )

        update_gate = self.continuous_update_gate(gate_input)

        updated_memory = (
            1.0 - update_gate
        ) * normalized_memory + update_gate * weighted_hidden_summary
        updated_memory = self._normalize_memory_vectors(updated_memory)

        with torch.no_grad():
            self.continuous_prototype_bank.copy_(updated_memory.detach())

        return updated_memory

    def _update_discrete_codebook_memory(
        self,
        hidden: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self.discrete_assignment is None
            or self.discrete_codebook is None
            or self.discrete_ema_counts is None
            or self.discrete_ema_sums is None
        ):
            raise ValueError("discrete memory state is not available")

        normalized_hidden = self._normalize_hidden_for_memory(hidden)
        if token_mask is not None:
            selected_hidden = normalized_hidden[token_mask]
            if selected_hidden.numel() == 0:
                assignment_logits = hidden.new_zeros(
                    hidden.shape[0],
                    hidden.shape[1],
                    self.discrete_codebook_size,
                )
                assignment_probabilities = torch.softmax(assignment_logits, dim=-1)
                return (
                    assignment_logits,
                    assignment_probabilities,
                    self._normalize_memory_vectors(self.discrete_codebook),
                )
            normalized_hidden = selected_hidden.reshape(1, -1, self.hidden_dim)
        assignment_logits = self.discrete_assignment(normalized_hidden)
        assignment_probabilities = F.gumbel_softmax(
            assignment_logits,
            tau=self.gumbel_temperature,
            hard=False,
            dim=-1,
        )
        flattened_probabilities = assignment_probabilities.reshape(
            -1,
            self.discrete_codebook_size,
        )
        flattened_hidden = normalized_hidden.reshape(-1, self.hidden_dim)
        batch_counts = flattened_probabilities.sum(dim=0)
        batch_sums = flattened_probabilities.T @ flattened_hidden

        with torch.no_grad():
            self.discrete_ema_counts.mul_(self.discrete_ema_decay).add_(
                (1.0 - self.discrete_ema_decay) * batch_counts.detach()
            )
            self.discrete_ema_sums.mul_(self.discrete_ema_decay).add_(
                (1.0 - self.discrete_ema_decay) * batch_sums.detach()
            )
            normalized_codebook = (
                self.discrete_ema_sums
                / self.discrete_ema_counts.clamp_min(
                    self.memory_norm_epsilon
                ).unsqueeze(-1)
            )
            normalized_codebook = self._normalize_memory_vectors(normalized_codebook)
            self.discrete_codebook.copy_(normalized_codebook)

        return (
            assignment_logits,
            assignment_probabilities,
            self._normalize_memory_vectors(self.discrete_codebook),
        )

    def _build_phase_passthrough_outputs(
        self,
        hidden: torch.Tensor,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        continuous_outputs = {
            "hidden": hidden,
            "prototype_context": hidden,
            "prototype_logits": None,
            "prototype_weights": None,
            "aux": {
                "branch_name": "continuous",
                "enabled": False,
                "num_prototypes": 0,
                "memory_bypass_active": True,
                "memory_initialized": self.memory_initialized,
            },
        }
        discrete_outputs = {
            "hidden": hidden,
            "quantized_hidden": hidden,
            "assignment_logits": None,
            "assignment_probabilities": None,
            "code_indices": None,
            "aux": {
                "branch_name": "discrete",
                "enabled": False,
                "num_codes": 0,
                "memory_bypass_active": True,
                "memory_initialized": self.memory_initialized,
                "query_mode": "phase_passthrough",
                "topk": 0,
                "query_temperature": 0.0,
            },
        }
        fusion_outputs = {
            "hidden_reconstruction": hidden,
            "hidden_classification": hidden,
            "alpha": hidden.new_zeros(hidden.shape[0]),
            "beta": hidden.new_zeros(hidden.shape[0]),
            "aux": {
                "fusion_mode": "phase_direct_passthrough",
                "alpha": 0.0,
                "beta": 0.0,
                "alpha_std": 0.0,
                "beta_std": 0.0,
                "alpha_logit": float(self.alpha_logit.detach().cpu()),
                "beta_logit": float(self.beta_logit.detach().cpu()),
                "cka_reconstruction_mean": 0.0,
                "cka_reconstruction_std": 0.0,
                "cka_classification_mean": 0.0,
                "cka_classification_std": 0.0,
                "warmup_active": self.schedule_state["warmup_active"],
                "temperature": self.gumbel_temperature,
            },
        }
        return continuous_outputs, discrete_outputs, fusion_outputs
