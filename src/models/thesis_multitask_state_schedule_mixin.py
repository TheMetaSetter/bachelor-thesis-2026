from __future__ import annotations

"""Schedule and stage-state helpers for the thesis multitask model."""

import math
from typing import Any

from src.core.console import console_print
from src.models.thesis_multitask_components import (
    STAGE3_PHASE_CANONICAL_NAME,
    TWO_STAGE_A_PHASE_NAME,
    TWO_STAGE_B_PHASE_NAME,
)


class ThesisMultitaskStateScheduleMixin:
    def _zero_loss(self, reference_tensor):
        return reference_tensor.new_zeros(())

    def _compute_temperature_for_epoch(
        self, epoch_index: int, total_epochs: int
    ) -> float:
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
