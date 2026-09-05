from __future__ import annotations

"""Self-contained multitask prototype-fusion model.

The public model stays in this file, while the implementation details are
split into small mixins so the codebase can stay below the 1000-line limit
without changing behavior.
"""

from typing import Any

import torch

from src.data.scalers import SequenceStandardScaler

from src.models.base_model import BaseModel
from src.models.thesis_multitask_impl.thesis_multitask_components import (
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
from src.models.thesis_multitask_impl.thesis_multitask_loss_mixin import (
    ThesisMultitaskLossMixin,
)
from src.models.thesis_multitask_impl.thesis_multitask_routing_mixin import (
    ThesisMultitaskRoutingMixin,
)
from src.models.thesis_multitask_impl.thesis_multitask_setup_mixin import (
    ThesisMultitaskSetupMixin,
)
from src.models.thesis_multitask_impl.thesis_multitask_state_mixin import (
    ThesisMultitaskStateMixin,
)


class ThesisMultitaskModel(
    ThesisMultitaskSetupMixin,
    ThesisMultitaskStateMixin,
    ThesisMultitaskRoutingMixin,
    ThesisMultitaskLossMixin,
    BaseModel,
):
    # File comment for a younger reader:
    # this class is the public door into the model, but the real steps are kept
    # in small implementation modules so each phase is easier to read and test.
    @staticmethod
    def _resolve_model_config(
        config: ThesisMultitaskModelConfig | None,
        flat_kwargs: dict[str, Any],
    ) -> ThesisMultitaskModelConfig:
        if config is not None and flat_kwargs:
            raise ValueError("Pass either config or flat keyword arguments, not both")
        if config is None:
            config = ThesisMultitaskModelConfig.from_flat_kwargs(flat_kwargs)
        if not isinstance(config, ThesisMultitaskModelConfig):
            raise TypeError("config must be a ThesisMultitaskModelConfig instance")
        return config

    def __init__(
        self,
        config: ThesisMultitaskModelConfig | None = None,
        **flat_kwargs: Any,
    ) -> None:
        super().__init__()
        self._point_score_calibration = None
        self.reconstruction_loss_space = "normalized_input"
        self.reconstruction_scaler = None
        config = self._resolve_model_config(config, flat_kwargs)

        self._store_config_values(config)
        self._build_encoder(config)
        self._build_prototype_memory(config)
        self._build_fusion_parameters(config)
        self._build_task_heads(config)
        self._build_synthetic_injectors(config)
        self._encoder_profiled_parameters = self._get_encoder_profiled_parameters()
        self._build_optional_loss_configs()
        self._configure_trainable_parameters_for_phase()
        self.set_epoch_context(epoch_index=0, total_epochs=1)
        self._print_model_summary(config)

    def configure_reconstruction_loss(
        self, space: str, scaler_state: dict[str, Any]
    ) -> None:
        """Use the train-fitted scaler at the loss boundary, without refitting."""
        if space not in {"normalized_input", "raw_input"}:
            raise ValueError("reconstruction_loss_space must be normalized_input or raw_input")
        scaler = None
        if space == "raw_input":
            scaler = SequenceStandardScaler()
            scaler.load_state_dict(scaler_state)
            scaler.inverse_transform_tensor(torch.zeros_like(scaler.feature_mean))
        self.reconstruction_loss_space = space
        self.reconstruction_scaler = scaler

    def reconstruction_squared_error(self, outputs, batch) -> torch.Tensor:
        """Compute errors before averaging stochastic reconstruction samples."""
        reconstruction = outputs["recon"]
        target = batch["x"]
        if self.reconstruction_loss_space == "normalized_input":
            return (reconstruction - target).square()
        samples = (outputs.get("aux", {}).get("stochastic_query") or {}).get(
            "reconstruction_samples"
        )
        if isinstance(samples, torch.Tensor):
            reconstruction = samples
            target = target.unsqueeze(1)
        raw_target = self.reconstruction_scaler.inverse_transform_tensor(target)
        raw_reconstruction = self.reconstruction_scaler.inverse_transform_tensor(reconstruction)
        errors = (raw_reconstruction - raw_target).square()
        return errors.mean(dim=1) if errors.ndim == 4 else errors
