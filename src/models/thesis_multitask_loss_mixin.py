from __future__ import annotations

"""Public loss mixin for the thesis multitask model.

The actual logic now lives in smaller base mixins so each source file stays
under the repository line limit while keeping the same runtime behavior.
"""

from src.models.thesis_multitask_loss_core_mixin import ThesisMultitaskLossCoreMixin
from src.models.thesis_multitask_loss_gradient_mixin import (
    ThesisMultitaskLossGradientMixin,
)
from src.models.thesis_multitask_loss_step_mixin import ThesisMultitaskLossStepMixin


class ThesisMultitaskLossMixin(
    ThesisMultitaskLossCoreMixin,
    ThesisMultitaskLossGradientMixin,
    ThesisMultitaskLossStepMixin,
):
    pass
