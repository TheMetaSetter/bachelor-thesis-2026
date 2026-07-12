from __future__ import annotations

"""Public state mixin for the thesis multitask model."""

from src.models.thesis_multitask_state_memory_mixin import (
    ThesisMultitaskStateMemoryMixin,
)
from src.models.thesis_multitask_state_passthrough_mixin import (
    ThesisMultitaskStatePassthroughMixin,
)
from src.models.thesis_multitask_state_schedule_mixin import (
    ThesisMultitaskStateScheduleMixin,
)
from src.models.thesis_multitask_state_serialization_mixin import (
    ThesisMultitaskStateSerializationMixin,
)


class ThesisMultitaskStateMixin(
    ThesisMultitaskStateScheduleMixin,
    ThesisMultitaskStateSerializationMixin,
    ThesisMultitaskStateMemoryMixin,
    ThesisMultitaskStatePassthroughMixin,
):
    pass
