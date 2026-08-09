from __future__ import annotations

"""Public state mixin for the thesis multitask model."""

from src.models.thesis_multitask_impl.thesis_multitask_state_memory_mixin import (
    ThesisMultitaskStateMemoryMixin,
)
from src.models.thesis_multitask_impl.thesis_multitask_state_passthrough_mixin import (
    ThesisMultitaskStatePassthroughMixin,
)
from src.models.thesis_multitask_impl.thesis_multitask_state_schedule_mixin import (
    ThesisMultitaskStateScheduleMixin,
)
from src.models.thesis_multitask_impl.thesis_multitask_state_serialization_mixin import (
    ThesisMultitaskStateSerializationMixin,
)
from src.protocols.point_score_calibration import (
    PointScoreCalibration,
    transform_point_scores,
)


class ThesisMultitaskStateMixin(
    ThesisMultitaskStateScheduleMixin,
    ThesisMultitaskStateSerializationMixin,
    ThesisMultitaskStateMemoryMixin,
    ThesisMultitaskStatePassthroughMixin,
):
    def set_point_score_calibration(
        self, calibration: PointScoreCalibration
    ) -> None:
        if not isinstance(calibration, PointScoreCalibration):
            raise TypeError("calibration must be a PointScoreCalibration")
        self._point_score_calibration = calibration

    def clear_point_score_calibration(self) -> None:
        self._point_score_calibration = None

    def get_point_score_calibration(self) -> PointScoreCalibration | None:
        return getattr(self, "_point_score_calibration", None)

    def transform_official_point_scores(self, raw_point_scores):
        calibration = self.get_point_score_calibration()
        if calibration is None:
            return raw_point_scores, False
        return transform_point_scores(raw_point_scores, calibration), True
