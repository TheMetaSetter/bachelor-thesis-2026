from __future__ import annotations

"""Local affiliation metric helpers adapted from the CARLA reference metric code.

The active runtime needs Affiliation-F1 as a first-class metric, but should not
depend on importing code from `bsc-thesis-ref-codebases/` at runtime. This file
keeps the required affiliation probability machinery local and dataset-agnostic.
"""

from itertools import groupby
import math
from operator import itemgetter

import numpy as np


def _convert_vector_to_events(binary_vector: np.ndarray) -> list[tuple[int, int]]:
    positive_indexes = [index for index, value in enumerate(binary_vector) if value > 0]
    events: list[tuple[int, int]] = []
    for _, grouped_indexes in groupby(
        enumerate(positive_indexes),
        lambda indexed_pair: indexed_pair[0] - indexed_pair[1],
    ):
        contiguous_indexes = list(map(itemgetter(1), grouped_indexes))
        events.append((contiguous_indexes[0], contiguous_indexes[-1] + 1))
    return events


def _sum_without_nan(values: list[float]) -> float:
    return float(sum(value for value in values if not math.isnan(value)))


def _len_without_nan(values: list[float]) -> int:
    return int(sum(0 if math.isnan(value) else 1 for value in values))


def _interval_length(interval: tuple[float, float] | None) -> float:
    if interval is None:
        return 0.0
    return float(interval[1] - interval[0])


def _sum_interval_lengths(intervals: list[tuple[float, float] | None]) -> float:
    return float(sum(_interval_length(interval) for interval in intervals))


def _interval_intersection(
    left_interval: tuple[float, float] | None,
    right_interval: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if left_interval is None or right_interval is None:
        return None
    intersection = (
        max(left_interval[0], right_interval[0]),
        min(left_interval[1], right_interval[1]),
    )
    if intersection[0] >= intersection[1]:
        return None
    return intersection


def _interval_subset(
    left_interval: tuple[float, float],
    right_interval: tuple[float, float],
) -> bool:
    return bool(
        left_interval[0] >= right_interval[0] and left_interval[1] <= right_interval[1]
    )


def _cut_interval_into_three(
    interval: tuple[float, float] | None,
    anchor_interval: tuple[float, float],
) -> tuple[
    tuple[float, float] | None,
    tuple[float, float] | None,
    tuple[float, float] | None,
]:
    if interval is None:
        return None, None, None

    intersection = _interval_intersection(interval, anchor_interval)
    if interval == intersection:
        return None, interval, None
    if interval[1] <= anchor_interval[0]:
        return interval, None, None
    if interval[0] >= anchor_interval[1]:
        return None, None, interval
    if interval[0] <= anchor_interval[0] and interval[1] >= anchor_interval[1]:
        return (
            (interval[0], intersection[0]),
            intersection,
            (intersection[1], interval[1]),
        )
    if interval[0] <= anchor_interval[0]:
        return (interval[0], intersection[0]), intersection, None
    if interval[1] >= anchor_interval[1]:
        return None, intersection, (intersection[1], interval[1])
    raise ValueError("Unhandled interval partition case")


def _get_pivot(
    source_interval: tuple[float, float],
    target_interval: tuple[float, float],
) -> float:
    if _interval_intersection(source_interval, target_interval) is not None:
        raise ValueError("source_interval and target_interval must be disjoint")
    if max(source_interval) <= min(target_interval):
        return float(min(target_interval))
    if min(source_interval) >= max(target_interval):
        return float(max(target_interval))
    raise ValueError("source_interval must be outside target_interval")


def _integral_mini_interval_probability_precision(
    interval: tuple[float, float],
    ground_truth_interval: tuple[float, float],
    affiliation_zone: tuple[float, float],
) -> float:
    if _interval_intersection(interval, ground_truth_interval) is not None:
        raise ValueError("interval and ground_truth_interval must be disjoint")
    if not _interval_subset(ground_truth_interval, affiliation_zone):
        raise ValueError("ground_truth_interval must be included in affiliation_zone")
    if not _interval_subset(interval, affiliation_zone):
        raise ValueError("interval must be included in affiliation_zone")

    zone_start = float(min(affiliation_zone))
    zone_end = float(max(affiliation_zone))
    truth_start = float(min(ground_truth_interval))
    truth_end = float(max(ground_truth_interval))
    interval_start = float(min(interval))
    interval_end = float(max(interval))

    minimum_distance = max(interval_start - truth_end, truth_start - interval_end)
    maximum_distance = max(interval_end - truth_end, truth_start - interval_start)
    max_probability_distance = min(truth_start - zone_start, zone_end - truth_end)
    quadratic_piece = (
        min(maximum_distance, max_probability_distance) ** 2
        - min(minimum_distance, max_probability_distance) ** 2
    )
    linear_piece = max(maximum_distance, max_probability_distance) - max(
        minimum_distance,
        max_probability_distance,
    )
    clamped_distance_integral = 0.5 * quadratic_piece + (
        max_probability_distance * linear_piece
    )
    raw_distance_integral = 0.5 * (maximum_distance**2 - minimum_distance**2)
    overlap_mass = (truth_end - truth_start) * (interval_end - interval_start)
    interval_length = interval_end - interval_start
    zone_length = zone_end - zone_start
    return (
        interval_length
        - (clamped_distance_integral + raw_distance_integral + overlap_mass)
        / zone_length
    )


def _integral_interval_probability_precision(
    interval: tuple[float, float] | None,
    ground_truth_interval: tuple[float, float],
    affiliation_zone: tuple[float, float],
) -> float:
    def _outside_piece(piece: tuple[float, float] | None) -> float:
        if piece is None:
            return 0.0
        return _integral_mini_interval_probability_precision(
            piece,
            ground_truth_interval,
            affiliation_zone,
        )

    def _inside_piece(piece: tuple[float, float] | None) -> float:
        if piece is None:
            return 0.0
        return float(max(piece) - min(piece))

    left_piece, middle_piece, right_piece = _cut_interval_into_three(
        interval,
        ground_truth_interval,
    )
    return (
        _outside_piece(left_piece)
        + _inside_piece(middle_piece)
        + _outside_piece(right_piece)
    )


def _cut_ground_truth_based_on_center(
    ground_truth_interval: tuple[float, float] | None,
    center_value: float,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    if ground_truth_interval is None:
        return None, None
    if center_value >= max(ground_truth_interval):
        return ground_truth_interval, None
    if center_value <= min(ground_truth_interval):
        return None, ground_truth_interval
    return (
        (min(ground_truth_interval), center_value),
        (center_value, max(ground_truth_interval)),
    )


def _integral_mini_interval_probability_recall(
    predicted_interval: tuple[float, float],
    ground_truth_interval: tuple[float, float],
    affiliation_zone: tuple[float, float],
) -> float:
    predicted_pivot = _get_pivot(ground_truth_interval, predicted_interval)
    zone_start = float(min(affiliation_zone))
    zone_end = float(max(affiliation_zone))
    zone_center = 0.5 * (zone_start + zone_end)

    if predicted_pivot <= zone_start or predicted_pivot >= zone_end:
        return 0.0

    before_center, after_center = _cut_ground_truth_based_on_center(
        ground_truth_interval,
        zone_center,
    )
    before_far, before_near = _cut_ground_truth_based_on_center(
        before_center,
        0.5 * (zone_start + predicted_pivot),
    )
    after_near, after_far = _cut_ground_truth_based_on_center(
        after_center,
        0.5 * (zone_end + predicted_pivot),
    )

    def _bounds(interval: tuple[float, float] | None) -> tuple[float, float]:
        if interval is None:
            return math.nan, math.nan
        return float(min(interval)), float(max(interval))

    before_far_start, before_far_end = _bounds(before_far)
    before_near_start, before_near_end = _bounds(before_near)
    after_near_start, after_near_end = _bounds(after_near)
    after_far_start, after_far_end = _bounds(after_far)

    if predicted_pivot >= max(ground_truth_interval):
        integral_parts = [
            (predicted_pivot - zone_start) * (before_far_end - before_far_start),
            2 * predicted_pivot * (before_near_end - before_near_start)
            - (before_near_end**2 - before_near_start**2),
            2 * predicted_pivot * (after_near_end - after_near_start)
            - (after_near_end**2 - after_near_start**2),
            (zone_end + predicted_pivot) * (after_far_end - after_far_start)
            - (after_far_end**2 - after_far_start**2),
        ]
    elif predicted_pivot <= min(ground_truth_interval):
        integral_parts = [
            (before_far_end**2 - before_far_start**2)
            - (zone_start + predicted_pivot) * (before_far_end - before_far_start),
            (before_near_end**2 - before_near_start**2)
            - 2 * predicted_pivot * (before_near_end - before_near_start),
            (after_near_end**2 - after_near_start**2)
            - 2 * predicted_pivot * (after_near_end - after_near_start),
            (zone_end - predicted_pivot) * (after_far_end - after_far_start),
        ]
    else:
        raise ValueError("predicted_pivot must lie outside ground_truth_interval")

    integral_min_plus_distance = _sum_without_nan(integral_parts)
    truth_length = float(max(ground_truth_interval) - min(ground_truth_interval))
    zone_length = float(max(affiliation_zone) - min(affiliation_zone))
    return truth_length - integral_min_plus_distance / zone_length


def _integral_interval_probability_recall(
    predicted_interval: tuple[float, float],
    ground_truth_interval: tuple[float, float],
    affiliation_zone: tuple[float, float],
) -> float:
    def _outside_piece(piece: tuple[float, float] | None) -> float:
        if piece is None:
            return 0.0
        return _integral_mini_interval_probability_recall(
            predicted_interval,
            piece,
            affiliation_zone,
        )

    def _inside_piece(piece: tuple[float, float] | None) -> float:
        if piece is None:
            return 0.0
        return float(max(piece) - min(piece))

    left_piece, middle_piece, right_piece = _cut_interval_into_three(
        ground_truth_interval,
        predicted_interval,
    )
    return (
        _outside_piece(left_piece)
        + _inside_piece(middle_piece)
        + _outside_piece(right_piece)
    )


def _t_start(
    index: int,
    ground_truth_events: list[tuple[float, float]],
    time_range: tuple[float, float],
) -> float:
    if index == len(ground_truth_events):
        return 2 * max(time_range) - _t_stop(
            len(ground_truth_events) - 1,
            ground_truth_events,
            time_range,
        )
    return float(ground_truth_events[index][0])


def _t_stop(
    index: int,
    ground_truth_events: list[tuple[float, float]],
    time_range: tuple[float, float],
) -> float:
    if index == -1:
        return 2 * min(time_range) - _t_start(0, ground_truth_events, time_range)
    return float(ground_truth_events[index][1])


def _get_all_affiliation_zones(
    ground_truth_events: list[tuple[float, float]],
    time_range: tuple[float, float],
) -> list[tuple[float, float]]:
    return [
        (
            0.5
            * (
                _t_stop(index - 1, ground_truth_events, time_range)
                + _t_start(index, ground_truth_events, time_range)
            ),
            0.5
            * (
                _t_stop(index, ground_truth_events, time_range)
                + _t_start(index + 1, ground_truth_events, time_range)
            ),
        )
        for index in range(len(ground_truth_events))
    ]


def _affiliation_partition(
    predicted_events: list[tuple[float, float]],
    affiliation_zones: list[tuple[float, float]],
) -> list[list[tuple[float, float] | None]]:
    partition: list[list[tuple[float, float] | None]] = [None] * len(affiliation_zones)
    for zone_index, affiliation_zone in enumerate(affiliation_zones):
        # Keep one slot per input event so later zips remain aligned with the
        # original event ordering, matching the reference affiliation code.
        partition[zone_index] = [
            _interval_intersection(predicted_event, affiliation_zone)
            for predicted_event in predicted_events
        ]
    return partition


def _compute_affiliation_precision_probability(
    predicted_event_fragments: list[tuple[float, float] | None],
    ground_truth_event: tuple[float, float],
    affiliation_zone: tuple[float, float],
) -> float:
    if all(fragment is None for fragment in predicted_event_fragments):
        return math.nan
    return float(
        sum(
            _integral_interval_probability_precision(
                fragment,
                ground_truth_event,
                affiliation_zone,
            )
            for fragment in predicted_event_fragments
        )
        / _sum_interval_lengths(predicted_event_fragments)
    )


def _compute_affiliation_recall_probability(
    predicted_event_fragments: list[tuple[float, float] | None],
    ground_truth_event: tuple[float, float],
    affiliation_zone: tuple[float, float],
) -> float:
    predicted_event_fragments = [
        fragment for fragment in predicted_event_fragments if fragment is not None
    ]
    if not predicted_event_fragments:
        return 0.0

    recall_zones = _get_all_affiliation_zones(
        predicted_event_fragments,
        affiliation_zone,
    )
    ground_truth_fragments = _affiliation_partition(
        [ground_truth_event],
        recall_zones,
    )
    return float(
        sum(
            _integral_interval_probability_recall(
                predicted_fragment,
                ground_truth_fragment_list[0],
                affiliation_zone,
            )
            for predicted_fragment, ground_truth_fragment_list in zip(
                predicted_event_fragments,
                ground_truth_fragments,
            )
        )
        / _interval_length(ground_truth_event)
    )


def compute_affiliation_precision_recall(
    point_labels: np.ndarray,
    binary_predictions: np.ndarray,
) -> tuple[float, float]:
    label_array = np.asarray(point_labels).astype(np.int64).reshape(-1)
    prediction_array = np.asarray(binary_predictions).astype(np.int64).reshape(-1)
    if label_array.shape != prediction_array.shape:
        raise ValueError("point_labels and binary_predictions must have the same shape")

    ground_truth_events = _convert_vector_to_events(label_array)
    predicted_events = _convert_vector_to_events(prediction_array)
    if not ground_truth_events:
        return float("nan"), float("nan")
    if not predicted_events:
        return 0.0, 0.0

    time_range = (0.0, float(label_array.shape[0]))
    affiliation_zones = _get_all_affiliation_zones(ground_truth_events, time_range)
    event_partition = _affiliation_partition(predicted_events, affiliation_zones)

    precision_values = [
        _compute_affiliation_precision_probability(
            predicted_event_fragments,
            ground_truth_event,
            affiliation_zone,
        )
        for predicted_event_fragments, ground_truth_event, affiliation_zone in zip(
            event_partition,
            ground_truth_events,
            affiliation_zones,
        )
    ]
    recall_values = [
        _compute_affiliation_recall_probability(
            predicted_event_fragments,
            ground_truth_event,
            affiliation_zone,
        )
        for predicted_event_fragments, ground_truth_event, affiliation_zone in zip(
            event_partition,
            ground_truth_events,
            affiliation_zones,
        )
    ]

    if _len_without_nan(precision_values) > 0:
        precision = _sum_without_nan(precision_values) / _len_without_nan(
            precision_values
        )
    else:
        precision = float("nan")
    recall = float(sum(recall_values) / len(recall_values))
    return float(precision), float(recall)
