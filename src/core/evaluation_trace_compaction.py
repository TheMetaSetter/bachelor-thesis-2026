from __future__ import annotations

"""Helpers to compact evaluation traces without dropping UQ summary data.

The evaluator still produces the full in-memory payload so internal audits can
inspect the raw Monte Carlo tensors.  When the traces are written to disk, this
module strips the heavy sample payload and keeps only the fields that are
needed later for backfill, reporting, and provenance checks.
"""

from typing import Any


_TRACE_FIELDS_TO_KEEP = (
    "batch_index",
    "entity_ids",
    "point_score_summary",
    "window_score_summary",
    "point_score_history",
    "window_score_history",
    "uncertainty_history",
    "sample_retention_policy",
)

_STOCHASTIC_QUERY_METADATA_FIELDS = (
    "schema_version",
    "enabled",
    "num_samples",
    "continuous_temperature",
    "discrete_temperature",
    "return_mc_samples",
    "sample_retention_policy",
)


def _compact_stochastic_query(stochastic_query: Any) -> dict[str, Any] | None:
    if not isinstance(stochastic_query, dict):
        return None
    compacted = {
        key_name: stochastic_query.get(key_name)
        for key_name in _STOCHASTIC_QUERY_METADATA_FIELDS
        if key_name in stochastic_query
    }
    return compacted or None


def compact_evaluation_trace_payload(trace_payload: dict[str, Any]) -> dict[str, Any]:
    compacted = {
        key_name: trace_payload.get(key_name)
        for key_name in _TRACE_FIELDS_TO_KEEP
        if key_name in trace_payload
    }
    compacted_stochastic_query = _compact_stochastic_query(
        trace_payload.get("stochastic_query")
    )
    if compacted_stochastic_query is not None:
        compacted["stochastic_query"] = compacted_stochastic_query
    return compacted


def compact_evaluation_trace_payloads(
    trace_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        compact_evaluation_trace_payload(trace_payload)
        for trace_payload in trace_payloads
    ]
