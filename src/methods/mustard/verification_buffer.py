"""Non-overlapping verification buffer for MUSTARD gray-zone windows."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerificationBuffer:
    """Keep a small non-overlapping history of unresolved gray-zone windows."""

    max_size: int = 32
    non_overlap_gap: int = 0
    _items: list[dict[str, Any]] = field(default_factory=list)
    verification_capacity: int = 8
    default_ttl: int = 2
    _new_since_cycle: bool = False

    def try_admit(self, entry: dict[str, Any]) -> bool:
        """Admit a non-overlapping entry and initialize its TTL."""
        start, stop = int(entry["window_start"]), int(entry["window_end"])
        for item in self._items:
            if not (
                stop <= int(item["window_start"]) - self.non_overlap_gap
                or start >= int(item["window_end"]) + self.non_overlap_gap
            ):
                return False
        if len(self._items) >= self.max_size:
            self._items.pop(0)
        normalized = dict(entry)
        normalized.setdefault("entry_id", str(start))
        normalized.setdefault("ttl_remaining", self.default_ttl)
        normalized.setdefault("was_adapted", False)
        self._items.append(normalized)
        self._new_since_cycle = True
        return True

    def should_verify(self) -> bool:
        return len(self._items) >= self.verification_capacity and self._new_since_cycle

    def mark_verification_result(self, entry_id: str, adapted: bool) -> None:
        for item in self._items:
            if item.get("entry_id") == entry_id:
                item["was_adapted"] = bool(adapted)
                return
        raise KeyError(f"unknown verification entry: {entry_id}")

    def finish_verification_cycle(self) -> None:
        """Decrement TTL for unadapted windows and drop expired or adapted entries."""
        kept: list[dict[str, Any]] = []
        for item in self._items:
            if item.get("was_adapted"):
                continue
            item["ttl_remaining"] = int(item.get("ttl_remaining", self.default_ttl)) - 1
            if item["ttl_remaining"] > 0:
                kept.append(item)
        self._items = kept
        self._new_since_cycle = False

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._items]

    def state_dict(self) -> dict[str, Any]:
        return deepcopy(
            {
                "items": self._items,
                "new_since_cycle": self._new_since_cycle,
            }
        )

    def load_state_dict(self, state: dict[str, Any], device: Any) -> None:
        saved = deepcopy(state)
        for entry in saved["items"]:
            entry["x"] = entry["x"].to(device)
        self._items = saved["items"]
        self._new_since_cycle = bool(saved["new_since_cycle"])
