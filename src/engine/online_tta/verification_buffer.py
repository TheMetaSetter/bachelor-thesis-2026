from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerificationBuffer:
    """Keep a small verified window history with a non-overlap guard."""

    max_size: int = 32
    non_overlap_gap: int = 0
    _items: list[dict[str, Any]] = field(default_factory=list)
    verification_capacity: int = 8
    default_ttl: int = 2
    _new_since_cycle: bool = False

    def admit(self, *, window_start: int, window_end: int) -> bool:
        for item in self._items:
            existing_start = int(item["window_start"])
            existing_end = int(item["window_end"])
            overlaps = not (
                window_end <= existing_start - self.non_overlap_gap
                or window_start >= existing_end + self.non_overlap_gap
            )
            if overlaps:
                return False
        return True

    def add(self, item: dict[str, Any]) -> None:
        if len(self._items) >= self.max_size:
            self._items.pop(0)
        normalized = dict(item)
        normalized.setdefault("entry_id", f"entry-{len(self._items)}")
        normalized.setdefault("status", "unresolved")
        normalized.setdefault("ttl_remaining", self.default_ttl)
        normalized.setdefault("was_adapted", False)
        self._items.append(normalized)
        self._new_since_cycle = True

    def try_admit(self, entry: dict[str, Any]) -> bool:
        """Admit a non-overlapping unresolved entry and initialize its TTL."""
        if not self.admit(window_start=int(entry["window_start"]), window_end=int(entry["window_end"])):
            return False
        self.add(entry)
        return True

    def should_verify(self) -> bool:
        return len(self._items) >= self.verification_capacity and self._new_since_cycle

    def mark_verification_result(self, entry_id: str, adapted: bool) -> None:
        for item in self._items:
            if item.get("entry_id") == entry_id:
                item["was_adapted"] = bool(adapted)
                item["status"] = "adapted" if adapted else "unresolved"
                return
        raise KeyError(f"unknown verification entry: {entry_id}")

    def finish_verification_cycle(self) -> dict[str, int]:
        """Apply one TTL tick only after a verification cycle completes."""
        kept: list[dict[str, Any]] = []
        removed = 0
        for item in self._items:
            if item.get("was_adapted"):
                removed += 1
                continue
            item["ttl_remaining"] = int(item.get("ttl_remaining", self.default_ttl)) - 1
            if item["ttl_remaining"] <= 0:
                removed += 1
                continue
            kept.append(item)
        self._items = kept
        self._new_since_cycle = False
        return {"remaining": len(kept), "removed": removed}

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._items]
