from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerificationBuffer:
    """Keep a small verified window history with a non-overlap guard."""

    max_size: int = 32
    non_overlap_gap: int = 0
    _items: list[dict[str, Any]] = field(default_factory=list)

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
        self._items.append(dict(item))

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._items]
