from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TTLBuffer:
    """Keep recent items alive for a fixed number of stream steps."""

    ttl_steps: int
    _items: list[dict[str, object]] = field(default_factory=list)

    def add(self, item: object, current_step: int) -> None:
        self._items.append(
            {
                "item": item,
                "expires_at": int(current_step) + int(self.ttl_steps),
            }
        )

    def expire(self, current_step: int) -> None:
        self._items = [
            entry for entry in self._items if int(entry["expires_at"]) > current_step
        ]

    def contains(self, item: object) -> bool:
        return any(entry["item"] == item for entry in self._items)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
