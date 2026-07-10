from __future__ import annotations

from collections import deque


class NonOverlapGuard:
    """Accept at most bounded, non-overlapping successful update intervals."""

    def __init__(self, max_size: int = 1) -> None:
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self._intervals: deque[tuple[int, int]] = deque(maxlen=max_size)

    def accept(self, interval: tuple[int, int]) -> bool:
        start, end = interval
        if end <= start:
            raise ValueError("interval end must exceed start")
        return all(end <= left or start >= right for left, right in self._intervals)

    def add(self, interval: tuple[int, int]) -> None:
        if not self.accept(interval):
            raise ValueError("interval overlaps a guarded update")
        self._intervals.append(interval)

    def intervals(self) -> list[tuple[int, int]]:
        return list(self._intervals)
