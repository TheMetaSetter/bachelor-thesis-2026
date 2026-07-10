"""Small producer/consumer controller for deterministic demo replay."""

from __future__ import annotations

from queue import Empty, Queue
from threading import Event, Thread
import time
from collections.abc import Iterable, Iterator
from typing import Any


class StreamQueueController:
    """Own queue lifecycle; consumers only receive immutable stream items."""

    _STOP = object()

    def __init__(
        self,
        items: Iterable[Any] | None = None,
        maxsize: int = 128,
        delay_seconds: float = 0.05,
    ) -> None:
        self._source = iter(items or ())
        self._delay_seconds = float(delay_seconds)
        self._queue: Queue[Any] = Queue(maxsize=maxsize)
        self._paused = Event()
        self._stopped = Event()
        self._paused.set()
        self._thread: Thread | None = None

    def start(self, sequence: Iterable[Any] | None = None) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if sequence is not None:
            self._source = iter(sequence)
        self._stopped.clear()
        self._thread = Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _produce(self) -> None:
        try:
            for item in self._source:
                if self._stopped.is_set():
                    break
                self._paused.wait()
                self._queue.put(item)
                if self._delay_seconds > 0:
                    time.sleep(self._delay_seconds)
        finally:
            self._queue.put(self._STOP)

    def get(self, timeout: float | None = None) -> Any:
        item = self._queue.get(timeout=timeout)
        if item is self._STOP:
            self._queue.put(self._STOP)
            raise StopIteration
        return item

    def pause(self) -> None:
        self._paused.clear()

    def resume(self) -> None:
        self._paused.set()

    def stop(self) -> None:
        self._stopped.set()
        self._paused.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def __iter__(self) -> Iterator[Any]:
        while True:
            try:
                yield self.get(timeout=1.0)
            except StopIteration:
                return
            except Empty:
                if self._stopped.is_set():
                    return
