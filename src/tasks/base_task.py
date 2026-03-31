from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTask(ABC):
    @abstractmethod
    def training_step(self, model: Any, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, model: Any, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def test_step(self, model: Any, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

