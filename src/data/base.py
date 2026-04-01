from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSequenceParser(ABC):
    @abstractmethod
    def parse(self) -> dict[str, list[dict[str, Any]]]:
        raise NotImplementedError


class BaseDatasetBuilder(ABC):
    @abstractmethod
    def build(self, data_config: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def __call__(self, data_config: dict[str, Any]) -> dict[str, Any]:
        return self.build(data_config)
