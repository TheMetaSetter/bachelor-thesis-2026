from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSequenceParser(ABC):
    @abstractmethod
    def parse(self) -> dict[str, list[dict[str, Any]]]:
        raise NotImplementedError

