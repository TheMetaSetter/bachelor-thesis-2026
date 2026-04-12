from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseBatchAdapter(ABC):
    @abstractmethod
    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def forward_prepared(self, prepared_batch: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def postprocess_outputs(
        self,
        model_outputs: Any,
        prepared_batch: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError
