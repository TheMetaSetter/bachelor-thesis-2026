from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


# Unified API for all models
class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def training_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Training step nằm ngay bên trong model."""
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def test_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
