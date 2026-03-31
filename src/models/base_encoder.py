from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Return {'hidden': [B, L, H], 'pooled': Optional[B, H], 'aux': dict}."""
        raise NotImplementedError
