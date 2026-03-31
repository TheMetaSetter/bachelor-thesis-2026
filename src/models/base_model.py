from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn


class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch: dict) -> dict:
        raise NotImplementedError

