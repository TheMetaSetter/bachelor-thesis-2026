from __future__ import annotations

import random

import numpy as np
import torch

from src.core.console import console_print


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    console_print("CONFIG", "Seeded runtime", seed=seed, cuda_available=torch.cuda.is_available())
