from __future__ import annotations

import torch
from torch import nn

from scripts.train import build_optimizer_from_experiment_config


def test_build_optimizer_defaults_to_adam_when_name_is_omitted() -> None:
    model = nn.Linear(2, 2)

    optimizer = build_optimizer_from_experiment_config(
        model,
        {"optimizer": {"learning_rate": 0.001, "weight_decay": 0.0}},
    )

    assert isinstance(optimizer, torch.optim.Adam)


def test_build_optimizer_supports_adamw() -> None:
    model = nn.Linear(2, 2)

    optimizer = build_optimizer_from_experiment_config(
        model,
        {
            "optimizer": {
                "optimizer_name": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
            }
        },
    )

    assert isinstance(optimizer, torch.optim.AdamW)
