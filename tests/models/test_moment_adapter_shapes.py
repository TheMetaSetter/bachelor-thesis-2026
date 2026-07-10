from __future__ import annotations

from types import SimpleNamespace

import torch

from src.adapters.moment import MomentWindowAdapter


class _FakeMomentModel(torch.nn.Module):
    def forward(self, x_enc: torch.Tensor, input_mask: torch.Tensor) -> SimpleNamespace:
        embeddings = x_enc.mean(dim=-1)
        return SimpleNamespace(embeddings=embeddings, input_mask=input_mask)


def test_moment_adapter_prepares_and_postprocesses_batch_shapes() -> None:
    batch = {
        "x": torch.randn(3, 100, 38),
        "point_labels": torch.zeros(3, 100, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-1-1"} for _ in range(3)],
    }
    adapter = MomentWindowAdapter(moment_model=_FakeMomentModel(), context_length=512)

    prepared_batch = adapter.prepare_batch(batch)
    embedded_bundle = adapter.embed_batch(batch)

    assert prepared_batch["x_enc"].shape == (3, 38, 512)
    assert prepared_batch["input_mask"].shape == (3, 512)
    assert embedded_bundle["embeddings"].shape == (3, 38)
    assert embedded_bundle["window_labels"].shape == (3,)
    assert len(embedded_bundle["meta"]) == 3
