import pytest
import torch

from src.encoders import create_encoder


SETTINGS = dict(
    hidden_channels=8, output_dim=5, kernel_size=4, num_layers=3, dropout=0.1
)


def test_encoder_preserves_time_and_supports_gradients():
    model = create_encoder(SETTINGS, input_features=3, window_size=11)
    x = torch.randn(2, 11, 3, requires_grad=True)
    encoded = model(x)
    assert encoded.shape == (2, 11, 5)
    encoded.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None for parameter in model.parameters())
    with pytest.raises(ValueError, match="shape"):
        model(x[:, :-1])


def test_encoders_own_independent_weights():
    one = create_encoder(SETTINGS, input_features=3, window_size=11)
    two = create_encoder(SETTINGS, input_features=3, window_size=11)
    for first, second in zip(one.parameters(), two.parameters(), strict=True):
        assert first.data_ptr() != second.data_ptr()
