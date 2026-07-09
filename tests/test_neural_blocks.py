import torch

from src.models.neural_blocks import (
    SimpleWindowCnnEncoder,
    build_multilayer_perceptron,
)


def test_shared_mlp_preserves_batch_and_time_dimensions() -> None:
    network = build_multilayer_perceptron(
        input_dim=3,
        intermediate_dim=5,
        output_dim=2,
        num_linear_layers=3,
        dropout=0.0,
        apply_output_activation=False,
    )

    output = network(torch.zeros(2, 4, 3))

    assert output.shape == (2, 4, 2)


def test_shared_cnn_preserves_window_length() -> None:
    encoder = SimpleWindowCnnEncoder(
        input_dim=3,
        output_dim=5,
        hidden_channels=4,
        kernel_size=3,
        num_layers=3,
        dropout=0.0,
    )

    output = encoder(torch.zeros(2, 7, 3))

    assert output.shape == (2, 7, 5)
