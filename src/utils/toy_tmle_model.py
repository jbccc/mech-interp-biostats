from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from .torch_utils import build_mlp


class ToyTMLE(nn.Module):
    """
    Toy TMLE (Targeted Maximum Likelihood Estimation) neural network.

    This network uses a shared representation for features followed by
    separate heads for outcome prediction and propensity score estimation,
    which is useful for causal inference with TMLE methodology.

    Args:
        hidden_layers (int): Number of hidden layers in the network
        hidden_size (int): Size of each hidden layer
        input_dim (int): Dimension of input covariates W
        dropout_rate (float): Dropout rate to apply within layers
    """

    def __init__(
        self,
        hidden_layers=6,
        hidden_size=10,
        input_dim=3,
        dropout_rate=0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        # shared representation layers
        hidden_dims = [hidden_size] * (hidden_layers - 1)
        self.shared = build_mlp(
            input_dim,
            hidden_size,
            hidden_dims[:-1],
            activation=nn.ReLU,
            dropout_rate=dropout_rate,
        )

        # outcome head (Q)
        self.Q = nn.Linear(hidden_size + 1, 1)

        # propensity head (g)
        propensity_layers = []
        propensity_layers.append(nn.Linear(hidden_size, 1))
        propensity_layers.append(nn.Sigmoid())
        self.g = nn.Sequential(*propensity_layers)

        self.layer_names = [f"shared.{i}" for i in range(len(self.shared))] + ["Q", "g"]

    def forward(self, W: Tensor, A: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            W (torch.Tensor): Covariate tensor (batch_size, input_dim)
            A (torch.Tensor): Treatment assignment tensor (batch_size, 1)

        Returns:
            tuple: Contains:
                - Y_hat (torch.Tensor): Predicted outcome (Q function output)
                - A_hat (torch.Tensor): Predicted propensity score (g function output)
        """
        h = self.shared(W)

        A_hat = self.g(h)
        Y_hat = self.Q(torch.concat((h, A), dim=-1))

        return Y_hat.flatten(), A_hat.flatten()

    __call__: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] = forward
