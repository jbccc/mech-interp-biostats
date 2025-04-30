from typing import NamedTuple

import torch.nn as nn
from torch.utils.data import DataLoader

DataLoaders = NamedTuple("DataLoaders", (("train", DataLoader), ("val", DataLoader)))


def build_mlp(input_dim, output_dim, hidden_dims, activation=nn.ReLU, dropout_rate=0.0):
    """
    Build a multi-layer perceptron with configurable architecture.

    Creates a PyTorch Sequential module representing an MLP with specified
    layer dimensions, activation functions, batch normalization, and dropout.

    Args:
        input_dim (int): Dimension of input features
        output_dim (int): Dimension of output features
        hidden_dims (list): List of hidden layer dimensions
        activation (nn.Module): Activation function class to use after each hidden layer
        dropout_rate (float): Dropout probability (0.0 means no dropout)

    Returns:
        nn.Sequential: A PyTorch Sequential module containing the constructed MLP
    """
    layers = []
    current_dim = input_dim
    if hidden_dims:
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(activation())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)
