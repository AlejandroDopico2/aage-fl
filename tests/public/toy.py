"""Modelo y datos de juguete para los tests: no dependen de vuestros modelos."""

import torch
from torch import nn
from torch.utils.data import TensorDataset


def toy_model(hidden: int = 16, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, hidden), nn.ReLU(), nn.Linear(hidden, 10))


def toy_data(n: int = 64, seed: int = 0) -> TensorDataset:
    g = torch.Generator().manual_seed(seed)
    return TensorDataset(torch.randn(n, 1, 28, 28, generator=g), torch.randint(0, 10, (n,), generator=g))
