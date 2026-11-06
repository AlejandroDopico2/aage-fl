"""Tests públicos de los agregadores (fl/aggregators/)."""

import torch

from fl.aggregators import FedAvg
from fl.clients import ClientUpdate


def test_fedavg():
    a = ClientUpdate(cid=0, state_dict={"w": torch.tensor([0.0, 0.0]), "b": torch.tensor([1.0])}, n_samples=1)
    b = ClientUpdate(cid=1, state_dict={"w": torch.tensor([4.0, 8.0]), "b": torch.tensor([5.0])}, n_samples=3)
    result = FedAvg().aggregate([a, b])  # pesos 1/4 y 3/4
    assert torch.allclose(result["w"], torch.tensor([3.0, 6.0])), "la media debe ponderarse por n_samples"
    assert torch.allclose(result["b"], torch.tensor([4.0])), "la media debe ponderarse por n_samples"
