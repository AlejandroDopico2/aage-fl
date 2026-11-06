"""Tests públicos del entrenamiento local (fl/clients/)."""

import math

import torch
from toy import toy_data, toy_model

from fl.clients import Client, ClientUpdate, FedProxClient


def distance(state_a, state_b):
    return math.sqrt(sum(((state_a[k] - state_b[k]) ** 2).sum().item() for k in state_a))


def test_local_train():
    model = toy_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    update = Client(7, toy_data(n=50), lr=0.1).local_train(model)

    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), f"se ha modificado {name} del modelo global"
    assert isinstance(update, ClientUpdate)
    assert update.cid == 7 and update.n_samples == 50
    assert list(update.state_dict) == list(before), "el state_dict debe tener las mismas claves que el modelo"
    assert distance(update.state_dict, before) > 0, "los pesos locales no han cambiado al entrenar"
    assert math.isfinite(update.metrics["train_loss"]), 'falta metrics["train_loss"] o no es un número'


def test_fedprox():
    model, data = toy_model(), toy_data()
    distances = {}
    for name, client in [
        ("fedavg", Client(0, data, lr=0.1, local_epochs=5)),
        ("fedprox", FedProxClient(0, data, mu=1.0, lr=0.1, local_epochs=5)),
    ]:
        torch.manual_seed(0)  # mismo orden de mini-lotes en ambos casos
        distances[name] = distance(client.local_train(model).state_dict, model.state_dict())
    assert distances["fedprox"] < distances["fedavg"], "con FedProx el modelo local debe quedar más cerca del global"
