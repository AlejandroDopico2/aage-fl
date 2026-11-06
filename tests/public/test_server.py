"""Tests públicos del servidor (fl/server.py)."""

import numpy as np
from toy import toy_data, toy_model
from torch.utils.data import Subset

from fl.clients import Client
from fl.codecs import state_dict_nbytes
from fl.server import LocalOnly, Server, sample_clients


def make_server(server_cls=Server):
    """4 clientes con 30 ejemplos cada uno; en cada ronda participa la mitad."""
    data = toy_data(n=120)
    clients = [Client(i, Subset(data, range(i, 120, 4)), lr=0.05) for i in range(4)]
    return server_cls(toy_model(hidden=8), clients, toy_data(n=50, seed=1), fraction=0.5, seed=0)


def test_sample_clients():
    rng = np.random.default_rng(0)
    for fraction, expected in [(0.3, 3), (0.5, 5), (1.0, 10)]:
        selected = sample_clients(10, fraction, rng)
        assert len(selected) == expected, f"con fraction={fraction} se esperaban {expected} clientes"
        assert len(set(selected)) == expected, "no se puede elegir dos veces al mismo cliente"
    assert len(sample_clients(10, 0.01, rng)) == 1, "siempre debe participar al menos un cliente"


def test_run_round():
    server = make_server()
    before = {k: v.clone() for k, v in server.model.state_dict().items()}
    row = server.run_round()

    missing = {"test_acc", "test_loss", "train_loss", "bytes_up", "bytes_down"} - set(row)
    assert not missing, f"faltan métricas: {missing}"
    assert any(not (v == before[k]).all() for k, v in server.model.state_dict().items()), (
        "el modelo global no ha cambiado tras la ronda"
    )
    # Sin compresión, cada uno de los 2 clientes elegidos recibe y envía un modelo entero.
    model_bytes = state_dict_nbytes(before)
    assert row["bytes_down"] == 2 * model_bytes, "bytes_down mal contados"
    assert row["bytes_up"] == 2 * model_bytes, "bytes_up mal contados"


def test_local_only():
    rows = make_server(LocalOnly).run(rounds=2)
    assert all(r["bytes_up"] == r["bytes_down"] == 0 for r in rows), "sin comunicación no se envían bytes"
    assert all(0 <= r["test_acc"] <= 1 for r in rows)
