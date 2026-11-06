"""Servidor: selecciona clientes, agrega sus actualizaciones y evalúa el modelo global.

Los agregadores están en ``fl/aggregators/`` y los codecs en ``fl/codecs/``.
"""

from __future__ import annotations

import copy
import time

import numpy as np
from torch import nn
from torch.utils.data import Dataset

from fl.aggregators import Aggregator, FedAvg
from fl.clients import Client
from fl.codecs import Codec, state_dict_nbytes
from fl.models import evaluate
from fl.utils import RunLogger


def sample_clients(n_clients: int, fraction: float, rng: np.random.Generator) -> list[int]:
    """Elige al azar, sin repetición, ``max(1, round(fraction * n_clients))`` clientes.

    Devuelve sus identificadores (0..n_clients-1) ordenados de menor a mayor.
    """
    # TODO: usa rng (no np.random) para que el muestreo sea reproducible.
    raise NotImplementedError("TODO: muestreo de clientes")


class Server:
    """Orquesta el entrenamiento federado: ``run`` (dado) repite ``run_round`` (vuestro)."""

    def __init__(
        self,
        model: nn.Module,
        clients: list[Client],
        test_set: Dataset,
        *,
        fraction: float = 1.0,
        aggregator: Aggregator | None = None,
        codec: Codec | None = None,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.clients = clients
        self.test_set = test_set
        self.fraction = fraction
        self.aggregator = aggregator or FedAvg()
        self.codec = codec or Codec()
        self.rng = np.random.default_rng(seed)
        self.device = device

    def run(self, rounds: int, logger: RunLogger | None = None) -> list[dict]:
        """Ejecuta ``rounds`` rondas y devuelve (y registra) una fila de métricas por ronda."""
        history, bytes_cum, start = [], 0, time.perf_counter()
        for rnd in range(1, rounds + 1):
            row = {"round": rnd, **self.run_round()}
            bytes_cum += row["bytes_up"] + row["bytes_down"]
            row.update(bytes_cum=bytes_cum, time_s=round(time.perf_counter() - start, 2))
            history.append(row)
            if logger:
                logger.log(row)
        return history

    def run_round(self) -> dict:
        """Una ronda de FL: el servidor elige clientes, estos entrenan a partir del modelo global y
        envían su actualización (a través de ``self.codec``), el servidor las agrega y evalúa el
        nuevo modelo global sobre ``self.test_set``.

        Devuelve un diccionario con ``test_acc``, ``test_loss``, ``train_loss`` (media de los
        clientes) y los bytes enviados en la ronda: ``bytes_up`` (clientes -> servidor) y
        ``bytes_down`` (servidor -> clientes).
        """
        raise NotImplementedError("TODO: una ronda de aprendizaje federado")


class LocalOnly(Server):
    """Referencia «suelo»: cada cliente entrena solo con sus datos y nunca se comunica.

    Cada cliente conserva su propio modelo entre rondas (todos parten del mismo modelo inicial).
    En cada ronda entrenan todos y se registra la media de sus métricas sobre el test global.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_models = [copy.deepcopy(self.model) for _ in self.clients]

    def run_round(self) -> dict:
        # TODO (ligero): entrena cada modelo local con su cliente, cárgale los pesos resultantes y
        #                evalúalo. No hay comunicación: los bytes son 0.
        raise NotImplementedError("TODO: ronda sin comunicación")
