"""FedAvg (McMahan et al., 2017): media ponderada por el nº de ejemplos de cada cliente."""

from __future__ import annotations

from fl.aggregators.base import Aggregator
from fl.clients import ClientUpdate
from fl.models import StateDict


class FedAvg(Aggregator):
    def aggregate(self, updates: list[ClientUpdate]) -> StateDict:
        """Devuelve, para cada tensor, ``sum_k (n_k / n) * w_k`` con ``n = sum_k n_k``.

        - Conserva las claves, su orden y el ``dtype`` de cada tensor.
        - Los tensores enteros (p. ej. ``num_batches_tracked`` de BatchNorm) no pueden quedar
          como float: redondea la media y vuelve al tipo original.
        """
        # TODO: recorre las claves del primer state_dict y combina los tensores de todos los clientes.
        raise NotImplementedError("TODO: FedAvg")
