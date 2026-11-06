"""FedProx (Li et al., 2020): cliente de FedAvg con un término proximal en la pérdida."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset

from fl.clients.base import Client


class FedProxClient(Client):
    """Penaliza alejarse del modelo global: ``loss + (mu / 2) * ||w - w_global||²``.

    ``w`` son los parámetros locales y ``w_global`` los del modelo recibido. El servidor no
    cambia: FedProx se agrega con FedAvg.
    """

    def __init__(self, cid: int, dataset: Dataset, *, mu: float, **kwargs):
        super().__init__(cid, dataset, **kwargs)
        self.mu = mu

    def loss(self, model: nn.Module, global_model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # TODO: suma a la pérdida base (super().loss) el término proximal. Recorre a la vez
        #       model.parameters() y global_model.parameters(); usa .detach() en los globales.
        raise NotImplementedError("TODO: término proximal de FedProx")
