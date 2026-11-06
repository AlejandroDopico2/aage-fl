"""Interfaz común de los agregadores."""

from __future__ import annotations

from fl.clients import ClientUpdate
from fl.models import StateDict


class Aggregator:
    """Combina las actualizaciones de los clientes en el nuevo modelo global.

    Para crear uno nuevo, heredad de esta clase en un archivo propio e implementad ``aggregate``.
    """

    def aggregate(self, updates: list[ClientUpdate]) -> StateDict:
        """Devuelve el ``state_dict`` del nuevo modelo global."""
        raise NotImplementedError
