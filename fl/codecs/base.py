"""Comunicación: interfaz de los codecs y medida de lo que ocupa un modelo."""

from __future__ import annotations

from fl.models import StateDict


def state_dict_nbytes(state_dict: StateDict) -> int:
    """Bytes que ocupan todos los tensores de ``state_dict``."""
    raise NotImplementedError("TODO: tamaño en bytes de un state_dict")


class Codec:
    """Transforma lo que un cliente envía al servidor. Este, el de por defecto, no comprime nada.

    - ``encode(state, reference)``: en el cliente; devuelve el ``payload`` que viaja por la red.
    - ``decode(payload, reference)``: en el servidor; reconstruye los pesos a partir del ``payload``.

    ``reference`` es el modelo global de la ronda (lo conocen cliente y servidor).
    """

    def encode(self, state: StateDict, reference: StateDict) -> StateDict:
        return state

    def decode(self, payload: StateDict, reference: StateDict) -> StateDict:
        return payload
