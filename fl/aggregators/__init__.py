"""Agregadores: un archivo por método. Para añadir uno, registradlo en ``build_aggregator``."""

from fl.aggregators.base import Aggregator
from fl.aggregators.fedavg import FedAvg
from fl.utils import Config


__all__ = ["Aggregator", "FedAvg", "build_aggregator"]


def build_aggregator(cfg: Config) -> Aggregator:
    """Crea el agregador indicado en ``cfg.aggregator`` (sección ``server`` de la configuración)."""
    if cfg.aggregator == "fedavg":
        return FedAvg()
    raise ValueError(f"Agregador desconocido: {cfg.aggregator!r}")
