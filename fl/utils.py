"""Utilidades (DADO): configuración, semillas, dispositivo y registro de resultados."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
class Config(dict):
    """Diccionario con acceso por atributo: ``cfg.client.lr`` equivale a ``cfg["client"]["lr"]``."""

    def __init__(self, data: dict = ()):
        super().__init__({k: Config(v) if isinstance(v, dict) else v for k, v in dict(data).items()})

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"La configuración no tiene la clave {key!r}") from None


def load_config(path: str | None = None, overrides: list[str] = ()) -> Config:
    """Carga ``configs/default.yaml`` y le aplica, en orden, el YAML ``path`` y los ``overrides``.

    Cada override tiene la forma ``"seccion.clave=valor"`` (p. ej. ``"client.lr=0.05"``).
    Las claves que no existen en la configuración base dan error, así una errata no pasa desapercibida.
    """
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if path:
        _merge(cfg, yaml.safe_load(Path(path).read_text()) or {}, source=str(path))
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set espera 'clave=valor', no {item!r}")
        nested = _parse_value(value)
        for part in reversed(key.split(".")):
            nested = {part: nested}
        _merge(cfg, nested, source=f"--set {item}")
    return Config(cfg)


def _merge(base: dict, new: dict, source: str, prefix: str = "") -> None:
    for key, value in new.items():
        if key not in base:
            raise KeyError(f"{source}: clave desconocida '{prefix}{key}'. Válidas: {', '.join(base)}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f"{source}: '{prefix}{key}' es una sección, no un valor")
            _merge(base[key], value, source, f"{prefix}{key}.")
        else:
            base[key] = value


def _parse_value(text: str):
    value = yaml.safe_load(text)  # "0.1" -> 0.1, "true" -> True, "null" -> None
    if isinstance(value, str):  # YAML lee "1e-3" como texto
        try:
            return float(value)
        except ValueError:
            pass
    return value


# ---------------------------------------------------------------------------
# Reproducibilidad y dispositivo
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Fija las semillas de ``random``, NumPy y PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str = "auto") -> str:
    """``"auto"`` elige la GPU (CUDA) si está disponible y, si no, la CPU."""
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


# ---------------------------------------------------------------------------
# Registro de resultados
# ---------------------------------------------------------------------------
def run_name(cfg: Config) -> str:
    """Nombre legible a partir de lo que suele variar entre experimentos."""
    p, s, c = cfg.partition, cfg.server, cfg.client
    parts = [p.type]
    if p.type in ("dirichlet", "shards"):
        parts.append(f"a{p.alpha}")
    if p.type == "shards":
        parts.append(f"k{p.classes_per_client}")
    parts += [cfg.model.name, f"n{p.n_clients}", f"C{s.fraction}", f"E{c.local_epochs}"]
    if c.fedprox_mu > 0:
        parts.append(f"mu{c.fedprox_mu}")
    if s.aggregator != "fedavg":
        parts.append(s.aggregator)
    if s.codec != "none":
        parts.append(s.codec)
    if cfg.mode != "federated":
        parts.insert(0, cfg.mode)
    return "_".join(parts) + f"_s{cfg.seed}"


class RunLogger:
    """Guarda la configuración y las métricas de cada ronda con un formato común para todos.

    - ``<dir>/config.json``: configuración completa de la ejecución.
    - ``<dir>/metrics.csv``: una fila por ronda con las columnas de ``COLUMNS``.
    """

    COLUMNS = ["round", "test_acc", "test_loss", "train_loss", "bytes_up", "bytes_down", "bytes_cum", "time_s"]

    def __init__(self, cfg: Config):
        self.dir = Path(cfg.output.dir) / (cfg.output.name or run_name(cfg))
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(cfg, indent=2))
        self.metrics_path = self.dir / "metrics.csv"
        with open(self.metrics_path, "w", newline="") as f:
            csv.DictWriter(f, self.COLUMNS).writeheader()

    def log(self, row: dict) -> None:
        with open(self.metrics_path, "a", newline="") as f:
            csv.DictWriter(f, self.COLUMNS).writerow(row)
        print(
            f"ronda {row['round']:3d} | acc {row['test_acc']:.4f} | loss {row['test_loss']:.4f} "
            f"| {row['bytes_cum'] / 1e6:8.1f} MB | {row['time_s']:6.1f} s"
        )
