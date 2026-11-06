"""Tests públicos de la comunicación (fl/codecs/)."""

import torch
from toy import toy_model

from fl.codecs import build_codec, state_dict_nbytes
from fl.utils import load_config


def test_state_dict_nbytes():
    state = {
        "a": torch.zeros(10, dtype=torch.float32),  # 10 × 4 = 40 bytes
        "b": torch.zeros(2, 3, dtype=torch.int8),  # 6 × 1 = 6 bytes
        "c": torch.zeros(4, dtype=torch.float64),  # 4 × 8 = 32 bytes
    }
    assert state_dict_nbytes(state) == 78


def test_int8_codec():
    """Con ``server.codec: int8`` en la config, la subida ocupa < 30 % y los pesos se recuperan bien."""
    codec = build_codec(load_config(overrides=["server.codec=int8"]).server)
    torch.manual_seed(0)
    reference = toy_model(hidden=32).state_dict()  # el modelo global de la ronda
    state = {k: v + 0.01 * torch.randn_like(v) for k, v in reference.items()}  # el modelo local
    payload = codec.encode(state, reference)
    decoded = codec.decode(payload, reference)
    assert state_dict_nbytes(payload) < 0.3 * state_dict_nbytes(state), "lo enviado debe ocupar < 30 %"
    assert list(decoded) == list(state), "decode debe devolver las mismas claves, en el mismo orden"
    for name in state:
        assert torch.allclose(decoded[name], state[name], atol=1e-3), f"{name} no se recupera bien"
