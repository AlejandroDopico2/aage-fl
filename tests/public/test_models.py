"""Tests públicos de los modelos (fl/models.py)."""

import torch

from fl.models import CNN, CNN_MAX_PARAMS, MLP, count_params

IMAGES = torch.randn(5, 1, 28, 28)  # un lote de 5 imágenes


def test_mlp():
    assert MLP(hidden=16)(IMAGES).shape == (5, 10), "la salida debe tener forma (lote, 10 clases)"


def test_cnn():
    model = CNN()
    assert model(IMAGES).shape == (5, 10), "la salida debe tener forma (lote, 10 clases)"
    assert count_params(model) <= CNN_MAX_PARAMS, f"la CNN no puede pasar de {CNN_MAX_PARAMS:,} parámetros"
