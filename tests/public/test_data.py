"""Tests públicos del particionado (fl/data.py)."""

import numpy as np

from fl.data import dirichlet, dirichlet_shards, iid

# 6000 ejemplos, 600 de cada una de las 10 clases, desordenados.
LABELS = np.random.default_rng(0).permutation(np.repeat(np.arange(10), 600))


def check_is_partition(partition, n_clients=10):
    """Cada índice de LABELS aparece en exactamente un cliente."""
    assert len(partition) == n_clients, f"se esperaban {n_clients} clientes y hay {len(partition)}"
    all_indices = np.concatenate(partition)
    assert len(all_indices) == len(np.unique(all_indices)), "hay índices repetidos entre clientes"
    assert len(all_indices) == len(LABELS), "hay ejemplos que no se han asignado a ningún cliente"


def class_frequencies(indices):
    return np.bincount(LABELS[indices], minlength=10) / len(indices)


def test_iid():
    partition = iid(LABELS, 10, seed=0)
    check_is_partition(partition)
    sizes = [len(p) for p in partition]
    assert max(sizes) - min(sizes) <= 1, "en IID todos los clientes deben tener el mismo tamaño (±1)"
    for idx in partition:
        assert np.all(np.abs(class_frequencies(idx) - 0.1) < 0.05), "en IID cada cliente ve todas las clases (~10 %)"


def test_dirichlet():
    partition = dirichlet(LABELS, 10, alpha=0.05, seed=0)
    check_is_partition(partition)
    dominant = [class_frequencies(idx).max() for idx in partition]
    assert np.mean(dominant) > 0.5, "con alpha pequeño en cada cliente debería dominar alguna clase"


def test_shards():
    for k in (1, 2, 3):
        partition = dirichlet_shards(LABELS, 10, classes_per_client=k, alpha=1.0, seed=0)
        check_is_partition(partition)
        for idx in partition:
            assert len(np.unique(LABELS[idx])) <= k, f"con k={k} un cliente no puede tener más de {k} clases"
