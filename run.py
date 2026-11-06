"""Lanza un experimento de aprendizaje federado

Se parte siempre de configs/default.yaml:

    python run.py
    python run.py --config configs/examples/iid.yaml
    python run.py --set partition.alpha=1.0 client.local_epochs=5 server.fraction=1.0
"""

import argparse

from fl.aggregators import build_aggregator
from fl.clients import build_clients
from fl.codecs import build_codec
from fl.data import load_dataset, make_partition, plot_partition
from fl.models import build_model, count_params
from fl.server import LocalOnly, Server
from fl.utils import RunLogger, get_device, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML con los cambios respecto a configs/default.yaml")
    parser.add_argument(
        "--set", nargs="+", action="extend", default=[], metavar="CLAVE=VALOR", help="p. ej. client.lr=0.05"
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    logger = RunLogger(cfg)

    # Datos: cada cliente recibe un trozo del conjunto de entrenamiento; el test es global.
    train, test = load_dataset(cfg.data.dataset, cfg.data.root)
    labels = train.tensors[1].numpy()
    partition = make_partition(labels, cfg.partition, cfg.seed)
    plot_partition(labels, partition, logger.dir / "partition.png")
    clients = build_clients(train, partition, cfg, device)

    model = build_model(cfg.model)
    print(f"{logger.dir.name}: {count_params(model):,} parámetros, {len(clients)} clientes, {device}")

    server_cls = {"federated": Server, "local_only": LocalOnly}[cfg.mode]
    server = server_cls(
        model,
        clients,
        test,
        fraction=cfg.server.fraction,
        aggregator=build_aggregator(cfg.server),
        codec=build_codec(cfg.server),
        seed=cfg.seed,
        device=device,
    )
    server.run(cfg.server.rounds, logger)
    print(f"Resultados en {logger.dir}")


if __name__ == "__main__":
    main()
