"""Compara experimentos a partir de sus metrics.csv (DADO, ampliable).

    python plot.py runs/*                            # todas las ejecuciones
    python plot.py runs/iid_* runs/dirichlet_* -o comparativa.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_metrics(run_dir: Path) -> dict[str, list[float]]:
    """Lee ``metrics.csv`` como ``{columna: [valor por ronda]}``."""
    with open(run_dir / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return {col: [float(row[col]) for row in rows] for col in rows[0]} if rows else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="carpetas de resultados (runs/<nombre>)")
    parser.add_argument("-o", "--out", type=Path, default=Path("runs/comparison.png"))
    args = parser.parse_args()

    fig, (ax_acc, ax_loss, ax_mb) = plt.subplots(1, 3, figsize=(16, 4.5))
    for run_dir in sorted(d for d in args.runs if (d / "metrics.csv").exists()):
        m = read_metrics(run_dir)
        if not m:
            continue
        ax_acc.plot(m["round"], m["test_acc"], label=run_dir.name)
        ax_loss.plot(m["round"], m["test_loss"])
        ax_mb.plot([b / 1e6 for b in m["bytes_cum"]], m["test_acc"])

    ax_acc.set(xlabel="ronda", ylabel="accuracy (test)")
    ax_loss.set(xlabel="ronda", ylabel="pérdida (test)", yscale="log")
    ax_mb.set(xlabel="MB transmitidos (acumulado)", ylabel="accuracy (test)")
    for ax in (ax_acc, ax_loss, ax_mb):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="small")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"Figura guardada en {args.out}")


if __name__ == "__main__":
    main()
