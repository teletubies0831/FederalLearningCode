"""Reproduce CIFAR-10 experiments under different unreliable-user ratios.

This module intentionally keeps the orchestration logic minimal so that the
focus stays on calling the existing :func:`main.simulate` entry point.  Every
experiment simply fixes a proportion ``p`` of unreliable clients, enables the
low-quality data simulator, and records the resulting test accuracy.  The
script mirrors the evaluation procedure described in the accompanying paper.
"""
from __future__ import annotations

import argparse
import csv
import math
from itertools import product
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from main import simulate


def sample_faulty_clients(
    rng: np.random.Generator, n_clients: int, proportion: float
) -> List[int]:
    """Return the client identifiers that act on low-quality data.

    The helper relies entirely on :mod:`numpy` utilities to avoid duplicating
    sampling logic.  It always rounds up to at least one faulty client when the
    proportion is positive, matching the experiment design in the paper.
    """

    proportion = float(np.clip(proportion, 0.0, 1.0))
    if proportion == 0.0 or n_clients <= 0:
        return []
    target = max(1, int(math.ceil(proportion * n_clients)))
    target = min(target, n_clients)
    population = np.arange(1, n_clients + 1)
    selection = rng.choice(population, size=target, replace=False)
    return sorted(int(uid) for uid in selection.tolist())


def run_experiments(
    methods: Sequence[str],
    proportions: Sequence[float],
    *,
    dataset: str,
    clients: int,
    rounds: int,
    threshold: int,
    lr: float,
    local_epochs: int,
    batch_size: int,
    b_mode: str,
    low_quality_family: str,
    low_quality_severity: float,
    samples_per_client: int | None,
    data_root: str,
    seed: int,
    output_dir: Path,
    verbose: bool,
) -> List[dict]:
    """Execute all experiment combinations and return raw result rows."""

    rng = np.random.default_rng(seed)
    rows: List[dict] = []

    for proportion, method in product(proportions, methods):
        faulty_uids = sample_faulty_clients(rng, clients, proportion)
        has_low_quality = bool(faulty_uids)

        simulation = simulate(
            rounds=rounds,
            n_clients=clients,
            dim=8,  # ignored by the CIFAR-10 path but kept for compatibility
            t_thr=threshold,
            seed=seed,
            dataset=dataset,
            lr=lr,
            local_epochs=local_epochs,
            batch=batch_size,
            b_mode=b_mode,
            faulty_uids=faulty_uids,
            faulty_ratio=proportion,
            method=method,
            data_root=data_root,
            low_quality_family=low_quality_family,
            low_quality_severity=low_quality_severity,
            cifar_samples_per_client=samples_per_client,
            verbose=verbose,
        )

        final_acc = simulation.get("final_test_acc")
        if final_acc is None and simulation.get("round_metrics"):
            final_acc = simulation["round_metrics"][-1].get("test_acc")

        rows.append(
            {
                "proportion": proportion,
                "method": method,
                "accuracy": final_acc,
                "faulty_clients": ",".join(str(uid) for uid in faulty_uids),
                "low_quality": has_low_quality,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    return rows


def summarise(rows: Iterable[dict], methods: Sequence[str], proportions: Sequence[float]) -> str:
    """Return a compact text table of accuracies."""

    lookup = {
        (row["proportion"], row["method"]): row.get("accuracy") for row in rows
    }
    header = ["p"] + [method.upper() for method in methods]
    lines = [" | ".join(header)]
    lines.append("-+-".join("-" * len(col) for col in header))
    for proportion in proportions:
        entries = [f"{proportion:.2f}"]
        for method in methods:
            acc = lookup.get((proportion, method))
            entries.append("-" if acc is None else f"{acc * 100:.2f}%")
        lines.append(" | ".join(entries))
    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["proportion", "method", "accuracy", "faulty_clients", "low_quality"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce CIFAR-10 experiments with low-quality users",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--methods", nargs="*", default=["ours", "esfl", "ppfdl"], help="methods to evaluate")
    parser.add_argument("--proportions", nargs="*", type=float, default=[0.0, 0.1, 0.2, 0.3], help="unreliable user ratios")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"], help="dataset to simulate")
    parser.add_argument("--clients", type=int, default=10, help="number of federated clients")
    parser.add_argument("--rounds", type=int, default=3, help="federated rounds")
    parser.add_argument("--threshold", type=int, default=5, help="Shamir threshold")
    parser.add_argument("--lr", type=float, default=0.01, help="local learning rate")
    parser.add_argument("--local_epochs", type=int, default=1, help="local epochs per round")
    parser.add_argument("--batch_size", type=int, default=64, help="local batch size")
    parser.add_argument("--b_mode", type=str, default="per_round", choices=["per_round", "per_user"], help="mask generation mode")
    parser.add_argument("--family", type=str, default="symmetric", help="low-quality policy family")
    parser.add_argument("--severity", type=float, default=0.3, help="low-quality severity parameter")
    parser.add_argument("--samples_per_client", type=int, default=600, help="training samples per client")
    parser.add_argument("--data_root", type=str, default="./data", help="dataset cache directory")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--output_dir", type=Path, default=Path("results"), help="where to store CSV summaries")
    parser.add_argument("--verbose", action="store_true", help="print verbose simulator logs")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    methods = [m.lower() for m in args.methods]
    proportions = [float(p) for p in args.proportions]

    rows = run_experiments(
        methods,
        proportions,
        dataset=args.dataset,
        clients=args.clients,
        rounds=args.rounds,
        threshold=args.threshold,
        lr=args.lr,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        b_mode=args.b_mode,
        low_quality_family=args.family,
        low_quality_severity=args.severity,
        samples_per_client=args.samples_per_client,
        data_root=str(args.data_root),
        seed=args.seed,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    csv_path = args.output_dir / f"cifar10_{args.family}_summary.csv"
    write_csv(csv_path, rows)
    print("\nExperiment summary:")
    print(summarise(rows, methods, proportions))
    print(f"\nDetailed results saved to {csv_path.resolve()}")


if __name__ == "__main__":
    main()

