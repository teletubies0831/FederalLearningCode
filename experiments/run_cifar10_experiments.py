"""Run CIFAR-10 low-quality data experiments across multiple baselines."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from main import simulate


def choose_faulty_clients(n_clients: int, ratio: float, seed: int) -> List[int]:
    if ratio <= 0:
        return []
    rng = np.random.RandomState(seed)
    target = min(n_clients, max(1, int(math.ceil(ratio * n_clients))))
    population = np.arange(1, n_clients + 1)
    if target >= len(population):
        return population.tolist()
    selected = rng.choice(population, size=target, replace=False)
    return sorted(int(x) for x in selected.tolist())


def format_table(ratios: Sequence[float], methods: Sequence[str], data: Dict[tuple, float]) -> str:
    header = ["Proportion"] + [m.upper() for m in methods]
    rows = []
    for ratio in ratios:
        row = [f"{ratio:.2f}"]
        for method in methods:
            acc = data.get((ratio, method))
            if acc is None or (isinstance(acc, float) and math.isnan(acc)):
                row.append("-")
            else:
                row.append(f"{acc * 100:.2f}%")
        rows.append(row)

    col_widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    parts = []
    parts.append(" | ".join(h.ljust(col_widths[i]) for i, h in enumerate(header)))
    parts.append("-+-".join("-" * w for w in col_widths))
    for row in rows:
        parts.append(" | ".join(row[i].ljust(col_widths[i]) for i in range(len(header))))
    return "\n".join(parts)


def save_csv(output_dir: Path, filename: str, fieldnames: Sequence[str], rows: List[Dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / filename).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_curves(output_dir: Path, family: str, ratios: Sequence[float], methods: Sequence[str], data: Dict[tuple, float]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    for method in methods:
        ys = [data.get((ratio, method), np.nan) for ratio in ratios]
        plt.plot(ratios, ys, marker="o", label=method.upper())
    plt.xlabel("Proportion of unreliable users")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    out_path = output_dir / f"accuracy_{family}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def run(args: argparse.Namespace) -> None:
    methods = [m.lower() for m in args.methods]
    ratios = [float(r) for r in args.ratios]
    output_dir = Path(args.output_dir)
    results: Dict[tuple, float] = {}
    csv_rows: List[Dict[str, float]] = []

    for ratio in ratios:
        faulty_uids = choose_faulty_clients(args.clients, ratio, seed=args.seed)
        for method in methods:
            sim_res = simulate(
                rounds=args.rounds,
                n_clients=args.clients,
                dim_m=args.dim,
                t_thr=args.threshold,
                seed=args.seed,
                dataset="cifar10",
                lr=args.lr,
                local_epochs=args.local_epochs,
                batch=args.batch,
                b_mode=args.b_mode,
                log_weights=False,
                log_local_eval=False,
                faulty_uids=faulty_uids,
                faulty_ratio=ratio,
                method=method,
                data_root=args.data_root,
                low_quality_family=args.family,
                low_quality_severity=args.severity,
                cifar_samples_per_client=args.samples_per_client,
                verbose=False,
            )
            final_acc = sim_res.get("final_test_acc")
            if final_acc is None and sim_res.get("round_metrics"):
                final_acc = sim_res["round_metrics"][-1].get("test_acc")
            results[(ratio, method)] = final_acc if final_acc is not None else float("nan")
            csv_rows.append(
                {
                    "ratio": ratio,
                    "method": method,
                    "accuracy": final_acc,
                    "faulty_clients": ",".join(str(uid) for uid in faulty_uids),
                }
            )

    print("\nAccuracy comparison (CIFAR-10, family=%s, severity=%.2f):" % (args.family, args.severity))
    print(format_table(ratios, methods, results))

    save_csv(output_dir, f"cifar10_{args.family}_summary.csv", ["ratio", "method", "accuracy", "faulty_clients"], csv_rows)
    plot_path = plot_curves(output_dir, args.family, ratios, methods, results)
    print(f"\nResults saved to {output_dir.resolve()}")
    print(f"Plot saved to {plot_path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CIFAR-10 low-quality robustness study")
    parser.add_argument("--methods", nargs="*", default=["ESFL", "PPFDL", "OURS"], help="methods to evaluate")
    parser.add_argument("--ratios", nargs="*", type=float, default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], help="proportions of unreliable users")
    parser.add_argument("--clients", type=int, default=10, help="number of federated clients")
    parser.add_argument("--dim", type=int, default=8, help="ignored for cifar10 (compatibility)")
    parser.add_argument("--threshold", type=int, default=5, help="Shamir threshold")
    parser.add_argument("--rounds", type=int, default=3, help="federated rounds")
    parser.add_argument("--local_epochs", type=int, default=1, help="local epochs per round")
    parser.add_argument("--batch", type=int, default=64, help="local batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="local learning rate")
    parser.add_argument("--b_mode", type=str, default="per_round", choices=["per_round", "per_user"], help="mask generation mode")
    parser.add_argument("--family", type=str, default="symmetric", choices=["symmetric", "class_dependent", "robust", "erasing"], help="low-quality data family")
    parser.add_argument("--severity", type=float, default=0.3, help="low-quality severity parameter")
    parser.add_argument("--samples_per_client", type=int, default=600, help="number of training samples per client")
    parser.add_argument("--data_root", type=str, default="./data", help="torchvision data root")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--output_dir", type=str, default="results", help="directory to store artifacts")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
