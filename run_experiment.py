"""Command line entry point for running the federated learning experiments."""
from __future__ import annotations

import argparse
import random
from typing import Dict, List, Optional

import numpy as np
import torch

from federated.data import create_data_loaders
from federated.models import get_model_builder
from federated.trainer import FedAvgTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated learning with low-quality data simulation")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "femnist"], default="mnist")
    parser.add_argument("--data-dir", default="data", help="Directory for downloading datasets")
    parser.add_argument("--num-clients", type=int, default=10, help="Number of simulated clients")
    parser.add_argument("--clients-per-round", type=int, default=None, help="Number of clients participating in each round")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Number of local epochs per client")
    parser.add_argument("--batch-size", type=int, default=32, help="Local batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for SGD")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for SGD")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--low-quality-fraction", type=float, default=0.0, help="Fraction of clients with degraded data")
    parser.add_argument("--label-noise", type=float, default=0.0, help="Probability of random label flips")
    parser.add_argument("--gaussian-noise-std", type=float, default=0.0, help="Standard deviation of additive Gaussian noise")
    parser.add_argument("--gaussian-blur-sigma", type=float, default=0.0, help="Sigma for Gaussian blur (kernel size inferred)")
    parser.add_argument("--pixel-dropout", type=float, default=0.0, help="Pixel dropout probability for low-quality clients")
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_low_quality_config(args: argparse.Namespace) -> Optional[Dict[str, object]]:
    methods: List[Dict[str, float]] = []
    if args.label_noise > 0:
        methods.append({"type": "label_noise", "prob": args.label_noise})
    if args.gaussian_noise_std > 0:
        methods.append({"type": "gaussian_noise", "std": args.gaussian_noise_std})
    if args.gaussian_blur_sigma > 0:
        # Follow torchvision convention of using odd kernel sizes.
        methods.append({"type": "gaussian_blur", "sigma": args.gaussian_blur_sigma, "kernel_size": 5})
    if args.pixel_dropout > 0:
        methods.append({"type": "pixel_dropout", "drop_prob": args.pixel_dropout})

    if not methods or args.low_quality_fraction <= 0:
        return None
    return {"fraction": args.low_quality_fraction, "methods": methods}


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)

    # 根据命令行参数组合低质量数据策略
    low_quality_config = build_low_quality_config(args)
    client_loaders, test_loader, info = create_data_loaders(
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        low_quality_config=low_quality_config,
        seed=args.seed,
        data_dir=args.data_dir,
    )

    clients_per_round = args.clients_per_round or args.num_clients
    if clients_per_round > len(client_loaders):
        raise ValueError("clients_per_round cannot exceed the total number of clients")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_builder = get_model_builder(args.dataset, info.num_classes)
    trainer = FedAvgTrainer(
        model_builder=model_builder,
        client_loaders=client_loaders,
        test_loader=test_loader,
        device=device,
        lr=args.lr,
        local_epochs=args.local_epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    _, history = trainer.train(num_rounds=args.rounds, clients_per_round=clients_per_round)

    final_loss, final_acc = history[-1].test_loss, history[-1].test_accuracy
    print("=" * 80)
    print(f"Final test loss: {final_loss:.4f} | Final test accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main()
