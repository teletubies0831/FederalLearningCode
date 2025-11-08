"""Batch runner that executes all configured experiments and produces summary artefacts."""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from typing import Any, Callable, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from federated.aggregation import SecureAggregationController
from federated.data import FederatedDataBundle, create_data_loaders
from federated.models import get_model_builder
from federated.trainer import FederatedTrainer, TrainingHistory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all configured federated learning experiments")
    parser.add_argument(
        "--config",
        default=os.path.join("configs", "experiments.json"),
        help="Path to the JSON configuration file",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_aggregator_factory(name: str, experiment_cfg: Dict[str, Any]) -> Callable[[int, int], Any]:
    name = name.lower()
    if name in {"ours", "secure_truth", "truth_discovery"}:
        max_iters = int(experiment_cfg.get("max_truth_iters", 5))
        alpha = float(experiment_cfg.get("truth_alpha", 0.05))
        scaling = float(experiment_cfg.get("truth_scaling", 1.0))
        variance_floor = float(experiment_cfg.get("variance_floor", 1e-9))

        def factory(num_clients: int, dropout_tolerance: int) -> SecureAggregationController:
            return SecureAggregationController(
                num_clients=num_clients,
                dropout_tolerance=dropout_tolerance,
                max_truth_iters=max_iters,
                truth_strategy="iterative",
                truth_alpha=alpha,
                truth_scaling=scaling,
                variance_floor=variance_floor,
            )

        return factory
    if name in {"esfl", "fedavg"}:

        def factory(num_clients: int, dropout_tolerance: int) -> SecureAggregationController:
            return SecureAggregationController(
                num_clients=num_clients,
                dropout_tolerance=dropout_tolerance,
                truth_strategy="uniform",
            )

        return factory
    if name == "ppfdl":
        alpha = float(experiment_cfg.get("truth_alpha", 0.05))
        scaling = float(experiment_cfg.get("truth_scaling", 1.0))
        variance_floor = float(experiment_cfg.get("variance_floor", 1e-9))

        def factory(num_clients: int, dropout_tolerance: int) -> SecureAggregationController:
            return SecureAggregationController(
                num_clients=num_clients,
                dropout_tolerance=dropout_tolerance,
                truth_strategy="ppfdl",
                truth_alpha=alpha,
                truth_scaling=scaling,
                variance_floor=variance_floor,
            )

        return factory
    if name in {"ppfdl_anchor", "ppfdl_kmeans"}:
        alpha = float(experiment_cfg.get("truth_alpha", 0.5))
        scaling = float(experiment_cfg.get("truth_scaling", 1.0))
        variance_floor = float(experiment_cfg.get("variance_floor", 1e-9))

        def factory(num_clients: int, dropout_tolerance: int) -> SecureAggregationController:
            return SecureAggregationController(
                num_clients=num_clients,
                dropout_tolerance=dropout_tolerance,
                truth_strategy="ppfdl_anchor",
                truth_alpha=alpha,
                truth_scaling=scaling,
                variance_floor=variance_floor,
            )

        return factory
    raise ValueError(f"Unsupported aggregator strategy: {name}")


def build_low_quality_config(methods: List[Dict[str, Any]], fraction: float) -> Dict[str, Any] | None:
    if not methods or fraction <= 0:
        return None
    return {"fraction": fraction, "methods": methods}


def _format_low_quality_methods(methods: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for method in methods:
        method_type = str(method.get("type", "unknown"))
        if method_type == "label_noise":
            parts.append(f"label_noise(p={float(method.get('prob', 0.0)):.2f})")
        elif method_type == "gaussian_noise":
            parts.append(f"gaussian_noise(std={float(method.get('std', 0.0)):.2f})")
        elif method_type == "gaussian_blur":
            parts.append(f"gaussian_blur(sigma={float(method.get('sigma', 0.0)):.2f})")
        elif method_type == "pixel_dropout":
            parts.append(f"pixel_dropout(p={float(method.get('drop_prob', 0.0)):.2f})")
        else:
            parts.append(method_type)
    return ", ".join(parts) if parts else "none"


def log_low_quality_pipeline(scope: str, config: Dict[str, Any] | None) -> None:
    if not config:
        print(f"[{scope}] Low-quality data pipeline: disabled")
        return
    fraction = float(config.get("fraction", 0.0))
    methods = config.get("methods", [])
    method_desc = _format_low_quality_methods(methods if isinstance(methods, list) else [])
    print(f"[{scope}] Low-quality data pipeline: fraction={fraction:.2f} | methods={method_desc}")


def extract_shared_parameters(global_cfg: Dict[str, Any], experiment_cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "dataset",
        "data_dir",
        "num_clients",
        "rounds",
        "local_epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "dropout_rate",
        "dropout_tolerance",
        "seed",
    ]
    params = {key: global_cfg.get(key) for key in keys}
    for key in keys:
        if key in experiment_cfg:
            params[key] = experiment_cfg[key]
    missing = [key for key, value in params.items() if value is None]
    if missing:
        raise ValueError(f"Missing configuration values for keys: {missing}")
    return params


def run_single_configuration(
    label: str,
    aggregator_factory: Callable[[int, int], Any],
    shared_params: Dict[str, Any],
    fractions: Iterable[float],
    methods: List[Dict[str, Any]],
) -> Dict[str, Any]:
    experiment_results: List[Dict[str, Any]] = []
    history_payload: List[Dict[str, Any]] = []

    for fraction in fractions:
        setup_seed(int(shared_params["seed"]))
        low_quality_config = build_low_quality_config(methods, float(fraction))
        log_low_quality_pipeline(f"{label} | fraction={fraction:.2f}", low_quality_config)

        data_bundle: FederatedDataBundle = create_data_loaders(
            dataset_name=str(shared_params["dataset"]),
            num_clients=int(shared_params["num_clients"]),
            batch_size=int(shared_params["batch_size"]),
            low_quality_config=low_quality_config,
            seed=int(shared_params["seed"]),
            data_dir=str(shared_params["data_dir"]),
        )

        if int(shared_params["dropout_tolerance"]) >= len(data_bundle.client_loaders):
            raise ValueError("dropout_tolerance must be smaller than the total number of clients")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_builder = get_model_builder(
            str(shared_params["dataset"]), data_bundle.info.num_classes
        )

        trainer = FederatedTrainer(
            model_builder=model_builder,
            client_loaders=data_bundle.client_loaders,
            test_loader=data_bundle.test_loader,
            device=device,
            lr=float(shared_params["lr"]),
            local_epochs=int(shared_params["local_epochs"]),
            dropout_tolerance=int(shared_params["dropout_tolerance"]),
            weight_decay=float(shared_params["weight_decay"]),
            aggregator_factory=aggregator_factory,
            low_quality_clients=data_bundle.low_quality_clients,
        )

        _, history = trainer.train(
            num_rounds=int(shared_params["rounds"]),
            dropout_rate=float(shared_params["dropout_rate"]),
        )
        final_metrics: TrainingHistory = history[-1]
        experiment_results.append(
            {
                "fraction": float(fraction),
                "final_loss": float(final_metrics.test_loss),
                "final_accuracy": float(final_metrics.test_accuracy),
            }
        )
        history_payload.append(
            {
                "fraction": float(fraction),
                "history": [asdict(entry) for entry in history],
                "low_quality_clients": data_bundle.low_quality_clients,
            }
        )

    return {
        "label": label,
        "results": sorted(experiment_results, key=lambda item: item["fraction"]),
        "histories": sorted(history_payload, key=lambda item: item["fraction"]),
    }


def save_histories(histories: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(histories, handle, indent=2)


def save_plot_data(experiments: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = ["label", "fraction", "final_loss", "final_accuracy"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for experiment in experiments:
            for record in experiment["results"]:
                writer.writerow(
                    {
                        "label": experiment["label"],
                        "fraction": record["fraction"],
                        "final_loss": record["final_loss"],
                        "final_accuracy": record["final_accuracy"],
                    }
                )


def plot_accuracy_curves(experiments: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(6, 5))
    for experiment in experiments:
        fractions = [record["fraction"] for record in experiment["results"]]
        accuracies = [record["final_accuracy"] for record in experiment["results"]]
        plt.plot(fractions, accuracies, marker="o", label=experiment["label"])
    plt.xlabel("Proportion of unreliable users")
    plt.ylabel("Accuracy")
    plt.title("Comparison of FL strategies under unreliable clients")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    global_cfg: Dict[str, Any] = config.get("global", {})
    experiments_cfg: List[Dict[str, Any]] = config.get("experiments", [])
    output_cfg: Dict[str, Any] = config.get("output", {})

    if not experiments_cfg:
        raise ValueError("Configuration file must define at least one experiment entry")

    results_dir = output_cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    aggregated_results: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []

    for experiment_cfg in experiments_cfg:
        label = experiment_cfg.get("label", experiment_cfg.get("name", "Unnamed"))
        aggregator_name = experiment_cfg.get("aggregator", experiment_cfg.get("name", ""))
        aggregator_factory = resolve_aggregator_factory(aggregator_name, experiment_cfg)
        shared_params = extract_shared_parameters(global_cfg, experiment_cfg)
        methods = experiment_cfg.get(
            "low_quality_methods",
            global_cfg.get("low_quality_methods", []),
        )
        fractions = experiment_cfg.get("fractions") or experiment_cfg.get("low_quality_fractions")
        if not fractions:
            raise ValueError(f"Experiment '{label}' must provide a list of fractions")

        print("=" * 80)
        print(f"Running experiment: {label} | Strategy: {aggregator_name}")
        experiment_result = run_single_configuration(
            label=label,
            aggregator_factory=aggregator_factory,
            shared_params=shared_params,
            fractions=fractions,
            methods=list(methods),
        )
        aggregated_results.append(experiment_result)
        histories.append({"label": label, "histories": experiment_result["histories"]})

    data_filename = output_cfg.get("data_filename", "accuracy_plot_data.csv")
    plot_filename = output_cfg.get("plot_filename", "accuracy_comparison.png")
    history_filename = output_cfg.get("history_filename", "training_histories.json")

    data_path = os.path.join(results_dir, data_filename)
    plot_path = os.path.join(results_dir, plot_filename)
    history_path = os.path.join(results_dir, history_filename)

    save_plot_data(aggregated_results, data_path)
    save_histories(histories, history_path)
    plot_accuracy_curves(aggregated_results, plot_path)

    print("=" * 80)
    print(f"Stored plot data at: {data_path}")
    print(f"Stored detailed histories at: {history_path}")
    print(f"Saved comparison plot to: {plot_path}")


if __name__ == "__main__":
    main()

