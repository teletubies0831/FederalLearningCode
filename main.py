#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal federated learning simulator with verifiable secure aggregation.

The module keeps the original functionality—toy gradients, MNIST, and CIFAR-10
experiments—but rewrites the data preparation pipeline so that shared pieces of
logic are factored into small helpers.  The resulting script is considerably
shorter and avoids duplicated training routines while remaining fully
configurable from the command line.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from field import FixedField
from low_quality import LowQualityPolicy
from protocol import TA, Client, Server, client_side_verify_and_update


# ---------------------------------------------------------------------------
# Neural network definitions and tensor helpers
# ---------------------------------------------------------------------------


class SimpleCNN(nn.Module):
    """LeNet style network for MNIST."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = torch.relu(nn.functional.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class SimpleCIFAR10CNN(nn.Module):
    """Compact CIFAR-10 convolutional model."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def torch_parameters_to_vector(model: nn.Module) -> np.ndarray:
    with torch.no_grad():
        vec = torch.cat([p.view(-1) for p in model.parameters()])
    return vec.detach().cpu().numpy().astype(np.float64)


def torch_vector_to_parameters(model: nn.Module, vec: np.ndarray) -> None:
    with torch.no_grad():
        tensor = torch.from_numpy(vec.astype(np.float32))
        pointer = 0
        for param in model.parameters():
            numel = param.numel()
            param.copy_(tensor[pointer : pointer + numel].view_as(param))
            pointer += numel


def evaluate_model_from_vector(
    model_cls: Callable[[], nn.Module], vec: np.ndarray, loader: DataLoader, device: str
) -> Tuple[float, float]:
    model = model_cls().to(device)
    torch_vector_to_parameters(model, vec)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += float(loss.item())
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total_samples += yb.size(0)
    denom = max(1, total_samples)
    return total_loss / denom, total_correct / denom


# ---------------------------------------------------------------------------
# Dataset building blocks
# ---------------------------------------------------------------------------


def build_mnist_loaders(n_clients: int, batch: int, data_root: str) -> Tuple[Dict[int, DataLoader], DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    indices = np.arange(len(train_ds))
    np.random.shuffle(indices)
    splits = np.array_split(indices, n_clients)

    client_loaders: Dict[int, DataLoader] = {}
    for idx, part in enumerate(splits, start=1):
        subset = Subset(train_ds, part.tolist())
        client_loaders[idx] = DataLoader(subset, batch_size=batch, shuffle=True, num_workers=0)

    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)
    return client_loaders, test_loader


class ClientCIFAR10Dataset(Dataset):
    """Client-side CIFAR-10 dataset that applies optional low-quality policies."""

    def __init__(
        self,
        base_dataset: datasets.CIFAR10,
        indices: Iterable[int],
        transform: Optional[transforms.Compose],
        policy: Optional[LowQualityPolicy],
        seed: int,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform
        self.policy = policy.spawn(seed) if policy else None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[self.indices[idx]]
        if self.policy is not None:
            image, label = self.policy.apply(image, label)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_cifar10_loaders(
    n_clients: int,
    batch: int,
    *,
    data_root: str,
    seed: int,
    samples_per_client: Optional[int],
    low_quality_policies: Optional[Dict[int, LowQualityPolicy]],
) -> Tuple[Dict[int, DataLoader], DataLoader]:
    transform_train = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    base_train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=None)
    test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

    indices = np.arange(len(base_train))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    if samples_per_client is not None:
        total = min(len(indices), int(samples_per_client) * n_clients)
        indices = indices[:total]

    splits = np.array_split(indices, n_clients)
    client_loaders: Dict[int, DataLoader] = {}
    for idx, part in enumerate(splits, start=1):
        policy = low_quality_policies[idx] if low_quality_policies and idx in low_quality_policies else None
        subset = ClientCIFAR10Dataset(
            base_dataset=base_train,
            indices=part.tolist(),
            transform=transform_train,
            policy=policy,
            seed=seed + 97 * idx,
        )
        client_loaders[idx] = DataLoader(subset, batch_size=batch, shuffle=True, num_workers=0)

    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)
    return client_loaders, test_loader


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


@dataclass
class DatasetSetup:
    init_vector: np.ndarray
    updater: Callable[[np.ndarray, int, int], np.ndarray]
    evaluator: Optional[Callable[[np.ndarray], Tuple[float, float]]] = None
    note: Optional[str] = None


def select_faulty_clients(
    n_clients: int, explicit: Sequence[int], ratio: float, seed: int
) -> List[int]:
    base = {int(uid) for uid in explicit if 1 <= int(uid) <= n_clients}
    ratio = float(np.clip(ratio, 0.0, 1.0))
    if ratio > 0.0:
        rng = np.random.default_rng(seed + 54321)
        target = max(1, int(math.ceil(ratio * n_clients)))
        if len(base) < target:
            remaining = [uid for uid in range(1, n_clients + 1) if uid not in base]
            if remaining:
                need = min(len(remaining), target - len(base))
                sampled = rng.choice(remaining, size=need, replace=False)
                base.update(int(uid) for uid in sampled.tolist())
    return sorted(base)


def make_faulty_hook(
    faulty_clients: Iterable[int], mode: str, std: float, seed: int
) -> Callable[[np.ndarray, np.ndarray, int], np.ndarray]:
    faulty_set = set(int(uid) for uid in faulty_clients)
    mode = (mode or "none").lower()
    rngs = {uid: np.random.default_rng(seed + 12345 + uid) for uid in faulty_set}

    def apply(x_prev: np.ndarray, candidate: np.ndarray, uid: int) -> np.ndarray:
        if uid not in faulty_set:
            return candidate
        if mode == "sign_flip":
            return x_prev - (candidate - x_prev)
        if mode == "gaussian":
            noise = rngs[uid].normal(loc=0.0, scale=float(std), size=x_prev.shape)
            return x_prev + noise
        return candidate

    return apply


def make_torch_updater(
    *,
    model_cls: Callable[[], nn.Module],
    loaders: Dict[int, DataLoader],
    device: str,
    lr: float,
    local_epochs: int,
    faulty_hook: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    faulty_label_clients: Iterable[int],
    weight_decay: float = 0.0,
) -> Callable[[np.ndarray, int, int], np.ndarray]:
    faulty_labels = set(int(uid) for uid in faulty_label_clients)
    criterion = nn.CrossEntropyLoss()

    def run_local_training(x_prev_vec: np.ndarray, uid: int, _: int) -> np.ndarray:
        model = model_cls().to(device)
        torch_vector_to_parameters(model, x_prev_vec)
        model.train()
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
        loader = loaders[uid]
        for _ in range(local_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                if uid in faulty_labels:
                    yb = torch.randint(low=0, high=10, size=yb.shape, device=yb.device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
        vec = torch_parameters_to_vector(model)
        return faulty_hook(x_prev_vec, vec, uid)

    return run_local_training


def make_grad_fn_factory(seed: int, lr: float = 0.5) -> Callable[[np.ndarray, int, int], np.ndarray]:
    rng = np.random.default_rng(seed)

    def make_update(x_prev: np.ndarray, uid: int, _: int) -> np.ndarray:
        mu = rng.normal(loc=uid * 0.05, scale=0.5, size=x_prev.size)
        grad = (x_prev - mu)
        noise = rng.normal(0.0, 0.01, size=x_prev.size)
        return x_prev - lr * grad + noise

    return make_update


def prepare_dataset(
    dataset: str,
    *,
    n_clients: int,
    batch: int,
    data_root: str,
    lr: float,
    local_epochs: int,
    faulty_hook: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    faulty_mode: str,
    faulty_clients: Iterable[int],
    low_quality_family: Optional[str],
    low_quality_severity: float,
    samples_per_client: Optional[int],
    seed: int,
    toy_dim: int,
) -> DatasetSetup:
    dataset = dataset.lower()
    if dataset == "mnist":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_cls = SimpleCNN
        model = model_cls().to(device)
        init_vec = torch_parameters_to_vector(model)
        loaders, test_loader = build_mnist_loaders(n_clients, batch, data_root)
        updater = make_torch_updater(
            model_cls=model_cls,
            loaders=loaders,
            device=device,
            lr=lr,
            local_epochs=local_epochs,
            faulty_hook=faulty_hook,
            faulty_label_clients=faulty_clients if faulty_mode == "label_noise" else [],
        )
        evaluator = lambda vec: evaluate_model_from_vector(model_cls, vec, test_loader, device)
        return DatasetSetup(init_vector=init_vec, updater=updater, evaluator=evaluator)

    if dataset == "cifar10":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_cls = SimpleCIFAR10CNN
        model = model_cls().to(device)
        init_vec = torch_parameters_to_vector(model)
        policies = None
        if faulty_clients and low_quality_family:
            base_policy = LowQualityPolicy(
                family=low_quality_family,
                severity=low_quality_severity,
                num_classes=10,
                seed=seed,
            )
            policies = {uid: base_policy.spawn(seed + 789 * uid) for uid in faulty_clients}
        loaders, test_loader = build_cifar10_loaders(
            n_clients,
            batch,
            data_root=data_root,
            seed=seed,
            samples_per_client=samples_per_client,
            low_quality_policies=policies,
        )
        updater = make_torch_updater(
            model_cls=model_cls,
            loaders=loaders,
            device=device,
            lr=lr,
            local_epochs=local_epochs,
            faulty_hook=faulty_hook,
            faulty_label_clients=faulty_clients if faulty_mode == "label_noise" else [],
            weight_decay=5e-4,
        )
        evaluator = lambda vec: evaluate_model_from_vector(model_cls, vec, test_loader, device)
        note = "Local loaders use torchvision; training runs sequentially."
        return DatasetSetup(init_vector=init_vec, updater=updater, evaluator=evaluator, note=note)

    # toy synthetic gradients
    init_vec = np.zeros(toy_dim, dtype=float)
    base_update = make_grad_fn_factory(seed=seed, lr=lr)

    def updater(x_prev: np.ndarray, uid: int, round_t: int) -> np.ndarray:
        vec = base_update(x_prev, uid, round_t)
        return faulty_hook(x_prev, vec, uid)

    return DatasetSetup(init_vector=init_vec, updater=updater)


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------


def simulate(
    *,
    rounds: int = 3,
    n_clients: int = 5,
    dim: int = 8,
    t_thr: int = 3,
    seed: int = 0,
    dataset: str = "toy",
    lr: float = 0.1,
    local_epochs: int = 1,
    batch: int = 256,
    b_mode: str = "per_round",
    method: str = "ours",
    faulty_uids: Sequence[int] | None = None,
    faulty_mode: str = "none",
    faulty_std: float = 0.5,
    faulty_ratio: float = 0.0,
    offline_uid: Optional[int] = None,
    offline_from_round: int = 10**9,
    data_root: str = "./data",
    low_quality_family: Optional[str] = None,
    low_quality_severity: float = 0.3,
    cifar_samples_per_client: Optional[int] = None,
    log_weights: bool = False,
    log_local_eval: bool = False,
    verbose: bool = True,
) -> Dict[str, object]:
    assert 1 <= t_thr <= n_clients, "threshold t must satisfy 1 <= t <= n_clients"

    method_key = (method or "ours").lower()
    dataset_key = (dataset or "toy").lower()
    weight_rule_map = {"ours": "inverse_l2", "esfl": "uniform", "ppfdl": "ppfdl"}
    os.environ["WEIGHT_RULE"] = weight_rule_map.get(method_key, method_key)

    faulty_uids = faulty_uids or []
    faulty_clients = select_faulty_clients(n_clients, faulty_uids, faulty_ratio, seed)
    faulty_hook = make_faulty_hook(faulty_clients, faulty_mode, faulty_std, seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    setup = prepare_dataset(
        dataset_key,
        n_clients=n_clients,
        batch=batch,
        data_root=data_root,
        lr=lr,
        local_epochs=local_epochs,
        faulty_hook=faulty_hook,
        faulty_mode=faulty_mode,
        faulty_clients=faulty_clients,
        low_quality_family=low_quality_family,
        low_quality_severity=low_quality_severity,
        samples_per_client=cifar_samples_per_client,
        seed=seed,
        toy_dim=dim,
    )

    ff = FixedField()
    ta = TA.initialize(ff, setup.init_vector.size, b_mode=b_mode)
    clients = [Client(uid=i, ff=ff, m=setup.init_vector.size, ta=ta) for i in range(1, n_clients + 1)]

    metrics: List[Dict[str, float]] = []
    final_test_loss: Optional[float] = None
    final_test_acc: Optional[float] = None
    x_prev = setup.init_vector.astype(float).copy()

    for round_t in range(1, rounds + 1):
        if verbose:
            print(f"\n=== Round {round_t} ===")
            if round_t == 1 and faulty_clients:
                extra = f", std={faulty_std}" if faulty_mode == "gaussian" else ""
                ratio_info = f", ratio~{faulty_ratio:.3f}" if faulty_ratio > 0 else ""
                print(f"  faulty clients={faulty_clients}, mode={faulty_mode}{extra}{ratio_info}")
            if round_t == 1 and setup.note:
                print(f"  note: {setup.note}")

        online_clients = [c for c in clients if not (offline_uid and c.uid == offline_uid and round_t >= offline_from_round)]
        if len(online_clients) < t_thr:
            raise RuntimeError(
                f"online clients {len(online_clients)} < threshold t={t_thr}; decrease --thr or adjust offline settings."
            )

        server = Server(ff=ff, m=setup.init_vector.size, ta=ta, t_thr=t_thr, round_t=round_t)
        reg_msgs = [dict(uid=c.uid, pk=c.pk) for c in online_clients]
        server.register_users(reg_msgs)
        peer_pubkeys = {c.uid: c.pk for c in online_clients}

        user_msgs: List[Dict[str, object]] = []
        for client in online_clients:
            msg = client.local_train_and_package(
                round_t=round_t,
                x_prev_float=x_prev,
                make_grad_fn=setup.updater,
                n_online=len(online_clients),
                t_thr=t_thr,
                peer_pubkeys=peer_pubkeys,
            )
            user_msgs.append(msg)

        inbox = server.distribute_peer_shares(user_msgs)
        sum_R1_points: List[Tuple[int, np.ndarray]] = []
        sum_R2_points: List[Tuple[int, int]] = []
        for client in online_clients:
            R1_j, R2_j = client.decrypt_sum_share(
                round_t=round_t, from_users=inbox.get(client.uid, {}), peer_pubkeys=peer_pubkeys
            )
            sum_R1_points.append((client.uid, R1_j))
            sum_R2_points.append((client.uid, R2_j))

        agg_msg = server.aggregate_and_prove(user_msgs, sum_R1_points, sum_R2_points)
        if ta.b_mode == "per_round":
            b1_mean, b2_mean = ta.derive_b1_b2(round_t, uid=0)
        else:
            b1_sum = np.zeros(setup.init_vector.size, dtype=object)
            b2_sum = 0
            for client in online_clients:
                b1_i, b2_i = ta.derive_b1_b2(round_t, uid=client.uid)
                b1_sum = (b1_sum + b1_i) % ff.q
                b2_sum = (b2_sum + b2_i) % ff.q
            count = len(online_clients)
            b1_mean = (b1_sum * pow(count, -1, ff.q)) % ff.q
            b2_mean = (b2_sum * pow(count, -1, ff.q)) % ff.q

        ok, x_next, diag = client_side_verify_and_update(ff, ta, agg_msg, b1_mean, b2_mean)
        if not ok:
            raise RuntimeError("verification failed: aggregated labels do not match")

        if verbose:
            delta = x_next - x_prev
            print(
                f"online={len(online_clients)} / total={n_clients}, dim={setup.init_vector.size}, thr={t_thr}"
            )
            print(
                "  ||Δx||2={:.3e}, x_prev[:4]={}, x_next[:4]={}".format(
                    np.linalg.norm(delta), np.round(x_prev[:4], 6), np.round(x_next[:4], 6)
                )
            )
        if log_weights and verbose:
            weights = [getattr(client, "_last_weight", float("nan")) for client in online_clients]
            weight_array = np.array(weights, dtype=float)
            mask = np.isfinite(weight_array)
            if mask.any():
                finite = weight_array[mask]
                print(
                    "  weight stats: mean={:.4e}, min={:.4e}, max={:.4e}, sum={:.4e}".format(
                        np.mean(finite), np.min(finite), np.max(finite), np.sum(finite)
                    )
                )
            else:
                print("  weight stats: unavailable (all NaN)")
            print(
                "  aggregate diag: weight_sum={:.4e}, numerator_l2={:.4e}, numerator_mean={:.4e}".format(
                    diag["weight_sum"], diag["numerator_l2"], diag["numerator_mean"]
                )
            )

        if setup.evaluator is not None:
            test_loss, test_acc = setup.evaluator(x_next)
            final_test_loss, final_test_acc = test_loss, test_acc
            metrics.append({"round": round_t, "test_loss": test_loss, "test_acc": test_acc})
            if verbose:
                print(f"  test: loss={test_loss:.4f}, acc={test_acc * 100:.2f}%")
            if log_local_eval and verbose:
                for client in online_clients:
                    local_vec = getattr(client, "_last_local_vec", None)
                    if local_vec is None:
                        continue
                    loss_i, acc_i = setup.evaluator(local_vec)
                    weight_i = getattr(client, "_last_weight", float("nan"))
                    print(
                        f"    client {client.uid}: w={weight_i:.4e}, local loss={loss_i:.4f}, local acc={acc_i * 100:.2f}%"
                    )
        else:
            norm = float(np.linalg.norm(x_next))
            metrics.append({"round": round_t, "l2_norm": norm})
            if verbose:
                print(f"  toy: ||x||2={norm:.3e}")

        x_prev = x_next

    if verbose:
        preview = np.round(x_prev[: min(8, x_prev.size)], 6)
        print("\nTraining finished. x_final[:8] =", preview)

    return {
        "final_vector": x_prev.copy(),
        "round_metrics": metrics,
        "final_test_loss": final_test_loss,
        "final_test_acc": final_test_acc,
        "faulty_clients": faulty_clients,
        "method": method_key,
        "dataset": dataset_key,
    }


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated learning simulator with verifiable secure aggregation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--clients", type=int, default=5, help="number of clients")
    parser.add_argument("--dim", type=int, default=8, help="model dimension for toy mode")
    parser.add_argument("--thr", type=int, default=3, help="Shamir threshold t")
    parser.add_argument("--rounds", type=int, default=3, help="training rounds")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--dataset",
        type=str,
        default="toy",
        choices=["toy", "mnist", "cifar10"],
        help="dataset / simulation mode",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="local learning rate")
    parser.add_argument("--local_epochs", type=int, default=1, help="local epochs per round")
    parser.add_argument("--batch", type=int, default=256, help="local batch size for torch datasets")
    parser.add_argument(
        "--b_mode",
        type=str,
        default="per_round",
        choices=["per_round", "per_user"],
        help="mask derivation mode",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ours",
        choices=["ours", "esfl", "ppfdl", "uniform", "inverse_l2", "chi2_rule"],
        help="aggregation weighting rule",
    )
    parser.add_argument("--faulty_uids", type=str, default="", help="comma separated unreliable client ids")
    parser.add_argument(
        "--faulty_mode",
        type=str,
        default="none",
        choices=["none", "sign_flip", "gaussian", "label_noise"],
        help="faulty update behaviour",
    )
    parser.add_argument("--faulty_std", type=float, default=0.5, help="gaussian noise std when faulty_mode=gaussian")
    parser.add_argument("--faulty_ratio", type=float, default=0.0, help="fraction of clients to mark as faulty")
    parser.add_argument("--offline_uid", type=int, default=None, help="client id that becomes offline")
    parser.add_argument(
        "--offline_from_round", type=int, default=10**9, help="round when the offline client disconnects"
    )
    parser.add_argument("--data_root", type=str, default="./data", help="dataset cache directory")
    parser.add_argument(
        "--samples_per_client", type=int, default=None, help="cap CIFAR-10 samples per client"
    )
    parser.add_argument(
        "--low_quality_family",
        type=str,
        default="symmetric",
        choices=["symmetric", "class_dependent", "robust", "erasing", "symmetric_noise"],
        help="low-quality data family for CIFAR-10",
    )
    parser.add_argument(
        "--low_quality_severity", type=float, default=0.3, help="severity for low-quality data manipulation"
    )
    parser.add_argument("--log_weights", action="store_true", help="print per-round weight statistics")
    parser.add_argument(
        "--log_local_eval",
        action="store_true",
        help="evaluate each client's local model when datasets provide evaluators",
    )
    parser.add_argument("--silent", action="store_true", help="suppress verbose logging")
    parser.add_argument("--save_metrics", type=str, default=None, help="optional JSON path for metrics")
    return parser.parse_args(argv)


def parse_faulty_uids(text: str) -> List[int]:
    tokens = [tok.strip() for tok in text.split(",") if tok.strip()]
    return [int(tok) for tok in tokens]


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    faulty_uids = parse_faulty_uids(args.faulty_uids) if args.faulty_uids else []

    result = simulate(
        rounds=args.rounds,
        n_clients=args.clients,
        dim=args.dim,
        t_thr=args.thr,
        seed=args.seed,
        dataset=args.dataset,
        lr=args.lr,
        local_epochs=args.local_epochs,
        batch=args.batch,
        b_mode=args.b_mode,
        method=args.method,
        faulty_uids=faulty_uids,
        faulty_mode=args.faulty_mode,
        faulty_std=args.faulty_std,
        faulty_ratio=args.faulty_ratio,
        offline_uid=args.offline_uid,
        offline_from_round=args.offline_from_round,
        data_root=args.data_root,
        low_quality_family=args.low_quality_family,
        low_quality_severity=args.low_quality_severity,
        cifar_samples_per_client=args.samples_per_client,
        log_weights=args.log_weights,
        log_local_eval=args.log_local_eval,
        verbose=not args.silent,
    )

    if args.save_metrics:
        path = Path(args.save_metrics)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(result)
        final_vec = payload.get("final_vector")
        if isinstance(final_vec, np.ndarray):
            payload["final_vector"] = final_vec.astype(float).tolist()
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        if not args.silent:
            print(f"metrics saved to {path}")


if __name__ == "__main__":
    main()
