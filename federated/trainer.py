"""Federated averaging training loop with detailed logging."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainingHistory:
    """Stores metrics collected during federated training."""

    round: int
    test_loss: float
    test_accuracy: float


class FedAvgTrainer:
    """Lightweight implementation of the FedAvg algorithm."""

    def __init__(
        self,
        model_builder: Callable[[], nn.Module],
        client_loaders: List[DataLoader],
        test_loader: DataLoader,
        device: torch.device,
        lr: float,
        local_epochs: int,
        weight_decay: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.model_builder = model_builder
        self.client_loaders = client_loaders
        self.test_loader = test_loader
        self.device = device
        self.lr = lr
        self.local_epochs = local_epochs
        self.weight_decay = weight_decay
        self.rng = random.Random(seed)

    def train(self, num_rounds: int, clients_per_round: int) -> Tuple[nn.Module, List[TrainingHistory]]:
        """Run the federated training loop."""

        global_model = self.model_builder().to(self.device)
        history: List[TrainingHistory] = []

        for round_idx in range(1, num_rounds + 1):
            # 在每一轮随机抽取部分客户端参与训练
            selected_indices = self._sample_clients(clients_per_round)
            # 聚合客户端更新，返回新的全局模型参数
            aggregated_state = self._aggregate_client_updates(global_model, selected_indices)
            global_model.load_state_dict(aggregated_state)

            test_loss, test_acc = self.evaluate(global_model)
            history.append(TrainingHistory(round=round_idx, test_loss=test_loss, test_accuracy=test_acc))
            print(
                f"Round {round_idx:03d}/{num_rounds:03d} | Selected clients: {selected_indices} | "
                f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}"
            )

        return global_model, history

    def _sample_clients(self, clients_per_round: int) -> List[int]:
        if clients_per_round <= 0 or clients_per_round > len(self.client_loaders):
            raise ValueError("clients_per_round must be in [1, num_clients]")
        return self.rng.sample(range(len(self.client_loaders)), clients_per_round)

    def _aggregate_client_updates(self, global_model: nn.Module, client_indices: Iterable[int]) -> Dict[str, torch.Tensor]:
        """Train each sampled client and average their resulting models."""

        global_state = global_model.state_dict()
        # 以全局模型参数形状初始化累积容器
        aggregated_state = {key: torch.zeros_like(param) for key, param in global_state.items()}
        num_clients = 0

        for client_idx in client_indices:
            client_model = self.model_builder().to(self.device)
            client_model.load_state_dict(global_state)
            self._train_single_client(client_model, self.client_loaders[client_idx])
            client_state = {k: v.detach().cpu() for k, v in client_model.state_dict().items()}
            for key, value in client_state.items():
                aggregated_state[key] += value
            num_clients += 1

        if num_clients == 0:
            raise RuntimeError("No clients were aggregated during this round")

        for key in aggregated_state:
            aggregated_state[key] /= float(num_clients)

        return aggregated_state

    def _train_single_client(self, model: nn.Module, loader: DataLoader) -> None:
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(self.local_epochs):
            for images, targets in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                logits = model(images)
                loss = loss_fn(logits, targets)
                loss.backward()
                optimizer.step()

    def evaluate(self, model: nn.Module) -> Tuple[float, float]:
        """Evaluate ``model`` on the shared test set."""

        model.eval()
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, targets in self.test_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                logits = model(images)
                loss = loss_fn(logits, targets)
                total_loss += loss.item() * images.size(0)
                predictions = logits.argmax(dim=1)
                total_correct += (predictions == targets).sum().item()
                total_samples += images.size(0)

        mean_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return mean_loss, accuracy
