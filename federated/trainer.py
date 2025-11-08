"""Federated training with configurable aggregation strategies."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .aggregation import ClientReport, build_aggregator


@dataclass
class TrainingHistory:
    """Stores metrics collected during federated training."""

    round: int
    test_loss: float
    test_accuracy: float


class FederatedTrainer:
    """Secure federated learning trainer that follows the thesis protocol."""

    def __init__(
        self,
        model_builder: Callable[[], nn.Module],
        client_loaders: List[DataLoader],
        test_loader: DataLoader,
        device: torch.device,
        lr: float,
        local_epochs: int,
        dropout_tolerance: int,
        weight_decay: float = 0.0,
        aggregator: str = "truth_discovery",
    ) -> None:
        self.model_builder = model_builder
        self.client_loaders = client_loaders
        self.test_loader = test_loader
        self.device = device
        self.lr = lr
        self.local_epochs = local_epochs
        self.dropout_tolerance = dropout_tolerance
        self.weight_decay = weight_decay
        self.aggregator_name = aggregator

    def train(
        self,
        num_rounds: int,
        dropout_rate: float,
    ) -> Tuple[nn.Module, List[TrainingHistory]]:
        """Run the federated training loop using secure aggregation."""

        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0.0, 1.0)")

        global_model = self.model_builder().to(self.device)
        history: List[TrainingHistory] = []
        num_clients = len(self.client_loaders)
        aggregator, requires_commitment = build_aggregator(
            self.aggregator_name,
            num_clients=num_clients,
            dropout_tolerance=self.dropout_tolerance,
        )

        for round_idx in range(1, num_rounds + 1):
            global_state = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
            client_reports = list(
                self._collect_client_reports(
                    round_idx=round_idx,
                    global_state=global_state,
                    dropout_rate=dropout_rate,
                    require_commitment=requires_commitment,
                )
            )

            aggregated_state = aggregator.aggregate(global_state, client_reports)
            global_model.load_state_dict(aggregated_state)

            test_loss, test_acc = self.evaluate(global_model)
            history.append(TrainingHistory(round=round_idx, test_loss=test_loss, test_accuracy=test_acc))
            active_clients = sorted(report.client_id for report in client_reports)
            print(
                f"Round {round_idx:03d}/{num_rounds:03d} | Active clients: {active_clients} | "
                f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}"
            )

        return global_model, history

    def _collect_client_reports(
        self,
        round_idx: int,
        global_state: Dict[str, torch.Tensor],
        dropout_rate: float,
        require_commitment: bool,
    ) -> Iterable[ClientReport]:
        """Train every client sequentially and emit secure reports."""

        generator = torch.Generator().manual_seed(round_idx)
        dropout_mask = torch.rand(len(self.client_loaders), generator=generator)

        # 先根据 ``dropout_rate`` 预选掉线候选客户端，再根据 ``dropout_tolerance``
        # 控制实际掉线数量不超过协议允许的范围。这样可以解释用户在问题中遇到的现象：
        # 当随机掉线客户端数量超过 ``dropout_tolerance`` 时，之前的实现会直接报错。
        tentative_dropouts = [
            client_id
            for client_id, mask_prob in enumerate(dropout_mask.tolist())
            if mask_prob < dropout_rate
        ]
        if len(tentative_dropouts) > self.dropout_tolerance:
            # 使用与 ``dropout_mask`` 相同的随机源，保证实验可复现。
            selection_order = torch.randperm(len(tentative_dropouts), generator=generator).tolist()
            allowed_dropouts = {
                tentative_dropouts[idx] for idx in selection_order[: self.dropout_tolerance]
            }
        else:
            allowed_dropouts = set(tentative_dropouts)

        for client_id, loader in enumerate(self.client_loaders):
            if client_id in allowed_dropouts:
                continue

            client_model = self.model_builder().to(self.device)
            client_model.load_state_dict({k: v.clone().to(self.device) for k, v in global_state.items()})
            training_loss = self._train_single_client(client_model, loader)
            client_state = {k: v.detach().cpu() for k, v in client_model.state_dict().items()}
            commitment = self._commit_state(client_state) if require_commitment else None
            yield ClientReport(
                client_id=client_id,
                updated_state=client_state,
                training_loss=training_loss,
                commitment=commitment,
            )

    def _train_single_client(self, model: nn.Module, loader: DataLoader) -> float:
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        cumulative_loss = 0.0
        num_batches = 0

        for _ in range(self.local_epochs):
            for images, targets in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                logits = model(images)
                loss = loss_fn(logits, targets)
                loss.backward()
                optimizer.step()
                cumulative_loss += loss.item()
                num_batches += 1

        return cumulative_loss / max(num_batches, 1)

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

    def _commit_state(self, state: Dict[str, torch.Tensor]) -> str:
        """Compute a SHA256 commitment for ``state``."""

        hasher = hashlib.sha256()
        for key in sorted(state.keys()):
            tensor_bytes = state[key].detach().cpu().contiguous().numpy().tobytes()
            hasher.update(key.encode("utf-8"))
            hasher.update(tensor_bytes)
        return hasher.hexdigest()
