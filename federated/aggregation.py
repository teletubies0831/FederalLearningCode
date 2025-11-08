"""Aggregation strategies for different federated learning baselines."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch


@dataclass
class ClientReport:
    """Client-side artifact produced after local optimisation."""

    client_id: int
    updated_state: Dict[str, torch.Tensor]
    training_loss: float
    commitment: str | None = None


class FedAvgAggregator:
    """Coordinate aggregation with FedAvg-style averaging and dropout tolerance."""

    def __init__(self, num_clients: int, dropout_tolerance: int) -> None:
        if dropout_tolerance < 0:
            raise ValueError("dropout_tolerance must be non-negative")
        if dropout_tolerance >= num_clients:
            raise ValueError("dropout_tolerance must be smaller than the number of clients")
        self.num_clients = num_clients
        self.dropout_tolerance = dropout_tolerance

    def aggregate(
        self,
        global_state: Dict[str, torch.Tensor],
        client_reports: Iterable[ClientReport],
    ) -> Dict[str, torch.Tensor]:
        reports = list(client_reports)
        self._validate_reports(reports)

        aggregated_state: Dict[str, torch.Tensor] = {}
        for key, param in global_state.items():
            if torch.is_floating_point(param):
                stacked = torch.stack(
                    [report.updated_state[key].to(torch.float32) for report in reports]
                )
                mean_param = stacked.mean(dim=0).to(dtype=param.dtype)
                aggregated_state[key] = mean_param.clone()
            else:
                aggregated_state[key] = reports[0].updated_state[key].clone()

        return aggregated_state

    def _validate_reports(self, reports: List[ClientReport]) -> None:
        if not reports:
            raise RuntimeError("No available client reports for aggregation")

        if len(reports) < self.num_clients - self.dropout_tolerance:
            raise RuntimeError(
                "Not enough client updates to satisfy dropout tolerance: "
                f"expected at least {self.num_clients - self.dropout_tolerance}, got {len(reports)}"
            )


class PPFDLAggregator(FedAvgAggregator):
    """Weighted averaging that down-weights high-loss clients (PPFDL baseline)."""

    def aggregate(
        self,
        global_state: Dict[str, torch.Tensor],
        client_reports: Iterable[ClientReport],
    ) -> Dict[str, torch.Tensor]:
        reports = list(client_reports)
        self._validate_reports(reports)

        weights = [1.0 / (report.training_loss + 1e-8) for report in reports]
        normaliser = sum(weights)
        aggregated_state: Dict[str, torch.Tensor] = {}

        for key, param in global_state.items():
            if torch.is_floating_point(param):
                accumulator = torch.zeros_like(param, dtype=torch.float32)
                for weight, report in zip(weights, reports):
                    accumulator += weight * report.updated_state[key].to(torch.float32)
                averaged = (accumulator / normaliser).to(dtype=param.dtype)
                aggregated_state[key] = averaged.clone()
            else:
                aggregated_state[key] = reports[0].updated_state[key].clone()

        return aggregated_state


class TruthDiscoveryAggregator(FedAvgAggregator):
    """Secure aggregation with commitment checks and truth discovery weighting."""

    def __init__(
        self,
        num_clients: int,
        dropout_tolerance: int,
        max_truth_iters: int = 5,
    ) -> None:
        super().__init__(num_clients=num_clients, dropout_tolerance=dropout_tolerance)
        self.max_truth_iters = max_truth_iters

    def aggregate(
        self,
        global_state: Dict[str, torch.Tensor],
        client_reports: Iterable[ClientReport],
    ) -> Dict[str, torch.Tensor]:
        reports = list(client_reports)
        self._validate_reports(reports)
        self._verify_commitments(reports)

        floating_keys = [key for key, param in global_state.items() if torch.is_floating_point(param)]
        if not floating_keys:
            return {key: reports[0].updated_state[key].clone() for key in global_state.keys()}

        deltas = self._compute_deltas(global_state, reports, floating_keys)
        weights = self._truth_discovery_weights(deltas)

        aggregated_updates = {
            key: torch.zeros_like(global_state[key], dtype=torch.float32)
            for key in floating_keys
        }
        normaliser = sum(weights) + 1e-12

        for weight, delta in zip(weights, deltas):
            for key in floating_keys:
                aggregated_updates[key] += weight * delta[key]

        aggregated_state: Dict[str, torch.Tensor] = {}
        for key, base_param in global_state.items():
            if key in aggregated_updates:
                base = base_param.detach().to(dtype=torch.float32)
                update = aggregated_updates[key] / normaliser + base
                aggregated_state[key] = update.to(dtype=base_param.dtype)
            else:
                aggregated_state[key] = reports[0].updated_state[key].clone()

        return aggregated_state

    def _verify_commitments(self, reports: List[ClientReport]) -> None:
        seen = set()
        for report in reports:
            if not report.commitment:
                raise RuntimeError(
                    "TruthDiscoveryAggregator requires commitments on every client report"
                )
            if report.commitment in seen:
                raise RuntimeError(
                    "Commitment collision detected; secure aggregation aborted"
                )
            recalculated = self._commit(report.updated_state)
            if recalculated != report.commitment:
                raise RuntimeError(
                    f"Commitment mismatch for client {report.client_id}; integrity cannot be guaranteed"
                )
            seen.add(report.commitment)

    def _compute_deltas(
        self,
        global_state: Dict[str, torch.Tensor],
        reports: Iterable[ClientReport],
        floating_keys: Iterable[str],
    ) -> List[Dict[str, torch.Tensor]]:
        deltas: List[Dict[str, torch.Tensor]] = []
        float_keys = list(floating_keys)
        for report in reports:
            deltas.append(
                {
                    key: report.updated_state[key].to(torch.float32)
                    - global_state[key].to(torch.float32)
                    for key in float_keys
                }
            )
        return deltas

    def _truth_discovery_weights(
        self,
        deltas: List[Dict[str, torch.Tensor]],
    ) -> List[float]:
        if not deltas:
            return []

        stacked = [torch.cat([tensor.flatten() for tensor in delta.values()]) for delta in deltas]
        stacked_tensor = torch.stack(stacked)

        reliability = torch.ones(len(deltas))
        consensus = torch.zeros_like(stacked_tensor[0])

        for _ in range(self.max_truth_iters):
            weighted_sum = torch.sum(reliability[:, None] * stacked_tensor, dim=0)
            consensus = weighted_sum / (reliability.sum() + 1e-12)
            distances = torch.norm(stacked_tensor - consensus, dim=1) + 1e-6
            reliability = 1.0 / distances

        return reliability.tolist()

    def _commit(self, state: Dict[str, torch.Tensor]) -> str:
        hasher = hashlib.sha256()
        for key in sorted(state.keys()):
            hasher.update(key.encode("utf-8"))
            hasher.update(state[key].detach().cpu().contiguous().numpy().tobytes())
        return hasher.hexdigest()


def build_aggregator(
    name: str,
    num_clients: int,
    dropout_tolerance: int,
) -> Tuple[FedAvgAggregator, bool]:
    """Construct an aggregation strategy and whether commitments are required."""

    canonical = name.lower()
    if canonical in {"truth_discovery", "ours"}:
        return TruthDiscoveryAggregator(num_clients, dropout_tolerance), True
    if canonical in {"esfl", "fedavg"}:
        return FedAvgAggregator(num_clients, dropout_tolerance), False
    if canonical == "ppfdl":
        return PPFDLAggregator(num_clients, dropout_tolerance), False
    supported = ["truth_discovery", "ours", "esfl", "fedavg", "ppfdl"]
    raise ValueError(
        f"Unsupported aggregation strategy '{name}'. Supported strategies: {', '.join(supported)}"
    )


__all__ = [
    "ClientReport",
    "FedAvgAggregator",
    "PPFDLAggregator",
    "TruthDiscoveryAggregator",
    "build_aggregator",
]
