"""Secure aggregation primitives closely matching the thesis protocol."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import hashlib

import torch


@dataclass
class ClientReport:
    """Client-side artifact produced after local optimisation."""

    client_id: int
    updated_state: Dict[str, torch.Tensor]
    training_loss: float
    commitment: str


class SecureAggregationController:
    """Coordinate secure aggregation, dropout handling, and truth discovery variants."""

    def __init__(
        self,
        num_clients: int,
        dropout_tolerance: int,
        max_truth_iters: int = 5,
        truth_strategy: str | None = "iterative",
        truth_alpha: float = 0.05,
        truth_scaling: float = 1.0,
        variance_floor: float = 1e-9,
    ) -> None:
        if dropout_tolerance < 0:
            raise ValueError("dropout_tolerance must be non-negative")
        if dropout_tolerance >= num_clients:
            raise ValueError("dropout_tolerance must be smaller than the number of clients")
        self.num_clients = num_clients
        self.dropout_tolerance = dropout_tolerance
        self.max_truth_iters = max_truth_iters
        strategy = (truth_strategy or "iterative").lower()
        valid_strategies = {"iterative", "uniform", "fedavg", "esfl", "ppfdl"}
        if strategy not in valid_strategies:
            raise ValueError(
                "truth_strategy must be one of 'iterative', 'uniform', 'fedavg', 'esfl', or 'ppfdl'"
            )
        self.truth_strategy = strategy
        self.truth_alpha = float(truth_alpha)
        if not 0.0 < self.truth_alpha < 1.0:
            raise ValueError("truth_alpha must be between 0 and 1")
        self.truth_scaling = float(truth_scaling)
        if self.truth_scaling <= 0:
            raise ValueError("truth_scaling must be positive")
        self.variance_floor = float(variance_floor)
        if self.variance_floor <= 0:
            raise ValueError("variance_floor must be positive")

    def aggregate(
        self,
        global_state: Dict[str, torch.Tensor],
        client_reports: Iterable[ClientReport],
    ) -> Dict[str, torch.Tensor]:
        reports = list(client_reports)
        if not reports:
            raise RuntimeError("No available client reports for aggregation")

        if len(reports) < self.num_clients - self.dropout_tolerance:
            raise RuntimeError(
                "Not enough client updates to satisfy dropout tolerance: "
                f"expected at least {self.num_clients - self.dropout_tolerance}, got {len(reports)}"
            )

        self._verify_commitments(reports)

        floating_keys = [
            key for key, param in global_state.items() if torch.is_floating_point(param)
        ]

        if not floating_keys:
            # 模型中不存在需要聚合的浮点参数，直接返回任意一个客户端的更新即可。
            return {
                key: reports[0].updated_state[key].clone()
                for key in global_state.keys()
            }

        deltas = self._compute_deltas(global_state, reports, floating_keys)
        weights = self._compute_weights(deltas, reports)

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

        flattened_updates = [
            torch.cat([tensor.flatten().to(dtype=torch.float64) for tensor in delta.values()])
            for delta in deltas
        ]
        updates = torch.stack(flattened_updates)

        if updates.size(0) == 1:
            return [1.0]

        mean_update = updates.mean(dim=0)
        diffs = updates - mean_update
        variances = torch.var(updates, dim=0, unbiased=False)
        variances = torch.clamp(variances, min=self.variance_floor)
        normalised = diffs * diffs / variances
        mahalanobis = normalised.sum(dim=1)

        degrees_of_freedom = float(updates.size(1))
        chi2 = torch.distributions.chi2.Chi2(
            torch.tensor(degrees_of_freedom, dtype=updates.dtype)
        )
        critical_value = chi2.icdf(
            torch.tensor(1.0 - self.truth_alpha / 2.0, dtype=updates.dtype)
        )
        # Ensure numerical stability in extreme cases.
        critical_value = max(float(critical_value.item()), self.variance_floor)

        raw_weights = torch.clamp(critical_value - mahalanobis, min=0.0)
        scaled_weights = self.truth_scaling * raw_weights / (critical_value + 1e-12)

        if float(scaled_weights.sum()) <= 0.0:
            return [1.0 for _ in deltas]

        return scaled_weights.to(dtype=torch.float32).tolist()

    def _compute_weights(
        self,
        deltas: List[Dict[str, torch.Tensor]],
        reports: List[ClientReport],
    ) -> List[float]:
        if self.truth_strategy in {"uniform", "fedavg", "esfl"}:
            return [1.0 for _ in reports]
        if self.truth_strategy in {"iterative", "ppfdl"}:
            return self._truth_discovery_weights(deltas)
        raise RuntimeError(f"Unsupported truth discovery strategy: {self.truth_strategy}")

    def _commit(self, state: Dict[str, torch.Tensor]) -> str:
        hasher = hashlib.sha256()
        for key in sorted(state.keys()):
            hasher.update(key.encode("utf-8"))
            hasher.update(state[key].detach().cpu().contiguous().numpy().tobytes())
        return hasher.hexdigest()

