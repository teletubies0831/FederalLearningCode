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
    """Coordinate secure aggregation, dropout handling, and truth discovery."""

    def __init__(self, num_clients: int, dropout_tolerance: int, max_truth_iters: int = 5) -> None:
        if dropout_tolerance < 0:
            raise ValueError("dropout_tolerance must be non-negative")
        if dropout_tolerance >= num_clients:
            raise ValueError("dropout_tolerance must be smaller than the number of clients")
        self.num_clients = num_clients
        self.dropout_tolerance = dropout_tolerance
        self.max_truth_iters = max_truth_iters

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

        stacked = [
            torch.cat([tensor.flatten() for tensor in delta.values()]) for delta in deltas
        ]
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
