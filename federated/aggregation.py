"""Secure aggregation primitives closely matching the thesis protocol."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import hashlib

import torch
import math

try:  # Optional SciPy for accurate chi-square quantiles
    import scipy.stats as _scipy_stats  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # SciPy not installed
    _HAVE_SCIPY = False


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
        self._prev_update_vector: torch.Tensor | None = None

    def aggregate(
        self,
        global_state: Dict[str, torch.Tensor],
        client_reports: Iterable[ClientReport],
    ) -> Tuple[Dict[str, torch.Tensor], List[float]]:
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
            aggregated = {
                key: reports[0].updated_state[key].clone()
                for key in global_state.keys()
            }
            weights = [1.0 for _ in reports]
            return aggregated, weights

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
        aggregated_delta_vector: List[torch.Tensor] = []
        for key, base_param in global_state.items():
            if key in aggregated_updates:
                base = base_param.detach().to(dtype=torch.float32)
                delta = aggregated_updates[key] / normaliser
                update = delta + base
                aggregated_state[key] = update.to(dtype=base_param.dtype)
                aggregated_delta_vector.append(delta.flatten().to(torch.float64))
            else:
                aggregated_state[key] = reports[0].updated_state[key].clone()

        if aggregated_delta_vector:
            self._prev_update_vector = torch.cat(aggregated_delta_vector)

        return aggregated_state, weights

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

    def _truth_discovery_weights(self, deltas):
        if not deltas:
            return []

        updates = torch.stack([
            torch.cat([t.flatten().to(torch.float64) for t in delta.values()])
            for delta in deltas
        ])  # [n, m]
        if updates.size(0) == 1:
            return [1.0]

        m = float(updates.size(1))
        if _HAVE_SCIPY:
            chi_quantile = float(_scipy_stats.chi2.ppf(1.0 - self.truth_alpha / 2.0, df=m))
        else:
            z = torch.distributions.Normal(0.0, 1.0)
            z_high = z.icdf(torch.tensor(1.0 - self.truth_alpha / 2.0, dtype=updates.dtype)).item()
            gamma = 2.0 / (9.0 * m)
            chi_quantile = m * (1.0 - gamma + z_high * math.sqrt(gamma)) ** 3

        scale = self.truth_scaling * max(chi_quantile, self.variance_floor)

        prev_truth = self._prev_update_vector
        if prev_truth is None or prev_truth.numel() != updates.size(1):
            truth = updates.mean(dim=0)
        else:
            truth = prev_truth.clone()

        weights = torch.ones(updates.size(0), dtype=torch.float32, device=updates.device)
        for _ in range(self.max_truth_iters):
            dist2 = (updates - truth).pow(2).sum(dim=1)
            norm2 = updates.pow(2).sum(dim=1)
            weights = (scale * norm2 / (dist2 + self.variance_floor)).to(torch.float32)

            if not torch.isfinite(weights).all():
                weights = torch.where(
                    torch.isfinite(weights), weights, torch.full_like(weights, 1.0)
                )

            weight_sum = weights.sum().item()
            if weight_sum <= 0.0:
                weights = torch.ones_like(weights)
                break

            updated_truth = (weights[:, None] * updates).sum(dim=0) / (weight_sum + 1e-12)
            if torch.allclose(updated_truth, truth, atol=1e-8, rtol=1e-5):
                truth = updated_truth
                break
            truth = updated_truth

        if float(weights.sum()) <= 0.0:
            return [1.0 for _ in deltas]
        return weights.tolist()

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

