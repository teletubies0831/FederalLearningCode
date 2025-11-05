# -*- coding: utf-8 -*-
"""Truth discovery style client weighting rules."""
from __future__ import annotations

import os
from typing import Callable, Dict

import numpy as np


def _inverse_l2_weight(x_i: np.ndarray, x_prev: np.ndarray, C: float = 1.0, eps: float = 1e-8) -> float:
    """Inverse squared distance weighting (default in the paper)."""
    dist2 = float(np.sum((x_i - x_prev) ** 2))
    return C / (dist2 + eps)


def _uniform_weight(*_: np.ndarray) -> float:
    """Baseline FedAvg style weighting (ESFL)."""

    return 1.0


def _chi2_weight(x_i: np.ndarray, x_prev: np.ndarray, alpha: float = 0.05) -> float:
    """Simplified χ^2 mapping used for ablation studies in the original paper."""

    m = x_i.size
    z = 1.959963984540054  # 95% quantile approximation
    Q = m + z * (2 * m) ** 0.5
    dist2 = float(np.sum((x_i - x_prev) ** 2))
    return Q / (dist2 + 1e-8)


def _ppfdl_weight(x_i: np.ndarray, x_prev: np.ndarray, tau: float = 5.0, eps: float = 1e-8) -> float:
    """
    A smooth clipping rule inspired by PPFDL: the further a client drifts from the
    previous global model, the more its contribution is attenuated but never fully
    discarded. ``tau`` controls how aggressively unreliable updates are suppressed.
    """

    dist = float(np.linalg.norm(x_i - x_prev))
    scaled = dist / max(tau, eps)
    return 1.0 / (1.0 + scaled)


_DISPATCH: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "inverse_l2": _inverse_l2_weight,
    "uniform": _uniform_weight,
    "esfl": _uniform_weight,
    "chi2_rule": _chi2_weight,
    "ppfdl": _ppfdl_weight,
}


def compute_weight(x_i: np.ndarray, x_prev: np.ndarray) -> float:
    """Select and evaluate the weighting rule based on ``WEIGHT_RULE`` env var."""

    rule = os.environ.get("WEIGHT_RULE", "inverse_l2").lower()
    func = _DISPATCH.get(rule, _inverse_l2_weight)
    return float(func(x_i, x_prev))


def weight_inverse_l2(x_i: np.ndarray, x_prev: np.ndarray, C: float = 1.0, eps: float = 1e-8) -> float:
    """Backward compatibility shim for existing imports."""

    return _inverse_l2_weight(x_i, x_prev, C=C, eps=eps)


def weight_chi2_rule(x_i: np.ndarray, x_prev: np.ndarray, alpha: float = 0.05) -> float:
    """Backward compatibility shim returning χ^2-style weights."""

    return _chi2_weight(x_i, x_prev, alpha=alpha)
