# -*- coding: utf-8 -*-
"""
质量加权（真值发现范式）
- 论文式(1)描述“权重与距离的反比”，式(6)给出一种 χ^2 基准的具体化。
- 这里提供两种可选：'inverse_l2' 与 'chi2_rule'，默认 'inverse_l2' 更稳健。
"""
from __future__ import annotations
import os
import numpy as np

def weight_inverse_l2(x_i: np.ndarray, x_prev: np.ndarray, C: float=1.0, eps: float=1e-8) -> float:
    """
    w_i = C / ( ||x_i - x_prev||^2 + eps )
    - 与式(1)“C / F(·)”一致；eps 防止除零。
    """
    # 可选：通过环境变量切换为均匀权重
    if os.environ.get("WEIGHT_RULE", "inverse_l2").lower() == "uniform":
        return 1.0
    dist2 = float(np.sum((x_i - x_prev)**2))
    return C / (dist2 + eps)

def weight_chi2_rule(x_i: np.ndarray, x_prev: np.ndarray, alpha: float=0.05) -> float:
    """
    论文式(6)的 χ^2 思路：用阈值/分位函数把距离映射到权重（实现上用简单单调映射近似）
    - 简化：w_i = Q / (||x_i - x_prev||^2 + eps)，其中 Q≈chi2_{1-alpha/2, m}
    """
    m = x_i.size
    # 正态近似分位：Q ≈ m + z*sqrt(2m) （z≈1.96 对 95%）
    z = 1.959963984540054
    Q = m + z * (2*m)**0.5
    dist2 = float(np.sum((x_i - x_prev)**2))
    return Q / (dist2 + 1e-8)
