# -*- coding: utf-8 -*-
"""
Shamir (t,n) 阈值秘密共享（支持标量与“逐元素向量”）
- 与论文 3.3 节、式(3)(4)一致；加性同态用于“和”的重构（式(5)）。
- 我们对“向量秘密”采取逐元素同一多项式次数的办法：每个坐标独立评估 f(k)。
"""
from __future__ import annotations
import secrets
import numpy as np
from typing import Dict, List, Tuple, Sequence
from field import FixedField

def _poly_eval_scalar(coeffs: List[int], x: int, ff: FixedField) -> int:
    """多项式 f(x)=c0 + c1 x + ... + c_{t-1} x^{t-1} 在点 x 的值（标量）"""
    res = 0
    xp = 1
    for c in coeffs:
        res = ff.add(res, ff.mul(c, xp))
        xp = ff.mul(xp, x)
    return res

def _poly_eval_vector(coeffs: List[np.ndarray], x: int, ff: FixedField) -> np.ndarray:
    """同上，但系数是向量：每个坐标独立计算"""
    res = np.zeros_like(coeffs[0], dtype=object)
    xp = 1
    for c in coeffs:
        res = (res + (c * xp) % ff.q) % ff.q
        xp = (xp * x) % ff.q
    return res

def share_scalar(secret: int, n: int, t: int, ff: FixedField) -> Dict[int, int]:
    """标量秘密 s → n 份（索引从 1..n），返回 {i: f(i)}"""
    coeffs = [secret] + [secrets.randbelow(ff.q) for _ in range(t - 1)]
    return {i: _poly_eval_scalar(coeffs, i, ff) for i in range(1, n + 1)}

def share_vector(secret_vec: np.ndarray, n: int, t: int, ff: FixedField) -> Dict[int, np.ndarray]:
    """向量秘密 r ∈ F^m → n 份（每个坐标同次多项式），返回 {i: f(i) (向量)}"""
    m = len(secret_vec)
    coeffs = [secret_vec] + [np.array([secrets.randbelow(ff.q) for _ in range(m)], dtype=object) for _ in range(t - 1)]
    return {i: _poly_eval_vector(coeffs, i, ff) for i in range(1, n + 1)}

def share_scalar_at(secret: int, x_points: Sequence[int], t: int, ff: FixedField) -> Dict[int, int]:
    """
    按指定评估点 x_points 生成标量份额，返回 {x: f(x)}，便于处理“掉线/非连续ID”。
    """
    coeffs = [secret] + [secrets.randbelow(ff.q) for _ in range(t - 1)]
    return {int(x): _poly_eval_scalar(coeffs, int(x), ff) for x in x_points}

def share_vector_at(secret_vec: np.ndarray, x_points: Sequence[int], t: int, ff: FixedField) -> Dict[int, np.ndarray]:
    """
    按指定评估点 x_points 生成向量份额，返回 {x: f(x)}。
    """
    m = len(secret_vec)
    coeffs = [secret_vec] + [np.array([secrets.randbelow(ff.q) for _ in range(m)], dtype=object) for _ in range(t - 1)]
    return {int(x): _poly_eval_vector(coeffs, int(x), ff) for x in x_points}

def lagrange_reconstruct(points: List[Tuple[int, int]], ff: FixedField) -> int:
    """标量秘密重构（给出 t 个 (x_j, y_j)）——式(4)"""
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    s = 0
    for j, xj in enumerate(xs):
        num, den = 1, 1
        for m, xm in enumerate(xs):
            if m == j: 
                continue
            num = ff.mul(num, xm % ff.q)
            den = ff.mul(den, (xm - xj) % ff.q)
        lj = ff.mul(num, ff.inv(den))
        s = ff.add(s, ff.mul(ys[j], lj))
    return s

def lagrange_reconstruct_vector(points: List[Tuple[int, np.ndarray]], ff: FixedField) -> np.ndarray:
    """向量秘密重构：坐标独立做拉格朗日插值（与上同理）"""
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    m = len(ys[0])
    s = np.zeros(m, dtype=object)
    for j, xj in enumerate(xs):
        num, den = 1, 1
        for m2, xm in enumerate(xs):
            if m2 == j: 
                continue
            num = ff.mul(num, xm % ff.q)
            den = ff.mul(den, (xm - xj) % ff.q)
        lj = ff.mul(num, ff.inv(den))
        s = (s + (ys[j] * lj) % ff.q) % ff.q
    return s
