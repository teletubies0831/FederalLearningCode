# -*- coding: utf-8 -*-
"""
有限域与定点量化工具
- 论文中所有“向量/标量”的加减乘校验等，均在某个有限域 F 中进行（见第 3 节与式(3)(4)(5)）。
- 我们选用大素数 q 做模，且用“定点量化”把浮点梯度/权重映射到 Z_q，再进行同态加减。
- 量化/反量化要与 q、scale 协同设置，避免溢出。
"""
from __future__ import annotations
import os
import numpy as np
from dataclasses import dataclass

# 选用一个很大的素数作为模（Mersenne 素数 2^127-1）
Q = (1 << 127) - 1

@dataclass
class FixedField:
    q: int = Q
    scale: int = 10**6  # 定点缩放系数：1e-6 精度

    def encode_vec(self, x: np.ndarray) -> np.ndarray:
        """浮点向量→整数域（取整后 mod q）"""
        xi = np.round(x * self.scale).astype(object)
        return np.mod(xi, self.q)

    def decode_vec(self, xi: np.ndarray) -> np.ndarray:
        """整数域→浮点向量"""
        # 还原到有符号范围（使用 Python 大整数避免溢出），再除以 scale
        return np.array([self._to_signed(int(v)) / float(self.scale) for v in xi], dtype=float)

    def encode_scalar(self, x: float | int) -> int:
        """标量定点编码：对正数避免量化为 0（至少编码为 1 单位），以防止聚合分母为 0。"""
        if isinstance(x, (int, np.integer)):
            xi = int(x)
        else:
            xi = int(round(float(x) * self.scale))
        if xi == 0 and float(x) > 0.0:
            xi = 1
        return xi % self.q

    def decode_scalar(self, xi: int) -> float:
        return self._to_signed(int(xi)) / float(self.scale)

    def add(self, a, b):
        return (a + b) % self.q

    def sub(self, a, b):
        return (a - b) % self.q

    def mul(self, a, b):
        return (a * b) % self.q

    def inv(self, x: int) -> int:
        """乘法逆元（费马小定理：x^(q-2) mod q）"""
        return pow(x, self.q - 2, self.q)

    def hadamard(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Hadamard 逐元素乘法（论文式(12)里的 a ∘ (...) ）"""
        return (a * b) % self.q

    def _to_signed(self, v: int) -> int:
        """把 [0, q) 映射回有符号整数区间，便于 decode（只用于展示/训练，不影响协议正确性）"""
        if v > self.q // 2:
            return v - self.q
        return v
