# -*- coding: utf-8 -*-
"""
工具函数
"""
from __future__ import annotations
import numpy as np
from typing import Tuple

def hadamard(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """逐元素乘法（式(12)）"""
    return (a * b) % q

def pack_vec_and_scalar(vec: np.ndarray, scalar: int) -> bytes:
    """把向量与标量序列化成 bytes（简洁实现，实际可用 protobuf/CBOR）"""
    out = len(vec).to_bytes(4, "big")
    for v in vec:
        out += int(v).to_bytes(32, "big", signed=False)
    out += int(scalar).to_bytes(32, "big", signed=False)
    return out

def unpack_vec_and_scalar(blob: bytes) -> Tuple[np.ndarray, int]:
    m = int.from_bytes(blob[:4], "big")
    off = 4
    vec = []
    for _ in range(m):
        vec.append(int.from_bytes(blob[off:off+32], "big"))
        off += 32
    scalar = int.from_bytes(blob[off:off+32], "big")
    return np.array(vec, dtype=object), scalar
