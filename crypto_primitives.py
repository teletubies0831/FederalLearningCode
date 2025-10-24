# -*- coding: utf-8 -*-
"""
密码学原语与双层掩码 PRG
- ECDH 采用 X25519（Curve25519），见 cryptography 文档；
- 共享密钥经 HKDF 派生为 AES-GCM 密钥，用于加密发给各对等方的“掩码份额”；
- 全局随机种子 b 通过 HKDF → 为每轮每个客户端派生第二层掩码 b1(向量)、b2(标量)。
"""
from __future__ import annotations
import os, struct, hmac, hashlib
import numpy as np
from typing import Tuple, Dict
from cryptography.hazmat.primitives.asymmetric import x25519  # :contentReference[oaicite:8]{index=8}
from cryptography.hazmat.primitives.kdf.hkdf import HKDF       # :contentReference[oaicite:9]{index=9}
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def gen_x25519_kp() -> Tuple[bytes, bytes]:
    """生成 X25519 密钥对（私钥 bytes, 公钥 bytes）"""
    sk = x25519.X25519PrivateKey.generate()
    pk = sk.public_key()
    return sk.private_bytes_raw(), pk.public_bytes_raw()

def ecdh_shared_key(sk_bytes: bytes, pk_bytes: bytes, info: bytes=b"vswa-ecdh") -> bytes:
    """X25519 ECDH → HKDF 导出 32 字节对称密钥"""
    sk = x25519.X25519PrivateKey.from_private_bytes(sk_bytes)
    pk = x25519.X25519PublicKey.from_public_bytes(pk_bytes)
    shared = sk.exchange(pk)
    kdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info)
    return kdf.derive(shared)

def aead_encrypt(key: bytes, plaintext: bytes, aad: bytes=b"") -> bytes:
    """AES-GCM 加密：nonce|ciphertext|tag"""
    aes = AESGCM(key)
    nonce = os.urandom(12)
    ct = aes.encrypt(nonce, plaintext, aad)
    return nonce + ct

def aead_decrypt(key: bytes, blob: bytes, aad: bytes=b"") -> bytes:
    aes = AESGCM(key)
    nonce, ct = blob[:12], blob[12:]
    return aes.decrypt(nonce, ct, aad)

def prg_b1_b2(global_seed_b: bytes, round_t: int, user_id: int, dim_m: int) -> Tuple[np.ndarray, int]:
    """
    全局种子 b → (b1 向量, b2 标量)（第二层掩码，式(7)(8)中的 b1,b2）
    - 以 (b || round || uid) 作为 HKDF 的输入材料，稳定可复现。
    """
    info = b"vswa-b-layer" + struct.pack(">Q", round_t) + struct.pack(">I", user_id)
    total_len = 32 + 8*dim_m
    keymat = _hkdf_stream(global_seed_b, info, total_len)
    b2 = int.from_bytes(keymat[:32], "big")
    vec = []
    off = 32
    for _ in range(dim_m):
        v = int.from_bytes(keymat[off:off+8], "big")
        vec.append(v)
        off += 8
    return (np.array(vec, dtype=object), b2)

def prg_b1_b2_round(global_seed_b: bytes, round_t: int, dim_m: int) -> Tuple[np.ndarray, int]:
    """
    从全局种子 b 为“某一轮”派生统一的第二层掩码 (b1 向量, b2 标量)，所有客户端相同。
    便于与论文中 Xagg/Wagg 的 |U2|*b1 / |U2|*b2 形式严格对齐。
    """
    info = b"vswa-b-layer/common" + struct.pack(">Q", round_t)
    total_len = 32 + 8*dim_m
    keymat = _hkdf_stream(global_seed_b, info, total_len)
    b2 = int.from_bytes(keymat[:32], "big")
    vec = []
    off = 32
    for _ in range(dim_m):
        v = int.from_bytes(keymat[off:off+8], "big")
        vec.append(v)
        off += 8
    return (np.array(vec, dtype=object), b2)

def _hkdf_stream(seed: bytes, info: bytes, total_len: int) -> bytes:
    """先用 HKDF-SHA256 导出 32 字节密钥 K，再用 HMAC-SHA256 计数器扩展到 total_len 字节。"""
    kdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"vswa-seed" + info)
    K = kdf.derive(seed)
    out = bytearray()
    counter = 1
    while len(out) < total_len:
        msg = info + counter.to_bytes(4, "big")
        block = hmac.new(K, msg, hashlib.sha256).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:total_len])
