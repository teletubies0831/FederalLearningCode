# -*- coding: utf-8 -*-
"""
协议主流程（四个阶段）
对应论文第 5 节：5.1 概述，5.2 详细流程；含式(7)–(23)所有关键计算与验证
"""
from __future__ import annotations
import os, math, secrets
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from field import FixedField
from shamir import (
    share_scalar,
    share_vector,
    lagrange_reconstruct,
    lagrange_reconstruct_vector,
    share_scalar_at,
    share_vector_at,
)
from crypto_primitives import (
    gen_x25519_kp, ecdh_shared_key, aead_encrypt, aead_decrypt,
    prg_b1_b2, prg_b1_b2_round,
)
from truth_discovery import weight_inverse_l2
from utils import hadamard, pack_vec_and_scalar, unpack_vec_and_scalar

# ===== 实体定义 =====

@dataclass
class TA:
    """可信权威（只在 Setup 阶段参与，之后离线）"""
    ff: FixedField
    m: int                    # 梯度维度
    global_seed_b: bytes      # 全局随机种子 b（用于第二层掩码 b1,b2 的 PRG）
    a_vec: np.ndarray         # 随机向量 a ∈ F^m（用于 V1 标签）
    c_sca: int                # 随机标量 c ∈ F（用于 V2 标签）
    b_mode: str = "per_round" # b 派生模式：per_round（默认，论文对齐）或 per_user

    @staticmethod
    def initialize(ff: FixedField, m: int, b_mode: str = "per_round") -> "TA":
        global_seed_b = os.urandom(32)          # b
        a_vec = np.array([secrets.randbelow(ff.q) for _ in range(m)], dtype=object)  # a
        c_sca = secrets.randbelow(ff.q)         # c
        return TA(ff=ff, m=m, global_seed_b=global_seed_b, a_vec=a_vec, c_sca=c_sca, b_mode=b_mode)

    def derive_b1_b2(self, round_t: int, uid: int | None) -> Tuple[np.ndarray, int]:
        """按配置派生第二层掩码：per_round -> 所有客户端相同；per_user -> 每用户不同。"""
        if self.b_mode == "per_round":
            return prg_b1_b2_round(self.global_seed_b, round_t, self.m)
        else:
            if uid is None:
                raise ValueError("per_user b_mode requires a valid uid")
            return prg_b1_b2(self.global_seed_b, round_t, int(uid), self.m)

@dataclass
class Client:
    """客户端（半诚实；按协议行事但好奇）"""
    uid: int
    ff: FixedField
    m: int
    ta: TA
    sk: bytes = field(init=False)
    pk: bytes = field(init=False)

    def __post_init__(self):
        self.sk, self.pk = gen_x25519_kp()

    def local_train_and_package(self, round_t: int, x_prev_float: np.ndarray, make_grad_fn, 
                                n_online: int, t_thr: int, peer_pubkeys: Dict[int, bytes]) -> Dict:
        """
        Local computation 阶段（论文 5.2 Phase 2）：
        1) 本地训练得到 x_i^T；2) 计算 w_i；3) 生成双层掩码与标签；
        4) 对 r1_i（向量）与 r2_i（标量）做 Shamir 拆分，并用 ECDH+AESGCM 加密发给每个对等方。
        """
        ff = self.ff
        # 1) 本地“梯度”向量（示例：由用户自定义 make_grad_fn 产生）
        x_i = make_grad_fn(x_prev_float, self.uid)

        # 2) 权重（默认用 “距离反比” 规则；可替换为 chi2_rule）
        w_i = weight_inverse_l2(x_i, x_prev_float, C=1.0)

        # 3) 编码到有限域
        x_prev = ff.encode_vec(x_prev_float)
        x_i_enc = ff.encode_vec(x_i)
        w_i_enc = ff.encode_scalar(w_i)

        # 4) 计算加权梯度（标量*向量，在域上做乘法）
        wix = (x_i_enc * w_i_enc) % ff.q  # wix^T_i

        # 5) 第一层随机掩码 r1_i（向量）、r2_i（标量）
        r1_i = np.array([secrets.randbelow(ff.q) for _ in range(self.m)], dtype=object)
        r2_i = secrets.randbelow(ff.q)

        # 6) 第二层掩码（基于全局种子 b 的 PRG）b1（向量）、b2（标量）——式(7)(8)
        b1_vec, b2_sca = self.ta.derive_b1_b2(round_t, self.uid)

        # 7) 密文载荷（式(7)(8)）
        enc_wix = (wix + r1_i + b1_vec) % ff.q         # [[wix^T_i]]
        enc_wi  = (w_i_enc + r2_i + b2_sca) % ff.q     # [[w_i]]

        # 8) 标签（式(12)(13)）V1_i = a ∘ (wix + r1_i)；V2_i = c*(w_i + r2_i)
        V1_i = (self.ta.a_vec * ((wix + r1_i) % ff.q)) % ff.q
        V2_i = (self.ta.c_sca * ((w_i_enc + r2_i) % ff.q)) % ff.q

        # 9) Shamir 拆分（式(9)(10)）—按“在线对等方ID”作为评估点，适配掉线/非连续ID
        #    注意：重构总掩码需要 R_j = sum_i f_i(j)。因此每个客户端必须包含“对自己的份额” f_i(self.uid)。
        #    我们在本地生成对所有在线 ID（含自己）的份额，并将“自份额”保存在实例上，供后续解密累加时加入。
        all_ids = list(peer_pubkeys.keys())
        shares_r1_all = share_vector_at(r1_i, all_ids, t_thr, ff)
        shares_r2_all = share_scalar_at(r2_i, all_ids, t_thr, ff)
        # 保存自份额，用于 decrypt_sum_share 阶段加入
        self._self_share_r1 = shares_r1_all[self.uid]
        self._self_share_r2 = shares_r2_all[self.uid]
        # 发给对等方的仅包含“非自己”的份额
        shares_r1 = {pid: val for pid, val in shares_r1_all.items() if pid != self.uid}
        shares_r2 = {pid: val for pid, val in shares_r2_all.items() if pid != self.uid}

        # 10) 用 ECDH+AESGCM 加密发给每个在线对等方（式(11)）
        encrypted_shares = {}
        for peer_id, peer_pk in peer_pubkeys.items():
            if peer_id == self.uid:
                continue
            key = ecdh_shared_key(self.sk, peer_pk)
            blob = pack_vec_and_scalar(shares_r1[peer_id], shares_r2[peer_id])
            encrypted_shares[peer_id] = aead_encrypt(key, blob, aad=b"vswa-share")

        # 11) 返回给服务器的消息包
        return dict(
            uid=self.uid,
            pk=self.pk,
            enc_shares=encrypted_shares,   # 让服务器转发
            enc_wix=enc_wix,
            enc_wi=enc_wi,
            V1_i=V1_i,
            V2_i=V2_i
        )

    def decrypt_sum_share(self, round_t: int, from_users: Dict[int, bytes], peer_pubkeys: Dict[int, bytes]) -> Tuple[np.ndarray, int]:
        """
        Secure aggregation 阶段的用户侧：解密收到的“对自己”的份额，加总成 R1_j、R2_j（式(14)）
        返回 (R1_j, R2_j)
        """
        ff = self.ff
        R1_j = np.zeros(self.m, dtype=object)
        R2_j = 0
        for uid_i, blob in from_users.items():
            key = ecdh_shared_key(self.sk, peer_pubkeys[uid_i])
            vec, sca = unpack_vec_and_scalar(aead_decrypt(key, blob, aad=b"vswa-share"))
            R1_j = (R1_j + vec) % ff.q
            R2_j = (R2_j + sca) % ff.q
        # 加上“自己生成的自份额”，使 R_j = sum_i f_i(j)
        if hasattr(self, "_self_share_r1") and hasattr(self, "_self_share_r2"):
            R1_j = (R1_j + self._self_share_r1) % ff.q
            R2_j = (R2_j + self._self_share_r2) % ff.q
        return R1_j, R2_j

@dataclass
class Server:
    """云服务器（可能作恶；因此需要可验证）"""
    ff: FixedField
    m: int
    ta: TA
    t_thr: int
    round_t: int = 1

    # 在线用户的公钥登记（Phase 1）
    online_pubkeys: Dict[int, bytes] = field(default_factory=dict)

    def register_users(self, user_msgs: List[Dict]):
        """收集/广播公钥（5.2 Phase 1）"""
        for msg in user_msgs:
            self.online_pubkeys[msg["uid"]] = msg["pk"]

    def distribute_peer_shares(self, user_msgs: List[Dict]) -> Dict[int, Dict[int, bytes]]:
        """把每个用户打包好的“发给别人”的密文份额按目的地转发"""
        inbox: Dict[int, Dict[int, bytes]] = {uid: {} for uid in self.online_pubkeys}
        for msg in user_msgs:
            for peer_id, blob in msg["enc_shares"].items():
                inbox.setdefault(peer_id, {})[msg["uid"]] = blob
        return inbox

    def aggregate_and_prove(self, user_msgs: List[Dict], sum_R1_points: List[Tuple[int, np.ndarray]], sum_R2_points: List[Tuple[int, int]]) -> Dict:
        """
        Phase 3：服务器侧聚合 + 证明材料（式(15)–(20)）
        - 用 t 个 (j, R1_j) / (j, R2_j) 做拉格朗日重构总掩码 R1、R2；
        - 计算 Xagg、Wagg 与 V1_agg、V2_agg，并把 (R1,R2) 一并发给用户。
        """
        ff = self.ff
        # 重构两类总掩码（式(15)(16)）
        R1 = lagrange_reconstruct_vector(sum_R1_points, ff)
        R2 = lagrange_reconstruct(sum_R2_points, ff)

        # 取出密文加权梯度与权重
        enc_wix_all = [msg["enc_wix"] for msg in user_msgs]
        enc_wi_all  = [msg["enc_wi"]  for msg in user_msgs]
        V1_all = [msg["V1_i"] for msg in user_msgs]
        V2_all = [msg["V2_i"] for msg in user_msgs]

        # Xagg, Wagg（式(17)(18)）
        Xagg = np.sum(enc_wix_all, axis=0, dtype=object) % ff.q
        Xagg = (Xagg - R1) % ff.q  # 注意：式(17)里还剩 |U2|*b1，验证时再扣
        Wagg = (np.sum(enc_wi_all, dtype=object) - R2) % ff.q

        # 验证标签聚合（式(19)(20)）
        V1_agg = np.sum(V1_all, axis=0, dtype=object) % ff.q
        V2_agg = np.sum(V2_all, dtype=object) % ff.q

        return dict(Xagg=Xagg, Wagg=Wagg, V1_agg=V1_agg, V2_agg=V2_agg, R1=R1, R2=R2, n=len(user_msgs))

def client_side_verify_and_update(ff: FixedField, ta: TA, agg_msg: Dict, b1_vec: np.ndarray, b2_sca: int) -> Tuple[bool, np.ndarray]:
    """
    Phase 4：客户端校验并更新（式(21)(22)(23)）
    - 校验通过：输出新一轮全局梯度 x^{T+1}（浮点）
    """
    nU2 = agg_msg["n"]
    Xagg = agg_msg["Xagg"]; Wagg = agg_msg["Wagg"]
    V1_agg = agg_msg["V1_agg"]; V2_agg = agg_msg["V2_agg"]
    R1 = agg_msg["R1"]; R2 = agg_msg["R2"]

    # 验证等式（式(21)(22)）
    left1 = V1_agg
    right1 = (ta.a_vec * ((Xagg - (nU2 * b1_vec) % ff.q + R1) % ff.q)) % ff.q
    left2 = V2_agg
    right2 = (ta.c_sca * ((Wagg - (nU2 * b2_sca) % ff.q + R2) % ff.q)) % ff.q

    ok = (np.all(left1 == right1) and (int(left2) % ff.q) == (int(right2) % ff.q))
    if not ok:
        return False, None

    # 通过则更新全局梯度（式(23)）：(Xagg - |U2|b1)/(Wagg - |U2|b2)
    # 为提高数值稳定性，这里采用：先解码再做实数除法，避免大模数下的奇异情况。
    num_enc = (Xagg - (nU2 * b1_vec) % ff.q) % ff.q
    den_enc = (Wagg - (nU2 * b2_sca) % ff.q) % ff.q
    num = ff.decode_vec(num_enc)
    den = ff.decode_scalar(int(den_enc))
    if abs(den) < 1e-18:
        return False, None
    x_next = num / den
    return True, x_next
