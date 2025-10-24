# -*- coding: utf-8 -*-
"""
鍗忚涓绘祦绋嬶紙鍥涗釜闃舵锛?
瀵瑰簲璁烘枃绗?5 鑺傦細5.1 姒傝堪锛?.2 璇︾粏娴佺▼锛涘惈寮?7)鈥?23)鎵€鏈夊叧閿绠椾笌楠岃瘉
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

# ===== 瀹炰綋瀹氫箟 =====

@dataclass
class TA:
    """鍙俊鏉冨▉锛堝彧鍦?Setup 闃舵鍙備笌锛屼箣鍚庣绾匡級"""
    ff: FixedField
    m: int                    # 姊害缁村害
    global_seed_b: bytes      # 鍏ㄥ眬闅忔満绉嶅瓙 b锛堢敤浜庣浜屽眰鎺╃爜 b1,b2 鐨?PRG锛?    a_vec: np.ndarray         # 闅忔満鍚戦噺 a 鈭?F^m锛堢敤浜?V1 鏍囩锛?    c_sca: int                # 闅忔満鏍囬噺 c 鈭?F锛堢敤浜?V2 鏍囩锛?    b_mode: str = "per_round" # b 娲剧敓妯″紡锛歱er_round锛堥粯璁わ紝璁烘枃瀵归綈锛夋垨 per_user

    @staticmethod
    def initialize(ff: FixedField, m: int, b_mode: str = "per_round") -> "TA":
        global_seed_b = os.urandom(32)          # b
        a_vec = np.array([secrets.randbelow(ff.q) for _ in range(m)], dtype=object)  # a
        c_sca = secrets.randbelow(ff.q)         # c
        return TA(ff=ff, m=m, global_seed_b=global_seed_b, a_vec=a_vec, c_sca=c_sca, b_mode=b_mode)

    def derive_b1_b2(self, round_t: int, uid: int | None) -> Tuple[np.ndarray, int]:
        """鎸夐厤缃淳鐢熺浜屽眰鎺╃爜锛歱er_round -> 鎵€鏈夊鎴风鐩稿悓锛沺er_user -> 姣忕敤鎴蜂笉鍚屻€?""
        if self.b_mode == "per_round":
            return prg_b1_b2_round(self.global_seed_b, round_t, self.m)
        else:
            if uid is None:
                raise ValueError("per_user b_mode requires a valid uid")
            return prg_b1_b2(self.global_seed_b, round_t, int(uid), self.m)

@dataclass
class Client:
    """瀹㈡埛绔紙鍗婅瘹瀹烇紱鎸夊崗璁浜嬩絾濂藉锛?""
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
        Local computation 闃舵锛堣鏂?5.2 Phase 2锛夛細
        1) 鏈湴璁粌寰楀埌 x_i^T锛?) 璁＄畻 w_i锛?) 鐢熸垚鍙屽眰鎺╃爜涓庢爣绛撅紱
        4) 瀵?r1_i锛堝悜閲忥級涓?r2_i锛堟爣閲忥級鍋?Shamir 鎷嗗垎锛屽苟鐢?ECDH+AESGCM 鍔犲瘑鍙戠粰姣忎釜瀵圭瓑鏂广€?
        """
        ff = self.ff
        # 1) 鏈湴鈥滄搴︹€濆悜閲忥紙绀轰緥锛氱敱鐢ㄦ埛鑷畾涔?make_grad_fn 浜х敓锛?
        x_i = make_grad_fn(x_prev_float, self.uid)

        # 2) 鏉冮噸锛堥粯璁ょ敤 鈥滆窛绂诲弽姣斺€?瑙勫垯锛涘彲鏇挎崲涓?chi2_rule锛?
        w_i = weight_inverse_l2(x_i, x_prev_float, C=1.0)

        # 3) 缂栫爜鍒版湁闄愬煙
        x_prev = ff.encode_vec(x_prev_float)
        x_i_enc = ff.encode_vec(x_i)
        w_i_enc = ff.encode_scalar(w_i)

        # 4) 璁＄畻鍔犳潈姊害锛堟爣閲?鍚戦噺锛屽湪鍩熶笂鍋氫箻娉曪級
        wix = (x_i_enc * w_i_enc) % ff.q  # wix^T_i

        # 5) 绗竴灞傞殢鏈烘帺鐮?r1_i锛堝悜閲忥級銆乺2_i锛堟爣閲忥級
        r1_i = np.array([secrets.randbelow(ff.q) for _ in range(self.m)], dtype=object)
        r2_i = secrets.randbelow(ff.q)

        # 6) 绗簩灞傛帺鐮侊紙鍩轰簬鍏ㄥ眬绉嶅瓙 b 鐨?PRG锛塨1锛堝悜閲忥級銆乥2锛堟爣閲忥級鈥斺€斿紡(7)(8)
        b1_vec, b2_sca = self.ta.derive_b1_b2(round_t, self.uid)

        # 7) 瀵嗘枃杞借嵎锛堝紡(7)(8)锛?
        enc_wix = (wix + r1_i + b1_vec) % ff.q         # [[wix^T_i]]
        enc_wi  = (w_i_enc + r2_i + b2_sca) % ff.q     # [[w_i]]

        # 8) 鏍囩锛堝紡(12)(13)锛塚1_i = a 鈭?(wix + r1_i)锛沄2_i = c*(w_i + r2_i)
        V1_i = (self.ta.a_vec * ((wix + r1_i) % ff.q)) % ff.q
        V2_i = (self.ta.c_sca * ((w_i_enc + r2_i) % ff.q)) % ff.q

        # 9) Shamir 鎷嗗垎锛堝紡(9)(10)锛夆€旀寜鈥滃湪绾垮绛夋柟ID鈥濅綔涓鸿瘎浼扮偣锛岄€傞厤鎺夌嚎/闈炶繛缁璉D
        peer_ids = [pid for pid in peer_pubkeys.keys() if pid != self.uid]
        shares_r1 = share_vector_at(r1_i, peer_ids, t_thr, ff)
        shares_r2 = share_scalar_at(r2_i, peer_ids, t_thr, ff)

        # 10) 鐢?ECDH+AESGCM 鍔犲瘑鍙戠粰姣忎釜鍦ㄧ嚎瀵圭瓑鏂癸紙寮?11)锛?
        encrypted_shares = {}
        for peer_id, peer_pk in peer_pubkeys.items():
            if peer_id == self.uid:
                continue
            key = ecdh_shared_key(self.sk, peer_pk)
            blob = pack_vec_and_scalar(shares_r1[peer_id], shares_r2[peer_id])
            encrypted_shares[peer_id] = aead_encrypt(key, blob, aad=b"vswa-share")

        # 11) 杩斿洖缁欐湇鍔″櫒鐨勬秷鎭寘
        return dict(
            uid=self.uid,
            pk=self.pk,
            enc_shares=encrypted_shares,   # 璁╂湇鍔″櫒杞彂
            enc_wix=enc_wix,
            enc_wi=enc_wi,
            V1_i=V1_i,
            V2_i=V2_i
        )

    def decrypt_sum_share(self, round_t: int, from_users: Dict[int, bytes], peer_pubkeys: Dict[int, bytes]) -> Tuple[np.ndarray, int]:
        """
        Secure aggregation 闃舵鐨勭敤鎴蜂晶锛氳В瀵嗘敹鍒扮殑鈥滃鑷繁鈥濈殑浠介锛屽姞鎬绘垚 R1_j銆丷2_j锛堝紡(14)锛?
        杩斿洖 (R1_j, R2_j)
        """
        ff = self.ff
        R1_j = np.zeros(self.m, dtype=object)
        R2_j = 0
        for uid_i, blob in from_users.items():
            key = ecdh_shared_key(self.sk, peer_pubkeys[uid_i])
            vec, sca = unpack_vec_and_scalar(aead_decrypt(key, blob, aad=b"vswa-share"))
            R1_j = (R1_j + vec) % ff.q
            R2_j = (R2_j + sca) % ff.q
        return R1_j, R2_j

@dataclass
class Server:
    """浜戞湇鍔″櫒锛堝彲鑳戒綔鎭讹紱鍥犳闇€瑕佸彲楠岃瘉锛?""
    ff: FixedField
    m: int
    ta: TA
    t_thr: int
    round_t: int = 1

    # 鍦ㄧ嚎鐢ㄦ埛鐨勫叕閽ョ櫥璁帮紙Phase 1锛?
    online_pubkeys: Dict[int, bytes] = field(default_factory=dict)

    def register_users(self, user_msgs: List[Dict]):
        """鏀堕泦/骞挎挱鍏挜锛?.2 Phase 1锛?""
        for msg in user_msgs:
            self.online_pubkeys[msg["uid"]] = msg["pk"]

    def distribute_peer_shares(self, user_msgs: List[Dict]) -> Dict[int, Dict[int, bytes]]:
        """鎶婃瘡涓敤鎴锋墦鍖呭ソ鐨勨€滃彂缁欏埆浜衡€濈殑瀵嗘枃浠介鎸夌洰鐨勫湴杞彂"""
        inbox: Dict[int, Dict[int, bytes]] = {uid: {} for uid in self.online_pubkeys}
        for msg in user_msgs:
            for peer_id, blob in msg["enc_shares"].items():
                inbox.setdefault(peer_id, {})[msg["uid"]] = blob
        return inbox

    def aggregate_and_prove(self, user_msgs: List[Dict], sum_R1_points: List[Tuple[int, np.ndarray]], sum_R2_points: List[Tuple[int, int]]) -> Dict:
        """
        Phase 3锛氭湇鍔″櫒渚ц仛鍚?+ 璇佹槑鏉愭枡锛堝紡(15)鈥?20)锛?
        - 鐢?t 涓?(j, R1_j) / (j, R2_j) 鍋氭媺鏍兼湕鏃ラ噸鏋勬€绘帺鐮?R1銆丷2锛?
        - 璁＄畻 Xagg銆乄agg 涓?V1_agg銆乂2_agg锛屽苟鎶?(R1,R2) 涓€骞跺彂缁欑敤鎴枫€?
        """
        ff = self.ff
        # 閲嶆瀯涓ょ被鎬绘帺鐮侊紙寮?15)(16)锛?
        R1 = lagrange_reconstruct_vector(sum_R1_points, ff)
        R2 = lagrange_reconstruct(sum_R2_points, ff)

        # 鍙栧嚭瀵嗘枃鍔犳潈姊害涓庢潈閲?
        enc_wix_all = [msg["enc_wix"] for msg in user_msgs]
        enc_wi_all  = [msg["enc_wi"]  for msg in user_msgs]
        V1_all = [msg["V1_i"] for msg in user_msgs]
        V2_all = [msg["V2_i"] for msg in user_msgs]

        # Xagg, Wagg锛堝紡(17)(18)锛?
        Xagg = np.sum(enc_wix_all, axis=0, dtype=object) % ff.q
        Xagg = (Xagg - R1) % ff.q  # 娉ㄦ剰锛氬紡(17)閲岃繕鍓?|U2|*b1锛岄獙璇佹椂鍐嶆墸
        Wagg = (np.sum(enc_wi_all, dtype=object) - R2) % ff.q

        # 楠岃瘉鏍囩鑱氬悎锛堝紡(19)(20)锛?
        V1_agg = np.sum(V1_all, axis=0, dtype=object) % ff.q
        V2_agg = np.sum(V2_all, dtype=object) % ff.q

        return dict(Xagg=Xagg, Wagg=Wagg, V1_agg=V1_agg, V2_agg=V2_agg, R1=R1, R2=R2, n=len(user_msgs))

def client_side_verify_and_update(ff: FixedField, ta: TA, agg_msg: Dict, b1_vec: np.ndarray, b2_sca: int) -> Tuple[bool, np.ndarray]:
    """
    Phase 4锛氬鎴风鏍￠獙骞舵洿鏂帮紙寮?21)(22)(23)锛?
    - 鏍￠獙閫氳繃锛氳緭鍑烘柊涓€杞叏灞€姊害 x^{T+1}锛堟诞鐐癸級
    """
    nU2 = agg_msg["n"]
    Xagg = agg_msg["Xagg"]; Wagg = agg_msg["Wagg"]
    V1_agg = agg_msg["V1_agg"]; V2_agg = agg_msg["V2_agg"]
    R1 = agg_msg["R1"]; R2 = agg_msg["R2"]

    # 楠岃瘉绛夊紡锛堝紡(21)(22)锛?
    left1 = V1_agg
    right1 = (ta.a_vec * ((Xagg - (nU2 * b1_vec) % ff.q + R1) % ff.q)) % ff.q
    left2 = V2_agg
    right2 = (ta.c_sca * ((Wagg - (nU2 * b2_sca) % ff.q + R2) % ff.q)) % ff.q

    ok = (np.all(left1 == right1) and (int(left2) % ff.q) == (int(right2) % ff.q))
    if not ok:
        return False, None

    # 閫氳繃鍒欐洿鏂板叏灞€姊害锛堝紡(23)锛夛細(Xagg - |U2|b1)/(Wagg - |U2|b2)
    num_vec = (Xagg - (nU2 * b1_vec) % ff.q) % ff.q
    den = (Wagg - (nU2 * b2_sca) % ff.q) % ff.q
    den_inv = ff.inv(int(den))
    x_next_enc = (num_vec * den_inv) % ff.q
    x_next = ff.decode_vec(x_next_enc)
    return True, x_next

