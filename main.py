#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦学习 + 可验证安全聚合 的最小可运行示例（单机模拟多客户端）

阶段：
- Phase 1: 注册/广播公钥
- Phase 2: 客户端本地训练 + 打包密文载荷 + 加密份额
- Phase 3: 服务器重构掩码并聚合，生成验证材料
- Phase 4: 客户端侧验证并更新全局模型

运行：
  python main.py --clients 5 --dim 8 --thr 3 --rounds 3
依赖：numpy, cryptography
"""
from __future__ import annotations
import argparse
import math
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from field import FixedField
from protocol import TA, Client, Server, client_side_verify_and_update
from crypto_primitives import prg_b1_b2


# ====== MNIST 相关（PyTorch CNN 版）======
class SimpleCNN(nn.Module):
    """LeNet 风格的轻量 CNN，用于 MNIST。"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = torch.relu(nn.functional.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def torch_parameters_to_vector(model: nn.Module) -> np.ndarray:
    with torch.no_grad():
        vec = torch.cat([p.view(-1) for p in model.parameters()])
    return vec.detach().cpu().numpy().astype(np.float64)

def torch_vector_to_parameters(model: nn.Module, vec: np.ndarray):
    with torch.no_grad():
        t = torch.from_numpy(vec.astype(np.float32))
        pointer = 0
        for p in model.parameters():
            numel = p.numel()
            p.copy_(t[pointer:pointer+numel].to(p.device).view_as(p))
            pointer += numel
def evaluate_cnn_from_vector(vec: np.ndarray, test_loader: DataLoader, device: str) -> tuple[float, float]:
    model = SimpleCNN().to(device)
    torch_vector_to_parameters(model, vec)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss_sum += float(loss.item())
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += yb.size(0)
    return loss_sum / max(1, total), (correct / max(1, total))

def local_train_softmax(x_prev: np.ndarray, X: np.ndarray, y: np.ndarray, d: int, k: int,
                        lr: float = 0.1, epochs: int = 1, batch: int = 256, l2: float = 0.0) -> np.ndarray:
    """单客户端在本地数据 (X,y) 上做若干步 mini-batch SGD，返回更新后的参数向量 x_i。"""
    W, b = _unpack_params(x_prev, d, k)
    n = X.shape[0]
    idx = np.arange(n)
    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, n, batch):
            sel = idx[s : s + batch]
            Xb, yb = X[sel], y[sel]
            P = _softmax(Xb @ W + b)
            Y = _one_hot(yb, k)
            err = (P - Y) / Xb.shape[0]
            gW = Xb.T @ err + l2 * W
            gb = np.sum(err, axis=0)
            W -= lr * gW
            b -= lr * gb
    return _pack_params(W, b)

def build_mnist_loaders(n_clients: int, batch: int, data_root: str = "./data"):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    try:
        train_ds = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
        test_ds  = datasets.MNIST(root=data_root, train=False, download=True, transform=tfm)
    except Exception as e:
        print(f"TorchVision download failed: {e}\n请确认网络可用，或手动准备 MNIST 到 {data_root}。")
        raise

    idxs = np.arange(len(train_ds))
    np.random.shuffle(idxs)
    splits = np.array_split(idxs, n_clients)
    client_loaders = {}
    for i, arr in enumerate(splits):
        subset = Subset(train_ds, arr.tolist())
        client_loaders[i + 1] = DataLoader(subset, batch_size=batch, shuffle=True, num_workers=0)

    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)
    return client_loaders, test_loader


def make_grad_fn_factory(seed: int = 0, lr: float = 0.5):
    rng = np.random.RandomState(seed)

    # 为每个客户端生成一个“本地最优点” mu_i（只为模拟差异数据分布）
    def make_mu(uid: int, m: int):
        rng2 = np.random.RandomState(seed + uid)
        return rng2.normal(loc=uid * 0.05, scale=0.5, size=m)

    def make_grad_fn(x_prev: np.ndarray, uid: int, round_t: int) -> np.ndarray:
        m = x_prev.size
        mu_i = make_mu(uid, m)
        # 简单一阶“朝向本地最优”的步进 + 少量噪声
        grad = (x_prev - mu_i)
        noise = rng.normal(0, 0.01, size=m)
        return x_prev - lr * grad + noise

    return make_grad_fn


def simulate(rounds: int = 3, n_clients: int = 5, dim_m: int = 8, t_thr: int = 3, seed: int = 0,
             dataset: str = "toy", mnist_npz: str = "mnist.npz",
             offline_uid: int | None = None, offline_from_round: int = 10**9,
             lr: float = 0.1, local_epochs: int = 1, batch: int = 256,
             b_mode: str = "per_round", auto_download: bool = True,
             log_weights: bool = False, log_local_eval: bool = False,
             faulty_uids: list[int] | None = None, faulty_mode: str = "none",
             faulty_std: float = 0.5, faulty_ratio: float = 0.0,
             parallelism: int = 1):
    assert 1 <= t_thr <= n_clients, "threshold t must satisfy 1 <= t <= n_clients"

    ff = FixedField()
    faulty_uids = faulty_uids or []
    faulty_mode = (faulty_mode or "none").lower()
    faulty_ratio = float(max(0.0, min(1.0, faulty_ratio)))
    parallelism = max(1, int(parallelism))

    base_faulty_set = set(int(uid) for uid in faulty_uids)
    if faulty_ratio > 0.0:
        rng_ratio = np.random.RandomState(seed + 54321)
        target_count = min(
            n_clients,
            max(1, int(math.ceil(faulty_ratio * n_clients)))
        )
        if len(base_faulty_set) < target_count:
            remaining = [uid for uid in range(1, n_clients + 1) if uid not in base_faulty_set]
            need = target_count - len(base_faulty_set)
            if need >= len(remaining):
                base_faulty_set.update(remaining)
            elif need > 0:
                sampled = rng_ratio.choice(remaining, size=need, replace=False)
                base_faulty_set.update(int(uid) for uid in sampled)
    faulty_set = base_faulty_set
    faulty_rngs = {uid: np.random.RandomState(seed + 12345 + uid) for uid in faulty_set}

    def apply_faulty_behavior(x_prev_vec: np.ndarray, x_vec: np.ndarray, uid: int) -> np.ndarray:
        if uid not in faulty_set:
            return x_vec
        if faulty_mode == "sign_flip":
            return x_prev_vec - (x_vec - x_prev_vec)
        if faulty_mode == "gaussian":
            noise = faulty_rngs[uid].normal(loc=0.0, scale=faulty_std, size=x_prev_vec.shape)
            return x_prev_vec + noise
        # label_noise 或未知模式直接返回（label_noise 在训练时处理）
        return x_vec
    # Reproducibility (best-effort)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
    except Exception:
        pass

    # 数据与模型维度
    parallel_notice = None

    if dataset == "mnist":
        if parallelism > 1:
            parallel_notice = "MNIST uses PyTorch DataLoader; forcing sequential local training to avoid thread contention."
            parallelism = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 模型与维度
        model_proto = SimpleCNN().to(device)
        dim_m = int(sum(p.numel() for p in model_proto.parameters()))
        # 数据加载与客户端划分
        client_loaders, test_loader = build_mnist_loaders(n_clients, batch)

        # 本地训练函数：从向量加载模型，训练若干 epoch，返回更新后的参数向量
        def make_grad_fn(x_prev: np.ndarray, uid: int, round_t: int) -> np.ndarray:
            model = SimpleCNN().to(device)
            torch_vector_to_parameters(model, x_prev)
            model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            for _ in range(local_epochs):
                for xb, yb in client_loaders[uid]:
                    xb, yb = xb.to(device), yb.to(device)
                    if uid in faulty_set and faulty_mode == "label_noise":
                        yb = torch.randint(low=0, high=10, size=yb.shape, device=yb.device)
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
            vec = torch_parameters_to_vector(model)
            return apply_faulty_behavior(x_prev, vec, uid)

        # 初始全局参数（从随机初始化模型展开）
        x_prev = torch_parameters_to_vector(model_proto)
    else:
        # toy：使用之前的“朝向局部mu”的示例
        x_prev = np.zeros(dim_m, dtype=float)
        base_grad_fn = make_grad_fn_factory(seed=seed)

        def make_grad_fn(x_prev: np.ndarray, uid: int, round_t: int) -> np.ndarray:
            vec = base_grad_fn(x_prev, uid, round_t)
            return apply_faulty_behavior(x_prev, vec, uid)

    ta = TA.initialize(ff, dim_m, b_mode=b_mode)
    clients = [Client(uid=i, ff=ff, m=dim_m, ta=ta) for i in range(1, n_clients + 1)]

    for t in range(1, rounds + 1):
        print(f"\n=== Round {t} ===")
        if faulty_set and t == 1:
            extra = f", std={faulty_std}" if faulty_mode == "gaussian" else ""
            ratio_info = f", ratio~{faulty_ratio:.3f}" if faulty_ratio > 0 else ""
            print(f"  faulty clients={sorted(faulty_set)}, mode={faulty_mode}{extra}{ratio_info}")
        if parallel_notice and t == 1:
            print(f"  note: {parallel_notice}")
        if parallelism > 1 and t == 1:
            print(f"  parallelism={parallelism} (local training in thread pool)")
        # 本轮在线用户（模拟掉线）
        online_clients = [c for c in clients if not (offline_uid is not None and c.uid == offline_uid and t >= offline_from_round)]
        if len(online_clients) < t_thr:
            raise RuntimeError(f"在线用户数 {len(online_clients)} 小于门限 t={t_thr}，无法重构。请调小 --thr 或减少掉线。")

        # Phase 1: 注册/广播公钥（仅在线）
        server = Server(ff=ff, m=dim_m, ta=ta, t_thr=t_thr, round_t=t)
        reg_msgs = [dict(uid=c.uid, pk=c.pk) for c in online_clients]
        server.register_users(reg_msgs)
        peer_pubkeys = {c.uid: c.pk for c in online_clients}

        # Phase 2: 客户端本地训练 + 打包密文载荷
        def _package_for_client(client: Client):
            msg = client.local_train_and_package(
                round_t=t,
                x_prev_float=x_prev,
                make_grad_fn=make_grad_fn,
                n_online=len(online_clients),
                t_thr=t_thr,
                peer_pubkeys=peer_pubkeys,
            )
            return client.uid, msg

        if parallelism > 1 and len(online_clients) > 1:
            order_map = {c.uid: idx for idx, c in enumerate(online_clients)}
            results: list[tuple[int, dict]] = []
            with ThreadPoolExecutor(max_workers=parallelism) as executor:
                futures = [executor.submit(_package_for_client, c) for c in online_clients]
                for fut in as_completed(futures):
                    uid, msg = fut.result()
                    results.append((uid, msg))
            results.sort(key=lambda item: order_map[item[0]])
            user_msgs = [msg for _, msg in results]
        else:
            user_msgs = []
            for c in online_clients:
                _, msg = _package_for_client(c)
                user_msgs.append(msg)

        # 服务器转发对等方份额（每个用户收到“别人给自己”的密文份额）
        inbox = server.distribute_peer_shares(user_msgs)

        # 每个用户解密并累加份额，得到 (j, R1_j), (j, R2_j)
        sum_R1_points = []  # List[(j, np.ndarray)]
        sum_R2_points = []  # List[(j, int)]
        for c in online_clients:
            R1_j, R2_j = c.decrypt_sum_share(
                round_t=t,
                from_users=inbox.get(c.uid, {}),
                peer_pubkeys=peer_pubkeys,
            )
            sum_R1_points.append((c.uid, R1_j))
            sum_R2_points.append((c.uid, R2_j))

        # Phase 3: 服务器侧重构总掩码并聚合，生成证明材料
        agg_msg = server.aggregate_and_prove(user_msgs, sum_R1_points, sum_R2_points)

        # 为验证准备第二层掩码参数：
        # - per_round：b 在一轮内对所有客户端相同，直接传入公共 b；
        # - per_user：使用在线集合 U2 的平均值 b_mean，使得 |U2|*b_mean == sum_i b_i 成立。
        if ta.b_mode == "per_round":
            b1_mean, b2_mean = ta.derive_b1_b2(t, uid=0)
        else:
            b1_sum = np.zeros(dim_m, dtype=object)
            b2_sum = 0
            for c in online_clients:
                b1_i, b2_i = ta.derive_b1_b2(t, uid=c.uid)
                b1_sum = (b1_sum + b1_i) % ff.q
                b2_sum = (b2_sum + b2_i) % ff.q
            n_inv = ff.inv(len(online_clients) % ff.q)
            b1_mean = (b1_sum * n_inv) % ff.q
            b2_mean = (b2_sum * n_inv) % ff.q

        # Phase 4: 客户端侧验证并更新全局模型
        ok, x_next, agg_diag = client_side_verify_and_update(ff, ta, agg_msg, b1_mean, b2_mean)
        if not ok:
            raise RuntimeError("Verification failed: aggregated labels do not match.")

        # 打印本轮信息
        print(f"online={len(online_clients)} / total={n_clients}, dim={dim_m}, thr={t_thr}")
        dx = x_next - x_prev
        print(f"x_prev[:4]={np.round(x_prev[:4], 6)} -> x_next[:4]={np.round(x_next[:4], 6)}  |  ||Δx||2={np.linalg.norm(dx):.3e}")
        if log_weights:
            weights = [getattr(c, "_last_weight", float("nan")) for c in online_clients]
            if weights:
                w_arr = np.array(weights, dtype=float)
                print(
                    "  weights stats: mean={:.4e}, min={:.4e}, max={:.4e}, sum={:.4e}".format(
                        np.mean(w_arr), np.min(w_arr), np.max(w_arr), np.sum(w_arr)
                    )
                )
            print(
                "  aggregate diag: weight_sum={:.4e}, numerator_l2={:.4e}, numerator_mean={:.4e}".format(
                    agg_diag["weight_sum"], agg_diag["numerator_l2"], agg_diag["numerator_mean"]
                )
            )

        # 评估（MNIST 测试集）
        if dataset == "mnist":
            test_loss, test_acc = evaluate_cnn_from_vector(x_next, test_loader, device)
            print(f"test: loss={test_loss:.4f}, acc={test_acc*100:.2f}%")
            if log_local_eval:
                for c in online_clients:
                    local_vec = getattr(c, "_last_local_vec", None)
                    if local_vec is None:
                        continue
                    loss_i, acc_i = evaluate_cnn_from_vector(local_vec, test_loader, device)
                    weight_i = getattr(c, "_last_weight", float("nan"))
                    print(
                        f"    client {c.uid}: w={weight_i:.4e}, local loss={loss_i:.4f}, local acc={acc_i*100:.2f}%"
                    )
        else:
            # toy 情况下没有标签，打印范数以辅助判断收敛
            print(f"toy: ||x||2={np.linalg.norm(x_next):.3e}")

        x_prev = x_next

    print("\nTraining finished.")
    print("x_final (first 8 dims):", np.round(x_prev[:8], 6))


def main():
    p = argparse.ArgumentParser(description="Federated Learning + Verifiable Secure Aggregation (simulation)")
    p.add_argument("--clients", type=int, default=5, help="number of clients")
    p.add_argument("--dim", type=int, default=8, help="model/gradient dimension (toy only)")
    p.add_argument("--thr", type=int, default=3, help="Shamir threshold t")
    p.add_argument("--rounds", type=int, default=3, help="training rounds")
    p.add_argument("--seed", type=int, default=0, help="random seed for toy data")
    p.add_argument("--dataset", type=str, default="toy", choices=["toy", "mnist"], help="dataset/mode")
    p.add_argument("--mnist_npz", type=str, default="mnist.npz", help="(deprecated) legacy npz path, unused in torch mode")
    p.add_argument("--offline_uid", type=int, default=None, help="a client id to drop (simulate offline)")
    p.add_argument("--offline_from_round", type=int, default=10**9, help="start round for the offline client")
    p.add_argument("--lr", type=float, default=0.1, help="local lr (mnist)")
    p.add_argument("--local_epochs", type=int, default=1, help="local epochs per round (mnist)")
    p.add_argument("--batch", type=int, default=256, help="local batch size (mnist)")
    p.add_argument("--b_mode", type=str, default="per_round", choices=["per_round", "per_user"], help="2nd-layer mask mode (paper usually per_round)")
    p.add_argument("--auto_download", dest="auto_download", action="store_true", default=True, help="(deprecated) kept for compatibility")
    p.add_argument("--no_auto_download", dest="auto_download", action="store_false", help="(deprecated) kept for compatibility")
    p.add_argument("--weight_rule", type=str, default="inverse_l2", choices=["inverse_l2","uniform"], help="client weight rule in truth discovery")
    p.add_argument("--log_weights", action="store_true", help="print per-round weight and aggregation diagnostics")
    p.add_argument("--log_local_eval", action="store_true", help="evaluate each client's local model on the test set (mnist only)")
    p.add_argument("--faulty_uids", type=str, default="", help="comma separated client ids that behave unreliably")
    p.add_argument(
        "--faulty_mode",
        type=str,
        default="none",
        choices=["none", "sign_flip", "gaussian", "label_noise"],
        help="type of unreliable behavior to simulate",
    )
    p.add_argument("--faulty_std", type=float, default=0.5, help="stddev of Gaussian noise when faulty_mode=gaussian")
    p.add_argument("--faulty_ratio", type=float, default=0.0, help="fraction of total clients to mark as faulty")
    p.add_argument("--parallelism", type=int, default=1, help="number of threads for parallel client packaging")
    args = p.parse_args()

    # 控制加权规则（truth_discovery 使用环境变量切换）
    os.environ["WEIGHT_RULE"] = args.weight_rule

    if args.faulty_uids.strip():
        faulty_uids = [int(tok) for tok in args.faulty_uids.split(',') if tok.strip()]
    else:
        faulty_uids = []

    simulate(
        rounds=args.rounds,
        n_clients=args.clients,
        dim_m=args.dim,
        t_thr=args.thr,
        seed=args.seed,
        dataset=args.dataset,
        mnist_npz=args.mnist_npz,
        offline_uid=args.offline_uid,
        offline_from_round=args.offline_from_round,
        lr=args.lr,
        local_epochs=args.local_epochs,
        batch=args.batch,
        b_mode=args.b_mode,
        auto_download=args.auto_download,
        log_weights=args.log_weights,
        log_local_eval=args.log_local_eval,
        faulty_uids=faulty_uids,
        faulty_mode=args.faulty_mode,
        faulty_std=args.faulty_std,
        faulty_ratio=args.faulty_ratio,
        parallelism=args.parallelism,
    )


if __name__ == "__main__":
    main()
