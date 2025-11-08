# 联邦学习低质量数据实验

该仓库提供了一个简洁的联邦学习实验脚本，用于在 **MNIST**、**CIFAR-10** 和 **FEMNIST**（使用 `torchvision` 中的 EMNIST balanced 切分近似实现）数据集上复现论文 *Verifiable and Secure Aggregation for FL with Low-quality Data* 中的实验思路。脚本完全基于 `PyTorch` 与 `torchvision`，实现了**所有客户端同时参与**的安全聚合流程，并支持论文中常见的几种低质量数据模拟方式：

- **标签噪声**：按给定概率随机翻转标签；
- **特征噪声**：向像素添加高斯噪声；
- **图像模糊**：使用高斯模糊降低图像清晰度；
- **像素丢弃**：随机将像素置零以模拟遮挡或采集缺陷。

> 以上策略可组合使用，从而模拟多种低质量数据场景。

## 环境准备

1. 创建虚拟环境（可选）：
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows 使用 .venv\\Scripts\\activate
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
   若需要 GPU 版本的 PyTorch，请参考 [PyTorch 官网](https://pytorch.org/) 的指引安装匹配版本。

首次运行脚本时会自动下载所需数据集到 `--data-dir` 指定的目录（默认 `./data`）。

## 快速开始

运行 MNIST 基线（无低质量客户端与掉线）：
```bash
python run_experiment.py --dataset mnist --rounds 5 --num-clients 10
```

在 CIFAR-10 上模拟 30% 低质量客户端，并允许 10% 掉线：
```bash
python run_experiment.py \
  --dataset cifar10 \
  --num-clients 20 \
  --rounds 20 \
  --local-epochs 2 \
  --batch-size 64 \
  --low-quality-fraction 0.3 \
  --label-noise 0.2 \
  --gaussian-noise-std 0.1 \
  --dropout-rate 0.1 \
  --dropout-tolerance 2
```

在 FEMNIST 上模拟模糊和像素丢弃：
```bash
python run_experiment.py \
  --dataset femnist \
  --num-clients 30 \
  --rounds 15 \
  --gaussian-blur-sigma 1.2 \
  --pixel-dropout 0.1 \
  --low-quality-fraction 0.4
```

## 参数说明

| 参数 | 含义 |
|------|------|
| `--dataset` | 数据集名称，可选 `mnist`、`cifar10`、`femnist` |
| `--data-dir` | 数据下载与缓存目录 |
| `--num-clients` | 模拟的总客户端数量 |
| `--rounds` | 联邦训练轮数 |
| `--local-epochs` | 每个客户端本地训练的 epoch 数 |
| `--batch-size` | 客户端本地训练的批大小 |
| `--lr` / `--weight-decay` | 本地优化器（SGD）的学习率与权重衰减 |
| `--dropout-rate` | 每轮模拟客户端掉线的概率 |
| `--dropout-tolerance` | 安全聚合所能容忍的最大掉线数量 |
| `--aggregator` | 聚合策略，支持 `truth_discovery`、`esfl`/`fedavg`、`ppfdl` |
| `--low-quality-fraction` | 被设定为低质量客户端的比例（0~1） |
| `--label-noise` | 标签翻转概率 |
| `--gaussian-noise-std` | 添加到像素的高斯噪声标准差 |
| `--gaussian-blur-sigma` | 高斯模糊的 sigma，核大小固定为 5 |
| `--pixel-dropout` | 像素随机置零的概率 |
| `--seed` | 随机种子，确保可复现 |

> **说明 1**：当 `dropout_rate` 产生的候选掉线客户端数量大于 `dropout_tolerance` 时，代码会自动裁剪掉线集合，仅保留 `dropout_tolerance`
> 个客户端真正掉线，以符合协议中的容错约束。若希望模拟更多掉线，请同步增大 `dropout_tolerance`。
>
> **说明 2**：`--low-quality-fraction` 按客户端层面计算比例。例如 `--num-clients 20 --low-quality-fraction 0.3` 会随机挑选 6 个客户端，这些
> 客户端在整个训练过程中始终使用退化后的数据集，从而模拟论文中的低质量参与方。

## 结果输出

脚本会在每轮训练后输出测试集损失与精度，并在训练结束时打印最终指标。可根据 `TrainingHistory` 数据结构扩展保存或可视化逻辑。

## 结构说明

```
.
├── federated
│   ├── aggregation.py  # 安全聚合与真值发现流程
│   ├── data.py         # 数据加载与低质量模拟
│   ├── models.py       # 针对不同数据集的模型定义
│   └── trainer.py      # 遵循论文协议的训练流程
├── run_experiment.py   # 命令行入口
├── requirements.txt    # 依赖列表
└── Verifiable...pdf    # 论文原文
```

### 协议实现概述

- **安全聚合**：`SecureAggregationController` 对所有客户端的模型更新进行加权求和，并对每个客户端的状态字典执行 SHA256 承诺校验，确保聚合器只接触经过验证的更新结果。
- **容忍掉线**：当客户端在某轮训练中掉线时，只要仍有不少于 `num_clients - dropout_tolerance` 的客户端提交更新，聚合仍会继续执行。
- **真值发现**：在聚合前会对客户端更新执行基于残差的迭代加权（truth discovery），以降低低质量或恶意更新的影响。

若需要在真实部署中获得更强的安全保证，可考虑集成以下项目提供的可验证安全聚合实现：

- [FATE](https://github.com/FederatedAI/FATE) 的安全聚合模块，支持多种密码学协议；
- [OpenMined PySyft](https://github.com/OpenMined/PySyft) 提供的 SMPC 工具；
- [Flower](https://flower.dev/) 的 `flower_secure_aggregation` 示例，可直接在联邦学习框架中启用安全聚合。

欢迎根据实验需求调整轮数、客户端数量或模型结构。若需进一步扩展低质量数据策略，可在 `federated/data.py` 中新增对应的变换组件；如需替换更高强度的安全聚合协议，可在 `federated/aggregation.py` 中自定义控制器。祝实验顺利！