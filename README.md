# 联邦学习低质量数据实验

该仓库提供了一个简洁的联邦学习实验脚本，用于在 **MNIST**、**CIFAR-10** 和 **FEMNIST**（使用 `torchvision` 中的 EMNIST balanced 切分近似实现）数据集上复现论文 *Verifiable and Secure Aggregation for FL with Low-quality Data* 中的实验思路。脚本完全基于 `PyTorch` 与 `torchvision`，实现了 FedAvg 聚合，并支持论文中常见的几种低质量数据模拟方式：

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

运行 MNIST 基线（无低质量客户端）：
```bash
python run_experiment.py --dataset mnist --rounds 5 --num-clients 10 --clients-per-round 5
```

在 CIFAR-10 上模拟 30% 低质量客户端，包含标签噪声和特征噪声：
```bash
python run_experiment.py \
  --dataset cifar10 \
  --num-clients 20 \
  --clients-per-round 10 \
  --rounds 20 \
  --local-epochs 2 \
  --batch-size 64 \
  --low-quality-fraction 0.3 \
  --label-noise 0.2 \
  --gaussian-noise-std 0.1
```

在 FEMNIST 上模拟模糊和像素丢弃：
```bash
python run_experiment.py \
  --dataset femnist \
  --num-clients 30 \
  --clients-per-round 15 \
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
| `--clients-per-round` | 每轮参与训练的客户端数量，缺省时等于 `--num-clients` |
| `--rounds` | 联邦训练轮数 |
| `--local-epochs` | 每个客户端本地训练的 epoch 数 |
| `--batch-size` | 客户端本地训练的批大小 |
| `--lr` / `--weight-decay` | 本地优化器（SGD）的学习率与权重衰减 |
| `--low-quality-fraction` | 被设定为低质量客户端的比例（0~1） |
| `--label-noise` | 标签翻转概率 |
| `--gaussian-noise-std` | 添加到像素的高斯噪声标准差 |
| `--gaussian-blur-sigma` | 高斯模糊的 sigma，核大小固定为 5 |
| `--pixel-dropout` | 像素随机置零的概率 |
| `--seed` | 随机种子，确保可复现 |

## 结果输出

脚本会在每轮训练后输出测试集损失与精度，并在训练结束时打印最终指标。可根据 `TrainingHistory` 数据结构扩展保存或可视化逻辑。

## 结构说明

```
.
├── federated
│   ├── data.py        # 数据加载与低质量模拟
│   ├── models.py      # 针对不同数据集的模型定义
│   └── trainer.py     # FedAvg 训练流程
├── run_experiment.py  # 命令行入口
├── requirements.txt   # 依赖列表
└── Verifiable...pdf   # 论文原文
```

欢迎根据实验需求调整轮数、客户端数量或模型结构。若需进一步扩展低质量数据策略，可在 `federated/data.py` 中新增对应的变换组件。祝实验顺利！
