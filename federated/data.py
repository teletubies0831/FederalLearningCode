"""Utilities for preparing datasets and simulating low-quality client data."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


@dataclass
class DatasetInfo:
    """Container describing the metadata of a dataset."""

    name: str
    num_classes: int
    base_transform: Callable


class AddGaussianNoise:
    """Additive Gaussian noise transform for tensors."""

    def __init__(self, std: float) -> None:
        if std < 0:
            raise ValueError("Standard deviation must be non-negative")
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std == 0:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, min=-1.0, max=1.0)


class RandomPixelDropout:
    """Randomly zero out pixels without rescaling like standard dropout."""

    def __init__(self, drop_prob: float) -> None:
        if not 0 <= drop_prob < 1:
            raise ValueError("drop_prob must be in [0, 1)")
        self.drop_prob = drop_prob

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0:
            return tensor
        mask = torch.rand_like(tensor)
        return tensor * (mask >= self.drop_prob).float()


class LabelNoiseTransform:
    """Randomly replace labels to simulate annotation errors."""

    def __init__(self, num_classes: int, noise_prob: float, rng: random.Random) -> None:
        if not 0 <= noise_prob <= 1:
            raise ValueError("noise_prob must be in [0, 1]")
        self.num_classes = num_classes
        self.noise_prob = noise_prob
        self.rng = rng

    def __call__(self, label: int) -> int:
        if self.noise_prob == 0 or self.rng.random() > self.noise_prob:
            return label
        new_label = self.rng.randrange(self.num_classes)
        while new_label == label and self.num_classes > 1:
            new_label = self.rng.randrange(self.num_classes)
        return new_label


class ClientDataset(Dataset):
    """Dataset wrapper applying base and optional degradation transforms."""

    def __init__(
        self,
        base_dataset: Dataset,
        indices: Sequence[int],
        base_transform: Callable,
        image_transform: Optional[Callable] = None,
        label_transform: Optional[Callable[[int], int]] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.base_transform = base_transform
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        raw_image, raw_label = self.base_dataset[self.indices[index]]
        image = self.base_transform(raw_image)
        if self.image_transform is not None:
            image = self.image_transform(image)
        label = raw_label
        if self.label_transform is not None:
            label = self.label_transform(raw_label)
        return image, label


class TransformedDataset(Dataset):
    """Applies a transform at access time to reuse a shared base dataset."""

    def __init__(self, base_dataset: Dataset, transform: Callable) -> None:
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image, label = self.base_dataset[index]
        return self.transform(image), label


def _mnist_info(root: str) -> Tuple[Dataset, Dataset, DatasetInfo]:
    mean, std = (0.1307,), (0.3081,)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train = datasets.MNIST(root=root, train=True, download=True)
    test = datasets.MNIST(root=root, train=False, download=True)
    info = DatasetInfo(name="mnist", num_classes=10, base_transform=base_transform)
    return train, test, info


def _cifar10_info(root: str) -> Tuple[Dataset, Dataset, DatasetInfo]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train = datasets.CIFAR10(root=root, train=True, download=True)
    test = datasets.CIFAR10(root=root, train=False, download=True)
    info = DatasetInfo(name="cifar10", num_classes=10, base_transform=base_transform)
    return train, test, info


def _femnist_info(root: str) -> Tuple[Dataset, Dataset, DatasetInfo]:
    # FEMNIST is derived from EMNIST; we approximate it with the balanced split.
    mean, std = (0.1307,), (0.3081,)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train = datasets.EMNIST(root=root, split="balanced", train=True, download=True)
    test = datasets.EMNIST(root=root, split="balanced", train=False, download=True)
    info = DatasetInfo(name="femnist", num_classes=len(train.classes), base_transform=base_transform)
    return train, test, info


DATASET_LOADERS: Dict[str, Callable[[str], Tuple[Dataset, Dataset, DatasetInfo]]] = {
    "mnist": _mnist_info,
    "cifar10": _cifar10_info,
    "femnist": _femnist_info,
}


def load_datasets(name: str, root: str = "data") -> Tuple[Dataset, Dataset, DatasetInfo]:
    """Load the specified dataset.

    Args:
        name: Name of the dataset to load.

    Returns:
        Tuple containing the training dataset, the test dataset and metadata.
    """

    name = name.lower()
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unsupported dataset: {name}")
    return DATASET_LOADERS[name](root)


def _split_indices(indices: Sequence[int], num_clients: int) -> List[List[int]]:
    """Split indices into approximately equal per-client subsets."""

    splits: List[List[int]] = []
    total = len(indices)
    base_size = total // num_clients
    remainder = total % num_clients
    start = 0
    for client_idx in range(num_clients):
        extra = 1 if client_idx < remainder else 0
        end = start + base_size + extra
        splits.append(list(indices[start:end]))
        start = end
    return splits


def _build_low_quality_transforms(
    methods: Sequence[Dict[str, float]],
    num_classes: int,
    rng: random.Random,
) -> Tuple[Optional[Callable], Optional[Callable]]:
    """Create transforms that simulate low-quality data.

    The supported methods are:
        - label_noise: add label noise with probability ``prob``.
        - gaussian_noise: add Gaussian noise with standard deviation ``std``.
        - gaussian_blur: blur images with ``sigma`` (and optional ``kernel_size``).
        - pixel_dropout: randomly zero pixels with probability ``drop_prob``.
    """

    image_transforms: List[Callable] = []
    label_transform: Optional[Callable[[int], int]] = None

    for method in methods:
        method_type = method.get("type")
        if method_type == "label_noise":
            prob = float(method.get("prob", 0.0))
            label_transform = LabelNoiseTransform(num_classes, prob, rng)
        elif method_type == "gaussian_noise":
            std = float(method.get("std", 0.0))
            image_transforms.append(AddGaussianNoise(std))
        elif method_type == "gaussian_blur":
            sigma = float(method.get("sigma", 1.0))
            kernel = int(method.get("kernel_size", 5))
            image_transforms.append(transforms.GaussianBlur(kernel, sigma=(sigma, sigma)))
        elif method_type == "pixel_dropout":
            drop_prob = float(method.get("drop_prob", 0.0))
            image_transforms.append(RandomPixelDropout(drop_prob))
        elif method_type is None:
            raise ValueError("Low-quality method entries must define a 'type' key")
        else:
            raise ValueError(f"Unsupported low-quality method: {method_type}")

    image_transform: Optional[Callable]
    if image_transforms:
        image_transform = transforms.Compose(image_transforms)
    else:
        image_transform = None

    return image_transform, label_transform


def create_data_loaders(
    dataset_name: str,
    num_clients: int,
    batch_size: int,
    low_quality_config: Optional[Dict[str, object]] = None,
    seed: int = 42,
    data_dir: str = "data",
) -> Tuple[List[DataLoader], DataLoader, DatasetInfo]:
    """Prepare federated data loaders with optional low-quality clients.

    Args:
        dataset_name: Name of the dataset to load (mnist, cifar10 or femnist).
        num_clients: Number of simulated clients.
        batch_size: Local batch size for each client.
        low_quality_config: Optional configuration describing degraded clients.
        seed: Deterministic seed for client sampling and label noise.
        data_dir: Directory where torchvision datasets will be stored.

    Returns:
        A tuple with client loaders, a shared test loader and dataset metadata.
    """

    if num_clients <= 0:
        raise ValueError("num_clients must be positive")

    train_dataset, test_dataset, info = load_datasets(dataset_name, data_dir)
    rng = random.Random(seed)
    indices = list(range(len(train_dataset)))
    rng.shuffle(indices)  # 打乱整体样本次序，确保划分公平
    client_splits = _split_indices(indices, num_clients)

    low_quality_clients: Sequence[int] = []
    methods: Sequence[Dict[str, float]] = []
    if low_quality_config:
        fraction = float(low_quality_config.get("fraction", 0.0))
        raw_methods = low_quality_config.get("methods", [])
        if raw_methods:
            methods = list(raw_methods)  # type: ignore[arg-type]
        if methods and fraction > 0:
            count = max(1, int(round(num_clients * fraction)))
            count = min(count, num_clients)
            low_quality_clients = rng.sample(range(num_clients), count)
        else:
            low_quality_clients = []
    client_loaders: List[DataLoader] = []
    low_quality_set = set(low_quality_clients)

    for client_idx, client_indices in enumerate(client_splits):
        image_transform: Optional[Callable] = None
        label_transform: Optional[Callable] = None
        if low_quality_config and client_idx in low_quality_set:
            # 为指定客户端构建低质量数据增强（图像与标签）
            image_transform, label_transform = _build_low_quality_transforms(methods, info.num_classes, rng)
        client_dataset = ClientDataset(
            base_dataset=train_dataset,
            indices=client_indices,
            base_transform=info.base_transform,
            image_transform=image_transform,
            label_transform=label_transform,
        )
        loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)

    test_loader = DataLoader(TransformedDataset(test_dataset, info.base_transform), batch_size=batch_size)
    return client_loaders, test_loader, info
