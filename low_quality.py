"""Utilities for constructing low-quality client datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class LowQualityPolicy:
    """Describe how unreliable clients manipulate their local data."""

    family: str
    severity: float = 0.3
    num_classes: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        self.family = self.family.lower()
        self._rng = np.random.RandomState(self.seed)

    def spawn(self, new_seed: int) -> "LowQualityPolicy":
        """Return an independent policy with the same configuration."""

        return LowQualityPolicy(
            family=self.family,
            severity=self.severity,
            num_classes=self.num_classes,
            seed=new_seed,
        )

    def apply(self, image: Image.Image, label: int) -> Tuple[Image.Image, int]:
        """Mutate ``(image, label)`` in-place according to the policy."""

        handler = {
            "symmetric": self._apply_symmetric_noise,
            "symmetric_noise": self._apply_symmetric_noise,
            "class_dependent": self._apply_class_dependent_noise,
            "robust": self._apply_robust_corruption,
            "blur": self._apply_robust_corruption,
            "erasing": self._apply_erasing,
        }.get(self.family)
        if handler is None:
            raise ValueError(f"Unsupported low-quality family: {self.family}")
        return handler(image, int(label))

    # --- family handlers -------------------------------------------------

    def _apply_symmetric_noise(self, image: Image.Image, label: int) -> Tuple[Image.Image, int]:
        if self._rng.rand() < float(self.severity):
            label = int(self._rng.randint(self.num_classes))
        return image, label

    def _apply_class_dependent_noise(self, image: Image.Image, label: int) -> Tuple[Image.Image, int]:
        if self._rng.rand() < float(self.severity):
            label = (label + 1) % self.num_classes
        return image, label

    def _apply_robust_corruption(self, image: Image.Image, label: int) -> Tuple[Image.Image, int]:
        radius = 0.5 + 2.5 * float(self.severity)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        arr = np.array(image).astype(np.float32)
        noise_scale = 25.0 * float(self.severity)
        if noise_scale > 0:
            noise = self._rng.normal(loc=0.0, scale=noise_scale, size=arr.shape)
            arr = np.clip(arr + noise, 0.0, 255.0)
        return Image.fromarray(arr.astype(np.uint8)), label

    def _apply_erasing(self, image: Image.Image, label: int) -> Tuple[Image.Image, int]:
        arr = np.array(image)
        h, w, _ = arr.shape
        patch_ratio = 0.05 + 0.25 * float(self.severity)
        patch_h = max(1, int(round(h * patch_ratio)))
        patch_w = max(1, int(round(w * patch_ratio)))
        top = int(self._rng.randint(0, max(1, h - patch_h + 1)))
        left = int(self._rng.randint(0, max(1, w - patch_w + 1)))
        arr[top : top + patch_h, left : left + patch_w, :] = 0
        return Image.fromarray(arr.astype(np.uint8)), label
