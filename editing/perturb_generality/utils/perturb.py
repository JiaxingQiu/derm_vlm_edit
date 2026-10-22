"""Light clinical-photo perturbation: rotation and color jitter. No crop, no mosaic."""

from __future__ import annotations

import hashlib
import random

import torch
import torchvision.transforms as T
from PIL import Image

IMG_AUG = T.Compose([
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
])


def stable_seed(*parts) -> int:
    text = "|".join(str(p) for p in parts)
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def perturb_image(path, seed: int) -> Image.Image:
    """Same seed returns the same augmentation."""
    img = Image.open(path).convert("RGB")
    torch_state = torch.random.get_rng_state()
    py_state = random.getstate()
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        return IMG_AUG(img)
    finally:
        torch.random.set_rng_state(torch_state)
        random.setstate(py_state)
