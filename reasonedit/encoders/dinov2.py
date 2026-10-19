"""DINOv2 image encoder wrapper.

Self-supervised ViT trained with DINOv2 — excellent visual features with no
text contamination, making it clean for the Vision Q / Language Q modularity
decomposition.

Models available on HuggingFace:
    facebook/dinov2-small   (384-d)
    facebook/dinov2-base    (768-d)
    facebook/dinov2-large   (1024-d)
    facebook/dinov2-giant   (1536-d)
"""

from __future__ import annotations

from typing import List, Union

import torch
from PIL import Image

from .base import ImageEncoder


class DINOv2Encoder(ImageEncoder):
    """DINOv2 image encoder with configurable pooling.

    Args:
        model_name: HuggingFace model id (default: dinov2-large, 1024-d).
        device: torch device.
        pool: Pooling strategy — ``"cls"`` uses the [CLS] token,
              ``"mean"`` averages all patch tokens (excluding CLS).
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        device: Union[str, torch.device] = "cpu",
        pool: str = "cls",
    ):
        self._model_name = model_name
        self._device = torch.device(device) if isinstance(device, str) else device
        self._pool = pool
        self._model = None
        self._transform = None

    # ---- lazy loading ----

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModel, AutoImageProcessor

        print(f"[DINOv2Encoder] Loading {self._model_name} ...")
        self._processor = AutoImageProcessor.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name).to(self._device)
        self._model.eval()
        self._dim = self._model.config.hidden_size
        print(f"[DINOv2Encoder] Loaded ({self._dim}-d, pool={self._pool})")

    # ---- interface ----

    @torch.no_grad()
    def encode(self, images: Union[List[Image.Image], Image.Image]) -> torch.Tensor:
        self._load()
        images = self._ensure_list(images)
        images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        outputs = self._model(**inputs)
        # outputs.last_hidden_state: [B, 1+num_patches, hidden]
        if self._pool == "cls":
            emb = outputs.last_hidden_state[:, 0, :]
        else:
            emb = outputs.last_hidden_state[:, 1:, :].mean(dim=1)
        return emb.to(torch.float32)

    @property
    def dim(self) -> int:
        self._load()
        return self._dim

    @property
    def name(self) -> str:
        short = self._model_name.split("/")[-1]
        return f"{short}_{self._pool}"
