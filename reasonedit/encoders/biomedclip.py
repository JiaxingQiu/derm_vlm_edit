"""BiomedCLIP image encoder wrapper.

Contrastive vision-language model trained on PubMed biomedical image-text
pairs.  The image tower provides domain-specific visual embeddings that
should transfer well to dermatology images.

HuggingFace id:
    microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224   (512-d)
"""

from __future__ import annotations

from typing import List, Union

import torch
from PIL import Image

from .base import ImageEncoder


class BiomedCLIPEncoder(ImageEncoder):
    """BiomedCLIP image encoder (vision tower only).

    Uses the ``open_clip`` backend that BiomedCLIP was released with.

    Args:
        model_name: HuggingFace model id.
        device: torch device.
    """

    def __init__(
        self,
        model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        device: Union[str, torch.device] = "cpu",
    ):
        self._model_name = model_name
        self._device = torch.device(device) if isinstance(device, str) else device
        self._model = None

    # ---- lazy loading ----

    def _load(self):
        if self._model is not None:
            return
        import open_clip

        print(f"[BiomedCLIPEncoder] Loading {self._model_name} ...")
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            "hf-hub:" + self._model_name, device=self._device,
        )
        self._model = model
        self._preprocess = preprocess_val
        self._model.eval()
        self._dim = model.visual.output_dim
        print(f"[BiomedCLIPEncoder] Loaded ({self._dim}-d)")

    # ---- interface ----

    @torch.no_grad()
    def encode(self, images: Union[List[Image.Image], Image.Image]) -> torch.Tensor:
        self._load()
        images = self._ensure_list(images)
        images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
        pixel_tensors = torch.stack([self._preprocess(img) for img in images]).to(self._device)
        emb = self._model.encode_image(pixel_tensors)
        return emb.to(torch.float32)

    @property
    def dim(self) -> int:
        self._load()
        return self._dim

    @property
    def name(self) -> str:
        return "biomedclip"
