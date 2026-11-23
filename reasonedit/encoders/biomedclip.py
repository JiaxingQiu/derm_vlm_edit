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


def _vision_dim(model) -> int:
    """Embedding width of an open_clip vision tower.

    A custom CLIP tower stores ``output_dim``. BiomedCLIP's tower is a timm
    model, whose projected width lives on the linear head instead.
    """
    visual = model.visual
    if hasattr(visual, "output_dim"):
        return int(visual.output_dim)
    head = getattr(visual, "head", None)
    if head is not None:
        for module in reversed(list(head.modules())):
            out = getattr(module, "out_features", None)
            if out:
                return int(out)
    text_proj = getattr(model, "text_projection", None)
    shape = getattr(text_proj, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[-1])
    out = getattr(text_proj, "out_features", None)
    if out:
        return int(out)
    raise AttributeError(
        f"{type(visual).__name__} has no recorded embedding width"
    )


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
        pool: str = "cls",
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
        self._dim = _vision_dim(model)
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
