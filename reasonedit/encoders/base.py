"""Abstract base class for standalone image encoders.

Any image encoder that can be used as the vision half of a dual embedding
must subclass ImageEncoder and implement encode(), dim, and name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Union

import torch
from PIL import Image


class ImageEncoder(ABC):
    """Abstract wrapper for any image embedding model.

    Subclasses must implement:
        encode(images) -> [B, dim] float32 tensor
        dim            -> int (embedding dimensionality)
        name           -> str (short identifier for logging/saving)
    """

    @abstractmethod
    def encode(self, images: Union[List[Image.Image], Image.Image]) -> torch.Tensor:
        """Encode one or more PIL images into a dense embedding.

        Args:
            images: Single PIL Image or list of PIL Images.

        Returns:
            torch.Tensor of shape [B, dim], dtype float32, on self.device.
        """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality (e.g. 1024 for DINOv2-large)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name used in logs and saved checkpoints."""

    # ------------------------------------------------------------------
    # Convenience helpers shared by all subclasses
    # ------------------------------------------------------------------

    def _ensure_list(self, images: Union[List[Image.Image], Image.Image]) -> List[Image.Image]:
        if isinstance(images, Image.Image):
            return [images]
        return list(images)
