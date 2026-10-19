"""Sentence embedding wrappers for the text half of a dual embedding."""

from __future__ import annotations

from typing import List, Union

import torch


class SentenceEncoder:
    """Sentence-transformers wrapper. ``encode`` returns ``[B, dim]`` float32."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        device: Union[str, torch.device] = "cpu",
        short_name: str = "sbert",
    ):
        self._model_name = model_name
        self._device = torch.device(device) if isinstance(device, str) else device
        self._short_name = short_name
        self._model = None
        self._dim = None

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        print(f"[SentenceEncoder] Loading {self._model_name} ...")
        self._model = SentenceTransformer(self._model_name, device=str(self._device))
        self._dim = int(self._model.get_sentence_embedding_dimension())
        print(f"[SentenceEncoder] Loaded {self._short_name} ({self._dim}-d)")

    def encode(self, texts: Union[List[str], str]) -> torch.Tensor:
        self._load()
        if isinstance(texts, str):
            texts = [texts]
        emb = self._model.encode(list(texts), convert_to_tensor=True, device=str(self._device))
        return emb.to(torch.float32)

    @property
    def dim(self) -> int:
        self._load()
        return self._dim

    @property
    def name(self) -> str:
        return self._short_name
