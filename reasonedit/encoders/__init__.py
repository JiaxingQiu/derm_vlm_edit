"""Standalone encoders for dual-embedding codebook keys.

    from reasonedit.encoders import get_image_encoder, get_text_encoder

    image_enc = get_image_encoder("dinov2", device="cuda:0", pool="cls")
    text_enc = get_text_encoder("sbert", device="cuda:0")
"""

from __future__ import annotations

import inspect

from .base import ImageEncoder
from .biomedclip import BiomedCLIPEncoder
from .dinov2 import DINOv2Encoder
from .text import SentenceEncoder

# short name -> (class, default kwargs)
_IMAGE = {
    "dinov2": (DINOv2Encoder, {}),
    "dinov2-small": (DINOv2Encoder, {"model_name": "facebook/dinov2-small"}),
    "dinov2-base": (DINOv2Encoder, {"model_name": "facebook/dinov2-base"}),
    "dinov2-large": (DINOv2Encoder, {"model_name": "facebook/dinov2-large"}),
    "dinov2-giant": (DINOv2Encoder, {"model_name": "facebook/dinov2-giant"}),
    "biomedclip": (BiomedCLIPEncoder, {}),
}

_TEXT = {
    "sbert": (SentenceEncoder, {
        "model_name": "sentence-transformers/paraphrase-mpnet-base-v2",
        "short_name": "sbert",
    }),
    "minilm": (SentenceEncoder, {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "short_name": "minilm",
    }),
}


def _construct(cls, defaults: dict, device, kwargs: dict):
    """Build ``cls``, forwarding only parameters its constructor accepts."""
    raw = {**defaults, "device": device, **kwargs}
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return cls(**raw)
    accepted = set(params) - {"self"}
    dropped = sorted(k for k in raw if k not in accepted)
    if dropped:
        print(f"[encoders] {cls.__name__} ignores {dropped}")
    return cls(**{k: v for k, v in raw.items() if k in accepted})


def _lookup(registry: dict, name: str, kind: str):
    key = name.lower().strip()
    if key not in registry:
        raise ValueError(
            f"Unknown {kind} encoder {name!r}. Available: {sorted(registry)}"
        )
    return registry[key]


def get_image_encoder(name: str, device="cpu", **kwargs) -> ImageEncoder:
    """Create an image encoder by short name. Unknown kwargs are ignored."""
    cls, defaults = _lookup(_IMAGE, name, "image")
    return _construct(cls, defaults, device, kwargs)


def get_text_encoder(name: str, device="cpu", **kwargs) -> SentenceEncoder:
    """Create a text encoder by short name. Unknown kwargs are ignored."""
    cls, defaults = _lookup(_TEXT, name, "text")
    return _construct(cls, defaults, device, kwargs)


def encoder_tag(name: str, kind: str = "image", **kwargs) -> str:
    """Resolved tag used in the balancing-weight key. Does not load weights."""
    if kind == "text":
        return get_text_encoder(name, device="cpu", **kwargs).name
    return get_image_encoder(name, device="cpu", **kwargs).name


__all__ = [
    "ImageEncoder",
    "DINOv2Encoder",
    "BiomedCLIPEncoder",
    "SentenceEncoder",
    "get_image_encoder",
    "get_text_encoder",
    "encoder_tag",
]
