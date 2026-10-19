"""Tiny config loader: model YAML + a few globals. No Hydra, no editor presets."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

CONFIGS = Path(__file__).resolve().parent

SHORT_TO_YAML = {
    "medgemma": "medgemma.yaml",
    "dermatollama": "dermatollama.yaml",
    "dermato_llama": "dermatollama.yaml",
}


def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    return obj


def load_config(model_short: str, device=None, batch_size: int = 1,
                max_new_tokens: int = 32, seed: int = 42,
                mode: str = None, image_encoder: str = None,
                image_encoder_pool: str = "cls", text_encoder: str = None,
                scaler_dataset: str = None, lang_scaler: float = None):
    """Build the NestedConfig-shaped object ReasonEdit / VQAModel expect.

    Editor knobs are left empty so ReasonEdit.__init__ defaults apply
    (answer-only IKE: no merge, no hubness, cap_keys=1).

    Args:
        model_short: ``"medgemma"`` or ``"dermatollama"``.
        device: torch device (default: auto).
        batch_size: inference batch size.
        max_new_tokens: generation length.
        seed: random seed.
        mode: Embedding mode override — ``"standalone_dual"``,
              ``"dual_sbert"``, ``"vision"``, etc.  *None* keeps the
              model YAML default (which is ``dual_sbert``).
        image_encoder: Image encoder short name (e.g. ``"dinov2"``,
              ``"biomedclip"``).  Only used when *mode* is
              ``"standalone_dual"``.
        image_encoder_pool: Pooling strategy for the image encoder
              (``"cls"`` or ``"mean"``).
        text_encoder: Text encoder short name (``"sbert"`` or ``"minilm"``).
        scaler_dataset: Dataset name the balancing weight was recorded on.
        lang_scaler: Explicit balancing weight. Skips the recorded combo.
    """
    key = str(model_short).lower()
    yaml_name = SHORT_TO_YAML.get(key)
    if yaml_name is None:
        raise ValueError(f"Unknown model_short={model_short!r}. "
                         f"Expected one of {sorted(SHORT_TO_YAML)}")
    path = CONFIGS / yaml_name
    with open(path, encoding="utf-8") as f:
        model = yaml.safe_load(f) or {}

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    editor_kwargs = {"_name": "reasonedit"}
    if mode is not None:
        editor_kwargs["mode"] = mode
    if image_encoder is not None:
        editor_kwargs["image_encoder"] = image_encoder
    if image_encoder_pool is not None:
        editor_kwargs["image_encoder_pool"] = image_encoder_pool
    if text_encoder is not None:
        editor_kwargs["text_encoder"] = text_encoder
    if scaler_dataset is not None:
        editor_kwargs["scaler_dataset"] = scaler_dataset
    if lang_scaler is not None:
        editor_kwargs["lang_scaler"] = lang_scaler

    return SimpleNamespace(
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        seed=seed,
        device=device,
        model=_ns(model),
        editor=SimpleNamespace(**editor_kwargs),
        experiment=SimpleNamespace(task="qa", dataset_name="derm", pred_by="label_text"),
    )
