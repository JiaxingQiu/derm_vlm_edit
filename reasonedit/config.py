"""Tiny config loader: model YAML + a few globals. No Hydra, no editor presets."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

PKG = Path(__file__).resolve().parent
CONFIGS = PKG / "configs"

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
                max_new_tokens: int = 32, seed: int = 42):
    """Build the NestedConfig-shaped object ReasonEdit / VQAModel expect.

    Editor knobs are left empty so ReasonEdit.__init__ defaults apply
    (answer-only IKE: no merge, no hubness, cap_keys=1).
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

    return SimpleNamespace(
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        seed=seed,
        device=device,
        model=_ns(model),
        editor=SimpleNamespace(_name="reasonedit"),
        experiment=SimpleNamespace(task="qa", dataset_name="derm", pred_by="label_text"),
    )
