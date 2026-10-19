"""Recorded balancing weights, one entry per encoder/dataset combo.

The file is ``reasonedit/auto_scaler/scalers.yaml``. A combo is

    {image tag}__{text tag}__{dataset}

An existing entry is left in place unless the caller overwrites it.
"""

from __future__ import annotations

from pathlib import Path

import yaml

SCALERS_PATH = Path(__file__).resolve().parent / "scalers.yaml"


def combo_key(image_tag: str, text_tag: str, dataset: str) -> str:
    return f"{image_tag}__{text_tag}__{dataset}"


def load_scalers() -> dict:
    if not SCALERS_PATH.exists():
        return {}
    return yaml.safe_load(SCALERS_PATH.read_text(encoding="utf-8")) or {}


def lookup_scaler(image_tag: str, text_tag: str, dataset: str):
    entry = load_scalers().get(combo_key(image_tag, text_tag, dataset))
    if not entry or entry.get("lang_scaler") is None:
        return None
    return float(entry["lang_scaler"])


def save_scaler(image_tag: str, text_tag: str, dataset: str, lang_scaler: float, **meta) -> str:
    """Write one combo. Other combos in the file are preserved."""
    data = load_scalers()
    key = combo_key(image_tag, text_tag, dataset)
    data[key] = {
        "lang_scaler": float(lang_scaler),
        "image_encoder": image_tag,
        "text_encoder": text_tag,
        "dataset": dataset,
        **meta,
    }
    SCALERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCALERS_PATH.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    return key


def require_scaler(image_tag: str, text_tag: str, dataset: str, command: str) -> float:
    """Return the recorded weight, or raise with the command that records it."""
    value = lookup_scaler(image_tag, text_tag, dataset)
    if value is None:
        key = combo_key(image_tag, text_tag, dataset)
        raise RuntimeError(
            f"No balancing weight recorded for {key}.\nRecord it with:\n  {command}"
        )
    return value
