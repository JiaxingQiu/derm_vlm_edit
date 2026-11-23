"""Retrieval-image switch for the derm edit runs.

``combined`` leaves each row unchanged, so the codebook key is the same
combined canvas the VLM sees. ``photo`` and ``dscope`` set ``key_image`` to
the sibling file already stored under ``data/images``. Generation keeps
``image``.
"""

from __future__ import annotations

from pathlib import Path

import config as C

MODES = ("combined", "photo", "dscope")
DEFAULT_IMAGE_ENCODER = "dinov2"
DEFAULT_SCALER_DATASET = "coco"


def results_tag(embed_mode, reject_threshold, retrieval_image="combined",
                image_encoder=DEFAULT_IMAGE_ENCODER,
                scaler_dataset=DEFAULT_SCALER_DATASET) -> str:
    """Folder name for one run.

    ``dinov2`` on ``coco`` keeps ``standalone_dual_10`` and
    ``standalone_dual_10_photo``. Another encoder adds its short name.
    Another scaler dataset adds its name after that:
    ``standalone_dual_10_photo_panderm_derm``.
    """
    base = f"{embed_mode}_{float(reject_threshold):g}"
    mode = _mode(retrieval_image)
    if mode != "combined":
        base = f"{base}_{mode}"
    encoder = _token(image_encoder, DEFAULT_IMAGE_ENCODER, "image encoder")
    if encoder != DEFAULT_IMAGE_ENCODER:
        base = f"{base}_{encoder}"
    dataset = _token(scaler_dataset, DEFAULT_SCALER_DATASET, "scaler dataset")
    if dataset != DEFAULT_SCALER_DATASET:
        base = f"{base}_{dataset}"
    return base


def key_path(case_id: str, mode: str) -> Path:
    """Sibling file for a ``{id}_combined`` case. Combined returns the case image."""
    mode = _mode(mode)
    if mode == "combined":
        return C.image_path_for(case_id)
    stem = str(case_id).rsplit("_", 1)[0]
    return C.IMAGE_ROOT / f"{stem}_{mode}.jpg"


def attach(rows, mode="combined"):
    """Stamp ``key_image`` for photo and dscope. Combined rows stay as they are."""
    mode = _mode(mode)
    if mode == "combined":
        return rows
    for row in rows:
        row["key_image"] = str(key_path(row["case_id"], mode))
    return rows


def require_present(rows, label="rows"):
    """Fail before any model load when a retrieval file is missing."""
    missing = []
    for row in rows:
        raw = row.get("key_image") or row.get("image")
        if not raw or not Path(raw).exists():
            missing.append(f"{row.get('case_id')}: {raw}")
    if not missing:
        return
    shown = "\n".join(missing[:20])
    extra = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
    raise FileNotFoundError(
        f"missing retrieval image for {label} ({len(missing)} rows):\n{shown}{extra}"
    )


def _mode(mode: str) -> str:
    mode = (mode or "combined").strip()
    if mode not in MODES:
        raise ValueError(f"retrieval image must be one of {MODES}, got {mode!r}")
    return mode


def _token(name: str, default: str, kind: str) -> str:
    name = (name or default).strip().lower()
    if not name or not all(c.isalnum() or c in "._-" for c in name):
        raise ValueError(f"{kind} must be a short name, got {name!r}")
    return name
