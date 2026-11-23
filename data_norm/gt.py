"""Load MIDAS ground-truth y16 keyed by annotation case_id."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SHARED = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

import config as C  # noqa: E402

DATA_SHARE = C.MIDAS_DICT.resolve().parent


def load() -> dict[str, str | None]:
    """Return {case_id: y16}. None when the lesion has conflicting labels."""
    midas = pd.read_parquet(DATA_SHARE / "midas_share.parquet", columns=["lesion_id", "y16"])
    mapping = pd.read_parquet(DATA_SHARE / "case_mapping.parquet", columns=["case_id", "lesion_id"])
    per_lesion: dict[str, set] = {}
    for lid, y16 in zip(midas["lesion_id"], midas["y16"]):
        per_lesion.setdefault(lid, set()).add(y16)
    resolved = {k: next(iter(v)) if len(v) == 1 else None for k, v in per_lesion.items()}
    return {cid: resolved.get(lid) for cid, lid in zip(mapping["case_id"], mapping["lesion_id"])}
