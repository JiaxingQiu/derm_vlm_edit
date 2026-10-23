"""Put shared code on sys.path. Call once from a notebook before other imports."""

from __future__ import annotations

import sys
from pathlib import Path

OOD_ROOT = Path(__file__).resolve().parent.parent


def bootstrap() -> Path:
    shared = OOD_ROOT.parent / "shared"
    reliability = OOD_ROOT.parent / "reliability"
    for path in (shared, reliability):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import config as C

    root = str(C.DERM_EDIT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return OOD_ROOT


def results_dir(root, split, embed_mode) -> Path:
    """results/{embed_mode}_{threshold}/{split}. The tag includes the threshold."""
    mode = (embed_mode or "dual_sbert").strip()
    return Path(root) / "results" / mode / split
