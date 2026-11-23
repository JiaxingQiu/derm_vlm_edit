"""Download a small Fitzpatrick17k subset for the dermatology scaler.

Fifty clinical images, each paired with its diagnosis label. The metadata
matches the COCO loader: ``[{filename, caption}, ...]``. A second call keeps
a complete subset in place.

    python -m reasonedit.auto_scaler.download_derm
"""

from __future__ import annotations

import csv
import io
import random
import urllib.request
from pathlib import Path

from PIL import Image

DATA_DIR = Path(__file__).resolve().parent / "data_derm"
METADATA_FILE = DATA_DIR / "metadata.json"
CSV_URL = "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv"
DEFAULT_N = 50
DEFAULT_SEED = 42
_TIMEOUT = 20


def _existing(n: int) -> bool:
    if not METADATA_FILE.exists():
        return False
    from .download import load_pairs

    images, _ = load_pairs(METADATA_FILE, DATA_DIR)
    return len(images) >= n


def _fetch_csv() -> list:
    print(f"[auto_scaler] reading {CSV_URL}")
    request = urllib.request.Request(CSV_URL, headers={"User-Agent": "derm-vlm-edit"})
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        text = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _save_image(url: str, dest: Path) -> bool:
    request = urllib.request.Request(url, headers={"User-Agent": "derm-vlm-edit"})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            raw = response.read()
        image = Image.open(io.BytesIO(raw))
        image = image.convert("RGB")
        dest.parent.mkdir(parents=True, exist_ok=True)
        image.save(dest, format="JPEG")
        return True
    except Exception as exc:
        print(f"  skip {url}: {exc}")
        return False


def download(n: int = DEFAULT_N, seed: int = DEFAULT_SEED, force: bool = False) -> Path:
    """Save ``n`` diagnosis-labeled images under ``data_derm``. Idempotent."""
    if not force and _existing(n):
        print(f"[auto_scaler] {n} dermatology images already in {DATA_DIR}")
        return METADATA_FILE

    rows = _fetch_csv()
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata = []
    for index in order:
        if len(metadata) >= n:
            break
        row = rows[index]
        url = (row.get("url") or "").strip()
        label = (row.get("label") or "").strip()
        digest = (row.get("md5hash") or "").strip()
        if not url or not label or not digest:
            continue
        filename = f"{digest}.jpg"
        dest = DATA_DIR / filename
        if dest.exists() or _save_image(url, dest):
            if not dest.exists():
                continue
            metadata.append({"filename": filename, "caption": label, "url": url})
            print(f"  [{len(metadata)}/{n}] {label}")
    if len(metadata) < n:
        raise RuntimeError(f"Need {n} dermatology images, saved {len(metadata)}")
    import json
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[auto_scaler] Saved {len(metadata)} images → {DATA_DIR}")
    return METADATA_FILE


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download a Fitzpatrick17k subset for auto_scaler")
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    download(n=args.n, seed=args.seed, force=args.force)
