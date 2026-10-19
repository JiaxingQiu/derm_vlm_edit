"""Download a small COCO 2017 val subset for the auto_scaler step.

Captions come from the official annotations zip. Images are fetched
individually from the val2017 URL, so a 50-image subset does not pull
the full image archive.
"""

from __future__ import annotations

import json
import random
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Tuple

from PIL import Image

DATA_DIR = Path(__file__).resolve().parent / "data"
METADATA_FILE = DATA_DIR / "metadata.json"
CACHE_DIR = DATA_DIR / "_cache"

COCO_ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_ANN_MEMBER = "annotations/captions_val2017.json"
COCO_IMAGE_URL = "http://images.cocodataset.org/val2017/{filename}"


def load_pairs(metadata_path, image_dir) -> Tuple[List[Image.Image], List[str]]:
    """Read a ``[{filename, caption|question}, ...]`` file into PIL images and texts."""
    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    root = Path(image_dir)
    images, texts = [], []
    for entry in meta:
        path = root / entry["filename"]
        text = entry.get("caption", entry.get("question", ""))
        if path.exists() and text:
            images.append(Image.open(path).convert("RGB"))
            texts.append(text)
    return images, texts


def _captions():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / "captions_val2017.json"
    if dest.exists():
        try:
            data = json.loads(dest.read_text(encoding="utf-8"))
            if "images" in data and "annotations" in data:
                return data
        except json.JSONDecodeError:
            pass
        dest.unlink()

    zip_path = CACHE_DIR / "annotations_trainval2017.zip"
    print(f"[auto_scaler] Downloading COCO captions from {COCO_ANN_ZIP}")
    urllib.request.urlretrieve(COCO_ANN_ZIP, str(zip_path))
    with zipfile.ZipFile(zip_path) as zf:
        if COCO_ANN_MEMBER not in zf.namelist():
            raise RuntimeError(f"{COCO_ANN_MEMBER} not in {zip_path}")
        dest.write_bytes(zf.read(COCO_ANN_MEMBER))
    return json.loads(dest.read_text(encoding="utf-8"))


def download(n: int = 50, seed: int = 42, force: bool = False) -> Path:
    """Download ``n`` COCO val images plus one caption each. Idempotent."""
    if METADATA_FILE.exists() and not force:
        cached, _ = load_pairs(METADATA_FILE, DATA_DIR)
        if len(cached) >= n:
            print(f"[auto_scaler] {len(cached)} COCO images already in {DATA_DIR}")
            return METADATA_FILE

    print(f"[auto_scaler] Downloading {n} COCO val images (seed={seed})")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    coco = _captions()
    id_to_caption = {}
    for ann in coco["annotations"]:
        id_to_caption.setdefault(ann["image_id"], ann["caption"])
    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

    chosen = random.Random(seed).sample(sorted(id_to_caption), min(n, len(id_to_caption)))
    metadata = []
    for i, image_id in enumerate(chosen):
        filename = id_to_filename[image_id]
        dest = DATA_DIR / filename
        if not dest.exists():
            try:
                urllib.request.urlretrieve(COCO_IMAGE_URL.format(filename=filename), str(dest))
            except Exception as exc:
                print(f"  [{i + 1}/{n}] failed {filename}: {exc}")
                continue
        metadata.append({
            "filename": filename,
            "caption": id_to_caption[image_id],
            "coco_image_id": image_id,
        })
        if i == 0 or (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{n}] {filename}")

    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[auto_scaler] Saved {len(metadata)} images → {DATA_DIR}")
    return METADATA_FILE


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download a COCO val subset for auto_scaler")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    download(n=args.n, seed=args.seed, force=args.force)
