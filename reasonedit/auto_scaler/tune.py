"""Record the balancing weight for one image encoder, text encoder, and dataset.

An existing entry in ``auto_scaler/scalers.yaml`` is reused unless
``overwrite`` is set. The command line lives in ``reasonedit.run.auto_scaler``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from .download import DATA_DIR, METADATA_FILE, download, load_pairs
from .store import combo_key, lookup_scaler, save_scaler

PKG_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PKG_DIR / "results"


def _load_dataset(dataset: str, images_dir: Optional[str], captions_file: Optional[str], n_download: int):
    if images_dir or captions_file:
        if not images_dir or not captions_file:
            raise ValueError("Custom data needs both --images_dir and --captions_file")
        return load_pairs(captions_file, images_dir)
    if dataset != "coco":
        raise ValueError(
            f"Dataset {dataset!r} is not bundled. Pass --images_dir and --captions_file, "
            "or use --dataset coco."
        )
    download(n=n_download)
    return load_pairs(METADATA_FILE, DATA_DIR)


def tune_scaler(
    image_encoder: str = "dinov2",
    text_encoder: str = "sbert",
    dataset: str = "coco",
    device: str = "cuda:0",
    n_samples: int = 10,
    n_runs: int = 10,
    n_download: int = 50,
    images_dir: Optional[str] = None,
    captions_file: Optional[str] = None,
    image_encoder_pool: str = "cls",
    overwrite: bool = False,
) -> float:
    """Return the balancing weight for this combo, running the sweep only when needed."""
    from .scaler import StandaloneAutoScaler
    from ..encoders import encoder_tag, get_image_encoder, get_text_encoder

    image_tag = encoder_tag(image_encoder, kind="image", pool=image_encoder_pool)
    text_tag = encoder_tag(text_encoder, kind="text")
    key = combo_key(image_tag, text_tag, dataset)

    existing = lookup_scaler(image_tag, text_tag, dataset)
    if existing is not None and not overwrite:
        print(f"[auto_scaler] {key} already recorded: lang_scaler={existing}")
        return existing

    images, texts = _load_dataset(dataset, images_dir, captions_file, n_download)
    if len(images) < n_samples:
        raise RuntimeError(
            f"Need at least {n_samples} pairs, found {len(images)}."
        )
    print(f"[auto_scaler] {len(images)} pairs for {key}")

    img_enc = get_image_encoder(image_encoder, device=device, pool=image_encoder_pool)
    txt_enc = get_text_encoder(text_encoder, device=device)
    searcher = StandaloneAutoScaler(
        img_enc, device=device, n_samples=n_samples, text_encoder=txt_enc,
    )

    out_dir = str(RESULTS_DIR / key)
    for run_id in range(n_runs):
        if searcher.load_results(run_id=run_id, out_dir=out_dir) is not None and not overwrite:
            print(f"  Run {run_id + 1}/{n_runs} already cached")
            continue
        results = searcher.search(images, texts, verbose=(run_id == 0))
        searcher.save_results(results, run_id=run_id, out_dir=out_dir)
        print(f"  Run {run_id + 1}/{n_runs} done")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    agg = searcher.load_results_k(out_dir=out_dir)
    if agg is None:
        raise RuntimeError(f"No auto_scaler runs found for {key}")
    best = float(searcher.get_best_from_agg(agg))
    save_scaler(
        image_tag, text_tag, dataset, best,
        image_encoder_arg=image_encoder,
        text_encoder_arg=text_encoder,
        image_encoder_pool=image_encoder_pool,
        n_samples=n_samples,
        n_runs=n_runs,
    )
    summary = RESULTS_DIR / f"{key}.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({
        "combo": key,
        "lang_scaler": best,
        "scalers": {str(k): v for k, v in agg["scalers"].items()},
        "baselines": agg.get("baselines", {}),
    }, indent=2), encoding="utf-8")
    print(f"[auto_scaler] Recorded {key}: lang_scaler={best}")
    searcher.cleanup()
    return best
