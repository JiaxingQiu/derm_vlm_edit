"""First step: record the balancing weight for one encoder combo.

    python -m reasonedit.run.auto_scaler
    python -m reasonedit.run.auto_scaler --image_encoder biomedclip --text_encoder minilm
"""

from __future__ import annotations

import argparse

from reasonedit.auto_scaler import tune_scaler


def main():
    ap = argparse.ArgumentParser(description="Record a balancing weight for one encoder combo")
    ap.add_argument("--image_encoder", default="dinov2")
    ap.add_argument("--image_encoder_pool", default="cls", choices=["cls", "mean"])
    ap.add_argument("--text_encoder", default="sbert")
    ap.add_argument("--dataset", default="coco")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--n_runs", type=int, default=10)
    ap.add_argument("--n_download", type=int, default=50)
    ap.add_argument("--images_dir", default=None)
    ap.add_argument("--captions_file", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    best = tune_scaler(**vars(args))
    print(f"lang_scaler={best}")


if __name__ == "__main__":
    main()
