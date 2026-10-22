#!/usr/bin/env python3
"""Collective OOD generality. standalone_dual, then language_last.

Finished metrics are kept. Missing ones continue. Generality is last because
it generates on every leftover combined MIDAS case.

    python run.py
    python run.py --overwrite
    python run.py --no-resume
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from utils.bootstrap import bootstrap, results_dir
bootstrap()

import config as C
from utils.edit import edit
from utils.edits import collect_all, collect_locality, collect_ood
from utils.eval import load_suite, load_vlm, needs_compute, render_table, run_suite

USERS = [
    "jesse.y.liu.md@gmail.com",
    "achiou@stanford.edu",
    "lhunterellul",
]
MODELS = ["medgemma", "dermatollama"]
EMBED_MODES = ["standalone_dual", "language_last"]
POOL = "all"


def _defaults():
    return dict(
        rationale_mode=C.RATIONALE_MODE,
        norm_answer=C.NORM_ANSWER,
        subsample=0,
        max_new_tokens=C.MAX_NEW_TOKENS,
        seed=42,
        image_encoder="dinov2",
        image_pool="cls",
        text_encoder="sbert",
        scaler_dataset="coco",
    )


def run_collect(embed_mode, overwrite=False, resume=True):
    opts = _defaults()
    out = results_dir(ROOT, "collect", embed_mode)
    print(f"results: {out}  mode={embed_mode}", flush=True)
    rows = []
    for model_short in MODELS:
        if not needs_compute(out, POOL, model_short, overwrite=overwrite, resume=resume):
            print(f"{POOL} / {model_short}: loaded")
            rows.append(load_suite(out, POOL, model_short))
            continue
        edits, report = collect_all(
            USERS, model_short,
            rationale_mode=opts["rationale_mode"],
            norm_answer=opts["norm_answer"],
            subsample=opts["subsample"],
        )
        print(f"{model_short}: {len(edits)} unique edits, {len(report['conflicts'])} conflicting cases dropped")
        for row in report["conflicts"]:
            users = ", ".join(str(u).split("@")[0] for u in row["users"])
            print(f"  drop {row['case_id']}: {users} -> {', '.join(row['answers'])}")
        locality = collect_locality(USERS, model_short, subsample=opts["subsample"])
        ood = collect_ood(USERS, model_short, subsample=opts["subsample"])
        n_before_text = sum(1 for row in ood if row.get("pred_before"))
        print(f"{model_short}: {len(locality)} locality cases, {len(ood)} ood cases ({n_before_text} with a saved top-1)")
        if not edits:
            print(f"skip {model_short}")
            continue
        import torch
        vlm, config = load_vlm(
            model_short, max_new_tokens=opts["max_new_tokens"], seed=opts["seed"],
            embed_mode=embed_mode, image_encoder=opts["image_encoder"],
            image_pool=opts["image_pool"], text_encoder=opts["text_encoder"],
            scaler_dataset=opts["scaler_dataset"],
        )
        editor, queries = edit(vlm, config, edits)
        rows.append(run_suite(
            editor, vlm, queries, user=POOL, model_short=model_short,
            out_dir=out, locality_edits=locality, ood_rows=ood,
            norm_answer=opts["norm_answer"], max_new_tokens=opts["max_new_tokens"],
            overwrite=overwrite,
        ))
        del vlm, editor
        gc.collect()
        torch.cuda.empty_cache()
    print(render_table(rows))


def main():
    ap = argparse.ArgumentParser(description="Collective reliability, locality, and OOD generality.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    resume = not args.no_resume
    for mode in EMBED_MODES:
        print(f"\n===== collect  {mode} =====", flush=True)
        run_collect(mode, overwrite=args.overwrite, resume=resume)


if __name__ == "__main__":
    main()
