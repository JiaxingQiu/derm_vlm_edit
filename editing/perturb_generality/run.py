#!/usr/bin/env python3
"""Collective run only. Each embedding mode runs at reject percentiles 1 and 5.

Finished metrics are kept. Missing ones continue. Results land in
results/perturb_gen/{embed_mode}_{threshold}/collect/. The per-dermatologist notebook is separate.

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
from utils.edits import collect_all, collect_locality, collect_user
from utils.eval import load_suite, load_vlm, needs_compute, render_table, run_suite

USERS = [
    "jesse.y.liu.md@gmail.com",
    "achiou@stanford.edu",
    "lhunterellul",
]
MODELS = ["medgemma", "dermatollama"]
EMBED_MODES = ["standalone_dual", "language_last"]
REJECT_THRESHOLDS = [1, 5]
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


def _tag(embed_mode, reject_threshold):
    """1 and 1.0 both land in ``standalone_dual_1``."""
    return f"{embed_mode}_{float(reject_threshold):g}"


def run_collect(embed_mode, reject_threshold, overwrite=False, resume=True):
    opts = _defaults()
    tag = _tag(embed_mode, reject_threshold)
    out = results_dir(ROOT, "collect", tag)
    print(f"results: {out}  mode={embed_mode}  reject={reject_threshold}", flush=True)
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
        print(f"{model_short}: {len(locality)} locality cases")
        if not edits:
            print(f"skip {model_short}")
            continue
        import torch
        vlm, config = load_vlm(
            model_short, max_new_tokens=opts["max_new_tokens"], seed=opts["seed"],
            embed_mode=embed_mode, image_encoder=opts["image_encoder"],
            image_pool=opts["image_pool"], text_encoder=opts["text_encoder"],
            scaler_dataset=opts["scaler_dataset"], reject_threshold=reject_threshold,
        )
        editor, queries = edit(vlm, config, edits)
        rows.append(run_suite(
            editor, vlm, queries, user=POOL, model_short=model_short,
            out_dir=out, locality_edits=locality, seed=opts["seed"],
            norm_answer=opts["norm_answer"], max_new_tokens=opts["max_new_tokens"],
            overwrite=overwrite,
        ))
        del vlm, editor
        gc.collect()
        torch.cuda.empty_cache()
    print(render_table(rows))


def run_individual(embed_mode, reject_threshold, overwrite=False, resume=True):
    opts = _defaults()
    tag = _tag(embed_mode, reject_threshold)
    out = results_dir(ROOT, "individual", tag)
    print(f"results: {out}  mode={embed_mode}  reject={reject_threshold}", flush=True)
    rows = []
    for model_short in MODELS:
        cached, todo = {}, []
        for user in USERS:
            if needs_compute(out, user, model_short, overwrite=overwrite, resume=resume):
                todo.append(user)
            else:
                print(f"{user} / {model_short}: loaded")
                cached[user] = load_suite(out, user, model_short)
        if todo:
            import torch
            editor = None
            locality = collect_locality(USERS, model_short, subsample=opts["subsample"])
            vlm, config = load_vlm(
                model_short, max_new_tokens=opts["max_new_tokens"], seed=opts["seed"],
                embed_mode=embed_mode, image_encoder=opts["image_encoder"],
                image_pool=opts["image_pool"], text_encoder=opts["text_encoder"],
                scaler_dataset=opts["scaler_dataset"], reject_threshold=reject_threshold,
            )
            for user in todo:
                edits, _ = collect_user(
                    user, model_short,
                    rationale_mode=opts["rationale_mode"],
                    norm_answer=opts["norm_answer"],
                    subsample=opts["subsample"],
                )
                if not edits:
                    print(f"skip {user} / {model_short}")
                    continue
                editor, queries = edit(vlm, config, edits)
                cached[user] = run_suite(
                    editor, vlm, queries, user=user, model_short=model_short,
                    out_dir=out, locality_edits=locality, seed=opts["seed"],
                    norm_answer=opts["norm_answer"], max_new_tokens=opts["max_new_tokens"],
                    overwrite=overwrite,
                )
            del vlm, editor
            gc.collect()
            torch.cuda.empty_cache()
        for user in USERS:
            if user in cached:
                rows.append(cached[user])
    print(render_table(rows))


def main():
    ap = argparse.ArgumentParser(description="Collective run. Both embedding modes.")
    ap.add_argument("--pct_threshold", type=float, default=None,
                    help="One reject percentile, so two jobs can run in parallel. Omit to run 1 and 5.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    resume = not args.no_resume
    thresholds = REJECT_THRESHOLDS if args.pct_threshold is None else [args.pct_threshold]
    for mode in EMBED_MODES:
        for pct in thresholds:
            print(f"\n===== collect  {mode}  reject={pct} =====", flush=True)
            run_collect(mode, pct, overwrite=args.overwrite, resume=resume)


if __name__ == "__main__":
    main()
