#!/usr/bin/env python3
"""Collect, then individual. Each uses standalone_dual, then language_last.

Finished metrics are kept. Missing ones continue.

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
        print(f"{model_short}: {len(locality)} locality cases")
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
            out_dir=out, locality_edits=locality, seed=opts["seed"],
            norm_answer=opts["norm_answer"], max_new_tokens=opts["max_new_tokens"],
            overwrite=overwrite,
        ))
        del vlm, editor
        gc.collect()
        torch.cuda.empty_cache()
    print(render_table(rows))


def run_individual(embed_mode, overwrite=False, resume=True):
    opts = _defaults()
    out = results_dir(ROOT, "individual", embed_mode)
    print(f"results: {out}  mode={embed_mode}", flush=True)
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
                scaler_dataset=opts["scaler_dataset"],
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
    ap = argparse.ArgumentParser(description="Collect, then individual. Both embedding modes.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    resume = not args.no_resume
    for split, fn in (("collect", run_collect), ("individual", run_individual)):
        for mode in EMBED_MODES:
            print(f"\n===== {split}  {mode} =====", flush=True)
            fn(mode, overwrite=args.overwrite, resume=resume)


if __name__ == "__main__":
    main()
