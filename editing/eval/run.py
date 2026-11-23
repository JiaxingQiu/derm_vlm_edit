#!/usr/bin/env python3
"""One configuration, both models, all eight metrics.

A configuration is one retrieval image, one image encoder, one embed mode,
and one reject percentile. MedGemma runs first, then DermatoLlama. Each model
loads once and builds one codebook. Reliability, text generality, image
generality, and locality go to results/perturb_gen/{tag}/collect/. OOD
generality goes to results/ood_gen/{tag}/collect/. ``dinov2`` keeps the
current folder names.

    python run.py combined
    python run.py photo --image-encoder panderm
    python run.py dscope --pct_threshold 10
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

EDITING = Path(__file__).resolve().parent.parent
PERTURB_ROOT = EDITING / "perturb_generality"
OOD_ROOT = EDITING / "ood_generality"

for _entry in (EDITING / "shared", EDITING / "reliability"):
    _path = str(_entry)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config as C
from retrieval_images import require_present, results_tag

_repo = str(C.DERM_EDIT_ROOT)
if _repo not in sys.path:
    sys.path.insert(0, _repo)

USERS = [
    "jesse.y.liu.md@gmail.com",
    "achiou@stanford.edu",
    "lhunterellul",
]
MODELS = ["medgemma", "dermatollama"]
POOL = "all"


def _defaults():
    return dict(
        rationale_mode=C.RATIONALE_MODE,
        norm_answer=C.NORM_ANSWER,
        subsample=0,
        max_new_tokens=C.MAX_NEW_TOKENS,
        seed=42,
        image_pool="cls",
        text_encoder="sbert",
        scaler_dataset="coco",
    )


def _purge_utils():
    for key in list(sys.modules):
        if key == "utils" or key.startswith("utils."):
            del sys.modules[key]


def _load_suite(root: Path) -> dict:
    """Import this folder's ``utils`` package and keep the module objects."""
    _purge_utils()
    entry = str(root)
    if entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)
    import utils.bootstrap
    import utils.edit
    import utils.edits
    import utils.eval
    return {
        key: sys.modules[key]
        for key in list(sys.modules)
        if key == "utils" or key.startswith("utils.")
    }


class Suite:
    def __init__(self, root: Path):
        self.root = root
        self.modules = _load_suite(root)
        self.edit = self.modules["utils.edit"]
        self.edits = self.modules["utils.edits"]
        self.eval = self.modules["utils.eval"]

    def activate(self):
        """Point ``utils`` at this suite. The other suite's modules stay in memory."""
        _purge_utils()
        sys.modules.update(self.modules)
        entry = str(self.root)
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)


def _print_conflicts(report):
    for row in report["conflicts"]:
        users = ", ".join(str(user).split("@")[0] for user in row["users"])
        print(f"  drop {row['case_id']}: {users} -> {', '.join(row['answers'])}")


def run_config(retrieval_image, embed_mode, reject_threshold, image_encoder,
               scaler_dataset="coco", overwrite=False, resume=True):
    opts = _defaults()
    opts["scaler_dataset"] = scaler_dataset
    perturb = Suite(PERTURB_ROOT)
    ood = Suite(OOD_ROOT)
    tag = results_tag(
        embed_mode, reject_threshold, retrieval_image, image_encoder, scaler_dataset,
    )
    perturb_out = C.results_dir(PERTURB_ROOT, "collect", tag)
    ood_out = C.results_dir(OOD_ROOT, "collect", tag)
    print(
        f"perturb: {perturb_out}\n"
        f"ood:     {ood_out}\n"
        f"mode={embed_mode}  reject={reject_threshold}  "
        f"retrieval_image={retrieval_image}  image_encoder={image_encoder}  "
        f"scaler_dataset={scaler_dataset}",
        flush=True,
    )
    perturb_rows, ood_rows_out = [], []
    for model_short in MODELS:
        do_perturb = perturb.eval.needs_compute(
            perturb_out, POOL, model_short, overwrite=overwrite, resume=resume,
        )
        do_ood = ood.eval.needs_compute(
            ood_out, POOL, model_short, overwrite=overwrite, resume=resume,
        )
        if not do_perturb and not do_ood:
            print(f"{POOL} / {model_short}: loaded")
            perturb_rows.append(perturb.eval.load_suite(perturb_out, POOL, model_short))
            ood_rows_out.append(ood.eval.load_suite(ood_out, POOL, model_short))
            continue
        edits, report = perturb.edits.collect_all(
            USERS, model_short,
            rationale_mode=opts["rationale_mode"],
            norm_answer=opts["norm_answer"],
            subsample=opts["subsample"],
            retrieval_image=retrieval_image,
        )
        print(
            f"{model_short}: {len(edits)} unique edits, "
            f"{len(report['conflicts'])} conflicting cases dropped"
        )
        _print_conflicts(report)
        locality = perturb.edits.collect_locality(
            USERS, model_short, subsample=opts["subsample"], retrieval_image=retrieval_image,
        )
        ood_rows = ood.edits.collect_ood(
            USERS, model_short, subsample=opts["subsample"], retrieval_image=retrieval_image,
        )
        n_before_text = sum(1 for row in ood_rows if row.get("pred_before"))
        print(f"{model_short}: {len(locality)} locality cases")
        print(f"{model_short}: {len(ood_rows)} ood cases ({n_before_text} with a saved top-1)")
        if not edits:
            print(f"skip {model_short}")
            continue
        require_present(edits, f"{model_short} edits")
        require_present(locality, f"{model_short} locality")
        require_present(ood_rows, f"{model_short} ood")
        import torch
        perturb.activate()
        vlm, config = perturb.eval.load_vlm(
            model_short, max_new_tokens=opts["max_new_tokens"], seed=opts["seed"],
            embed_mode=embed_mode, image_encoder=image_encoder,
            image_pool=opts["image_pool"], text_encoder=opts["text_encoder"],
            scaler_dataset=opts["scaler_dataset"], reject_threshold=reject_threshold,
        )
        editor, queries = perturb.edit.edit(vlm, config, edits)
        if do_perturb:
            perturb_rows.append(perturb.eval.run_suite(
                editor, vlm, queries, user=POOL, model_short=model_short,
                out_dir=perturb_out, locality_edits=locality, seed=opts["seed"],
                norm_answer=opts["norm_answer"], max_new_tokens=opts["max_new_tokens"],
                overwrite=overwrite,
            ))
        else:
            print(f"{POOL} / {model_short}: perturb loaded")
            perturb_rows.append(perturb.eval.load_suite(perturb_out, POOL, model_short))
        ood.activate()
        if do_ood:
            ood_rows_out.append(ood.eval.run_suite(
                editor, vlm, queries, user=POOL, model_short=model_short,
                out_dir=ood_out, ood_rows=ood_rows,
                norm_answer=opts["norm_answer"], max_new_tokens=opts["max_new_tokens"],
                overwrite=overwrite,
            ))
        else:
            print(f"{POOL} / {model_short}: ood loaded")
            ood_rows_out.append(ood.eval.load_suite(ood_out, POOL, model_short))
        del vlm, editor
        gc.collect()
        torch.cuda.empty_cache()
    print(perturb.eval.render_table(perturb_rows))
    print(ood.eval.render_table(ood_rows_out))


def main():
    ap = argparse.ArgumentParser(description="One configuration, both models, all eight metrics.")
    ap.add_argument("retrieval_image", choices=["combined", "photo", "dscope"])
    ap.add_argument("--pct_threshold", type=float, default=10)
    ap.add_argument("--embed-mode", default="standalone_dual", choices=["standalone_dual", "language_last"])
    ap.add_argument("--image-encoder", default="dinov2",
                    help="Short name for get_image_encoder. dinov2 keeps the current result folders.")
    ap.add_argument("--scaler-dataset", default="coco", choices=["coco", "derm"],
                    help="Which recorded balancing weight to load. coco keeps the current folders.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    run_config(
        args.retrieval_image, args.embed_mode, args.pct_threshold, args.image_encoder,
        scaler_dataset=args.scaler_dataset,
        overwrite=args.overwrite, resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
