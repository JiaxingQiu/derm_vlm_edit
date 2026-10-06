"""Reliability runner: does ReasonEdit flip the model's top-1 to the human answer?

Pipeline for a given (user, model):
  1. Build the edit set = the model's top-1 errors (baseline top-1 acc = 0).
  2. Baseline: ask the same top-3 differential question as the annotation
     interface; extract and record top-1.
  3. Edit: store the human top-1 as an IKE fact
     ("New Fact: The most likely differential is {answer}.")
     keyed by <image, question> (dual_sbert).
  4. Apply retrieval + re-ask the same top-3 question; extract top-1.
  5. Report reliability (top-1 accuracy vs human correction) before vs after.

Run this AFTER creating a working env (see env/). This script does the compute.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))

import config as C
from data_engineering.normalize_dx import normalize
from data_engineering.build_edit_set import build as build_edit_set

sys.path.insert(0, str(C.DERM_EDIT_ROOT))


def is_correct(pred_text: str, human_norm: str, human_y16, mode: str | None = None) -> bool:
    """Match extracted top-1 to the human answer at the selected grain."""
    if not pred_text:
        return False
    p = normalize(pred_text)
    mode = mode or C.NORM_ANSWER
    if mode == "y16":
        return bool(human_y16 and p.y16 and p.y16 == human_y16)
    return bool(human_norm and p.norm == human_norm)


def resolve_layers(model, config):
    """Correct best-guess inner_params_* against the actually-loaded model tree."""
    from reasonedit.utils import validate_and_correct_param_name

    for attr in ("inner_params_vision", "inner_params_lang", "inner_params"):
        names = getattr(config.model, attr, None) or []
        resolved = [validate_and_correct_param_name(model.model, n) for n in names]
        setattr(config.model, attr, resolved)
        print(f"[layers] {attr}: {resolved}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Derm ReasonEdit reliability (0->1) run")
    ap.add_argument("--model", required=True, choices=list(C.MODEL_SHORT_TO_ANNOT.keys()))
    ap.add_argument("--user", default=C.DEFAULT_USER)
    ap.add_argument("--rationale_mode", default=C.RATIONALE_MODE, choices=["edited_only", "final"])
    ap.add_argument("--norm_answer", default=C.NORM_ANSWER, choices=["y16", "human"])
    ap.add_argument("--subsample", type=int, default=0, help="0 = all edits")
    ap.add_argument("--max_new_tokens", type=int, default=C.MAX_NEW_TOKENS)
    args = ap.parse_args()

    import torch
    from reasonedit import ReasonEdit, VQAModel, load_config
    from derm_dataset import DermEditDataset

    annot_model = C.MODEL_SHORT_TO_ANNOT[args.model]

    # ---- 1. Build edit set (model's top-1 errors) ----
    edits, summary = build_edit_set(
        args.user, annot_model, args.rationale_mode,
        write=True, norm_answer=args.norm_answer,
    )
    if args.subsample and len(edits) > args.subsample:
        edits = edits[: args.subsample]
    if not edits:
        print(f"No usable edits for user={args.user} model={annot_model}. Exiting.")
        return
    print(f"[data] {len(edits)} edits | user={args.user} | model={annot_model} | norm_answer={args.norm_answer}", flush=True)

    # ---- config ----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = load_config(
        args.model, device=device,
        batch_size=1, max_new_tokens=args.max_new_tokens, seed=42,
    )
    config.editor.fact_template = C.FACT_TEMPLATE

    # ---- 2. Load model, fix layer names ----
    model = VQAModel(config)
    resolve_layers(model, config)

    ds = DermEditDataset(config, edits)
    ds.set_dataloader()

    # ---- baseline predictions ----
    print("=" * 50, "\n[baseline] generating pre-edit top-3 (eval extracts top-1)", flush=True)
    ds.task_generate(model, use_cache=True, max_new_tokens=args.max_new_tokens)
    pre = [ex.get("pred", {}).get("answer", "") for ex in ds.data]
    pre_raw = [ex.get("pred", {}).get("raw", "") for ex in ds.data]

    # ---- 3-4. Edit + apply retrieval + regenerate ----
    print("=" * 50, "\n[edit] building ReasonEdit codebook", flush=True)
    editor = ReasonEdit(config, model)
    editor.edit(config, edit_ds=ds)
    editor.apply_to_dataset(ds)
    print("=" * 50, "\n[post] generating post-edit top-3 (eval extracts top-1)", flush=True)
    ds.task_generate(model, use_cache=True, max_new_tokens=args.max_new_tokens)
    post = [ex.get("pred", {}).get("answer", "") for ex in ds.data]
    post_raw = [ex.get("pred", {}).get("raw", "") for ex in ds.data]

    retr_log = {r["uid"]: r.get("facts", []) for r in (editor.last_retrieval_log or [])}

    # ---- 5. Reliability ----
    rows = []
    n_pre = n_post = 0
    for ex, pa, qa, pa_raw, qa_raw in zip(ds.data, pre, post, pre_raw, post_raw):
        hn, hy = ex["answer"], ex.get("human_top1_y16")
        pc, qc = is_correct(pa, hn, hy, args.norm_answer), is_correct(qa, hn, hy, args.norm_answer)
        n_pre += int(pc)
        n_post += int(qc)
        rows.append({
            "uid": ex["uid"], "case_id": ex.get("case_id"),
            "model_top1": ex.get("model_top1"),
            "human_top1": ex.get("human_top1_raw"), "human_norm": hn, "human_y16": hy,
            "pred_pre": pa, "pred_post": qa,
            "pred_pre_raw": pa_raw, "pred_post_raw": qa_raw,
            "correct_pre": pc, "correct_post": qc,
            "retrieved_facts": retr_log.get(ex["uid"], []),
        })

    n = len(rows)
    n_pre = C.cap_before(n_pre)
    result = {
        "user": args.user, "model": annot_model, "short_model": args.model,
        "rationale_mode": args.rationale_mode, "norm_answer": args.norm_answer,
        "question": C.QUESTION,
        "fact_template": C.FACT_TEMPLATE,
        "n_edits": n,
        "reliability_pre": n_pre / n if n else 0.0,
        "reliability_post": n_post / n if n else 0.0,
        "n_correct_pre": n_pre, "n_correct_post": n_post,
        "inner_params_vision": getattr(config.model, "inner_params_vision", None),
        "cases": rows,
    }

    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    safe_user = args.user.replace("@", "_at_").replace(".", "_")
    out = C.RESULTS_DIR / f"reliability__{safe_user}__{args.model}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("=" * 50)
    print(f"reliability_pre  = {result['reliability_pre']:.3f} ({n_pre}/{n})")
    print(f"reliability_post = {result['reliability_post']:.3f} ({n_post}/{n})")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
