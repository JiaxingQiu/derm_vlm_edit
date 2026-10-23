"""Score one codebook: reliability, text generality, image generality, then locality.

Each scorer copies its query set. The codebook itself is left as `edit` built it.
Locality generates the untouched cases twice: once with no fact, once with retrieval.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import config as C
from derm_dataset import DermEditDataset
from data_engineering.normalize_dx import normalize
from run_efficacy import is_correct

from utils.prompts import SENTENCES, make_question

METRICS = ("reliability", "text_generality", "image_generality", "locality")


def load_vlm(model_short, *, max_new_tokens, seed, embed_mode, image_encoder,
             image_pool, text_encoder, scaler_dataset, reject_threshold=0.1):
    import torch
    from reasonedit import VQAModel, load_config
    from run_efficacy import resolve_layers

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = load_config(
        model_short, device=device, batch_size=1,
        max_new_tokens=max_new_tokens, seed=seed,
        mode=embed_mode, image_encoder=image_encoder,
        image_encoder_pool=image_pool, text_encoder=text_encoder,
        scaler_dataset=scaler_dataset,
        reject_threshold_pct=reject_threshold,
    )
    config.editor.fact_template = C.FACT_TEMPLATE
    vlm = VQAModel(config)
    resolve_layers(vlm, config)
    return vlm, config


def _clone(queries):
    cloned = DermEditDataset(queries.config, [])
    cloned.data = copy.deepcopy(queries.data)
    return cloned


def _score(ds, editor, norm_answer):
    retr = {row["uid"]: row.get("facts", []) for row in (editor.last_retrieval_log or [])}
    rows, n_ok = [], 0
    for ex in ds.data:
        pred = ex.get("pred", {})
        ok = is_correct(pred.get("answer", ""), ex["answer"], ex.get("human_top1_y16"), norm_answer)
        n_ok += int(ok)
        rows.append({
            "uid": ex["uid"],
            "case_id": ex.get("case_id"),
            "user": ex.get("login_id"),
            "rephrase": ex.get("rephrase"),
            "human_answer": ex["answer"],
            "human_y16": ex.get("human_top1_y16"),
            "pred": pred.get("answer", ""),
            "correct": ok,
            "retrieved": bool(retr.get(ex["uid"])),
        })
    return n_ok, rows


def _run_generate(editor, vlm, ds, max_new_tokens, norm_answer):
    editor.apply_to_dataset(ds)
    ds.task_generate(vlm, use_cache=True, max_new_tokens=max_new_tokens)
    return _score(ds, editor, norm_answer)


def eval_reliability(editor, vlm, queries, *, norm_answer, max_new_tokens):
    n_ok, rows = _run_generate(editor, vlm, _clone(queries), max_new_tokens, norm_answer)
    return n_ok, rows


def eval_text(editor, vlm, queries, *, norm_answer, max_new_tokens, seed, user, model_short):
    ds = _clone(queries)
    rng = random.Random(f"{seed}|{user}|{model_short}|text")
    for ex in ds.data:
        sentence = rng.choice(SENTENCES)
        question = make_question(sentence)
        ex["rephrase"] = sentence
        ex["question"] = question
        ex["prompt"] = question
        ex["prompt_orig"] = question
    n_ok, rows = _run_generate(editor, vlm, ds, max_new_tokens, norm_answer)
    return n_ok, rows


def _preserved(before, after, mode):
    if not before or not after:
        return False
    left, right = normalize(before), normalize(after)
    if mode == "y16":
        return bool(left.y16 and right.y16 and left.y16 == right.y16)
    return bool(left.norm and left.norm == right.norm)


def eval_locality(editor, vlm, locality_edits, config, *, norm_answer, max_new_tokens):
    """Before-edit top-1 is the reference. After-edit top-1 should match it."""
    if not locality_edits:
        return 0, []
    base = DermEditDataset(config, locality_edits)
    base.set_dataloader()
    before = _clone(base)
    before.task_generate(vlm, use_cache=True, max_new_tokens=max_new_tokens)
    after = _clone(base)
    editor.apply_to_dataset(after)
    after.task_generate(vlm, use_cache=True, max_new_tokens=max_new_tokens)
    retr = {row["uid"]: row.get("facts", []) for row in (editor.last_retrieval_log or [])}
    rows, n_ok = [], 0
    for old, new in zip(before.data, after.data):
        pred_before = (old.get("pred") or {}).get("answer", "")
        pred_after = (new.get("pred") or {}).get("answer", "")
        ok = _preserved(pred_before, pred_after, norm_answer)
        n_ok += int(ok)
        rows.append({
            "uid": new["uid"],
            "case_id": new.get("case_id"),
            "user": new.get("login_id"),
            "pred_before": pred_before,
            "pred": pred_after,
            "correct": ok,
            "retrieved": bool(retr.get(new["uid"])),
        })
    return n_ok, rows


def eval_image(editor, vlm, queries, *, norm_answer, max_new_tokens, seed, user, model_short):
    from utils.perturb import perturb_image, stable_seed

    ds = _clone(queries)
    for ex in ds.data:
        aug_seed = stable_seed(seed, user, model_short, ex["uid"])
        ex["image"] = perturb_image(ex["image"], aug_seed)
    n_ok, rows = _run_generate(editor, vlm, ds, max_new_tokens, norm_answer)
    return n_ok, rows


def result_path(out_dir, metric, user, model_short) -> Path:
    safe = str(user).replace("@", "_at_").replace(".", "_")
    return Path(out_dir) / f"{metric}__{safe}__{model_short}.json"


def suite_complete(out_dir, user, model_short) -> bool:
    return all(result_path(out_dir, metric, user, model_short).exists() for metric in METRICS)


def needs_compute(out_dir, user, model_short, *, overwrite, resume) -> bool:
    """GPU work is required only to recompute or to finish a partial run."""
    if overwrite:
        return True
    if suite_complete(out_dir, user, model_short):
        return False
    return bool(resume)


def _read_metric(out_dir, metric, user, model_short):
    path = result_path(out_dir, metric, user, model_short)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _cell(payload):
    if not payload:
        return "--"
    return f"{payload['n_correct']}/{payload['n']}"


def _write(out_dir, metric, user, model_short, annot, n_ok, rows):
    n = len(rows)
    payload = {
        "user": user,
        "model": annot,
        "short_model": model_short,
        "metric": metric,
        "n": n,
        "n_correct": n_ok,
        "score": n_ok / n if n else 0.0,
        "cases": rows,
    }
    path = result_path(out_dir, metric, user, model_short)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"{metric}: {user} / {model_short}: {n_ok}/{n}")
    return payload


def _summary(user, model_short, payloads):
    present = next((p for p in payloads.values() if p), None)
    model = present["model"] if present else C.MODEL_SHORT_TO_ANNOT[model_short]
    return {
        "user": user,
        "model": model,
        "short_model": model_short,
        "reliability": _cell(payloads.get("reliability")),
        "text_generality": _cell(payloads.get("text_generality")),
        "image_generality": _cell(payloads.get("image_generality")),
        "locality": _cell(payloads.get("locality")),
    }


def load_suite(out_dir, user, model_short):
    payloads = {metric: _read_metric(out_dir, metric, user, model_short) for metric in METRICS}
    return _summary(user, model_short, payloads)


def run_suite(editor, vlm, queries, *, user, model_short, out_dir, locality_edits,
              seed=42, norm_answer=None, max_new_tokens=None, overwrite=False):
    """Reliability, text generality, image generality, then locality.

    A metric already on disk is kept when overwrite is false.
    """
    norm_answer = norm_answer or C.NORM_ANSWER
    max_new_tokens = max_new_tokens or C.MAX_NEW_TOKENS
    annot = C.MODEL_SHORT_TO_ANNOT[model_short]
    common = dict(norm_answer=norm_answer, max_new_tokens=max_new_tokens)
    payloads = {}

    def _kept(metric):
        if overwrite:
            return None
        saved = _read_metric(out_dir, metric, user, model_short)
        if saved:
            print(f"{metric}: {user} / {model_short}: loaded {saved['n_correct']}/{saved['n']}")
        return saved

    payloads["reliability"] = _kept("reliability") or _write(
        out_dir, "reliability", user, model_short, annot,
        *eval_reliability(editor, vlm, queries, **common))
    payloads["text_generality"] = _kept("text_generality") or _write(
        out_dir, "text_generality", user, model_short, annot,
        *eval_text(editor, vlm, queries, seed=seed, user=user,
                   model_short=model_short, **common))
    payloads["image_generality"] = _kept("image_generality") or _write(
        out_dir, "image_generality", user, model_short, annot,
        *eval_image(editor, vlm, queries, seed=seed, user=user,
                    model_short=model_short, **common))
    payloads["locality"] = _kept("locality") or _write(
        out_dir, "locality", user, model_short, annot,
        *eval_locality(editor, vlm, locality_edits, queries.config, **common))
    return _summary(user, model_short, payloads)


def render_table(rows):
    lines = [
        "| User | Model | Reliability | Text generality | Image generality | Locality |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        user = str(row["user"]).split("@")[0]
        lines.append(
            f"| {user} | {row['model']} | {row['reliability']} | {row['text_generality']} | "
            f"{row['image_generality']} | {row['locality']} |"
        )
    return "\n".join(lines)
