"""Score one collective codebook: reliability, locality, then OOD generality.

Generality uses leftover combined MIDAS cases. The before score is the saved
prediction against the case y16. The after score is a new generation with the
codebook applied, against the same y16. That after score is generality.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import config as C
from derm_dataset import DermEditDataset
from data_engineering.normalize_dx import normalize
from run_efficacy import is_correct

METRICS = ("reliability", "locality", "generality")


def load_vlm(model_short, *, max_new_tokens, seed, embed_mode, image_encoder,
             image_pool, text_encoder, scaler_dataset):
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


def eval_generality(editor, vlm, ood_rows, config, *, norm_answer, max_new_tokens):
    """Saved top-1 is the before score. One edited generation is the after score."""
    if not ood_rows:
        return 0, 0, []
    ds = DermEditDataset(config, ood_rows)
    ds.set_dataloader()
    editor.apply_to_dataset(ds)
    ds.task_generate(vlm, use_cache=True, max_new_tokens=max_new_tokens)
    retr = {row["uid"]: row.get("facts", []) for row in (editor.last_retrieval_log or [])}
    rows, n_before, n_after = [], 0, 0
    for src, ex in zip(ood_rows, ds.data):
        pred_before = src.get("pred_before") or ""
        pred_after = (ex.get("pred") or {}).get("answer", "")
        ok_before = is_correct(pred_before, ex["answer"], ex.get("human_top1_y16"), norm_answer)
        ok_after = is_correct(pred_after, ex["answer"], ex.get("human_top1_y16"), norm_answer)
        n_before += int(ok_before)
        n_after += int(ok_after)
        rows.append({
            "uid": ex["uid"],
            "case_id": ex.get("case_id"),
            "y16": ex.get("human_top1_y16"),
            "pred_before": pred_before,
            "pred": pred_after,
            "correct_before": ok_before,
            "correct": ok_after,
            "retrieved": bool(retr.get(ex["uid"])),
        })
    return n_before, n_after, rows


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


def _before_cell(payload):
    if not payload or "n_correct_before" not in payload:
        return "--"
    return f"{payload['n_correct_before']}/{payload['n']}"


def _overall(reliability, locality):
    if not reliability or not locality:
        return "--"
    correct = reliability["n_correct"] + locality["n_correct"]
    total = reliability["n"] + locality["n"]
    return f"{correct}/{total}"


def _write(out_dir, metric, user, model_short, annot, n_ok, rows, extra=None):
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
    if extra:
        payload.update(extra)
    path = result_path(out_dir, metric, user, model_short)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    if metric == "generality":
        before = payload.get("n_correct_before", 0)
        print(f"{metric}: {user} / {model_short}: before {before}/{n}  after {n_ok}/{n}")
    else:
        print(f"{metric}: {user} / {model_short}: {n_ok}/{n}")
    return payload


def _summary(user, model_short, payloads):
    present = next((p for p in payloads.values() if p), None)
    model = present["model"] if present else C.MODEL_SHORT_TO_ANNOT[model_short]
    generality = payloads.get("generality")
    return {
        "user": user,
        "model": model,
        "short_model": model_short,
        "reliability": _cell(payloads.get("reliability")),
        "generality_before": _before_cell(generality),
        "generality": _cell(generality),
        "locality": _cell(payloads.get("locality")),
        "overall": _overall(payloads.get("reliability"), payloads.get("locality")),
    }


def load_suite(out_dir, user, model_short):
    payloads = {metric: _read_metric(out_dir, metric, user, model_short) for metric in METRICS}
    return _summary(user, model_short, payloads)


def run_suite(editor, vlm, queries, *, user, model_short, out_dir, locality_edits,
              ood_rows, norm_answer=None, max_new_tokens=None, overwrite=False):
    """Reliability, locality, then generality. A metric on disk is kept."""
    norm_answer = norm_answer or C.NORM_ANSWER
    max_new_tokens = max_new_tokens or C.MAX_NEW_TOKENS
    annot = C.MODEL_SHORT_TO_ANNOT[model_short]
    common = dict(norm_answer=norm_answer, max_new_tokens=max_new_tokens)
    payloads = {}

    def _kept(metric):
        if overwrite:
            return None
        saved = _read_metric(out_dir, metric, user, model_short)
        if not saved:
            return None
        if metric == "generality" and "n_correct_before" in saved:
            print(
                f"{metric}: {user} / {model_short}: loaded "
                f"before {saved['n_correct_before']}/{saved['n']}  "
                f"after {saved['n_correct']}/{saved['n']}"
            )
        else:
            print(f"{metric}: {user} / {model_short}: loaded {saved['n_correct']}/{saved['n']}")
        return saved

    payloads["reliability"] = _kept("reliability") or _write(
        out_dir, "reliability", user, model_short, annot,
        *eval_reliability(editor, vlm, queries, **common))
    payloads["locality"] = _kept("locality") or _write(
        out_dir, "locality", user, model_short, annot,
        *eval_locality(editor, vlm, locality_edits, queries.config, **common))
    saved_generality = _kept("generality")
    if saved_generality:
        payloads["generality"] = saved_generality
    else:
        n_before, n_after, rows = eval_generality(
            editor, vlm, ood_rows, queries.config, **common,
        )
        n = len(rows)
        payloads["generality"] = _write(
            out_dir, "generality", user, model_short, annot, n_after, rows,
            extra={
                "n_correct_before": n_before,
                "score_before": n_before / n if n else 0.0,
                "n_retrieved": sum(1 for row in rows if row["retrieved"]),
            },
        )
    return _summary(user, model_short, payloads)


def render_table(rows):
    lines = [
        "| User | Model | Reliability | Generality before | Generality | Locality | Overall |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        user = str(row["user"]).split("@")[0]
        lines.append(
            f"| {user} | {row['model']} | {row['reliability']} | {row['generality_before']} | "
            f"{row['generality']} | {row['locality']} | {row['overall']} |"
        )
    return "\n".join(lines)
