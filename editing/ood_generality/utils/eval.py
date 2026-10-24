"""Score one collective codebook on leftover combined MIDAS cases.

Reliability and locality stay in perturb_generality. This run only applies
that same codebook to the OOD set. The before score is the saved prediction
against the case y16. The after score is a new generation with the codebook
applied, against the same y16. That after score is generality.
"""

from __future__ import annotations

import json
from pathlib import Path

import config as C
from derm_dataset import DermEditDataset
from run_efficacy import is_correct

METRICS = ("generality",)


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


def eval_generality(editor, vlm, ood_rows, config, *, norm_answer, max_new_tokens):
    """Saved top-1 is the before score. One edited generation is the after score."""
    if not ood_rows:
        return 0, 0, []
    ds = DermEditDataset(config, ood_rows)
    ds.set_dataloader()
    editor.apply_to_dataset(ds)
    ds.task_generate(vlm, use_cache=True, max_new_tokens=max_new_tokens)
    log = {row["uid"]: row for row in (editor.last_retrieval_log or [])}
    rows, n_before, n_after = [], 0, 0
    for src, ex in zip(ood_rows, ds.data):
        pred_before = src.get("pred_before") or ""
        pred_after = (ex.get("pred") or {}).get("answer", "")
        ok_before = is_correct(pred_before, ex["answer"], ex.get("human_top1_y16"), norm_answer)
        ok_after = is_correct(pred_after, ex["answer"], ex.get("human_top1_y16"), norm_answer)
        n_before += int(ok_before)
        n_after += int(ok_after)
        hit = log.get(ex["uid"]) or {}
        rows.append({
            "uid": ex["uid"],
            "case_id": ex.get("case_id"),
            "y16": ex.get("human_top1_y16"),
            "pred_before": pred_before,
            "pred": pred_after,
            "correct_before": ok_before,
            "correct": ok_after,
            "retrieved": bool(hit.get("facts")),
            "retrieved_case_id": hit.get("retrieved_case_id") or "",
            "retrieved_image": hit.get("retrieved_image") or "",
            "retrieved_fact": hit.get("retrieved_fact") or "",
            "retrieved_distance": hit.get("retrieved_distance"),
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


def _slice_counts(rows):
    """gen_acc is fixes among original errors. gen_loc is original successes that stay correct."""
    n_wrong = n_fixed = n_right = n_kept = 0
    for row in rows:
        if row.get("correct_before"):
            n_right += 1
            n_kept += int(bool(row.get("correct")))
        else:
            n_wrong += 1
            n_fixed += int(bool(row.get("correct")))
    return {
        "n_orig_wrong": n_wrong,
        "n_fixed": n_fixed,
        "n_orig_correct": n_right,
        "n_kept": n_kept,
        "n_broken": n_right - n_kept,
    }


def _with_slices(payload):
    if not payload:
        return None
    merged = dict(payload)
    if "n_kept" not in merged:
        if "n_fixed" not in merged or "n_broken" not in merged:
            merged.update(_slice_counts(merged.get("cases") or []))
        elif merged.get("n_orig_correct") is not None and merged.get("n_broken") is not None:
            merged["n_kept"] = merged["n_orig_correct"] - merged["n_broken"]
    if "n_correct_before" not in merged:
        merged["n_correct_before"] = sum(
            1 for row in (merged.get("cases") or []) if row.get("correct_before")
        )
    return merged


def _pair(numerator, denominator):
    if numerator is None or denominator is None:
        return "--"
    return f"{numerator}/{denominator}"


def _format_scores(payload):
    payload = _with_slices(payload)
    if not payload:
        return "gen_all before --  gen_all after --  gen_acc --  gen_loc --"
    return (
        f"gen_all before {_pair(payload.get('n_correct_before'), payload.get('n'))}  "
        f"gen_all after {_pair(payload.get('n_correct'), payload.get('n'))}  "
        f"gen_acc {_pair(payload.get('n_fixed'), payload.get('n_orig_wrong'))}  "
        f"gen_loc {_pair(payload.get('n_kept'), payload.get('n_orig_correct'))}"
    )


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
    payload.update(_slice_counts(rows))
    path = result_path(out_dir, metric, user, model_short)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"{metric}: {user} / {model_short}: {_format_scores(payload)}")
    return payload


def _summary(user, model_short, payloads):
    present = next((p for p in payloads.values() if p), None)
    model = present["model"] if present else C.MODEL_SHORT_TO_ANNOT[model_short]
    generality = _with_slices(payloads.get("generality"))
    return {
        "user": user,
        "model": model,
        "short_model": model_short,
        "gen_all_before": _pair(
            None if not generality else generality.get("n_correct_before"),
            None if not generality else generality.get("n"),
        ),
        "gen_all_after": _pair(
            None if not generality else generality.get("n_correct"),
            None if not generality else generality.get("n"),
        ),
        "gen_acc": _pair(
            None if not generality else generality.get("n_fixed"),
            None if not generality else generality.get("n_orig_wrong"),
        ),
        "gen_loc": _pair(
            None if not generality else generality.get("n_kept"),
            None if not generality else generality.get("n_orig_correct"),
        ),
    }


def load_suite(out_dir, user, model_short):
    payloads = {metric: _read_metric(out_dir, metric, user, model_short) for metric in METRICS}
    return _summary(user, model_short, payloads)


def run_suite(editor, vlm, queries, *, user, model_short, out_dir, ood_rows,
              norm_answer=None, max_new_tokens=None, overwrite=False):
    """Generality only. A result already on disk is kept."""
    norm_answer = norm_answer or C.NORM_ANSWER
    max_new_tokens = max_new_tokens or C.MAX_NEW_TOKENS
    annot = C.MODEL_SHORT_TO_ANNOT[model_short]
    common = dict(norm_answer=norm_answer, max_new_tokens=max_new_tokens)
    saved = None if overwrite else _read_metric(out_dir, "generality", user, model_short)
    if saved and "n_correct_before" in saved:
        print(f"generality: {user} / {model_short}: loaded {_format_scores(saved)}")
        payloads = {"generality": saved}
    else:
        n_before, n_after, rows = eval_generality(
            editor, vlm, ood_rows, queries.config, **common,
        )
        n = len(rows)
        payloads = {"generality": _write(
            out_dir, "generality", user, model_short, annot, n_after, rows,
            extra={
                "n_correct_before": n_before,
                "score_before": n_before / n if n else 0.0,
                "n_retrieved": sum(1 for row in rows if row["retrieved"]),
            },
        )}
    return _summary(user, model_short, payloads)


def render_table(rows):
    labeled = any(row.get("threshold") for row in rows)
    if labeled:
        lines = [
            "| Mode | Threshold | Model | gen_all before | gen_all after | gen_acc | gen_loc |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in rows:
            lines.append(
                f"| {row.get('embed_mode', '')} | {row.get('threshold', '')} | {row['model']} | "
                f"{row['gen_all_before']} | {row['gen_all_after']} | {row['gen_acc']} | {row['gen_loc']} |"
            )
        return "\n".join(lines)
    lines = [
        "| User | Model | gen_all before | gen_all after | gen_acc | gen_loc |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        user = str(row["user"]).split("@")[0]
        lines.append(
            f"| {user} | {row['model']} | {row['gen_all_before']} | {row['gen_all_after']} | "
            f"{row['gen_acc']} | {row['gen_loc']} |"
        )
    return "\n".join(lines)
