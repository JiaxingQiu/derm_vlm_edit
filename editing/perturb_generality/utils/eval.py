"""Score one codebook three ways: reliability, then text generality, then image generality.

Each scorer copies the query set, so a later pass still sees the original image and question.
The codebook itself is left as `edit` built it.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import config as C
from derm_dataset import DermEditDataset
from run_efficacy import is_correct

from utils.prompts import SENTENCES, make_question

METRICS = ("reliability", "text_generality", "image_generality")


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
    return {
        "user": user,
        "model": payloads["reliability"]["model"],
        "short_model": model_short,
        "reliability": f"{payloads['reliability']['n_correct']}/{payloads['reliability']['n']}",
        "text_generality": f"{payloads['text_generality']['n_correct']}/{payloads['text_generality']['n']}",
        "image_generality": f"{payloads['image_generality']['n_correct']}/{payloads['image_generality']['n']}",
    }


def load_suite(out_dir, user, model_short):
    payloads = {
        metric: json.loads(result_path(out_dir, metric, user, model_short).read_text())
        for metric in METRICS
    }
    return _summary(user, model_short, payloads)


def run_suite(editor, vlm, queries, *, user, model_short, out_dir, seed=42,
              norm_answer=None, max_new_tokens=None):
    """Reliability, then text generality, then image generality, on one codebook."""
    norm_answer = norm_answer or C.NORM_ANSWER
    max_new_tokens = max_new_tokens or C.MAX_NEW_TOKENS
    annot = C.MODEL_SHORT_TO_ANNOT[model_short]
    common = dict(norm_answer=norm_answer, max_new_tokens=max_new_tokens)

    reliability = _write(out_dir, "reliability", user, model_short, annot,
                         *eval_reliability(editor, vlm, queries, **common))
    text = _write(out_dir, "text_generality", user, model_short, annot,
                  *eval_text(editor, vlm, queries, seed=seed, user=user,
                             model_short=model_short, **common))
    image = _write(out_dir, "image_generality", user, model_short, annot,
                   *eval_image(editor, vlm, queries, seed=seed, user=user,
                               model_short=model_short, **common))
    return _summary(user, model_short, {
        "reliability": reliability,
        "text_generality": text,
        "image_generality": image,
    })


def render_table(rows):
    lines = [
        "| User | Model | Reliability | Text generality | Image generality |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        user = str(row["user"]).split("@")[0]
        lines.append(
            f"| {user} | {row['model']} | {row['reliability']} | "
            f"{row['text_generality']} | {row['image_generality']} |"
        )
    return "\n".join(lines)
