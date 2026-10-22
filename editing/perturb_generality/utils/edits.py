"""Collect the edit list and the locality list.

The edit list is cases a user marked wrong. The locality list is cases no user
marked wrong. Collective edits are one row per case: a disagreement about the
correction is logged and dropped, and a single "wrong" label wins over "correct".
"""

from __future__ import annotations

from collections import defaultdict

import config as C
from data_engineering.build_edit_set import build as build_edit_set
from data_engineering.parse_annotations import load_records


def collect_user(user, model_short, rationale_mode=None, norm_answer=None, subsample=0):
    """One dermatologist's top-1 errors for one model."""
    rationale_mode = rationale_mode or C.RATIONALE_MODE
    norm_answer = norm_answer or C.NORM_ANSWER
    annot = C.MODEL_SHORT_TO_ANNOT[model_short]
    edits, summary = build_edit_set(
        user, annot, rationale_mode, write=False, norm_answer=norm_answer,
    )
    if subsample:
        edits = edits[:subsample]
    return edits, summary


def dedupe_cases(edits):
    """One row per case. Differing corrections are returned separately and dropped."""
    groups = defaultdict(list)
    order = []
    for edit in edits:
        case_id = edit["case_id"]
        if case_id not in groups:
            order.append(case_id)
        groups[case_id].append(edit)
    kept, conflicts = [], []
    for case_id in order:
        items = groups[case_id]
        answers = list(dict.fromkeys(item.get("answer") for item in items))
        if len(answers) > 1:
            conflicts.append({
                "case_id": case_id,
                "model": items[0].get("model"),
                "users": [item.get("login_id") for item in items],
                "answers": answers,
            })
            continue
        kept.append(items[0])
    return kept, conflicts


def collect_all(users, model_short, rationale_mode=None, norm_answer=None, subsample=0):
    """Unique cases across users. Wrong wins because only error rows are collected."""
    edits, summaries = [], []
    for user in users:
        chunk, summary = collect_user(
            user, model_short, rationale_mode=rationale_mode, norm_answer=norm_answer,
        )
        edits.extend(chunk)
        summaries.append(summary)
    edits, conflicts = dedupe_cases(edits)
    if subsample:
        edits = edits[:subsample]
    return edits, {"summaries": summaries, "conflicts": conflicts}


def collect_locality(users, model_short, subsample=0):
    """Completed cases for this model that no listed user marked top-1 incorrect."""
    annot = C.MODEL_SHORT_TO_ANNOT[model_short]
    user_set = set(users)
    by_case = defaultdict(list)
    for rec in load_records(C.ANNOTATIONS_CSV):
        if rec.model != annot or rec.login_id not in user_set or not rec.marked_complete:
            continue
        by_case[rec.case_id].append(rec)
    rows = []
    for case_id, items in sorted(by_case.items()):
        if any(rec.is_top1_error for rec in items):
            continue
        image = C.image_path_for(case_id)
        if not image.exists():
            continue
        rec = items[0]
        rows.append({
            "uid": f"locality__{rec.model}__{case_id}",
            "image": str(image),
            "question": C.QUESTION,
            "answer": rec.model_top1 or "",
            "rationale": "",
            "cot": "",
            "login_id": "",
            "model": rec.model,
            "case_id": case_id,
            "model_top1": rec.model_top1,
            "human_top1_raw": "",
            "human_top1_y16": None,
        })
    if subsample:
        rows = rows[:subsample]
    return rows


def overlap_report(edits):
    """Cases stored by more than one user. Same image and same question, so the keys collide."""
    groups = defaultdict(list)
    for edit in edits:
        groups[(edit.get("model"), edit.get("case_id"))].append(edit)
    rows = []
    for (model, case_id), items in sorted(groups.items()):
        users = sorted({item.get("login_id") for item in items})
        if len(users) < 2:
            continue
        answers = sorted({item.get("answer") for item in items})
        rows.append({
            "model": model,
            "case_id": case_id,
            "n_users": len(users),
            "users": users,
            "answers_agree": len(answers) == 1,
            "answers": answers,
        })
    return rows
