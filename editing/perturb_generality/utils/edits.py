"""Collect the edit list. This is the only step that differs between notebooks."""

from __future__ import annotations

from collections import defaultdict

import config as C
from data_engineering.build_edit_set import build as build_edit_set


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


def collect_all(users, model_short, rationale_mode=None, norm_answer=None, subsample=0):
    """Concatenate every user's edits into one codebook list."""
    edits, summaries = [], []
    for user in users:
        chunk, summary = collect_user(
            user, model_short, rationale_mode=rationale_mode, norm_answer=norm_answer,
        )
        edits.extend(chunk)
        summaries.append(summary)
    if subsample:
        edits = edits[:subsample]
    return edits, summaries


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
