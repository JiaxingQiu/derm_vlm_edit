"""Collect the codebook, the locality list, and the leftover MIDAS set.

The codebook and the locality list use the same rules as perturb_generality.
The OOD list is combined-image MIDAS cases none of these users completed for
this model. Each row keeps the saved top-1 so generality can score accuracy
before the codebook is applied.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import config as C
from data_engineering.extract_dx import extract_top1
from data_engineering.parse_annotations import load_records

_PERTURB_EDITS = (
    Path(__file__).resolve().parents[2] / "perturb_generality" / "utils" / "edits.py"
)

PRED_FILES = {
    "medgemma": "medgemma_predictions_reason.csv",
    "dermatollama": "dermato_llama_predictions_reason.csv",
}


def _perturb_edits():
    name = "perturb_generality_edits"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _PERTURB_EDITS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def collect_user(user, model_short, rationale_mode=None, norm_answer=None, subsample=0):
    return _perturb_edits().collect_user(
        user, model_short, rationale_mode=rationale_mode,
        norm_answer=norm_answer, subsample=subsample,
    )


def collect_all(users, model_short, rationale_mode=None, norm_answer=None, subsample=0):
    return _perturb_edits().collect_all(
        users, model_short, rationale_mode=rationale_mode,
        norm_answer=norm_answer, subsample=subsample,
    )


def collect_locality(users, model_short, subsample=0):
    return _perturb_edits().collect_locality(users, model_short, subsample=subsample)


def reviewed_case_ids(users, model_short):
    """Completed cases these users already saw for this model."""
    annot = C.MODEL_SHORT_TO_ANNOT[model_short]
    user_set = set(users)
    ids = set()
    for rec in load_records(C.ANNOTATIONS_CSV):
        if rec.model != annot or rec.login_id not in user_set or not rec.marked_complete:
            continue
        ids.add(rec.case_id)
    return ids


def predictions_csv(model_short) -> Path:
    """Saved top-3 responses live next to the image symlink target."""
    return C.IMAGE_ROOT.resolve().parent / PRED_FILES[model_short]


def collect_ood(users, model_short, subsample=0):
    """Combined MIDAS cases outside the reviewed pool, with saved top-1 and y16."""
    csv.field_size_limit(min(sys.maxsize, 10 ** 7))
    reviewed = reviewed_case_ids(users, model_short)
    path = predictions_csv(model_short)
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for rec in csv.DictReader(handle):
            if (rec.get("image_mode") or "").strip() != "combined":
                continue
            case_id = (rec.get("id") or "").strip()
            y16 = (rec.get("y16") or "").strip()
            if not case_id or case_id in reviewed or not y16:
                continue
            image = C.image_path_for(case_id)
            if not image.exists():
                continue
            rows.append({
                "uid": f"ood__{C.MODEL_SHORT_TO_ANNOT[model_short]}__{case_id}",
                "image": str(image),
                "question": C.QUESTION,
                "answer": y16,
                "rationale": "",
                "cot": "",
                "login_id": "",
                "model": C.MODEL_SHORT_TO_ANNOT[model_short],
                "case_id": case_id,
                "model_top1": "",
                "human_top1_raw": y16,
                "human_top1_y16": y16,
                "pred_before": extract_top1(rec.get("reason_classify") or ""),
            })
    rows.sort(key=lambda row: row["case_id"])
    if subsample:
        rows = rows[:subsample]
    return rows
