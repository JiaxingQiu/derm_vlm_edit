"""Build a ReasonEdit-schema edit set for a given (user, model).

Output: one JSONL record per top-1 error, matching what ReasonEdit consumes
(`uid, image, question, answer, rationale, cot`) plus bookkeeping fields, and a
summary JSON with counts and normalization coverage.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402
from data_engineering.parse_annotations import (  # noqa: E402
    AnnotationRecord,
    filter_records,
    load_records,
    reconstruct_reasoning,
)
from data_engineering.normalize_dx import normalize, to_dict  # noqa: E402


def record_to_edit(rec: AnnotationRecord, rationale_mode: str) -> dict:
    """Convert one annotation record into a ReasonEdit-schema edit example."""
    nd = normalize(rec.human_top1_raw)
    md = normalize(rec.model_top1)
    reasoning = reconstruct_reasoning(rec.reasoning_lines, mode=rationale_mode)
    safe_user = rec.login_id.replace("@", "_at_").replace(".", "_")
    uid = f"{safe_user}__{rec.model}__{rec.case_id}"
    return {
        # ---- ReasonEdit-consumed fields ----
        "uid": uid,
        "image": str(C.image_path_for(rec.case_id)),
        "question": C.QUESTION,
        "answer": nd.norm,            # human top-1 (canonical string)
        "rationale": reasoning,       # human-edited reasoning
        "cot": reasoning,
        # ---- bookkeeping / eval ----
        "login_id": rec.login_id,
        "model": rec.model,
        "case_id": rec.case_id,
        "model_top1": rec.model_top1,
        "model_top1_norm": md.norm,
        "model_top1_y16": md.y16,
        "human_top1_raw": rec.human_top1_raw,
        "human_top1_y16": nd.y16,
        "human_top1_is_multi": nd.is_multi,
        "diagnosis_order": rec.diagnosis_order,
        "n_reasoning_lines": len(rec.reasoning_lines),
        "n_human_edited_lines": sum(1 for ln in rec.reasoning_lines if ln.edits_made),
        "image_exists": C.image_path_for(rec.case_id).exists(),
    }


def _keep_for_mode(e: dict, norm_answer: str) -> bool:
    """Keep the case if it is an error at this scoring grain."""
    if norm_answer == "y16":
        my, hy = e.get("model_top1_y16"), e.get("human_top1_y16")
        return my is not None and hy is not None and my != hy
    return bool(e.get("model_top1_norm") and e.get("answer")
                and e["model_top1_norm"] != e["answer"])


def build(
    user: str = C.DEFAULT_USER,
    model: str = C.DEFAULT_MODEL,
    rationale_mode: str = C.RATIONALE_MODE,
    write: bool = True,
    norm_answer: str = C.NORM_ANSWER,
) -> tuple[list[dict], dict]:
    all_records = load_records(C.ANNOTATIONS_CSV)
    recs = filter_records(
        all_records,
        user=user,
        model=model,
        require_complete=C.REQUIRE_MARKED_COMPLETE,
        top1_errors_only=True,
        require_correction=True,
    )
    raw = [record_to_edit(r, rationale_mode) for r in recs]
    edits = [e for e in raw if _keep_for_mode(e, norm_answer)]

    summary = {
        "user": user,
        "model": model,
        "rationale_mode": rationale_mode,
        "norm_answer": norm_answer,
        "question": C.QUESTION,
        "fact_template": C.FACT_TEMPLATE,
        "n_marked_incorrect": len(raw),
        "n_edits": len(edits),
        "n_dropped_same_grain": len(raw) - len(edits),
        "n_images_missing": sum(1 for e in edits if not e["image_exists"]),
        "n_multi_dx_top1": sum(1 for e in edits if e["human_top1_is_multi"]),
        "n_y16_unmapped": sum(1 for e in edits if e["human_top1_y16"] is None),
        "n_empty_rationale": sum(1 for e in edits if not e["rationale"].strip()),
        "human_top1_norm_counts": Counter(e["answer"] for e in edits).most_common(),
        "human_top1_y16_counts": Counter(
            e["human_top1_y16"] or "<unmapped>" for e in edits
        ).most_common(),
        "model_top1_counts": Counter(e["model_top1"] for e in edits).most_common(),
    }

    if write:
        os.makedirs(C.OUTPUT_DIR, exist_ok=True)
        with open(C.edit_set_path(user, model), "w", encoding="utf-8") as f:
            for e in edits:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        with open(C.summary_path(user, model), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    return edits, summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build a ReasonEdit edit set from human feedback")
    ap.add_argument("--user", default=C.DEFAULT_USER)
    ap.add_argument("--model", default=C.DEFAULT_MODEL, choices=list(C.TARGET_MODELS))
    ap.add_argument("--rationale_mode", default=C.RATIONALE_MODE, choices=["final", "edited_only"])
    ap.add_argument("--norm_answer", default=C.NORM_ANSWER, choices=["y16", "human"])
    args = ap.parse_args()

    edits, summary = build(args.user, args.model, args.rationale_mode, norm_answer=args.norm_answer)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(edits)} edits -> {C.edit_set_path(args.user, args.model)}")
