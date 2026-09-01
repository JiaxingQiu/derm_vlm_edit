"""Parse the annotation export into structured human-ground-truth records.

Each annotation row = one (annotator x case x model) review of the model's
top-3 differential. For the "simplest" top-1 experiment we reconstruct, per row:

  - model_top1        : model's #1 differential (diag_1_name)
  - is_top1_error     : human marked diag_1 incorrect
  - human_top1_raw    : human's corrected diagnosis (diag_1_correct_differential)
  - human_reasoning   : reconstructed human-edited reasoning for diag_1

The per-line reasoning edits live in `reasoning_1` as a JSON list of
{original, edited, edits_made}. Observed convention: the first entry's
`original` is "deleted" and the human writes their own leading observation.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# Allow very large CSV fields (reasoning/raw_response can be long).
csv.field_size_limit(min(sys.maxsize, 10 ** 7))


@dataclass
class ReasoningLine:
    original: str
    edited: str
    edits_made: bool


@dataclass
class AnnotationRecord:
    login_id: str
    case_id: str
    model: str
    marked_complete: bool
    model_top1: str
    diag_1_label: str
    human_top1_raw: str
    is_top1_error: bool
    reasoning_lines: List[ReasoningLine] = field(default_factory=list)
    diagnosis_order: str = ""
    other_feedback: str = ""


def _parse_reasoning(raw_json: str) -> List[ReasoningLine]:
    if not raw_json or not raw_json.strip():
        return []
    try:
        arr = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return []
    out: List[ReasoningLine] = []
    for d in arr:
        if not isinstance(d, dict):
            continue
        out.append(
            ReasoningLine(
                original=str(d.get("original", "")),
                edited=str(d.get("edited", "")),
                edits_made=bool(d.get("edits_made", False)),
            )
        )
    return out


def reconstruct_reasoning(lines: List[ReasoningLine], mode: str = "final") -> str:
    """Rebuild the human rationale from per-line edits.

    mode="final"       -> join all non-empty `edited` lines (human's final state)
    mode="edited_only" -> join only lines the human actually changed
    """
    picked: List[str] = []
    for ln in lines:
        e = ln.edited.strip()
        if not e or e.lower() == "deleted":
            continue
        if mode == "edited_only" and not ln.edits_made:
            continue
        picked.append(e)
    return " ".join(picked).strip()


def load_records(csv_path) -> List[AnnotationRecord]:
    """Load and structure every annotation row (no filtering)."""
    records: List[AnnotationRecord] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            records.append(
                AnnotationRecord(
                    login_id=row.get("login_id", "").strip(),
                    case_id=row.get("case_id", "").strip(),
                    model=row.get("model", "").strip(),
                    marked_complete=(row.get("marked_complete", "").strip() == "True"),
                    model_top1=row.get("diag_1_name", "").strip(),
                    diag_1_label=row.get("diag_1_label", "").strip(),
                    human_top1_raw=row.get("diag_1_correct_differential", "").strip(),
                    is_top1_error=(row.get("diag_1_label", "").strip() == "incorrect"),
                    reasoning_lines=_parse_reasoning(row.get("reasoning_1", "")),
                    diagnosis_order=row.get("diagnosis_order", "").strip(),
                    other_feedback=row.get("other_feedback", "").strip(),
                )
            )
    return records


def filter_records(
    records: List[AnnotationRecord],
    user: Optional[str] = None,
    model: Optional[str] = None,
    require_complete: bool = True,
    top1_errors_only: bool = True,
    require_correction: bool = True,
) -> List[AnnotationRecord]:
    out = []
    for r in records:
        if user is not None and r.login_id != user:
            continue
        if model is not None and r.model != model:
            continue
        if require_complete and not r.marked_complete:
            continue
        if top1_errors_only and not r.is_top1_error:
            continue
        if require_correction and not r.human_top1_raw:
            continue
        out.append(r)
    return out
