"""Check edit-set human corrections against MIDAS pathology ground truth.

Each edit set JSONL (one per annotator × model) is a filtered record where
the annotator marked the model top-1 wrong and provided a correction.
The `answer` field is the normalized correction that ReasonEdit uses.
We check whether `human_top1_y16` matches the MIDAS `gt_y16`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

SHARED = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

import config as C  # noqa: E402

DATA_SHARE = C.MIDAS_DICT.resolve().parent
EDIT_SET_DIR = C.OUTPUT_DIR


def _gt_by_case() -> dict:
    """case_id -> MIDAS pathology y16 (None when ambiguous)."""
    midas = pd.read_parquet(DATA_SHARE / "midas_share.parquet", columns=["lesion_id", "y16"])
    mapping = pd.read_parquet(DATA_SHARE / "case_mapping.parquet", columns=["case_id", "lesion_id"])
    per_lesion: dict = {}
    for lid, y16 in zip(midas["lesion_id"], midas["y16"]):
        per_lesion.setdefault(lid, set()).add(y16)
    resolved = {k: next(iter(v)) if len(v) == 1 else None for k, v in per_lesion.items()}
    return {cid: resolved.get(lid) for cid, lid in zip(mapping["case_id"], mapping["lesion_id"])}


def load_edit_sets() -> pd.DataFrame:
    """Load all edit-set JSONLs and attach MIDAS ground truth."""
    gt = _gt_by_case()
    rows = []
    for path in sorted(EDIT_SET_DIR.glob("edit_set__*.jsonl")):
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                rows.append({
                    "annotator": e["login_id"],
                    "model": e["model"],
                    "case_id": e["case_id"],
                    "model_top1": e["model_top1"],
                    "model_y16": e.get("model_top1_y16"),
                    "human_correction": e["human_top1_raw"],
                    "human_y16": e.get("human_top1_y16"),
                    "gt_y16": gt.get(e["case_id"]),
                })
    df = pd.DataFrame(rows)
    df["scorable"] = df["human_y16"].notna() & df["gt_y16"].notna()
    df["correct"] = df["scorable"] & (df["human_y16"] == df["gt_y16"])
    return df


def _summarize(df: pd.DataFrame) -> dict:
    s = df[df["scorable"]]
    n, c = len(s), int(s["correct"].sum())
    return {"Correct": c, "Total": n, "Acc%": f"{c/n*100:.1f}" if n else "—",
            "Skipped": len(df) - n or ""}


def accuracy_tables(df: pd.DataFrame) -> dict:
    """Return {section: DataFrame} for overall, per-annotator, per-model, per-annotator×model."""
    tables = {}
    tables["Overall"] = pd.DataFrame([_summarize(df)], index=["All edits"])

    tables["By annotator"] = pd.DataFrame(
        {a: _summarize(g) for a, g in df.groupby("annotator")}
    ).T.rename_axis("Annotator")

    tables["By model"] = pd.DataFrame(
        {m: _summarize(g) for m, g in df.groupby("model")}
    ).T.rename_axis("Model")

    tables["By annotator × model"] = pd.DataFrame(
        {f"{a} / {m}": _summarize(g)
         for (a, m), g in df.groupby(["annotator", "model"])}
    ).T.rename_axis("Annotator / Model")

    return tables


def error_table(df: pd.DataFrame) -> pd.DataFrame:
    return (df[df["scorable"] & ~df["correct"]]
            [["case_id", "annotator", "model", "model_top1", "model_y16",
              "human_correction", "human_y16", "gt_y16"]]
            .reset_index(drop=True))
