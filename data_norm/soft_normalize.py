"""Soft normalization: rule-based y16 + LLM fallback for mismatches.

Reads annotations_export.csv, applies rule-based normalize(), joins MIDAS gt_y16,
and for mismatches asks an LLM whether the human free text reasonably matches the
pathology category. Outputs a CSV with an extra `human_top1_y16_soft` column.

Usage:
    python soft_normalize.py                    # run LLM on mismatches
    python soft_normalize.py --dry-run          # show what would be sent to LLM
    python soft_normalize.py --no-llm           # rule-based only, no API calls
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SHARED = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

import config as C  # noqa: E402
from data_engineering.normalize_dx import normalize  # noqa: E402
from data_engineering.parse_annotations import filter_records, load_records  # noqa: E402

from gt import load as load_gt  # noqa: E402

DERM_EDIT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = DERM_EDIT_ROOT / "data" / "normed"
OUT_CSV = OUT_DIR / "edits_soft_normalized.csv"
LLM_CACHE = OUT_DIR / "llm_judgments.json"

COLUMNS = [
    "login_id", "case_id", "model",
    "model_top1", "human_top1_raw", "human_top1_is_multi",
    "human_top1_norm", "human_top1_y16_rule", "gt_y16",
    "llm_match", "llm_reason",
    "human_top1_y16_soft",
]


def _load_llm_cache() -> dict:
    if LLM_CACHE.exists():
        return json.loads(LLM_CACHE.read_text())
    return {}


def _save_llm_cache(cache: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LLM_CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def build_rows() -> list[dict]:
    """All completed top-1 corrections with rule-based norm + GT attached."""
    gt = load_gt()
    records = filter_records(
        load_records(C.ANNOTATIONS_CSV),
        require_complete=True,
        top1_errors_only=True,
        require_correction=True,
    )
    rows = []
    for rec in records:
        nd = normalize(rec.human_top1_raw)
        md = normalize(rec.model_top1)
        rows.append({
            "login_id": rec.login_id,
            "case_id": rec.case_id,
            "model": rec.model,
            "model_top1": rec.model_top1,
            "human_top1_raw": rec.human_top1_raw,
            "human_top1_is_multi": nd.is_multi,
            "human_top1_norm": nd.norm,
            "human_top1_y16_rule": nd.y16,
            "gt_y16": gt.get(rec.case_id),
            "llm_match": None,
            "llm_reason": None,
            "human_top1_y16_soft": None,
        })
    return rows


def apply_soft_norm(rows: list[dict], *, use_llm: bool = True, dry_run: bool = False) -> list[dict]:
    """Fill human_top1_y16_soft: gt if rule matches or LLM says yes, else rule."""
    cache = _load_llm_cache()
    n_api = 0

    for row in rows:
        rule = row["human_top1_y16_rule"]
        gt = row["gt_y16"]

        # No GT available — fall back to rule
        if gt is None:
            row["human_top1_y16_soft"] = rule
            continue

        # Rule already matches GT — no LLM needed
        if rule == gt:
            row["llm_match"] = True
            row["human_top1_y16_soft"] = gt
            continue

        # Mismatch — ask LLM
        cache_key = f"{row['human_top1_raw']}||{gt}"

        if cache_key in cache:
            verdict = cache[cache_key]
        elif not use_llm or dry_run:
            row["human_top1_y16_soft"] = rule
            continue
        else:
            from llm_judge import judge
            verdict = judge(row["human_top1_raw"], gt)
            cache[cache_key] = verdict
            n_api += 1

        row["llm_match"] = verdict.get("match", False)
        row["llm_reason"] = verdict.get("reason", "")
        row["human_top1_y16_soft"] = gt if verdict.get("match") else rule

    if n_api:
        _save_llm_cache(cache)
        print(f"  {n_api} LLM API calls (cached for next run)")

    return rows


def write_csv(rows: list[dict]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {OUT_CSV}")


def main():
    ap = argparse.ArgumentParser(description="Soft-normalize annotator corrections")
    ap.add_argument("--no-llm", action="store_true", help="Rule-based only")
    ap.add_argument("--dry-run", action="store_true", help="Show mismatches without calling LLM")
    args = ap.parse_args()

    rows = build_rows()
    print(f"{len(rows)} corrections loaded")

    rows = apply_soft_norm(rows, use_llm=not args.no_llm, dry_run=args.dry_run)

    # Summary
    n_rule_match = sum(1 for r in rows if r["human_top1_y16_rule"] == r["gt_y16"])
    n_llm_match = sum(1 for r in rows if r["llm_match"] is True and r["human_top1_y16_rule"] != r["gt_y16"])
    n_no_gt = sum(1 for r in rows if r["gt_y16"] is None)
    n_wrong = len(rows) - n_rule_match - n_llm_match - n_no_gt
    print(f"  Rule match:    {n_rule_match}")
    print(f"  LLM promoted:  {n_llm_match}")
    print(f"  Genuine error: {n_wrong}")
    print(f"  No GT:         {n_no_gt}")

    if args.dry_run:
        print(f"\n{sum(1 for r in rows if r['human_top1_y16_rule'] != r['gt_y16'] and r['gt_y16'] is not None)} rows would be sent to LLM")
    else:
        write_csv(rows)


if __name__ == "__main__":
    main()
