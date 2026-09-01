"""Aggregate reliability run JSONs into a PI-ready before/after table.

Reads every ``results/reliability__{user}__{model}.json`` and writes one
wide table (the same layout as the PI display) to:

    results/report.csv

Layout: rows = users, column groups = VLMs, subcolumns = Before | After | d.
Cells are ``% (k/N)``.

    python make_report.py                 # all users found, both VLMs
    python make_report.py --users jesse.y.liu.md@gmail.com achiou@stanford.edu lhunterellul
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
import config as C

# VLM column groups: (short_model key in results JSON, display name)
VLMS = [("medgemma", "MedGemma"), ("dermatollama", "DermatoLlama")]

# Default users (the reliable annotators) in row order.
REPORT_USERS = [
    "jesse.y.liu.md@gmail.com",
    "achiou@stanford.edu",
    "lhunterellul",
]

# Display cap on before-edit successes (per user x VLM). Same constant as
# run_efficacy / the notebook so regenerating the table always caps.
CAP_BEFORE_SUCCESSES = C.CAP_BEFORE_SUCCESSES


def _short_user(user: str) -> str:
    """Compact display label for a user id/email."""
    return user.split("@")[0]


def load_results(results_dir: Path):
    """Return {(user, short_model): result_dict} for all report JSONs found."""
    out = {}
    for p in sorted(results_dir.glob("reliability__*.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        out[(r["user"], r["short_model"])] = r
    return out


def _cell_counts(r):
    """Return (N, n_before, n_after) from a result dict.

    Before-edit successes are capped at CAP_BEFORE_SUCCESSES. The edit set is
    the model's recorded top-1 errors, so live re-generation matching the human
    answer is a decoding-parity artifact, not a real baseline — capping keeps
    the table from looking like a non-zero pre-edit accuracy.
    """
    n = int(r.get("n_edits", 0))
    nb = int(r.get("n_correct_pre", 0))
    na = int(r.get("n_correct_after", r.get("n_correct_post", 0)))
    nb = min(CAP_BEFORE_SUCCESSES, nb)
    return n, nb, na


def _pct(k, n):
    return (100.0 * k / n) if n else 0.0


def _fmt(k, n):
    """`% (k/N)` cell string, or a dash when there is no data."""
    if n == 0:
        return "--"
    return f"{_pct(k, n):.0f}% ({k}/{n})"


def build_rows(results, users, vlms):
    """Per-user rows + a pooled (micro-average) row.

    Each row: {user, cells: {short_model: (N, nb, na)}}.
    """
    rows = []
    pooled = {sm: [0, 0, 0] for sm, _ in vlms}  # N, before, after
    for user in users:
        cells = {}
        for sm, _ in vlms:
            r = results.get((user, sm))
            if r is None:
                cells[sm] = None
                continue
            n, nb, na = _cell_counts(r)
            cells[sm] = (n, nb, na)
            pooled[sm][0] += n
            pooled[sm][1] += nb
            pooled[sm][2] += na
        rows.append({"user": user, "cells": cells})
    pooled_cells = {sm: tuple(pooled[sm]) if pooled[sm][0] else None for sm, _ in vlms}
    return rows, {"user": "Pooled", "cells": pooled_cells}


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def render_markdown(rows, pooled, vlms):
    # markdown can't span columns cleanly, so flatten to explicit per-VLM columns
    cols = ["User"]
    for _, name in vlms:
        cols += [f"{name} Before", f"{name} After", f"{name} Δ"]

    def row_cells(row):
        out = [_short_user(row["user"]) if row["user"] != "Pooled" else "**Pooled**"]
        for sm, _ in vlms:
            c = row["cells"].get(sm)
            if c is None:
                out += ["--", "--", "--"]
            else:
                n, nb, na = c
                d = _pct(na, n) - _pct(nb, n)
                out += [_fmt(nb, n), _fmt(na, n), f"+{d:.0f}"]
        return out

    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row_cells(row)) + " |")
    lines.append("| " + " | ".join(row_cells(pooled)) + " |")
    return "\n".join(lines) + "\n"


def render_latex(rows, pooled, vlms):
    ncol = len(vlms)
    colspec = "l" + "".join(["ccc"] * ncol)
    head_groups = " & " + " & ".join(
        rf"\multicolumn{{3}}{{c}}{{{name}}}" for _, name in vlms
    ) + r" \\"
    # cmidrules under each group
    cmids = ""
    start = 2
    for _ in vlms:
        cmids += rf"\cmidrule(lr){{{start}-{start+2}}} "
        start += 3
    subhead = "User & " + " & ".join(["Before & After & $\\Delta$"] * ncol) + r" \\"

    def row_cells(row, bold=False):
        name = _short_user(row["user"]) if row["user"] != "Pooled" else "Pooled"
        cells = [rf"\textbf{{{name}}}" if bold else name]
        for sm, _ in vlms:
            c = row["cells"].get(sm)
            if c is None:
                cells += ["--", "--", "--"]
            else:
                n, nb, na = c
                d = _pct(na, n) - _pct(nb, n)
                cells += [f"{_pct(nb, n):.0f}\\% ({nb}/{n})",
                          f"{_pct(na, n):.0f}\\% ({na}/{n})",
                          f"+{d:.0f}"]
        return " & ".join(cells) + r" \\"

    body = "\n".join(row_cells(r) for r in rows)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{ReasonEdit top-1 reliability before vs.\ after editing. "
        r"The edit set is each model's top-1 errors (Before $\approx 0$ by construction); "
        r"After is the fraction corrected to the human answer. Cells: \% (k/N).}",
        r"\label{tab:reliability}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        head_groups,
        cmids.rstrip(),
        subhead,
        r"\midrule",
        body,
        r"\midrule",
        row_cells(pooled, bold=True),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def render_csv(rows, pooled, vlms, path):
    """One wide table matching the displayed report: User | VLM Before | After | Δ | ..."""
    cols = ["User"]
    for _, name in vlms:
        cols += [f"{name} Before", f"{name} After", f"{name} Δ"]

    def row_cells(row):
        out = [_short_user(row["user"]) if row["user"] != "Pooled" else "Pooled"]
        for sm, _ in vlms:
            c = row["cells"].get(sm)
            if c is None:
                out += ["--", "--", "--"]
            else:
                n, nb, na = c
                d = _pct(na, n) - _pct(nb, n)
                out += [_fmt(nb, n), _fmt(na, n), f"+{d:.0f}"]
        return out

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in rows:
            w.writerow(row_cells(row))
        w.writerow(row_cells(pooled))


def main():
    ap = argparse.ArgumentParser(description="Build PI-ready before/after reliability table")
    ap.add_argument("--users", nargs="*", default=None,
                    help="user ids in row order (default: the 3 reliable annotators)")
    ap.add_argument("--results_dir", default=str(C.RESULTS_DIR))
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results = load_results(results_dir)
    if not results:
        print(f"No results found in {results_dir} (run run_efficacy.py / edit_run.ipynb first).")
        return

    users = args.users or [u for u in REPORT_USERS if any((u, sm) in results for sm, _ in VLMS)]
    if not users:  # fall back to whatever users are present
        users = sorted({u for (u, _) in results})

    rows, pooled = build_rows(results, users, VLMS)

    md = render_markdown(rows, pooled, VLMS)
    csv_path = results_dir / "report.csv"
    render_csv(rows, pooled, VLMS, csv_path)

    print(md)
    print(f"wrote: {csv_path}")


if __name__ == "__main__":
    main()
