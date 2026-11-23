# Stage 1 — Soft-normalize annotator corrections

Map each completed top-1 correction onto the 16-class MIDAS `y16` label.

Rule-based `normalize()` handles abbreviations and synonyms. MIDAS pathology
ground truth is joined on `case_id`. Where the rule label already matches
pathology, it is kept. Where there is no pathology label, the rule label is
kept. Only mismatches are sent to GPT-4o, which decides whether the free text
is a reasonable name for that pathology category. The result column is
`human_top1_y16_soft`.

## Files

| File | Role |
|---|---|
| `soft_normalize.ipynb` | **Interactive notebook** — run this to inspect and produce the CSV |
| `gt.py` | Load MIDAS ground-truth `y16` keyed by `case_id` |
| `llm_judge.py` | GPT-4o judge: does the free text match the pathology category? |
| `tokens.py` | Azure OpenAI credentials (gitignored — create locally) |
| `annotator_accuracy.ipynb` | Diagnostic: score edit-set corrections against MIDAS |
| `utils.py` | Helper used by the accuracy notebook |

## Inputs

Resolved from `editing/shared/config.py`:

- `data/annotations_export.csv` — annotation export
- `data/midas_share.parquet`, `data/case_mapping.parquet` — MIDAS ground truth

## Run

```bash
cd derm_vlm_edit/data_norm
jupyter notebook soft_normalize.ipynb
```
Set `USE_LLM = True` or `False` in cell 2. Run all cells.

With `USE_LLM = True`, every run calls GPT-4o on the current mismatches. A previous `llm_judgments.json` is not reused, so new annotations are judged when the notebook is rerun. The same free text and pathology label is asked only once inside that run.

## Output

Written to `data/normed/`:

- `edits_soft_normalized.csv` — one row per correction with `human_top1_y16_soft`
- `llm_judgments.json` — verdicts from this run (a record, not an input to the next run)

> **Note:** Stage 2 and stage 3 do not read this CSV yet. They rebuild edits
> from `annotations_export.csv` using the rule-based `y16` label directly.
