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
| `soft_normalize.py` | CLI wrapper (same logic, `--no-llm` / `--dry-run` flags) |
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

**Notebook** (recommended):
```bash
cd derm_vlm_edit/data_norm
jupyter notebook soft_normalize.ipynb
```
Set `USE_LLM = True` or `False` in cell 2. Run all cells.

**CLI**:
```bash
cd derm_vlm_edit
python data_norm/soft_normalize.py            # rule + LLM
python data_norm/soft_normalize.py --no-llm   # rule only
python data_norm/soft_normalize.py --dry-run  # list mismatches, no API calls
```

## Output

Written to `data/normed/`:

- `edits_soft_normalized.csv` — one row per correction with `human_top1_y16_soft`
- `llm_judgments.json` — cached LLM verdicts (reruns only call the API for new mismatches)

> **Note:** Stage 2 and stage 3 do not read this CSV yet. They rebuild edits
> from `annotations_export.csv` using the rule-based `y16` label directly.
