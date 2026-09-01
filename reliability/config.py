"""Central configuration for the derm ReasonEdit experiment.

Everything is keyed by a switchable (user, model) pair so the same pipeline can
be re-run for any dermatologist / target model without code changes.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (resolved relative to the derm_vlm_edit repo so the symlinks are used)
# ---------------------------------------------------------------------------
# derm_vlm_edit/
#   data/                                           (folder of symlinks into the preserved repo)
#     annotations_export.csv -> /scratch/jq2uw/derm_vlms/data/annotations_export.csv
#     images                 -> /scratch/jq2uw/derm_vlms/results/images
#     midas_share_dictionary.json -> /scratch/jq2uw/derm_vlms/data_share/midas_share_dictionary.json
#   reasonedit/                                     (slim ReasonEdit: editor + VQA wrapper + 2 VLMs)
#   reliability/                                    (this package: acc-only experiment)
# NOTE: data/ symlinks target the absolute repo path, so no top-level
#       `derm_vlms` / `reasonedit` symlink is required.
DERM_EDIT_ROOT = Path(__file__).resolve().parent.parent          # derm_vlm_edit/
PKG_ROOT = Path(__file__).resolve().parent                        # reliability/

DATA_ROOT = DERM_EDIT_ROOT / "data"                              # symlinks into the preserved repo
ANNOTATIONS_CSV = DATA_ROOT / "annotations_export.csv"
IMAGE_ROOT = DATA_ROOT / "images"                                # {case_id}.jpg
MIDAS_DICT = DATA_ROOT / "midas_share_dictionary.json"

RESULTS_DIR = PKG_ROOT / "results"                               # run JSONs, edit sets, report.csv
OUTPUT_DIR = RESULTS_DIR                                         # edit_set / summary live here too

# Slim ReasonEdit package (editor + VQAModel + MedGemma/DermatoLlama loaders)
REASONEDIT_PKG = DERM_EDIT_ROOT / "reasonedit"

# CLI short model name <-> annotation `model` column value
MODEL_SHORT_TO_ANNOT = {
    "medgemma": "MedGemma",
    "dermatollama": "DermatoLlama",
    "dermato_llama": "DermatoLlama",
}

# ---------------------------------------------------------------------------
# Experiment switches
# ---------------------------------------------------------------------------
# Target models we support for this experiment (annotation `model` column values).
TARGET_MODELS = ("MedGemma", "DermatoLlama")

# Default (user, model) pair for the first end-to-end run. Switch freely.
DEFAULT_USER = "jesse.y.liu.md@gmail.com"
DEFAULT_MODEL = "DermatoLlama"

# Single-answer diagnosis question used for both editing and evaluation.
QUESTION = "What is the single most likely diagnosis for this lesion?"

# Only rows the annotator finished are considered.
REQUIRE_MARKED_COMPLETE = True

# How to reconstruct the human rationale from the per-line edit JSON.
# Rewriting the model's rationale for the WRONG diagnosis is optional for the
# annotator: the human ground truth is the *new* diagnosis + the *new* reasoning
# they authored. Unchanged model sentences (which still argue for the wrong
# diagnosis) are leftovers, not ground truth -> default to human-authored lines.
#   "edited_only" -> only lines the human actually changed (edits_made == True)  [human ground truth]
#   "final"       -> all non-empty `edited` lines (human obs + kept model text)
RATIONALE_MODE = "edited_only"

# Display/report cap on before-edit successes (per user x VLM). Live
# re-generation can match the human answer even though the edit set is
# recorded top-1 errors; we cap so Before does not look like a real baseline.
CAP_BEFORE_SUCCESSES = 0


def image_path_for(case_id: str) -> Path:
    """Resolve the image file for an annotation `case_id` (e.g. '1025_combined')."""
    return IMAGE_ROOT / f"{case_id}.jpg"


def edit_set_path(user: str, model: str) -> Path:
    """Output path for a per-(user, model) edit set (JSONL)."""
    safe_user = user.replace("@", "_at_").replace(".", "_")
    return OUTPUT_DIR / f"edit_set__{safe_user}__{model}.jsonl"


def summary_path(user: str, model: str) -> Path:
    safe_user = user.replace("@", "_at_").replace(".", "_")
    return OUTPUT_DIR / f"summary__{safe_user}__{model}.json"


# ---------------------------------------------------------------------------
# ReasonEdit model-config notes (for Phase 3; not used in Phase 1)
# ---------------------------------------------------------------------------
# Per user guidance, both new models use dual_sbert embeddings with:
#   - mode = "dual_sbert"
#   - inner_params_vision = a LATE vision-block MLP layer (rough convention)
#   - lang_scaler_sbert = 1  (concatenate SBERT lang embedding, unit-scaled)
REASONEDIT_MODEL_HINTS = {
    "MedGemma": {
        "name": "google/medgemma-1.5-4b-it",
        "mode": "dual_sbert",
        "lang_scaler_sbert": 1,
    },
    "DermatoLlama": {
        "name": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "adapter": "DermaVLM/DermatoLLama-full",
        "mode": "dual_sbert",
        "lang_scaler_sbert": 1,
    },
}
