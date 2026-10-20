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
#   editing/shared/                                  (this config + data engineering)
#   editing/reliability/                              (reliability metric)
#   editing/text_generality/                          (text generality metric)
#   editing/image_generality/                         (image generality metric)
# NOTE: data/ symlinks target the absolute repo path, so no top-level
#       `derm_vlms` / `reasonedit` symlink is required.
SHARED_ROOT = Path(__file__).resolve().parent                      # editing/shared/
EDITING_ROOT = SHARED_ROOT.parent                                 # editing/
DERM_EDIT_ROOT = EDITING_ROOT.parent                              # derm_vlm_edit/

DATA_ROOT = DERM_EDIT_ROOT / "data"                              # symlinks into the preserved repo
ANNOTATIONS_CSV = DATA_ROOT / "annotations_export.csv"
IMAGE_ROOT = DATA_ROOT / "images"                                # {case_id}.jpg
MIDAS_DICT = DATA_ROOT / "midas_share_dictionary.json"

# Each metric writes its own results under editing/<metric>/results/.
# edit_set and summary JSONLs live under reliability/results/ (the first
# metric that builds them), but the paths are overridable.
RESULTS_DIR = EDITING_ROOT / "reliability" / "results"
OUTPUT_DIR = RESULTS_DIR

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

# Same top-3 differential prompt as the annotation interface
# (collect_ai_response/{medgemma,dermato_llama}/predict_reason.py).
# Generation is top-3 + reasoning; eval still extracts and scores top-1 only.
QUESTION = (
    "You are an expert dermatologist. Please give the top 3 diagnoses in "
    "your differential, and provide reasoning for each diagnosis, in format:\n"
    "1. [Diagnosis]: [Reasoning]\n"
    "2. [Diagnosis]: [Reasoning]\n"
    "3. [Diagnosis]: [Reasoning]"
)

# ReasonEdit codebook value (answer-only IKE). Retrieval key is still
# <image, QUESTION>; this is the sentence prepended at generate time.
FACT_TEMPLATE = "New Fact: The top 1 most likely differential is {answer}."

# Top-3 + reasoning needs a long decode; original collection used 512.
MAX_NEW_TOKENS = 512

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
# How to decide membership + correctness.
#   "y16"   -> edit set = y16(model) ≠ y16(human); score by y16 equality
#   "human" -> edit set = normalized strings differ; score by exact canonical text
NORM_ANSWER = "y16"
CAP_BEFORE_SUCCESSES = None  # int = cap reported Before; None = use live n_correct_pre


def cap_before(n: int) -> int:
    if CAP_BEFORE_SUCCESSES is None:
        return int(n)
    return min(int(CAP_BEFORE_SUCCESSES), int(n))

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
