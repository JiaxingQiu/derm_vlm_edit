"""Normalize free-text dermatology diagnosis names.

Human corrections in the annotation export are messy free text
(e.g. "BCC", "dysplastic nevus", "Lichenoid keratosis", "Blue nevus vs melanoma").
This module canonicalizes them and, when confident, maps to the 16-class MIDAS
`y16` taxonomy. Matching for the efficacy experiment primarily uses `norm`;
`y16` is a best-effort coarse label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Optional


# 16-class MIDAS taxonomy (see data_share/midas_share_dictionary.json -> y16)
CANON_Y16 = [
    "Basal Cell Carcinoma",
    "Melanocytic Nevus",
    "Other Benign",
    "Seborrheic Keratosis",
    "Melanoma",
    "Squamous Cell Carcinoma",
    "Actinic Keratosis",
    "Squamous Cell Carcinoma In Situ",
    "Melanocytic Lesion",
    "Dermatofibroma",
    "Non-neoplastic",
    "Hemangioma",
    "Fibrous Papule",
    "Other Malignant",
    "Melanocytic Tumor",
    "Unknown",
]

# Common dermatology abbreviations -> full canonical phrase (applied token-wise
# after lowercasing). Order matters: multi-word keys are handled in _expand.
_ABBREV = {
    "sccis": "squamous cell carcinoma in situ",
    "scc is": "squamous cell carcinoma in situ",
    "scc in situ": "squamous cell carcinoma in situ",
    "bcc": "basal cell carcinoma",
    "scc": "squamous cell carcinoma",
    "ak": "actinic keratosis",
    "hak": "hypertrophic actinic keratosis",
    "sk": "seborrheic keratosis",
    "eic": "epidermal inclusion cyst",
    "blk": "benign lichenoid keratosis",
    "lplk": "lichenoid keratosis",
}

# Normalized (lowercased, abbrev-expanded) substring -> y16 canonical label.
# Best-effort clinical grouping; ambiguous entries are left unmapped (None).
_Y16_RULES = [
    ("basal cell carcinoma", "Basal Cell Carcinoma"),
    ("squamous cell carcinoma in situ", "Squamous Cell Carcinoma In Situ"),
    ("squamous cell carcinoma", "Squamous Cell Carcinoma"),
    ("keratoacanthoma", "Squamous Cell Carcinoma"),
    ("melanoma", "Melanoma"),
    ("actinic keratosis", "Actinic Keratosis"),
    ("seborrheic keratosis", "Seborrheic Keratosis"),
    ("lichenoid keratosis", "Seborrheic Keratosis"),
    ("dysplastic nevus", "Melanocytic Nevus"),
    ("atypical nevus", "Melanocytic Nevus"),
    ("blue nevus", "Melanocytic Nevus"),
    ("spitz nevus", "Melanocytic Nevus"),
    ("nevus", "Melanocytic Nevus"),
    ("dysplastic", "Melanocytic Nevus"),
    ("lentigo", "Melanocytic Lesion"),
    ("dermatofibroma", "Dermatofibroma"),
    ("fibrous papule", "Fibrous Papule"),
    ("fibroepithelial polyp", "Fibrous Papule"),
    ("hemangioma", "Hemangioma"),
    ("angiosarcoma", "Other Malignant"),
    ("neurofibroma", "Other Benign"),
    ("epidermal inclusion cyst", "Other Benign"),
    ("epidermoid cyst", "Other Benign"),
    ("sebaceous hyperplasia", "Other Benign"),
    ("cyst", "Other Benign"),
    ("wart", "Non-neoplastic"),
    ("acne", "Non-neoplastic"),
    ("rosacea", "Non-neoplastic"),
    ("scar", "Non-neoplastic"),
    ("eczema", "Non-neoplastic"),
    ("dermatitis", "Non-neoplastic"),
    ("prurigo", "Non-neoplastic"),
    ("lichen simplex", "Non-neoplastic"),
    ("comedone", "Non-neoplastic"),
    ("dilated pore", "Non-neoplastic"),
]

# Tokens that indicate the annotator listed several diagnoses (a differential),
# rather than a single top-1 answer.
_MULTI_RE = re.compile(r"\bvs\.?\b|/|,|\bor\b", re.IGNORECASE)


@dataclass
class NormDx:
    raw: str
    norm: str            # canonical single top-1 string (lowercased)
    y16: Optional[str]   # best-effort 16-class label, or None
    is_multi: bool       # True if the raw text listed multiple diagnoses


def _clean(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .;:-")
    return s


def _expand_abbrev(s: str) -> str:
    """Expand whole-string and whole-token abbreviations."""
    if s in _ABBREV:
        return _ABBREV[s]
    # multi-word abbrevs
    for k, v in _ABBREV.items():
        if " " in k and k in s:
            s = s.replace(k, v)
    # single-token abbrevs
    toks = [_ABBREV.get(t, t) for t in s.split(" ")]
    return " ".join(toks)


def _first_component(s: str) -> str:
    """For a multi-diagnosis string, return the first listed diagnosis."""
    parts = re.split(r"\bvs\.?\b|/|,|\bor\b", s, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    return parts[0] if parts else s


def _to_y16(norm: str) -> Optional[str]:
    for substr, label in _Y16_RULES:
        if substr in norm:
            return label
    return None


def normalize(raw: str) -> NormDx:
    """Normalize a free-text diagnosis into a canonical top-1 + coarse y16 label."""
    cleaned = _clean(raw)
    multi = bool(_MULTI_RE.search(cleaned))
    top1 = _first_component(cleaned) if multi else cleaned
    norm = _expand_abbrev(_clean(top1))
    return NormDx(raw=raw, norm=norm, y16=_to_y16(norm), is_multi=multi)


def to_dict(nd: NormDx) -> dict:
    return asdict(nd)


if __name__ == "__main__":  # quick manual check
    samples = [
        "BCC", "dysplastic nevus", "Lichenoid keratosis", "AK", "SCCis",
        "Blue nevus vs melanoma vs recurrent nevus", "Acne", "scar",
        "AK/HAK, SCC in situ, SCC, Benign Lichenoid Keratosis",
    ]
    for s in samples:
        nd = normalize(s)
        print(f"{s!r:55s} -> norm={nd.norm!r:35s} y16={nd.y16!r:30s} multi={nd.is_multi}")
