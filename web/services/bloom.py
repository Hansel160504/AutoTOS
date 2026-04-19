"""Bloom-level → TOS category mapping.

Single source of truth, replacing the duplicated dicts in ai_model.py and
dashboard.py. The categories are the three columns of a TOS:

    fam = Familiarisation  (Remembering, Understanding)
    int = Integration      (Applying, Analyzing)
    cre = Creation         (Evaluating, Creating)
"""
from __future__ import annotations

from typing import Dict, Tuple

# Canonical capitalised Bloom levels — must match the dataset and ai_model.
CANONICAL_BLOOMS: Tuple[str, ...] = (
    "Remembering", "Understanding",
    "Applying",    "Analyzing",
    "Evaluating",  "Creating",
)

# Anything the model or dataset might return → bucket
BLOOM_TO_CAT: Dict[str, str] = {
    "Remembering":   "fam", "Knowledge":      "fam",
    "Understanding": "fam", "Understand":     "fam",
    "Applying":      "int", "Apply":          "int",
    "Analyzing":     "int", "Analyze":        "int", "Analysing": "int",
    "Evaluating":    "cre", "Evaluate":       "cre",
    "Creating":      "cre", "Create":         "cre",
}

# Bucket → starting bloom for a generation slot.
SLOT_BLOOM_FOR_BUCKET: Dict[str, str] = {
    "fam": "remembering",
    "int": "applying",
    "cre": "creating",
}

# Default mix per subject type.
SUBJECT_DEFAULTS: Dict[str, Tuple[int, int, int]] = {
    "lab":    (20, 30, 50),
    "nonlab": (50, 30, 20),
}


def bucket_for_bloom(bloom: str) -> str:
    """Map a bloom string to its TOS bucket. Unknown → 'fam' (safest default)."""
    return BLOOM_TO_CAT.get((bloom or "").strip(), "fam")


def defaults_for(subject_type: str) -> Tuple[int, int, int]:
    """(fam_pct, int_pct, cre_pct) for a subject type. Falls back to nonlab."""
    return SUBJECT_DEFAULTS.get(subject_type, SUBJECT_DEFAULTS["nonlab"])