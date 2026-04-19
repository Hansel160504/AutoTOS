"""Configuration constants and environment loading.

All tunable values live here so other modules import from one place.
"""
from __future__ import annotations

import logging
import os
import pathlib

logger = logging.getLogger(__name__)

# ── Ollama ────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "autotos")
OLLAMA_TIMEOUT  = int(os.environ.get("OLLAMA_TIMEOUT", "300"))

# ── Concurrency ───────────────────────────────────────────────────────
GENERATION_WORKERS = int(os.environ.get("GENERATION_WORKERS", "1"))

# ── Text processing ───────────────────────────────────────────────────
MAX_RETURN    = 50_000
CHUNK_SIZE    = 600
CHUNK_OVERLAP = 20

# ── Caches ────────────────────────────────────────────────────────────
MEM_CACHE_MAX     = 512
DISK_CACHE_MAX    = 2_000
CHUNK_CACHE_MAX   = 64       # was unbounded — fixed memory leak
CACHE_CLEANUP_EVERY = 50

# ── Per-type generation budgets ───────────────────────────────────────
# v31: richer context (~460 tok avg) + MCQ needs room for 4 full choices
NUM_CTX = {"mcq": 1536, "tf": 1536, "open": 1536}
MAX_TOKENS = {"mcq": 300, "tf": 150, "open": 250}

# ── Validation thresholds ─────────────────────────────────────────────
ANSWER_TEXT_MAX_CHARS         = 220
MAX_GEN_ATTEMPTS              = 3
SEMANTIC_DUP_JACCARD_PAIRWISE = 0.65
SEMANTIC_DUP_COMMON_WORDS     = 3
CIRCULAR_CHOICE_JACCARD       = 0.55
TF_SEMANTIC_DUP_JACCARD       = 0.55
MCQ_SUBTOPIC_JACCARD          = 0.50

# ── Type maps (single source of truth) ────────────────────────────────
# v31 training uses: MCQ, True_False, Open_Ended — all display types
# must match EXACTLY or the model sees unfamiliar tokens at inference.
TYPE_INTERNAL: dict[str, str] = {
    "mcq": "mcq", "MCQ": "mcq",
    "truefalse": "tf", "true_false": "tf", "True_False": "tf", "tf": "tf",
    "open_ended": "open", "Open_Ended": "open", "open-ended": "open",
    "openended": "open", "open": "open",
}
TYPE_DISPLAY: dict[str, str] = {
    "mcq": "MCQ", "MCQ": "MCQ",
    "truefalse": "True_False", "true_false": "True_False",
    "True_False": "True_False", "tf": "True_False",
    "open_ended": "Open_Ended", "Open_Ended": "Open_Ended",
    "open-ended": "Open_Ended", "open": "Open_Ended",
}
INTERNAL_TO_DISPLAY: dict[str, str] = {
    "mcq": "MCQ", "tf": "True_False", "open": "Open_Ended",
}

# ── Bloom levels ──────────────────────────────────────────────────────
BLOOM_CANONICAL = frozenset({
    "Remembering", "Understanding", "Applying",
    "Analyzing",   "Evaluating",   "Creating",
})
BLOOM_ALIASES: dict[str, str] = {
    "knowledge": "Remembering", "remember": "Remembering", "remembering": "Remembering",
    "understand": "Understanding", "understanding": "Understanding",
    "apply": "Applying", "applying": "Applying",
    "analyze": "Analyzing", "analyzing": "Analyzing",
    "evaluate": "Evaluating", "evaluating": "Evaluating",
    "create": "Creating", "creating": "Creating",
}
# Bloom buckets (when input gives a coarse category)
BLOOM_CYCLE: dict[str, list[str]] = {
    "remembering": ["Remembering", "Understanding"],
    "applying":    ["Applying",    "Analyzing"],
    "creating":    ["Evaluating",  "Creating"],
}

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR        = pathlib.Path(__file__).resolve().parent.parent
CACHE_DIR       = BASE_DIR / ".extracted_cache"
MODEL_CACHE_DIR = BASE_DIR / ".model_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_bloom(bloom: str, slot_index: int = 0) -> str:
    """Map any bloom string to the canonical capitalised form."""
    stripped = (bloom or "").strip()
    if stripped in BLOOM_CANONICAL:
        return stripped
    key = stripped.lower()
    if key in BLOOM_CYCLE:
        return BLOOM_CYCLE[key][slot_index % 2]
    return BLOOM_ALIASES.get(key, stripped or "Remembering")


def normalize_type(qtype: str) -> str:
    """Map external type name to internal short name (mcq/tf/open)."""
    return TYPE_INTERNAL.get((qtype or "").strip().lower(), "mcq")


def normalize_out_type(raw_type: str) -> str:
    """Map internal/external type to canonical display form."""
    return TYPE_DISPLAY.get((raw_type or "").strip().lower(), raw_type or "mcq")