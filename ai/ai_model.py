# AutoTOS AI module — Ollama backend
#
# Migration from llama-cpp-python → Ollama:
#   - Removed: Llama, LlamaGrammar, model file loading
#   - Added:   requests HTTP calls to Ollama /api/generate
#   - format="json" replaces LlamaGrammar for structured output
#   - All business logic (chunking, fingerprinting, validators) unchanged
#
# ── Patch history ──────────────────────────────────────────────────────────
# v3.5  Fix 1 – AUTOTOS_INSTRUCTION: added "All choices MUST be unique"
#       Fix 2 – _TF_HINT: added "Do NOT use negation words"
#       Fix 3 – MAX_TOKENS_SINGLE[mcq]: 250 → 350
#       Fix 4 – _is_valid_answer: removed false-positive sentence_choices rule
#       Fix 5 – _is_valid_tf: added double-negative guard
#
# v3.6  Opt 1 – Token budgets recalibrated from real log data
#       Opt 2 – num_ctx=768 added to Ollama options
#       Opt 3 – Parallel generation via ThreadPoolExecutor
#       Opt 4 – Thread-safe seen_fps and seen_stems
#       Opt 5 – GENERATION_WORKERS exposed as env var
#
# v3.7  Fix 6 – Answer-based secondary fingerprint
#       Fix 7 – Semantic-duplicate choice detection via pairwise Jaccard
#       Fix 8 – AUTOTOS_INSTRUCTION reinforced
#       Fix 9 – Warm-up replaced with lightweight model-load ping
#
# v3.8  Fix 10 – Bloom label rename (dataset v5+)
#
# v3.9  Opt 6 – BLOOM_HINTS injected into build_training_prompt (reverted v4.0)
#
# v4.0  PROMPT-1 – build_training_prompt() rewritten to match training format.
#       PROMPT-2 – Verbose injection removed (BLOOM_HINTS, base_rules, type_rules).
#       PROMPT-3 – _TF_HINT replaced with single inline line for tf type.
#       PROMPT-4 – avoid_block trimmed from 6 → 3 most-recent questions.
#       PROMPT-5 – num_ctx raised from 768 → 2048.
#       PROMPT-6 – BLOOM_HINTS moved to retry path only (attempt >= 3).
#
# v4.1  Fix 11 – _strip_choice_letter_prefix(): strip embedded "A ", "B. ",
#                "C) " prefixes that the model echoes inside choice text.
#       Fix 12 – _is_valid_blank_completion(): reject MCQ where stem has blank
#                but choices are full sentences incompatible with the stem.
#       Fix 13 – chunk_text() word-boundary guard: never cut mid-token.
#
# v4.2  Opt 7  – CHUNK_SIZE: 800 → 1500 chars, CHUNK_OVERLAP: 30 → 60.
#                Safe within num_ctx=1024: worst-case total ~880 tokens.
#                Larger context gives the model more factual grounding per
#                question, reducing hallucination and improving answer accuracy.
#
#       Fix 14 – TF semantic-duplicate detection: _is_tf_semantic_duplicate()
#                computes Jaccard similarity on content words between the
#                candidate TF statement and all previously accepted TF
#                statements for the same concept. Threshold J >= 0.55 rejects
#                near-identical rewrites (e.g. Q21-Q25 all being
#                "ICTs used for warfare -> false").
#                Root cause: _question_fingerprint() strips so aggressively
#                that "primarily used for warfare" and "mainly used for
#                warfare" produce different fingerprints and both pass.
#                A content-word Jaccard check catches these before they escape.
#
#       Fix 15 – TF answer-balance nudge: _tf_balance_note() tracks true/false
#                answer counts per concept in a shared dict. When a concept
#                already has >= 2 TF answers of the same polarity and zero of
#                the opposite, a one-line note is appended to the prompt
#                ("Vary the answer — try a statement that is TRUE/FALSE.")
#                This directly caused Q21-Q25 and Q32-Q34 repetition.
#
#       Opt 8  – Open-ended bloom+concept tracker: _register_open_question()
#                records every (bloom, concept) pair that has been generated.
#                _open_diversity_note() returns a nudge when the same combo
#                is attempted again, steering the model toward a different
#                angle within the same Bloom's level and topic.
# ============================================================

import re
import json
import fitz
import base64
import requests
from docx import Document
from pptx import Presentation
from io import BytesIO
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Set, Tuple
import os
import hashlib
import logging
import random

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# =====================================================
# CONFIGURATION
# =====================================================
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "autotos")
OLLAMA_TIMEOUT  = int(os.environ.get("OLLAMA_TIMEOUT", "300"))

GENERATION_WORKERS = int(os.environ.get("GENERATION_WORKERS", "2"))

MAX_RETURN    = 50_000

# Opt 7 (v4.2): increased from 800 -> 1500 chars.
# At ~4 chars/token this is ~375 tokens of context.
# With num_ctx=1024: 375 (ctx) + 160 (system+spec) + 304 (output) + 60 (retry)
# = ~899 tokens worst-case — safely within the 1024 window.
CHUNK_SIZE    = 1500
CHUNK_OVERLAP = 60   # ~4% of chunk size (was 30/800 = 3.75%)

BASE_DIR        = os.path.dirname(__file__)
CACHE_DIR       = os.path.join(BASE_DIR, ".extracted_cache")
MODEL_CACHE_DIR = os.path.join(BASE_DIR, ".model_cache")
os.makedirs(CACHE_DIR,       exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

logger.info("Ollama config: base_url=%s model=%s timeout=%ds",
            OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT)

# =====================================================
# TOKEN BUDGETS
# =====================================================
MAX_TOKENS_SINGLE: Dict[str, int] = {
    "mcq":  304,
    "tf":   256,
    "open": 256,
}

# =====================================================
# BLOOM KEYWORDS
# =====================================================
BLOOM_KEYWORDS: Dict[str, List[str]] = {
    "Remembering":   ["Define", "Identify", "Describe", "Recognize", "Explain", "Recite", "Illustrate"],
    "Understanding": ["Summarize", "Interpret", "Classify", "Compare", "Contrast", "Infer", "Paraphrase"],
    "Applying":      ["Solve", "Use", "Complete", "Relate", "Transfer", "Articulate", "Discover"],
    "Analyzing":     ["Contrast", "Connect", "Devise", "Correlate", "Distill", "Conclude", "Categorize"],
    "Evaluating":    ["Criticize", "Judge", "Defend", "Appraise", "Prioritize", "Reframe", "Grade"],
    "Creating":      ["Design", "Develop", "Modify", "Invent", "Write", "Rewrite", "Collaborate"],
}

# PROMPT-6: BLOOM_HINTS retained but only injected on retry attempt >= 3
BLOOM_HINTS: Dict[str, str] = {
    "Remembering":   "Recall a specific fact, term, or definition from the context.",
    "Understanding": "Explain the main idea or the reason something works the way it does.",
    "Applying":      "Show how to use the concept in a concrete scenario or task.",
    "Analyzing":     "Identify a relationship, cause, difference, or underlying pattern.",
    "Evaluating":    "Judge which option is better, or justify a choice with evidence.",
    "Creating":      "Propose, design, or plan something new using the concept.",
}

# =====================================================
# INSTRUCTIONS
# =====================================================
# =====================================================
# INSTRUCTIONS
# =====================================================
AUTOTOS_INSTRUCTION = (
    "You are AutoTOS. Generate one exam question aligned with the Bloom's level and concept. Output ONLY this JSON:\n"
    "{\"type\":\"mcq|truefalse|open_ended\",\"concept\":\"...\",\"bloom\":\"...\",\"question\":\"...\","
    "\"choices\":[\"A text\",\"B text\",\"C text\",\"D text\"],\"answer\":\"A|B|C|D|true|false\","
    "\"answer_text\":\"Why the answer is correct (1-2 sentences).\"}\n"
    "choices field: mcq only. answer_text field: mcq and truefalse only. Answer based ONLY on the provided context.\n"
    "CRITICAL RULE FOR MCQ: The 'answer' field MUST contain ONLY the single uppercase letter (A, B, C, or D). Do NOT write out the full text of the choice.\n"
    "CRITICAL RULE: Write direct questions (e.g., 'What is...'). Do NOT generate fill-in-the-blank questions. NEVER use underscores (_____)."
)

AUTOTOS_INSTRUCTION_OPEN = (
    "You are AutoTOS. Generate one exam question aligned with the Bloom's level and concept. Output ONLY this JSON:\n"
    "{\"type\":\"open_ended\",\"concept\":\"...\",\"bloom\":\"...\",\"question\":\"...\",\"answer\":\"<complete specific answer>\"}\n"
    "Do NOT include answer_text or choices. Answer must be a complete specific response. Answer based ONLY on the provided context."
)

TYPE_MAP = {
    "mcq": "mcq", "truefalse": "tf", "true_false": "tf", "tf": "tf",
    "open_ended": "open", "open-ended": "open", "openended": "open", "open": "open",
}
OUT_TYPE_NORMALIZE = {
    "mcq": "mcq", "truefalse": "truefalse", "tf": "truefalse",
    "true_false": "truefalse", "open_ended": "open_ended",
    "open": "open_ended", "open-ended": "open_ended",
}
BLOOM_MAP_SINGLE = {
    "knowledge": "Remembering", "understand": "Understanding", "apply": "Applying",
    "analyze": "Analyzing", "evaluate": "Evaluating", "create": "Creating",
    "remembering": "Remembering", "understanding": "Understanding",
    "applying": "Applying", "analyzing": "Analyzing", "evaluating": "Evaluating",
    "creating": "Creating",
}
BLOOM_CYCLE = {
    "remembering": ["Remembering", "Understanding"],
    "applying":    ["Applying",    "Analyzing"],
    "creating":    ["Evaluating",  "Creating"],
}

def normalize_bloom(bloom: str, slot_index: int = 0) -> str:
    stripped = (bloom or "").strip()
    _CANONICAL = {"Remembering", "Understanding", "Applying",
                  "Analyzing", "Evaluating", "Creating"}
    if stripped in _CANONICAL:
        return stripped
    key = stripped.lower()
    cycle = BLOOM_CYCLE.get(key)
    if cycle: return cycle[slot_index % 2]
    return BLOOM_MAP_SINGLE.get(key, stripped or "Remembering")

def normalize_type(qtype: str) -> str:
    return TYPE_MAP.get((qtype or "").strip().lower(), "mcq")

def normalize_out_type(raw_type: str) -> str:
    return OUT_TYPE_NORMALIZE.get((raw_type or "").strip().lower(), raw_type or "mcq")

# =====================================================
# OLLAMA CONNECTIVITY CHECK
# =====================================================
_ollama_ready = False

def _check_ollama() -> bool:
    global _ollama_ready
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if r.ok:
            models = [m.get("name", "") for m in r.json().get("models", [])]
            available = any(
                OLLAMA_MODEL == m or OLLAMA_MODEL == m.split(":")[0]
                for m in models
            )
            if available:
                logger.info("Ollama ready. Model '%s' found.", OLLAMA_MODEL)
                _ollama_ready = True
            else:
                logger.warning(
                    "Ollama is running but model '%s' is NOT found. "
                    "Run: docker exec autotoss_ollama ollama create %s -f /models/Modelfile",
                    OLLAMA_MODEL, OLLAMA_MODEL
                )
        return _ollama_ready
    except Exception as e:
        logger.warning("Ollama not reachable: %s", e)
        return False

_check_ollama()

# =====================================================
# DISK CACHE
# =====================================================
def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def _cache_path_for_hash(h: str) -> str:
    return os.path.join(CACHE_DIR, f"{h}.txt")

def _prompt_hash_key(prompt: str, max_tokens: int, temperature: float) -> str:
    key = f"{OLLAMA_MODEL}|{max_tokens}:{temperature:.4f}:{prompt}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def _model_cache_path(key: str) -> str:
    return os.path.join(MODEL_CACHE_DIR, f"{key}.json")

def read_model_cache(key: str) -> Optional[Any]:
    p = _model_cache_path(key)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f: return json.load(f)
        except Exception:
            try: os.remove(p)
            except Exception: pass
    return None

def write_model_cache(key: str, obj: Any):
    try:
        with open(_model_cache_path(key), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
    except Exception:
        pass

CACHE_HITS   = 0
CACHE_MISSES = 0

# =====================================================
# GENERATION PROGRESS TRACKER
# =====================================================
_gen_progress: Dict[str, Any] = {"current": 0, "total": 0, "active": False}
_gen_progress_lock = threading.Lock()

# =====================================================
# UTILITIES
# =====================================================
# =====================================================
# UTILITIES
# =====================================================
def clean_text(txt) -> str:
    if txt is None: return ""
    if not isinstance(txt, str):
        try: txt = str(txt)
        except Exception: return ""
    return re.sub(r"\s+", " ", txt).strip()

def sanitize_prompt(prompt: str) -> str:
    return (prompt or "").strip()

_AWKWARD_PHRASING_RULES = [
    (re.compile(r'^Which of the following best describes which\b', re.IGNORECASE), "Which"),
    (re.compile(r'^Which of the following best describes what\b',  re.IGNORECASE), "What"),
    (re.compile(r'^Which best describes what\b',                   re.IGNORECASE), "What"),
    (re.compile(r'^Which best describes which\b',                  re.IGNORECASE), "Which"),
]

# --- NEW: Fill-in-the-blank Guard ---
_FILL_IN_BLANK_RE = re.compile(r'_{3,}')

def is_fill_in_the_blank(question: str) -> bool:
    """Returns True if the question contains 3 or more underscores."""
    if not question: return False
    return bool(_FILL_IN_BLANK_RE.search(question))

# --- NEW: MCQ Hallucination Fix ---
def normalize_mcq_answer(answer_value: str, choices: list) -> str:
    """Converts hallucinated full-text answers back to A, B, C, or D."""
    answer_value = answer_value.strip()
    if len(answer_value) == 1 and answer_value.upper() in ["A", "B", "C", "D"]:
        return answer_value.upper()
    
    answer_lower = answer_value.lower()
    for i, choice in enumerate(choices):
        if choice and answer_lower in choice.lower():
            return chr(65 + i)
            
    return answer_value

def _clean_question_phrasing(text: str) -> str:
    if not text: return text
    for pattern, replacement in _AWKWARD_PHRASING_RULES:
        if pattern.match(text):
            text = pattern.sub(replacement + " ", text, count=1).strip()
            if text: text = text[0].upper() + text[1:]
            break
    return text

_TF_QUESTION_PREFIX_RE = re.compile(r"^(true\s+or\s+false\s*[:\-]\s*)", re.IGNORECASE)
_ARTIFACT_DIGIT_RE = re.compile(r"\b\d+\s+(?=[A-Z])")

def strip_question_prefix(text: str, is_open_ended: bool = False) -> str:
    text = (text or "").strip()
    if not text: return text
    text = _ARTIFACT_DIGIT_RE.sub("", text).strip()
    if is_open_ended: return _clean_question_phrasing(text)
    text = _TF_QUESTION_PREFIX_RE.sub("", text).strip()
    text = re.sub(r"^it is true that\s+", "", text, flags=re.IGNORECASE).strip()
    if text: text = text[0].upper() + text[1:]
    return _clean_question_phrasing(text)

_ANSWER_TEXT_MAX_CHARS = 250

def _truncate_answer_text(text: str) -> str:
    if not text: return ""
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r'^(answer\s*[:\-]\s*)', '', t, flags=re.IGNORECASE).strip()
    sentences = re.split(r'(?<=[.!?])\s+', t)
    first = sentences[0].strip() if sentences else t.strip()
    if first and first[-1] not in ".!?": first = first + "."
    if len(first) > _ANSWER_TEXT_MAX_CHARS:
        truncated = first[:_ANSWER_TEXT_MAX_CHARS]
        last_space = truncated.rfind(" ")
        if last_space > 0: truncated = truncated[:last_space]
        return truncated.rstrip('.,;:') + "..."
    return first

# =====================================================
# CHUNKING
# =====================================================
_chunk_cache: Dict[str, List[str]] = {}

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks, snapping to sentence boundaries
    when possible, or word boundaries otherwise (Fix 13, v4.1).

    Opt 7 (v4.2): CHUNK_SIZE raised from 800 -> 1500 chars so each chunk
    delivers ~375 tokens of context — nearly double the previous budget —
    while staying safely within num_ctx=1024.
    """
    if not text: return []
    text = text.strip(); chunks = []; start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start, end),
                           text.rfind("! ", start, end),
                           text.rfind("? ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
            else:
                # Fix 13: never cut mid-token
                word_boundary = text.rfind(" ", start, end)
                if word_boundary > start:
                    end = word_boundary
        chunk = text[start:end].strip()
        if chunk: chunks.append(chunk)
        start += step
    return chunks

def _find_best_chunk_idx(chunks: List[str], topic: str) -> int:
    if not chunks or not topic: return 0
    topic_lower = topic.lower()
    for i, chunk in enumerate(chunks):
        if topic_lower in chunk.lower(): return i
    for word in topic_lower.split():
        if len(word) > 3:
            for i, chunk in enumerate(chunks):
                if word in chunk.lower(): return i
    return 0

def get_chunks_for_text(full_text: str) -> List[str]:
    key = hashlib.md5(full_text[:256].encode("utf-8", errors="ignore")).hexdigest()
    if key not in _chunk_cache:
        _chunk_cache[key] = chunk_text(full_text)
        logger.info("Chunked document: %d chars -> %d chunks", len(full_text), len(_chunk_cache[key]))
    return _chunk_cache[key]

# =====================================================
# FILE EXTRACTION
# =====================================================
def extract_text_from_bytes(file_bytes: bytes, filetype: str) -> str:
    text = ""
    try:
        if filetype == "pdf":
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            parts = []
            for page in doc:
                try:
                    ptext = page.get_text("text")
                    if ptext and ptext.strip(): parts.append(ptext)
                except Exception: continue
            text = " ".join(parts)
        elif filetype == "docx":
            d = Document(BytesIO(file_bytes))
            text = " ".join(p.text for p in d.paragraphs if p.text and p.text.strip())
        elif filetype == "pptx":
            prs = Presentation(BytesIO(file_bytes))
            slt = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text and shape.text.strip():
                        slt.append(shape.text)
            text = " ".join(slt)
        else:
            text = file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.warning("Extraction error: %s", e)
    return clean_text(text)

def extract_text_from_path(path: str, max_chars: int = MAX_RETURN) -> str:
    if not path or not os.path.exists(path): return ""
    try:
        with open(path, "rb") as f: b = f.read()
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e); return ""
    h = _sha256_bytes(b)
    cache_file = _cache_path_for_hash(h)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as rf:
                return clean_text(rf.read())[:max_chars]
        except Exception: pass
    ext = (os.path.splitext(path)[1] or "").lower()
    filetype = {".pdf": "pdf", ".docx": "docx", ".doc": "docx",
                ".pptx": "pptx", ".ppt": "pptx"}.get(ext, "")
    extracted = extract_text_from_bytes(b, filetype)
    try:
        with open(cache_file, "w", encoding="utf-8") as wf: wf.write(extracted)
    except Exception: pass
    return clean_text(extracted)[:max_chars]

def lesson_from_upload(data_or_text: Optional[str]) -> str:
    if not data_or_text: return ""
    if isinstance(data_or_text, str) and os.path.exists(data_or_text):
        try: return extract_text_from_path(data_or_text, max_chars=MAX_RETURN)
        except Exception as e:
            logger.warning("Error extracting %s: %s", data_or_text, e); return ""
    if isinstance(data_or_text, str) and data_or_text.startswith("data:"):
        try:
            header, encoded = data_or_text.split(",", 1)
            file_bytes = base64.b64decode(encoded)
            ft = ("pdf"  if "pdf"  in header else
                  "docx" if ("docx" in header or "word" in header) else
                  "pptx" if ("pptx" in header or "presentation" in header) else "")
            return extract_text_from_bytes(file_bytes, ft)[:MAX_RETURN]
        except Exception as e:
            logger.warning("Base64 decode fail: %s", e); return ""
    try: return clean_text(data_or_text or "")[:MAX_RETURN]
    except Exception: return ""

# =====================================================
# PROMPT BUILDER  (v4.0 — minimal, training-aligned)
# =====================================================
def _build_avoid_block(seen_questions: List[str]) -> str:
    if not seen_questions: return ""
    recent = seen_questions[-3:]
    items  = "; ".join(f'"{q[:40]}"' for q in recent)
    return f"\n[Different topic angle — avoid repeating: {items}]\n"


def build_training_prompt(
    instruction: str,
    qtype: str,
    bloom: str,
    concept: str,
    context: str,
    extra_note: str = "",
    attempt_note: str = "",
    avoid_questions: Optional[List[str]] = None,
    attempt: int = 1,
) -> str:
    avoid_block = _build_avoid_block(avoid_questions or [])

    tf_note = ""
    if qtype == "tf":
        tf_note = "\n- Type note: Declarative statement only. No question mark. No negation words.\n"

    bloom_note = ""
    if attempt >= 3:
        hint = BLOOM_HINTS.get(bloom, "")
        if hint:
            bloom_note = f"\n- Bloom hint: {hint}\n"

    retry_line = f"\n{attempt_note.strip()}\n" if attempt_note and attempt_note.strip() else ""
    ctx_suffix = f"{tf_note}{bloom_note}{retry_line}{avoid_block}"

    user_msg = (
        "### Target Specification:\n"
        f"- Question Type: {qtype}\n"
        f"- Bloom's Level: {bloom}\n"
        f"- Concept: {concept}\n\n"
        "### Context (Source Material):\n"
        f"{context}{ctx_suffix}"
    )

    return (
        f"<|im_start|>system\n{instruction}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# =====================================================
# JSON EXTRACTORS
# =====================================================
def _extract_first_json(text: str) -> Optional[str]:
    if not text: return None
    start = text.find('{')
    if start == -1: return None
    depth = 0; in_string = False; escape = False
    for i, c in enumerate(text[start:], start):
        if escape: escape = False; continue
        if c == '\\' and in_string: escape = True; continue
        if c == '"': in_string = not in_string; continue
        if in_string: continue
        if c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0: return text[start:i + 1]
    return text[start:]

def _try_parse_json(json_str: str) -> Optional[Any]:
    if not json_str or not isinstance(json_str, str): return None
    try: return json.loads(json_str)
    except Exception: pass
    cleaned = re.sub(r',\s*([}\]])', r'\1', json_str)
    if cleaned.count('"') % 2 == 1: cleaned = cleaned + '"'
    open_braces = cleaned.count('{') - cleaned.count('}')
    if 0 < open_braces <= 6:
        try: return json.loads(cleaned + ('}' * open_braces))
        except Exception: pass
    try: return json.loads(cleaned)
    except Exception: return None

# =====================================================
# MODEL CALLER
# =====================================================
def ask_model(prompt: str, max_tokens: int = 200,
              temperature: float = 0.45) -> Optional[dict]:
    global CACHE_HITS, CACHE_MISSES
    if not _ollama_ready and not _check_ollama():
        logger.error("Ollama not ready.")
        return None

    prompt    = sanitize_prompt(prompt)
    cache_key = _prompt_hash_key(prompt, max_tokens, temperature)
    cached    = read_model_cache(cache_key)
    if cached is not None:
        CACHE_HITS += 1
        return cached if isinstance(cached, dict) else None
    CACHE_MISSES += 1

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx":     1024,   # safe for CHUNK_SIZE=1500: worst-case ~899 tokens
            "top_p":       0.95,
            "stop":        ["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
        },
    }

    try:
        start = time.time()
        resp  = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        duration = time.time() - start

        if not resp.ok:
            logger.warning("Ollama returned %d: %s", resp.status_code, resp.text[:200])
            return None

        raw = resp.json().get("response", "")
        logger.info("ask_model duration=%.2fs raw_len=%d max_tok=%d temp=%.2f",
                    duration, len(raw), max_tokens, temperature)

        if not raw or not raw.strip():
            logger.warning("Empty Ollama response")
            return None

        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
        if not raw: return None

        json_str = _extract_first_json(raw)
        parsed   = _try_parse_json(json_str) if json_str else None

        if parsed is None:
            logger.warning("No JSON found in: %s", raw[:200])
            return None

        write_model_cache(cache_key, parsed)
        return parsed

    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out after %ds", OLLAMA_TIMEOUT)
        return None
    except Exception as e:
        logger.exception("ask_model error: %s", e)
        return None

# =====================================================
# MODEL WARM-UP
# =====================================================
def _warmup_model() -> None:
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": "Say OK",
                "stream": False,
                "options": {"num_predict": 3, "temperature": 0.0},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        if resp.ok:
            logger.info("Warm-up done — model loaded into memory.")
        else:
            logger.warning("Warm-up ping returned HTTP %d", resp.status_code)
    except Exception as e:
        logger.warning("Warm-up ping failed (non-fatal): %s", e)

try:
    if _ollama_ready:
        logger.info("Warming up Ollama (model load ping)...")
        _warmup_model()
except Exception:
    pass

# =====================================================
# NORMALIZATION / KEY MAPPING
# =====================================================
def _normalize_output_keys(out: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(out, dict): return {}
    mapping = {
        "statement": "question", "prompt": "question",
        "sample_answer": "answer_text", "sample_response": "answer_text",
        "model_answer": "answer", "explanation": "answer_text",
        "solution": "answer_text", "rationale": "answer_text",
        "ans": "answer", "correct": "answer",
    }
    for src, dst in mapping.items():
        if src in out and dst not in out: out[dst] = out[src]
    if "answer" not in out and "sample_answer" in out: out["answer"] = out["sample_answer"]
    if isinstance(out.get("answer"), bool): out["answer"] = "true" if out["answer"] else "false"
    return out

# =====================================================
# Fix 11 (v4.1) — CHOICE PREFIX STRIPPER
# =====================================================
_CHOICE_LETTER_PREFIX_RE = re.compile(r'^[A-Da-d][).:\s]+\s*')

def _strip_choice_letter_prefix(text: str) -> str:
    """Remove leading 'A ', 'B. ', 'C) ' style prefixes from a choice string."""
    return _CHOICE_LETTER_PREFIX_RE.sub("", text).strip()


def normalize_generated_question(q: dict, expected_display_type: str,
                                  topic: str, bloom_level: str) -> dict:
    q = q or {}
    if not isinstance(q, dict): q = {"question": str(q)}
    q = _normalize_output_keys(q)
    raw_out_type = q.get("type") or expected_display_type
    display_type = normalize_out_type(raw_out_type) or expected_display_type
    out = {
        "type":    display_type,
        "concept": topic,
        "bloom":   bloom_level,
        "question": strip_question_prefix(
            clean_text(q.get("question") or q.get("prompt") or ""),
            is_open_ended=(display_type == "open_ended")
        ),
        "choices":  [],
        "answer":   "",
    }
    if display_type != "open_ended":
        raw_answer_text = clean_text(q.get("answer_text") or q.get("explanation") or "")
        out["answer_text"] = _truncate_answer_text(raw_answer_text)
    choices_raw = q.get("choices") or q.get("options")
    if isinstance(choices_raw, dict):
        keys = sorted(choices_raw.keys(), key=lambda s: s.upper())
        out["choices"] = [
            _strip_choice_letter_prefix(clean_text(choices_raw[k])) for k in keys
        ][:4]
    elif isinstance(choices_raw, list):
        out["choices"] = [
            _strip_choice_letter_prefix(clean_text(x)) for x in choices_raw
        ][:4]
    ans = q.get("answer", "")
    if isinstance(ans, bool): out["answer"] = "true" if ans else "false"
    elif isinstance(ans, (int, float)):
        idx = int(ans)
        out["answer"] = (out["choices"][idx] if out["choices"] and 0 <= idx < len(out["choices"]) else str(ans))
    elif isinstance(ans, str):
        a = ans.strip()
        a_stripped = _CHOICE_LETTER_PREFIX_RE.sub("", a).strip()
        if display_type == "mcq":
            if re.fullmatch(r"[A-Da-d]", a.strip()) and out["choices"]:
                idx = ord(a.strip().upper()) - ord("A")
                out["answer"] = (out["choices"][idx] if 0 <= idx < len(out["choices"]) else a.strip())
            elif re.match(r"^[A-Da-d][).:\s]", a):
                matched = next(
                    (c for c in out["choices"] if c.lower() == a_stripped.lower()), None
                )
                out["answer"] = matched or a_stripped
            else:
                matched = next(
                    (c for c in out["choices"] if c.lower() == a.lower()), None
                )
                out["answer"] = matched or a
        elif display_type == "truefalse":
            a_lower = a.lower().rstrip(".")
            if a_lower in ("true", "false"):  out["answer"] = a_lower
            elif a_lower in ("1", "yes"):     out["answer"] = "true"
            elif a_lower in ("0", "no"):      out["answer"] = "false"
            else:                             out["answer"] = a
        else: out["answer"] = a
    else: out["answer"] = clean_text(str(ans or ""))
    return out

# =====================================================
# FINGERPRINTING
# =====================================================
_FP_STOPWORDS = re.compile(
    r'\b(a|an|the|and|or|of|in|on|to|for|is|are|was|were|by|at|be|its|it|'
    r'that|this|these|those|with|as|from|into|about|which|how|what|does|do)\b')
_FP_FILLER_RE = re.compile(
    r"^(what (is|are|does|do|was|were) (the )?(primary |main |key )?"
    r"(focus|purpose|function|role|goal|aim|definition|meaning|concept|"
    r"example|reason|impact|effect|difference|advantage|use|importance) of\s*|"
    r"which (of the following )?(best )?(describes?|defines?|explains?|is|are)\s*|"
    r"how (does?|do|is|are|can)\s*|"
    r"why (is|are|does|do)\s*)", re.IGNORECASE)
_FP_VERB_OPENER_RE = re.compile(
    r"^(define|explain|describe|summarize|identify|analyze|evaluate|compare|"
    r"contrast|apply|solve|create|design|develop|discuss|state|examine|"
    r"assess|illustrate|demonstrate|interpret|classify|infer|relate|conclude|"
    r"criticize|judge|defend|appraise|reframe|modify|invent|collaborate)\s+",
    re.IGNORECASE)
_FP_QUALIFIER_RE = re.compile(
    r"\b(primary|main|key|overall|general|core|basic|fundamental|"
    r"purpose|goal|aim|role|function|focus|use|importance|objective)\b",
    re.IGNORECASE)
_FP_EXTRA_FILLER = re.compile(
    r"\b(according|lesson|course|module|section|unit|chapter|text|context|"
    r"provided|reading|material|notes|slide|above|below|given|based)\b",
    re.IGNORECASE
)
_FP_TRAILING_VERB_RE = re.compile(
    r"\s+(ensure|refer|mean|indicate|show|suggest|imply|denote|involve|"
    r"describe|define|represent|state|explain|allow|enable|prevent|protect|"
    r"provide|require|help|support|include|contain|affect|impact|cause|create|"
    r"result|lead|contribute|determine|measure|assess|reflect)\s*$",
    re.IGNORECASE
)

def _question_fingerprint(q: dict) -> str:
    raw = (q.get("question") or "").lower().strip()
    raw = re.sub(r"[^\w\s]", " ", raw)
    raw = _FP_STOPWORDS.sub(" ", raw)
    raw = _FP_FILLER_RE.sub("", raw).strip()
    raw = _FP_VERB_OPENER_RE.sub("", raw).strip()
    raw = _FP_QUALIFIER_RE.sub(" ", raw)
    raw = _FP_EXTRA_FILLER.sub(" ", raw)
    raw = _FP_TRAILING_VERB_RE.sub("", raw).strip()
    raw = re.sub(r"\s+", " ", raw).strip()
    words = raw.split()
    words = [w[:-1] if w.endswith("s") and len(w) > 4 else w for w in words]
    raw   = " ".join(words)
    qtext   = raw[:35]
    concept = re.sub(r"\s+", "_", (q.get("concept") or "").lower().strip())
    return f"{concept}::{qtext}"

_ANS_FP_SW = re.compile(
    r'\b(a|an|the|and|or|of|in|on|to|for|is|are|was|were|by|at|be|its|it|'
    r'that|this|with|as|from|they|their|can|will|may|has|have|used|'
    r'also|both|each|such|than|then|when|where|while)\b',
    re.IGNORECASE
)

def _answer_fingerprint(q: dict) -> Optional[str]:
    if (q.get("type") or "").lower() != "mcq":
        return None
    ans = clean_text(str(q.get("answer") or "")).lower()
    if not ans or len(ans) < 8:
        return None
    ans_norm = _ANS_FP_SW.sub(" ", ans)
    ans_norm = re.sub(r"\s+", " ", ans_norm).strip()
    if not ans_norm:
        return None
    concept = re.sub(r"\s+", "_", (q.get("concept") or "").lower().strip())
    return f"{concept}::ans::{ans_norm[:50]}"

def _question_stem(q: dict) -> str:
    return (q.get("question") or "")[:80].strip()

def _answer_matches_explanation(answer_letter, choices, answer_text):
    if not choices or not answer_letter or answer_letter not in "ABCD": return True, answer_letter
    if not answer_text: return True, answer_letter
    idx = ord(answer_letter.upper()) - ord("A")
    if idx < 0 or idx >= len(choices): return True, answer_letter
    ans_low = answer_text.lower()
    scores = [(sum(1 for w in re.findall(r"\b\w{4,}\b", c.lower()) if w in ans_low), i)
              for i, c in enumerate(choices)]
    scores.sort(reverse=True)
    best_score, best_idx = scores[0]
    chosen_score = next(s for s, i in scores if i == idx)
    best_letter = chr(ord("A") + best_idx)
    if best_letter != answer_letter and (best_score - chosen_score) >= 2:
        return False, best_letter
    return True, answer_letter

_SEM_DUP_SW = re.compile(
    r'\b(a|an|the|and|or|of|in|on|to|for|is|are|was|were|by|at|be|its|it|'
    r'that|this|these|those|with|as|from|into|about|which|how|what|does|do|'
    r'not|can|will|may|has|have|had|been|being|they|their|such|each|used|'
    r'using|allows|allow|makes|make|uses|use|helps|help|enables|enable|'
    r'provides|provide|requires|require|ensures|ensure|offer|offers)\b',
    re.IGNORECASE
)

def _has_semantic_duplicate_choices(choices: list) -> bool:
    if not choices or len(choices) < 2:
        return False

    def _content(text: str) -> set:
        t = _SEM_DUP_SW.sub(" ", text.lower())
        return {w for w in re.findall(r'\b\w+\b', t) if len(w) > 4}

    word_sets = [_content(c) for c in choices]
    non_empty = [ws for ws in word_sets if ws]
    if len(non_empty) >= 2:
        common = non_empty[0].copy()
        for ws in non_empty[1:]:
            common &= ws
        if len(common) >= 3:
            logger.info("MCQ rejected: all-choices share %d content words: %r",
                        len(common), sorted(common))
            return True

    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            a, b = word_sets[i], word_sets[j]
            if len(a) >= 4 and len(b) >= 4:
                union = a | b
                if union:
                    jaccard = len(a & b) / len(union)
                    if jaccard >= 0.65:
                        logger.info(
                            "MCQ rejected: pairwise near-duplicate choices "
                            "(J=%.2f): %r vs %r",
                            jaccard, choices[i][:60], choices[j][:60],
                        )
                        return True
    return False

# =====================================================
# Fix 12 (v4.1) — BLANK-COMPLETION VALIDATOR
# =====================================================
_BLANK_STEM_RE = re.compile(r"_{4,}|\.{3,}\s*$")
_FULL_SENTENCE_OPENER_RE = re.compile(
    r"^(it (is|was|can|will|has|does|did|should|would|could|must|might)\b|"
    r"by (the|a|an|its|this|that|making|allowing|enabling|providing|using|doing|giving|increasing|reducing|removing|replacing|combining)\b|"
    r"by [a-z]+ing\b|"
    r"they (are|were|can|will|have|do)\b|"
    r"this (is|was|makes|allows|enables|provides|ensures|refers)\b|"
    r"these (are|were)\b|"
    r"there (is|are|was|were)\b|"
    r"to (store|perform|record|connect|display|process|enable|allow|provide|ensure|help|use|make|give|increase|reduce|replace|control|combine|measure|identify|define|describe|analyze|evaluate|create|develop|design|modify)\b|"
    r"(a|an) [a-z]+ (that|which|for|to|of|in)\b"
    r")",
    re.IGNORECASE
)

def _is_valid_blank_completion(q: dict) -> bool:
    question = (q.get("question") or "").strip()
    if not _BLANK_STEM_RE.search(question):
        return True
    choices = q.get("choices") or []
    if not choices:
        return True
    bad = sum(1 for c in choices if _FULL_SENTENCE_OPENER_RE.match(c))
    if bad >= 2:
        logger.info(
            "MCQ rejected: fill-in-blank stem with %d full-sentence choices "
            "(question=%r)", bad, question[:80]
        )
        return False
    return True

# =====================================================
# Fix 14 (v4.2) — TF SEMANTIC DUPLICATE DETECTION
# =====================================================
# Stored as: concept_key -> [(frozenset_of_content_words, answer_str), ...]
# Shared across threads via tf_concept_lock from generate_from_records().
# =====================================================
_TF_SEM_SW = re.compile(
    r'\b(a|an|the|and|or|of|in|on|to|for|is|are|was|were|by|at|be|its|it|'
    r'that|this|these|those|with|as|from|into|about|which|how|what|does|do|'
    r'not|can|will|may|has|have|had|been|being|they|their|such|each|used|'
    r'also|both|than|then|when|where|while|primarily|mainly|often|'
    r'generally|usually|typically|mostly|largely|rather|instead|'
    r'whether|either|neither|nor|just|only|always|never|still|even|'
    r'more|most|less|least|very|quite|somewhat)\b',
    re.IGNORECASE
)

def _tf_content_words(question: str) -> frozenset:
    """Return meaningful content words from a TF statement."""
    q = re.sub(r"[^\w\s]", " ", question.lower())
    q = _TF_SEM_SW.sub(" ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return frozenset(w for w in q.split() if len(w) > 3)

def _is_tf_semantic_duplicate(
    candidate_q: dict,
    seen_tf_by_concept: Dict[str, List[Tuple[frozenset, str]]],
    tf_lock: threading.Lock,
) -> bool:
    """
    Fix 14 (v4.2): Jaccard similarity check for TF questions.
    Rejects when J >= 0.55 against any previously accepted TF statement
    for the same concept.  Threshold is lower than MCQ choice dedup (0.65)
    because TF statements are shorter — fewer words yields higher Jaccard
    for the same semantic overlap.
    """
    concept  = (candidate_q.get("concept") or "").lower().strip()
    question = candidate_q.get("question") or ""
    cwords   = _tf_content_words(question)
    if not cwords:
        return False
    with tf_lock:
        existing = seen_tf_by_concept.get(concept, [])
        for ex_words, _ in existing:
            if not ex_words:
                continue
            union   = cwords | ex_words
            jaccard = len(cwords & ex_words) / len(union) if union else 0.0
            if jaccard >= 0.55:
                logger.info(
                    "TF rejected: semantic duplicate (J=%.2f) concept=%r q=%r",
                    jaccard, concept, question[:70]
                )
                return True
    return False

def _register_tf_question(
    candidate_q: dict,
    seen_tf_by_concept: Dict[str, List[Tuple[frozenset, str]]],
    tf_lock: threading.Lock,
) -> None:
    """Record a successfully accepted TF question in the per-concept tracker."""
    concept  = (candidate_q.get("concept") or "").lower().strip()
    question = candidate_q.get("question") or ""
    answer   = (candidate_q.get("answer") or "").lower().strip()
    cwords   = _tf_content_words(question)
    with tf_lock:
        if concept not in seen_tf_by_concept:
            seen_tf_by_concept[concept] = []
        seen_tf_by_concept[concept].append((cwords, answer))

# =====================================================
# Fix 15 (v4.2) — TF ANSWER-BALANCE NUDGE
# =====================================================
def _tf_balance_note(
    concept: str,
    seen_tf_by_concept: Dict[str, List[Tuple[frozenset, str]]],
    tf_lock: threading.Lock,
) -> str:
    """
    Fix 15 (v4.2): return a balance hint when a concept's TF answers are
    all the same polarity (>= 2 of one side, 0 of the other).
    Empty string when no nudge is needed.
    """
    with tf_lock:
        existing = seen_tf_by_concept.get(concept.lower().strip(), [])
    if len(existing) < 2:
        return ""
    true_count  = sum(1 for _, a in existing if a == "true")
    false_count = sum(1 for _, a in existing if a == "false")
    if false_count >= 2 and true_count == 0:
        return "Vary the answer — write a TRUE statement about this concept."
    if true_count >= 2 and false_count == 0:
        return "Vary the answer — write a FALSE statement about this concept."
    return ""

# =====================================================
# Opt 8 (v4.2) — OPEN-ENDED BLOOM+CONCEPT TRACKER
# =====================================================
def _open_bloom_concept_key(topic: str, bloom: str) -> str:
    return f"{topic.lower().strip()}::{bloom.lower().strip()}"

def _open_diversity_note(
    topic: str,
    bloom: str,
    seen_open_combos: Set[str],
    open_lock: threading.Lock,
) -> str:
    """
    Opt 8 (v4.2): return a diversity nudge when the (bloom, concept) pair
    has already produced an open-ended question.  Empty string otherwise.
    """
    key = _open_bloom_concept_key(topic, bloom)
    with open_lock:
        if key in seen_open_combos:
            return "Different angle from before — focus on a distinct aspect or example."
    return ""

def _register_open_question(
    topic: str,
    bloom: str,
    seen_open_combos: Set[str],
    open_lock: threading.Lock,
) -> None:
    key = _open_bloom_concept_key(topic, bloom)
    with open_lock:
        seen_open_combos.add(key)

# =====================================================
# VALIDATORS
# =====================================================
_ANSWER_ALWAYS_BAD = {"—", "-", "", "answer:"}
_OPEN_PLACEHOLDERS = {
    "model answer here", "model answer here.", "answer here",
    "write answer here", "<complete model answer based on context>",
    "<write a complete model answer>",
}

def _is_valid_answer(q: dict, display_type: str) -> bool:
    answer  = clean_text(str(q.get("answer") or "")).strip()
    ans_low = answer.lower().rstrip(".")
    if ans_low in _ANSWER_ALWAYS_BAD: return False
    if display_type == "mcq":
        choices = q.get("choices") or []
        if not answer: return False
        if re.fullmatch(r"[a-d]", ans_low) and not choices: return False
        if choices and len(choices) >= 2:
            stripped = [c.lower().strip() for c in choices if c.strip()]
            if len(stripped) != len(set(stripped)):
                logger.info("MCQ rejected: duplicate choices detected %r", stripped)
                return False
            if _has_semantic_duplicate_choices(choices):
                return False
            if not _is_valid_blank_completion(q):
                return False
            answer_text_expl = clean_text(str(q.get("answer_text") or ""))
            if answer_text_expl and len(choices) >= 3:
                ans_idx = next(
                    (i for i, c in enumerate(choices)
                     if c.lower().strip() == answer.lower().strip()),
                    None
                )
                if ans_idx is not None:
                    ans_letter = chr(ord("A") + ans_idx)
                    ok, _ = _answer_matches_explanation(ans_letter, choices, answer_text_expl)
                    if not ok:
                        logger.info("MCQ rejected: answer doesn't match its own explanation "
                                    "(record answer=%r)", answer[:60])
                        return False
        return True
    elif display_type == "truefalse":
        return ans_low in ("true", "false")
    elif display_type == "open_ended":
        if len(answer) < 15: return False
        if ans_low in _OPEN_PLACEHOLDERS: return False
        if re.match(r"^(model answer|answer\s*:)", ans_low): return False
    return True

_TF_NEGATED_RE = re.compile(
    r"^(it is (false|not true|incorrect|inaccurate|wrong) that\b|"
    r"it is (incorrect|inaccurate|wrong) to (state|say|claim|assert) that\b|"
    r"it is not the case that\b)",
    re.IGNORECASE
)
_TF_IT_IS_TRUE_RE  = re.compile(r"^it is true that\b", re.IGNORECASE)
_TF_META_STATEMENT_RE = re.compile(
    r"^the statement .{5,} is (true|false|correct|incorrect)\b",
    re.IGNORECASE
)
_TF_TASK_VERBS = re.compile(
    r'^(convert|calculate|compute|list|draw|design|write|find|determine|'
    r'show|give an example|describe how|explain how|create a?|propose|'
    r'evaluate|analyze|define|summarize|solve|identify|compare|'
    r'develop|construct|formulate|generate|'
    r'contrast|correlate|distill|conclude|categorize|'
    r'criticize|judge|defend|appraise|prioritize|reframe|grade|'
    r'modify|invent|rewrite|collaborate|'
    r'interpret|classify|infer|paraphrase|relate|transfer|articulate|discover|'
    r'connect|devise|describe|recognize|recite|illustrate|complete)\b', re.IGNORECASE)
_TF_WH_QUESTION = re.compile(r'^(what|which|how|why|who|where|when)\b', re.IGNORECASE)
_TF_NEGATION_IN_STMT = re.compile(
    r"\b(not|never|no|doesn't|don't|isn't|aren't|cannot|can't|won't|"
    r"wouldn't|shouldn't|couldn't|neither|nor)\b",
    re.IGNORECASE
)

def _is_valid_tf(q: dict) -> bool:
    answer = (q.get("answer") or "").strip().lower().rstrip(".")
    if answer not in ("true", "false"): return False
    question = (q.get("question") or "").strip()
    if not question: return False
    if _TF_TASK_VERBS.match(question) or _TF_WH_QUESTION.match(question): return False
    if _TF_NEGATED_RE.match(question): return False
    if _TF_IT_IS_TRUE_RE.match(question) and answer == "false": return False
    if re.search(r'(explanation|description|timeline|summary)\s*[.:]?\s*$', question, re.IGNORECASE): return False
    if question.rstrip().endswith(":"): return False
    if len(question.strip()) < 25: return False
    if _TF_META_STATEMENT_RE.match(question): return False
    if answer == "false" and _TF_NEGATION_IN_STMT.search(question):
        logger.info("TF rejected: double-negative (negation in stmt + answer=false): %r", question[:80])
        return False
    return True

# =====================================================
# SINGLE QUESTION GENERATOR
# =====================================================
_RETRY_NOTES = [
    "",
    "Try a different angle — focus on a specific component or detail.",
    "Ask about a consequence or real-world application.",
    "Use a concrete scenario or example.",
]

def _generate_single(
    topic, prompt_type, display_type, bloom, context, max_tok,
    seen_fps, record_idx,
    seen_questions=None, fp_lock=None,
    seen_answer_fps=None, answer_fp_lock=None,
    # Fix 14/15 (v4.2): TF diversity trackers
    seen_tf_by_concept=None, tf_concept_lock=None,
    # Opt 8 (v4.2): open-ended bloom+concept tracker
    seen_open_combos=None, open_combos_lock=None,
):
    instruction  = AUTOTOS_INSTRUCTION_OPEN if prompt_type == "open" else AUTOTOS_INSTRUCTION
    MAX_ATTEMPTS = 5
    avoid_list   = list(seen_questions or [])

    for attempt in range(1, MAX_ATTEMPTS + 1):
        temp = min(0.85, 0.45 + 0.15 * (attempt - 1))

        # Base retry note from the static list
        attempt_note = _RETRY_NOTES[min(attempt - 1, len(_RETRY_NOTES) - 1)]

        # Fix 15 (v4.2): append TF balance nudge, computed fresh each attempt
        # so it reflects any parallel workers that accepted new TF questions
        if prompt_type == "tf" and seen_tf_by_concept is not None and tf_concept_lock is not None:
            balance = _tf_balance_note(topic, seen_tf_by_concept, tf_concept_lock)
            if balance:
                attempt_note = f"{attempt_note} {balance}".strip() if attempt_note else balance

        # Opt 8 (v4.2): append open-ended diversity nudge when bloom+concept already seen
        if prompt_type == "open" and seen_open_combos is not None and open_combos_lock is not None:
            diversity = _open_diversity_note(topic, bloom, seen_open_combos, open_combos_lock)
            if diversity:
                attempt_note = f"{attempt_note} {diversity}".strip() if attempt_note else diversity

        prompt = build_training_prompt(
            instruction, prompt_type, bloom, topic, context,
            attempt_note=attempt_note,
            avoid_questions=avoid_list,
            attempt=attempt,
        )

        generated = ask_model(prompt, max_tokens=max_tok, temperature=temp)
        if generated is None:
            logger.info("Single gen failed record=%d attempt=%d", record_idx, attempt)
            time.sleep(0.1 * attempt)
            continue

        generated   = _normalize_output_keys(generated)
        candidate_q = normalize_generated_question(generated, display_type, topic, bloom)
        qtext       = (candidate_q.get("question") or "").strip()
        if not qtext: continue

        # TF-specific validation + semantic dedup
        if display_type == "truefalse":
            if not _is_valid_tf(candidate_q):
                time.sleep(0.1 * attempt); continue
            # Fix 14 (v4.2): reject near-identical TF rewrites
            if seen_tf_by_concept is not None and tf_concept_lock is not None:
                if _is_tf_semantic_duplicate(candidate_q, seen_tf_by_concept, tf_concept_lock):
                    avoid_list.append(_question_stem(candidate_q))
                    time.sleep(0.1 * attempt); continue

        if not _is_valid_answer(candidate_q, display_type):
            logger.info("Invalid answer record=%d attempt=%d ans=%r",
                        record_idx, attempt, candidate_q.get("answer", ""))
            avoid_list.append(qtext[:60]); time.sleep(0.1 * attempt); continue

        # Global fingerprint dedup (all types)
        fp = _question_fingerprint(candidate_q)
        is_dup = False
        if fp_lock:
            with fp_lock:
                if fp in seen_fps: is_dup = True
                else: seen_fps.add(fp)
        else:
            if fp in seen_fps: is_dup = True
            else: seen_fps.add(fp)
        if is_dup:
            avoid_list.append(_question_stem(candidate_q))
            time.sleep(0.1 * attempt); continue

        # MCQ answer fingerprint dedup
        if seen_answer_fps is not None:
            ans_fp = _answer_fingerprint(candidate_q)
            if ans_fp:
                is_ans_dup = False
                if answer_fp_lock:
                    with answer_fp_lock:
                        if ans_fp in seen_answer_fps: is_ans_dup = True
                        else: seen_answer_fps.add(ans_fp)
                else:
                    if ans_fp in seen_answer_fps: is_ans_dup = True
                    else: seen_answer_fps.add(ans_fp)
                if is_ans_dup:
                    logger.info("MCQ rejected: same-answer near-duplicate record=%d "
                                "ans_fp=%r", record_idx, ans_fp[:60])
                    avoid_list.append(_question_stem(candidate_q))
                    time.sleep(0.1 * attempt); continue

        # All checks passed — register in type-specific trackers before returning
        if display_type == "truefalse" and seen_tf_by_concept is not None and tf_concept_lock is not None:
            _register_tf_question(candidate_q, seen_tf_by_concept, tf_concept_lock)

        if display_type == "open_ended" and seen_open_combos is not None and open_combos_lock is not None:
            _register_open_question(topic, bloom, seen_open_combos, open_combos_lock)

        return candidate_q

    return None

# =====================================================
# BATCH GENERATOR
# =====================================================
def generate_from_records(records, max_items=None):
    limit = min(len(records), max_items) if max_items else len(records)

    topic_slot_counter: Dict[str, List[int]] = {}
    slots = []
    for i in range(limit):
        rec       = records[i]
        input_obj = rec.get("input", {}) if isinstance(rec, dict) else {}
        topic     = (input_obj.get("concept") or input_obj.get("topic") or rec.get("instruction", "General")) or "General"
        raw_bloom = (input_obj.get("bloom") or (rec.get("output", {}) or {}).get("bloom") or "Remembering")
        raw_type  = (input_obj.get("type")  or (rec.get("output", {}) or {}).get("type")  or "mcq")
        prompt_type  = normalize_type(raw_type)
        display_type = normalize_out_type({"mcq": "mcq", "tf": "truefalse", "open": "open_ended"}.get(prompt_type, prompt_type))
        bloom        = normalize_bloom(raw_bloom, slot_index=i)
        candidate    = (input_obj.get("context") or input_obj.get("learn_material") or
                        input_obj.get("file_path") or rec.get("file_path") or "")
        full_text = lesson_from_upload(candidate) if candidate else ""
        context   = ""
        if full_text:
            chunks    = get_chunks_for_text(full_text)
            text_hash = hashlib.md5(full_text[:256].encode("utf-8", errors="ignore")).hexdigest()[:8]
            topic_key = f"{topic}::{text_hash}"
            base_idx  = _find_best_chunk_idx(chunks, topic)
            if topic_key not in topic_slot_counter:
                idxs = list(range(len(chunks)))
                if base_idx in idxs: idxs.remove(base_idx)
                idxs.insert(0, base_idx); random.shuffle(idxs[1:])
                topic_slot_counter[topic_key] = idxs
            idxs      = topic_slot_counter[topic_key]
            chunk_idx = idxs.pop(0)
            if not idxs: topic_slot_counter.pop(topic_key, None)
            context = chunks[chunk_idx]
            logger.info("record=%d topic=%r bloom=%s -> chunk %d/%d", i+1, topic, bloom, chunk_idx+1, len(chunks))
        else:
            logger.warning("record=%d topic=%r has NO learning material.", i+1, topic)
        slots.append({"record_idx": i, "topic": topic, "bloom": bloom,
                      "prompt_type": prompt_type, "display_type": display_type,
                      "context": context, "record": rec})

    # Shared dedup state (all types)
    _fp_lock         = threading.Lock()
    _answer_fp_lock  = threading.Lock()
    _stems_lock      = threading.Lock()
    seen_fps:         set  = set()
    seen_answer_fps:  set  = set()
    seen_stems:       list = []

    # Fix 14/15 (v4.2): TF per-concept tracker
    _tf_concept_lock: threading.Lock = threading.Lock()
    seen_tf_by_concept: Dict[str, List[Tuple[frozenset, str]]] = {}

    # Opt 8 (v4.2): open-ended bloom+concept tracker
    _open_combos_lock: threading.Lock = threading.Lock()
    seen_open_combos: Set[str] = set()

    with _gen_progress_lock:
        _gen_progress["current"] = 0
        _gen_progress["total"]   = len(slots)
        _gen_progress["active"]  = True

    def _run_slot(slot: dict):
        with _stems_lock:
            stems_snapshot = list(seen_stems)

        q = _generate_single(
            slot["topic"], slot["prompt_type"], slot["display_type"],
            slot["bloom"], slot["context"],
            MAX_TOKENS_SINGLE.get(slot["prompt_type"], 256),
            seen_fps, slot["record_idx"] + 1,
            seen_questions=stems_snapshot,
            fp_lock=_fp_lock,
            seen_answer_fps=seen_answer_fps,
            answer_fp_lock=_answer_fp_lock,
            seen_tf_by_concept=seen_tf_by_concept,
            tf_concept_lock=_tf_concept_lock,
            seen_open_combos=seen_open_combos,
            open_combos_lock=_open_combos_lock,
        )

        with _gen_progress_lock:
            _gen_progress["current"] += 1

        if q is not None:
            stem = _question_stem(q)
            if stem:
                with _stems_lock:
                    seen_stems.append(stem)
                    if len(seen_stems) > 8:
                        seen_stems.pop(0)

        return slot["record_idx"], slot, q

    workers = min(GENERATION_WORKERS, len(slots))
    logger.info("Generating %d questions with %d worker(s)", len(slots), workers)
    slot_start = time.time()

    results: List[Optional[tuple]] = [None] * len(slots)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_run_slot, slot): i for i, slot in enumerate(slots)}
        for future in as_completed(future_map):
            orig_i, slot, q = future.result()
            results[orig_i] = (slot, q)

    elapsed = time.time() - slot_start
    logger.info("All %d questions generated in %.1fs (%.1f s/q avg)",
                len(slots), elapsed, elapsed / max(len(slots), 1))

    with _gen_progress_lock:
        _gen_progress["active"] = False

    out_questions = []
    for slot, q in results:        # type: ignore[misc]
        if q is not None:
            out_questions.append(q)
        else:
            rec      = slot["record"]
            fallback = rec.get("output") if isinstance(rec, dict) else None
            if fallback and isinstance(fallback, dict):
                out_questions.append(normalize_generated_question(
                    fallback, slot["display_type"], slot["topic"], slot["bloom"]))
            else:
                out_questions.append({
                    "type": slot["display_type"], "concept": slot["topic"],
                    "bloom": slot["bloom"],
                    "question": f"[GENERATION FAILED] Review this item — {slot['topic']}",
                    "choices": (["(Generation failed)"] * 4 if slot["display_type"] == "mcq" else []),
                    "answer": "", "answer_text": "Generation failed. Please delete or replace.",
                    "_generation_failed": True,
                })
    return out_questions

def generate_quiz_for_topics(records_or_topics, max_items=None, test_labels=None, *args, **kwargs):
    try: quizzes = generate_from_records(records_or_topics, max_items)
    except Exception as e:
        logger.exception("generate_from_records error: %s", e); quizzes = []
    if isinstance(quizzes, dict) and "quizzes" in quizzes: quizzes = quizzes["quizzes"]
    elif not isinstance(quizzes, list):
        try: quizzes = list(quizzes)
        except Exception: quizzes = []
    if test_labels and isinstance(test_labels, (list, tuple)):
        for idx, item in enumerate(quizzes):
            if isinstance(item, dict):
                item["test_header"] = test_labels[idx] if idx < len(test_labels) else ""
    return {"quizzes": quizzes}

def get_model_cache_stats(): return {"cache_hits": CACHE_HITS, "cache_misses": CACHE_MISSES}

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: records.append(json.loads(line))
            except Exception as e: logger.warning("Skipping bad line: %s", e)
    return records

# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(title="AutoTOS AI Service", version="4.2-ollama")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])

class GenerateRequest(BaseModel):
    records:     List[dict]
    max_items:   Optional[int]       = None
    test_labels: Optional[List[str]] = None

class ExtractRequest(BaseModel):
    data: str

@app.get("/health")
async def health():
    ready = _ollama_ready or _check_ollama()
    return {"status": "ok" if ready else "degraded",
            "ollama_ready": ready,
            "ollama_model": OLLAMA_MODEL,
            "chunk_size": CHUNK_SIZE}

@app.get("/cache_stats")
async def cache_stats():
    return get_model_cache_stats()

@app.get("/progress")
async def generation_progress():
    with _gen_progress_lock:
        return dict(_gen_progress)

@app.post("/extract")
async def extract_text(req: ExtractRequest):
    try:
        text = await run_in_threadpool(lambda: lesson_from_upload(req.data))
        return {"text": text or ""}
    except Exception as e:
        logger.exception("extract error: %s", e)
        raise HTTPException(status_code=500, detail="Extraction failed")

@app.post("/generate")
async def generate(req: GenerateRequest):
    if not req.records:
        raise HTTPException(status_code=400, detail="records must be non-empty")
    try:
        resp = await run_in_threadpool(
            lambda: generate_quiz_for_topics(req.records, max_items=req.max_items,
                                             test_labels=req.test_labels))
        return resp
    except Exception as e:
        logger.exception("generate error: %s", e)
        raise HTTPException(status_code=500, detail="Generation failed")

@app.post("/generate_from_records")
async def generate_from_records_endpoint(payload: GenerateRequest):
    try:
        out = await run_in_threadpool(
            lambda: generate_from_records(payload.records, max_items=payload.max_items))
        return {"quizzes": out}
    except Exception as e:
        logger.exception("generate_from_records error: %s", e)
        raise HTTPException(status_code=500, detail="Generation failed")

if __name__ == "__main__":
    import argparse, uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", "-j", required=False)
    parser.add_argument("--sample", "-n", type=int, default=None)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    if args.jsonl and not args.serve:
        recs = load_jsonl(args.jsonl)
        n = args.sample or min(5, len(recs))
        for r in generate_from_records(recs[:n], max_items=n):
            print(json.dumps(r, indent=2, ensure_ascii=False))
    elif args.serve:
        uvicorn.run("ai_model:app", host="0.0.0.0", port=8000, log_level="info")