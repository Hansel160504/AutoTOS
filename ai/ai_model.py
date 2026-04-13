# AutoTOS AI module — Ollama backend
#
# ── Patch history ──────────────────────────────────────────────────────────
# v4.9  – Instruction trim, Bloom pattern relaxation.
# v4.10 – Qwen3-4B token + negation fixes.
# v4.11 – Term-def expansion (Fix-44), "which statement best" hard-reject
#          (Fix-45), TF context-framing rejection (Fix-46), instruction
#          rewrite for small models (Fix-47), open-ended verb diversity (Fix-48).
#
# v4.12 (MCQ diversity + fallback validation pass)
#
#       Fix-49 – Fallback quality gate.
#                The dataset fallback path (rec.get("output")) previously
#                bypassed ALL validators. Analysis of the 3,603-record
#                JSONL showed: 28 term questions, 83 "which statement best",
#                218 same-first-word choice sets. Q5/Q7/Q13/Q19 in the
#                failing output were all fallbacks, not model outputs.
#                `_is_valid_fallback_question()` now screens fallbacks with
#                the same core checks before appending.
#
#       Fix-50 – Same-first-word choice detection.
#                Added `_choices_have_same_first_word()`. If all 4 choices
#                start with the same core word (e.g. all "they" / "by" /
#                "role"), the question is rejected. Root cause: the model
#                sometimes generates "A role is..." × 4 and the letter-prefix
#                stripper reveals the shared stem. 218 such cases exist in the
#                training dataset, which also seeds them into fallbacks.
#
#       Fix-51 – MCQ negation in question stem.
#                "Which of the following is NOT..." is poor test design.
#                Added `_is_mcq_negation_question()` to reject questions
#                containing "is not / are not / does not / cannot" in the
#                MCQ question stem. Also added to instruction FORBIDDEN list
#                and to the fallback gate.
#
#       Fix-52 – "How does" opener — per-concept cap.
#                After banning "Which statement best" (Fix-45), the model
#                defaulted to "How does X affect/impact/contribute" for every
#                Understanding MCQ (Q6, Q8, Q10 in test output — all "How
#                does"). Fix: cap at 1 "how" opener per concept total. On
#                cap breach, the retry note explicitly lists alternative
#                starters. Instruction starters for Understanding now avoid
#                "How does" entirely.
#
#       Fix-53 – MCQ sub-topic saturation via per-concept content-word Jaccard.
#                Q8 and Q10 were both about "consistency" for UI Basics.
#                Added `_is_mcq_subtopic_saturated()` with per-concept word
#                set (Jaccard ≥ 0.50). Content-word filter lowered from
#                len > 4 to len > 3 so "role/user/data/code" are now tracked.
#                Also lowered the choices-level Jaccard to the same rule so
#                that semantically identical choice sets are caught earlier.
#
#       Fix-54 – Choices content-word threshold lowered 5 → 4 chars.
#                Short domain words like "role", "user", "data", "code",
#                "port" are meaningful; excluding them caused the 218
#                same-stem choice sets to slip through `_has_semantic_
#                duplicate_choices`. Threshold now > 3 (i.e., 4+ chars).
# ============================================================

import os
import re
import json
import fitz
import base64
import hashlib
import logging
import random
import tempfile
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from docx import Document
from pptx import Presentation

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

GENERATION_WORKERS = int(os.environ.get("GENERATION_WORKERS", "1"))

MAX_RETURN    = 50_000
CHUNK_SIZE    = 600
CHUNK_OVERLAP = 20

CACHE_MAX_FILES = 2_000

BASE_DIR        = os.path.dirname(__file__)
CACHE_DIR       = os.path.join(BASE_DIR, ".extracted_cache")
MODEL_CACHE_DIR = os.path.join(BASE_DIR, ".model_cache")
os.makedirs(CACHE_DIR,       exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

logger.info("Ollama config: base_url=%s model=%s timeout=%ds",
            OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT)

SESSION = requests.Session()

# =====================================================
# IN-MEMORY LRU CACHE
# =====================================================
class _LRUMemCache:
    def __init__(self, maxsize: int = 512) -> None:
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

_mem_cache = _LRUMemCache(maxsize=512)

# =====================================================
# PER-TYPE num_ctx
# =====================================================
_NUM_CTX: Dict[str, int] = {
    "mcq":  1024,
    "tf":   1024,
    "open": 1024,
}

# =====================================================
# TOKEN BUDGETS
# =====================================================
MAX_TOKENS_SINGLE: Dict[str, int] = {
    "mcq":  160,
    "tf":    85,
    "open":  90,
}

# =====================================================
# INSTRUCTIONS  (v4.12)
#
# v4.12 changes from v4.11:
#   • "How does X affect/impact/contribute/influence" added to FORBIDDEN
#     (Fix-52). After banning "which statement best", the model defaulted
#     to "how does" for every Understanding MCQ.
#   • "Which of the following is not" added to FORBIDDEN (Fix-51).
#   • Understanding starters completely replaced — no "How does" at all
#     (Fix-52). Now: Why / What happens / In what way / What role.
#   • Remembering starters clarified to discourage "What is the term for".
# =====================================================
AUTOTOS_INSTRUCTION = (
    "You are AutoTOS. Output ONLY valid JSON.\n"
    "\n"
    "NEVER start a question with:\n"
    "  'What is the term' | 'Which term' | 'What does the term' | 'What is the definition'\n"
    "  'Which statement best summarizes' | 'Which statement best explains'\n"
    "  'Which statement best describes' | 'Which of the following statements'\n"
    "  'How does' | 'How do' | 'How is' | 'How can' | 'How would' | 'How will'\n"
    "  'Which of the following is not' | 'Which is not' | 'Which are not'\n"
    "  'Analyze [anything]: Which' | 'Evaluate [anything]: Which best'\n"
    "\n"
    "NEVER put these words in a True/False statement:\n"
    "  not | never | don't | doesn't | isn't | aren't | cannot | can't | won't | no\n"
    "NEVER start a True/False statement with:\n"
    "  'In the context of' | 'In a scenario where' | 'Given that'\n"
    "  'The term ... refers to' | 'X is defined as' | 'Assuming that'\n"
    "\n"
    "MCQ JSON:\n"
    "{\"type\":\"mcq\",\"concept\":\"...\",\"bloom\":\"...\",\"question\":\"...\","
    "\"choices\":[\"...\",\"...\",\"...\",\"...\"],\"answer\":\"A|B|C|D\","
    "\"answer_text\":\"1 sentence why correct.\"}\n"
    "\n"
    "TF JSON:\n"
    "{\"type\":\"truefalse\",\"concept\":\"...\",\"bloom\":\"...\",\"question\":\"...\","
    "\"answer\":\"true|false\",\"answer_text\":\"1 sentence why correct.\"}\n"
    "\n"
    "MCQ: exactly 4 choices, no A/B/C/D prefix, answer = A/B/C/D letter.\n"
    "Choices must be meaningfully different — never start all 4 with the same word.\n"
    "\n"
    "Question starters by Bloom level:\n"
    "  Remembering  → 'What is the primary purpose of X?' | 'What causes X?' | 'Which [noun] is used for X?' | 'When does X occur?'\n"
    "  Understanding → 'Why is X important for Y?' | 'What happens when X occurs?' | 'In what way does X affect Y?' | 'What role does X play in Y?'\n"
    "  Applying     → 'Given X, which approach should be used?' | 'How would you implement X?' | 'Which solution best addresses X?'\n"
    "  Analyzing    → 'What is the relationship between X and Y?' | 'What distinguishes X from Y?' | 'What is the main difference between X and Y?'\n"
    "  Evaluating   → 'Which approach is most effective for X?' | 'What is the main weakness of X?' | 'Assess X; what is its primary limitation?'\n"
    "  Creating     → 'Which design best addresses X?' | 'Select the approach that best achieves Y.' | 'Which plan most effectively combines X and Y?'\n"
)

AUTOTOS_INSTRUCTION_OPEN = (
    "You are AutoTOS. Output ONLY valid JSON.\n"
    "\n"
    "NEVER start a question with:\n"
    "  'What does the term' | 'What does ... mean' | 'What is the definition of'\n"
    "  'The term ... refers to' | 'What is the term' | 'Which term'\n"
    "\n"
    "Open-ended JSON:\n"
    "{\"type\":\"open_ended\",\"concept\":\"...\",\"bloom\":\"...\",\"question\":\"...\","
    "\"answer\":\"<exactly 1 complete sentence>\"}\n"
    "\n"
    "answer: exactly 1 complete sentence ending with a period.\n"
    "Question must require application, analysis, evaluation, or creation.\n"
    "\n"
    "Question starters by Bloom level:\n"
    "  Applying   → 'Apply X to show...' | 'Demonstrate how...' | 'Solve...' | 'Use X to...'\n"
    "  Analyzing  → 'Analyze...' | 'Examine the relationship between...' | 'Compare X and Y...'\n"
    "  Evaluating → 'Evaluate...' | 'Assess...' | 'Justify...' | 'Critique...' | 'Judge the effectiveness of...'\n"
    "  Creating   → 'Design...' | 'Propose...' | 'Formulate...' | 'Develop...' | 'Construct a plan for...'\n"
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
        r = SESSION.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
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

def _prompt_hash_key(prompt: str, max_tokens: int,
                     temperature: float, num_ctx: int) -> str:
    key = f"{OLLAMA_MODEL}|{max_tokens}|{temperature:.4f}|{num_ctx}|{prompt}"
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

_cache_write_count = 0
_CLEANUP_EVERY = 50

def write_model_cache(key: str, obj: Any) -> None:
    global _cache_write_count
    target = _model_cache_path(key)
    try:
        fd, tmp_path = tempfile.mkstemp(dir=MODEL_CACHE_DIR, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False)
            os.replace(tmp_path, target)
        except Exception:
            try: os.remove(tmp_path)
            except Exception: pass
            raise
    except Exception as e:
        logger.debug("write_model_cache failed: %s", e)
        return
    _cache_write_count += 1
    if _cache_write_count % _CLEANUP_EVERY == 0:
        _cleanup_model_cache()

def _cleanup_model_cache(max_files: int = CACHE_MAX_FILES) -> None:
    try:
        all_files = [
            os.path.join(MODEL_CACHE_DIR, f)
            for f in os.listdir(MODEL_CACHE_DIR)
            if f.endswith(".json")
        ]
        if len(all_files) <= max_files:
            return
        all_files.sort(key=lambda p: os.path.getmtime(p))
        to_delete = all_files[:len(all_files) - max_files]
        for path in to_delete:
            try: os.remove(path)
            except Exception: pass
        logger.info("Cache cleanup: removed %d old entries (%d remain).",
                    len(to_delete), max_files)
    except Exception as e:
        logger.warning("Cache cleanup failed (non-fatal): %s", e)

_cleanup_model_cache()

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

_FILL_IN_BLANK_RE = re.compile(r'_{3,}')

def is_fill_in_the_blank(question: str) -> bool:
    if not question: return False
    return bool(_FILL_IN_BLANK_RE.search(question))

def normalize_mcq_answer(answer_value: str, choices: list) -> str:
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

_ANSWER_TEXT_MAX_CHARS = 220

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

def _truncate_open_answer(text: str) -> str:
    if not text: return ""
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r'^(answer\s*[:\-]\s*)', '', t, flags=re.IGNORECASE).strip()
    if not t: return ""
    _ABR_RE = re.compile(r'\b(e\.g|i\.e|etc|vs|Mr|Mrs|Dr|Prof|Sr|Jr|St|approx|fig|no)\.\s', re.IGNORECASE)
    masked = _ABR_RE.sub(lambda m: m.group(0).replace('.', '\x00'), t)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$', masked)
    first = parts[0].replace('\x00', '.').strip() if parts else t.strip()
    if first and first[-1] not in ".!?":
        first = first + "."
    return first

# =====================================================
# CHUNKING
# =====================================================
_chunk_cache: Dict[str, List[str]] = {}

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
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
    key = hashlib.md5(full_text[:4096].encode("utf-8", errors="ignore")).hexdigest()
    if key not in _chunk_cache:
        _chunk_cache[key] = chunk_text(full_text)
        logger.info("Chunked document: %d chars -> %d chunks",
                    len(full_text), len(_chunk_cache[key]))
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
# PROMPT BUILDER
# =====================================================
def _build_avoid_block(seen_questions: List[str]) -> str:
    if not seen_questions: return ""
    recent = seen_questions[-3:]
    items  = "; ".join(f'"{q[:25]}"' for q in recent)
    return f"\n[Avoid repeating: {items}]\n"


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
        tf_note = (
            "\nTF RULES: Write a POSITIVE declarative statement (no question mark).\n"
            "FORBIDDEN WORDS in statement: not, never, don't, doesn't, isn't, aren't, cannot, can't, won't, no\n"
            "FORBIDDEN statement starts: 'In the context of', 'In a scenario where', 'Given that', "
            "'The term ... refers to', 'X is defined as'\n"
        )

    retry_line = f"\n{attempt_note.strip()}\n" if attempt_note and attempt_note.strip() else ""
    ctx_suffix = f"{tf_note}{retry_line}{avoid_block}"

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
              temperature: float = 0.45,
              num_ctx: int = 1024) -> Optional[dict]:
    global CACHE_HITS, CACHE_MISSES
    if not _ollama_ready and not _check_ollama():
        logger.error("Ollama not ready.")
        return None

    prompt    = sanitize_prompt(prompt)
    cache_key = _prompt_hash_key(prompt, max_tokens, temperature, num_ctx)

    mem_hit = _mem_cache.get(cache_key)
    if mem_hit is not None:
        CACHE_HITS += 1
        return mem_hit if isinstance(mem_hit, dict) else None

    cached = read_model_cache(cache_key)
    if cached is not None:
        CACHE_HITS += 1
        _mem_cache.set(cache_key, cached)
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
            "num_ctx":     num_ctx,
            "top_p":       0.95,
            "think":       False,
            "stop":        ["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
        },
    }

    try:
        start = time.time()
        resp  = SESSION.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        duration = time.time() - start

        if not resp.ok:
            logger.warning("Ollama returned %d: %s", resp.status_code, resp.text[:200])
            return None

        raw = resp.json().get("response", "")
        logger.info("ask_model duration=%.2fs raw_len=%d max_tok=%d temp=%.2f num_ctx=%d",
                    duration, len(raw), max_tokens, temperature, num_ctx)

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
        _mem_cache.set(cache_key, parsed)
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
        resp = SESSION.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": "Say OK",
                "stream": False,
                "options": {"num_predict": 3, "temperature": 0.0, "num_ctx": 1024},
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
# CHOICE PREFIX STRIPPER
# =====================================================
_CHOICE_LETTER_PREFIX_RE = re.compile(
    r'^(?:\([A-Da-d]\)|[A-Da-d][).:\-]\s*|[A-Da-d]\s+(?=[A-Z]))'
)

def _strip_choice_letter_prefix(text: str) -> str:
    if not text:
        return text
    stripped = _CHOICE_LETTER_PREFIX_RE.sub("", text).strip()
    if stripped == text and len(text) > 2 and text[0].upper() in "ABCD" and text[1] == " ":
        stripped = text[2:].strip()
    return stripped


def _choice_has_letter_prefix(text: str) -> bool:
    if not text or len(text) < 2:
        return False
    return bool(_CHOICE_LETTER_PREFIX_RE.match(text))


# =====================================================
# Fix-37 (v4.9): Bloom-level question-starter validator
# =====================================================
_BLOOM_MCQ_PATTERNS: Dict[str, re.Pattern] = {
    "Analyzing": re.compile(
        r'^(what (is|are) the (relationship|difference|distinction|cause|root cause|interaction|component|underlying)\b|'
        r'what (causes?|distinguishes?|differentiates?|connects?|contributes?)\b|'
        r'(analyze|compare|contrast|examine|distinguish|differentiate|categorize|dissect|break down|identify)\b|'
        r'which (architectural|component|structural|key|primary|underlying) (feature|difference|factor|element)\b|'
        r'(analyze|examine) .{3,60}; what (does|is|are)\b)',
        re.IGNORECASE
    ),
    "Evaluating": re.compile(
        r'^(which (approach|method|system|strategy|solution|framework|tool|option|argument|choice|design) '
        r'(is most|would be most|best|most effectively|is the most|is recommended|provides the strongest)\b|'
        r'what is the (strongest|most critical|weakest|most significant|most effective)\b|'
        r'(assess|evaluate|judge|defend|appraise|prioritize|grade|critique|rate|justify|recommend|value)\b|'
        r'assess .{3,60}; what (is|are)\b|'
        r'judge (the|whether)\b|'
        r'which .{5,60} (most effective|most reliable|most compliant|best suited|most appropriate)\b)',
        re.IGNORECASE
    ),
    "Creating": re.compile(
        r'^(which (of the following|design|approach|configuration|method|plan|outline|schema|strategy|combination|'
        r'blueprint|structure|pattern|formulation) (represents?|best (designs?|structures?|synthesizes?|invents?|'
        r'constructs?|formulates?|outlines?|proposes?|organizes?|creates?|builds?|establishes?)|'
        r'most (optimized|innovative|effective|novel))\b|'
        r'(design|develop|modify|invent|write|rewrite|collaborate|construct|formulate|compose|plan|propose|'
        r'produce|generate|choose|select|create|build)\b|'
        r'choose the .{3,60} that best\b|'
        r'select the .{3,60} that (best|most)\b)',
        re.IGNORECASE
    ),
}

def _is_valid_mcq_bloom_pattern(question: str, bloom: str) -> bool:
    if not question or len(question) < 10:
        return True
    pattern = _BLOOM_MCQ_PATTERNS.get(bloom)
    if pattern is None:
        return True
    return bool(pattern.match(question))


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
        elif display_type == "open_ended":
            out["answer"] = _truncate_open_answer(clean_text(a))
        else:
            out["answer"] = a
    else:
        raw_ans = clean_text(str(ans or ""))
        if display_type == "open_ended":
            out["answer"] = _truncate_open_answer(raw_ans)
        else:
            out["answer"] = raw_ans

    if display_type == "open_ended" and out["answer"]:
        out["answer"] = _truncate_open_answer(out["answer"])

    return out

# =====================================================
# FINGERPRINTING
# =====================================================
_FP_STOPWORDS = re.compile(
    r'\b(a|an|the|and|or|of|in|on|to|for|is|are|was|were|by|at|be|its|it|'
    r'that|this|these|those|with|as|from|into|about|which|how|what|does|do)\b')
_FP_FILLER_RE = re.compile(
    r"^(what (is|are|does|do) (the )?(primary |main |key )?"
    r"(focus|purpose|function|role|goal|aim|definition|meaning|concept|"
    r"example|reason|impact|effect|difference|advantage|use|importance) of\s*|"
    r"what does (the term|the word|the phrase|the concept)\s+.{0,40}(refer|mean|stand for|describe)\s*(to\s*)?\b|"
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
    if best_letter != answer_letter and (best_score - chosen_score) >= 3:
        return False, best_letter
    return True, answer_letter


# =====================================================
# Fix-50/54 (v4.12): SAME-FIRST-WORD CHOICE DETECTOR
#
# If all 4 choices start with the same core noun/pronoun word, the MCQ
# is definitionally bad — choices are variants of the same idea.
# 137 such cases were found in the training JSONL (they/it/role/user…).
#
# EXCEPTION: prepositions/conjunctions ("by/in/to/with/for/from…") are
# legitimate starters for How/Why answer options ("By reducing X",
# "By encrypting Y"). 81 such cases in the dataset are valid and should
# not be rejected — the Jaccard check handles semantic duplicates there.
# =====================================================
_PREPOSITION_STARTERS = frozenset({
    "by", "in", "to", "with", "for", "from", "at", "on", "through",
    "via", "using", "when", "if", "while", "after", "before",
    "during", "because", "although", "since", "as", "that", "upon",
})

def _choices_have_same_first_word(choices: list) -> bool:
    """Return True if all choices share the same leading content word (non-preposition)."""
    if not choices or len(choices) < 4:
        return False
    first_words = []
    for c in choices:
        words = (c or "").lower().split()
        if not words:
            return False
        fw = words[0]
        # Skip leading articles
        if fw in ("a", "an", "the") and len(words) > 1:
            fw = words[1]
        first_words.append(fw)
    if len(set(first_words)) != 1 or not first_words[0]:
        return False
    # Allow prepositions/conjunctions — they legitimately start clause fragments
    if first_words[0] in _PREPOSITION_STARTERS:
        return False
    logger.info("MCQ rejected: all choices start with same noun/pronoun: %r", first_words[0])
    return True


# =====================================================
# Fix-54 (v4.12): SEMANTIC DUPLICATE CHOICES
# Content-word threshold lowered from > 4 to > 3 chars.
# =====================================================
_SEM_DUP_SW = re.compile(
    r'\b(a|an|the|and|or|of|in|on|to|for|is|are|was|were|by|at|be|its|it|'
    r'that|this|these|those|with|as|from|they|their|such|each|used|'
    r'using|allows|allow|makes|make|uses|use|helps|help|enables|enable|'
    r'provides|provide|requires|require|ensures|ensure|offer|offers)\b',
    re.IGNORECASE
)

def _has_semantic_duplicate_choices(choices: list) -> bool:
    if not choices or len(choices) < 2:
        return False

    def _content(text: str) -> set:
        t = _SEM_DUP_SW.sub(" ", text.lower())
        # Fix-54: threshold lowered to > 3 (4+ chars) to include "role", "user", "data"
        return {w for w in re.findall(r'\b\w+\b', t) if len(w) > 3}

    # Fix-50: same-first-word check
    if _choices_have_same_first_word(choices):
        return True

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
# BLANK-COMPLETION VALIDATOR
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
# Fix-44 (v4.11) + Fix-49 (v4.12): TERM-DEFINITION REJECTION
# =====================================================
_TERM_DEF_QUESTION_RE = re.compile(
    r"^("
    r"what does (the term|the word|the phrase|the concept)\s*.{1,50}(refer to|mean|stand for|primarily refer|represent)\b|"
    r"what is (the term|a term) (used to|for|that)\b|"
    r"what (is|are) (the term|the definition of|the meaning of)\s|"
    r"which term (describes?|refers? to|identifies?|defines?|is used|best describes?|best refers?)\b|"
    r"what term (describes?|refers? to|identifies?|is used)\b"
    r")",
    re.IGNORECASE
)

def _is_term_definition_question(question: str) -> bool:
    return bool(_TERM_DEF_QUESTION_RE.match(question or ""))


# =====================================================
# Fix-45 (v4.11): "WHICH STATEMENT BEST" REJECTION
# =====================================================
_WHICH_STMT_BEST_RE = re.compile(
    r"^which (statement|of the following statements?|option|answer)\s+"
    r"(best |most )?"
    r"(summarizes?|explains?|describes?|illustrates?|represents?|captures?|"
    r"details?|outlines?|reflects?|shows?|demonstrates?|conveys?)",
    re.IGNORECASE
)

def _is_which_statement_best(question: str) -> bool:
    return bool(_WHICH_STMT_BEST_RE.match(question or ""))


# =====================================================
# Fix-51 (v4.12): MCQ NEGATION IN QUESTION STEM
#
# "Which of the following is NOT..." is poor test design — it tests
# elimination rather than recall/understanding, and is confusing.
# =====================================================
_MCQ_NEGATION_STEM_RE = re.compile(
    r"\b(is not|are not|does not|do not|cannot|can't|doesn't|aren't|isn't|"
    r"which is not|which are not|which does not|which cannot|"
    r"that is not|that are not|not a recognized|not considered|not an example)\b",
    re.IGNORECASE
)

def _is_mcq_negation_question(question: str) -> bool:
    """Reject MCQ questions containing negation in the question stem."""
    return bool(_MCQ_NEGATION_STEM_RE.search(question or ""))


# =====================================================
# Fix-52 (v4.12): MCQ "HOW DOES" OPENER — PER-CONCEPT CAP
#
# After banning "which statement best" (Fix-45), the model defaulted to
# "How does X affect/impact/contribute" for every Understanding MCQ.
# This extracts the opener category and enforces a per-concept cap.
# =====================================================
_MCQ_OPENER_CATEGORY_RE = re.compile(
    r"^(how does|how do|how is|how are|how can|how would|how will|how should|how could|"
    r"which of the following|"
    r"what is|what are|what was|"
    r"why is|why are|why does|why do|"
    r"what happens|"
    r"in what way|"
    r"what role|"
    r"what makes|"
    r"when does|when is|"
    r"where is|where are)",
    re.IGNORECASE
)

# Openers that trigger the per-concept cap (max 1 per concept)
_MCQ_CAPPED_OPENERS = {"how", "which_of"}

def _extract_mcq_opener(question: str) -> str:
    """Extract a normalized opener category from an MCQ question."""
    q = (question or "").strip().lower()
    m = _MCQ_OPENER_CATEGORY_RE.match(q)
    if not m:
        return "other"
    raw = m.group(1).lower()
    if raw.startswith("how"):
        return "how"
    if raw.startswith("which of"):
        return "which_of"
    if raw.startswith("what is") or raw.startswith("what are") or raw.startswith("what was"):
        return "what_is"
    if raw.startswith("why"):
        return "why"
    if raw.startswith("what happens"):
        return "what_happens"
    if raw.startswith("in what"):
        return "in_what_way"
    if raw.startswith("what role"):
        return "what_role"
    if raw.startswith("what makes"):
        return "what_makes"
    return "other"

def _is_mcq_opener_overused(
    candidate_q: dict,
    seen_mcq_openers: Dict[str, Dict[str, int]],
    mcq_opener_lock: threading.Lock,
) -> bool:
    """Return True if this opener has reached its per-concept cap."""
    concept = (candidate_q.get("concept") or "").lower().strip()
    opener  = _extract_mcq_opener(candidate_q.get("question") or "")
    if opener not in _MCQ_CAPPED_OPENERS:
        return False
    limit = 1  # max 1 of each capped opener per concept
    with mcq_opener_lock:
        counts = seen_mcq_openers.get(concept, {})
        return counts.get(opener, 0) >= limit

def _register_mcq_opener(
    candidate_q: dict,
    seen_mcq_openers: Dict[str, Dict[str, int]],
    mcq_opener_lock: threading.Lock,
) -> None:
    concept = (candidate_q.get("concept") or "").lower().strip()
    opener  = _extract_mcq_opener(candidate_q.get("question") or "")
    with mcq_opener_lock:
        if concept not in seen_mcq_openers:
            seen_mcq_openers[concept] = {}
        seen_mcq_openers[concept][opener] = seen_mcq_openers[concept].get(opener, 0) + 1

def _mcq_opener_hint(
    concept: str,
    seen_mcq_openers: Dict[str, Dict[str, int]],
    mcq_opener_lock: threading.Lock,
) -> str:
    """Return an attempt-note hint if capped openers are saturated for this concept."""
    with mcq_opener_lock:
        counts = seen_mcq_openers.get(concept.lower().strip(), {})
    notes = []
    if counts.get("how", 0) >= 1:
        notes.append(
            "Do NOT start with 'How does'. Use instead: "
            "'Why is X important?', 'What happens when X?', 'What role does X play?', 'In what way does X affect Y?'"
        )
    if counts.get("which_of", 0) >= 1:
        notes.append(
            "Do NOT start with 'Which of the following'. Ask a specific question instead."
        )
    return " ".join(notes)


# =====================================================
# Fix-53 (v4.12): MCQ SUB-TOPIC SATURATION
#
# Q8 and Q10 were both about "consistency" in UI Basics because the
# fingerprint didn't catch slightly differently worded questions about
# the same sub-concept. A per-concept Jaccard check (threshold 0.50)
# on content words catches this.
# =====================================================
_MCQ_SUBTOPIC_SW = re.compile(
    r'\b(a|an|the|and|or|of|in|on|to|for|is|are|was|were|by|at|be|its|it|'
    r'that|this|these|those|with|as|from|how|what|which|when|where|why|'
    r'does|do|can|will|may|has|have|been|being|they|their|such|each|used|'
    r'primarily|mainly|often|generally|usually|typically|between|among|'
    r'following|correctly|commonly|specifically|properly|typically|'
    r'given|provide|ensure|allow|make|help|support|affect|impact|cause)\b',
    re.IGNORECASE
)

def _mcq_subtopic_words(question: str) -> frozenset:
    q = re.sub(r"[^\w\s]", " ", question.lower())
    q = _MCQ_SUBTOPIC_SW.sub(" ", q)
    q = re.sub(r"\s+", " ", q).strip()
    # threshold: 4+ chars (includes "role", "user", "data")
    return frozenset(w for w in q.split() if len(w) >= 4)

def _is_mcq_subtopic_saturated(
    candidate_q: dict,
    seen_mcq_by_concept: Dict[str, List[frozenset]],
    mcq_concept_lock: threading.Lock,
    threshold: float = 0.50,
) -> bool:
    concept  = (candidate_q.get("concept") or "").lower().strip()
    question = candidate_q.get("question") or ""
    cwords   = _mcq_subtopic_words(question)
    if not cwords:
        return False
    with mcq_concept_lock:
        existing = seen_mcq_by_concept.get(concept, [])
        for ex_words in existing:
            if not ex_words:
                continue
            union   = cwords | ex_words
            jaccard = len(cwords & ex_words) / len(union) if union else 0.0
            if jaccard >= threshold:
                logger.info(
                    "MCQ rejected: sub-topic saturation (J=%.2f) concept=%r q=%r",
                    jaccard, concept, question[:70]
                )
                return True
    return False

def _register_mcq_subtopic(
    candidate_q: dict,
    seen_mcq_by_concept: Dict[str, List[frozenset]],
    mcq_concept_lock: threading.Lock,
) -> None:
    concept  = (candidate_q.get("concept") or "").lower().strip()
    question = candidate_q.get("question") or ""
    cwords   = _mcq_subtopic_words(question)
    with mcq_concept_lock:
        if concept not in seen_mcq_by_concept:
            seen_mcq_by_concept[concept] = []
        seen_mcq_by_concept[concept].append(cwords)


# =====================================================
# Fix-25: LAZY BLOOM OPENER VALIDATOR
# =====================================================
_LAZY_BLOOM_OPENER_RE = re.compile(
    r"^(analyze|evaluate|assess|create|design|develop)\s+.{5,60}[:]\s*"
    r"(which|what).{0,60}(best illustrates|best shows|best describes|best explains|best represents)",
    re.IGNORECASE
)

def _is_lazy_bloom_opener(question: str) -> bool:
    return bool(_LAZY_BLOOM_OPENER_RE.match(question))

# =====================================================
# TF SEMANTIC DUPLICATE DETECTION
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

@lru_cache(maxsize=1024)
def _tf_content_words(question: str) -> frozenset:
    q = re.sub(r"[^\w\s]", " ", question.lower())
    q = _TF_SEM_SW.sub(" ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return frozenset(w for w in q.split() if len(w) > 3)

def _is_tf_semantic_duplicate(
    candidate_q: dict,
    seen_tf_by_concept: Dict[str, List[Tuple[frozenset, str]]],
    tf_lock: threading.Lock,
) -> bool:
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
    concept  = (candidate_q.get("concept") or "").lower().strip()
    question = candidate_q.get("question") or ""
    answer   = (candidate_q.get("answer") or "").lower().strip()
    cwords   = _tf_content_words(question)
    with tf_lock:
        if concept not in seen_tf_by_concept:
            seen_tf_by_concept[concept] = []
        seen_tf_by_concept[concept].append((cwords, answer))

# =====================================================
# TF ANSWER-BALANCE NUDGE
# =====================================================
def _tf_balance_note(
    concept: str,
    seen_tf_by_concept: Dict[str, List[Tuple[frozenset, str]]],
    tf_lock: threading.Lock,
) -> str:
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
# OPEN-ENDED BLOOM+CONCEPT TRACKER (Fix-48 v4.11)
# =====================================================
_OPEN_STARTER_VERB_RE = re.compile(
    r"^(evaluate|assess|analyze|examine|compare|justify|critique|judge|"
    r"design|propose|formulate|develop|construct|create|build|apply|"
    r"demonstrate|solve|use|discuss|explain|describe)\b",
    re.IGNORECASE
)

def _extract_open_starter_verb(question: str) -> str:
    m = _OPEN_STARTER_VERB_RE.match((question or "").strip())
    return m.group(1).lower() if m else ""

def _open_bloom_concept_key(topic: str, bloom: str) -> str:
    return f"{topic.lower().strip()}::{bloom.lower().strip()}"

def _open_diversity_note(
    topic: str,
    bloom: str,
    question: str,
    seen_open_combos: Set[str],
    seen_open_verbs: Dict[str, Set[str]],
    open_lock: threading.Lock,
) -> str:
    key      = _open_bloom_concept_key(topic, bloom)
    verb     = _extract_open_starter_verb(question)
    verb_key = f"{key}::verb"
    notes = []
    with open_lock:
        if key in seen_open_combos:
            notes.append("Different angle — focus on a distinct aspect or example.")
        if verb and verb_key in seen_open_verbs and verb in seen_open_verbs[verb_key]:
            bloom_verbs = {
                "evaluating": ["Assess", "Critique", "Judge", "Justify"],
                "creating":   ["Design", "Formulate", "Propose", "Develop"],
                "applying":   ["Demonstrate", "Apply", "Solve", "Use"],
                "analyzing":  ["Compare", "Examine", "Analyze"],
            }
            alternatives = bloom_verbs.get(bloom.lower(), [])
            alt_str = " / ".join(alternatives) if alternatives else "a different verb"
            notes.append(f"Use a different question starter — try: {alt_str}.")
    return " ".join(notes)

def _register_open_question(
    topic: str,
    bloom: str,
    question: str,
    seen_open_combos: Set[str],
    seen_open_verbs: Dict[str, Set[str]],
    open_lock: threading.Lock,
) -> None:
    key      = _open_bloom_concept_key(topic, bloom)
    verb     = _extract_open_starter_verb(question)
    verb_key = f"{key}::verb"
    with open_lock:
        seen_open_combos.add(key)
        if verb:
            if verb_key not in seen_open_verbs:
                seen_open_verbs[verb_key] = set()
            seen_open_verbs[verb_key].add(verb)

# =====================================================
# VALIDATORS
# =====================================================
_ANSWER_ALWAYS_BAD = {"—", "-", "", "answer:"}
_OPEN_PLACEHOLDERS = {
    "model answer here", "model answer here.", "answer here",
    "write answer here", "<complete model answer based on context>",
    "<write a complete model answer>",
    "<exactly 1 complete sentence answer based on context>",
    "1 complete sentence answer based on context.",
}

_MCQ_CHOICE_PLACEHOLDER_RE = re.compile(
    r'^(choice text|option [abcd]|answer [abcd]|placeholder|n/a|none)\s*$',
    re.IGNORECASE
)

def _is_valid_answer(q: dict, display_type: str) -> bool:
    answer  = clean_text(str(q.get("answer") or "")).strip()
    ans_low = answer.lower().rstrip(".")
    if ans_low in _ANSWER_ALWAYS_BAD: return False

    if display_type == "mcq":
        choices = q.get("choices") or []
        if not answer: return False

        if len(choices) != 4:
            logger.info("MCQ rejected: expected 4 choices, got %d", len(choices))
            return False

        for c in choices:
            if not c or not c.strip():
                logger.info("MCQ rejected: empty choice detected")
                return False
            if _choice_has_letter_prefix(c):
                logger.info("MCQ rejected: choice still has letter prefix: %r", c[:60])
                return False
            if _MCQ_CHOICE_PLACEHOLDER_RE.match(c.strip()):
                logger.info("MCQ rejected: placeholder choice: %r", c[:60])
                return False

        choices_lower = [c.lower().strip() for c in choices]
        answer_lower  = answer.lower().strip()
        if answer_lower not in choices_lower:
            matched = any(
                answer_lower in c or c in answer_lower
                for c in choices_lower
                if len(c) > 5 and len(answer_lower) > 5
            )
            if not matched:
                logger.info(
                    "MCQ rejected: answer %r not found in choices %r",
                    answer[:60], choices_lower
                )
                return False

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
                 if c.lower().strip() == answer_lower),
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
        sentence_count = len(re.findall(r'(?<=[.!?])\s+[A-Z]', answer))
        if sentence_count >= 2:
            logger.info("Open-ended rejected: answer has %d+ sentences: %r",
                        sentence_count + 1, answer[:80])
            return False

    return True


# =====================================================
# Fix-46 (v4.11): TF CONTEXT-FRAMING REJECTION
# =====================================================
_TF_CONTEXT_FRAMING_RE = re.compile(
    r"^(in the context of\b|"
    r"in a scenario where\b|"
    r"given (the context|that [a-z].{3,60}(must|should|can|is|are|will|would))\b|"
    r"assuming (that\b|a\b|an\b)|"
    r"when considering\b|"
    r"under the assumption\b|"
    r"in (this|the) case (where|of)\b)",
    re.IGNORECASE
)

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
        logger.info("TF rejected: double-negative: %r", question[:80])
        return False
    if _TF_CONTEXT_FRAMING_RE.match(question):
        logger.info("TF rejected: context-framing starter: %r", question[:80])
        return False
    return True


# =====================================================
# Fix-49 (v4.12): FALLBACK QUALITY GATE
#
# The dataset fallback path (`rec.get("output")`) previously bypassed
# ALL validators. Analysis showed: 28 term questions, 83 "which statement
# best", 218 same-first-word choice sets in the 3,603-record JSONL.
# Q5/Q7/Q13/Q19 in the failing output were all confirmed fallbacks.
#
# Strategy: reject the most egregious fallbacks (term-defs, "which
# statement best", same-word choices, MCQ negation). Accept borderline
# fallbacks (e.g. slightly repetitive openers) to avoid inflating the
# [GENERATION FAILED] count unnecessarily.
# =====================================================
def _is_valid_fallback_question(q: dict, display_type: str) -> bool:
    """
    Quality gate for dataset fallback questions.
    Only the clearest violations are rejected to avoid inflating failures.
    """
    qtext = (q.get("question") or "").strip()
    if not qtext or len(qtext) < 20:
        return False

    # Always reject term-definition questions
    if _is_term_definition_question(qtext):
        logger.info("Fallback rejected: term-def question: %r", qtext[:70])
        return False

    if display_type == "mcq":
        # Reject "Which statement best..."
        if _is_which_statement_best(qtext):
            logger.info("Fallback rejected: which-statement-best: %r", qtext[:70])
            return False
        # Reject negation in stem
        if _is_mcq_negation_question(qtext):
            logger.info("Fallback rejected: negation in MCQ stem: %r", qtext[:70])
            return False
        choices = q.get("choices") or []
        if len(choices) != 4:
            return False
        # Reject same-first-word choices
        if _choices_have_same_first_word(choices):
            logger.info("Fallback rejected: same-first-word choices for: %r", qtext[:60])
            return False
        answer = (q.get("answer") or "").strip()
        if not answer:
            return False

    elif display_type == "truefalse":
        # Apply full TF validation (includes context-framing check)
        if not _is_valid_tf(q):
            logger.info("Fallback rejected: invalid TF: %r", qtext[:70])
            return False

    return True


# =====================================================
# SINGLE QUESTION GENERATOR
# =====================================================
_RETRY_NOTES = [
    "",
    "Try a different angle — focus on a specific component or detail.",
    "Use a concrete scenario or example from the context.",
]

_MAX_ATTEMPTS = 3

def _generate_single(
    topic, prompt_type, display_type, bloom, context, max_tok,
    seen_fps, record_idx,
    seen_questions=None, fp_lock=None,
    seen_answer_fps=None, answer_fp_lock=None,
    seen_tf_by_concept=None, tf_concept_lock=None,
    seen_open_combos=None, open_combos_lock=None,
    seen_open_verbs=None,
    seen_term_defs=None, term_def_lock=None,
    seen_mcq_by_concept=None, mcq_concept_lock=None,
    seen_mcq_openers=None, mcq_opener_lock=None,
):
    instruction = AUTOTOS_INSTRUCTION_OPEN if prompt_type == "open" else AUTOTOS_INSTRUCTION
    ctx_size    = _NUM_CTX.get(prompt_type, 1024)

    all_seen   = list(seen_questions or [])
    concept_lc = topic.lower()
    concept_specific = [s for s in all_seen if concept_lc in s.lower()]
    avoid_list = list(dict.fromkeys(concept_specific[-2:] + all_seen[-3:]))[-3:]

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        temp = min(0.80, 0.45 + 0.175 * (attempt - 1))

        attempt_note = _RETRY_NOTES[min(attempt - 1, len(_RETRY_NOTES) - 1)]

        if prompt_type == "tf" and seen_tf_by_concept is not None and tf_concept_lock is not None:
            balance = _tf_balance_note(topic, seen_tf_by_concept, tf_concept_lock)
            if balance:
                attempt_note = f"{attempt_note} {balance}".strip() if attempt_note else balance

        if prompt_type == "open" and seen_open_combos is not None and open_combos_lock is not None:
            diversity = _open_diversity_note(
                topic, bloom, "",
                seen_open_combos, seen_open_verbs or {},
                open_combos_lock
            )
            if diversity:
                attempt_note = f"{attempt_note} {diversity}".strip() if attempt_note else diversity

        # Fix-52 (v4.12): pre-generation opener hint for MCQ
        if prompt_type == "mcq" and seen_mcq_openers is not None and mcq_opener_lock is not None:
            opener_hint = _mcq_opener_hint(topic, seen_mcq_openers, mcq_opener_lock)
            if opener_hint:
                attempt_note = f"{attempt_note} {opener_hint}".strip() if attempt_note else opener_hint

        prompt = build_training_prompt(
            instruction, prompt_type, bloom, topic, context,
            attempt_note=attempt_note,
            avoid_questions=avoid_list,
            attempt=attempt,
        )

        generated = ask_model(prompt, max_tokens=max_tok, temperature=temp, num_ctx=ctx_size)
        if generated is None:
            logger.info("Single gen failed record=%d attempt=%d", record_idx, attempt)
            time.sleep(0.05 * attempt)
            continue

        generated   = _normalize_output_keys(generated)
        candidate_q = normalize_generated_question(generated, display_type, topic, bloom)
        qtext       = (candidate_q.get("question") or "").strip()
        if not qtext:
            time.sleep(0.05 * attempt)
            continue

        # Fix-44/49: reject term-definition questions
        if _is_term_definition_question(qtext):
            concept_key = concept_lc
            bloom_lower = bloom.lower()
            allowed = bloom_lower in ("remembering", "understanding")
            if allowed and seen_term_defs is not None and term_def_lock is not None:
                with term_def_lock:
                    count = seen_term_defs.get(concept_key, 0)
                    if count >= 1:
                        allowed = False
                    else:
                        seen_term_defs[concept_key] = count + 1
            if not allowed:
                logger.info("Rejected term-def question record=%d bloom=%s q=%r",
                            record_idx, bloom, qtext[:70])
                avoid_list.append(qtext[:25])
                avoid_list = avoid_list[-3:]
                time.sleep(0.05 * attempt); continue

        # Fix-45: reject "Which statement best..."
        if _is_which_statement_best(qtext):
            logger.info("Rejected 'which statement best' record=%d q=%r",
                        record_idx, qtext[:70])
            avoid_list.append(qtext[:25])
            avoid_list = avoid_list[-3:]
            time.sleep(0.05 * attempt); continue

        # Fix-51 (v4.12): reject MCQ negation in question stem
        if display_type == "mcq" and _is_mcq_negation_question(qtext):
            logger.info("MCQ rejected: negation in stem record=%d q=%r",
                        record_idx, qtext[:70])
            avoid_list.append(qtext[:25])
            avoid_list = avoid_list[-3:]
            time.sleep(0.05 * attempt); continue

        # Fix-52 (v4.12): MCQ opener overuse check
        if (display_type == "mcq" and
                seen_mcq_openers is not None and mcq_opener_lock is not None):
            if _is_mcq_opener_overused(candidate_q, seen_mcq_openers, mcq_opener_lock):
                logger.info("MCQ rejected: overused opener record=%d q=%r",
                            record_idx, qtext[:70])
                avoid_list.append(qtext[:25])
                avoid_list = avoid_list[-3:]
                time.sleep(0.05 * attempt); continue

        # Fix-25: reject lazy bloom opener
        if _is_lazy_bloom_opener(qtext):
            logger.info("Rejected lazy bloom opener record=%d q=%r", record_idx, qtext[:70])
            avoid_list.append(qtext[:25])
            avoid_list = avoid_list[-3:]
            time.sleep(0.05 * attempt); continue

        # Fix-37: MCQ Bloom-level question-starter check
        if display_type == "mcq" and not _is_valid_mcq_bloom_pattern(qtext, bloom):
            logger.info(
                "MCQ rejected: Bloom starter mismatch level=%s record=%d q=%r",
                bloom, record_idx, qtext[:70]
            )
            avoid_list.append(qtext[:25])
            avoid_list = avoid_list[-3:]
            time.sleep(0.05 * attempt); continue

        # TF fast-path validators
        if display_type == "truefalse":
            if not _is_valid_tf(candidate_q):
                time.sleep(0.05 * attempt); continue

        # Fix-48: open-ended verb diversity check
        if display_type == "open_ended" and seen_open_combos is not None and open_combos_lock is not None:
            verb     = _extract_open_starter_verb(qtext)
            verb_key = f"{_open_bloom_concept_key(topic, bloom)}::verb"
            if verb and seen_open_verbs is not None:
                with open_combos_lock:
                    used_verbs = seen_open_verbs.get(verb_key, set())
                    if verb in used_verbs:
                        logger.info(
                            "Open-ended rejected: verb '%s' reused for concept=%r bloom=%s",
                            verb, topic, bloom
                        )
                        avoid_list.append(qtext[:25])
                        avoid_list = avoid_list[-3:]
                        time.sleep(0.05 * attempt); continue

        # Fix-53 (v4.12): MCQ sub-topic saturation check
        if (display_type == "mcq" and
                seen_mcq_by_concept is not None and mcq_concept_lock is not None):
            if _is_mcq_subtopic_saturated(candidate_q, seen_mcq_by_concept, mcq_concept_lock):
                avoid_list.append(_question_stem(candidate_q)[:25])
                avoid_list = avoid_list[-3:]
                time.sleep(0.05 * attempt); continue

        # Global fingerprint dedup
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
            avoid_list.append(_question_stem(candidate_q)[:25])
            avoid_list = avoid_list[-3:]
            time.sleep(0.05 * attempt); continue

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
                    logger.info("MCQ rejected: same-answer near-duplicate record=%d",
                                record_idx)
                    avoid_list.append(_question_stem(candidate_q)[:25])
                    avoid_list = avoid_list[-3:]
                    time.sleep(0.05 * attempt); continue

        # TF semantic dedup
        if display_type == "truefalse":
            if seen_tf_by_concept is not None and tf_concept_lock is not None:
                if _is_tf_semantic_duplicate(candidate_q, seen_tf_by_concept, tf_concept_lock):
                    avoid_list.append(_question_stem(candidate_q)[:25])
                    avoid_list = avoid_list[-3:]
                    time.sleep(0.05 * attempt); continue

        # Full answer/choice validity (most expensive — last)
        if not _is_valid_answer(candidate_q, display_type):
            logger.info("Invalid answer record=%d attempt=%d ans=%r",
                        record_idx, attempt, candidate_q.get("answer", ""))
            avoid_list.append(qtext[:25])
            avoid_list = avoid_list[-3:]
            time.sleep(0.05 * attempt); continue

        # ── All checks passed — register state ──────────────────────────
        if display_type == "truefalse" and seen_tf_by_concept is not None:
            _register_tf_question(candidate_q, seen_tf_by_concept, tf_concept_lock)

        if display_type == "open_ended" and seen_open_combos is not None:
            _register_open_question(
                topic, bloom, qtext,
                seen_open_combos, seen_open_verbs or {},
                open_combos_lock
            )

        if display_type == "mcq":
            if seen_mcq_by_concept is not None and mcq_concept_lock is not None:
                _register_mcq_subtopic(candidate_q, seen_mcq_by_concept, mcq_concept_lock)
            if seen_mcq_openers is not None and mcq_opener_lock is not None:
                _register_mcq_opener(candidate_q, seen_mcq_openers, mcq_opener_lock)

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
            text_hash = hashlib.md5(full_text[:4096].encode("utf-8", errors="ignore")).hexdigest()[:8]
            topic_key = f"{topic}::{text_hash}"
            base_idx  = _find_best_chunk_idx(chunks, topic)
            if topic_key not in topic_slot_counter:
                idxs = list(range(len(chunks)))
                if base_idx in idxs: idxs.remove(base_idx)
                idxs.insert(0, base_idx)
                tail = idxs[1:]
                random.shuffle(tail)
                idxs[1:] = tail
                topic_slot_counter[topic_key] = idxs
            idxs      = topic_slot_counter[topic_key]
            chunk_idx = idxs.pop(0)
            if not idxs: topic_slot_counter.pop(topic_key, None)
            context = chunks[chunk_idx]
            logger.info("record=%d topic=%r bloom=%s -> chunk %d/%d",
                        i + 1, topic, bloom, chunk_idx + 1, len(chunks))
        else:
            logger.warning("record=%d topic=%r has NO learning material.", i + 1, topic)
        slots.append({"record_idx": i, "topic": topic, "bloom": bloom,
                      "prompt_type": prompt_type, "display_type": display_type,
                      "context": context, "record": rec})

    # ── Shared dedup / tracking state ───────────────────────────────────
    _fp_lock        = threading.Lock()
    _answer_fp_lock = threading.Lock()
    _stems_lock     = threading.Lock()
    seen_fps:        set = set()
    seen_answer_fps: set = set()
    seen_stems: deque = deque(maxlen=16)

    _tf_concept_lock: threading.Lock = threading.Lock()
    seen_tf_by_concept: Dict[str, List[Tuple[frozenset, str]]] = {}

    _open_combos_lock: threading.Lock = threading.Lock()
    seen_open_combos: Set[str] = set()
    seen_open_verbs:  Dict[str, Set[str]] = {}

    _term_def_lock: threading.Lock = threading.Lock()
    seen_term_defs: Dict[str, int] = {}

    # Fix-52/53 (v4.12): MCQ-specific trackers
    _mcq_concept_lock: threading.Lock = threading.Lock()
    seen_mcq_by_concept: Dict[str, List[frozenset]] = {}

    _mcq_opener_lock: threading.Lock = threading.Lock()
    seen_mcq_openers: Dict[str, Dict[str, int]] = {}

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
            MAX_TOKENS_SINGLE.get(slot["prompt_type"], 115),
            seen_fps, slot["record_idx"] + 1,
            seen_questions=stems_snapshot,
            fp_lock=_fp_lock,
            seen_answer_fps=seen_answer_fps,
            answer_fp_lock=_answer_fp_lock,
            seen_tf_by_concept=seen_tf_by_concept,
            tf_concept_lock=_tf_concept_lock,
            seen_open_combos=seen_open_combos,
            open_combos_lock=_open_combos_lock,
            seen_open_verbs=seen_open_verbs,
            seen_term_defs=seen_term_defs,
            term_def_lock=_term_def_lock,
            seen_mcq_by_concept=seen_mcq_by_concept,
            mcq_concept_lock=_mcq_concept_lock,
            seen_mcq_openers=seen_mcq_openers,
            mcq_opener_lock=_mcq_opener_lock,
        )

        with _gen_progress_lock:
            _gen_progress["current"] += 1

        if q is not None:
            stem = _question_stem(q)
            if stem:
                with _stems_lock:
                    seen_stems.append(stem)

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
            # ── Fix-49 (v4.12): validate dataset fallback before using ──
            rec      = slot["record"]
            fallback = rec.get("output") if isinstance(rec, dict) else None
            used_fallback = False
            if fallback and isinstance(fallback, dict):
                normalized_fb = normalize_generated_question(
                    fallback, slot["display_type"], slot["topic"], slot["bloom"])
                if _is_valid_fallback_question(normalized_fb, slot["display_type"]):
                    out_questions.append(normalized_fb)
                    used_fallback = True
                else:
                    logger.info(
                        "Fallback rejected for record=%d concept=%r — using placeholder.",
                        slot["record_idx"] + 1, slot["topic"]
                    )
            if not used_fallback:
                out_questions.append({
                    "type":    slot["display_type"],
                    "concept": slot["topic"],
                    "bloom":   slot["bloom"],
                    "question": f"[GENERATION FAILED] Review this item — {slot['topic']}",
                    "choices": (["(Generation failed)"] * 4 if slot["display_type"] == "mcq" else []),
                    "answer": "",
                    "answer_text": "Generation failed. Please delete or replace.",
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

def get_model_cache_stats():
    cache_files = sum(1 for f in os.listdir(MODEL_CACHE_DIR) if f.endswith(".json"))
    return {
        "cache_hits":       CACHE_HITS,
        "cache_misses":     CACHE_MISSES,
        "mem_cache_size":   len(_mem_cache._cache),
        "disk_cache_files": cache_files,
    }

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
app = FastAPI(title="AutoTOS AI Service", version="4.12-ollama")
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
            "ollama_ready":       ready,
            "ollama_model":       OLLAMA_MODEL,
            "chunk_size":         CHUNK_SIZE,
            "generation_workers": GENERATION_WORKERS,
            "version":            "4.12"}

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