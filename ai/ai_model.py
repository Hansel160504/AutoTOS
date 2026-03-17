# ai_model.py
# AutoTOS AI module — prompt format matched exactly to training data
#
# FIXES APPLIED (v2.15 + perf tweaks):
#  - smaller chunks, shuffled non-repeating chunk queue
#  - model-aware prompt cache key
#  - lower token budgets for faster generation
#  - improved avoid block and longer recent-stems history
# ============================================================

import re
import json
import fitz
import base64
from llama_cpp import Llama
from docx import Document
from pptx import Presentation
from io import BytesIO
import time
from typing import List, Dict, Any, Optional
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
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/autotos-q4_k_m.gguf")

N_THREADS  = int(os.environ.get("N_THREADS", 6))    # Ryzen 5600G: 6 cores / 12 threads — use all of them
N_BATCH    = int(os.environ.get("N_BATCH", 512))
N_CTX      = 1280  # ~110 instr + ~17 spec + ~375 ctx + ~160 out = ~662; 1280 is safe
BATCH_SIZE = 1     # CPU-only: no parallel benefit from batching

MAX_RETURN    = 50_000  # full lecture notes (was 8 000)
CHUNK_SIZE    = 800     # chars per context chunk (~65-90 tokens). Smaller → more chunks → more variety
CHUNK_OVERLAP = 30      # overlap so sentences not split at boundaries (reduced from 50)

BASE_DIR        = os.path.dirname(__file__)
CACHE_DIR       = os.path.join(BASE_DIR, ".extracted_cache")
MODEL_CACHE_DIR = os.path.join(BASE_DIR, ".model_cache")
os.makedirs(CACHE_DIR,       exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

logger.info("LLM config: n_ctx=%d n_threads=%d n_batch=%d chunk_size=%d max_return=%d",
            N_CTX, N_THREADS, N_BATCH, CHUNK_SIZE, MAX_RETURN)

# =====================================================
# TOKEN BUDGETS — tuned for speed without losing quality
# =====================================================
MAX_TOKENS_SINGLE: Dict[str, int] = {
    "mcq":  200,
    "tf":   140,
    "open": 160, # open-ended single specific answer
}

# =====================================================
# BLOOM KEYWORDS
# =====================================================
BLOOM_KEYWORDS: Dict[str, List[str]] = {
    "Knowledge":  ["Define", "Identify", "Describe", "Recognize", "Explain", "Recite", "Illustrate"],
    "Understand": ["Summarize", "Interpret", "Classify", "Compare", "Contrast", "Infer", "Paraphrase"],
    "Apply":      ["Solve", "Use", "Complete", "Relate", "Transfer", "Articulate", "Discover"],
    "Analyze":    ["Contrast", "Connect", "Devise", "Correlate", "Distill", "Conclude", "Categorize"],
    "Evaluate":   ["Criticize", "Judge", "Defend", "Appraise", "Prioritize", "Reframe", "Grade"],
    "Create":     ["Design", "Develop", "Modify", "Invent", "Write", "Rewrite", "Collaborate"],
}

def get_bloom_cue_line(bloom: str) -> str:
    cues = BLOOM_KEYWORDS.get(bloom, [])
    if not cues:
        return ""
    return (
        "- Cognitive cues (embody this level — do NOT open the question with these words): "
        + ", ".join(cues)
    )

# =====================================================
# INSTRUCTIONS (must match training exactly)
# =====================================================
AUTOTOS_INSTRUCTION = (
    "You are AutoTOS. Generate one exam question aligned with the Bloom\u2019s level and concept. "
    "Output ONLY this JSON:\n"
    '{"type":"mcq|truefalse|open_ended","concept":"...","bloom":"...","question":"...",'
    '"choices":["A text","B text","C text","D text"],"answer":"A|B|C|D|true|false",'
    '"answer_text":"Why the answer is correct (1-2 sentences)."}\n'
    "choices field: mcq only. answer_text field: mcq and truefalse only. "
    "Answer based ONLY on the provided context."
)

AUTOTOS_INSTRUCTION_OPEN = (
    "You are AutoTOS. Generate one exam question aligned with the Bloom\u2019s level and concept. "
    "Output ONLY this JSON:\n"
    '{"type":"open_ended","concept":"...","bloom":"...","question":"...","answer":"<complete specific answer>"}\n'
    "Do NOT include answer_text or choices. "
    "Answer must be a complete specific response. "
    "Answer based ONLY on the provided context."
)

# ── Type mappings ──
TYPE_MAP = {
    "mcq": "mcq", "truefalse": "tf", "true_false": "tf", "tf": "tf",
    "open_ended": "open", "open-ended": "open", "openended": "open", "open": "open",
}
OUT_TYPE_NORMALIZE = {
    "mcq": "mcq", "truefalse": "truefalse", "tf": "truefalse",
    "true_false": "truefalse", "open_ended": "open_ended",
    "open": "open_ended", "open-ended": "open_ended",
}

# ── Bloom mappings ──
BLOOM_MAP_SINGLE = {
    "knowledge": "Knowledge", "understand": "Understand", "apply": "Apply",
    "analyze": "Analyze", "analyse": "Analyze", "evaluate": "Evaluate",
    "create": "Create", "remembering": "Knowledge", "understanding": "Understand",
    "applying": "Apply", "analyzing": "Analyze", "evaluating": "Evaluate",
    "creating": "Create",
}
BLOOM_CYCLE = {
    "remembering": ["Knowledge", "Understand"],
    "applying":    ["Apply",     "Analyze"],
    "creating":    ["Evaluate",  "Create"],
}

def normalize_bloom(bloom: str, slot_index: int = 0) -> str:
    key = (bloom or "").strip().lower()
    cycle = BLOOM_CYCLE.get(key)
    if cycle:
        return cycle[slot_index % 2]
    return BLOOM_MAP_SINGLE.get(key, bloom or "Knowledge")

def normalize_type(qtype: str) -> str:
    return TYPE_MAP.get((qtype or "").strip().lower(), "mcq")

def normalize_out_type(raw_type: str) -> str:
    return OUT_TYPE_NORMALIZE.get((raw_type or "").strip().lower(), raw_type or "mcq")

# =====================================================
# MODEL LOAD
# =====================================================
llm = None
try:
    logger.info("Loading GGUF model from %s ...", MODEL_PATH)
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_batch=N_BATCH,
        n_gpu_layers=0,
        verbose=False,
    )
    logger.info("Model ready!")
except Exception as e:
    logger.exception("Failed to load GGUF model: %s", e)
    llm = None

# =====================================================
# DISK CACHE
# =====================================================
def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def _cache_path_for_hash(h: str) -> str:
    return os.path.join(CACHE_DIR, f"{h}.txt")

def _prompt_hash_key(prompt: str, max_tokens: int, temperature: float) -> str:
    # include model path so caches are model-specific (prevents reusing outputs across quantized model changes)
    model_id = MODEL_PATH or ""
    key = f"{model_id}|{max_tokens}:{temperature:.4f}:{prompt}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def _model_cache_path(key: str) -> str:
    return os.path.join(MODEL_CACHE_DIR, f"{key}.json")

def read_model_cache(key: str) -> Optional[Any]:
    p = _model_cache_path(key)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
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
# UTILITIES
# =====================================================
def clean_text(txt) -> str:
    """Normalize whitespace for OUTPUT fields only. NEVER use on prompts."""
    if txt is None: return ""
    if not isinstance(txt, str):
        try: txt = str(txt)
        except Exception: return ""
    return re.sub(r"\s+", " ", txt).strip()

def sanitize_prompt(prompt: str) -> str:
    """Strip leading/trailing whitespace only. Preserves internal \\n."""
    return (prompt or "").strip()

_AWKWARD_PHRASING_RULES = [
    (re.compile(r'^Which of the following best describes which\b', re.IGNORECASE), "Which"),
    (re.compile(r'^Which of the following best describes what\b',  re.IGNORECASE), "What"),
    (re.compile(r'^Which best describes what\b',                   re.IGNORECASE), "What"),
    (re.compile(r'^Which best describes which\b',                  re.IGNORECASE), "Which"),
]

def _clean_question_phrasing(text: str) -> str:
    if not text: return text
    for pattern, replacement in _AWKWARD_PHRASING_RULES:
        if pattern.match(text):
            text = pattern.sub(replacement + " ", text, count=1).strip()
            if text: text = text[0].upper() + text[1:]
            break
    return text

_TF_QUESTION_PREFIX_RE = re.compile(
    r"^(true\s+or\s+false\s*[:\-]\s*)", re.IGNORECASE
)

def strip_question_prefix(text: str, is_open_ended: bool = False) -> str:
    text = (text or "").strip()
    if not text or is_open_ended:
        return _clean_question_phrasing(text)
    text = _TF_QUESTION_PREFIX_RE.sub("", text).strip()
    if text:
        text = text[0].upper() + text[1:]
    return _clean_question_phrasing(text)


_ANSWER_TEXT_MAX_CHARS = 160  # keep as a guard — first sentence will be enforced

def _truncate_answer_text(text: str) -> str:
    """
    Return exactly one, well-formed sentence as the explanation.
    - Strips common 'Answer:' prefixes.
    - Picks the first full sentence (split on [.!?] followed by whitespace).
    - Ensures trailing punctuation, and clamps length to _ANSWER_TEXT_MAX_CHARS
      while avoiding cutting mid-word where reasonable.
    """
    if not text:
        return ""

    # Normalize whitespace and strip leading 'Answer:' style markers
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r'^(answer\s*[:\-]\s*)', '', t, flags=re.IGNORECASE).strip()

    # Split into sentences using punctuation as boundary
    sentences = re.split(r'(?<=[.!?])\s+', t)
    # find first non-empty sentence-like fragment
    first = ""
    for s in sentences:
        if s and s.strip():
            first = s.strip()
            break
    if not first:
        # fallback: take the start of the text
        first = t[:_ANSWER_TEXT_MAX_CHARS].strip()

    # Ensure it ends with punctuation
    if first and first[-1] not in ".!?":
        first = first + "."

    # If it's short enough, return it
    if len(first) <= _ANSWER_TEXT_MAX_CHARS:
        return first

    # Otherwise truncate gently at a word boundary but always end with a period.
    truncated = first[:_ANSWER_TEXT_MAX_CHARS]
    last_space = truncated.rfind(" ")
    if last_space > int(_ANSWER_TEXT_MAX_CHARS * 0.6):
        truncated = truncated[:last_space]
    truncated = truncated.rstrip('.,;:') + "."
    return truncated

# =====================================================
# CHUNKING
# =====================================================
_chunk_cache: Dict[str, List[str]] = {}

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    text   = text.strip()
    chunks = []
    start  = 0
    step   = max(1, chunk_size - overlap)

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(
                text.rfind(". ", start, end),
                text.rfind("! ", start, end),
                text.rfind("? ", start, end),
            )
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks

def _find_best_chunk_idx(chunks: List[str], topic: str) -> int:
    if not chunks or not topic:
        return 0
    topic_lower = topic.lower()
    for i, chunk in enumerate(chunks):
        if topic_lower in chunk.lower():
            return i
    for word in topic_lower.split():
        if len(word) > 3:
            for i, chunk in enumerate(chunks):
                if word in chunk.lower():
                    return i
    return 0

def get_chunks_for_text(full_text: str) -> List[str]:
    key = hashlib.md5(full_text[:256].encode("utf-8", errors="ignore")).hexdigest()
    if key not in _chunk_cache:
        _chunk_cache[key] = chunk_text(full_text)
        logger.info("Chunked document: %d chars → %d chunks", len(full_text), len(_chunk_cache[key]))
    return _chunk_cache[key]

# =====================================================
# FILE EXTRACTION (unchanged)
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
            d    = Document(BytesIO(file_bytes))
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
    h          = _sha256_bytes(b)
    cache_file = _cache_path_for_hash(h)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as rf:
                return clean_text(rf.read())[:max_chars]
        except Exception: pass
    ext      = (os.path.splitext(path)[1] or "").lower()
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
# PROMPT BUILDERS (with improved avoid block)
# =====================================================
_TF_HINT = "[Declarative factual statement only — do NOT ask a question, do NOT give instructions.]"

_RETRY_NOTES = [
    "",
    "Different angle — ask about a specific component or detail.\n",
    "Ask about a consequence or real-world application.\n",
    "Use a scenario or example.\n",
]

def _build_avoid_block(seen_questions: List[str]) -> str:
    """Compact avoid block — include up to 6 recent stems and a short diversity hint."""
    if not seen_questions:
        return ""
    recent = seen_questions[-6:]
    items  = "; ".join(f'"{q[:50]}"'  for q in recent)
    return f"\n[Avoid repeating: {items}]\n[Focus on a DIFFERENT ASPECT of the topic; do NOT restate definitions.]\n"

def build_training_prompt(instruction: str, qtype: str, bloom: str,
                           concept: str, context: str,
                           extra_note: str = "",
                           attempt_note: str = "",
                           avoid_questions: Optional[List[str]] = None) -> str:
    avoid_block = _build_avoid_block(avoid_questions or [])
    ctx_block   = f"{context}\n{extra_note}{avoid_block}" if (extra_note or avoid_block) else context
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Target Specification:\n"
        f"- Question Type: {qtype}\n"
        f"- Bloom's Level: {bloom}\n"
        f"- Concept: {concept}\n\n"
        "### Context (Source Material):\n"
        f"{ctx_block}\n\n"
        "### Response:\n"
    )

# =====================================================
# JSON EXTRACTORS
# =====================================================
# ===== Replace _extract_first_json and _try_parse_json with improved versions =====
def _extract_first_json(text: str) -> Optional[str]:
    """
    Extract first complete JSON object { ... } using brace counter.
    If the model output was truncated (missing trailing brace), return the
    longest substring starting at the first '{' (so repair can be attempted).
    """
    if not text:
        return None
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    last_close_idx = -1
    for i, c in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if c == '\\' and in_string:
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            last_close_idx = i
            if depth == 0:
                # found a complete JSON object
                return text[start:i + 1]
    # no balanced closing brace found — return the fragment from '{' to end for repair attempts
    return text[start:]  # may be truncated; repair function may fix it


def _try_parse_json(json_str: str) -> Optional[Any]:
    """
    Try to parse JSON. If direct parsing fails, attempt minor cleanups:
      - remove trailing commas before } or ]
      - try to balance braces by appending closing braces (heuristic, up to 6)
    Returns the parsed object or None.
    """
    if not json_str or not isinstance(json_str, str):
        return None

    # quick attempt
    try:
        return json.loads(json_str)
    except Exception:
        pass

    # Basic cleanup: remove trailing commas before } or ]
    cleaned = re.sub(r',\s*([}\]])', r'\1', json_str)

    # Try to close unbalanced quotes by ensuring even number of quotes
    if cleaned.count('"') % 2 == 1:
        # attempt to append a closing quote
        cleaned = cleaned + '"'

    # Try incremental brace balancing: append '}' until braces balance or up to N times.
    open_braces = cleaned.count('{') - cleaned.count('}')
    if open_braces > 0 and open_braces <= 6:
        cleaned_candidate = cleaned + ('}' * open_braces)
        try:
            return json.loads(cleaned_candidate)
        except Exception:
            pass

    # Last attempt: try simple json.loads on the cleaned string
    try:
        return json.loads(cleaned)
    except Exception:
        return None
# =====================================================
# MODEL CALLER
# =====================================================
# ===== Replace ask_model with new behavior that attempts repair + retry =====
def ask_model(prompt: str, max_tokens: int = 160,
              temperature: float = 0.45) -> Optional[dict]:
    global CACHE_HITS, CACHE_MISSES
    if llm is None:
        logger.error("LLM not initialized.")
        return None

    prompt = sanitize_prompt(prompt)
    cache_key = _prompt_hash_key(prompt, max_tokens, temperature)
    cached = read_model_cache(cache_key)
    if cached is not None:
        CACHE_HITS += 1
        return cached if isinstance(cached, dict) else None
    CACHE_MISSES += 1

    def _call_model(max_tokens_call, temp_call):
        try:
            start = time.time()
            resp = llm.create_completion(
                prompt=prompt, max_tokens=max_tokens_call,
                temperature=temp_call, top_p=0.95,
                stop=["### Instruction:", "### Target", "\n###", "<think>", "</s>"],
                echo=False,
            )
            raw = ""
            if isinstance(resp, dict) and "choices" in resp and resp["choices"]:
                raw = resp["choices"][0].get("text", "") or ""
            elif isinstance(resp, str):
                raw = resp
            duration = time.time() - start
            logger.info("ask_model duration=%.2fs raw_len=%d max_tok=%d temp=%.2f",
                        duration, len(raw), max_tokens_call, temp_call)
            return raw
        except Exception as e:
            logger.exception("LLM call failed: %s", e)
            return ""

    raw = _call_model(max_tokens, temperature)
    if not raw or not raw.strip():
        logger.warning("Empty model output")
        return None

    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
    if not raw:
        return None

    json_str = _extract_first_json(raw)
    parsed = None
    if json_str:
        parsed = _try_parse_json(json_str)
    # If not parsed, attempt heuristical repair/parsing on the fragment
    if parsed is None and json_str:
        logger.warning("No JSON parse on first attempt; trying repair heuristics.")
        parsed = _try_parse_json(json_str)  # _try_parse_json already tries cleaning
        if parsed:
            logger.info("Repaired truncated JSON successfully.")
    # If still None, retry the model once with more tokens and lower temperature
    if parsed is None:
        logger.info("Retrying model with larger token budget to avoid truncation.")
        retry_tokens = min(max_tokens + 80, 1024)  # small bump but bounded
        raw2 = _call_model(retry_tokens, max(0.0, temperature - 0.15))
        if raw2 and raw2.strip():
            raw2 = re.sub(r"<think>[\s\S]*?</think>", "", raw2).strip()
            json_str2 = _extract_first_json(raw2)
            if json_str2:
                parsed = _try_parse_json(json_str2)
                if parsed:
                    logger.info("Parsed JSON from retry call.")
                    write_model_cache(cache_key, parsed)
                    return parsed
        # If retry failed, log the first 1000 chars for debugging
        logger.warning("Retry failed. Raw output sample: %s", raw[:1000].replace("\n", " "))
        return None

    # parsed successfully
    write_model_cache(cache_key, parsed)
    return parsed

# Warm-up
try:
    if llm:
        logger.info("Warming up model...")
        _wp = build_training_prompt(
            AUTOTOS_INSTRUCTION, "mcq", "Knowledge", "test",
            "This is a warm-up context."
        )
        ask_model(_wp, max_tokens=20, temperature=0.0)
        logger.info("Warm-up done.")
except Exception:
    pass

# =====================================================
# CONTEXT FINDER
# =====================================================
def get_relevant_context(full_text: str, topic: str,
                          window_size: int = 600) -> str:
    if not full_text: return ""
    ft_lower    = full_text.lower()
    topic_lower = (topic or "").lower()
    idx = ft_lower.find(topic_lower) if topic_lower else -1
    if idx == -1 and topic_lower:
        for word in [w for w in topic_lower.split() if len(w) > 3]:
            idx = ft_lower.find(word)
            if idx != -1: break
    if idx == -1:
        return full_text[:window_size]
    return full_text[max(0, idx - 100): min(len(full_text), idx + window_size)]

# =====================================================
# NORMALIZATION (unchanged)
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
        if src in out and dst not in out:
            out[dst] = out[src]
    if "answer" not in out and "sample_answer" in out:
        out["answer"] = out["sample_answer"]
    if isinstance(out.get("answer"), bool):
        out["answer"] = "true" if out["answer"] else "false"
    return out

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
        "question": strip_question_prefix(clean_text(q.get("question") or q.get("prompt") or ""), is_open_ended=(display_type == "open_ended")),
        "choices":  [],
        "answer":   "",
    }

    if display_type != "open_ended":
        raw_answer_text = clean_text(q.get("answer_text") or q.get("explanation") or "")
        out["answer_text"] = _truncate_answer_text(raw_answer_text)

    choices_raw = q.get("choices") or q.get("options")
    if isinstance(choices_raw, dict):
        keys = sorted(choices_raw.keys(), key=lambda s: s.upper())
        out["choices"] = [clean_text(choices_raw[k]) for k in keys][:4]
    elif isinstance(choices_raw, list):
        out["choices"] = [clean_text(x) for x in choices_raw][:4]

    ans = q.get("answer", "")
    if isinstance(ans, bool):
        out["answer"] = "true" if ans else "false"
    elif isinstance(ans, (int, float)):
        idx = int(ans)
        out["answer"] = (out["choices"][idx]
                         if out["choices"] and 0 <= idx < len(out["choices"])
                         else str(ans))
    elif isinstance(ans, str):
        a = ans.strip()
        if display_type == "mcq":
            if re.fullmatch(r"^[A-Da-d]$", a) and out["choices"]:
                idx = ord(a.upper()) - ord("A")
                out["answer"] = (out["choices"][idx]
                                 if 0 <= idx < len(out["choices"]) else a)
            elif re.fullmatch(r"^[A-Da-d][).\s].*", a):
                m = re.match(r"^[A-Da-d][).\s]\s*(.+)$", a)
                out["answer"] = m.group(1).strip() if m else a
            else:
                out["answer"] = a
        elif display_type == "truefalse":
            a_lower = a.lower().rstrip(".")
            if a_lower in ("true", "false"):    out["answer"] = a_lower
            elif a_lower in ("1", "yes"):       out["answer"] = "true"
            elif a_lower in ("0", "no"):        out["answer"] = "false"
            else:                               out["answer"] = a
        else:
            out["answer"] = a
    else:
        out["answer"] = clean_text(str(ans or ""))

    return out

# =====================================================
# DUPLICATE FINGERPRINT (unchanged)
# =====================================================
_FP_STOPWORDS = re.compile(
    r'\b(a|an|the|and|or|of|in|on|to|for|is|are|was|were|by|at|be|its|it|'
    r'that|this|these|those|with|as|from|into|about|which|how|what|does|do)\b'
)

_FP_FILLER_RE = re.compile(
    r"^(what (is|are|does|do|was|were) (the )?(primary |main |key )?"
    r"(focus|purpose|function|role|goal|aim|definition|meaning|concept|"
    r"example|reason|impact|effect|difference|advantage|use|importance) of\s*|"
    r"which (of the following )?(best )?(describes?|defines?|explains?|is|are)\s*|"
    r"how (does?|do|is|are|can)\s*|"
    r"why (is|are|does|do)\s*)",
    re.IGNORECASE,
)

_FP_VERB_OPENER_RE = re.compile(
    r"^(define|explain|describe|summarize|identify|analyze|evaluate|compare|"
    r"contrast|apply|solve|create|design|develop|discuss|state|examine|"
    r"assess|illustrate|demonstrate|interpret|classify|infer|relate|conclude|"
    r"criticize|judge|defend|appraise|reframe|modify|invent|collaborate)\s+",
    re.IGNORECASE,
)

_FP_QUALIFIER_RE = re.compile(
    r"\b(primary|main|key|overall|general|core|basic|fundamental|"
    r"purpose|goal|aim|role|function|focus|use|importance|objective)\b",
    re.IGNORECASE,
)

def _question_fingerprint(q: dict) -> str:
    raw = (q.get("question") or "").lower().strip()
    raw = re.sub(r"[^\w\s]", " ", raw)
    raw = _FP_STOPWORDS.sub(" ", raw)
    raw = _FP_FILLER_RE.sub("", raw).strip()
    raw = _FP_VERB_OPENER_RE.sub("", raw).strip()
    raw = _FP_QUALIFIER_RE.sub(" ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    words = raw.split()
    words = [w[:-1] if w.endswith("s") and len(w) > 4 else w for w in words]
    raw   = " ".join(words)
    qtext   = raw[:45]
    concept = re.sub(r"\s+", "_", (q.get("concept") or "").lower().strip())
    return f"{concept}::{qtext}"

def _question_stem(q: dict) -> str:
    return (q.get("question") or "")[:80].strip()

# =====================================================
# ANSWER ↔ EXPLANATION CONSISTENCY CHECKER (unchanged)
# =====================================================
def _answer_matches_explanation(answer_letter: str, choices: list,
                                 answer_text: str) -> tuple:
    if not choices or not answer_letter or answer_letter not in "ABCD":
        return True, answer_letter
    if not answer_text:
        return True, answer_letter

    idx = ord(answer_letter.upper()) - ord("A")
    if idx < 0 or idx >= len(choices):
        return True, answer_letter

    ans_low = answer_text.lower()

    scores = []
    for i, choice in enumerate(choices):
        words = re.findall(r"\b\w{4,}\b", choice.lower())
        score = sum(1 for w in words if w in ans_low)
        scores.append((score, i))

    scores.sort(reverse=True)
    best_score, best_idx = scores[0]
    chosen_score = next(s for s, i in scores if i == idx)

    best_letter = chr(ord("A") + best_idx)

    if best_letter != answer_letter and (best_score - chosen_score) >= 2:
        return False, best_letter

    return True, answer_letter

# =====================================================
# ANSWER VALIDATOR (unchanged)
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

    if ans_low in _ANSWER_ALWAYS_BAD:
        return False

    if display_type == "mcq":
        choices = q.get("choices") or []
        if not answer:
            return False
        if re.fullmatch(r"[a-d]", ans_low) and not choices:
            return False
        resolved_letter = answer
        if re.fullmatch(r"[A-Da-d]", answer):
            resolved_letter = answer.upper()
        consistent, _ = _answer_matches_explanation(
            resolved_letter, choices, q.get("answer_text", "")
        )
        if not consistent:
            return False
        return True

    elif display_type == "truefalse":
        return ans_low in ("true", "false")

    elif display_type == "open_ended":
        if len(answer) < 15:
            return False
        if ans_low in _OPEN_PLACEHOLDERS:
            return False
        if re.match(r"^(model answer|answer\s*:)", ans_low):
            return False

    return True

# =====================================================
# TF VALIDATOR (unchanged)
# =====================================================
_TF_TASK_VERBS = re.compile(
    r'^(convert|calculate|compute|list|draw|design|write|find|determine|'
    r'show|give an example|describe how|explain how|create a?|propose|'
    r'evaluate|analyze|define|summarize|solve|identify|compare|'
    r'develop|construct|formulate|generate|'
    r'contrast|correlate|distill|conclude|categorize|'
    r'criticize|judge|defend|appraise|prioritize|reframe|grade|'
    r'modify|invent|rewrite|collaborate|'
    r'interpret|classify|infer|paraphrase|relate|transfer|articulate|discover|'
    r'connect|devise|describe|recognize|recite|illustrate|complete)\b',
    re.IGNORECASE,
)
_TF_WH_QUESTION = re.compile(
    r'^(what|which|how|why|who|where|when)\b', re.IGNORECASE,
)

def _is_valid_tf(q: dict) -> bool:
    answer = (q.get("answer") or "").strip().lower().rstrip(".")
    if answer not in ("true", "false"): return False
    question = (q.get("question") or "").strip()
    if not question: return False
    if _TF_TASK_VERBS.match(question) or _TF_WH_QUESTION.match(question): return False
    if re.search(r'(explanation|description|timeline|summary)\s*[.:]?\s*$', question, re.IGNORECASE): return False
    return True

# =====================================================
# SINGLE-QUESTION GENERATOR (unchanged)
# =====================================================
def _generate_single(topic: str, prompt_type: str, display_type: str,
                     bloom: str, context: str, max_tok: int,
                     seen_fps: set, record_idx: int,
                     seen_questions: Optional[List[str]] = None) -> Optional[dict]:
    instruction  = AUTOTOS_INSTRUCTION_OPEN if prompt_type == "open" else AUTOTOS_INSTRUCTION
    extra_note   = _TF_HINT if prompt_type == "tf" else ""
    MAX_ATTEMPTS = 4
    avoid_list   = list(seen_questions or [])

    for attempt in range(1, MAX_ATTEMPTS + 1):
        temp         = min(0.85, 0.45 + 0.15 * (attempt - 1))
        attempt_note = _RETRY_NOTES[min(attempt - 1, len(_RETRY_NOTES) - 1)]
        retry_bloom  = bloom
        if attempt > 2 and prompt_type == "tf":
            retry_bloom = "Knowledge" if attempt == 3 else "Understand"

        prompt = build_training_prompt(
            instruction, prompt_type, retry_bloom, topic,
            context, extra_note=extra_note,
            attempt_note=attempt_note,
            avoid_questions=avoid_list,
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
        if display_type == "truefalse" and not _is_valid_tf(candidate_q):
            time.sleep(0.1 * attempt); continue
        if not _is_valid_answer(candidate_q, display_type):
            logger.info("Invalid answer record=%d attempt=%d ans=%r", record_idx, attempt, candidate_q.get("answer",""))
            avoid_list.append(qtext[:60])
            time.sleep(0.1 * attempt); continue
        fp = _question_fingerprint(candidate_q)
        if fp in seen_fps:
            avoid_list.append(_question_stem(candidate_q))
            time.sleep(0.1 * attempt); continue

        seen_fps.add(fp)
        return candidate_q

    return None

# =====================================================
# MAIN GENERATOR (with shuffled chunk queue)
# =====================================================
def generate_from_records(records: List[dict],
                           max_items: Optional[int] = None) -> List[dict]:
    out_questions: List[dict] = []
    limit         = min(len(records), max_items) if max_items else len(records)
    seen_fps:  set       = set()
    seen_stems: List[str] = []   # rolling list of recent question stems for avoid injection
    topic_slot_counter: Dict[str, List[int]] = {}   # per-topic shuffled chunk index queue

    # ── Build slot metadata ──
    slots = []
    for i in range(limit):
        rec       = records[i]
        input_obj = rec.get("input", {}) if isinstance(rec, dict) else {}

        topic     = (input_obj.get("concept") or input_obj.get("topic")
                     or rec.get("instruction", "General")) or "General"
        raw_bloom = (input_obj.get("bloom") or
                     (rec.get("output", {}) or {}).get("bloom") or "Remembering")
        raw_type  = (input_obj.get("type") or
                     (rec.get("output", {}) or {}).get("type") or "mcq")

        prompt_type  = normalize_type(raw_type)
        display_type = normalize_out_type(
            {"mcq": "mcq", "tf": "truefalse", "open": "open_ended"}.get(
                prompt_type, prompt_type)
        )
        bloom     = normalize_bloom(raw_bloom, slot_index=i)

        candidate = (input_obj.get("context") or input_obj.get("learn_material") or
                     input_obj.get("file_path") or rec.get("file_path") or "")
        full_text = lesson_from_upload(candidate) if candidate else ""

        # ── Assign a unique chunk for this slot ──────────────────────────────
        context = ""
        if full_text:
            chunks    = get_chunks_for_text(full_text)
            text_hash = hashlib.md5(full_text[:256].encode("utf-8", errors="ignore")).hexdigest()[:8]
            topic_key = f"{topic}::{text_hash}"
            base_idx  = _find_best_chunk_idx(chunks, topic)

            # prepare a shuffled queue of indices (biased to keep base_idx first)
            if topic_key not in topic_slot_counter:
                idxs = list(range(len(chunks)))
                if base_idx in idxs:
                    idxs.remove(base_idx)
                idxs.insert(0, base_idx)
                random.shuffle(idxs[1:])  # keep base_idx first, shuffle the rest
                topic_slot_counter[topic_key] = idxs

            idxs = topic_slot_counter[topic_key]
            chunk_idx = idxs.pop(0)

            # recycle idx list when exhausted (so future runs recreate it)
            if not idxs:
                topic_slot_counter.pop(topic_key, None)

            context = chunks[chunk_idx]
            logger.info(
                "record=%d topic=%r bloom=%s → chunk %d/%d",
                i + 1, topic, bloom, chunk_idx + 1, len(chunks)
            )
        else:
            logger.warning(
                "record=%d topic=%r has NO learning material — "
                "question will be generic. Upload a file to improve variety.",
                i + 1, topic
            )

        slots.append({
            "record_idx":   i,
            "topic":        topic,
            "bloom":        bloom,
            "prompt_type":  prompt_type,
            "display_type": display_type,
            "context":      context,
            "record":       rec,
        })

    # ── Process each slot ──
    for slot in slots:
        q = _generate_single(
            slot["topic"], slot["prompt_type"], slot["display_type"],
            slot["bloom"], slot["context"],
            MAX_TOKENS_SINGLE.get(slot["prompt_type"], 160),
            seen_fps, slot["record_idx"] + 1,
            seen_questions=seen_stems,
        )

        if q is not None:
            out_questions.append(q)
            stem = _question_stem(q)
            if stem:
                seen_stems.append(stem)
                if len(seen_stems) > 8:
                    seen_stems.pop(0)
        else:
            rec      = slot["record"]
            fallback = rec.get("output") if isinstance(rec, dict) else None
            if fallback and isinstance(fallback, dict):
                out_questions.append(
                    normalize_generated_question(
                        fallback, slot["display_type"],
                        slot["topic"], slot["bloom"]
                    )
                )
            else:
                out_questions.append({
                    "type":        slot["display_type"],
                    "concept":     slot["topic"],
                    "bloom":       slot["bloom"],
                    "question":    f"[GENERATION FAILED] Review this item — {slot['topic']}",
                    "choices":     (["(Generation failed)"] * 4 if slot["display_type"] == "mcq" else []),
                    "answer":      "",
                    "answer_text": "Generation failed. Please delete or replace.",
                    "_generation_failed": True,
                })

    return out_questions

# =====================================================
# WRAPPER, MISC, API (unchanged)
# =====================================================
def generate_quiz_for_topics(records_or_topics,
                              max_items: Optional[int] = None,
                              test_labels: Optional[list] = None,
                              *args, **kwargs):
    try:
        quizzes = generate_from_records(records_or_topics, max_items)
    except Exception as e:
        logger.exception("generate_from_records error: %s", e)
        quizzes = []

    if isinstance(quizzes, dict) and "quizzes" in quizzes:
        quizzes = quizzes["quizzes"]
    elif not isinstance(quizzes, list):
        try: quizzes = list(quizzes)
        except Exception: quizzes = []

    if test_labels and isinstance(test_labels, (list, tuple)):
        for idx, item in enumerate(quizzes):
            if isinstance(item, dict):
                item["test_header"] = test_labels[idx] if idx < len(test_labels) else ""

    return {"quizzes": quizzes}

def get_model_cache_stats() -> Dict[str, int]:
    return {"cache_hits": CACHE_HITS, "cache_misses": CACHE_MISSES}

def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: records.append(json.loads(line))
            except Exception as e: logger.warning("Skipping bad line: %s", e)
    return records

def validate_dataset_records(records: List[dict], sample_limit: int = 3) -> Dict[str, Any]:
    report = {"total": len(records), "missing_instruction": 0,
              "missing_input": 0, "missing_output": 0, "examples": []}
    for rec in records:
        if not isinstance(rec, dict): continue
        if not rec.get("instruction"): report["missing_instruction"] += 1
        if not isinstance(rec.get("input"), dict): report["missing_input"] += 1
        if not isinstance(rec.get("output"), dict): report["missing_output"] += 1
        if len(report["examples"]) < sample_limit: report["examples"].append(rec)
    return report

app = FastAPI(title="AutoTOS AI Service", version="2.15")
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
    return {"status": "ok", "model_loaded": llm is not None,
            "chunk_size": CHUNK_SIZE, "max_return": MAX_RETURN}

@app.get("/cache_stats")
async def cache_stats():
    return get_model_cache_stats()

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
            lambda: generate_quiz_for_topics(
                req.records, max_items=req.max_items, test_labels=req.test_labels
            )
        )
        return resp
    except Exception as e:
        logger.exception("generate error: %s", e)
        raise HTTPException(status_code=500, detail="Generation failed")

@app.post("/generate_from_records")
async def generate_from_records_endpoint(payload: GenerateRequest):
    try:
        out = await run_in_threadpool(
            lambda: generate_from_records(payload.records, max_items=payload.max_items)
        )
        return {"quizzes": out}
    except Exception as e:
        logger.exception("generate_from_records error: %s", e)
        raise HTTPException(status_code=500, detail="Generation failed")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl",  "-j", required=False)
    parser.add_argument("--sample", "-n", type=int, default=None)
    parser.add_argument("--serve",  action="store_true")
    args = parser.parse_args()

    if args.jsonl and not args.serve:
        recs    = load_jsonl(args.jsonl)
        report  = validate_dataset_records(recs, sample_limit=2)
        logger.info("Validation: %s", json.dumps(report, indent=2, default=str))
        n       = args.sample or min(5, len(recs))
        results = generate_from_records(recs[:n], max_items=n)
        for r in results:
            print(json.dumps(r, indent=2, ensure_ascii=False))
    elif args.serve:
        import uvicorn
        uvicorn.run("ai_model:app", host="0.0.0.0", port=8000, log_level="info")
    else:
        logger.info("Usage: python ai_model.py --jsonl data.jsonl --sample 5")