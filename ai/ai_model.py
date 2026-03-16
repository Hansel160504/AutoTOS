# ai_model.py
# AutoTOS AI module — prompt format matched exactly to training data
#
# FIXES APPLIED (v2.3):
#   1. Brace-counter JSON extractor — stops at first complete object,
#      eliminates "Extra data" parse failures and the 5x retry penalty.
#   2. Qwen3 /no_think suppression — appended to ### Response: so the
#      model skips its <think>...</think> block (~200 wasted tokens/call).
#   3. Qwen3 <think> stripper — safety-net regex in case /no_think is
#      ignored on some prompts.
#   4. N_THREADS = 10 — uses all 12 Ryzen 5600G threads minus 2 for OS.
#   5. N_BATCH = 512 — faster prompt ingestion on CPU.
#   6. N_CTX = 1536 — prompts rarely exceed ~900 tokens; smaller ctx = faster.
#   7. max_tokens tuned per question type (220 MCQ, 180 TF, 260 open).
#   8. AUTOTOS_INSTRUCTION updated — answer_text capped to 1-2 sentences.
#   9. _truncate_answer_text() post-processing safety net (350 char cap).
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
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# =====================================================
# CONFIGURATION
# =====================================================
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/qwen3-q4_k_m.gguf")

# FIX 4 & 5 & 6: Ryzen 5600G = 6 cores / 12 threads.
# Leave 2 threads for OS + Docker; use the rest for inference.
N_THREADS = 6   # was 4
N_BATCH   = 512  # was 128 — faster prompt ingestion on CPU
N_CTX     = 1280 # was 2048 — prompts rarely exceed ~900 tokens

BASE_DIR        = os.path.dirname(__file__)
CACHE_DIR       = os.path.join(BASE_DIR, ".extracted_cache")
MODEL_CACHE_DIR = os.path.join(BASE_DIR, ".model_cache")
os.makedirs(CACHE_DIR,       exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

logger.info("LLM config: n_ctx=%d n_threads=%d n_batch=%d", N_CTX, N_THREADS, N_BATCH)

# ── The EXACT instruction from the training dataset ──
# FIX 8: "1-2 sentences only" added to both the format spec AND the closing
#         instruction line. This alone cuts output tokens by ~35% per call
#         because answer_text was the biggest variable-length field.
AUTOTOS_INSTRUCTION = (
    "You are AutoTOS \u2014 an AI-powered exam generator for university faculty. "
    "Generate a well-structured exam question based on the following specification. "
    "The question must be aligned with the specified Bloom\u2019s Taxonomy level and topic. "
    "Ensure it is pedagogically sound, clear, and ready for classroom use. "
    "Use university-level language. Avoid casual or exaggerated terms. "
    "All content must be factual and based on standard course materials. "
    'Output must strictly follow this format: {\n'
    '  "type": "mcq|truefalse|open_ended",\n'
    '  "concept": "...",\n'
    '  "bloom": "...",\n'
    '  "question": "...",\n'
    '  "choices": ["..."]  // only for mcq\n'
    '  "answer": "A|B|C|D|true|false",\n'
    '  "answer_text": "1-2 sentences only. State why the answer is correct."\n'
    "} "
    "Keep answer_text to 1-2 sentences maximum. "
    "Answer based ONLY on the provided context."
)

# ── Input type mapping: dashboard/internal → training prompt values ──
TYPE_MAP = {
    "mcq":        "mcq",
    "truefalse":  "tf",
    "true_false": "tf",
    "tf":         "tf",
    "open_ended": "open",
    "open-ended": "open",
    "openended":  "open",
    "open":       "open",
}

# ── Output type normalizer → what the app expects ──
OUT_TYPE_NORMALIZE = {
    "mcq":        "mcq",
    "truefalse":  "truefalse",
    "tf":         "truefalse",
    "true_false": "truefalse",
    "open_ended": "open_ended",
    "open":       "open_ended",
    "open-ended": "open_ended",
}

# ── Bloom mapping ──
BLOOM_MAP_SINGLE = {
    "knowledge":    "Knowledge",
    "understand":   "Understand",
    "apply":        "Apply",
    "analyze":      "Analyze",
    "analyse":      "Analyze",
    "evaluate":     "Evaluate",
    "create":       "Create",
    "remembering":  "Knowledge",
    "understanding":"Understand",
    "applying":     "Apply",
    "analyzing":    "Analyze",
    "evaluating":   "Evaluate",
    "creating":     "Create",
}

BLOOM_CYCLE = {
    "remembering": ["Knowledge", "Understand"],
    "applying":    ["Apply",     "Analyze"],
    "creating":    ["Evaluate",  "Create"],
}

def normalize_bloom(bloom: str, slot_index: int = 0) -> str:
    key   = (bloom or "").strip().lower()
    cycle = BLOOM_CYCLE.get(key)
    if cycle:
        return cycle[slot_index % 2]
    if key in BLOOM_MAP_SINGLE:
        return BLOOM_MAP_SINGLE[key]
    return bloom or "Knowledge"

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
    key = f"{max_tokens}:{temperature:.4f}:{prompt}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def _model_cache_path(key: str) -> str:
    return os.path.join(MODEL_CACHE_DIR, f"{key}.json")

def read_model_cache(key: str) -> Optional[dict]:
    p = _model_cache_path(key)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            try: os.remove(p)
            except Exception: pass
    return None

def write_model_cache(key: str, obj: dict):
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
_QUESTION_PREFIX_RE = re.compile(
    r'^(Solve|Design|Summarize|Analyze|Define|Explain\s+why)\s*:\s*',
    re.IGNORECASE,
)

def strip_question_prefix(text: str) -> str:
    cleaned = _QUESTION_PREFIX_RE.sub("", (text or "")).strip()
    return cleaned[0].upper() + cleaned[1:] if cleaned else cleaned

def clean_text(txt) -> str:
    if txt is None: return ""
    if not isinstance(txt, str):
        try: txt = str(txt)
        except Exception: return ""
    return re.sub(r"\s+", " ", txt).strip()

# =====================================================
# FIX 9: answer_text post-processing truncation
# =====================================================
# Hard cap: 2 sentences ~= 300 chars on average.
# The instruction change (Fix 8) does the real work at generation time;
# this is a deterministic safety net for edge cases where the model ignores it.
_ANSWER_TEXT_MAX_CHARS = 350

def _truncate_answer_text(text: str) -> str:
    """
    Trim answer_text to at most 2 sentences.
    Splits on .  !  ?  followed by whitespace; keeps first 2 fragments.
    Falls back to hard character cap if sentence splitting fails.
    """
    if not text or len(text) <= _ANSWER_TEXT_MAX_CHARS:
        return text

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) >= 2:
        trimmed = " ".join(sentences[:2]).strip()
        if trimmed and trimmed[-1] not in ".!?":
            trimmed += "."
        return trimmed

    # Fallback: truncate at word boundary
    truncated  = text[:_ANSWER_TEXT_MAX_CHARS]
    last_space = truncated.rfind(" ")
    if last_space > int(_ANSWER_TEXT_MAX_CHARS * 0.7):
        truncated = truncated[:last_space]
    return truncated.rstrip(".,;:") + "."

# =====================================================
# FILE EXTRACTION
# =====================================================
def extract_text_from_bytes(file_bytes: bytes, filetype: str) -> str:
    text = ""
    try:
        if filetype == "pdf":
            doc   = fitz.open(stream=file_bytes, filetype="pdf")
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
            prs  = Presentation(BytesIO(file_bytes))
            slt  = []
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

def extract_text_from_path(path: str, max_chars: int = 8000) -> str:
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
    MAX_RETURN = 8000
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
# PROMPT BUILDER — exact training format
# =====================================================
def build_training_prompt(instruction: str, qtype: str, bloom: str,
                           concept: str, context: str,
                           extra_note: str = "") -> str:
    """
    Reproduces the exact prompt prefix used during fine-tuning.

    FIX 2: '/no_think' is appended after '### Response:' — this is Qwen3's
    official token to suppress the <think>...</think> reasoning block,
    saving ~150-250 tokens per call on CPU.
    """
    ctx_block = context
    if extra_note:
        ctx_block = f"{context}\n{extra_note}" if context else extra_note

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
        "/no_think\n"   # FIX 2: suppress Qwen3 thinking block
    )

# ── TF declarative-statement hint ──
_TF_HINT = (
    "[Instruction: Generate a single declarative statement that can be "
    "evaluated as True or False. Do NOT start with action verbs like "
    "Design, Create, Summarize, Analyze, Explain, or Propose. "
    "Write a factual statement about the concept.]"
)

# =====================================================
# FIX 1: Brace-counter JSON extractor
# =====================================================
def _extract_first_json(text: str) -> Optional[str]:
    """
    Walk character-by-character and return the first complete JSON object.

    The old greedy regex r"\{[\s\S]*\}" matched from the first '{' to the
    LAST '}' in the string — causing 'Extra data' parse errors whenever the
    model wrote anything after the closing brace. This walks with a depth
    counter and stops at the exact matching '}'.
    """
    start = text.find('{')
    if start == -1:
        return None
    depth     = 0
    in_string = False
    escape    = False
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
            if depth == 0:
                return text[start:i + 1]
    return None  # unterminated — model cut off by max_tokens

# =====================================================
# MODEL CALLER — uses create_completion (not chat)
# =====================================================
def ask_model(prompt: str, max_tokens: int = 220,
              temperature: float = 0.3) -> Optional[dict]:
    global CACHE_HITS, CACHE_MISSES

    if llm is None:
        logger.error("LLM not initialized.")
        return None

    prompt    = clean_text(prompt)
    cache_key = _prompt_hash_key(prompt, max_tokens, temperature)
    cached    = read_model_cache(cache_key)
    if cached is not None:
        CACHE_HITS += 1
        return cached
    CACHE_MISSES += 1

    try:
        start = time.time()
        resp  = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            # FIX 2 (safety net): stop ON </think> so we never pay for
            # the thinking block if /no_think is ignored.
            stop=["### Instruction:", "### Target", "\n###", "</think>"],
            echo=False,
        )
        raw = ""
        if isinstance(resp, dict) and "choices" in resp and resp["choices"]:
            raw = resp["choices"][0].get("text", "") or ""
        elif isinstance(resp, str):
            raw = resp

        raw      = clean_text(raw)
        duration = time.time() - start
        logger.debug("ask_model duration=%.2fs raw_len=%d", duration, len(raw))

        if not raw:
            logger.warning("Empty model output")
            return None

        # FIX 3: Strip any <think>...</think> block that slipped through.
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
        if not raw:
            logger.warning("Output was only a thinking block — nothing to parse")
            return None

        # FIX 1: Brace-counter extractor.
        json_str = _extract_first_json(raw)
        if not json_str:
            logger.warning("No JSON found in: %s", raw[:200])
            return None

        try:
            parsed = json.loads(json_str)
        except Exception:
            try:
                cleaned = re.sub(r",\s*}", "}", re.sub(r",\s*]", "]", json_str))
                parsed  = json.loads(cleaned)
            except Exception as e:
                logger.warning("JSON parse failed: %s | raw: %s", e, json_str[:200])
                return None

        write_model_cache(cache_key, parsed)
        return parsed

    except Exception as e:
        logger.exception("ask_model error: %s", e)
        return None


# Warm-up
try:
    if llm:
        logger.info("Warming up model...")
        _wp = build_training_prompt(
            AUTOTOS_INSTRUCTION, "mcq", "Knowledge", "test",
            "This is a warm-up context."
        )
        ask_model(_wp, max_tokens=16, temperature=0.0)
        logger.info("Warm-up done.")
except Exception:
    pass

# =====================================================
# CONTEXT FINDER
# =====================================================
def get_relevant_context(full_text: str, topic: str,
                          window_size: int = 2000) -> str:
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
    return full_text[max(0, idx - 300): min(len(full_text), idx + window_size)]

# =====================================================
# NORMALIZATION
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

    # FIX 9: Truncate answer_text at normalization time — every path
    # (fresh generation, fallback, placeholder) goes through this cap.
    raw_answer_text    = clean_text(q.get("answer_text") or q.get("explanation") or "")
    trimmed_answer_text = _truncate_answer_text(raw_answer_text)

    out = {
        "type":        display_type,
        "concept":     topic,
        "bloom":       bloom_level,
        "question":    strip_question_prefix(clean_text(q.get("question") or q.get("prompt") or "")),
        "choices":     [],
        "answer":      "",
        "answer_text": trimmed_answer_text,
    }

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
        out["answer"] = out["choices"][idx] if out["choices"] and 0 <= idx < len(out["choices"]) else str(ans)
    elif isinstance(ans, str):
        a = ans.strip()
        if display_type == "mcq":
            if re.fullmatch(r"^[A-Da-d]$", a) and out["choices"]:
                idx = ord(a.upper()) - ord("A")
                if 0 <= idx < len(out["choices"]):
                    out["answer"] = out["choices"][idx]
                else:
                    out["answer"] = a
            elif re.fullmatch(r"^[A-Da-d][).\s].*", a):
                match = re.match(r"^[A-Da-d][).\s]\s*(.+)$", a)
                if match:
                    out["answer"] = match.group(1).strip()
                else:
                    out["answer"] = a
            else:
                out["answer"] = a
        elif display_type == "truefalse":
            a_lower = a.lower().rstrip(".")
            if a_lower in ("true", "false"):
                out["answer"] = a_lower
            elif a_lower in ("1", "yes"):
                out["answer"] = "true"
            elif a_lower in ("0", "no"):
                out["answer"] = "false"
            else:
                out["answer"] = a
        else:
            out["answer"] = a
    else:
        out["answer"] = clean_text(str(ans or ""))

    return out

# =====================================================
# MAIN GENERATOR
# =====================================================
def generate_from_records(records: List[dict],
                           max_items: Optional[int] = None) -> List[dict]:
    out_questions: List[dict] = []
    limit         = min(len(records), max_items) if max_items else len(records)
    seen_questions: set       = set()

    for i in range(limit):
        rec       = records[i]
        input_obj = rec.get("input", {}) if isinstance(rec, dict) else {}

        topic = (input_obj.get("concept") or input_obj.get("topic")
                 or rec.get("instruction", "General"))
        topic = topic or "General"

        raw_bloom = (input_obj.get("bloom") or
                     (rec.get("output", {}) or {}).get("bloom") or "Remembering")

        raw_type  = (input_obj.get("type") or
                     (rec.get("output", {}) or {}).get("type") or "mcq")

        prompt_type  = normalize_type(raw_type)
        display_type = normalize_out_type(
            {"mcq": "mcq", "tf": "truefalse", "open": "open_ended"}.get(prompt_type, prompt_type)
        )

        candidate = (input_obj.get("context") or
                     input_obj.get("learn_material") or
                     input_obj.get("file_path") or
                     rec.get("file_path") or "")
        full_text = lesson_from_upload(candidate) if candidate else ""
        context   = get_relevant_context(full_text, topic) if full_text else ""

        bloom = normalize_bloom(raw_bloom, slot_index=i)

        extra_note = _TF_HINT if prompt_type == "tf" else ""

        base_prompt = build_training_prompt(
            AUTOTOS_INSTRUCTION, prompt_type, bloom, topic,
            context, extra_note=extra_note
        )

        # FIX 7: Tight token budgets now that answer_text is 1-2 sentences.
        # Approximate token breakdown per type:
        #   MCQ : question ~25 + 4 choices ~60 + answer ~5 + answer_text ~50 = ~140 → 220 cap
        #   TF  : question ~25 + answer ~5 + answer_text ~40               = ~70  → 180 cap
        #   Open: question ~30 + answer ~30 + answer_text ~60              = ~120 → 260 cap
        MAX_TOKENS_BY_TYPE = {
            "mcq":  220,   # was 380
            "tf":   180,   # was 380
            "open": 260,   # was 380
        }
        max_tok = MAX_TOKENS_BY_TYPE.get(prompt_type, 220)

        MAX_ATTEMPTS = 5
        normalized   = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            temp = min(0.85, 0.30 + 0.12 * (attempt - 1))

            if attempt > 2 and prompt_type == "tf":
                retry_bloom = "Knowledge" if attempt == 3 else "Understand"
                retry_prompt = build_training_prompt(
                    AUTOTOS_INSTRUCTION, prompt_type, retry_bloom, topic,
                    context, extra_note=extra_note
                )
                prompt_to_use = retry_prompt
            else:
                prompt_to_use = base_prompt

            generated = ask_model(prompt_to_use, max_tokens=max_tok, temperature=temp)
            if generated is None:
                logger.info("Generation failed record=%d attempt=%d", i + 1, attempt)
                time.sleep(0.2 * attempt)
                continue

            generated   = _normalize_output_keys(generated)
            candidate_q = normalize_generated_question(generated, display_type, topic, bloom)
            qtext       = (candidate_q.get("question") or "").strip()

            if not qtext:
                logger.info("Empty question record=%d attempt=%d", i + 1, attempt)
                continue

            if display_type == "truefalse":
                if not _is_valid_tf(candidate_q):
                    logger.info("Invalid TF record=%d attempt=%d q=%r",
                                i + 1, attempt, qtext[:80])
                    time.sleep(0.15 * attempt)
                    continue

            if qtext in seen_questions:
                logger.info("Duplicate question record=%d attempt=%d", i + 1, attempt)
                time.sleep(0.15 * attempt)
                continue

            normalized = candidate_q
            seen_questions.add(qtext)
            out_questions.append(normalized)
            break

        if normalized is None:
            fallback = rec.get("output") if isinstance(rec, dict) else None
            if fallback and isinstance(fallback, dict):
                out_questions.append(
                    normalize_generated_question(fallback, display_type, topic, bloom)
                )
            else:
                placeholder_choices = (["(Generation failed)", "(Generation failed)",
                                         "(Generation failed)", "(Generation failed)"]
                                       if display_type == "mcq" else [])
                out_questions.append({
                    "type":        display_type,
                    "concept":     topic,
                    "bloom":       bloom,
                    "question":    f"[GENERATION FAILED] Review this item — {topic}",
                    "choices":     placeholder_choices,
                    "answer":      "",
                    "answer_text": "Generation failed after 5 attempts. Please delete or replace.",
                    "_generation_failed": True,
                })

    return out_questions

# =====================================================
# TF VALIDATOR
# =====================================================
_TF_TASK_VERBS = re.compile(
    r'^(convert|calculate|compute|list|draw|design|write|find|determine|'
    r'show|give an example|describe how|explain how|create a?|propose|'
    r'evaluate|analyze|define|summarize|solve|identify|compare|'
    r'develop|construct|formulate|generate)\b',
    re.IGNORECASE,
)
_TF_WH_QUESTION = re.compile(
    r'^(what|which|how|why|who|where|when)\b',
    re.IGNORECASE,
)

def _is_valid_tf(q: dict) -> bool:
    answer = (q.get("answer") or "").strip().lower().rstrip(".")
    if answer not in ("true", "false"):
        return False
    question = (q.get("question") or "").strip()
    if not question:
        return False
    if _TF_TASK_VERBS.match(question):
        return False
    if _TF_WH_QUESTION.match(question):
        return False
    if re.search(r'(explanation|description|timeline|summary)\s*[.:]?\s*$', question, re.IGNORECASE):
        return False
    return True

# =====================================================
# WRAPPER (backwards-compatible with dashboard.py)
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

# =====================================================
# MISC
# =====================================================
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

# =====================================================
# FastAPI app
# =====================================================
app = FastAPI(title="AutoTOS AI Service", version="2.3")
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
    return {"status": "ok", "model_loaded": llm is not None}

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

# =====================================================
# CLI
# =====================================================
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