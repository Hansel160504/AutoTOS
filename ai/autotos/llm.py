"""Ollama HTTP client, prompt construction, and response normalization.

Lazy connectivity check (no module-level blocking I/O) and response caching.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import cache
from .config import (
    INTERNAL_TO_DISPLAY,
    NUM_CTX,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
)
from .io_utils import (
    clean_text,
    strip_question_prefix,
    truncate_answer_text,
    truncate_open_answer,
)
from .config import normalize_out_type

logger = logging.getLogger(__name__)


# ── HTTP session with conservative retry ──────────────────────────────
def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=2, connect=2, read=0, backoff_factor=0.3,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
    )
    s.mount("http://",  HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8))
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8))
    return s


SESSION = _make_session()


# ── Connectivity (lazy, thread-safe) ──────────────────────────────────
class _OllamaState:
    ready = False
    checked_at = 0.0
    lock = threading.Lock()


def is_ready(force_recheck: bool = False) -> bool:
    """Cheap status check; re-pings every 30s on failure."""
    if _OllamaState.ready and not force_recheck:
        return True
    if not force_recheck and (time.time() - _OllamaState.checked_at) < 30:
        return _OllamaState.ready
    with _OllamaState.lock:
        try:
            r = SESSION.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
            if r.ok:
                models = [m.get("name", "") for m in r.json().get("models", [])]
                _OllamaState.ready = any(
                    OLLAMA_MODEL == m or OLLAMA_MODEL == m.split(":")[0]
                    for m in models
                )
                if not _OllamaState.ready:
                    logger.warning(
                        "Ollama is up but model %r missing. "
                        "Run: docker exec autotoss_ollama ollama create %s -f /models/Modelfile",
                        OLLAMA_MODEL, OLLAMA_MODEL,
                    )
                else:
                    logger.info("Ollama ready (model %r found).", OLLAMA_MODEL)
        except requests.RequestException as exc:
            logger.warning("Ollama unreachable: %s", exc)
            _OllamaState.ready = False
        finally:
            _OllamaState.checked_at = time.time()
    return _OllamaState.ready


def warmup() -> None:
    """Background warm-up — never raises."""
    try:
        SESSION.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":   OLLAMA_MODEL, "prompt": "Say OK", "stream": False,
                "options": {"num_predict": 3, "temperature": 0.0, "num_ctx": 1024},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        logger.info("Ollama warm-up complete.")
    except Exception as exc:
        logger.debug("Warm-up ping failed (non-fatal): %s", exc)


# ── Instructions (training-aligned, < 80 tokens each) ─────────────────
INSTRUCTION_MCQ = (
    "You are AutoTOS. Generate one exam question. Output ONLY valid JSON.\n"
    "Format: {\"type\":\"MCQ\",\"concept\":\"...\",\"bloom\":\"...\","
    "\"question\":\"...\",\"choices\":[\"...\",\"...\",\"...\",\"...\"],"
    "\"answer\":\"A|B|C|D\",\"answer_text\":\"One sentence.\"}"
)

INSTRUCTION_TF = (
    "You are AutoTOS. Generate one exam question. Output ONLY valid JSON.\n"
    "Format: {\"type\":\"True_False\",\"concept\":\"...\",\"bloom\":\"...\","
    "\"question\":\"...\",\"answer\":\"true|false\","
    "\"answer_text\":\"One sentence.\"}"
)

INSTRUCTION_OPEN = (
    "You are AutoTOS. Generate one exam question. Output ONLY valid JSON.\n"
    "Answer must be 2–4 complete sentences.\n"
    "Format: {\"type\":\"Open_Ended\",\"concept\":\"...\",\"bloom\":\"...\","
    "\"question\":\"...\",\"answer\":\"2-4 sentences.\"}"
)

INSTRUCTIONS = {
    "mcq":  INSTRUCTION_MCQ,
    "tf":   INSTRUCTION_TF,
    "open": INSTRUCTION_OPEN,
}


# ── Prompt builder ────────────────────────────────────────────────────
def build_prompt(
    *,
    prompt_type: str,
    bloom: str,
    concept: str,
    context: str,
    attempt_note: str = "",
    avoid_questions: Optional[List[str]] = None,
) -> str:
    from .config import INTERNAL_TO_DISPLAY
    instruction = INSTRUCTIONS.get(prompt_type, INSTRUCTION_MCQ)
    display_type = INTERNAL_TO_DISPLAY.get(prompt_type, prompt_type)
    avoid_block = ""
    if avoid_questions:
        recent = avoid_questions[-3:]
        items  = "; ".join(f'"{q[:25]}"' for q in recent)
        avoid_block = f"\n[Avoid repeating: {items}]\n"

    tf_note = (
        "\nTF RULES: Write a POSITIVE declarative statement (no question mark). "
        "Avoid misleading or tricky negative phrasing.\n"
    ) if prompt_type == "tf" else ""

    note   = f"\n{attempt_note.strip()}\n" if attempt_note and attempt_note.strip() else ""
    suffix = f"{tf_note}{note}{avoid_block}"

    user_msg = (
        "### Target Specification:\n"
        f"- Question Type: {display_type}\n"
        f"- Bloom's Level: {bloom}\n"
        f"- Concept: {concept}\n\n"
        "### Context (Source Material):\n"
        f"{context}{suffix}"
    )

    return (
        f"<|im_start|>system\n{instruction}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ── JSON extraction from the model response ───────────────────────────
def _extract_first_json(text: str) -> Optional[str]:
    """Find the first balanced {...} block in a (possibly noisy) string."""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def _try_parse_json(s: str) -> Optional[Any]:
    if not s or not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r",\s*([}\]])", r"\1", s)
    if cleaned.count('"') % 2 == 1:
        cleaned += '"'
    open_braces = cleaned.count("{") - cleaned.count("}")
    if 0 < open_braces <= 6:
        try:
            return json.loads(cleaned + "}" * open_braces)
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


# ── Generate ──────────────────────────────────────────────────────────
def ask_model(
    prompt: str,
    *,
    max_tokens: int = 200,
    temperature: float = 0.45,
    num_ctx: int = 1024,
) -> Optional[dict]:
    """One call to Ollama, with full caching. Returns parsed JSON dict or None."""
    if not is_ready():
        return None

    prompt = (prompt or "").strip()
    key = cache.prompt_hash_key(prompt, max_tokens, temperature, num_ctx)

    cached = cache.get(key)
    if cached is not None:
        return cached if isinstance(cached, dict) else None

    payload = {
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json",
        "options": {
            "temperature": temperature, "num_predict": max_tokens, "num_ctx": num_ctx,
            "top_p": 0.95, "think": False,
            "stop": ["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
        },
    }

    try:
        t0 = time.time()
        resp = SESSION.post(f"{OLLAMA_BASE_URL}/api/generate",
                            json=payload, timeout=OLLAMA_TIMEOUT)
        dt = time.time() - t0
    except requests.exceptions.Timeout:
        logger.error("Ollama timeout after %ds", OLLAMA_TIMEOUT)
        return None
    except requests.RequestException as exc:
        logger.exception("ask_model HTTP error: %s", exc)
        return None

    if not resp.ok:
        logger.warning("Ollama %d: %s", resp.status_code, resp.text[:200])
        return None

    raw = resp.json().get("response", "")
    logger.info(
        "ask_model dur=%.2fs raw_len=%d max_tok=%d temp=%.2f",
        dt, len(raw), max_tokens, temperature,
    )
    if not raw or not raw.strip():
        return None

    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
    if not raw:
        return None

    json_str = _extract_first_json(raw)
    parsed = _try_parse_json(json_str) if json_str else None
    if parsed is None:
        logger.warning("No JSON in response: %s", raw[:200])
        return None

    cache.put(key, parsed)
    return parsed


# ── Response normalization ────────────────────────────────────────────
_KEY_ALIASES = {
    "statement": "question", "prompt": "question",
    "sample_answer": "answer_text", "sample_response": "answer_text",
    "model_answer": "answer", "explanation": "answer_text",
    "solution": "answer_text", "rationale": "answer_text",
    "ans": "answer", "correct": "answer",
}

_CHOICE_PREFIX_RE = re.compile(r"^(?:\([A-Da-d]\)|[A-Da-d][).:\-]\s*|[A-Da-d]\s+(?=[A-Z]))")


def _strip_choice_prefix(text: str) -> str:
    if not text:
        return text
    out = _CHOICE_PREFIX_RE.sub("", text).strip()
    if out == text and len(text) > 2 and text[0].upper() in "ABCD" and text[1] == " ":
        out = text[2:].strip()
    return out


def choice_has_letter_prefix(text: str) -> bool:
    return bool(text) and len(text) >= 2 and bool(_CHOICE_PREFIX_RE.match(text))


def _alias_keys(out: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(out, dict):
        return {}
    for src, dst in _KEY_ALIASES.items():
        if src in out and dst not in out:
            out[dst] = out[src]
    if "answer" not in out and "sample_answer" in out:
        out["answer"] = out["sample_answer"]
    if isinstance(out.get("answer"), bool):
        out["answer"] = "true" if out["answer"] else "false"
    return out


def _coerce_choices(raw) -> List[str]:
    if isinstance(raw, dict):
        keys = sorted(raw.keys(), key=lambda s: s.upper())
        return [_strip_choice_prefix(clean_text(raw[k])) for k in keys][:4]
    if isinstance(raw, list):
        return [_strip_choice_prefix(clean_text(x)) for x in raw][:4]
    return []


def _coerce_mcq_answer(ans: Any, choices: List[str]) -> str:
    """Normalize MCQ answer to a single uppercase letter A/B/C/D.

    The v31-trained model outputs answer as a letter ('A', 'B', 'C', 'D').
    This function MUST keep it as a letter so the dashboard can render
    'A) <choice>' style output. Earlier versions converted letters into
    full choice text — that broke the preview.
    """
    if isinstance(ans, str):
        a = ans.strip()
        # Already a single letter (most common with v31 model)
        if re.fullmatch(r"[A-Da-d]", a):
            return a.upper()
        # 'A)', 'A.', '(A)' style → strip wrapper, take letter
        m = re.match(r"^[\(]?([A-Da-d])[\)\.\:\-\s]", a)
        if m:
            return m.group(1).upper()
        # Full text → look up which choice index it matches
        if choices:
            a_low = a.lower().rstrip(".")
            for i, c in enumerate(choices):
                if isinstance(c, str) and c.lower().strip().rstrip(".") == a_low:
                    return "ABCD"[i]
            # Substring fallback
            for i, c in enumerate(choices):
                if isinstance(c, str) and len(a_low) > 8 and a_low in c.lower():
                    return "ABCD"[i]
        return a  # Give up — return raw

    if isinstance(ans, (int, float)):
        idx = int(ans)
        if 0 <= idx < 4:
            return "ABCD"[idx]
        return str(ans)
    return clean_text(str(ans or ""))


def normalize_question(
    raw: Optional[dict],
    *,
    expected_display_type: str,
    topic: str,
    bloom: str,
    answer_text_max: int,
) -> dict:
    """Convert any model output shape into the canonical question dict."""
    raw = _alias_keys(raw or {})
    if not isinstance(raw, dict):
        raw = {"question": str(raw)}

    display = normalize_out_type(raw.get("type") or expected_display_type) or expected_display_type
    out: Dict[str, Any] = {
        "type":    display,
        "concept": topic,
        "bloom":   bloom,
        "question": strip_question_prefix(
            clean_text(raw.get("question") or raw.get("prompt") or ""),
            is_open_ended=(display == "Open_Ended"),
        ),
        "choices": [],
        "answer":  "",
    }
    if display != "Open_Ended":
        out["answer_text"] = truncate_answer_text(
            clean_text(raw.get("answer_text") or raw.get("explanation") or ""),
            answer_text_max,
        )

    out["choices"] = _coerce_choices(raw.get("choices") or raw.get("options"))

    ans = raw.get("answer", "")
    if isinstance(ans, bool):
        out["answer"] = "true" if ans else "false"
    elif display == "MCQ":
        out["answer"] = _coerce_mcq_answer(ans, out["choices"])
    elif display == "True_False":
        a = str(ans).strip().lower().rstrip(".")
        out["answer"] = {"1": "true", "yes": "true", "0": "false", "no": "false"}.get(a, a)
    elif display == "Open_Ended":
        out["answer"] = truncate_open_answer(clean_text(str(ans)))
    else:
        out["answer"] = clean_text(str(ans))

    if display == "Open_Ended" and out["answer"]:
        out["answer"] = truncate_open_answer(out["answer"])
    return out