"""TOS save pipeline — parsing, validation, distribution, persistence.

Extracted from the old 200-line `save_tos` route handler. Each stage is a
pure function (or small class) and returns a result the route can serialise.
The route itself just orchestrates these calls.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from werkzeug.utils import secure_filename

from .bloom import SLOT_BLOOM_FOR_BUCKET, bucket_for_bloom, defaults_for
from .external_ai import extract_lesson

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Limits
# ──────────────────────────────────────────────────────────────────────
# Raised from 30 MB → 100 MB. Users can upload larger PDFs / PPTX now.
# Note: Flask MAX_CONTENT_LENGTH must also be raised (in web/config.py).
MAX_FILE_BYTES       = 100 * 1024 * 1024

# JSON-storage caps — bumped to take advantage of MEDIUMTEXT (16 MB) DB columns.
# Was 60_000 bytes (fit in TEXT 64KB column). Now 1 MB per JSON column.
# Allows large 50-100 item exams + many topics with full learn_material.
MAX_TOPICS_JSON      = 1_000_000
MAX_QUIZZES_JSON     = 1_000_000

# Per-topic learn_material cap. Kept at 3000 chars — matches num_ctx 1536.
# Raising this risks truncation at the model context boundary.
LEARN_MATERIAL_LIMIT = 3_000

DATA_URL_SIZE_HINT   = 5_000
MAX_CILOS            = 20
CILO_MAX_CHARS       = 500
ALLOWED_TEST_TYPES   = frozenset({
    "mcq", "truefalse", "open_ended",          # from legacy dashboard forms
    "MCQ", "True_False", "Open_Ended",          # v34 canonical names
})

# ──────────────────────────────────────────────────────────────────────
# topics_json parsing (supports legacy list + new {_cilos, topics} dict)
# ──────────────────────────────────────────────────────────────────────
def parse_topics_json(raw_json: str) -> Tuple[List[dict], List[str]]:
    """Returns (topics, cilos). Accepts both old list and new dict formats."""
    try:
        data = json.loads(raw_json or "[]")
    except json.JSONDecodeError:
        return [], []

    if isinstance(data, list):
        return data, []
    if isinstance(data, dict):
        return data.get("topics") or [], data.get("_cilos") or []
    return [], []


def dump_topics_json(topics: List[dict], cilos: List[str]) -> str:
    return json.dumps({"_cilos": cilos, "topics": topics}, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────
# Range string <-> index list
# ──────────────────────────────────────────────────────────────────────
def parse_range_string(r_str: str) -> List[int]:
    """Parse '1-3,5,7-8' → [0,1,2,4,6,7]. Silently skips malformed parts."""
    out: List[int] = []
    if not r_str:
        return out
    for part in r_str.split(","):
        part = part.strip()
        if "-" in part:
            try:
                s, e = part.split("-")
                out.extend(range(int(s) - 1, int(e)))
            except ValueError:
                continue
        elif part.isdigit():
            out.append(int(part) - 1)
    return out


# ──────────────────────────────────────────────────────────────────────
# Upload helpers
# ──────────────────────────────────────────────────────────────────────
def _is_data_url_or_large(s: Any, threshold: int = DATA_URL_SIZE_HINT) -> bool:
    if not s:
        return False
    if isinstance(s, str) and s.startswith("data:"):
        return True
    try:
        return len(s) > threshold
    except TypeError:
        return False


_DATA_URL_EXT_MAP = (
    ("pdf",  "pdf"),
    ("docx", "docx"), ("word", "docx"),
    ("pptx", "pptx"), ("presentation", "pptx"),
    ("plain", "txt"), ("text", "txt"),
)


def _ext_from_data_url_header(header: str) -> str:
    h = header.lower()
    for keyword, ext in _DATA_URL_EXT_MAP:
        if keyword in h:
            return ext
    return "bin"


def save_data_url_to_file(data_url: str, uploads_dir: str) -> Optional[str]:
    """Save a data: URL to disk. Returns path or None on failure."""
    if not data_url or not isinstance(data_url, str) or not data_url.startswith("data:"):
        return None
    try:
        header, encoded = data_url.split(",", 1)
    except ValueError:
        return None

    ext = _ext_from_data_url_header(header)
    try:
        decoded = base64.b64decode(encoded)
    except Exception as exc:
        logger.warning("base64 decode failed: %s", exc)
        return None

    if len(decoded) > MAX_FILE_BYTES:
        logger.warning("File exceeds MAX_FILE_BYTES (%d MB)", MAX_FILE_BYTES // (1024*1024))
        return None

    os.makedirs(uploads_dir, exist_ok=True)
    dest = os.path.join(uploads_dir, secure_filename(f"{uuid4().hex}.{ext}"))
    try:
        with open(dest, "wb") as f:
            f.write(decoded)
        return dest
    except OSError as exc:
        logger.exception("Failed to save upload: %s", exc)
        return None


# ──────────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────────
@dataclass
class ValidationError(Exception):
    message: str
    def __str__(self) -> str:  # pragma: no cover
        return self.message


@dataclass
class PreparedTopic:
    topic: str
    hours: int
    learn_material: str = ""
    learn_material_is_truncated: bool = False
    learn_material_was_file: bool = False
    file_path: Optional[str] = None
    quiz_items: int = 0
    items: int = 0
    fam: int = 0
    int_: int = 0
    cre: int = 0
    fam_range: str = ""
    int_range: str = ""
    cre_range: str = ""
    fam_pct: int = 0
    int_pct: int = 0
    cre_pct: int = 0

    def to_dict(self) -> dict:
        d = {
            "topic": self.topic,
            "hours": self.hours,
            "learn_material": self.learn_material,
            "learn_material_is_truncated": self.learn_material_is_truncated,
            "learn_material_was_file": self.learn_material_was_file,
            "file_path": self.file_path,
            "quiz_items": self.quiz_items,
            "items": self.items,
            "fam": self.fam,
            "int": self.int_,
            "cre": self.cre,
            "fam_range": self.fam_range or None,
            "int_range": self.int_range or None,
            "cre_range": self.cre_range or None,
            "fam_pct": self.fam_pct,
            "int_pct": self.int_pct,
            "cre_pct": self.cre_pct,
        }
        return d


# ──────────────────────────────────────────────────────────────────────
# Input validation
# ──────────────────────────────────────────────────────────────────────
def sanitise_cilos(cilos_in: List[Any]) -> List[str]:
    return [str(c).strip()[:CILO_MAX_CHARS] for c in cilos_in if str(c).strip()][:MAX_CILOS]


def validate_percentages(subject_type: str, data: dict) -> Tuple[int, int, int]:
    """Return (fam, int, cre) percentages. Raises ValidationError if invalid."""
    if subject_type == "custom":
        try:
            fam = int(data.get("fam_pct", 0))
            int_ = int(data.get("int_pct", 0))
            cre = int(data.get("cre_pct", 0))
        except (TypeError, ValueError):
            raise ValidationError("Custom percentages must be integer values.")
        if any(x < 0 or x > 100 for x in (fam, int_, cre)):
            raise ValidationError("Each custom percentage must be between 0 and 100.")
        if fam + int_ + cre != 100:
            raise ValidationError("Custom percentages must sum to exactly 100.")
        return fam, int_, cre
    return defaults_for(subject_type)


def validate_basic(title: str, total_quiz_raw: Any) -> int:
    """Validate title + totalQuizItems. Returns the parsed total."""
    if not title:
        raise ValidationError("Missing TOS title")
    if not total_quiz_raw:
        raise ValidationError("Missing total quiz items")
    try:
        total = int(total_quiz_raw)
        if total <= 0:
            raise ValueError
    except (TypeError, ValueError):
        raise ValidationError("Total quiz items must be a positive integer")
    return total


# ──────────────────────────────────────────────────────────────────────
# Topic processing
# ──────────────────────────────────────────────────────────────────────
def _prepare_learn_material(
    raw_lm: Any,
    uploads_dir: str,
) -> Tuple[str, bool, bool, Optional[str]]:
    """Return (stored_lm, truncated, was_file, saved_path)."""
    if not isinstance(raw_lm, str) or not raw_lm:
        return "", False, False, None

    try:
        if raw_lm.startswith("data:"):
            saved_path = save_data_url_to_file(raw_lm, uploads_dir)
            extracted = extract_lesson(raw_lm) or ""
            truncated = len(extracted) > LEARN_MATERIAL_LIMIT
            return extracted[:LEARN_MATERIAL_LIMIT], truncated, True, saved_path

        if _is_data_url_or_large(raw_lm):
            extracted = extract_lesson(raw_lm) or ""
            truncated = len(extracted) > LEARN_MATERIAL_LIMIT
            return extracted[:LEARN_MATERIAL_LIMIT], truncated, False, None

        truncated = len(raw_lm) > LEARN_MATERIAL_LIMIT
        return raw_lm[:LEARN_MATERIAL_LIMIT], truncated, False, None
    except Exception as exc:
        logger.warning("learn_material processing error: %s", exc)
        return "", True, False, None


def validate_topics(topics_in: List[dict], uploads_dir: str) -> List[PreparedTopic]:
    """Parse + validate incoming topics. Skips invalid entries silently."""
    out: List[PreparedTopic] = []
    for t in topics_in:
        name = (t.get("topic") or "").strip()
        try:
            hours = int(t.get("hours") or 0)
        except (TypeError, ValueError):
            continue
        if not name or hours <= 0:
            continue

        lm, truncated, was_file, saved_path = _prepare_learn_material(
            t.get("learn_material") or "", uploads_dir
        )
        out.append(PreparedTopic(
            topic=name, hours=hours,
            learn_material=lm,
            learn_material_is_truncated=truncated,
            learn_material_was_file=was_file,
            file_path=saved_path,
        ))
    return out


# ──────────────────────────────────────────────────────────────────────
# Quiz count distribution
# ──────────────────────────────────────────────────────────────────────
def distribute_quiz_items(topics: List[PreparedTopic], total_quiz: int) -> None:
    """Apportion `total_quiz` items across topics by hours. Mutates in place."""
    total_hours = sum(t.hours for t in topics)
    assigned = 0
    for i, t in enumerate(topics):
        if i == len(topics) - 1:
            t.quiz_items = total_quiz - assigned
        else:
            share = round((t.hours / total_hours) * total_quiz)
            t.quiz_items = share
            assigned += share
        t.items = t.quiz_items


def apply_bloom_distribution(
    topics: List[PreparedTopic],
    fam_pct: int,
    int_pct: int,
    cre_pct: int,
) -> None:
    """Compute per-topic fam/int/cre counts from percentages. Mutates in place."""
    for t in topics:
        fam = round(t.items * fam_pct / 100)
        intg = round(t.items * int_pct / 100)
        t.fam = fam
        t.int_ = intg
        t.cre = t.items - (fam + intg)
        t.fam_pct, t.int_pct, t.cre_pct = fam_pct, int_pct, cre_pct


def compute_item_ranges(topics: List[PreparedTopic]) -> None:
    """Assign contiguous item numbers across fam→int→cre columns."""
    fam_no = 1
    int_no = sum(t.fam for t in topics) + 1
    cre_no = sum(t.fam + t.int_ for t in topics) + 1

    def _range(start: int, n: int) -> Tuple[str, int]:
        if n <= 0:
            return "", start
        end = start + n - 1
        return (f"{start}-{end}" if start != end else str(start)), end + 1

    for t in topics:
        t.fam_range, fam_no = _range(fam_no, t.fam)
        t.int_range, int_no = _range(int_no, t.int_)
        t.cre_range, cre_no = _range(cre_no, t.cre)


# ──────────────────────────────────────────────────────────────────────
# Tests section
# ──────────────────────────────────────────────────────────────────────
def validate_tests(tests_in: List[dict], total_quiz: int) -> List[dict]:
    """Parse + validate test definitions. Raises on mismatch with total_quiz."""
    tests: List[dict] = []
    total = 0
    for t in tests_in:
        ttype = (t.get("type") or "").strip()
        try:
            titems = int(t.get("items", 0))
        except (TypeError, ValueError):
            titems = 0
        if ttype not in ALLOWED_TEST_TYPES or titems <= 0:
            continue
        tests.append({
            "type": ttype,
            "items": titems,
            "description": (t.get("description") or "").strip(),
        })
        total += titems

    if tests and total != total_quiz:
        raise ValidationError("Test items do not match total quiz count")
    return tests


def build_test_labels(tests: List[dict], total_quiz: int) -> Tuple[List[str], List[str]]:
    """Return (test_labels_per_slot, question_types_per_slot)."""
    if not tests:
        return ["Test 1"] * total_quiz, ["mcq"] * total_quiz

    labels: List[str] = []
    types: List[str] = []
    for i, t in enumerate(tests):
        label = f"Test {i + 1}"
        labels.extend([label] * t["items"])
        types.extend([t["type"]] * t["items"])
    return labels, types


# ──────────────────────────────────────────────────────────────────────
# Question-slot construction
# ──────────────────────────────────────────────────────────────────────
def build_question_slots(
    topics: List[PreparedTopic],
    total_quiz: int,
    question_types: List[str],
) -> List[dict]:
    """Create one slot per quiz item, mapping to the topic + bloom bucket."""
    slots: List[Optional[dict]] = [None] * total_quiz

    for t in topics:
        for bucket, range_str in (
            ("fam", t.fam_range), ("int", t.int_range), ("cre", t.cre_range),
        ):
            bloom = SLOT_BLOOM_FOR_BUCKET[bucket]
            for idx in parse_range_string(range_str):
                if 0 <= idx < total_quiz:
                    slots[idx] = {
                        "topic": t.topic,
                        "learn_material": t.learn_material,
                        "file_path": t.file_path,
                        "bloom": bloom,
                    }

    # Any unfilled slot → falls back to first topic's familiarisation.
    fallback = {
        "topic": topics[0].topic,
        "learn_material": topics[0].learn_material,
        "file_path": topics[0].file_path,
        "bloom": "remembering",
    }
    for i in range(total_quiz):
        if slots[i] is None:
            slots[i] = fallback

    records: List[dict] = []
    for i in range(total_quiz):
        s = slots[i]
        qtype = question_types[i] if i < len(question_types) else "mcq"
        records.append({
            "instruction": "Generate a single exam question strictly from the provided context.",
            "input": {
                "concept": s["topic"],
                "context": s.get("learn_material") or "",
                "file_path": s.get("file_path"),
                "bloom": s.get("bloom", "Remembering"),
                "type": qtype,
            },
        })
    return records


# ──────────────────────────────────────────────────────────────────────
# Post-processing (dedup + TF validity flagging)
# ──────────────────────────────────────────────────────────────────────
_TF_TASK_VERBS = re.compile(
    r"^(convert|calculate|compute|list|draw|design|write|find|determine|"
    r"show|give an example|describe how|explain how|create a|propose|"
    r"evaluate|analyze|define|summarize|solve)\b", re.IGNORECASE,
)
_TF_WH = re.compile(r"^(what|which|how|why|who|where|when)\b", re.IGNORECASE)
_NORM_NON_WORD = re.compile(r"[^\w\s]")
_NORM_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    t = _NORM_NON_WORD.sub("", (text or "").lower().strip())
    return _NORM_WS.sub(" ", t)


def _is_valid_tf(q: dict) -> bool:
    answer = (q.get("answer") or "").strip().upper().rstrip(".")
    if answer not in ("TRUE", "FALSE"):
        return False
    question = (q.get("question") or "").strip()
    return not (_TF_TASK_VERBS.match(question) or _TF_WH.match(question))


def _deduplicate(quizzes: List[dict]) -> List[dict]:
    seen: set = set()
    out: List[dict] = []
    dropped = 0
    for q in quizzes:
        if not isinstance(q, dict):
            out.append(q)
            continue
        fp = _normalize(q.get("question", ""))[:70]
        if fp in seen:
            dropped += 1
            continue
        seen.add(fp)
        out.append(q)
    if dropped:
        logger.info("Dedup: removed %d duplicate(s).", dropped)
    return out


def postprocess_quizzes(quizzes: List[dict]) -> List[dict]:
    quizzes = _deduplicate(quizzes)
    for q in quizzes:
        if not isinstance(q, dict):
            continue
        if (q.get("type") or "").lower() in ("tf", "truefalse", "true_false", "true false"):
            if not _is_valid_tf(q):
                q["_invalid_tf"] = True
                q["answer_text"] = (
                    "⚠️ This question was flagged as an invalid True/False statement. "
                    "Please review or deselect it before saving."
                )
            # ── Capitalize for display — validation already passed using .lower() ──
            ans = (q.get("answer") or "").strip().lower()
            if ans in ("true", "false"):
                q["answer"] = ans.capitalize()   # "true" → "True", "false" → "False"

    return quizzes


# ──────────────────────────────────────────────────────────────────────
# Recompute TOS for derived (selected) exams
# ──────────────────────────────────────────────────────────────────────
def recompute_topics_for_derived(topics_json: str, quizzes: List[dict]) -> List[dict]:
    """Rebuild per-topic fam/int/cre counts + ranges from the actual quizzes."""
    import copy
    topics, _cilos = parse_topics_json(topics_json)
    if not topics or not quizzes:
        return topics

    topics = copy.deepcopy(topics)
    by_name = {t["topic"].strip().lower(): t for t in topics}

    for t in topics:
        t["fam"] = t["int"] = t["cre"] = t["items"] = t["quiz_items"] = 0
        t["fam_range"] = t["int_range"] = t["cre_range"] = ""

    for q in quizzes:
        if not isinstance(q, dict):
            continue
        concept = (q.get("concept") or "").strip()
        bucket = bucket_for_bloom(q.get("bloom") or "")
        cl = concept.lower()

        matched = by_name.get(cl)
        if matched is None:
            for key, topic in by_name.items():
                if cl in key or key in cl:
                    matched = topic
                    break
        target = matched if matched is not None else topics[0]
        target[bucket] += 1
        target["items"] += 1
        target["quiz_items"] += 1

    # Reassign contiguous item ranges.
    fam_no = 1
    int_no = sum(t["fam"] for t in topics) + 1
    cre_no = sum(t["fam"] + t["int"] for t in topics) + 1

    def _range(start: int, n: int) -> Tuple[str, int]:
        if n <= 0:
            return "—", start
        end = start + n - 1
        return (f"{start}-{end}" if start != end else str(start)), end + 1

    for t in topics:
        t["fam_range"], fam_no = _range(fam_no, t["fam"])
        t["int_range"], int_no = _range(int_no, t["int"])
        t["cre_range"], cre_no = _range(cre_no, t["cre"])
    return topics


# ──────────────────────────────────────────────────────────────────────
# JSON size budgets
# ──────────────────────────────────────────────────────────────────────
def prepare_persisted_topics_json(
    topic_dicts: List[dict], cilos: List[str],
) -> str:
    """Serialise topics+cilos, trimming learn_material if the blob is too large."""
    blob = dump_topics_json(topic_dicts, cilos)
    if len(blob) <= MAX_TOPICS_JSON:
        return blob
    trimmed = []
    for t in topic_dicts:
        t = dict(t)
        t["learn_material"] = (t.get("learn_material") or "")[:500]
        t["learn_material_is_truncated"] = True
        trimmed.append(t)
    return dump_topics_json(trimmed, cilos)


def prepare_persisted_quizzes_json(quizzes: List[dict]) -> str:
    blob = json.dumps(quizzes, ensure_ascii=False)
    if len(blob) <= MAX_QUIZZES_JSON:
        return blob

    trimmed: List[dict] = []
    for q in quizzes:
        if not isinstance(q, dict):
            continue
        slim = {
            k: q.get(k)
            for k in ("type", "concept", "bloom", "test_header", "test_description")
        }
        slim["question"]    = (q.get("question")    or "")[:800]
        slim["answer"]      = (q.get("answer")      or "")[:200]
        slim["answer_text"] = (q.get("answer_text") or "")[:300]
        choices = q.get("choices") or []
        if isinstance(choices, list) and len(json.dumps(choices)) < 1000:
            slim["choices"] = [str(c)[:300] for c in choices][:4]
        else:
            slim["choices"] = []
        trimmed.append(slim)

    blob = json.dumps(trimmed, ensure_ascii=False)
    return blob[:MAX_QUIZZES_JSON]


# ──────────────────────────────────────────────────────────────────────
# Derive Bloom percentages from stored topics (view/download paths)
# ──────────────────────────────────────────────────────────────────────
def extract_bloom_percentages(
    topics: List[dict], subject_type: str,
) -> Tuple[int, int, int]:
    """Use the *saved* percentages when present; fall back to subject defaults."""
    if topics and topics[0].get("fam_pct") is not None:
        return (
            int(topics[0]["fam_pct"]),
            int(topics[0]["int_pct"]),
            int(topics[0]["cre_pct"]),
        )
    return defaults_for(subject_type)