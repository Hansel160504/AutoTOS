"""Generation orchestration.

The old `_generate_single` had 18+ parameters threaded through. This module
splits responsibilities:

    GenerationContext   — small per-call value object
    Generator           — owns model client + DedupTracker; pure methods
    BatchRunner         — fans out to a thread pool, fills failures with fallbacks
"""
from __future__ import annotations

import hashlib
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from . import cache
from .config import (
    ANSWER_TEXT_MAX_CHARS,
    GENERATION_WORKERS,
    INTERNAL_TO_DISPLAY,
    MAX_GEN_ATTEMPTS,
    MAX_TOKENS,
    NUM_CTX,
    normalize_bloom,
    normalize_out_type,
    normalize_type,
)
from .io_utils import (
    find_best_chunk_idx,
    get_chunks_for_text,
    lesson_from_upload,
)
from .llm import ask_model, build_prompt, normalize_question
from .validators import (
    DedupTracker,
    answer_fingerprint,
    extract_mcq_opener,
    extract_open_starter_verb,
    is_mcq_negation,
    is_term_definition,
    is_valid_answer,
    is_valid_fallback,
    is_valid_tf,
    is_which_statement_best,
    mcq_subtopic_words,
    question_fingerprint,
    question_stem,
    tf_content_words,
    WHICH_STMT_BEST_EXEMPT,
)

logger = logging.getLogger(__name__)


_RETRY_NOTES = (
    "",
    "Try a different angle — focus on a specific component or detail.",
    "Use a concrete scenario or example from the context.",
)


# ──────────────────────────────────────────────────────────────────────
# Progress tracker (thread-safe singleton)
# ──────────────────────────────────────────────────────────────────────
class Progress:
    _lock = threading.Lock()
    state: Dict[str, Any] = {"current": 0, "total": 0, "active": False}

    @classmethod
    def reset(cls, total: int) -> None:
        with cls._lock:
            cls.state.update(current=0, total=total, active=True)

    @classmethod
    def tick(cls) -> None:
        with cls._lock:
            cls.state["current"] += 1

    @classmethod
    def finish(cls) -> None:
        with cls._lock:
            cls.state["active"] = False

    @classmethod
    def snapshot(cls) -> Dict[str, Any]:
        with cls._lock:
            return dict(cls.state)


# ──────────────────────────────────────────────────────────────────────
# Per-slot value object
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Slot:
    record_idx: int
    topic: str
    bloom: str
    prompt_type: str       # internal: mcq / tf / open
    display_type: str      # display: mcq / truefalse / open_ended
    context: str
    record: dict


# ──────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────
class Generator:
    """Generates one validated question per call. Stateless except for tracker."""

    def __init__(self, tracker: DedupTracker) -> None:
        self.tracker = tracker

    # ── high-level: call the model and validate, with retries ────────
    def generate(self, slot: Slot) -> Optional[dict]:
        ctx_size = NUM_CTX.get(slot.prompt_type, 1024)
        max_tok  = MAX_TOKENS.get(slot.prompt_type, 115)
        avoid    = self._initial_avoid_list(slot)

        for attempt in range(1, MAX_GEN_ATTEMPTS + 1):
            temperature = min(0.80, 0.45 + 0.175 * (attempt - 1))
            note = self._build_attempt_note(slot, attempt)

            prompt = build_prompt(
                prompt_type=slot.prompt_type,
                bloom=slot.bloom,
                concept=slot.topic,
                context=slot.context,
                attempt_note=note,
                avoid_questions=avoid,
            )

            raw = ask_model(prompt, max_tokens=max_tok, temperature=temperature, num_ctx=ctx_size)
            if raw is None:
                time.sleep(0.05 * attempt)
                continue

            candidate = normalize_question(
                raw,
                expected_display_type=slot.display_type,
                topic=slot.topic, bloom=slot.bloom,
                answer_text_max=ANSWER_TEXT_MAX_CHARS,
            )

            verdict = self._validate(slot, candidate)
            if verdict is True:
                self._register(slot, candidate)
                return candidate

            avoid = self._update_avoid(avoid, candidate)
            time.sleep(0.05 * attempt)

        return None

    # ── helpers ─────────────────────────────────────────────────────
    def _initial_avoid_list(self, slot: Slot) -> List[str]:
        all_seen = self.tracker.stems_snapshot()
        cl       = slot.topic.lower()
        concept_specific = [s for s in all_seen if cl in s.lower()]
        merged   = list(dict.fromkeys(concept_specific[-2:] + all_seen[-3:]))
        return merged[-3:]

    def _build_attempt_note(self, slot: Slot, attempt: int) -> str:
        note = _RETRY_NOTES[min(attempt - 1, len(_RETRY_NOTES) - 1)]

        if slot.prompt_type == "tf":
            balance = self.tracker.tf_balance_note(slot.topic)
            if balance:
                note = f"{note} {balance}".strip()
        elif slot.prompt_type == "open":
            diversity = self.tracker.open_diversity_note(slot.topic, slot.bloom, "")
            if diversity:
                note = f"{note} {diversity}".strip()
        elif slot.prompt_type == "mcq":
            hint = self.tracker.mcq_opener_hint(slot.topic.lower().strip())
            if hint:
                note = f"{note} {hint}".strip()
        return note

    @staticmethod
    def _update_avoid(avoid: List[str], candidate: dict) -> List[str]:
        stem = question_stem(candidate) or ""
        avoid = avoid + [stem[:25]]
        return avoid[-3:]

    # ── validation chain — returns True or False ────────────────────
    def _validate(self, slot: Slot, candidate: dict) -> bool:
        qtext = (candidate.get("question") or "").strip()
        if not qtext:
            return False

        # Term-definition limit
        if is_term_definition(qtext):
            if not self.tracker.allow_term_def(slot.topic.lower(), slot.bloom):
                logger.info("Reject: term-def quota exceeded (concept=%r)", slot.topic)
                return False

        # "Which statement best" only allowed for Analyzing/Understanding
        if is_which_statement_best(qtext) and slot.bloom not in WHICH_STMT_BEST_EXEMPT:
            logger.info("Reject: 'which statement best' (bloom=%s)", slot.bloom)
            return False

        # MCQ stem must not negate
        if slot.display_type == "MCQ" and is_mcq_negation(qtext):
            logger.info("Reject: MCQ negation in stem")
            return False

        # MCQ overused opener
        if slot.display_type == "MCQ":
            opener = extract_mcq_opener(qtext)
            if self.tracker.is_mcq_opener_overused(slot.topic.lower().strip(), opener):
                logger.info("Reject: MCQ overused opener=%s", opener)
                return False

        # TF fast-path
        if slot.display_type == "True_False" and not is_valid_tf(candidate):
            return False

        # Open-ended verb diversity
        if slot.display_type == "Open_Ended":
            verb = extract_open_starter_verb(qtext)
            if self.tracker.is_open_verb_repeat(slot.topic, slot.bloom, verb):
                logger.info("Reject: open-ended verb '%s' reused", verb)
                return False

        # MCQ subtopic saturation
        if slot.display_type == "MCQ":
            words = mcq_subtopic_words(qtext)
            if self.tracker.is_mcq_subtopic_saturated(slot.topic.lower().strip(), words):
                logger.info("Reject: MCQ subtopic saturated")
                return False

        # Global question fingerprint
        if self.tracker.is_fp_dup(question_fingerprint(candidate)):
            logger.info("Reject: question fingerprint duplicate")
            return False

        # MCQ same-answer dup
        if self.tracker.is_ans_fp_dup(answer_fingerprint(candidate)):
            logger.info("Reject: same-answer near-duplicate")
            return False

        # TF semantic dup
        if slot.display_type == "True_False":
            if self.tracker.is_tf_semantic_dup(slot.topic.lower().strip(),
                                               tf_content_words(qtext)):
                logger.info("Reject: TF semantic duplicate")
                return False

        # Final: full answer/choice validity
        if not is_valid_answer(candidate, slot.display_type):
            logger.info("Reject: invalid answer/choice (ans=%r)",
                        (candidate.get("answer") or "")[:40])
            return False
        return True

    # ── post-acceptance state updates ───────────────────────────────
    def _register(self, slot: Slot, candidate: dict) -> None:
        qtext = candidate["question"]

        if slot.display_type == "True_False":
            self.tracker.register_tf(
                slot.topic.lower().strip(),
                tf_content_words(qtext),
                candidate.get("answer", ""),
            )
        elif slot.display_type == "Open_Ended":
            self.tracker.register_open(slot.topic, slot.bloom, qtext)
        elif slot.display_type == "MCQ":
            self.tracker.register_mcq_subtopic(
                slot.topic.lower().strip(), mcq_subtopic_words(qtext)
            )
            self.tracker.register_mcq_opener(
                slot.topic.lower().strip(), extract_mcq_opener(qtext)
            )

        stem = question_stem(candidate)
        if stem:
            self.tracker.push_stem(stem)


# ──────────────────────────────────────────────────────────────────────
# Slot construction (record → Slot list, with chunk assignment)
# ──────────────────────────────────────────────────────────────────────
def _build_slots(records: List[dict], limit: int) -> List[Slot]:
    """Convert raw records to Slot objects. Assigns one context chunk
    per slot, rotating through chunks for a given (topic, document)."""
    chunk_queues: Dict[str, List[int]] = {}
    slots: List[Slot] = []

    for i in range(limit):
        rec       = records[i]
        input_obj = rec.get("input", {}) if isinstance(rec, dict) else {}

        topic = (input_obj.get("concept")
                 or input_obj.get("topic")
                 or rec.get("instruction", "General")
                 or "General")
        bloom = normalize_bloom(
            input_obj.get("bloom") or (rec.get("output") or {}).get("bloom") or "Remembering",
            slot_index=i,
        )
        prompt_type  = normalize_type(
            input_obj.get("type") or (rec.get("output") or {}).get("type") or "mcq"
        )
        display_type = normalize_out_type(INTERNAL_TO_DISPLAY.get(prompt_type, prompt_type))

        candidate = (input_obj.get("context") or input_obj.get("learn_material")
                     or input_obj.get("file_path") or rec.get("file_path") or "")
        full_text = lesson_from_upload(candidate) if candidate else ""

        context = ""
        if full_text:
            chunks = get_chunks_for_text(full_text)
            text_hash = hashlib.md5(full_text[:4096].encode("utf-8", errors="ignore")).hexdigest()[:8]
            queue_key = f"{topic}::{text_hash}"

            if queue_key not in chunk_queues:
                base = find_best_chunk_idx(chunks, topic)
                idxs = list(range(len(chunks)))
                if base in idxs:
                    idxs.remove(base)
                random.shuffle(idxs)
                idxs.insert(0, base)
                chunk_queues[queue_key] = idxs

            queue = chunk_queues[queue_key]
            chunk_idx = queue.pop(0) if queue else 0
            if not queue:
                chunk_queues.pop(queue_key, None)
            context = chunks[chunk_idx] if chunks else ""

            logger.info(
                "record=%d topic=%r bloom=%s -> chunk %d/%d",
                i + 1, topic, bloom, chunk_idx + 1, len(chunks),
            )
        else:
            logger.warning("record=%d topic=%r has NO learning material.", i + 1, topic)

        slots.append(Slot(
            record_idx=i, topic=topic, bloom=bloom,
            prompt_type=prompt_type, display_type=display_type,
            context=context, record=rec,
        ))
    return slots


# ──────────────────────────────────────────────────────────────────────
# Failure-resilient batch runner
# ──────────────────────────────────────────────────────────────────────
def _placeholder(slot: Slot) -> dict:
    base: Dict[str, Any] = {
        "type":    slot.display_type,
        "concept": slot.topic,
        "bloom":   slot.bloom,
        "question": f"[GENERATION FAILED] Review this item — {slot.topic}",
        "answer":  "",
        "answer_text": "Generation failed. Please delete or replace.",
        "_generation_failed": True,
    }
    if slot.display_type == "MCQ":
        base["choices"] = ["(Generation failed)"] * 4
    return base


def _try_fallback(slot: Slot) -> Optional[dict]:
    """Use the dataset's existing 'output' if it passes the fallback gate."""
    rec = slot.record
    if not isinstance(rec, dict):
        return None
    fb = rec.get("output")
    if not isinstance(fb, dict):
        return None

    normalized = normalize_question(
        fb,
        expected_display_type=slot.display_type,
        topic=slot.topic, bloom=slot.bloom,
        answer_text_max=ANSWER_TEXT_MAX_CHARS,
    )
    if is_valid_fallback(normalized, slot.display_type):
        return normalized
    logger.info("Fallback rejected for record=%d concept=%r",
                slot.record_idx + 1, slot.topic)
    return None


def generate_from_records(records: List[dict],
                          max_items: Optional[int] = None) -> List[dict]:
    """Top-level batch generator. Always returns len(records[:max_items]) items."""
    if max_items is not None and len(records) < max_items:
        logger.warning(
            "Got %d records but max_items=%d — likely a frontend off-by-one. "
            "Verify range(start, end+1) in the TOS→records conversion.",
            len(records), max_items,
        )

    limit = min(len(records), max_items) if max_items else len(records)
    slots = _build_slots(records, limit)

    tracker = DedupTracker()
    gen = Generator(tracker)
    Progress.reset(len(slots))

    workers = max(1, min(GENERATION_WORKERS, len(slots)))
    logger.info("Generating %d questions with %d worker(s)", len(slots), workers)
    t0 = time.time()

    results: List[Optional[dict]] = [None] * len(slots)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(gen.generate, slot): slot for slot in slots}
        for future in as_completed(futures):
            slot = futures[future]
            try:
                results[slot.record_idx] = future.result()
            except Exception as exc:
                logger.error(
                    "Worker crashed for slot %d (concept=%r): %s",
                    slot.record_idx, slot.topic, exc,
                )
                results[slot.record_idx] = None
            finally:
                Progress.tick()

    Progress.finish()
    dt = time.time() - t0
    logger.info("Batch done in %.1fs (%.1fs/q avg)", dt, dt / max(len(slots), 1))

    out: List[dict] = []
    for slot, q in zip(slots, results):
        if q is not None:
            out.append(q)
            continue
        fallback = _try_fallback(slot)
        out.append(fallback if fallback is not None else _placeholder(slot))
    return out


def generate_quiz_for_topics(records_or_topics, max_items=None,
                              test_labels=None, *args, **kwargs) -> dict:
    """Backwards-compatible wrapper used by the dashboard."""
    try:
        quizzes = generate_from_records(records_or_topics, max_items)
    except Exception as exc:
        logger.exception("generate_from_records error: %s", exc)
        quizzes = []

    if isinstance(quizzes, dict) and "quizzes" in quizzes:
        quizzes = quizzes["quizzes"]
    elif not isinstance(quizzes, list):
        try:
            quizzes = list(quizzes)
        except Exception:
            quizzes = []

    if test_labels:
        for idx, item in enumerate(quizzes):
            if isinstance(item, dict):
                item["test_header"] = test_labels[idx] if idx < len(test_labels) else ""
    return {"quizzes": quizzes}


def get_model_cache_stats() -> dict:
    return cache.stats()