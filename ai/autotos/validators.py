"""Validators + dedup trackers.

Each validator returns a bool (True = accept). Side-effects (logging) happen
at the call site, not inside the validator, so the same predicate can be
reused for both runtime acceptance and offline dataset audits.
"""
from __future__ import annotations

import logging
import re
import threading
from collections import deque
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from .config import (
    CIRCULAR_CHOICE_JACCARD,
    MCQ_SUBTOPIC_JACCARD,
    SEMANTIC_DUP_COMMON_WORDS,
    SEMANTIC_DUP_JACCARD_PAIRWISE,
    TF_SEMANTIC_DUP_JACCARD,
)
from .llm import choice_has_letter_prefix

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Compiled regex (module-level for hot-path speed)
# ──────────────────────────────────────────────────────────────────────
_FP_STOPWORDS = re.compile(
    r"\b(a|an|the|and|or|of|in|on|to|for|is|are|was|were|by|at|be|its|it|"
    r"that|this|these|those|with|as|from|into|about|which|how|what|does|do)\b"
)
_FP_FILLER = re.compile(
    r"^(what (is|are|does|do) (the )?(primary |main |key )?"
    r"(focus|purpose|function|role|goal|aim|definition|meaning|concept|"
    r"example|reason|impact|effect|difference|advantage|use|importance) of\s*|"
    r"what does (the term|the word|the phrase|the concept)\s+.{0,40}(refer|mean|stand for|describe)\s*(to\s*)?\b|"
    r"which (of the following )?(best )?(describes?|defines?|explains?|is|are)\s*|"
    r"how (does?|do|is|are|can)\s*|"
    r"why (is|are|does|do)\s*)",
    re.IGNORECASE,
)
_FP_VERB_OPENER = re.compile(
    r"^(define|explain|describe|summarize|identify|analyze|evaluate|compare|"
    r"contrast|apply|solve|create|design|develop|discuss|state|examine|"
    r"assess|illustrate|demonstrate|interpret|classify|infer|relate|conclude|"
    r"criticize|judge|defend|appraise|reframe|modify|invent|collaborate)\s+",
    re.IGNORECASE,
)
_FP_QUALIFIER = re.compile(
    r"\b(primary|main|key|overall|general|core|basic|fundamental|"
    r"purpose|goal|aim|role|function|focus|use|importance|objective)\b",
    re.IGNORECASE,
)
_FP_FILLER_CTX = re.compile(
    r"\b(according|lesson|course|module|section|unit|chapter|text|context|"
    r"provided|reading|material|notes|slide|above|below|given|based)\b",
    re.IGNORECASE,
)
_FP_TRAILING_VERB = re.compile(
    r"\s+(ensure|refer|mean|indicate|show|suggest|imply|denote|involve|"
    r"describe|define|represent|state|explain|allow|enable|prevent|protect|"
    r"provide|require|help|support|include|contain|affect|impact|cause|create|"
    r"result|lead|contribute|determine|measure|assess|reflect)\s*$",
    re.IGNORECASE,
)
_ANS_FP_STOPWORDS = re.compile(
    r"\b(a|an|the|and|or|of|in|on|to|for|is|are|was|were|by|at|be|its|it|"
    r"that|this|with|as|from|they|their|can|will|may|has|have|used|"
    r"also|both|each|such|than|then|when|where|while)\b",
    re.IGNORECASE,
)
_SEM_DUP_STOPWORDS = re.compile(
    r"\b(a|an|the|and|or|of|in|on|to|for|is|are|was|were|by|at|be|its|it|"
    r"that|this|these|those|with|as|from|they|their|such|each|used|"
    r"using|allows|allow|makes|make|uses|use|helps|help|enables|enable|"
    r"provides|provide|requires|require|ensures|ensure|offer|offers)\b",
    re.IGNORECASE,
)
_BLANK_STEM_RE = re.compile(r"_{4,}|\.{3,}\s*$")
_FULL_SENTENCE_OPENER = re.compile(
    r"^(it (is|was|can|will|has|does|did|should|would|could|must|might)\b|"
    r"by (the|a|an|its|this|that|making|allowing|enabling|providing|using|doing|giving|increasing|reducing|removing|replacing|combining)\b|"
    r"by [a-z]+ing\b|"
    r"they (are|were|can|will|have|do)\b|"
    r"this (is|was|makes|allows|enables|provides|ensures|refers)\b|"
    r"these (are|were)\b|"
    r"there (is|are|was|were)\b|"
    r"to (store|perform|record|connect|display|process|enable|allow|provide|ensure|help|use|make|give|increase|reduce|replace|control|combine|measure|identify|define|describe|analyze|evaluate|create|develop|design|modify)\b|"
    r"(a|an) [a-z]+ (that|which|for|to|of|in)\b)",
    re.IGNORECASE,
)
_JUNK_DISTRACTOR_RE = re.compile(
    r"\b(color (theme|of)|desk space|cable connection|count.*cable|"
    r"color.*icon|icon.*color|physical.*space|measuring.*space|"
    r"checking.*color|counting.*cable|counting.*wire)\b",
    re.IGNORECASE,
)
_JUNK_AT_RE = re.compile(
    r"best (aligns? with|represents?) the principles? of .{3,60} at the \w+ level\b|"
    r"is (the )?correct because it best (aligns?|represents?)\b",
    re.IGNORECASE,
)
_CIRC_STOPWORDS = re.compile(
    r"\b(a|an|the|and|or|of|in|on|to|for|is|are|was|were|by|it|"
    r"that|this|with|as|from|they|their|you|when|what|how|why|which|"
    r"does|do|can|will|would|describe|explain|happens?|encounter)\b",
    re.IGNORECASE,
)
_TERM_DEF_RE = re.compile(
    r"^("
    r"what does (the term|the word|the phrase|the concept)\s*.{1,50}(refer to|mean|stand for|primarily refer|represent)\b|"
    r"what is (the term|a term) (used to|for|that)\b|"
    r"what (is|are) (the term|the definition of|the meaning of)\s|"
    r"which term (describes?|refers? to|identifies?|defines?|is used|best describes?|best refers?)\b|"
    r"what term (describes?|refers? to|identifies?|is used)\b|"
    r"define (the )?(term|concept|word|phrase|meaning of)\b|"
    r"define [a-z].{2,50}\.$|"
    r"define [a-z].{2,40}\?|"
    r"explain (the meaning|what .{2,40}means)\b|"
    r"what does .{2,60} (involve|consist of|entail|comprise)\b)",
    re.IGNORECASE,
)
_WHICH_STMT_BEST_RE = re.compile(
    r"^which (statement|of the following statements?|option|answer)\s+"
    r"(best |most )?"
    r"(summarizes?|explains?|describes?|illustrates?|represents?|captures?|"
    r"details?|outlines?|reflects?|shows?|demonstrates?|conveys?)",
    re.IGNORECASE,
)
WHICH_STMT_BEST_EXEMPT = frozenset({"Analyzing", "Understanding"})

_MCQ_NEG_STEM_RE = re.compile(
    r"\b(is not|are not|does not|do not|cannot|can't|doesn't|aren't|isn't|"
    r"which is not|which are not|which does not|which cannot|"
    r"that is not|that are not|not a recognized|not considered|not an example)\b",
    re.IGNORECASE,
)
_MCQ_OPENER_RE = re.compile(
    r"^(how (does|do|is|are|can|would|will|should|could)|"
    r"which of the following|"
    r"what is|what are|what was|"
    r"why is|why are|why does|why do|"
    r"what happens|in what way|what role|what makes|"
    r"when does|when is|where is|where are)",
    re.IGNORECASE,
)
MCQ_CAPPED_OPENERS = frozenset({"which_of", "what_is_purpose", "how_would_describe"})

_MCQ_SUBTOPIC_STOPWORDS = re.compile(
    r"\b(a|an|the|and|or|of|in|on|to|for|is|are|was|were|by|at|be|its|it|"
    r"that|this|these|those|with|as|from|how|what|which|when|where|why|"
    r"does|do|can|will|may|has|have|been|being|they|their|such|each|used|"
    r"primarily|mainly|often|generally|usually|typically|between|among|"
    r"following|correctly|commonly|specifically|properly|typically|"
    r"purpose|goal|aim|role|function|importance|benefit|feature|"
    r"given|provide|ensure|allow|make|help|support|affect|impact|cause)\b",
    re.IGNORECASE,
)
_TF_SEM_STOPWORDS = re.compile(
    r"\b(a|an|the|and|or|of|in|on|to|for|is|are|was|were|by|at|be|its|it|"
    r"that|this|these|those|with|as|from|into|about|which|how|what|does|do|"
    r"not|can|will|may|has|have|had|been|being|they|their|such|each|used|"
    r"also|both|than|then|when|where|while|primarily|mainly|often|"
    r"generally|usually|typically|mostly|largely|rather|instead|"
    r"whether|either|neither|nor|just|only|always|never|still|even|"
    r"more|most|less|least|very|quite|somewhat)\b",
    re.IGNORECASE,
)
_TF_LAZY_RE = re.compile(
    r"correctly can improve system reliability\b|"
    r"is a common assumption that accurately reflects\b|"
    r"is commonly used to store and manage sensitive data\b|"
    r"is a foundational concept that underpins many advanced computing\b|"
    r"requires configuration for all deployment scenarios\b|"
    r"requires examining system architecture to detect vulnerabilities\b|"
    r"differ(s)? in scope and application\b|"
    r"can be used to solve (complex )?problems in practice\b|"
    r"underpins many advanced (computing|system)\b|"
    r"^(applying|analyzing|evaluating|creating|remembering|understanding|comparing)\s+"
    r"\w.{4,60}(in practice|in the field|in real.world scenarios?|"
    r"correctly|accurately|effectively|efficiently)\s*\.?\s*$",
    re.IGNORECASE,
)
_TF_JUNK_AT_RE = re.compile(
    r"as established in the relevant study material\b|"
    r"does not match the documented technical definition of this concept\b|"
    r"it does not match the documented\b|"
    r"as (documented|stated|noted|described|outlined) in the (study material|lesson|context|source)\b|"
    r"this is consistent with the (study material|lesson|source material|course content)\b",
    re.IGNORECASE,
)
_TF_CONTEXT_FRAMING_RE = re.compile(
    r"^(in the context of\b|"
    r"in a scenario where\b|"
    r"given (the context|that [a-z].{3,60}(must|should|can|is|are|will|would))\b|"
    r"assuming (that\b|a\b|an\b)|"
    r"when considering\b|"
    r"under the assumption\b|"
    r"in (this|the) case (where|of)\b)",
    re.IGNORECASE,
)
_TF_NEGATED_RE = re.compile(
    r"^(it is (false|not true|incorrect|inaccurate|wrong) that\b|"
    r"it is (incorrect|inaccurate|wrong) to (state|say|claim|assert) that\b|"
    r"it is not the case that\b)",
    re.IGNORECASE,
)
_TF_IT_IS_TRUE_RE = re.compile(r"^it is true that\b", re.IGNORECASE)
_TF_META_RE = re.compile(
    r"^the statement .{5,} is (true|false|correct|incorrect)\b", re.IGNORECASE
)
_TF_TASK_VERBS = re.compile(
    r"^(convert|calculate|compute|list|draw|design|write|find|determine|"
    r"show|give an example|describe how|explain how|create a?|propose|"
    r"evaluate|analyze|define|summarize|solve|identify|compare|"
    r"develop|construct|formulate|generate|"
    r"contrast|correlate|distill|conclude|categorize|"
    r"criticize|judge|defend|appraise|prioritize|reframe|grade|"
    r"modify|invent|rewrite|collaborate|"
    r"interpret|classify|infer|paraphrase|relate|transfer|articulate|discover|"
    r"connect|devise|describe|recognize|recite|illustrate|complete)\b",
    re.IGNORECASE,
)
_TF_WH = re.compile(r"^(what|which|how|why|who|where|when)\b", re.IGNORECASE)
_TF_NEG_IN_STMT = re.compile(
    r"\b(not|never|no|doesn't|don't|isn't|aren't|cannot|can't|won't|"
    r"wouldn't|shouldn't|couldn't|neither|nor)\b",
    re.IGNORECASE,
)
_OPEN_VERB_RE = re.compile(
    r"^(evaluate|assess|analyze|examine|compare|justify|critique|judge|"
    r"design|propose|formulate|develop|construct|create|build|apply|"
    r"demonstrate|solve|use|discuss|explain|describe)\b",
    re.IGNORECASE,
)
_MCQ_PLACEHOLDER_CHOICE = re.compile(
    r"^(choice text|option [abcd]|answer [abcd]|placeholder|n/a|none)\s*$",
    re.IGNORECASE,
)

ANSWER_ALWAYS_BAD = frozenset({"—", "-", "", "answer:"})
OPEN_PLACEHOLDERS = frozenset({
    "model answer here", "model answer here.", "answer here",
    "write answer here", "<complete model answer based on context>",
    "<write a complete model answer>",
    "<exactly 1 complete sentence answer based on context>",
    "<exactly 2 complete sentences.>",
    "1 complete sentence answer based on context.",
    "1–2 sentences.", "1-2 sentences.",
    "2-4 sentences.", "2–4 sentences.",
})


# ──────────────────────────────────────────────────────────────────────
# Fingerprinting (for global dedup)
# ──────────────────────────────────────────────────────────────────────
def question_fingerprint(q: dict) -> str:
    raw = (q.get("question") or "").lower().strip()
    raw = re.sub(r"[^\w\s]", " ", raw)
    for pat in (_FP_STOPWORDS, _FP_FILLER, _FP_VERB_OPENER):
        raw = pat.sub(" " if pat is _FP_STOPWORDS else "", raw).strip()
    raw = _FP_QUALIFIER.sub(" ", raw)
    raw = _FP_FILLER_CTX.sub(" ", raw)
    raw = _FP_TRAILING_VERB.sub("", raw).strip()
    raw = re.sub(r"\s+", " ", raw).strip()
    words = [w[:-1] if w.endswith("s") and len(w) > 4 else w for w in raw.split()]
    qtext = " ".join(words)[:35]
    concept = re.sub(r"\s+", "_", (q.get("concept") or "").lower().strip())
    return f"{concept}::{qtext}"


def answer_fingerprint(q: dict) -> Optional[str]:
    """Build a dedup key from the MCQ correct-answer text.

    With v31, q['answer'] is a letter A/B/C/D, so we look up the actual
    choice string before fingerprinting.
    """
    if (q.get("type") or "") != "MCQ":
        return None
    ans_letter = (q.get("answer") or "").strip().upper().rstrip(".")
    choices = q.get("choices") or []
    if ans_letter not in ("A", "B", "C", "D") or len(choices) < 4:
        return None
    idx = ord(ans_letter) - ord("A")
    if idx >= len(choices):
        return None
    ans_text = (choices[idx] or "").strip().lower()
    if not ans_text or len(ans_text) < 8:
        return None
    norm = re.sub(r"\s+", " ", _ANS_FP_STOPWORDS.sub(" ", ans_text)).strip()
    if not norm:
        return None
    concept = re.sub(r"\s+", "_", (q.get("concept") or "").lower().strip())
    return f"{concept}::ans::{norm[:50]}"


def question_stem(q: dict) -> str:
    return (q.get("question") or "")[:80].strip()


# ──────────────────────────────────────────────────────────────────────
# Choice-quality predicates
# ──────────────────────────────────────────────────────────────────────
def _content_words(text: str) -> set[str]:
    t = _SEM_DUP_STOPWORDS.sub(" ", text.lower())
    return {w for w in re.findall(r"\b\w+\b", t) if len(w) > 3}


def has_semantic_duplicate_choices(choices: list[str]) -> bool:
    if not choices or len(choices) < 2:
        return False
    word_sets = [_content_words(c) for c in choices]
    non_empty = [ws for ws in word_sets if ws]
    if len(non_empty) >= 2:
        common = non_empty[0].copy()
        for ws in non_empty[1:]:
            common &= ws
        if len(common) >= SEMANTIC_DUP_COMMON_WORDS:
            return True
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            a, b = word_sets[i], word_sets[j]
            if len(a) >= 4 and len(b) >= 4:
                union = a | b
                if union and len(a & b) / len(union) >= SEMANTIC_DUP_JACCARD_PAIRWISE:
                    return True
    return False


def is_valid_blank_completion(q: dict) -> bool:
    if not _BLANK_STEM_RE.search((q.get("question") or "").strip()):
        return True
    choices = q.get("choices") or []
    if not choices:
        return True
    return sum(1 for c in choices if _FULL_SENTENCE_OPENER.match(c)) < 2


def has_junk_distractors(choices: list[str]) -> bool:
    return any(_JUNK_DISTRACTOR_RE.search(c or "") for c in choices)


def has_junk_answer_text_mcq(at: str) -> bool:
    return bool(at and _JUNK_AT_RE.search(at))


def _circular_keywords(text: str) -> FrozenSet[str]:
    t = _CIRC_STOPWORDS.sub(" ", text.lower())
    t = re.sub(r"[^\w\s]", " ", t)
    return frozenset(w for w in t.split() if len(w) >= 4)


def has_circular_choice(question: str, choices: list[str]) -> bool:
    if not question or not choices:
        return False
    qw = _circular_keywords(question)
    if not qw:
        return False
    for c in choices:
        cw = _circular_keywords(c or "")
        if not cw:
            continue
        union = qw | cw
        if union and len(qw & cw) / len(union) >= CIRCULAR_CHOICE_JACCARD:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────
# Question-stem predicates
# ──────────────────────────────────────────────────────────────────────
def is_term_definition(q: str) -> bool:
    return bool(_TERM_DEF_RE.match(q or ""))


def is_which_statement_best(q: str) -> bool:
    return bool(_WHICH_STMT_BEST_RE.match(q or ""))


def is_mcq_negation(q: str) -> bool:
    return bool(_MCQ_NEG_STEM_RE.search(q or ""))


def is_tf_lazy(q: str) -> bool:
    return bool(_TF_LAZY_RE.search(q or ""))


def has_tf_junk_answer_text(at: str) -> bool:
    return bool(at and _TF_JUNK_AT_RE.search(at))


def extract_open_starter_verb(q: str) -> str:
    m = _OPEN_VERB_RE.match((q or "").strip())
    return m.group(1).lower() if m else ""


# ──────────────────────────────────────────────────────────────────────
# MCQ opener categorisation
# ──────────────────────────────────────────────────────────────────────
def extract_mcq_opener(question: str) -> str:
    q = (question or "").strip().lower()
    m = _MCQ_OPENER_RE.match(q)
    if not m:
        return "other"
    raw = m.group(1).lower()
    if raw.startswith("how"):
        if re.match(r"^how would you (describe|explain|summarize|outline|characterize)\b", q):
            return "how_would_describe"
        return "how"
    if raw.startswith("which of"):
        return "which_of"
    if raw.startswith(("what is", "what are", "what was")):
        if re.search(r"\b(primary|main|key|core)\s+(purpose|goal|aim|role|function)\b", q):
            return "what_is_purpose"
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


# ──────────────────────────────────────────────────────────────────────
# Sub-topic & TF semantic dedup — keyword sets
# ──────────────────────────────────────────────────────────────────────
def mcq_subtopic_words(q: str) -> FrozenSet[str]:
    s = re.sub(r"[^\w\s]", " ", q.lower())
    s = _MCQ_SUBTOPIC_STOPWORDS.sub(" ", s)
    return frozenset(w for w in s.split() if len(w) >= 4)


@lru_cache(maxsize=2048)
def tf_content_words(q: str) -> FrozenSet[str]:
    s = re.sub(r"[^\w\s]", " ", q.lower())
    s = _TF_SEM_STOPWORDS.sub(" ", s)
    return frozenset(w for w in s.split() if len(w) > 3)


# ──────────────────────────────────────────────────────────────────────
# TF master validator
# ──────────────────────────────────────────────────────────────────────
def is_valid_tf(q: dict) -> bool:
    answer = (q.get("answer") or "").strip().lower().rstrip(".")
    if answer not in ("true", "false"):
        return False
    question = (q.get("question") or "").strip()
    if not question or len(question) < 25:
        return False
    if _TF_TASK_VERBS.match(question) or _TF_WH.match(question):
        return False
    if _TF_NEGATED_RE.match(question):
        return False
    if _TF_IT_IS_TRUE_RE.match(question) and answer == "false":
        return False
    if re.search(r"(explanation|description|timeline|summary)\s*[.:]?\s*$", question, re.IGNORECASE):
        return False
    if question.rstrip().endswith(":"):
        return False
    if _TF_META_RE.match(question):
        return False
    if answer == "false" and _TF_NEG_IN_STMT.search(question):
        return False
    if _TF_CONTEXT_FRAMING_RE.match(question):
        return False
    if is_tf_lazy(question):
        return False
    if has_tf_junk_answer_text(q.get("answer_text") or ""):
        return False
    return True


# ──────────────────────────────────────────────────────────────────────
# MCQ master validator
# ──────────────────────────────────────────────────────────────────────
def _answer_matches_explanation(answer_letter, choices, answer_text):
    """Confidence check: answer's chosen option is consistent with the
    explanation. Returns (ok, suggested_letter)."""
    if not (choices and answer_letter and answer_letter in "ABCD" and answer_text):
        return True, answer_letter
    idx = ord(answer_letter.upper()) - ord("A")
    if not 0 <= idx < len(choices):
        return True, answer_letter
    ans_low = answer_text.lower()
    scores = sorted(
        (
            (sum(1 for w in re.findall(r"\b\w{4,}\b", c.lower()) if w in ans_low), i)
            for i, c in enumerate(choices)
        ),
        reverse=True,
    )
    best_score, best_idx = scores[0]
    chosen = next(s for s, i in scores if i == idx)
    best_letter = chr(ord("A") + best_idx)
    if best_letter != answer_letter and best_score - chosen >= 3:
        return False, best_letter
    return True, answer_letter


def is_valid_answer(q: dict, display_type: str) -> bool:
    answer = (q.get("answer") or "").strip()
    ans_low = answer.lower().rstrip(".")
    if ans_low in ANSWER_ALWAYS_BAD:
        return False

    if display_type == "MCQ":
        choices = q.get("choices") or []
        if not answer or len(choices) != 4:
            return False
        for c in choices:
            if not c or not c.strip():
                return False
            if choice_has_letter_prefix(c):
                return False
            if _MCQ_PLACEHOLDER_CHOICE.match(c.strip()):
                return False

        # v31 model outputs answer as a single letter A/B/C/D
        ans_letter = answer.strip().upper().rstrip(".")
        if ans_letter not in ("A", "B", "C", "D"):
            return False

        clow = [c.lower().strip() for c in choices]
        if len(set(clow)) != len(clow):
            return False
        if has_semantic_duplicate_choices(choices):
            return False
        if not is_valid_blank_completion(q):
            return False
        if has_junk_distractors(choices):
            return False
        if has_circular_choice(q.get("question", ""), choices):
            return False
        at = (q.get("answer_text") or "").strip()
        if at and has_junk_answer_text_mcq(at):
            return False
        if at and len(choices) >= 3:
            ok, _ = _answer_matches_explanation(ans_letter, choices, at)
            if not ok:
                return False
        return True

    if display_type == "True_False":
        return ans_low in ("true", "false")

    if display_type == "Open_Ended":
        if len(answer) < 15 or ans_low in OPEN_PLACEHOLDERS:
            return False
        if re.match(r"^(model answer|answer\s*:)", ans_low):
            return False
        # >4 sentence answers are rejected (v31 trains 2-4 sentences).
        if len(re.findall(r"(?<=[.!?])\s+[A-Z]", answer)) >= 4:
            return False
    return True


def is_valid_fallback(q: dict, display_type: str) -> bool:
    """Stricter gate for the dataset-fallback case."""
    qtext = (q.get("question") or "").strip()
    if len(qtext) < 20 or is_term_definition(qtext):
        return False
    if display_type == "MCQ":
        bloom = (q.get("bloom") or "").strip()
        if is_which_statement_best(qtext) and bloom not in WHICH_STMT_BEST_EXEMPT:
            return False
        if is_mcq_negation(qtext):
            return False
        if len(q.get("choices") or []) != 4:
            return False
        at = (q.get("answer_text") or "").strip()
        if at and has_junk_answer_text_mcq(at):
            return False
        if not (q.get("answer") or "").strip():
            return False
    elif display_type == "True_False":
        if not is_valid_tf(q):
            return False
    return True


# ══════════════════════════════════════════════════════════════════════
# DEDUP TRACKERS
# ══════════════════════════════════════════════════════════════════════
class DedupTracker:
    """Holds all per-batch dedup state. One instance per generation batch.

    Centralising this gives a clean single-argument interface to the generator
    (vs the old 18-parameter `_generate_single`).
    """

    __slots__ = (
        "_lock",
        "fps", "answer_fps", "stems",
        "tf_by_concept", "open_combos", "open_verbs",
        "term_defs", "mcq_by_concept", "mcq_openers",
    )

    def __init__(self, stem_history: int = 16) -> None:
        self._lock = threading.Lock()
        self.fps:        Set[str] = set()
        self.answer_fps: Set[str] = set()
        self.stems = deque(maxlen=stem_history)

        self.tf_by_concept:  Dict[str, List[Tuple[FrozenSet[str], str]]] = {}
        self.open_combos:    Set[str] = set()
        self.open_verbs:     Dict[str, Set[str]] = {}
        self.term_defs:      Dict[str, int] = {}
        self.mcq_by_concept: Dict[str, List[FrozenSet[str]]] = {}
        self.mcq_openers:    Dict[str, Dict[str, int]] = {}

    # ── stems ────────────────────────────────────────────────────────
    def stems_snapshot(self) -> List[str]:
        with self._lock:
            return list(self.stems)

    def push_stem(self, stem: str) -> None:
        if stem:
            with self._lock:
                self.stems.append(stem)

    # ── global FPs ───────────────────────────────────────────────────
    def is_fp_dup(self, fp: str) -> bool:
        with self._lock:
            if fp in self.fps:
                return True
            self.fps.add(fp)
            return False

    def is_ans_fp_dup(self, fp: Optional[str]) -> bool:
        if not fp:
            return False
        with self._lock:
            if fp in self.answer_fps:
                return True
            self.answer_fps.add(fp)
            return False

    # ── term-def quota (max 1 per concept for low-bloom levels) ──────
    def allow_term_def(self, concept: str, bloom: str) -> bool:
        if bloom.lower() not in ("remembering", "understanding"):
            return False
        with self._lock:
            n = self.term_defs.get(concept, 0)
            if n >= 1:
                return False
            self.term_defs[concept] = n + 1
            return True

    # ── MCQ openers ──────────────────────────────────────────────────
    def is_mcq_opener_overused(self, concept: str, opener: str) -> bool:
        if opener not in MCQ_CAPPED_OPENERS:
            return False
        with self._lock:
            return self.mcq_openers.get(concept, {}).get(opener, 0) >= 1

    def register_mcq_opener(self, concept: str, opener: str) -> None:
        with self._lock:
            self.mcq_openers.setdefault(concept, {})[opener] = (
                self.mcq_openers.get(concept, {}).get(opener, 0) + 1
            )

    def mcq_opener_hint(self, concept: str) -> str:
        with self._lock:
            counts = self.mcq_openers.get(concept, {})
        notes: List[str] = []
        if counts.get("which_of", 0) >= 1:
            notes.append("Do NOT start with 'Which of the following'. Ask a specific question instead.")
        if counts.get("what_is_purpose", 0) >= 1:
            notes.append("Do NOT ask 'What is the primary purpose'. Ask about a specific component, scenario, or consequence.")
        if counts.get("how_would_describe", 0) >= 1:
            notes.append("Do NOT use 'How would you describe/explain'. Ask about a specific fact, mechanism, or real-world consequence instead.")
        return " ".join(notes)

    # ── MCQ subtopic saturation ──────────────────────────────────────
    def is_mcq_subtopic_saturated(self, concept: str, words: FrozenSet[str]) -> bool:
        if not words:
            return False
        with self._lock:
            for ex in self.mcq_by_concept.get(concept, []):
                union = words | ex
                if union and len(words & ex) / len(union) >= MCQ_SUBTOPIC_JACCARD:
                    return True
        return False

    def register_mcq_subtopic(self, concept: str, words: FrozenSet[str]) -> None:
        with self._lock:
            self.mcq_by_concept.setdefault(concept, []).append(words)

    # ── TF semantic dedup ────────────────────────────────────────────
    def is_tf_semantic_dup(self, concept: str, words: FrozenSet[str]) -> bool:
        if not words:
            return False
        with self._lock:
            for ex_words, _ in self.tf_by_concept.get(concept, []):
                union = words | ex_words
                if union and len(words & ex_words) / len(union) >= TF_SEMANTIC_DUP_JACCARD:
                    return True
        return False

    def register_tf(self, concept: str, words: FrozenSet[str], answer: str) -> None:
        with self._lock:
            self.tf_by_concept.setdefault(concept, []).append((words, answer))

    def tf_balance_note(self, concept: str) -> str:
        with self._lock:
            existing = self.tf_by_concept.get(concept, [])
        if len(existing) < 2:
            return ""
        true_n  = sum(1 for _, a in existing if a == "true")
        false_n = sum(1 for _, a in existing if a == "false")
        if false_n >= 2 and true_n == 0:
            return "Vary the answer — write a TRUE statement about this concept."
        if true_n >= 2 and false_n == 0:
            return "Vary the answer — write a FALSE statement about this concept."
        return ""

    # ── Open-ended diversity ─────────────────────────────────────────
    @staticmethod
    def _open_key(topic: str, bloom: str) -> str:
        return f"{topic.lower().strip()}::{bloom.lower().strip()}"

    def open_diversity_note(self, topic: str, bloom: str, question: str) -> str:
        key  = self._open_key(topic, bloom)
        verb = extract_open_starter_verb(question)
        verb_key = f"{key}::verb"
        notes: List[str] = []
        with self._lock:
            if key in self.open_combos:
                notes.append("Different angle — focus on a distinct aspect or example.")
            if verb and verb in self.open_verbs.get(verb_key, set()):
                alt_map = {
                    "evaluating": ["Assess", "Critique", "Judge", "Justify"],
                    "creating":   ["Design", "Formulate", "Propose", "Develop"],
                    "applying":   ["Demonstrate", "Apply", "Solve", "Use"],
                    "analyzing":  ["Compare", "Examine", "Analyze"],
                }
                alts = alt_map.get(bloom.lower(), [])
                notes.append(f"Use a different question starter — try: {' / '.join(alts) or 'a different verb'}.")
        return " ".join(notes)

    def is_open_verb_repeat(self, topic: str, bloom: str, verb: str) -> bool:
        if not verb:
            return False
        verb_key = f"{self._open_key(topic, bloom)}::verb"
        with self._lock:
            return verb in self.open_verbs.get(verb_key, set())

    def register_open(self, topic: str, bloom: str, question: str) -> None:
        key  = self._open_key(topic, bloom)
        verb = extract_open_starter_verb(question)
        verb_key = f"{key}::verb"
        with self._lock:
            self.open_combos.add(key)
            if verb:
                self.open_verbs.setdefault(verb_key, set()).add(verb)