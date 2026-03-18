# routes/dashboard.py
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from extensions import db
from models import TosRecord
from datetime import datetime
import json
import re
import traceback
import logging
import os
import base64
from uuid import uuid4
from werkzeug.utils import secure_filename
import requests
import time
import importlib
from functools import wraps

def faculty_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if current_user.is_authenticated and current_user.is_admin is True:
            # Admins don't belong here — send to their panel
            return redirect(url_for("admin.index"))
        return f(*args, **kwargs)
    return decorated

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/dashboard")

# Where to store uploads (project_root/uploads)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Maximum file bytes to accept for saving (10 MB default)
MAX_FILE_BYTES = 30 * 1024 * 1024

# ----------------------------------------------------
# Helper: call external model service (if configured)
# ----------------------------------------------------
def call_model_service(expanded_records, test_labels):
    model_url = os.getenv("AUTO_TOS_MODEL_URL") or current_app.config.get("AUTO_TOS_MODEL_URL")
    
    payload = {"records": expanded_records, "test_labels": test_labels}

    if model_url:
        endpoint = model_url.rstrip("/") + "/generate"
        try:
            start = time.time()
            logger.info("Calling external model service %s (no timeout) ...", endpoint)
            resp = requests.post(endpoint, json=payload, timeout=None)
            duration = time.time() - start
            logger.info("Model service responded status=%s in %.2fs", resp.status_code, duration)
            if resp.ok:
                try:
                    data = resp.json()
                    return data
                except Exception as e:
                    logger.warning("Failed to parse JSON from model service: %s", e)
            else:
                logger.warning("Model service returned non-ok status %d: %s", resp.status_code, resp.text[:500])
        except Exception as e:
            logger.exception("External model service call failed: %s", e)

    # Fallback to local model call (dynamic import to avoid loading model at module import)
    try:
        logger.info("Falling back to local model generation (this may be slower).")
        start = time.time()
        ai_model = importlib.import_module("ai_model")
        # call wrapper in ai_model
        local = ai_model.generate_quiz_for_topics(expanded_records, max_items=None, test_labels=test_labels)
        duration = time.time() - start
        logger.info("Local model generation completed in %.2fs", duration)
        return local
    except Exception as e:
        logger.exception("Local model generation failed: %s", e)
        return None

# ----------------------------------------------------
# Helper: extract lesson text (via model service /extract or local fallback)
# ----------------------------------------------------
def extract_lesson(data_or_text):
    model_url = os.getenv("AUTO_TOS_MODEL_URL") or current_app.config.get("AUTO_TOS_MODEL_URL")
    
    if model_url:
        endpoint = model_url.rstrip("/") + "/extract"
        try:
            resp = requests.post(endpoint, json={"data": data_or_text}, timeout=None)
            if resp.ok:
                try:
                    payload = resp.json()
                    if isinstance(payload, dict) and "text" in payload:
                        return payload.get("text") or ""
                    if isinstance(payload, str):
                        return payload
                except Exception as e:
                    logger.warning("Failed to parse /extract response JSON: %s", e)
            else:
                logger.warning("/extract returned status %d: %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.debug("Remote /extract failed: %s", e)

    # local fallback
    try:
        ai_model = importlib.import_module("ai_model")
        return ai_model.lesson_from_upload(data_or_text)
    except Exception as e:
        logger.exception("Local lesson_from_upload fallback failed: %s", e)
        return ""

# ----------------------------------------------------
# Helper: get cache stats from model service or local ai_model
# ----------------------------------------------------
def get_cache_stats():
    model_url = os.getenv("AUTO_TOS_MODEL_URL") or current_app.config.get("AUTO_TOS_MODEL_URL")
    timeout = int(os.getenv("AUTO_TOS_MODEL_TIMEOUT", "5"))
    if model_url:
        try:
            resp = requests.get(model_url.rstrip("/") + "/cache_stats", timeout=timeout)
            if resp.ok:
                try:
                    return resp.json()
                except Exception:
                    logger.warning("Failed to parse cache_stats JSON from model service")
            else:
                logger.debug("cache_stats endpoint returned %d", resp.status_code)
        except Exception as e:
            logger.debug("Failed to call model cache_stats: %s", e)

    # local fallback
    try:
        ai_model = importlib.import_module("ai_model")
        if hasattr(ai_model, "get_model_cache_stats"):
            return ai_model.get_model_cache_stats()
    except Exception as e:
        logger.debug("Local get_model_cache_stats failed: %s", e)
    return {}

# ============================================================
# HELPERS: Post-generation deduplication & T/F validation
# ============================================================

# Action verbs that mean the model generated a task, not a T/F statement
_TF_TASK_VERBS = re.compile(
    r'^(convert|calculate|compute|list|draw|design|write|find|determine|'
    r'show|give an example|describe how|explain how|create a|propose|'
    r'evaluate|analyze|define|summarize|solve)\b',
    re.IGNORECASE,
)

# Wh-questions are not valid True/False statements
_TF_WH_QUESTION = re.compile(
    r'^(what|which|how|why|who|where|when)\b',
    re.IGNORECASE,
)


def _normalize_question(text: str) -> str:
    """Lowercase + strip punctuation for duplicate fingerprinting."""
    t = (text or "").lower().strip()
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\s+', ' ', t)
    return t


def _validate_tf_question(q: dict) -> bool:
    """
    Return True only if a T/F question is a valid declarative statement.
    Rejects questions where:
      - Answer is not True/False
      - Question starts with an action/task verb (Convert, Solve, Design …)
      - Question is a Wh-question (What is …, Which …, How …)
    """
    answer = (q.get("answer") or "").strip().upper().rstrip('.')
    if answer not in ("TRUE", "FALSE"):
        logger.warning("T/F rejected — bad answer value: %r", answer)
        return False

    question = (q.get("question") or "").strip()
    if _TF_TASK_VERBS.match(question):
        logger.warning("T/F rejected — task verb: %r", question[:80])
        return False

    if _TF_WH_QUESTION.match(question):
        logger.warning("T/F rejected — wh-question: %r", question[:80])
        return False

    return True


def _deduplicate_quizzes(quizzes: list) -> list:
    """
    Remove near-duplicate questions using first-70-char fingerprint.
    First occurrence is kept; duplicates are logged and dropped.
    """
    seen_fps = set()
    unique   = []
    dropped  = 0

    for q in quizzes:
        if not isinstance(q, dict):
            unique.append(q)
            continue

        fp = _normalize_question(q.get("question", ""))[:70]

        if fp in seen_fps:
            dropped += 1
            logger.info(
                "Dedup: dropped duplicate (concept=%s bloom=%s): %r",
                q.get("concept", "?"),
                q.get("bloom",   "?"),
                q.get("question", "")[:80],
            )
            continue

        seen_fps.add(fp)
        unique.append(q)

    if dropped:
        logger.info("Dedup: removed %d duplicate(s), kept %d.", dropped, len(unique))

    return unique


def postprocess_quizzes(quizzes: list) -> list:
    """
    Run all post-generation cleanup:
      1. Deduplicate near-identical questions.
      2. Flag invalid T/F questions so faculty can spot and deselect them.
    """
    quizzes = _deduplicate_quizzes(quizzes)

    for q in quizzes:
        if not isinstance(q, dict):
            continue

        qtype = (q.get("type") or "").lower().strip()
        if qtype in ("tf", "truefalse", "true_false"):
            if not _validate_tf_question(q):
                q["_invalid_tf"]  = True
                q["answer_text"]  = (
                    "⚠️ This question was flagged as an invalid True/False "
                    "statement (it appears to be a task, not a statement). "
                    "Please review or deselect it before saving."
                )

    return quizzes


# =========================================================
# 1. DASHBOARD HOME (List Records)
# =========================================================
@dashboard_bp.route("/")
@login_required
@faculty_required  
def index():
    records = (
        TosRecord.query.filter_by(user_id=current_user.id)
        .order_by(TosRecord.id.desc())
        .all()
    )
    # Ensure date_created is always a string so templates can slice it with [:10]
    for r in records:
        if hasattr(r.date_created, 'strftime'):
            r.date_created = r.date_created.strftime("%Y-%m-%d %H:%M:%S")
    return render_template("dashboard.html", records=records)

# =========================================================
# 2. CREATE PAGE
# =========================================================
@dashboard_bp.route("/create")
@login_required
@faculty_required  
def create():
    return render_template("create_tos.html")

# small helper to parse "1-3,5,7-8" -> zero-based indexes
def parse_range_string(r_str):
    indices = []
    if not r_str:
        return indices
    parts = r_str.split(",")
    for p in parts:
        p = p.strip()
        if "-" in p:
            try:
                s, e = p.split("-")
                s_i = int(s) - 1
                e_i = int(e) - 1
                indices.extend(range(s_i, e_i + 1))
            except Exception:
                pass
        elif p.isdigit():
            try:
                indices.append(int(p) - 1)
            except Exception:
                pass
    return indices

# tiny helper to decide whether a learn_material is "large" or a data URL
def _is_large_or_data_url(s: str, size_threshold: int = 5000) -> bool:
    if not s:
        return False
    if isinstance(s, str) and s.startswith("data:"):
        return True
    try:
        return len(s) > size_threshold
    except Exception:
        return False

# save a data:...;base64,... data URL to uploads folder and return filepath (or None)
def save_data_url_to_file(data_url: str) -> str:
    if not data_url or not isinstance(data_url, str) or not data_url.startswith("data:"):
        return None
    try:
        header, encoded = data_url.split(",", 1)
    except Exception:
        return None

    ext = None
    if "pdf" in header:
        ext = "pdf"
    elif "vnd.openxmlformats-officedocument.wordprocessingml.document" in header or "docx" in header:
        ext = "docx"
    elif "presentation" in header or "pptx" in header:
        ext = "pptx"
    elif "plain" in header or "text" in header:
        ext = "txt"
    else:
        ext = "bin"

    try:
        decoded = base64.b64decode(encoded)
    except Exception as e:
        logger.warning("Failed to base64-decode data URL: %s", e)
        return None

    if len(decoded) > MAX_FILE_BYTES:
        logger.warning("Upload size %d exceeds MAX_FILE_BYTES (%d). Skipping file save.", len(decoded), MAX_FILE_BYTES)
        return None

    fname = f"{uuid4().hex}.{ext}"
    safe_name = secure_filename(fname)
    dest_path = os.path.join(UPLOADS_DIR, safe_name)

    try:
        with open(dest_path, "wb") as f:
            f.write(decoded)
        logger.info("Saved uploaded file to %s (bytes=%d)", dest_path, len(decoded))
        return dest_path
    except Exception as e:
        logger.exception("Failed to write uploaded file to disk: %s", e)
        return None

# =========================================================
# 3. SAVE TOS LOGIC (Generates Quiz & Saves to DB)
# =========================================================
@dashboard_bp.route("/save_tos", methods=["POST"])
@login_required
@faculty_required  
def save_tos():
    data = request.get_json() or {}

    title = (data.get("title") or "").strip()
    subject_type = data.get("subjectType", "nonlab")
    total_quiz = data.get("totalQuizItems")
    topics_in = data.get("topics", []) or []
    tests_in = data.get("tests", []) or []

    # allow optional custom percents for custom subjectType
    fam_pct = None
    int_pct = None
    cre_pct = None
    if subject_type == "custom":
        try:
            fam_pct = int(data.get("fam_pct", data.get("famPct", 0)))
            int_pct = int(data.get("int_pct", data.get("intPct", 0)))
            cre_pct = int(data.get("cre_pct", data.get("crePct", 0)))
        except Exception:
            return jsonify({"error": "Custom percentages must be integer values."}), 400

        # validate ranges and sum
        if any(x < 0 or x > 100 for x in (fam_pct, int_pct, cre_pct)):
            return jsonify({"error": "Each custom percentage must be between 0 and 100."}), 400
        if fam_pct + int_pct + cre_pct != 100:
            return jsonify({"error": "Custom percentages must sum to exactly 100."}), 400

    # -----------------------------
    # VALIDATE TITLE & TOTAL ITEMS
    # -----------------------------
    if not title:
        return jsonify({"error": "Missing TOS title"}), 400

    if not total_quiz:
        return jsonify({"error": "Missing total quiz items"}), 400

    try:
        total_quiz = int(total_quiz)
        if total_quiz <= 0:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Total quiz items must be a positive integer"}), 400

    # -----------------------------
    # VALIDATE TOPICS
    # -----------------------------
    valid_topics = []
    for t in topics_in:
        topic_name = (t.get("topic") or "").strip()
        hours_raw = t.get("hours")

        if not topic_name:
            continue

        try:
            hours_val = int(hours_raw)
        except (TypeError, ValueError):
            continue

        if hours_val <= 0:
            continue

        # handle learn_material carefully
        raw_lm = t.get("learn_material") or ""
        stored_lm = ""
        lm_truncated = False
        lm_was_file = False
        saved_file_path = None

        try:
            if isinstance(raw_lm, str) and raw_lm.startswith("data:"):
                saved = save_data_url_to_file(raw_lm)
                if saved:
                    saved_file_path = saved
                    lm_was_file = True
                else:
                    lm_was_file = True

                extracted = extract_lesson(raw_lm)
                if extracted:
                    if len(extracted) > 3000:
                        stored_lm = extracted[:3000]
                        lm_truncated = True
                    else:
                        stored_lm = extracted
                else:
                    stored_lm = "" 
            elif _is_large_or_data_url(raw_lm, size_threshold=5000):
                extracted = extract_lesson(raw_lm)
                if extracted:
                    if len(extracted) > 3000:
                        stored_lm = extracted[:3000]
                        lm_truncated = True
                    else:
                        stored_lm = extracted
                else:
                    stored_lm = ""
            else:
                stored_lm = raw_lm if isinstance(raw_lm, str) else str(raw_lm or "")
                if len(stored_lm) > 3000:
                    stored_lm = stored_lm[:3000]
                    lm_truncated = True
        except Exception as e:
            logger.warning("learn_material extraction error for topic %s: %s", topic_name, e)
            stored_lm = ""
            lm_truncated = True

        topic_entry = {
            "topic": topic_name,
            "hours": hours_val,
            "learn_material": stored_lm,
            "learn_material_is_truncated": lm_truncated,
            "learn_material_was_file": lm_was_file,
            "file_path": saved_file_path,
            "quiz_items": 0,
        }

        valid_topics.append(topic_entry)

    if not valid_topics:
        return jsonify({"error": "Add at least one valid topic (name + hours)"}), 400

    # ---------------------------------------------------
    # DISTRIBUTE QUIZ ITEMS BY HOURS PROPORTION
    # ---------------------------------------------------
    total_hours = sum(t["hours"] for t in valid_topics)
    assigned = 0

    for i, t in enumerate(valid_topics):
        if i == len(valid_topics) - 1:
            t["quiz_items"] = total_quiz - assigned
        else:
            q = round((t["hours"] / total_hours) * total_quiz)
            t["quiz_items"] = q
            assigned += q

        t["items"] = t["quiz_items"]

    # ---------------------------------------------------
    # BLOOM DISTRIBUTION BASED ON SUBJECT TYPE
    # ---------------------------------------------------
    if subject_type == "lab":
        fam_pct, int_pct, cre_pct = 20, 30, 50
    elif subject_type == "custom":
        fam_pct, int_pct, cre_pct = fam_pct, int_pct, cre_pct  
    else:
        fam_pct, int_pct, cre_pct = 50, 30, 20

    fam_pct = int(fam_pct)
    int_pct = int(int_pct)
    cre_pct = int(cre_pct)

    for t in valid_topics:
        items = t["items"]

        fam = round(items * fam_pct / 100)
        inte = round(items * int_pct / 100)
        cre = items - (fam + inte)

        t["fam"] = fam
        t["int"] = inte
        t["cre"] = cre
        t["fam_range"] = None
        t["int_range"] = None
        t["cre_range"] = None
        
        # --- FIX: Save exact percentages explicitly to the JSON payload ---
        t["fam_pct"] = fam_pct
        t["int_pct"] = int_pct
        t["cre_pct"] = cre_pct

    # ---------------------------------------------------
    # BUILD RANGE NUMBERS
    # ---------------------------------------------------
    fam_no = 1
    int_no = sum(t["fam"] for t in valid_topics) + 1
    cre_no = sum((t["fam"] + t["int"]) for t in valid_topics) + 1

    for t in valid_topics:
        if t["fam"] > 0:
            s, e = fam_no, fam_no + t["fam"] - 1
            t["fam_range"] = f"{s}-{e}" if s != e else str(s)
            fam_no = e + 1

        if t["int"] > 0:
            s, e = int_no, int_no + t["int"] - 1
            t["int_range"] = f"{s}-{e}" if s != e else str(s)
            int_no = e + 1

        if t["cre"] > 0:
            s, e = cre_no, cre_no + t["cre"] - 1
            t["cre_range"] = f"{s}-{e}" if s != e else str(s)
            cre_no = e + 1

    # ---------------------------------------------------
    # VALIDATE TEST BREAKDOWN 
    # ---------------------------------------------------
    tests = []
    if tests_in:
        allowed_types = {"mcq", "truefalse", "open_ended"}
        total_from_tests = 0

        for t in tests_in:
            ttype = (t.get("type") or "").strip()
            try:
                titems = int(t.get("items", 0))
            except (TypeError, ValueError):
                titems = 0
            
            description = (t.get("description") or "").strip()

            if ttype not in allowed_types or titems <= 0:
                continue

            tests.append({"type": ttype, "items": titems, "description": description})
            total_from_tests += titems

        if total_from_tests != total_quiz:
            return jsonify({"error": "Test items do not match total quiz count"}), 400

    # ---------------------------------------------------
    # BUILD TEST LABELS FOR AI GENERATOR
    # ---------------------------------------------------
    test_labels = []
    question_types_by_slot = []  
    if tests:
        for i, t in enumerate(tests):
            label = f"Test {i+1}"
            for _ in range(t["items"]):
                test_labels.append(label)
                question_types_by_slot.append(t["type"])
        if len(test_labels) != total_quiz:
            return jsonify({"error": "Test items mismatch"}), 400
    else:
        test_labels = ["Test 1"] * total_quiz
        question_types_by_slot = ["mcq"] * total_quiz

    # ---------------------------------------------------
    # BUILD question_slots 
    # ---------------------------------------------------
    question_slots = [None] * total_quiz

    for t in valid_topics:
        topic_name = t["topic"]
        lm = t.get("learn_material")
        fp = t.get("file_path")

        for idx in parse_range_string(t.get("fam_range", "")):
            if 0 <= idx < total_quiz:
                question_slots[idx] = {"topic": topic_name, "learn_material": lm, "file_path": fp, "bloom": "Remembering"}

        for idx in parse_range_string(t.get("int_range", "")):
            if 0 <= idx < total_quiz:
                question_slots[idx] = {"topic": topic_name, "learn_material": lm, "file_path": fp, "bloom": "Applying"}

        for idx in parse_range_string(t.get("cre_range", "")):
            if 0 <= idx < total_quiz:
                question_slots[idx] = {"topic": topic_name, "learn_material": lm, "file_path": fp, "bloom": "Creating"}

    fallback_topic = valid_topics[0]["topic"]
    fallback_lm = valid_topics[0].get("learn_material")
    for i in range(total_quiz):
        if question_slots[i] is None:
            question_slots[i] = {"topic": fallback_topic, "learn_material": fallback_lm, "file_path": valid_topics[0].get("file_path"), "bloom": "Remembering"}

    # ---------------------------------------------------
    # EXPAND slots into records
    # ---------------------------------------------------
    expanded_records = []
    for i in range(total_quiz):
        slot = question_slots[i]
        qtype = question_types_by_slot[i] if i < len(question_types_by_slot) else "mcq"
        rec = {
            "instruction": "Generate a single exam question strictly from the provided context.",
            "input": {
                "concept": slot["topic"],
                "context": slot.get("learn_material") or "",
                "file_path": slot.get("file_path"),  
                "bloom": slot.get("bloom", "Remembering"),
                "type": qtype
            }
        }
        expanded_records.append(rec)

    # ---------------------------------------------------
    # GENERATE QUIZ
    # ---------------------------------------------------
    try:
        model_result = call_model_service(expanded_records, test_labels)
    except Exception as e:
        logger.exception("Error during quiz generation orchestrator: %s", e)
        traceback.print_exc()
        return jsonify({"error": "Internal error generating quiz items."}), 500

    quizzes_data = []
    if isinstance(model_result, dict) and "quizzes" in model_result:
        quizzes_data = model_result.get("quizzes") or []
    elif isinstance(model_result, list):
        quizzes_data = model_result
    elif isinstance(model_result, dict):
        possible = model_result.get("results") or model_result.get("items") or []
        if isinstance(possible, list):
            quizzes_data = possible
    
    if not quizzes_data:
        logger.error("No quizzes returned by model")
        return jsonify({"error": "Failed to generate quiz items."}), 500
    
    # ---------------------------------------------------
    # FORCE INJECT HEADERS AND DESCRIPTIONS 
    # ---------------------------------------------------
    # Map "Test 1" -> description
    desc_map = {}
    for i, t in enumerate(tests):
        label = f"Test {i+1}"
        desc_map[label] = t.get("description", "")

    for i, q in enumerate(quizzes_data):
        if isinstance(q, dict):
            # 1. Force inject test_header by matching index to the original list we sent the model!
            if i < len(test_labels):
                expected_header = test_labels[i]
            else:
                expected_header = q.get("test_header") or "Test 1"
            
            q["test_header"] = expected_header

            # 2. Attach the description matching the guaranteed header
            if expected_header in desc_map:
                q["test_description"] = desc_map[expected_header]

    # ---------------------------------------------------
    # POST-PROCESS: Deduplicate + Validate T/F
    # Run AFTER headers are injected so test_header is preserved.
    # ---------------------------------------------------
    pre_count    = len(quizzes_data)
    quizzes_data = postprocess_quizzes(quizzes_data)
    post_count   = len(quizzes_data)

    if post_count < pre_count:
        logger.info(
            "Post-process: %d → %d questions after dedup/T/F validation.",
            pre_count, post_count,
        )
        # Rebuild test_labels from the already-injected headers
        test_labels = [q.get("test_header", "Test 1") for q in quizzes_data]

    # ---------------------------------------------------
    # SAVE TOS TO DATABASE
    # ---------------------------------------------------
    try:
        topics_json_safe = json.dumps(valid_topics, ensure_ascii=False)
        quizzes_json_safe = json.dumps(quizzes_data, ensure_ascii=False)

        MAX_TOPICS_JSON = 60_000
        MAX_QUIZZES_JSON = 60_000

        if len(topics_json_safe) > MAX_TOPICS_JSON:
            for t in valid_topics:
                t["learn_material"] = (t.get("learn_material") or "")[:500]
                t["learn_material_is_truncated"] = True
            topics_json_safe = json.dumps(valid_topics, ensure_ascii=False)

        if len(quizzes_json_safe) > MAX_QUIZZES_JSON:
            trimmed_quizzes = []
            for q in quizzes_data:
                if not isinstance(q, dict):
                    continue
                q_trim = {
                    "type": q.get("type"),
                    "concept": q.get("concept"),
                    "bloom": q.get("bloom"),
                    "test_header": q.get("test_header"),        
                    "test_description": q.get("test_description"),
                    "question": (q.get("question") or "")[:800],
                    "choices": [],
                    "answer": (q.get("answer") or "")[:200],
                    "answer_text": (q.get("answer_text") or "")[:300],
                }
                choices = q.get("choices") or []
                if isinstance(choices, list) and len(json.dumps(choices)) < 1000:
                    q_trim["choices"] = [str(c)[:300] for c in choices][:4]
                trimmed_quizzes.append(q_trim)

            quizzes_json_safe = json.dumps(trimmed_quizzes, ensure_ascii=False)
            if len(quizzes_json_safe) > MAX_QUIZZES_JSON:
                quizzes_json_safe = quizzes_json_safe[:MAX_QUIZZES_JSON]

        tos = TosRecord(
            user_id=current_user.id,
            title=title,
            topics_json=topics_json_safe,
            quizzes_json=quizzes_json_safe,
            total_items=total_quiz,
            date_created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        db.session.add(tos)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.exception("DB save error: %s", e)
        return jsonify({"error": "Database error while saving TOS. Check server logs."}), 500

    cache_stats = {}
    try:
        cache_stats = get_cache_stats() or {}
    except Exception:
        pass

    preview_html = ""
    try:
        preview_html = render_template(
            "partials/quiz_preview.html",
            title=title,
            subject_type=subject_type, 
            quizzes=quizzes_data,
            total_items=total_quiz,
            topics=valid_topics,
            fam_pct=fam_pct,
            int_pct=int_pct,
            cre_pct=cre_pct
        )
    except Exception as e:
        logger.exception("Failed to render quiz preview template: %s", e)

    return jsonify(
        {
            "title": title,
            "subject_type": subject_type,
            "master_id": tos.id,   
            "fam_pct": fam_pct,
            "int_pct": int_pct,
            "cre_pct": cre_pct,
            "totalQuiz": total_quiz,
            "totalHours": total_hours,
            "topics": valid_topics,
            "tests": tests,
            "quizzes": quizzes_data,
            "cache_stats": cache_stats,
            "preview_html": preview_html,
            "redirect_url": url_for('dashboard.index')  
        }
    )
# =========================================================
# 3b. SAVE SELECTED QUIZ ITEMS
# =========================================================
@dashboard_bp.route("/save_selected", methods=["POST"])
@login_required
@faculty_required  
def save_selected():
    data = request.get_json() or {}
    parent_id        = data.get("parent_id")
    selected_indices = data.get("selected_indices", [])  # 1-based

    if not parent_id:
        return jsonify({"error": "Missing parent record ID."}), 400
    if not selected_indices:
        return jsonify({"error": "No questions selected."}), 400

    parent = TosRecord.query.get_or_404(parent_id)
    if parent.user_id != current_user.id:
        return jsonify({"error": "Permission denied."}), 403

    try:
        all_quizzes = json.loads(parent.quizzes_json or "[]")
    except Exception:
        return jsonify({"error": "Could not load quiz data from parent record."}), 500

    # selected_indices are 1-based
    selected_quizzes = []
    for idx in selected_indices:
        zero_based = idx - 1
        if 0 <= zero_based < len(all_quizzes):
            selected_quizzes.append(all_quizzes[zero_based])

    if not selected_quizzes:
        return jsonify({"error": "None of the selected indices matched valid questions."}), 400

    try:
        new_record = TosRecord(
            user_id      = current_user.id,
            title        = parent.title + " (Selected)",
            topics_json  = parent.topics_json,
            quizzes_json = json.dumps(selected_quizzes, ensure_ascii=False),
            total_items  = len(selected_quizzes),
            date_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            is_derived   = True,          # ← ADD THIS
            parent_id    = parent.id,     # ← ADD THIS
        )
        db.session.add(new_record)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.exception("DB error in save_selected: %s", e)
        return jsonify({"error": "Database error while saving selected items."}), 500

    return jsonify({
        "total_items":   len(selected_quizzes),
        "record_id":     new_record.id,
        "redirect_url":  url_for("dashboard.index"),
    })
# =========================================================
# 4. VIEW RECORD
# 1. Change view_tos — remove @faculty_required, add admin bypass
@dashboard_bp.route("/view/<int:id>")
@login_required
def view_tos(id):
    record = TosRecord.query.get_or_404(id)

    # Admins can view any record; faculty only their own
    if not current_user.is_admin and record.user_id != current_user.id:
        flash("You do not have permission to view this.", "error")
        return redirect(url_for("dashboard.index"))

    try:
        topics_data = json.loads(record.topics_json)
    except:
        topics_data = []

    quizzes_data = []
    if hasattr(record, 'quizzes_json') and record.quizzes_json:
        try:
            quizzes_data = json.loads(record.quizzes_json)
        except:
            quizzes_data = []

    parent_record = None
    if record.is_derived and record.parent_id:
        parent_record = TosRecord.query.get(record.parent_id)

    return render_template("view_tos.html", record=record, topics=topics_data,
                           quizzes=quizzes_data, parent_record=parent_record)


# 2. Change delete_tos — same pattern, redirect admin back to admin panel
# =========================================================
# Fixed delete_tos route — paste this into routes/dashboard.py
# replacing the existing delete_tos function
# =========================================================

# 2. Change delete_tos — same pattern, redirect admin back to admin panel
# =========================================================
# Fixed delete_tos — v2
# Replace the existing delete_tos function in routes/dashboard.py
# =========================================================

# =========================================================
# Fixed delete_tos — v3
# Replace the existing delete_tos function in routes/dashboard.py
#
# Root cause: SQLAlchemy batches parent + child DELETEs into one
# statement. MySQL FK constraint fires because parent is referenced
# by children still in the same batch.
# Fix: db.session.flush() after bulk-deleting children forces the
# child DELETE SQL to execute BEFORE the parent DELETE SQL,
# satisfying the FK constraint within the same transaction.
# =========================================================

@dashboard_bp.route("/delete/<int:id>")
@login_required
def delete_tos(id):
    record = TosRecord.query.get_or_404(id)

    if not current_user.is_admin and record.user_id != current_user.id:
        flash("You do not have permission to delete this.", "error")
        return redirect(url_for("dashboard.index"))

    try:
        child_count = TosRecord.query.filter_by(parent_id=record.id).count()

        # Delete children and FLUSH immediately so their DELETE SQL hits
        # the DB before we register the parent deletion.
        TosRecord.query.filter_by(parent_id=record.id).delete(synchronize_session=False)
        db.session.flush()          # ← sends child DELETE to DB now

        db.session.delete(record)   # parent DELETE queued
        db.session.commit()         # parent DELETE sent + committed

        if child_count:
            flash(
                f"Deleted '{record.title}' and {child_count} "
                f"derived exam(s) successfully.",
                "success",
            )
        else:
            flash(f"Deleted '{record.title}' successfully.", "success")

    except Exception as e:
        db.session.rollback()
        logger.exception("delete_tos failed for record id=%d: %s", id, e)
        flash(f"Error deleting record: {e}", "error")

    if current_user.is_admin:
        return redirect(url_for("admin.records"))
    return redirect(url_for("dashboard.index")) 