# routes/dashboard.py
# CILOs are stored inside topics_json as:
#   {"_cilos": ["...", "..."], "topics": [...]}
# Old records (plain list) are handled transparently.
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
import threading
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
            return redirect(url_for("admin.index"))
        return f(*args, **kwargs)
    return decorated

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/dashboard")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOADS_DIR  = os.path.join(PROJECT_ROOT, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

MAX_FILE_BYTES = 30 * 1024 * 1024

# ============================================================
# THREAD-SAFE GENERATION PROGRESS TRACKER
# ============================================================
_gen_lock     = threading.Lock()
_gen_progress = {"current": 0, "total": 0, "active": False}

def _reset_progress():
    with _gen_lock:
        _gen_progress.update({"current": 0, "total": 0, "active": False})

def _tick_progress(current: int, total: int):
    with _gen_lock:
        _gen_progress.update({"current": current, "total": total, "active": True})


# ============================================================
# HELPER: Parse topics_json — supports both old (list) and
#         new (dict with _cilos key) formats.
# ============================================================
def _parse_topics_json(raw_json: str):
    """
    Returns (topics: list, cilos: list).
    Supports:
      - Old format: JSON array  -> cilos = []
      - New format: {"_cilos": [...], "topics": [...]}
    """
    try:
        data = json.loads(raw_json or "[]")
    except Exception:
        return [], []

    if isinstance(data, list):
        return data, []
    if isinstance(data, dict):
        topics = data.get("topics") or []
        cilos  = data.get("_cilos") or []
        return topics, cilos
    return [], []


def _dump_topics_json(topics: list, cilos: list) -> str:
    """Serialise topics + cilos into the new format."""
    return json.dumps({"_cilos": cilos, "topics": topics}, ensure_ascii=False)


# ============================================================
# EXTERNAL MODEL + PROGRESS
# ============================================================
def call_model_service(expanded_records, test_labels):
    model_url = os.getenv("AUTO_TOS_MODEL_URL") or current_app.config.get("AUTO_TOS_MODEL_URL")
    total = len(expanded_records)
    _tick_progress(0, total)

    if model_url:
        endpoint = model_url.rstrip("/") + "/generate"
        try:
            _poll_stop = threading.Event()

            def _mirror_external_progress():
                prog_url = model_url.rstrip("/") + "/progress"
                while not _poll_stop.is_set():
                    try:
                        r = requests.get(prog_url, timeout=2)
                        if r.ok:
                            d = r.json()
                            _tick_progress(d.get("current", 0), d.get("total", total))
                    except Exception:
                        pass
                    time.sleep(0.8)

            mirror_thread = threading.Thread(target=_mirror_external_progress, daemon=True)
            mirror_thread.start()

            resp = requests.post(endpoint, json={"records": expanded_records, "test_labels": test_labels}, timeout=None)

            _poll_stop.set()
            mirror_thread.join(timeout=2)

            if resp.ok:
                try:
                    data = resp.json()
                    _tick_progress(total, total)
                    return data
                except Exception as e:
                    logger.warning("Failed to parse JSON from model service: %s", e)
            else:
                logger.warning("Model service returned %d: %s", resp.status_code, resp.text[:500])
        except Exception as e:
            logger.exception("External model service call failed: %s", e)

    try:
        ai_model = importlib.import_module("ai_model")
        if hasattr(ai_model, "generate_quiz_item_by_item"):
            results = []
            for i, record in enumerate(expanded_records, start=1):
                results.append(ai_model.generate_quiz_item_by_item(record))
                _tick_progress(i, total)
            return {"quizzes": results}
        local = ai_model.generate_quiz_for_topics(expanded_records, max_items=None, test_labels=test_labels)
        _tick_progress(total, total)
        return local
    except Exception as e:
        logger.exception("Local model generation failed: %s", e)
        return None


def extract_lesson(data_or_text):
    model_url = os.getenv("AUTO_TOS_MODEL_URL") or current_app.config.get("AUTO_TOS_MODEL_URL")
    if model_url:
        endpoint = model_url.rstrip("/") + "/extract"
        try:
            resp = requests.post(endpoint, json={"data": data_or_text}, timeout=None)
            if resp.ok:
                payload = resp.json()
                if isinstance(payload, dict) and "text" in payload:
                    return payload.get("text") or ""
                if isinstance(payload, str):
                    return payload
        except Exception as e:
            logger.debug("Remote /extract failed: %s", e)
    try:
        ai_model = importlib.import_module("ai_model")
        return ai_model.lesson_from_upload(data_or_text)
    except Exception as e:
        logger.exception("Local lesson_from_upload failed: %s", e)
        return ""


def get_cache_stats():
    model_url = os.getenv("AUTO_TOS_MODEL_URL") or current_app.config.get("AUTO_TOS_MODEL_URL")
    timeout   = int(os.getenv("AUTO_TOS_MODEL_TIMEOUT", "5"))
    if model_url:
        try:
            resp = requests.get(model_url.rstrip("/") + "/cache_stats", timeout=timeout)
            if resp.ok:
                return resp.json()
        except Exception:
            pass
    try:
        ai_model = importlib.import_module("ai_model")
        if hasattr(ai_model, "get_model_cache_stats"):
            return ai_model.get_model_cache_stats()
    except Exception:
        pass
    return {}


# ============================================================
# POST-PROCESSING
# ============================================================
_TF_TASK_VERBS   = re.compile(r'^(convert|calculate|compute|list|draw|design|write|find|determine|show|give an example|describe how|explain how|create a|propose|evaluate|analyze|define|summarize|solve)\b', re.IGNORECASE)
_TF_WH_QUESTION  = re.compile(r'^(what|which|how|why|who|where|when)\b', re.IGNORECASE)

def _normalize_question(text: str) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r'[^\w\s]', '', t)
    return re.sub(r'\s+', ' ', t)

def _validate_tf_question(q: dict) -> bool:
    answer   = (q.get("answer") or "").strip().upper().rstrip('.')
    if answer not in ("TRUE", "FALSE"):
        return False
    question = (q.get("question") or "").strip()
    if _TF_TASK_VERBS.match(question) or _TF_WH_QUESTION.match(question):
        return False
    return True

def _deduplicate_quizzes(quizzes: list) -> list:
    seen, unique, dropped = set(), [], 0
    for q in quizzes:
        if not isinstance(q, dict):
            unique.append(q); continue
        fp = _normalize_question(q.get("question", ""))[:70]
        if fp in seen:
            dropped += 1; continue
        seen.add(fp); unique.append(q)
    if dropped:
        logger.info("Dedup: removed %d duplicate(s).", dropped)
    return unique

def postprocess_quizzes(quizzes: list) -> list:
    quizzes = _deduplicate_quizzes(quizzes)
    for q in quizzes:
        if not isinstance(q, dict): continue
        if (q.get("type") or "").lower() in ("tf", "truefalse", "true_false"):
            if not _validate_tf_question(q):
                q["_invalid_tf"]  = True
                q["answer_text"]  = "⚠️ This question was flagged as an invalid True/False statement. Please review or deselect it before saving."
    return quizzes


# ============================================================
# RECOMPUTE TOS FOR DERIVED EXAMS
# ============================================================
def recompute_topics_for_derived(topics_json_str: str, quizzes: list) -> list:
    BLOOM_TO_CAT = {
        "Remembering":"fam","Knowledge":"fam","Understand":"fam",
        "Applying":"int","Analyzing":"int","Apply":"int","Analyze":"int",
        "Evaluating":"cre","Creating":"cre","Evaluate":"cre","Create":"cre",
    }
    topics, _ = _parse_topics_json(topics_json_str)
    if not topics or not quizzes:
        return topics

    import copy
    topics    = copy.deepcopy(topics)
    topic_map = {t["topic"].strip().lower(): t for t in topics}

    for t in topics:
        t["fam"] = t["int"] = t["cre"] = t["items"] = t["quiz_items"] = 0
        t["fam_range"] = t["int_range"] = t["cre_range"] = ""

    for q in quizzes:
        if not isinstance(q, dict): continue
        concept = (q.get("concept") or "").strip()
        cat     = BLOOM_TO_CAT.get((q.get("bloom") or "").strip(), "fam")
        cl      = concept.lower()
        matched = None
        if cl in topic_map:
            matched = topic_map[cl]["topic"]
        else:
            for tk, tv in topic_map.items():
                if cl in tk or tk in cl:
                    matched = tv["topic"]; break
        if not matched:
            matched = topics[0]["topic"]
        for t in topics:
            if t["topic"] == matched:
                t[cat] += 1; t["items"] += 1; t["quiz_items"] += 1; break

    fc = 1
    ic = sum(t["fam"] for t in topics) + 1
    cc = sum(t["fam"] + t["int"] for t in topics) + 1
    for t in topics:
        for key, cur in [("fam", "fam"), ("int", "int"), ("cre", "cre")]:
            n = t[key]
            if key == "fam": start = fc
            elif key == "int": start = ic
            else: start = cc
            if n > 0:
                e = start + n - 1
                t[f"{key}_range"] = f"{start}-{e}" if start != e else str(start)
                if key == "fam": fc = e + 1
                elif key == "int": ic = e + 1
                else: cc = e + 1
            else:
                t[f"{key}_range"] = ""
    return topics


# ============================================================
# GENERATION PROGRESS ROUTE
# ============================================================
@dashboard_bp.route("/generation_progress")
@login_required
def generation_progress():
    model_url = os.getenv("AUTO_TOS_MODEL_URL") or current_app.config.get("AUTO_TOS_MODEL_URL")
    if model_url:
        try:
            resp = requests.get(model_url.rstrip("/") + "/progress", timeout=2)
            if resp.ok:
                return jsonify(resp.json())
        except Exception:
            pass
    with _gen_lock:
        return jsonify(dict(_gen_progress))


# ============================================================
# 1. DASHBOARD HOME
# ============================================================
@dashboard_bp.route("/")
@login_required
@faculty_required
def index():
    records = TosRecord.query.filter_by(user_id=current_user.id).order_by(TosRecord.id.desc()).all()
    for r in records:
        if hasattr(r.date_created, 'strftime'):
            r.date_created = r.date_created.strftime("%Y-%m-%d %H:%M:%S")
    return render_template("dashboard.html", records=records)


# ============================================================
# 2. CREATE PAGE
# ============================================================
@dashboard_bp.route("/create")
@login_required
@faculty_required
def create():
    return render_template("create_tos.html")


# ============================================================
# HELPERS
# ============================================================
def parse_range_string(r_str):
    indices = []
    if not r_str: return indices
    for p in r_str.split(","):
        p = p.strip()
        if "-" in p:
            try:
                s, e = p.split("-"); indices.extend(range(int(s) - 1, int(e)))
            except Exception: pass
        elif p.isdigit():
            try: indices.append(int(p) - 1)
            except Exception: pass
    return indices

def _is_large_or_data_url(s, size_threshold=5000):
    if not s: return False
    if isinstance(s, str) and s.startswith("data:"): return True
    try: return len(s) > size_threshold
    except Exception: return False

def save_data_url_to_file(data_url: str) -> str:
    if not data_url or not isinstance(data_url, str) or not data_url.startswith("data:"): return None
    try: header, encoded = data_url.split(",", 1)
    except Exception: return None
    ext = ("pdf" if "pdf" in header else "docx" if ("docx" in header or "word" in header)
           else "pptx" if ("pptx" in header or "presentation" in header)
           else "txt" if ("plain" in header or "text" in header) else "bin")
    try: decoded = base64.b64decode(encoded)
    except Exception as e: logger.warning("base64 decode failed: %s", e); return None
    if len(decoded) > MAX_FILE_BYTES: return None
    dest = os.path.join(UPLOADS_DIR, secure_filename(f"{uuid4().hex}.{ext}"))
    try:
        with open(dest, "wb") as f: f.write(decoded)
        return dest
    except Exception as e:
        logger.exception("Failed to save upload: %s", e); return None


# ============================================================
# 3. SAVE TOS
# ============================================================
@dashboard_bp.route("/save_tos", methods=["POST"])
@login_required
@faculty_required
def save_tos():
    data = request.get_json() or {}

    title        = (data.get("title") or "").strip()
    subject_type = data.get("subjectType", "nonlab")
    total_quiz   = data.get("totalQuizItems")
    topics_in    = data.get("topics", []) or []
    tests_in     = data.get("tests",  []) or []
    cilos_in     = data.get("cilos",  []) or []

    # Sanitise CILOs — keep non-empty strings, max 20, each max 500 chars
    cilos = [str(c).strip()[:500] for c in cilos_in if str(c).strip()][:20]

    fam_pct = int_pct = cre_pct = None
    if subject_type == "custom":
        try:
            fam_pct = int(data.get("fam_pct", 0))
            int_pct = int(data.get("int_pct", 0))
            cre_pct = int(data.get("cre_pct", 0))
        except Exception:
            return jsonify({"error": "Custom percentages must be integer values."}), 400
        if any(x < 0 or x > 100 for x in (fam_pct, int_pct, cre_pct)):
            return jsonify({"error": "Each custom percentage must be between 0 and 100."}), 400
        if fam_pct + int_pct + cre_pct != 100:
            return jsonify({"error": "Custom percentages must sum to exactly 100."}), 400

    if not title:       return jsonify({"error": "Missing TOS title"}), 400
    if not total_quiz:  return jsonify({"error": "Missing total quiz items"}), 400
    try:
        total_quiz = int(total_quiz)
        if total_quiz <= 0: raise ValueError
    except ValueError:
        return jsonify({"error": "Total quiz items must be a positive integer"}), 400

    # ── Validate topics ──────────────────────────────────
    valid_topics = []
    for t in topics_in:
        topic_name = (t.get("topic") or "").strip()
        try:    hours_val = int(t.get("hours") or 0)
        except: continue
        if not topic_name or hours_val <= 0: continue

        raw_lm = t.get("learn_material") or ""
        stored_lm, lm_truncated, lm_was_file, saved_file_path = "", False, False, None
        try:
            if isinstance(raw_lm, str) and raw_lm.startswith("data:"):
                saved_file_path = save_data_url_to_file(raw_lm)
                lm_was_file = True
                extracted   = extract_lesson(raw_lm)
                if extracted:
                    stored_lm = extracted[:3000]; lm_truncated = len(extracted) > 3000
            elif _is_large_or_data_url(raw_lm, 5000):
                extracted = extract_lesson(raw_lm)
                if extracted:
                    stored_lm = extracted[:3000]; lm_truncated = len(extracted) > 3000
            else:
                stored_lm = raw_lm if isinstance(raw_lm, str) else str(raw_lm or "")
                if len(stored_lm) > 3000:
                    stored_lm = stored_lm[:3000]; lm_truncated = True
        except Exception as e:
            logger.warning("learn_material error for %s: %s", topic_name, e)
            stored_lm = ""; lm_truncated = True

        valid_topics.append({
            "topic": topic_name, "hours": hours_val,
            "learn_material": stored_lm,
            "learn_material_is_truncated": lm_truncated,
            "learn_material_was_file": lm_was_file,
            "file_path": saved_file_path,
            "quiz_items": 0,
        })

    if not valid_topics:
        return jsonify({"error": "Add at least one valid topic (name + hours)"}), 400

    # ── Distribute quiz items by hours ───────────────────
    total_hours = sum(t["hours"] for t in valid_topics)
    assigned = 0
    for i, t in enumerate(valid_topics):
        if i == len(valid_topics) - 1:
            t["quiz_items"] = total_quiz - assigned
        else:
            q = round((t["hours"] / total_hours) * total_quiz)
            t["quiz_items"] = q; assigned += q
        t["items"] = t["quiz_items"]

    # ── Bloom distribution ───────────────────────────────
    if subject_type == "lab":
        fam_pct, int_pct, cre_pct = 20, 30, 50
    elif subject_type != "custom":
        fam_pct, int_pct, cre_pct = 50, 30, 20
    fam_pct, int_pct, cre_pct = int(fam_pct), int(int_pct), int(cre_pct)

    for t in valid_topics:
        items = t["items"]
        fam   = round(items * fam_pct / 100)
        inte  = round(items * int_pct / 100)
        cre   = items - (fam + inte)
        t.update({"fam": fam, "int": inte, "cre": cre,
                  "fam_range": None, "int_range": None, "cre_range": None,
                  "fam_pct": fam_pct, "int_pct": int_pct, "cre_pct": cre_pct})

    # ── Item ranges ──────────────────────────────────────
    fam_no = 1
    int_no = sum(t["fam"] for t in valid_topics) + 1
    cre_no = sum(t["fam"] + t["int"] for t in valid_topics) + 1
    for t in valid_topics:
        if t["fam"] > 0:
            s, e = fam_no, fam_no + t["fam"] - 1
            t["fam_range"] = f"{s}-{e}" if s != e else str(s); fam_no = e + 1
        if t["int"] > 0:
            s, e = int_no, int_no + t["int"] - 1
            t["int_range"] = f"{s}-{e}" if s != e else str(s); int_no = e + 1
        if t["cre"] > 0:
            s, e = cre_no, cre_no + t["cre"] - 1
            t["cre_range"] = f"{s}-{e}" if s != e else str(s); cre_no = e + 1

    # ── Validate tests ───────────────────────────────────
    tests, total_from_tests = [], 0
    allowed_types = {"mcq", "truefalse", "open_ended"}
    for t in tests_in:
        ttype = (t.get("type") or "").strip()
        try:    titems = int(t.get("items", 0))
        except: titems = 0
        desc = (t.get("description") or "").strip()
        if ttype not in allowed_types or titems <= 0: continue
        tests.append({"type": ttype, "items": titems, "description": desc})
        total_from_tests += titems
    if tests and total_from_tests != total_quiz:
        return jsonify({"error": "Test items do not match total quiz count"}), 400

    # ── Test labels ──────────────────────────────────────
    test_labels, question_types_by_slot = [], []
    if tests:
        for i, t in enumerate(tests):
            label = f"Test {i+1}"
            for _ in range(t["items"]):
                test_labels.append(label); question_types_by_slot.append(t["type"])
    else:
        test_labels            = ["Test 1"] * total_quiz
        question_types_by_slot = ["mcq"]    * total_quiz

    # ── Question slots ───────────────────────────────────
    question_slots = [None] * total_quiz
    for t in valid_topics:
        lm = t.get("learn_material"); fp = t.get("file_path"); tn = t["topic"]
        for idx in parse_range_string(t.get("fam_range", "")):
            if 0 <= idx < total_quiz: question_slots[idx] = {"topic": tn, "learn_material": lm, "file_path": fp, "bloom": "remembering"}
        for idx in parse_range_string(t.get("int_range", "")):
            if 0 <= idx < total_quiz: question_slots[idx] = {"topic": tn, "learn_material": lm, "file_path": fp, "bloom": "applying"}
        for idx in parse_range_string(t.get("cre_range", "")):
            if 0 <= idx < total_quiz: question_slots[idx] = {"topic": tn, "learn_material": lm, "file_path": fp, "bloom": "creating"}

    fb = {"topic": valid_topics[0]["topic"], "learn_material": valid_topics[0].get("learn_material"), "file_path": valid_topics[0].get("file_path"), "bloom": "remembering"}
    for i in range(total_quiz):
        if question_slots[i] is None: question_slots[i] = fb

    expanded_records = []
    for i in range(total_quiz):
        slot  = question_slots[i]
        qtype = question_types_by_slot[i] if i < len(question_types_by_slot) else "mcq"
        expanded_records.append({
            "instruction": "Generate a single exam question strictly from the provided context.",
            "input": {"concept": slot["topic"], "context": slot.get("learn_material") or "",
                      "file_path": slot.get("file_path"), "bloom": slot.get("bloom", "Remembering"), "type": qtype}
        })

    # ── Generate ─────────────────────────────────────────
    try:
        model_result = call_model_service(expanded_records, test_labels)
    except Exception as e:
        logger.exception("Generation error: %s", e); _reset_progress()
        return jsonify({"error": "Internal error generating quiz items."}), 500

    quizzes_data = []
    if isinstance(model_result, dict) and "quizzes" in model_result:
        quizzes_data = model_result.get("quizzes") or []
    elif isinstance(model_result, list):
        quizzes_data = model_result
    elif isinstance(model_result, dict):
        possible = model_result.get("results") or model_result.get("items") or []
        if isinstance(possible, list): quizzes_data = possible

    if not quizzes_data:
        _reset_progress()
        return jsonify({"error": "Failed to generate quiz items."}), 500

    # ── Inject test headers ──────────────────────────────
    desc_map = {f"Test {i+1}": t.get("description", "") for i, t in enumerate(tests)}
    for i, q in enumerate(quizzes_data):
        if isinstance(q, dict):
            hdr = test_labels[i] if i < len(test_labels) else (q.get("test_header") or "Test 1")
            q["test_header"] = hdr
            if hdr in desc_map: q["test_description"] = desc_map[hdr]

    quizzes_data = postprocess_quizzes(quizzes_data)
    test_labels  = [q.get("test_header", "Test 1") for q in quizzes_data]

    # ── Save to DB ───────────────────────────────────────
    try:
        # Store topics + cilos together in topics_json
        topics_json_safe  = _dump_topics_json(valid_topics, cilos)
        quizzes_json_safe = json.dumps(quizzes_data, ensure_ascii=False)

        MAX_TOPICS_JSON  = 60_000
        MAX_QUIZZES_JSON = 60_000

        if len(topics_json_safe) > MAX_TOPICS_JSON:
            for t in valid_topics:
                t["learn_material"] = (t.get("learn_material") or "")[:500]
                t["learn_material_is_truncated"] = True
            topics_json_safe = _dump_topics_json(valid_topics, cilos)

        if len(quizzes_json_safe) > MAX_QUIZZES_JSON:
            trimmed = []
            for q in quizzes_data:
                if not isinstance(q, dict): continue
                q_trim = {k: q.get(k) for k in ("type","concept","bloom","test_header","test_description")}
                q_trim["question"]    = (q.get("question")    or "")[:800]
                q_trim["answer"]      = (q.get("answer")      or "")[:200]
                q_trim["answer_text"] = (q.get("answer_text") or "")[:300]
                choices = q.get("choices") or []
                q_trim["choices"] = [str(c)[:300] for c in choices][:4] if isinstance(choices, list) and len(json.dumps(choices)) < 1000 else []
                trimmed.append(q_trim)
            quizzes_json_safe = json.dumps(trimmed, ensure_ascii=False)
            if len(quizzes_json_safe) > MAX_QUIZZES_JSON:
                quizzes_json_safe = quizzes_json_safe[:MAX_QUIZZES_JSON]

        tos = TosRecord(
            user_id      = current_user.id,
            title        = title,
            topics_json  = topics_json_safe,
            quizzes_json = quizzes_json_safe,
            total_items  = total_quiz,
            date_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            subject_type = subject_type,
        )
        db.session.add(tos); db.session.commit()
    except Exception as e:
        db.session.rollback(); logger.exception("DB save error: %s", e); _reset_progress()
        return jsonify({"error": "Database error while saving TOS."}), 500

    preview_html = ""
    try:
        preview_html = render_template(
            "partials/quiz_preview.html",
            title        = title,
            subject_type = subject_type,
            quizzes      = quizzes_data,
            total_items  = total_quiz,
            topics       = valid_topics,
            cilos        = cilos,
            fam_pct      = fam_pct,
            int_pct      = int_pct,
            cre_pct      = cre_pct,
        )
    except Exception as e:
        logger.exception("Failed to render quiz preview template: %s", e)

    _reset_progress()

    return jsonify({
        "title":        title,
        "subject_type": subject_type,
        "master_id":    tos.id,
        "fam_pct":      fam_pct,
        "int_pct":      int_pct,
        "cre_pct":      cre_pct,
        "totalQuiz":    total_quiz,
        "totalHours":   total_hours,
        "topics":       valid_topics,
        "cilos":        cilos,
        "tests":        tests,
        "quizzes":      quizzes_data,
        "cache_stats":  get_cache_stats() or {},
        "preview_html": preview_html,
        "redirect_url": url_for('dashboard.index'),
    })


# ============================================================
# 3b. SAVE SELECTED
# ============================================================
@dashboard_bp.route("/save_selected", methods=["POST"])
@login_required
@faculty_required
def save_selected():
    data             = request.get_json() or {}
    parent_id        = data.get("parent_id")
    selected_indices = data.get("selected_indices", [])

    if not parent_id:        return jsonify({"error": "Missing parent record ID."}), 400
    if not selected_indices: return jsonify({"error": "No questions selected."}), 400

    parent = TosRecord.query.get_or_404(parent_id)
    if parent.user_id != current_user.id:
        return jsonify({"error": "Permission denied."}), 403

    try:
        all_quizzes = json.loads(parent.quizzes_json or "[]")
    except Exception:
        return jsonify({"error": "Could not load quiz data from parent record."}), 500

    selected_quizzes = [all_quizzes[i - 1] for i in selected_indices if 0 <= i - 1 < len(all_quizzes)]
    if not selected_quizzes:
        return jsonify({"error": "None of the selected indices matched valid questions."}), 400

    # Preserve cilos from parent
    _, parent_cilos = _parse_topics_json(parent.topics_json or "[]")

    try:
        new_record = TosRecord(
            user_id      = current_user.id,
            title        = parent.title + " (Selected)",
            topics_json  = parent.topics_json,   # keeps cilos
            quizzes_json = json.dumps(selected_quizzes, ensure_ascii=False),
            total_items  = len(selected_quizzes),
            date_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            subject_type = getattr(parent, 'subject_type', 'nonlab') or 'nonlab',
            is_derived   = True,
            parent_id    = parent.id,
        )
        db.session.add(new_record); db.session.commit()
    except Exception as e:
        db.session.rollback(); logger.exception("DB error in save_selected: %s", e)
        return jsonify({"error": "Database error while saving selected items."}), 500

    return jsonify({
        "total_items":  len(selected_quizzes),
        "record_id":    new_record.id,
        "redirect_url": url_for("dashboard.index"),
    })


# ============================================================
# 4. VIEW RECORD
# ============================================================
@dashboard_bp.route("/view/<int:id>")
@login_required
def view_tos(id):
    record = TosRecord.query.get_or_404(id)
    if not current_user.is_admin and record.user_id != current_user.id:
        flash("You do not have permission to view this.", "error")
        return redirect(url_for("dashboard.index"))

    # Parse topics + CILOs from topics_json
    topics_data, cilos = _parse_topics_json(record.topics_json)

    quizzes_data = []
    if hasattr(record, 'quizzes_json') and record.quizzes_json:
        try: quizzes_data = json.loads(record.quizzes_json)
        except Exception: pass

    if record.is_derived and quizzes_data:
        topics_data = recompute_topics_for_derived(record.topics_json, quizzes_data)

    total_items = len(quizzes_data) if record.is_derived else (record.total_items or 0)

    # Always use the *configured* Bloom percentages — never back-calculate from
# actual item counts, because rounding on small item totals will skew them.
# Topics store fam_pct/int_pct/cre_pct from the original save; fall back to
# subject_type defaults only for legacy records that predate this field.
    if topics_data and topics_data[0].get("fam_pct") is not None:
        fam_pct = int(topics_data[0]["fam_pct"])
        int_pct = int(topics_data[0]["int_pct"])
        cre_pct = int(topics_data[0]["cre_pct"])
    else:
        stype = getattr(record, 'subject_type', 'nonlab') or 'nonlab'
        if stype == 'lab': fam_pct, int_pct, cre_pct = 20, 30, 50
        else:              fam_pct, int_pct, cre_pct = 50, 30, 20
        
    parent_record = None
    if record.is_derived and record.parent_id:
        parent_record = TosRecord.query.get(record.parent_id)

    return render_template(
        "view_tos.html",
        record        = record,
        topics        = topics_data,
        quizzes       = quizzes_data,
        cilos         = cilos,
        parent_record = parent_record,
        fam_pct       = fam_pct,
        int_pct       = int_pct,
        cre_pct       = cre_pct,
        total_items   = total_items,
    )


# ============================================================
# 5. DELETE RECORD
# ============================================================
@dashboard_bp.route("/delete/<int:id>")
@login_required
def delete_tos(id):
    record = TosRecord.query.get_or_404(id)
    if not current_user.is_admin and record.user_id != current_user.id:
        flash("You do not have permission to delete this.", "error")
        return redirect(url_for("dashboard.index"))
    try:
        child_count = TosRecord.query.filter_by(parent_id=record.id).count()
        TosRecord.query.filter_by(parent_id=record.id).delete(synchronize_session=False)
        db.session.flush()
        db.session.delete(record); db.session.commit()
        msg = f"Deleted '{record.title}' and {child_count} derived exam(s) successfully." if child_count else f"Deleted '{record.title}' successfully."
        flash(msg, "success")
    except Exception as e:
        db.session.rollback(); logger.exception("delete_tos failed: %s", e)
        flash(f"Error deleting record: {e}", "error")
    if current_user.is_admin:
        return redirect(url_for("admin.records"))
    return redirect(url_for("dashboard.index"))