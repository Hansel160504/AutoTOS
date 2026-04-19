# routes/dashboard.py — slim route handlers.
#
# Business logic lives in services/tos_processor.py and services/docx_builder.py.
# External-AI communication lives in services/external_ai.py.
# Bloom mappings come from services/bloom.py.
#
# This file only deals with: auth, request parsing, DB access, and rendering.

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Tuple

from flask import (
    Blueprint, current_app, flash, jsonify, redirect, render_template,
    request, send_file, url_for,
)
from flask_login import current_user, login_required
from werkzeug.security import check_password_hash, generate_password_hash

from extensions import db
from models import TosRecord, User

from services.bloom import defaults_for
from services.docx_builder import build_docx
from services.external_ai import (
    call_model_service,
    fetch_remote_progress,
    get_cache_stats,
    progress,
)
from services.tos_processor import (
    PreparedTopic,
    ValidationError,
    apply_bloom_distribution,
    build_question_slots,
    build_test_labels,
    compute_item_ranges,
    distribute_quiz_items,
    dump_topics_json,
    extract_bloom_percentages,
    parse_topics_json,
    postprocess_quizzes,
    prepare_persisted_quizzes_json,
    prepare_persisted_topics_json,
    recompute_topics_for_derived,
    sanitise_cilos,
    validate_basic,
    validate_percentages,
    validate_tests,
    validate_topics,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/dashboard")

# ── Paths (consistent with docker-compose volume mount) ──────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Decorators
# ──────────────────────────────────────────────────────────────────────
def faculty_required(f):
    """Redirect admins away from faculty-only pages."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if current_user.is_authenticated and current_user.is_admin is True:
            return redirect(url_for("admin.index"))
        return f(*args, **kwargs)
    return wrapper


# ──────────────────────────────────────────────────────────────────────
# Authorisation helpers
# ──────────────────────────────────────────────────────────────────────
def _require_owner_or_admin(record: TosRecord) -> bool:
    return current_user.is_admin or record.user_id == current_user.id


# ──────────────────────────────────────────────────────────────────────
# 1. Home
# ──────────────────────────────────────────────────────────────────────
@dashboard_bp.route("/")
@login_required
@faculty_required
def index():
    records = (
        TosRecord.query
        .filter_by(user_id=current_user.id)
        .order_by(TosRecord.id.desc())
        .all()
    )
    for r in records:
        if hasattr(r.date_created, "strftime"):
            r.date_created = r.date_created.strftime("%Y-%m-%d %H:%M:%S")
    return render_template("dashboard.html", records=records)


# ──────────────────────────────────────────────────────────────────────
# 2. Create form
# ──────────────────────────────────────────────────────────────────────
@dashboard_bp.route("/create")
@login_required
@faculty_required
def create():
    return render_template("create_tos.html")


# ──────────────────────────────────────────────────────────────────────
# 3. Save TOS + generate quizzes
# ──────────────────────────────────────────────────────────────────────
@dashboard_bp.route("/save_tos", methods=["POST"])
@login_required
@faculty_required
def save_tos():
    data = request.get_json() or {}

    # ── Parse + validate input ────────────────────────────────
    title = (data.get("title") or "").strip()
    subject_type = data.get("subjectType", "nonlab")
    cilos = sanitise_cilos(data.get("cilos", []) or [])

    try:
        total_quiz = validate_basic(title, data.get("totalQuizItems"))
        fam_pct, int_pct, cre_pct = validate_percentages(subject_type, data)
    except ValidationError as exc:
        return jsonify({"error": str(exc)}), 400

    topics = validate_topics(data.get("topics", []) or [], UPLOADS_DIR)
    if not topics:
        return jsonify({"error": "Add at least one valid topic (name + hours)"}), 400

    # ── Compute quiz counts + Bloom distribution + ranges ─────
    distribute_quiz_items(topics, total_quiz)
    apply_bloom_distribution(topics, fam_pct, int_pct, cre_pct)
    compute_item_ranges(topics)

    # ── Tests ─────────────────────────────────────────────────
    try:
        tests = validate_tests(data.get("tests", []) or [], total_quiz)
    except ValidationError as exc:
        return jsonify({"error": str(exc)}), 400

    test_labels, question_types = build_test_labels(tests, total_quiz)

    # ── Build records + generate ──────────────────────────────
    expanded_records = build_question_slots(topics, total_quiz, question_types)

    try:
        model_result = call_model_service(expanded_records, test_labels)
    except Exception as exc:
        logger.exception("Generation error: %s", exc)
        progress.reset()
        return jsonify({"error": "Internal error generating quiz items."}), 500

    quizzes = _extract_quizzes_from_result(model_result)
    if not quizzes:
        progress.reset()
        return jsonify({"error": "Failed to generate quiz items."}), 500

    # ── Attach test headers + postprocess ─────────────────────
    desc_map = {f"Test {i+1}": t.get("description", "") for i, t in enumerate(tests)}
    for i, q in enumerate(quizzes):
        if not isinstance(q, dict):
            continue
        header = test_labels[i] if i < len(test_labels) else q.get("test_header") or "Test 1"
        q["test_header"] = header
        if header in desc_map:
            q["test_description"] = desc_map[header]

    quizzes = postprocess_quizzes(quizzes)

    # ── Persist ───────────────────────────────────────────────
    topic_dicts = [t.to_dict() for t in topics]
    try:
        tos = TosRecord(
            user_id=current_user.id,
            title=title,
            topics_json=prepare_persisted_topics_json(topic_dicts, cilos),
            quizzes_json=prepare_persisted_quizzes_json(quizzes),
            total_items=total_quiz,
            date_created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            subject_type=subject_type,
        )
        db.session.add(tos)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        logger.exception("DB save error: %s", exc)
        progress.reset()
        return jsonify({"error": "Database error while saving TOS."}), 500

    # ── Build preview (best-effort; failure non-fatal) ────────
    preview_html = _render_preview(
        title=title, subject_type=subject_type, quizzes=quizzes,
        total_quiz=total_quiz, topics=topic_dicts, cilos=cilos,
        fam_pct=fam_pct, int_pct=int_pct, cre_pct=cre_pct, master_id=tos.id,
    )

    progress.reset()

    return jsonify({
        "title":        title,
        "subject_type": subject_type,
        "master_id":    tos.id,
        "fam_pct":      fam_pct,
        "int_pct":      int_pct,
        "cre_pct":      cre_pct,
        "totalQuiz":    total_quiz,
        "totalHours":   sum(t.hours for t in topics),
        "topics":       topic_dicts,
        "cilos":        cilos,
        "tests":        tests,
        "quizzes":      quizzes,
        "cache_stats":  get_cache_stats() or {},
        "preview_html": preview_html,
        "redirect_url": url_for("dashboard.index"),
    })


def _extract_quizzes_from_result(result: Any) -> List[dict]:
    """Coerce the model service's response into a list of quiz dicts."""
    if isinstance(result, dict):
        if "quizzes" in result:
            return result.get("quizzes") or []
        maybe = result.get("results") or result.get("items")
        if isinstance(maybe, list):
            return maybe
    if isinstance(result, list):
        return result
    return []


def _render_preview(**ctx) -> str:
    try:
        return render_template("partials/quiz_preview.html", **ctx)
    except Exception as exc:
        logger.exception("Failed to render quiz preview template: %s", exc)
        return ""


# ──────────────────────────────────────────────────────────────────────
# 3b. Save selected subset
# ──────────────────────────────────────────────────────────────────────
@dashboard_bp.route("/save_selected", methods=["POST"])
@login_required
@faculty_required
def save_selected():
    data = request.get_json() or {}
    parent_id = data.get("parent_id")
    selected = data.get("selected_indices") or []

    if not parent_id:
        return jsonify({"error": "Missing parent record ID."}), 400
    if not selected:
        return jsonify({"error": "No questions selected."}), 400

    parent = TosRecord.query.get_or_404(parent_id)
    if parent.user_id != current_user.id:
        return jsonify({"error": "Permission denied."}), 403

    try:
        all_quizzes = json.loads(parent.quizzes_json or "[]")
    except json.JSONDecodeError:
        return jsonify({"error": "Could not load quiz data from parent record."}), 500

    picked = [
        all_quizzes[i - 1]
        for i in selected
        if 0 <= i - 1 < len(all_quizzes)
    ]
    if not picked:
        return jsonify({"error": "None of the selected indices matched valid questions."}), 400

    try:
        new_record = TosRecord(
            user_id=current_user.id,
            title=parent.title + " (Selected)",
            topics_json=parent.topics_json,
            quizzes_json=json.dumps(picked, ensure_ascii=False),
            total_items=len(picked),
            date_created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            subject_type=getattr(parent, "subject_type", "nonlab") or "nonlab",
            is_derived=True,
            parent_id=parent.id,
        )
        db.session.add(new_record)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        logger.exception("DB error in save_selected: %s", exc)
        return jsonify({"error": "Database error while saving selected items."}), 500

    return jsonify({
        "total_items":  len(picked),
        "record_id":    new_record.id,
        "redirect_url": url_for("dashboard.index"),
    })


# ──────────────────────────────────────────────────────────────────────
# 4. View record
# ──────────────────────────────────────────────────────────────────────
def _load_for_display(record: TosRecord) -> Tuple[List[dict], List[str], List[dict], int, Tuple[int, int, int]]:
    """Return (topics, cilos, quizzes, total_items, (fam, int, cre))."""
    topics, cilos = parse_topics_json(record.topics_json)

    quizzes: List[dict] = []
    if record.quizzes_json:
        try:
            quizzes = json.loads(record.quizzes_json)
        except json.JSONDecodeError:
            pass

    if record.is_derived and quizzes:
        topics = recompute_topics_for_derived(record.topics_json, quizzes)

    total_items = len(quizzes) if record.is_derived else (record.total_items or 0)
    fam, int_, cre = extract_bloom_percentages(
        topics, getattr(record, "subject_type", "nonlab") or "nonlab",
    )
    return topics, cilos, quizzes, total_items, (fam, int_, cre)


@dashboard_bp.route("/view/<int:id>")
@login_required
def view_tos(id: int):
    record = TosRecord.query.get_or_404(id)
    if not _require_owner_or_admin(record):
        flash("You do not have permission to view this.", "error")
        return redirect(url_for("dashboard.index"))

    topics, cilos, quizzes, total_items, (fam_pct, int_pct, cre_pct) = _load_for_display(record)

    parent_record = None
    if record.is_derived and record.parent_id:
        parent_record = TosRecord.query.get(record.parent_id)

    return render_template(
        "view_tos.html",
        record=record,
        topics=topics,
        quizzes=quizzes,
        cilos=cilos,
        parent_record=parent_record,
        fam_pct=fam_pct,
        int_pct=int_pct,
        cre_pct=cre_pct,
        total_items=total_items,
    )


# ──────────────────────────────────────────────────────────────────────
# 5. Download DOCX
# ──────────────────────────────────────────────────────────────────────
_FILENAME_SAFE_RE = re.compile(r"[^\w\s-]")


def _safe_filename(title: str) -> str:
    clean = _FILENAME_SAFE_RE.sub("", title or "").strip().replace(" ", "_")[:60]
    return f"{clean or 'TOS'}.docx"


@dashboard_bp.route("/download_docx/<int:id>")
@login_required
def download_docx(id: int):
    record = TosRecord.query.get_or_404(id)
    if not _require_owner_or_admin(record):
        flash("You do not have permission to download this.", "error")
        return redirect(url_for("dashboard.index"))

    topics, cilos, quizzes, total_items, (fam_pct, int_pct, cre_pct) = _load_for_display(record)

    try:
        buf = build_docx(
            title=record.title, cilos=cilos, topics=topics,
            quizzes=quizzes, fam_pct=fam_pct, int_pct=int_pct,
            cre_pct=cre_pct, total_items=total_items,
        )
    except Exception as exc:
        logger.exception("DOCX generation failed: %s", exc)
        flash("Failed to generate DOCX file.", "error")
        return redirect(url_for("dashboard.view_tos", id=id))

    return send_file(
        buf, as_attachment=True, download_name=_safe_filename(record.title),
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


# ──────────────────────────────────────────────────────────────────────
# 6. Delete
# ──────────────────────────────────────────────────────────────────────
@dashboard_bp.route("/delete/<int:id>")
@login_required
def delete_tos(id: int):
    record = TosRecord.query.get_or_404(id)
    if not _require_owner_or_admin(record):
        flash("You do not have permission to delete this.", "error")
        return redirect(url_for("dashboard.index"))

    try:
        child_count = TosRecord.query.filter_by(parent_id=record.id).count()
        TosRecord.query.filter_by(parent_id=record.id).delete(synchronize_session=False)
        db.session.flush()
        db.session.delete(record)
        db.session.commit()

        msg = (
            f"Deleted '{record.title}' and {child_count} derived exam(s) successfully."
            if child_count else
            f"Deleted '{record.title}' successfully."
        )
        flash(msg, "success")
    except Exception as exc:
        db.session.rollback()
        logger.exception("delete_tos failed: %s", exc)
        flash(f"Error deleting record: {exc}", "error")

    target = "admin.records" if current_user.is_admin else "dashboard.index"
    return redirect(url_for(target))


# ──────────────────────────────────────────────────────────────────────
# 7. Generation progress (remote-first, local fallback)
# ──────────────────────────────────────────────────────────────────────
@dashboard_bp.route("/generation_progress")
@login_required
def generation_progress():
    remote = fetch_remote_progress()
    if remote is not None:
        return jsonify(remote)
    return jsonify(progress.snapshot())


# ──────────────────────────────────────────────────────────────────────
# 8. Profile — view
# ──────────────────────────────────────────────────────────────────────
@dashboard_bp.route("/profile", methods=["GET"])
@login_required
@faculty_required
def profile():
    """Render the user's profile page with TOS stats."""
    user_id = current_user.id

    # ── Aggregate TOS stats for this user ──
    user_records = TosRecord.query.filter_by(user_id=user_id).all()
    master_count  = sum(1 for r in user_records if not r.is_derived)
    derived_count = sum(1 for r in user_records if r.is_derived)
    total_items   = sum(
        (r.total_items or 0)
        for r in user_records
        if not r.is_derived  # only count master items, derived are subsets
    )

    # ── Joined date (date_created is stored as a 'YYYY-MM-DD' string) ──
    joined_on = current_user.date_created or "—"
    try:
        joined_on = datetime.strptime(joined_on[:10], "%Y-%m-%d").strftime("%B %d, %Y")
    except (ValueError, TypeError):
        pass

    stats = {
        "master_count":  master_count,
        "derived_count": derived_count,
        "total_items":   total_items,
        "joined_on":     joined_on,
    }

    return render_template("profile.html", stats=stats)


# ──────────────────────────────────────────────────────────────────────
# 9. Profile — update name
# ──────────────────────────────────────────────────────────────────────
@dashboard_bp.route("/profile/update", methods=["POST"])
@login_required
@faculty_required
def update_profile():
    """Update the user's display name. Username is immutable."""
    name = (request.form.get("name") or "").strip()

    # ── Validation ──
    if not name:
        flash("Name cannot be empty.", "error")
        return redirect(url_for("dashboard.profile"))

    if len(name) > 120:
        flash("Name is too long (max 120 characters).", "error")
        return redirect(url_for("dashboard.profile"))

    if name == current_user.name:
        flash("No changes to save.", "error")
        return redirect(url_for("dashboard.profile"))

    # ── Update ──
    try:
        user = User.query.get(current_user.id)
        user.name = name
        db.session.commit()
        flash("Profile updated successfully.", "success")
    except Exception as exc:
        db.session.rollback()
        logger.exception("Profile update failed: %s", exc)
        flash("Could not update profile. Please try again.", "error")

    return redirect(url_for("dashboard.profile"))


# ──────────────────────────────────────────────────────────────────────
# 10. Profile — change password
# ──────────────────────────────────────────────────────────────────────
@dashboard_bp.route("/profile/change-password", methods=["POST"])
@login_required
@faculty_required
def change_password():
    """Verify current password, then update to new password."""
    current_pw = request.form.get("current_password") or ""
    new_pw     = request.form.get("new_password") or ""
    confirm_pw = request.form.get("confirm_new_password") or ""

    # ── Validation ──
    if not current_pw or not new_pw or not confirm_pw:
        flash("All password fields are required.", "error")
        return redirect(url_for("dashboard.profile"))

    if new_pw != confirm_pw:
        flash("New password and confirmation do not match.", "error")
        return redirect(url_for("dashboard.profile"))

    if len(new_pw) < 6:
        flash("New password must be at least 6 characters.", "error")
        return redirect(url_for("dashboard.profile"))

    # ── Verify current password (uses werkzeug — match your signup) ──
    if not check_password_hash(current_user.password, current_pw):
        flash("Current password is incorrect.", "error")
        return redirect(url_for("dashboard.profile"))

    if new_pw == current_pw:
        flash("New password must be different from current password.", "error")
        return redirect(url_for("dashboard.profile"))

    # ── Update ──
    try:
        user = User.query.get(current_user.id)
        user.password = generate_password_hash(new_pw)
        db.session.commit()
        flash("Password updated successfully.", "success")
    except Exception as exc:
        db.session.rollback()
        logger.exception("Password change failed: %s", exc)
        flash("Could not update password. Please try again.", "error")

    return redirect(url_for("dashboard.profile"))