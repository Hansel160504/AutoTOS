# web/routes/admin.py
from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, current_app
from flask_login import login_required, current_user
from functools import wraps
from extensions import db
from models import TosRecord, User
from datetime import datetime, timedelta
import os
import importlib
import requests
import logging

logger = logging.getLogger(__name__)

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


# =========================================================
# ADMIN GUARD
# =========================================================
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for("auth.login"))
        if current_user.is_admin is not True:
            # Don't send to dashboard — that causes a redirect loop.
            # Send non-admins back to login with a message.
            flash("Admin access required.", "error")
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated


# =========================================================
# HELPERS
# =========================================================
def _get_model_stats():
    model_url = os.getenv("AUTO_TOS_MODEL_URL") or current_app.config.get("AUTO_TOS_MODEL_URL")
    timeout   = int(os.getenv("AUTO_TOS_MODEL_TIMEOUT", "4"))
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


def _get_model_health():
    model_url = os.getenv("AUTO_TOS_MODEL_URL") or current_app.config.get("AUTO_TOS_MODEL_URL")
    if not model_url:
        return {"status": "local", "url": "local fallback"}
    try:
        resp = requests.get(model_url.rstrip("/") + "/health", timeout=4)
        if resp.ok:
            data           = resp.json()
            data["url"]    = model_url
            data["status"] = "online"
            return data
        return {"status": "error", "url": model_url, "code": resp.status_code}
    except Exception as e:
        return {"status": "offline", "url": model_url, "error": str(e)}


class _RowWrapper:
    """
    Wraps a SQLAlchemy KeyedTuple row and attaches a .children list.
    Allows templates to do: row[0] for the record, row.user_name, row.children
    """
    def __init__(self, row, children):
        self._row     = row
        self.children = children

    def __getitem__(self, idx):
        return self._row[idx]

    def __getattr__(self, name):
        return getattr(self._row, name)


def _attach_children(master_rows):
    """
    Fetch all derived records whose parent_id matches any master in master_rows,
    then attach them as .children on each row wrapper.
    """
    master_ids = [row[0].id for row in master_rows]
    if not master_ids:
        return [_RowWrapper(r, []) for r in master_rows]

    derived = (
        TosRecord.query
        .filter(TosRecord.parent_id.in_(master_ids), TosRecord.is_derived == True)
        .order_by(TosRecord.id.asc())
        .all()
    )
    children_map = {}
    for d in derived:
        children_map.setdefault(d.parent_id, []).append(d)

    return [_RowWrapper(row, children_map.get(row[0].id, [])) for row in master_rows]


# =========================================================
# 1. ADMIN DASHBOARD
# =========================================================
@admin_bp.route("/")
@login_required
@admin_required
def index():
    total_users  = User.query.count()
    active_users = User.query.filter(User.active == True).count()
    admin_users  = User.query.filter_by(is_admin=True).count()

    week_ago         = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
    new_users_week   = User.query.filter(User.date_created >= week_ago).count()

    master_records  = TosRecord.query.filter_by(is_derived=False).count()
    derived_records = TosRecord.query.filter_by(is_derived=True).count()
    total_records   = master_records
    total_items     = db.session.query(db.func.sum(TosRecord.total_items)).scalar() or 0
    new_records_week = TosRecord.query.filter(
        TosRecord.is_derived == False,
        TosRecord.date_created >= week_ago
    ).count()

    # Recent masters (with user info) — attach derived children
    recent_rows = (
        TosRecord.query
        .filter_by(is_derived=False)
        .join(User, TosRecord.user_id == User.id)
        .add_columns(User.name.label("user_name"), User.username.label("user_email"))
        .order_by(TosRecord.id.desc())
        .limit(8)
        .all()
    )
    recent_records = _attach_children(recent_rows)

    top_users = (
        db.session.query(
            User.id, User.name, User.username,
            db.func.count(TosRecord.id).label("record_count"),
            db.func.sum(TosRecord.total_items).label("total_items"),
        )
        .join(TosRecord, TosRecord.user_id == User.id, isouter=True)
        .group_by(User.id)
        .order_by(db.text("record_count DESC"))
        .limit(5)
        .all()
    )

    model_health = _get_model_health()
    cache_stats  = _get_model_stats()

    return render_template(
        "admin/dashboard.html",
        total_users=total_users,
        active_users=active_users,
        admin_users=admin_users,
        new_users_week=new_users_week,
        total_records=total_records,
        master_records=master_records,
        derived_records=derived_records,
        total_items=total_items,
        new_records_week=new_records_week,
        recent_records=recent_records,
        top_users=top_users,
        model_health=model_health,
        cache_stats=cache_stats,
    )


# =========================================================
# 2. USER MANAGEMENT
# =========================================================
@admin_bp.route("/users")
@login_required
@admin_required
def users():
    all_users = (
        User.query
        .outerjoin(TosRecord, TosRecord.user_id == User.id)
        .add_columns(
            db.func.count(TosRecord.id).label("record_count"),
            db.func.sum(TosRecord.total_items).label("total_items"),
        )
        .group_by(User.id)
        .order_by(User.id.desc())
        .all()
    )
    return render_template("admin/users.html", all_users=all_users)


@admin_bp.route("/users/<int:user_id>/toggle_active", methods=["POST"])
@login_required
@admin_required
def toggle_active(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        return jsonify({"error": "Cannot deactivate your own account."}), 400
    user.active = not user.active
    db.session.commit()
    state = "activated" if user.active else "deactivated"
    return jsonify({"success": True, "is_active": user.active, "message": f"User {state}."})


@admin_bp.route("/users/<int:user_id>/toggle_admin", methods=["POST"])
@login_required
@admin_required
def toggle_admin(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        return jsonify({"error": "Cannot change your own admin status."}), 400
    user.is_admin = not user.is_admin
    db.session.commit()
    role = "Admin" if user.is_admin else "User"
    return jsonify({"success": True, "is_admin": user.is_admin, "message": f"Role set to {role}."})


@admin_bp.route("/users/<int:user_id>/delete", methods=["POST"])
@login_required
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash("Cannot delete your own account.", "error")
        return redirect(url_for("admin.users"))
    try:
        TosRecord.query.filter_by(user_id=user.id).delete()
        db.session.delete(user)
        db.session.commit()
        flash(f"User '{user.name}' and all their records deleted.", "success")
    except Exception as e:
        db.session.rollback()
        logger.exception("Error deleting user %d: %s", user_id, e)
        flash("Error deleting user.", "error")
    return redirect(url_for("admin.users"))


@admin_bp.route("/users/<int:user_id>/reset_password", methods=["POST"])
@login_required
@admin_required
def reset_password(user_id):
    from werkzeug.security import generate_password_hash
    user     = User.query.get_or_404(user_id)
    new_pass = request.get_json(force=True).get("password", "").strip()
    if len(new_pass) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400
    user.password = generate_password_hash(new_pass)
    db.session.commit()
    return jsonify({"success": True, "message": f"Password reset for {user.name}."})


# =========================================================
# 3. ALL TOS RECORDS — masters only, children nested
# =========================================================
@admin_bp.route("/records")
@login_required
@admin_required
def records():
    page     = request.args.get("page", 1, type=int)
    per_page = 15
    search   = request.args.get("q", "").strip()

    query = (
        TosRecord.query
        .filter_by(is_derived=False)          # masters only
        .join(User, TosRecord.user_id == User.id)
        .add_columns(User.name.label("user_name"), User.username.label("user_email"))
        .order_by(TosRecord.id.desc())
    )

    if search:
        query = query.filter(
            db.or_(
                TosRecord.title.ilike(f"%{search}%"),
                User.name.ilike(f"%{search}%"),
                User.email.ilike(f"%{search}%"),
            )
        )

    paginated       = query.paginate(page=page, per_page=per_page, error_out=False)
    paginated.items = _attach_children(paginated.items)

    return render_template("admin/records.html", paginated=paginated, search=search)


@admin_bp.route("/records/<int:record_id>/delete", methods=["POST"])
@login_required
@admin_required
def delete_record(record_id):
    record = TosRecord.query.get_or_404(record_id)
    try:
        TosRecord.query.filter_by(parent_id=record_id).delete()
        db.session.delete(record)
        db.session.commit()
        flash(f"Record '{record.title}' deleted.", "success")
    except Exception as e:
        db.session.rollback()
        logger.exception("Error deleting record %d: %s", record_id, e)
        flash("Error deleting record.", "error")
    return redirect(url_for("admin.records"))


# =========================================================
# 4. SYSTEM STATS (live JSON)
# =========================================================
@admin_bp.route("/system_stats")
@login_required
@admin_required
def system_stats():
    return jsonify({
        "model_health":  _get_model_health(),
        "cache_stats":   _get_model_stats(),
        "total_users":   User.query.count(),
        "total_records": TosRecord.query.filter_by(is_derived=False).count(),
        "total_items":   db.session.query(db.func.sum(TosRecord.total_items)).scalar() or 0,
    })