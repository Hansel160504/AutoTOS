from flask import Blueprint, render_template, request, redirect, url_for, flash
from werkzeug.security import check_password_hash, generate_password_hash
from flask_login import login_user, logout_user, login_required, current_user
from models import User
from extensions import db
from datetime import datetime

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        if current_user.is_admin:
            return redirect(url_for("admin.index"))
        return redirect(url_for("dashboard.index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()

        if not user or not check_password_hash(user.password, password):
            flash("Invalid username or password.", "error")
            return redirect(url_for("auth.login"))

        if not user.active:
            flash("Your account has been deactivated. Please contact an administrator.", "error")
            return redirect(url_for("auth.login"))

        login_user(user)

        if user.is_admin:
            return redirect(url_for("admin.index"))
        return redirect(url_for("dashboard.index"))

    return render_template("login.html")


@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        if current_user.is_admin:
            return redirect(url_for("admin.index"))
        return redirect(url_for("dashboard.index"))

    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not username or not password:
            flash("All fields are required.", "error")
            return redirect(url_for("auth.signup"))

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return redirect(url_for("auth.signup"))

        existing = User.query.filter_by(username=username).first()
        if existing:
            flash("Username already taken.", "error")
            return redirect(url_for("auth.signup"))

        # create new user here
        user = User(
            name         = name,
            username=username,
            password     = generate_password_hash(password),
            is_admin     = False,
            active       = True,
            date_created = datetime.now().strftime("%Y-%m-%d"),
        )
        db.session.add(user)
        db.session.commit()

        flash("Account created! You can now sign in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("signup.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))