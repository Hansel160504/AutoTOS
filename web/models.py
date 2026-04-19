from extensions import db
from flask_login import UserMixin
from datetime import datetime


class User(db.Model, UserMixin):
    """
    Faculty / Admin user.

    Flask-Login's UserMixin already declares an `is_active` property that
    returns True. We store the column as `active` in the DB and expose it
    as `is_active` via a Python property so Flask-Login works correctly
    without SQLAlchemy column-vs-property conflicts.
    """
    __tablename__ = "users"

    id           = db.Column(db.Integer, primary_key=True)
    name         = db.Column(db.String(120), nullable=False)
    username     = db.Column(db.String(80), unique=True, nullable=False)  # ← replaces email
    password     = db.Column(db.String(256), nullable=False)

    is_admin     = db.Column(db.Boolean, default=False, nullable=False)

    # Stored as `is_active` in DB; accessed via property below to avoid
    # Flask-Login UserMixin clash.
    active       = db.Column("is_active", db.Boolean, default=True, nullable=False)

    date_created = db.Column(
        db.String(50),
        default=lambda: datetime.now().strftime("%Y-%m-%d")
    )

    # ── Flask-Login integration ──
    @property
    def is_active(self):
        return self.active

    @is_active.setter
    def is_active(self, value):
        self.active = value

    def get_id(self):
        return str(self.id)

    def __repr__(self):
        return f"<User {self.username} admin={self.is_admin} active={self.active}>"



class TosRecord(db.Model):  
    """
    A generated Table of Specification (TOS) record.

    Master records hold the full question bank.
    Derived records are subsets saved via 'Save Selected'.
    """
    __tablename__ = "tos_records"

    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    title        = db.Column(db.String(255), nullable=False)
    topics_json  = db.Column(db.Text, default="[]")
    quizzes_json = db.Column(db.Text, default="[]")
    total_items  = db.Column(db.Integer, default=0)
    date_created = db.Column(db.String(50))
    subject_type = db.Column(db.String(20), default='nonlab')  # ← ADD THIS
    # ── Master / Derived relationship ──
    is_derived   = db.Column(db.Boolean, default=False, nullable=False)
    parent_id    = db.Column(db.Integer, db.ForeignKey("tos_records.id"), nullable=True)

    def __repr__(self):
        return f"<TosRecord #{self.id} '{self.title}' derived={self.is_derived}>"