from flask import Flask, render_template
from config import Config
from extensions import db
from flask_login import LoginManager
from flask_migrate import Migrate

login_manager = LoginManager()
migrate       = Migrate()

# ── Upload size caps ──────────────────────────────────────────────────
# Files arrive in the JSON body as base64 (~33% larger than raw bytes).
# 100 MB raw file ≈ 133 MB base64. We allow 200 MB total request size to
# accommodate multi-file uploads. Without these, Flask returns 413
# "Request Entity Too Large" before our route runs.
MAX_REQUEST_BYTES = 200 * 1024 * 1024   # 200 MB total request


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # ── Big uploads ──
    app.config["MAX_CONTENT_LENGTH"]    = MAX_REQUEST_BYTES
    app.config["MAX_FORM_MEMORY_SIZE"]  = MAX_REQUEST_BYTES   # Werkzeug ≥ 3.1
    app.config["MAX_FORM_PARTS"]        = 1000

    db.init_app(app)
    migrate.init_app(app, db)

    login_manager.init_app(app)
    login_manager.login_view = "auth.login"

    from routes.auth      import auth_bp
    from routes.dashboard import dashboard_bp
    from routes.admin     import admin_bp        # ← NEW

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(admin_bp)             # ← NEW

    @app.route("/")
    def home():
        return render_template("login.html")

    return app


@login_manager.user_loader
def load_user(user_id):
    from models import User
    return User.query.get(int(user_id))


app = create_app()


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)