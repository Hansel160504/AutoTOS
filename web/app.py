from flask import Flask, render_template
from config import Config
from extensions import db
from flask_login import LoginManager
from flask_migrate import Migrate

login_manager = LoginManager()
migrate       = Migrate()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

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
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)  # ← add threaded=True