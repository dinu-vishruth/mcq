"""
Flask blueprints (Phase 6). app.py remains the entry point and keeps its
module-level `app` instance (so vercel.json's app.py target and Procfile's
`gunicorn app:app` are unchanged); it just calls register_blueprints(app).

Endpoints are blueprint-namespaced (auth.home, teacher.teacher_dashboard,
student.student_dashboard, student.mcq_test, student.student_login); all in-code
url_for() calls use those names. Templates use hardcoded paths, so they are
untouched.
"""
from core.routes.auth import auth_bp
from core.routes.teacher import teacher_bp
from core.routes.student import student_bp
from core.routes.documents import documents_bp
from core.routes.api import api_bp


def register_blueprints(app):
    app.register_blueprint(auth_bp)
    app.register_blueprint(teacher_bp)
    app.register_blueprint(student_bp)
    app.register_blueprint(documents_bp)
    app.register_blueprint(api_bp)
