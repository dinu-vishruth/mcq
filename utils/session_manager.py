# utils/session_manager.py
"""
Session-key helpers. Public API preserved for app.py and any callers:
    create_session_key(teacher, difficulty, timer, mcqs) -> str
    validate_session_key(key) -> bool
The bodies now delegate to app.repositories.session_repo so all session SQL
lives in one place.
"""
from core.repositories import session_repo


def create_session_key(teacher, difficulty, timer, mcqs):
    return session_repo.create_session(teacher, difficulty, timer, mcqs)


def validate_session_key(key):
    return session_repo.exists(key)
