"""
Session data access. Backs utils.session_manager (whose public functions are
preserved) and the RAG pipeline, which needs to link a session to the
document it was generated from (sessions.document_id, added in Phase 2).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime

from core.models.db import get_db
from core.models.questions import validate_questions


def create_session(teacher, difficulty, timer, mcqs, document_id=None, *, owner_id=None) -> str:
    valid, rejected = validate_questions(mcqs, trusted_metadata=True)
    if rejected or not valid:
        raise ValueError("The generated quiz did not pass question validation. Please retry.")
    mcqs = valid
    key = uuid.uuid4().hex
    conn = get_db()
    try:
        if owner_id is None:
            user = conn.execute("SELECT id FROM users WHERE username=?", (teacher,)).fetchone()
            owner_id = user["id"] if user else None
        conn.execute(
            "INSERT INTO sessions (session_key, teacher, teacher_user_id, created_at, difficulty, timer, mcqs_json, document_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (key, teacher, owner_id, datetime.utcnow().isoformat(), difficulty, timer, json.dumps(mcqs), document_id),
        )
        conn.commit()
    finally:
        conn.close()
    return key


def exists(session_key) -> bool:
    conn = get_db()
    try:
        return conn.execute("SELECT 1 FROM sessions WHERE session_key=?", (session_key,)).fetchone() is not None
    finally:
        conn.close()


def get(session_key):
    conn = get_db()
    try:
        return conn.execute("SELECT * FROM sessions WHERE session_key=?", (session_key,)).fetchone()
    finally:
        conn.close()
