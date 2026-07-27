"""
Session data access. Backs utils.session_manager (whose public functions are
preserved) and the Phase 4 pipeline, which needs to link a session to the
document it was generated from (sessions.document_id, added in Phase 2).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime

from core.models.db import get_db


def create_session(teacher, difficulty, timer, mcqs, document_id=None) -> str:
    key = str(uuid.uuid4())[:8]
    conn = get_db()
    try:
        # document_id column exists only after the Phase 2 migration; fall back
        # to the legacy insert if it isn't there yet.
        try:
            conn.execute(
                "INSERT INTO sessions (session_key, teacher, created_at, difficulty, timer, mcqs_json, document_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (key, teacher, datetime.utcnow().isoformat(), difficulty, timer, json.dumps(mcqs), document_id),
            )
        except Exception:
            conn.execute(
                "INSERT INTO sessions (session_key, teacher, created_at, difficulty, timer, mcqs_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (key, teacher, datetime.utcnow().isoformat(), difficulty, timer, json.dumps(mcqs)),
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
