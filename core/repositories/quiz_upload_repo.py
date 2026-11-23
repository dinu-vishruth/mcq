"""Temporary quiz input, separate from the user's saved Library resources."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from core.models.db import get_db


def create(user_id: int, title: str, text: str) -> str:
    upload_id = uuid.uuid4().hex
    now = datetime.utcnow()
    conn = get_db()
    try:
        # Expired inputs are discarded on the next upload. They are never
        # listed in the Library or copied to original-file/object storage.
        conn.execute("DELETE FROM quiz_uploads WHERE expires_at <= ?", (now.isoformat(),))
        conn.execute(
            "INSERT INTO quiz_uploads (id, user_id, title, content, expires_at) VALUES (?, ?, ?, ?, ?)",
            (upload_id, user_id, title, text, (now + timedelta(hours=24)).isoformat()),
        )
        conn.commit()
        return upload_id
    finally:
        conn.close()


def get(upload_id: str, user_id: int) -> dict | None:
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT id, title, content FROM quiz_uploads WHERE id=? AND user_id=? AND expires_at > ?",
            (upload_id, user_id, datetime.utcnow().isoformat()),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete(upload_id: str, user_id: int) -> None:
    conn = get_db()
    try:
        conn.execute("DELETE FROM quiz_uploads WHERE id=? AND user_id=?", (upload_id, user_id))
        conn.commit()
    finally:
        conn.close()
