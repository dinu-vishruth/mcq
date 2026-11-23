"""Persisted, per-user AI reservations, including failed and queued requests."""
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass

from fastapi import HTTPException

import config
from core.models.db import get_db, is_postgres


@dataclass
class Reservation:
    id: str
    transferred: bool = False


def release(reservation_id):
    conn = get_db()
    try:
        conn.execute("UPDATE generation_requests SET active=0 WHERE id=?", (reservation_id,))
        conn.commit()
    finally:
        conn.close()


def lease_remaining(reservation_id, user_id):
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT expires_at FROM generation_requests WHERE id=? AND user_id=? AND active=1",
            (reservation_id, user_id),
        ).fetchone()
        return max(0, row["expires_at"] - time.time() - 15) if row else 0
    finally:
        conn.close()


@contextmanager
def generation_slot(user_id, username, *, seconds=None):
    from fastapi_app.rate_limits import check_generation_throttle

    conn = get_db()
    slot = Reservation(uuid.uuid4().hex)
    try:
        # Serialize admission for this user across processes and instances.
        if is_postgres():
            conn.execute("SELECT id FROM users WHERE id=? FOR UPDATE", (user_id,))
        else:
            conn.execute("BEGIN IMMEDIATE")
        now = time.time()
        conn.execute("UPDATE generation_requests SET active=0 WHERE user_id=? AND expires_at<=?",
                     (user_id, now))
        if conn.execute("SELECT 1 FROM generation_requests WHERE user_id=? AND active=1", (user_id,)).fetchone():
            raise HTTPException(429, "A quiz is already being generated for your account. Please wait.")
        # Retain compatibility with sessions created before reservations existed.
        message = check_generation_throttle(conn, username, user_id=user_id)
        if message:
            raise HTTPException(429, message)
        stats = conn.execute(
            "SELECT COUNT(*) AS count, MAX(created_at) AS latest FROM generation_requests "
            "WHERE user_id=? AND created_at>?", (user_id, now - 300),
        ).fetchone()
        if config.GENERATION_MAX_PER_5MIN > 0 and stats["count"] >= config.GENERATION_MAX_PER_5MIN:
            raise HTTPException(429, "Too many generation attempts. Please wait five minutes before retrying.")
        if stats["latest"] is not None and now - stats["latest"] < config.GENERATION_COOLDOWN_SECONDS:
            raise HTTPException(429, "Please wait a few seconds between generation attempts.")
        conn.execute("DELETE FROM generation_requests WHERE active=0 AND created_at<?", (now - 86400,))
        conn.execute("INSERT INTO generation_requests(id,user_id,created_at,expires_at) VALUES (?,?,?,?)",
                     (slot.id, user_id, now, now + (seconds or config.PIPELINE_DEADLINE_SECONDS) + 30))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    try:
        yield slot
    finally:
        if not slot.transferred:
            release(slot.id)
