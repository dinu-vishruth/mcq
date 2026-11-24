"""Atomic attempt accounting and expiring execution leases across workers."""
import time
import uuid
from contextlib import contextmanager

import config
from core.models.db import get_db, is_postgres


class GenerationBusy(Exception):
    pass


@contextmanager
def admission(username, *, seconds, charge=True):
    token = uuid.uuid4().hex
    now = time.time()
    conn = get_db()
    try:
        if is_postgres():
            conn.execute("SELECT id FROM generation_control WHERE id=1 FOR UPDATE")
        else:
            conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM generation_leases WHERE expires_at <= ?", (now,))
        conn.execute("DELETE FROM generation_admissions WHERE created_at < ?", (now - 300,))
        if charge:
            attempts = conn.execute("SELECT created_at FROM generation_admissions WHERE username=? ORDER BY created_at DESC",
                                    (username,)).fetchall()
            if attempts and now - attempts[0]["created_at"] < config.GENERATION_COOLDOWN_SECONDS:
                raise GenerationBusy("Please wait a few seconds between generation requests.")
            if config.GENERATION_MAX_PER_5MIN > 0 and len(attempts) >= config.GENERATION_MAX_PER_5MIN:
                raise GenerationBusy("Generation request limit reached. Please try again later.")
        if conn.execute("SELECT scope FROM generation_leases WHERE scope=?", ("user:" + username,)).fetchone():
            raise GenerationBusy("A quiz is already being generated for your account.")
        capacity = max(1, int(getattr(config, "GENERATION_MAX_CONCURRENT", 4)))
        slot = next((f"provider:{i}" for i in range(capacity) if not conn.execute(
            "SELECT scope FROM generation_leases WHERE scope=?", (f"provider:{i}",)).fetchone()), None)
        if slot is None:
            raise GenerationBusy("Quiz generation is busy. Please try again shortly.")
        for scope in ["user:" + username, slot]:
            conn.execute("INSERT INTO generation_leases(scope,token,expires_at) VALUES (?,?,?)",
                         (scope, token, now + seconds + 30))
        if charge:
            conn.execute("INSERT INTO generation_admissions(token,username,created_at) VALUES (?,?,?)",
                         (token, username, now))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    try:
        yield
    finally:
        try:
            conn = get_db()
            try:
                conn.execute("DELETE FROM generation_leases WHERE token=?", (token,))
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            # A cleanup outage must not hide a successfully persisted quiz.
            # The lease remains bounded and will be reclaimed after expiry.
            from core.observability import log_event
            log_event("generation.lease_release_failed", error_type=type(exc).__name__)
