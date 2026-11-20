"""Per-user generation throttles.

The old helper counted rows with a ``LIMIT`` and inspected the oldest of the
most recent N -- which happens to be per-username only because the query was
filtered by ``teacher``, but it was never called from the JSON generation route,
so in practice nothing was enforced there. This version is explicit:

  * cooldown: seconds since this user's last generation session
  * window:   total generations by this user in the last 5 minutes

Both read the ``sessions`` table, which is where a successful generation is
recorded. Sessions marked ``difficulty='review'`` are rebuilt from stored
questions with no LLM call, so they don't count against the limit.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import config

WINDOW = timedelta(minutes=5)
_ELIGIBLE = "COALESCE(difficulty, '') != 'review'"


def _parse(value):
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def check_generation_throttle(conn, username: str) -> str | None:
    """Return an error message when this user must wait, else None."""
    cooldown = config.GENERATION_COOLDOWN_SECONDS
    max_per_window = config.GENERATION_MAX_PER_5MIN

    if cooldown > 0:
        row = conn.execute(
            f"SELECT created_at FROM sessions WHERE teacher=? AND {_ELIGIBLE} "
            "ORDER BY created_at DESC LIMIT 1",
            (username,),
        ).fetchone()
        if row is not None:
            last = _parse(row["created_at"])
            if last is not None and (datetime.utcnow() - last).total_seconds() < cooldown:
                return "Please wait a few seconds between generation requests."

    if max_per_window > 0:
        cutoff = (datetime.utcnow() - WINDOW).isoformat()
        row = conn.execute(
            f"SELECT COUNT(*) FROM sessions WHERE teacher=? AND {_ELIGIBLE} "
            "AND created_at >= ?",
            (username, cutoff),
        ).fetchone()
        if row is not None and row[0] >= max_per_window:
            return (f"You can only generate {max_per_window} quiz sets every 5 minutes. "
                    "Please wait a moment.")
    return None


def check_generation_throttle_for_user(username: str) -> str | None:
    """Throttle check that owns its connection.

    ``/api/practice/generate`` is a sync handler (it blocks on the LLM), so
    FastAPI runs it in the threadpool. An injected ``Db`` connection is created
    on the event-loop thread by the async dependency, and SQLite refuses to use
    a connection from another thread -- so the route opens its own short-lived
    connection here instead of borrowing the request's.
    """
    from core.models.db import get_db

    conn = get_db()
    try:
        return check_generation_throttle(conn, username)
    finally:
        conn.close()
