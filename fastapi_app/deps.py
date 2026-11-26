"""Reusable dependencies.

FastAPI's dependency system replaces the repeated
``if not session.get("user_id"): return redirect(...)`` guard that opened almost
every Flask view. Declaring ``user: CurrentUser`` on a route is now enough --
the guard cannot be forgotten, and the redirect/401 behaviour is defined once.
"""
from __future__ import annotations

import sqlite3
import time
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, status

from core.models.db import get_db
from core.services.account_identity import SESSION_VERSION


class RedirectException(HTTPException):
    """Raised to send a browser somewhere else from inside a dependency.

    Dependencies cannot ``return`` a response, so an anonymous visitor hitting a
    protected HTML page raises this and an exception handler in main.py turns it
    into the 302 the Flask version produced.
    """

    def __init__(self, location: str) -> None:
        super().__init__(status_code=status.HTTP_302_FOUND, detail=location)
        self.location = location


class User:
    """The authenticated user as carried in the session cookie."""

    __slots__ = ("id", "username")

    def __init__(self, user_id: int, username: str) -> None:
        self.id = user_id
        self.username = username


def _session_user(request: Request) -> User | None:
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    if request.session.get("identity_version") != SESSION_VERSION:
        request.session.clear()
        return None
    conn = get_db()
    try:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        if (row is not None and row["session_epoch"] == request.session.get("session_epoch", 0)
                and "last_seen_at" in row.keys() and not row["is_disabled"]):
            now = time.time()
            if not row["last_seen_at"] or now - row["last_seen_at"] >= 60:
                conn.execute("UPDATE users SET last_seen_at=? WHERE id=? AND (last_seen_at IS NULL OR last_seen_at<?)",
                             (now, user_id, now - 60))
                conn.commit()
    finally:
        conn.close()
    if (row is None or row["session_epoch"] != request.session.get("session_epoch", 0)
            or ("is_disabled" in row.keys() and row["is_disabled"])):
        request.session.clear()
        return None
    request.session["username"] = row["username"]
    return User(row["id"], row["username"])


def require_user(request: Request) -> User:
    """Session-backed auth guard for HTML pages -- redirects home when absent."""
    user = _session_user(request)
    if user is None:
        raise RedirectException("/")
    if request.session.get("password_upgrade_required") and request.url.path != "/profile":
        raise RedirectException("/profile")
    return user


def require_api_user(request: Request) -> User:
    """Auth guard for JSON endpoints -- 401 JSON instead of a redirect.

    Matches the old ``_require_user()`` helper in core/routes/api.py so the React
    client's error handling is unchanged.
    """
    user = _session_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if request.session.get("password_upgrade_required"):
        raise HTTPException(403, "Update your password in Settings before continuing.")
    return user


async def db() -> Any:
    """Yield a SQLite connection and always close it.

    The Flask code closed connections by hand at every return point, which is
    easy to get wrong on an error path. A yield-dependency closes it once, even
    when the handler raises.

    Declared ``async`` deliberately. A sync dependency would be run in FastAPI's
    threadpool while the ``async def`` route handlers run on the event loop
    thread -- and a sqlite3 connection may only be used from the thread that
    created it, so the handler would fail with "SQLite objects created in a
    thread can only be used in that same thread". Async keeps creation and use on
    one thread.
    """
    conn = get_db()
    try:
        yield conn
    finally:
        conn.close()


CurrentUser = Annotated[User, Depends(require_user)]
CurrentApiUser = Annotated[User, Depends(require_api_user)]
Db = Annotated[sqlite3.Connection, Depends(db)]
