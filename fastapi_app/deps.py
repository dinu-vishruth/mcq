"""Reusable dependencies.

FastAPI's dependency system replaces the repeated
``if not session.get("user_id"): return redirect(...)`` guard that opened almost
every Flask view. Declaring ``user: CurrentUser`` on a route is now enough --
the guard cannot be forgotten, and the redirect/401 behaviour is defined once.
"""
from __future__ import annotations

import sqlite3
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse

from core.models.db import get_db


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


def require_user(request: Request) -> User:
    """Session-backed auth guard for HTML pages -- redirects home when absent."""
    user_id = request.session.get("user_id")
    if not user_id:
        raise RedirectException("/")
    return User(user_id, request.session.get("username", ""))


def require_api_user(request: Request) -> User:
    """Auth guard for JSON endpoints -- 401 JSON instead of a redirect.

    Matches the old ``_require_user()`` helper in core/routes/api.py so the React
    client's error handling is unchanged.
    """
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return User(user_id, request.session.get("username", ""))


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
