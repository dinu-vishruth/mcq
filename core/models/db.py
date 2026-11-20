"""
Central SQLite access.

`get_db()` mirrors the connection app.py already uses (Row factory, same path).
It reads config.DB_PATH at call time (not import time) so tests that repoint the
DB continue to work. This module does NOT own schema creation for the legacy
tables — app.init_db() remains the source of truth for those — it adds the new
RAG/agentic tables via core.models.migrations, which is safe to call repeatedly.

`open_connection()` is the single place a connection is built, so the bootstrap
in fastapi_app/database.py gets the same locking behaviour as every repository.
"""
from __future__ import annotations

import sqlite3

import config
from core.models import pg_adapter


#: True when DATABASE_URL points the app at Postgres. SQLite remains the
#: default backend; Postgres is only used when the operator sets DATABASE_URL.
def is_postgres() -> bool:
    return bool(config.DATABASE_URL)

#: Seconds to wait for a write lock before raising. SQLite's 5s default is
#: guessable: a long generation writes while its own request is still open, so
#: concurrent requests need to queue rather than fail with "database is locked".
CONNECT_TIMEOUT_SECONDS = 30


def open_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=CONNECT_TIMEOUT_SECONDS)
    conn.row_factory = sqlite3.Row
    # WAL lets readers proceed while a writer commits. Persistent per file, but
    # cheap to re-assert and correct even for a database created in journal mode.
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        # Filesystems without the locking WAL needs (some network mounts) fall
        # back to the default journal instead of failing the request.
        pass
    return conn


def get_db():
    """Return a connection for the configured backend.

    Postgres returns the sqlite3-compatible adapter from pg_adapter (same
    execute/fetchone/Row surface, so repositories are unchanged); SQLite keeps
    the native connection. Code must not commit to sqlite3-only APIs here:
    repositories are shared by both backends.
    """
    if is_postgres():
        return pg_adapter.open_postgres(config.DATABASE_URL)
    return open_connection(config.DB_PATH)
