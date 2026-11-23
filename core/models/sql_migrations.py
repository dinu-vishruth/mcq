"""Numbered SQL migration files with a tiny runner.

Layout: ``database/migrations/NNN_<sqlite|postgres>_<name>.sql``. The dialect is
part of the filename, so each file stays pure SQL (index names, ALTER syntax and
 SERIAL/IDENTITY differ enough between SQLite and Postgres that sharing one body
would need a preprocessor -- not worth it for a handful of files).

Runner rules:
  * Files are applied in filename order, at most once, tracked in the
    ``_sql_migrations`` table.
  * Adding a migration = adding a file. Nothing else changes, which keeps the
    "additive and idempotent" move already used by fastapi_app/database.py and
    core.models.migrations.
  * A file only ever runs against the dialect named in its filename; the other
    backend ignores it.
"""
from __future__ import annotations

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIGRATIONS_DIR = os.path.join(_ROOT, "database", "migrations")

_TRACKING_TABLE = "sql_migrations"


def _applied(conn) -> set[str]:
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {_TRACKING_TABLE} ("
        "name TEXT PRIMARY KEY, applied_at TEXT DEFAULT CURRENT_TIMESTAMP)"
    )
    applied = conn.execute(f"SELECT name FROM {_TRACKING_TABLE}").fetchall()
    return {row["name"] for row in applied}


def run_sql_migrations(conn, backend: str) -> list[str]:
    """Apply pending migrations for `backend` ("sqlite" or "postgres").

    Returns the names of migrations applied this call. Adding a file without
    bumping any code is the point of this runner.
    """
    if backend not in ("sqlite", "postgres"):
        raise ValueError(f"unknown backend {backend!r}")
    applied_now: list[str] = []
    done = _applied(conn)

    if not os.path.isdir(MIGRATIONS_DIR):  # pragma: no cover - repo layout guard
        return applied_now

    for filename in sorted(os.listdir(MIGRATIONS_DIR)):
        parts = filename.split("_")
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        name_base, dialect = parts[0], parts[1]
        if dialect != backend:
            continue
        if filename in done or name_base in {p.split("_")[0] for p in done}:
            continue
        with open(os.path.join(MIGRATIONS_DIR, filename), "r", encoding="utf-8") as fh:
            sql = fh.read()
        conn.executescript(sql)
        conn.execute(f"INSERT INTO {_TRACKING_TABLE} (name) VALUES (?)", (filename,))
        conn.commit()
        applied_now.append(filename)

    return applied_now
