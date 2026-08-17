"""Database bootstrap for the FastAPI app.

Same schema work the Flask ``init_db()`` did: create the base tables, apply the
additive ALTERs, ensure login_attempts, seed demo accounts on an empty database,
then run the agentic/RAG migrations.

Why it runs on every boot: on Vercel the database lives at /tmp/mcq.db, which is
part of the ephemeral instance. A cold start gets an empty filesystem, so the
schema has to be (re)created before the first request. Every statement is
idempotent, so doing this locally against a real file is harmless.
"""
from __future__ import annotations

import os
import sqlite3

import config

#: Project root, so data files resolve independently of the process CWD.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCHEMA_FILE = os.path.join(_ROOT, "database", "schema.sql")

_BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('teacher', 'student')),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT UNIQUE,
    teacher TEXT,
    created_at TEXT,
    difficulty TEXT,
    timer INTEGER,
    mcqs_json TEXT
);
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT,
    student_name TEXT,
    user_id INTEGER,
    score INTEGER,
    total INTEGER,
    submitted_at TEXT,
    detail_json TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
);
CREATE TABLE IF NOT EXISTS login_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ip TEXT,
    username TEXT,
    attempt_time TEXT,
    success INTEGER
);
"""

# (table, column, type) triples applied with ALTER TABLE. SQLite has no
# "ADD COLUMN IF NOT EXISTS", so a duplicate-column OperationalError is the
# expected signal that the column is already there.
_ADD_COLUMNS = [
    ("users", "email", "TEXT"),
    ("sessions", "archived", "INTEGER DEFAULT 0"),
    ("results", "time_spent", "INTEGER DEFAULT 0"),
    # Google sign-in: OAuth accounts have no password, so password_hash gets a
    # placeholder and these columns carry the provider identity instead.
    ("users", "google_id", "TEXT"),
    ("users", "picture", "TEXT"),
    ("users", "auth_provider", "TEXT DEFAULT 'local'"),
]


def init_db() -> None:
    """Create/upgrade the schema. Safe to call repeatedly.

    DB_PATH is read from `config` at call time rather than bound at import.
    A module-level `from config import DB_PATH` froze whichever value existed
    when this module first loaded, so anything repointing config.DB_PATH
    afterwards (the test suite, a CLI override) had its tables created in one
    database while queries ran against another -- surfacing as
    "no such table: login_attempts".
    """
    db_path = config.DB_PATH
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        try:
            if os.path.exists(_SCHEMA_FILE):
                with open(_SCHEMA_FILE, "r", encoding="utf-8") as fh:
                    conn.executescript(fh.read())
            else:
                conn.executescript(_BASE_SCHEMA)
        except Exception as exc:
            print(f"[WARNING] Schema execution error: {exc}")

        # schema.sql may predate login_attempts; ensure it either way.
        conn.executescript(_BASE_SCHEMA)

        for table, column, coltype in _ADD_COLUMNS:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
                conn.commit()
            except sqlite3.OperationalError:
                pass  # already present

        # A unique index (not a UNIQUE column) because ALTER TABLE cannot add
        # constraints. Partial so the many NULLs on local accounts don't collide.
        try:
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_id "
                "ON users(google_id) WHERE google_id IS NOT NULL"
            )
            conn.commit()
        except sqlite3.OperationalError as exc:
            print(f"[WARNING] google_id index skipped: {exc}")

        _seed_demo_users(conn)

        try:
            from core.models.migrations import run_migrations
            run_migrations(conn)
        except Exception as exc:
            print(f"[WARNING] Agentic/RAG migrations skipped: {exc}")
    finally:
        conn.close()


def _seed_demo_users(conn: sqlite3.Connection) -> None:
    """Seed teacher/student demo accounts, but only into an empty users table."""
    try:
        from werkzeug.security import generate_password_hash

        if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                [
                    ("teacher", generate_password_hash("teacher123"), "teacher"),
                    ("student", generate_password_hash("student123"), "student"),
                ],
            )
            conn.commit()
    except Exception as exc:
        print(f"[WARNING] Failed to seed default users: {exc}")
