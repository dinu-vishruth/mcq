"""
Additive schema migrations for the agentic/RAG features.

Design rules (mirror the existing app.init_db idioms):
  - Only CREATE TABLE IF NOT EXISTS and try/except ALTER ADD COLUMN.
  - NEVER drop, rename, or alter the type of an existing column/table.
  - Idempotent: safe to call on every boot, like init_db().

Tables added:
  documents        one row per uploaded source document
  chunks           text chunks of a document (RAG retrieval unit)
  embeddings       vector per chunk (BLOB) keyed by content_hash for dedup/cache
  chat_history     conversational/agent interaction log per user
  learning_history per-student per-topic performance events
  weak_topics      derived weak-concept summary per student

Plus sessions.document_id linking a generated MCQ set to its source document.
"""
from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_hash      TEXT UNIQUE,              -- sha256 of cleaned full text (dedup)
    owner         TEXT,                     -- username who uploaded
    title         TEXT,
    source_type   TEXT,                     -- pdf | docx | pptx | txt | paste
    char_count    INTEGER DEFAULT 0,
    chunk_count   INTEGER DEFAULT 0,
    status        TEXT DEFAULT 'pending',   -- pending | chunked | embedded | ready | error
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
    meta_json     TEXT                      -- arbitrary metadata (page count, etc.)
);

CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   INTEGER NOT NULL,
    chunk_index   INTEGER NOT NULL,         -- ordinal within the document
    content       TEXT NOT NULL,
    content_hash  TEXT NOT NULL,            -- sha256 of chunk content (embed cache key)
    char_start    INTEGER DEFAULT 0,        -- offset in the cleaned document
    char_end      INTEGER DEFAULT 0,
    token_estimate INTEGER DEFAULT 0,
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(document_id) REFERENCES documents(id)
);

CREATE TABLE IF NOT EXISTS embeddings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash  TEXT UNIQUE NOT NULL,     -- shared with chunks.content_hash
    model         TEXT NOT NULL,            -- embedding model / backend id
    dim           INTEGER NOT NULL,
    vector        BLOB NOT NULL,            -- float32 little-endian bytes
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER,
    document_id   INTEGER,
    role          TEXT,                     -- user | assistant | agent name
    intent        TEXT,                     -- planner-detected intent
    content       TEXT,
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS learning_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER,
    session_key   TEXT,
    topic         TEXT,                     -- concept/topic label
    question      TEXT,
    is_correct    INTEGER DEFAULT 0,
    difficulty    TEXT,
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS weak_topics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER,
    topic         TEXT,
    wrong_count   INTEGER DEFAULT 0,
    total_count   INTEGER DEFAULT 0,
    last_seen     TEXT,
    updated_at    TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, topic)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document   ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash       ON chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_learning_user     ON learning_history(user_id);
CREATE INDEX IF NOT EXISTS idx_weak_user         ON weak_topics(user_id);
"""

# Additive columns on existing tables: (table, column, definition).
_ADD_COLUMNS = [
    ("sessions", "document_id", "INTEGER"),
]

# Personalization prefs captured during first-login onboarding (goal, learning
# style, daily study minutes). Additive: one row per user, created on demand.
_PREFS_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_prefs (
    user_id       INTEGER PRIMARY KEY,
    goal          TEXT DEFAULT '',
    style         TEXT DEFAULT '',
    daily_minutes INTEGER DEFAULT 30,
    xp            INTEGER DEFAULT 0,
    streak        INTEGER DEFAULT 0,
    onboarded     INTEGER DEFAULT 0,
    updated_at    TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def run_migrations(conn: sqlite3.Connection) -> None:
    """Apply all additive migrations. Idempotent."""
    conn.executescript(SCHEMA)
    conn.executescript(_PREFS_SCHEMA)
    for table, column, ddl in _ADD_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        except sqlite3.OperationalError:
            pass  # Column already exists.
    conn.commit()
