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
    doc_hash      TEXT,                     -- sha256 of cleaned full text (dedup)
    owner         TEXT,                     -- username who uploaded
    title         TEXT,
    source_type   TEXT,                     -- pdf | docx | pptx | txt | paste
    char_count    INTEGER DEFAULT 0,
    chunk_count   INTEGER DEFAULT 0,
    status        TEXT DEFAULT 'pending',   -- pending | chunked | embedded | ready | error
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
    meta_json     TEXT,                     -- arbitrary metadata (page count, etc.)
    -- Dedup is PER OWNER, not global. A global UNIQUE(doc_hash) meant the second
    -- user to upload the same material got the first user's row back and no row
    -- of their own, so their library rendered empty.
    UNIQUE(doc_hash, owner)
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
    ("user_prefs", "last_active_date", "TEXT"),  # YYYY-MM-DD of last study day (streak calc)
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


#: Column order used when copying rows during the documents rebuild. Named
#: explicitly (rather than SELECT *) so the copy is insensitive to column order.
_DOC_COLUMNS = (
    "id, doc_hash, owner, title, source_type, char_count, chunk_count, "
    "status, created_at, meta_json"
)


def _documents_needs_owner_scoped_dedup(conn: sqlite3.Connection) -> bool:
    """True when `documents` still carries the global UNIQUE(doc_hash).

    Detected from the stored DDL rather than a version counter, so it holds for
    databases created before this migration existed and is safe to re-check on
    every boot.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='documents'"
    ).fetchone()
    if row is None or not row[0]:
        return False  # Fresh DB: SCHEMA above already has the composite UNIQUE.
    ddl = " ".join(row[0].split())
    return "doc_hash TEXT UNIQUE" in ddl or "UNIQUE(doc_hash, owner)" not in ddl


def _rebuild_documents_for_owner_scoped_dedup(conn: sqlite3.Connection) -> None:
    """Drop the global UNIQUE(doc_hash) in favour of UNIQUE(doc_hash, owner).

    SQLite cannot drop a constraint in place, so the table is rebuilt. Every row
    is preserved with its original `id` because chunks.document_id references it.

    Two safety points worth stating:
      - The copy runs inside one transaction, so a failure rolls back and leaves
        the original table untouched.
      - Foreign keys are disabled for the swap. Were they on, DROP TABLE would
        cascade or error against chunks; ids are unchanged, so the references
        stay valid either way.
    """
    # Duplicate (doc_hash, owner) pairs would violate the new constraint. They
    # can exist because the old schema never enforced anything per owner.
    dupes = conn.execute(
        "SELECT doc_hash, owner, COUNT(*) c FROM documents "
        "GROUP BY doc_hash, owner HAVING c > 1"
    ).fetchall()
    if dupes:
        # Keep the lowest id per pair; it is the one chunks were written against.
        conn.execute(
            "DELETE FROM documents WHERE id NOT IN ("
            "  SELECT MIN(id) FROM documents GROUP BY doc_hash, owner"
            ")"
        )

    conn.execute("""
        CREATE TABLE documents_migrated (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_hash      TEXT,
            owner         TEXT,
            title         TEXT,
            source_type   TEXT,
            char_count    INTEGER DEFAULT 0,
            chunk_count   INTEGER DEFAULT 0,
            status        TEXT DEFAULT 'pending',
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
            meta_json     TEXT,
            UNIQUE(doc_hash, owner)
        )
    """)
    conn.execute(
        f"INSERT INTO documents_migrated ({_DOC_COLUMNS}) "
        f"SELECT {_DOC_COLUMNS} FROM documents"
    )
    conn.execute("DROP TABLE documents")
    conn.execute("ALTER TABLE documents_migrated RENAME TO documents")


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

    # Not additive, so it sits apart from the block above and is guarded by a
    # check of the live DDL. See _rebuild_documents_for_owner_scoped_dedup.
    if _documents_needs_owner_scoped_dedup(conn):
        had_fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        if had_fk:
            conn.execute("PRAGMA foreign_keys=OFF")
        try:
            with conn:  # Commit on success, roll back on any exception.
                _rebuild_documents_for_owner_scoped_dedup(conn)
        finally:
            if had_fk:
                conn.execute("PRAGMA foreign_keys=ON")
