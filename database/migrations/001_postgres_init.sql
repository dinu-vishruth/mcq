-- 001_postgres_init.sql: initial schema for a fresh Postgres database.
-- Mirrors the SQLite schema that fastapi_app/database.py and
-- core.models.migrations build locally, in Postgres dialect. Nothing here is
-- destructive; Postgres starts from an empty database (existing SQLite files
-- are not migrated automatically -- if that is ever needed it should be a
-- one-off copy script run by an operator, not boot code).

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('teacher', 'student')),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    email TEXT,
    google_id TEXT,
    picture TEXT,
    auth_provider TEXT DEFAULT 'local'
);

CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    session_key TEXT UNIQUE,
    teacher TEXT,
    created_at TEXT,
    difficulty TEXT,
    timer INTEGER,
    mcqs_json TEXT,
    archived INTEGER DEFAULT 0,
    document_id INTEGER
);

CREATE TABLE IF NOT EXISTS results (
    id SERIAL PRIMARY KEY,
    session_key TEXT,
    student_name TEXT,
    user_id INTEGER REFERENCES users(id),
    score INTEGER,
    total INTEGER,
    submitted_at TEXT,
    detail_json TEXT,
    time_spent INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS login_attempts (
    id SERIAL PRIMARY KEY,
    ip TEXT,
    username TEXT,
    attempt_time TEXT,
    success INTEGER
);

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    doc_hash TEXT,
    owner TEXT,
    title TEXT,
    source_type TEXT,
    char_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    meta_json TEXT,
    UNIQUE(doc_hash, owner)
);

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    char_start INTEGER DEFAULT 0,
    char_end INTEGER DEFAULT 0,
    token_estimate INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);

CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    content_hash TEXT UNIQUE NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BYTEA NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    document_id INTEGER,
    role TEXT,
    intent TEXT,
    content TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS learning_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    session_key TEXT,
    topic TEXT,
    question TEXT,
    is_correct INTEGER DEFAULT 0,
    difficulty TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_learning_user ON learning_history(user_id);

CREATE TABLE IF NOT EXISTS weak_topics (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    topic TEXT,
    wrong_count INTEGER DEFAULT 0,
    total_count INTEGER DEFAULT 0,
    last_seen TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, topic)
);
CREATE INDEX IF NOT EXISTS idx_weak_user ON weak_topics(user_id);

CREATE TABLE IF NOT EXISTS user_prefs (
    user_id INTEGER PRIMARY KEY,
    goal TEXT DEFAULT '',
    style TEXT DEFAULT '',
    daily_minutes INTEGER DEFAULT 30,
    xp INTEGER DEFAULT 0,
    streak INTEGER DEFAULT 0,
    onboarded INTEGER DEFAULT 0,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    last_active_date TEXT
);

-- One identity per Google account; mirrors the partial unique index used
-- locally (Postgres can express partial indexes directly).
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_id
    ON users(google_id) WHERE google_id IS NOT NULL;
