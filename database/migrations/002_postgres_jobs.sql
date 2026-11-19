-- 002_postgres_jobs.sql: background generation jobs (Postgres dialect).
-- Mirrors 002_sqlite_jobs.sql exactly; see that file for the status contract.

CREATE TABLE IF NOT EXISTS generation_jobs (
    id TEXT PRIMARY KEY,
    user_id INTEGER,
    username TEXT,
    document_id INTEGER,
    status TEXT NOT NULL DEFAULT 'queued',
    progress INTEGER DEFAULT 0,
    total INTEGER DEFAULT 0,
    params_json TEXT,
    result_json TEXT,
    error TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_generation_jobs_user ON generation_jobs(username);
CREATE INDEX IF NOT EXISTS idx_generation_jobs_status ON generation_jobs(status);
