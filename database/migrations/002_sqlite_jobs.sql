-- 002_sqlite_jobs.sql: background generation jobs (SQLite dialect).
-- One row per queued/running generation. The request path writes 'queued',
-- the worker advances it, and GET /api/jobs/{id} reads it. Rows are additive
-- history; nothing prunes them yet (results are small and useful for support).

CREATE TABLE IF NOT EXISTS generation_jobs (
    id TEXT PRIMARY KEY,
    user_id INTEGER,
    username TEXT,
    document_id INTEGER,
    status TEXT NOT NULL DEFAULT 'queued',   -- queued | running | done | failed
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
