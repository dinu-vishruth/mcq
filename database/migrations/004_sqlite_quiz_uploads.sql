CREATE TABLE IF NOT EXISTS quiz_uploads (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_quiz_uploads_expiry ON quiz_uploads(expires_at);
