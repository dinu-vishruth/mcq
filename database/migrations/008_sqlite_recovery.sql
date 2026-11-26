ALTER TABLE users ADD COLUMN session_epoch INTEGER NOT NULL DEFAULT 0;
ALTER TABLE generation_jobs ADD COLUMN heartbeat_at TEXT;
ALTER TABLE generation_jobs ADD COLUMN started_at TEXT;
ALTER TABLE generation_jobs ADD COLUMN run_token TEXT;
ALTER TABLE generation_jobs ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0;
CREATE TABLE account_tokens (
 digest TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 purpose TEXT NOT NULL, email TEXT NOT NULL, expires_at DOUBLE PRECISION NOT NULL,
 created_at DOUBLE PRECISION NOT NULL
);
CREATE TABLE upload_objects (storage_key TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE);
CREATE TABLE deletion_tasks (id TEXT PRIMARY KEY, document_ids TEXT NOT NULL, storage_keys TEXT NOT NULL);
CREATE TABLE upload_prefixes (user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, prefix TEXT NOT NULL, PRIMARY KEY(user_id,prefix));
INSERT INTO upload_prefixes SELECT id, username || '/' FROM users;
CREATE TABLE recovery_requests (ip_hash TEXT NOT NULL, requested_at DOUBLE PRECISION NOT NULL);
