CREATE TABLE generation_requests (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    expires_at DOUBLE PRECISION NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE UNIQUE INDEX idx_generation_active_user ON generation_requests(user_id) WHERE active=1;
CREATE INDEX idx_generation_user_time ON generation_requests(user_id, created_at);
-- Existing results must not be analyzed again during the upgrade.
ALTER TABLE results ADD COLUMN enrichment_status TEXT NOT NULL DEFAULT 'done';
ALTER TABLE results ADD COLUMN enrichment_deadline DOUBLE PRECISION;
